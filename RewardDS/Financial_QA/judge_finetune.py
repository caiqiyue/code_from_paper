import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append('../')
sys.path.append('../../')
# print(sys.path)
import config
from training_interface import DP_DDP_trainer, DDP_trainer, DDP_QLora_trainer
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoConfig
from transformers.models.qwen2 import Qwen2ForSequenceClassification
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from peft import PrefixTuningConfig, TaskType, get_peft_model, PeftModel, LoraConfig
import json
from torch.nn.utils.rnn import pad_sequence
import argparse
import importlib
from PrivLM_Bench.eval.metrics import Meter
import deepspeed
import hjson
import random
from torch.utils.data.distributed import DistributedSampler
from deepspeed import comm as dist 
import numpy as np
from private_transformers import PrivacyEngine

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)

class DP_Judge_Finetune(DP_DDP_trainer):
    def __init__(self, config):
        self.meter = Meter()
        super().__init__(**config)
    

    def initialize(self):
        super(DP_DDP_trainer, self).initialize()

        if self.reduce_then_dp:
            self.actual_batch_size = self.world_size * self.micro_batch_size * self.n_accumulation_steps

        self.actual_batch_size *= 2 # reject and chosen double
        self.privacy_engine = PrivacyEngine(
            self.model,
            micro_batch_size = self.micro_batch_size,
            actual_batch_size= self.actual_batch_size,
            sample_size=self.train_sample_size,
            epochs=self.epochs,
            max_grad_norm=self.max_grad_norm,
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            clipping_mode=self.clipping_mode,
            ## hr: this flag can bypass the model constraints, make sure to implement carefully
            skip_checks=True,
            use_DDP= True,
            reduce_then_dp=self.reduce_then_dp
        )

        if self.dp:
            self.privacy_engine.attach(self.optimizer)
            print("privacy engine is on")


        # print(self.tokenizer.pad_token_id)
        if self.tokenizer.pad_token_id == None:
            print(f"Setting Pad token to Eos token")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.model.config.pad_token_id = self.tokenizer.pad_token_id
        if self.model.model.config.pad_token_id == None:
            self.model.model.config.pad_token_id = self.tokenizer.pad_token_id


    def get_model(self):
        if self.past_training_dir=="":
            model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, 
                                                                   device_map=self.device,
                                                                    num_labels=1)
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(self.past_training_dir, device_map=self.device,
                                                                       attn_implementation="eager",
                                                                    num_labels=1)
        
        return model
    
    def get_dataloader(self, mode = 'train', tokenizer = None, sampler = None):
        dataset, collate_fn = self.get_dataset(mode=mode, tokenizer= tokenizer)
        print(f"TOTAL DATA of {mode} LEN: {len(dataset)}")

        # shuffle data when Train
        shuffle = True if mode == 'train' else False
        return DataLoader(
                dataset=dataset,
                shuffle=shuffle,
                batch_size=self.micro_batch_size if mode == 'train' else 1,
                collate_fn = lambda x: collate_fn(x, tokenizer),
                sampler = None
            )
    
    def train_on_batch(self, batch_text):
        chosen_sample_ids, reject_sample_ids, chosen_samples, reject_samples, all_sample_ids = \
            batch_text['chosen_sample_ids'], batch_text['reject_sample_ids'], batch_text['chosen_samples'], batch_text['reject_samples'], batch_text['all_sample_ids']
        chosen_sample_ids = chosen_sample_ids.to(self.device)
        reject_sample_ids = reject_sample_ids.to(self.device)
        all_sample_ids = all_sample_ids.to(self.device)


        # all_sample_ids = torch.cat([chosen_sample_ids, reject_sample_ids], dim = 0)
        # all_sample_ids = pad_sequence(all_sample_ids, batch_first=True, padding_value = self.tokenizer.pad_token_id)
        final_logits = self.model(input_ids = all_sample_ids).logits
        chosen_logits = final_logits[:chosen_sample_ids.shape[0]]
        reject_logits = final_logits[chosen_sample_ids.shape[0]:]

        chosen_logits_detach = chosen_logits.detach()
        reject_logits_detach = reject_logits.detach()
        # print(chosen_logits_detach.dtype)
        # print(reject_logits_detach)
        
        # 这一步求导的正负有问题
        coff_chosen = - (1 / F.sigmoid(chosen_logits_detach - reject_logits_detach)) * \
                    F.sigmoid(chosen_logits_detach - reject_logits_detach) * (1 - F.sigmoid(chosen_logits_detach - reject_logits_detach))
        coff_reject = (1 / F.sigmoid(chosen_logits_detach - reject_logits_detach)) * \
                    F.sigmoid(chosen_logits_detach - reject_logits_detach) * (1 - F.sigmoid(chosen_logits_detach - reject_logits_detach))

        # print(f"coff_chosen: {coff_chosen}")
        # print(f"coff_reject: {coff_reject}")

        final_logits[:chosen_sample_ids.shape[0]] *= coff_chosen
        final_logits[chosen_sample_ids.shape[0]:] *= coff_reject

        loss = final_logits.squeeze(-1)
        # print(loss)

        # print(final_logits.shape)

        return loss
    
    def utility_evaluate(self):
        self.model.eval()
        with torch.no_grad():
            diff_list = []
            for idx, batch_text in enumerate(tqdm(self.dev_dataloader)):
                chosen_sample_ids, reject_sample_ids, chosen_sample, reject_sample = \
                    batch_text['chosen_sample_ids'], batch_text['reject_sample_ids'], batch_text['chosen_samples'], batch_text['reject_samples']
                chosen_sample_ids = chosen_sample_ids.to(self.device)
                reject_sample_ids = reject_sample_ids.to(self.device)

                chosen_logits = self.model(input_ids = chosen_sample_ids).logits
                reject_logits = self.model(input_ids = reject_sample_ids).logits
                diff = chosen_logits - reject_logits
                diff = diff.squeeze(-1).mean()
                diff_list.append(diff.detach().cpu().item())
                if idx % 100 == 0 and idx != 0:
                    print("*********************************")
                    print("[Dev sample]")
                    print(f"chosen sample: {chosen_sample}")
                    print(f"reject sample: {reject_sample}")
                    print(f"diff: {diff}", flush = True)
                    print("*********************************")

            avg_diff = sum(diff_list) / len(diff_list)
            
            print("=================================")
            print("[FINAL Dev result]")
            print(f"Loss: {avg_diff}", flush=True)
            print("=================================")

        self.model.train()
        return avg_diff


    def main_train(self):
        self.model.train()
        print(f">>>>>>>>>>>>>>>>>LLM is loaded")
        print(">>>>>>>>>>>>>>>>>>Begin training")
        best_diff = 0
        for epoch in range(self.epochs):
            train_loss_list = []
            ##### training code #####
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(f"Epoch {epoch}")
            self.model.train()
            for idx, batch_text in enumerate(tqdm(self.train_dataloader)):
                record_loss = self.train_on_batch(batch_text=batch_text)
                
                self.step(idx, record_loss)
                torch.cuda.empty_cache()
                train_loss_list.append(record_loss.mean().item())
                if idx % self.train_log_steps == 0 and idx != 0:
                    print(f"===============================================")
                    print(f"[TRAINING SMAPLE]")
                    print(f"{epoch}th epoch, {idx}th batch: training loss: {np.mean(train_loss_list)}", flush=True)
                    print(f"===============================================")

                if ((idx + 1) % self.n_eval_steps == 0) or ((idx + 1) == len(self.train_dataloader)):
                    train_loss_list = []
                    avg_diff = self.utility_evaluate()
                    if avg_diff > best_diff:
                        print(f"STEP {idx}/EPOCH {epoch} SAVE MODEL")
                        best_diff = avg_diff
                        self.save_checkpoints(avg_diff)
    
    
    def save_checkpoints(self, best_diff):
        self.model.save_pretrained(self.model_save_dir)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(f"best avg_diff: {best_diff}")
        print(f"[SAVE] Best model to {self.model_save_dir}")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")



class Judge_Finetune(DDP_trainer):
    def __init__(self, config):
        self.meter = Meter()

        #* add new parameter here
        super().__init__(**config)

    def get_model(self):
        if self.past_training_dir=="":
            model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, 
                                                                   device_map=self.device,
                                                                    num_labels=1)
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            model.config.pad_token_id = tokenizer.pad_token_id
        else:
            model = AutoModelForSequenceClassification.from_pretrained(self.past_training_dir, device_map=self.device,
                                                                       attn_implementation="eager",
                                                                        num_labels=1)
        return model
    
    def get_dataloader(self, mode = 'train', tokenizer = None, sampler = None):
        dataset, collate_fn = self.get_dataset(mode=mode, tokenizer= tokenizer)
        print(f"TOTAL DATA of {mode} LEN: {len(dataset)}")

        # shuffle data when Train
        shuffle = True if mode == 'train' else False
        return DataLoader(
                dataset=dataset,
                shuffle=shuffle,
                batch_size=self.micro_batch_size if mode == 'train' else 1,
                collate_fn = lambda x: collate_fn(x, tokenizer),
                sampler=sampler
            )
    
    def train_on_batch(self, batch_text):
        chosen_sample_ids, reject_sample_ids, chosen_samples, reject_samples, all_sample_ids = \
            batch_text['chosen_sample_ids'], batch_text['reject_sample_ids'], batch_text['chosen_samples'], batch_text['reject_samples'], batch_text['all_sample_ids']
        chosen_sample_ids = chosen_sample_ids.to(self.device)
        reject_sample_ids = reject_sample_ids.to(self.device)
        all_sample_ids = all_sample_ids.to(self.device)


        # all_sample_ids = torch.cat([chosen_sample_ids, reject_sample_ids], dim = 0)
        # all_sample_ids = pad_sequence(all_sample_ids, batch_first=True, padding_value = self.tokenizer.pad_token_id)
        final_logits = self.model(input_ids = all_sample_ids).logits
        chosen_logits = final_logits[:chosen_sample_ids.shape[0]]
        reject_logits = final_logits[chosen_sample_ids.shape[0]:]

        chosen_logits_detach = chosen_logits.detach()
        reject_logits_detach = reject_logits.detach()
        # print(chosen_logits_detach.dtype)
        # print(reject_logits_detach)
        
        # 这一步求导的正负有问题
        coff_chosen = - (1 / F.sigmoid(chosen_logits_detach - reject_logits_detach)) * \
                    F.sigmoid(chosen_logits_detach - reject_logits_detach) * (1 - F.sigmoid(chosen_logits_detach - reject_logits_detach))
        coff_reject = (1 / F.sigmoid(chosen_logits_detach - reject_logits_detach)) * \
                    F.sigmoid(chosen_logits_detach - reject_logits_detach) * (1 - F.sigmoid(chosen_logits_detach - reject_logits_detach))

        # print(f"coff_chosen: {coff_chosen}")
        # print(f"coff_reject: {coff_reject}")

        final_logits[:chosen_sample_ids.shape[0]] *= coff_chosen
        final_logits[chosen_sample_ids.shape[0]:] *= coff_reject

        loss = final_logits.squeeze(-1)
        # print(loss)

        # print(final_logits.shape)

        return loss
    
    def utility_evaluate(self):
        self.model.eval()
        with torch.no_grad():
            diff_list = []
            for idx, batch_text in enumerate(tqdm(self.dev_dataloader)):
                chosen_sample_ids, reject_sample_ids, chosen_sample, reject_sample = \
                    batch_text['chosen_sample_ids'], batch_text['reject_sample_ids'], batch_text['chosen_samples'], batch_text['reject_samples']
                chosen_sample_ids = chosen_sample_ids.to(self.device)
                reject_sample_ids = reject_sample_ids.to(self.device)

                chosen_logits = self.model(input_ids = chosen_sample_ids).logits
                reject_logits = self.model(input_ids = reject_sample_ids).logits
                diff = chosen_logits - reject_logits
                diff = diff.squeeze(-1).mean()
                diff_list.append(diff.detach().cpu().item())
                if idx % 100 == 0 and idx != 0:
                    print("*********************************")
                    print("[Dev sample]")
                    print(f"chosen sample: {chosen_sample}")
                    print(f"reject sample: {reject_sample}")
                    print(f"diff: {diff}", flush = True)
                    print("*********************************")

            avg_diff = sum(diff_list) / len(diff_list)
            
            print("=================================")
            print("[FINAL Dev result]")
            print(f"Loss: {avg_diff}", flush=True)
            print("=================================")

        self.model.train()
        return avg_diff

    def main_train(self):
        self.model.train()
        print(f">>>>>>>>>>>>>>>>>LLM is loaded")
        print(">>>>>>>>>>>>>>>>>>Begin training")
        best_diff = 0
        for epoch in range(self.epochs):
            train_loss_list = []
            ##### training code #####
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(f"Epoch {epoch}")
            self.model.train()
            for idx, batch_text in enumerate(tqdm(self.train_dataloader)):
                record_loss = self.train_on_batch(batch_text=batch_text)
                
                self.step(idx, record_loss)
                torch.cuda.empty_cache()
                train_loss_list.append(record_loss.mean().item())
                if idx % self.train_log_steps == 0 and idx != 0:
                    print(f"===============================================")
                    print(f"[TRAINING SMAPLE]")
                    print(f"{epoch}th epoch, {idx}th batch: training loss: {np.mean(train_loss_list)}", flush=True)
                    print(f"===============================================")

                if ((idx + 1) % self.n_eval_steps == 0) or ((idx + 1) == len(self.train_dataloader)):
                    train_loss_list = []
                    avg_diff = self.utility_evaluate()
                    torch.cuda.empty_cache()
                    if avg_diff > best_diff:
                        print(f"STEP {idx}/EPOCH {epoch} SAVE MODEL")
                        best_diff = avg_diff
                        self.save_checkpoints(avg_diff)
    
    
    def save_checkpoints(self, best_diff):
        self.model.save_pretrained(self.model_save_dir)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(f"best avg_diff: {best_diff}")
        print(f"[SAVE] Best model to {self.model_save_dir}")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")



def replace_config_file(config, file_path):
    if file_path == "": return
    with open(file_path, 'r') as file:
        data = hjson.load(file)
    for key, value in data.items():
        if key in config.keys():
            config[key] = value

if __name__ == "__main__":
    def str2bool(s):
        return s.lower() in ("true", "t", "yes", "y", "1", "True")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help='model name')
    parser.add_argument("--use_DDP", type=str2bool, default="True", help='model name')
    parser.add_argument("--use_dp", type=str2bool, default="False", help='model name')
    parser.add_argument("--use_lora", type=str2bool, default="False", help='model name')
    parser.add_argument('--use_4_bits', type=str2bool, default="False",)
    parser.add_argument('--use_8_bits', type=str2bool, default="False",)


    parser.add_argument("--model_dir", type=str, default="", help='local directory with model data')
    parser.add_argument("--dataset_name", type=str, default="baize_dataset", help='model name')
    parser.add_argument("--dataset_dir", type=str, default="", help='local directory with model data')
    parser.add_argument("--train_data_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--dev_data_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--test_data_path", type=str, default="", help='local directory with model data')
    parser.add_argument("--collate_fn_name", type=str, default="collate_fn", help='local directory with model data')

    parser.add_argument("--max_generation_len", type=int, default=512, help='') # duplicate

    parser.add_argument("--model_save_dir", type=str, default="", help='local directory with model data')


    parser.add_argument("--freeze_embedding", type=str2bool, default="False", help='local directory with model data')
    parser.add_argument("--epoch", type=int, default=3, help='model name')
    # parser.add_argument("--eval_times", type=int, default=2, help='model name')
    parser.add_argument("--past_training_dir", type=str, default="", help='local directory with model data')


    parser.add_argument("--micro_batch_size", type=int, default=4, help='model name')
    parser.add_argument("--n_accumulation_steps", type=int, default=256, help='model name')
    parser.add_argument("--n_eval_steps", type=int, default=256, help='model name')
    parser.add_argument('--lr', type=float, default=1e-4, help='as name')   
    parser.add_argument("--config_path", type=str, default="", help='model name')

    parser.add_argument("--is_single", type=str2bool, default=True, help='model name')

    parser.add_argument("--target_delta", type=float, default=1e-5, help='model name')
    parser.add_argument("--target_epsilon", type=float, default=8, help='model name')

    # Include DeepSpeed configuration arguments
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
    

    # Lora setting
    parser.add_argument('--lora_rank', type=int, default=64,
                    help='lora setting')
    parser.add_argument('--lora_alpha', type=int, default=16,
                    help='lora setting')
    parser.add_argument('--lora_droupout', type=float, default=0.0,
                    help='lora setting')

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    set_seed(args.seed)
    print(args)
    
    # change to config param format
    config = {}

    # general
    config['model_dir'] = args.model_dir
    config['optimizer_type'] = "adamw"
    config['lr'] = args.lr
    config['epochs'] = args.epoch

    config['device_num'] = int(os.environ.get("WORLD_SIZE"))
    config['micro_batch_size'] = args.micro_batch_size
    config['n_accumulation_steps'] = args.n_accumulation_steps
    config['n_eval_steps'] = args.n_eval_steps
    # config['eval_times'] = args.eval_times

    config['dataset_name'] = args.dataset_name
    config['dataset_dir'] = args.dataset_dir
    config['train_data_path'] = args.train_data_path
    config['dev_data_path'] = args.dev_data_path
    config['test_data_path'] = args.test_data_path
    config['collate_fn_name'] = args.collate_fn_name
    config['max_generation_len'] = args.max_generation_len
    config['is_single'] = args.is_single
    config['model_save_dir'] = args.model_save_dir

    config['past_training_dir'] = args.past_training_dir

    
    # lora setting & quantization setting
    config['use_lora'] = args.use_lora
    config['lora_rank'] = args.lora_rank
    config['lora_alpha'] = args.lora_alpha
    config['lora_droupout'] = args.lora_droupout
    config['use_4_bits'] = args.use_4_bits
    config['use_8_bits'] = args.use_8_bits

    # DDP setting
    config['use_DDP'] = args.use_DDP
    if args.use_DDP:
        with open(args.deepspeed_config, 'r') as f:
            deepspeed_config = hjson.load(f)
            # rewrite the deepspeed config file
            deepspeed_config['optimizer'] = {
                "type": config['optimizer_type'],
                "params": {
                    "lr": config['lr']
                }
            }
            deepspeed_config['train_micro_batch_size_per_gpu'] = config['micro_batch_size']
        config['deepspeed_config'] = deepspeed_config
        if args.use_dp:
            config['reduce_then_dp'] = False
    
    # dp setting
    config['dp'] = args.use_dp
    if args.use_dp:
        config['clipping_mode'] = "ghost"
        config['target_epsilon'] = args.target_epsilon
        config['max_grad_norm'] = 1
        config['target_delta'] = args.target_delta
        # ghost clipping not support parameter-sharing(embedding-sharing)--gpt2
        config['freeze_embedding'] = args.freeze_embedding
        replace_config_file(config, args.config_path)
        trainer = DP_Judge_Finetune(config)
    else:
        replace_config_file(config, args.config_path)
        trainer = Judge_Finetune(config)
    
    print(f"CONFIG: {config}")
    print(f"[Simplified CONFIG]: \n\t[Use DDP]: {args.use_DDP}\n\t[Use Mulit-Node]: {config['device_num']>1}\n\t[Use DP]: {args.use_dp}\n\t[Use LoRA]: " + \
          f"{args.use_lora}\n\t[Use 4 bits]: {args.use_4_bits}\n\t[Use 8 bits]: {args.use_8_bits}")
    trainer.main_train()