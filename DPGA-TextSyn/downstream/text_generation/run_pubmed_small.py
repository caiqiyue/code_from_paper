from datasets import load_dataset,Dataset
import tiktoken
import math
import evaluate
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator

parser = argparse.ArgumentParser()
parser.add_argument('--train_filepath', type=str, help='train_filepath')
args = parser.parse_args()
data_files={"train": args.train_filepath, "validation": "./data/pubmed/dev.csv", "test": "./data/pubmed/test.csv"}
print(data_files)
raw_datasets = load_dataset("csv", data_files = data_files)

def num_tokens_from_string(string, encoding):
    """Returns the number of tokens in a text string."""
    
    try:
        num_tokens = len(encoding.encode(string))
    except:
        num_tokens = 0
    return num_tokens

min_token_threshold = 50

training_dataset= raw_datasets['train']

raw_datasets['train'] = training_dataset
print(f"!!!! clean_dataset after {len(raw_datasets['train']['text'])}")

model_path = './text_generation/model/bert/bert_small'
tokenizer = AutoTokenizer.from_pretrained(model_path)

def tokenize_function(examples):
    output = tokenizer(examples['text'], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    
    # Assign -100 to padded positions of the labels
    output["labels"] = output["input_ids"].clone()
    output["labels"][output["labels"]==tokenizer.pad_token_id]=-100
    return output

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)

train_dataset = tokenized_datasets['train'].shuffle()
eval_dataset = tokenized_datasets['validation']
eval_dataset = Dataset.from_dict(eval_dataset)  # 将字典转换为Dataset对象
test_dataset = tokenized_datasets['test']

print("train_dataset", len(train_dataset)) # 36718
print("train_dataset 0", len(train_dataset[0]["input_ids"]))

model = AutoModelForCausalLM.from_pretrained(model_path, is_decoder=True)
model.config.is_decoder

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

metric = evaluate.load("accuracy")
# metric = evaluate.combine(["accuracy","perplexity"])

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    
    preds= preds[labels != -100]
    labels= labels[labels != -100]

    return metric.compute(predictions=preds, references=labels)

save_path = model_path+'_best'

training_args = TrainingArguments(output_dir=save_path,      
                                num_train_epochs = 15,
                                per_device_train_batch_size=64,  
                                per_device_eval_batch_size=64,  
                                logging_steps=50,                
                                evaluation_strategy="epoch",     
                                save_strategy="epoch",          
                                save_total_limit=1,             
                                learning_rate=3e-4,             
                                weight_decay=0.01,              
                                metric_for_best_model="accuracy",  
                                load_best_model_at_end=True) 

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

train_result = trainer.train()
trainer.save_model()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

metrics = trainer.evaluate(test_dataset)
try:
    perplexity = math.exp(metrics["eval_loss"])
except OverflowError:
    perplexity = float("inf")
metrics["perplexity"] = perplexity

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

print(trainer.predict(test_dataset))