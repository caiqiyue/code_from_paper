import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from models_Cprompt.vision_transformer import VisionTransformer
import numpy as np
import math
import copy
from models_Cprompt.vit_coda_p import DualPrompt, L2P, CodaPrompt, CodaPrompt_weight, CodaPrompt_2d_v2
from torch.utils.data import DataLoader

from collections import defaultdict
import random

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import CosineAnnealingLR

import torch_optimizer as optim
# from sam import SAM

DEBUG_METRICS=True

class Linear_mine(nn.Module):
    def __init__(self, in_dim, out_dim, args=None):
        super(Linear_mine, self).__init__() 
        self.args = args
        self.task_class_num = self.args.class_per_task
        self.global_class_min_output = None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.not_trained_task_id = None
        fc_init = nn.Linear(in_dim, self.task_class_num)
        weight_init = fc_init.weight.data
        bias_init = fc_init.bias.data
        self.fc = nn.Linear(in_dim, out_dim)
        
        for i in range(int(out_dim//self.task_class_num)):
            self.fc.weight.data[i*self.task_class_num:(i+1)*self.task_class_num] = weight_init
            self.fc.bias.data[i*self.task_class_num:(i+1)*self.task_class_num] = bias_init
        

        '''
        self.fc = nn.Linear(in_dim, out_dim)

        '''
        #self.fc_ova = nn.Linear(in_dim, out_dim)
        #self.fc_ova = nn.Linear(in_dim, out_dim * 2)
        #self.fc_ova = nn.Linear(in_dim, 50)
    
    def process_frequency(self, task_id, class_distribution):
        slice_before = []
        for i in range(task_id + 1):
            for j in range(10):
               slice_before.extend(class_distribution[j][i]) 
        slice = [slice_before.index(i) for i in slice_before]
        weight = self.fc.weight.data.clone()
        #print(weight.size())
        #bias = self.fc.bias.data.clone()
        weight = weight[slice_before]
        #bias = bias[slice_before]
        weight = weight.view((task_id + 1) * 10, 4, -1) #TODO
        #bias = bias.view((task_id + 1) * 10, -1)
        #print(weight.size())
        weight = self.gram_schmidt(weight, task_id)
        #print(weight.size())
        #print(bias.size())
        #bias = self.gram_schmidt(bias, task_id)
        #print(weight[0][1])
        weight = weight.reshape((task_id + 1) * 40, -1)
        #print(weight[1])
        #bias = bias.transpose(0, 1).view(200)
        self.fc.weight.data[slice_before] = weight[slice]
        #self.bias.weight.data[slice_before] = bias[slice]

    def gram_schmidt(self, vv, task_id):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = (len(vv.shape) >= 3)
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        s = int(task_id * 10)
        f = int((task_id + 1) * 10)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
            
        return uu


    def forward(self, input, class_min_output, class_max_output):
        weight = self.fc.weight
        bias = self.fc.bias
        
        slice_before = class_min_output + class_max_output
        slice = [slice_before.index(i) for i in range(self.out_dim)]
        weight = torch.cat((weight[class_min_output, :].detach().clone(), weight[class_max_output, :]), dim=0)[slice, :]
        bias = torch.cat((bias[class_min_output].detach().clone(), bias[class_max_output]), dim=0)[slice]
        
        return F.linear(input, weight, bias)
    
    def forward_for_ova(self, prompt_proto, client_learned_task_id=None):
        output = self.fc_ova(prompt_proto)
        #print(output.size())
        #print(output[0])
        '''
        client_unlearned_task_id = []
        for i in range(50):
            if i not in client_learned_task_id:
                client_unlearned_task_id.append(i)
        '''
        #print(self.not_trained_task_id)
        min_list = []
        for i in self.global_class_min_output:
            min_list.append(int(i // 4))
        output[:,list(set(min_list))] = -float('inf')
        #output = output.view(-1, 2, self.out_dim)
        #output = F.log_softmax(output, dim=1)[:, 0, :]
        return output




def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p        





import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version


class ResNetZoo_hard(nn.Module):
    def __init__(self, num_classes=10, pt=False, mode=1, prompt_flag=False, prompt_param=None, task_size=10, device='cuda:0', local_clients=10, num_clients=10, class_distribution=None, tasks_global=3, class_distribution_real=None, class_distribution_proportion=None, class_distribution_client_di=None, params=None, args=None):
        super(ResNetZoo_hard, self).__init__()

        # get last layer
        self.params = params 
        self.args = args
        self.fc = nn.Linear(512, num_classes, bias=True)
        self.numclass = num_classes
        self.total_class_list = list(range(self.numclass))
        self.prompt_flag = prompt_flag
        self.task_id = None
        self.task_size = task_size
        self.client_index = -1
        self.class_distribution = class_distribution
        self.class_distribution_real = class_distribution_real
        self.class_distribution_proportion = class_distribution_proportion
        self.class_distribution_client_di = class_distribution_client_di
        self.client_class_min_output = []
        self.client_class_max_output = []
        self.global_class_max_output_previous = []
        self.client_class_min_output_not_contain_previous = []
        self.global_class_min_output_contain_previous = []
        self.global_class_min_output = []
        self.global_class_max_output = []
        self.ep_g = 0
        self.tasks_global = tasks_global
        self.learned_classes = []
        self.unlearned_classes = []
        self.device = device
        self.num_clients = num_clients
        self.current_class = []
        #self.initial_promptchoosing = {}

        # get feature encoder
        if mode == 0:
            if pt:
                print("++++++++++++++++++ in feature+++++++++++++++++++++")
                zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                           num_heads=12, use_grad_checkpointing=False, ckpt_layer=0,
                                           drop_path_rate=0, device=device, args=self.args
                                          )
                from timm.models import vit_base_patch16_224_in21k, vit_base_patch16_224
                load_dict = vit_base_patch16_224(pretrained=True).state_dict()
                del load_dict['head.weight']; del load_dict['head.bias']
                zoo_model.load_state_dict(load_dict)
                '''
                if prompt_flag:
                    print(" freezing original model")
                    for n,p  in zoo_model.named_parameters():
                        if not "prompt" in n:
                            print(f"freezing {n}")
                            p.requires_grad = False
                '''

            # classifier
            #self.fc = nn.Linear(768, num_classes)
            self.fc = Linear_mine(768, num_classes, args=self.args)
            self.criterion_fn = nn.CrossEntropyLoss(reduction='none').cuda(self.device)

        
        self.prompt = CodaPrompt_2d_v2(768, task_size, prompt_param, device=device, clients_local=local_clients, num_clients=num_clients, args=self.args)
        
        
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model
    
    
    def calculate_prompt_choosing(self, train_dataset, c, t, trained_task_id, current_trained_task_id, finished_task, class_features, label_mapping):
        with torch.no_grad():
            indices =[]
            for i in current_trained_task_id:
                indices.append(trained_task_id.index(i))
            #print(indices)
            choosing_class = {}
            classes = self.class_distribution[c][t]
            classes_real = self.class_distribution_real[c][t]
            classes_proportion = self.class_distribution_proportion[c][t]
            if self.class_distribution_client_di is not None:
                class_distribution_client_di = self.class_distribution_client_di[c][t]
            else:
                class_distribution_client_di = None
            mean_aqk_task = None


            if label_mapping == None:
                label_mapping = dict(zip(classes, classes_real))
            else:
                new_label_mapping = dict(zip(classes, classes_real))

                for new_class, new_class_real in new_label_mapping.items():
                    if new_class not in label_mapping:
                        label_mapping[new_class] = new_class_real

            for i in range(len(classes)):
                train_dataset.getTrainData([classes[i]], [], [], c, classes_real=[classes_real[i]], classes_proportion=classes_proportion, class_distribution_client_di=class_distribution_client_di)
                train_loader = DataLoader(dataset=train_dataset,
                                    shuffle=True,
                                    batch_size=self.args.batch_size,
                                    num_workers=8,
                                    pin_memory=True)
                mean_aqk_class = None
                features = []
                for step, (indexs, images, target) in enumerate(train_loader):
                    if isinstance(self.device, int):
                        images, target = images.cuda(self.device), target.cuda(self.device)
                    else:
                        images, target = images.cuda(), target.cuda()
                    
                    with torch.no_grad():
                        q, _, _, q_map = self.feat(images)
                        q = q[:,0,:]
                        features.append(q.cpu().numpy())
                    mean_aqk_list = self.feat.get_aqk(images, prompt=self.prompt, client_index=c, q=q, task_id=t, trained_task_id = trained_task_id, finished_task=finished_task).unsqueeze(0)
                    #print(mean_aqk_list.size())
                    mean_aqk_list = mean_aqk_list.reshape(mean_aqk_list.shape[0], mean_aqk_list.shape[1], len(trained_task_id), -1)
                    mean_aqk_list = mean_aqk_list[:, :, indices, :]
                    mean_aqk_list = mean_aqk_list.reshape(mean_aqk_list.shape[0], mean_aqk_list.shape[1], -1)
                    #print(mean_aqk_list.size())
                    if mean_aqk_class is None:
                        mean_aqk_class = mean_aqk_list
                    else:
                        mean_aqk_class = torch.cat((mean_aqk_class, mean_aqk_list), dim=0)

                features = np.concatenate(features, axis=0)
                
                class_id = label_mapping[classes[i]]
                                
                if class_id not in class_features:
                    class_features[class_id] = {}

                if c not in class_features[class_id]:
                    class_features[class_id][c] = []

                if class_features[class_id][c] == []:
                    class_features[class_id][c].append(features)
                else:
                    class_features[class_id][c] = np.vstack(class_features[class_id][c] + [features]) 
                 

                mean_aqk_class = torch.mean(mean_aqk_class, dim=0)
                choosing_class[classes[i]] = mean_aqk_class
                if mean_aqk_task is None:
                    mean_aqk_task = mean_aqk_class.unsqueeze(0)
                else:
                    mean_aqk_task = torch.cat((mean_aqk_task, mean_aqk_class.unsqueeze(0)), dim=0)

        prompt_weights_container = []
              
        if choosing_class:
            example_tensor = next(iter(choosing_class.values()))
            num_layers, num_prompts = example_tensor.shape
            for prompt_id in range(num_prompts):
                prompt_dict = {}
                for class_id, class_tensor in choosing_class.items():
                    weights = class_tensor[:, prompt_id].tolist()
                    prompt_dict[class_id] = weights
                prompt_weights_container.append(prompt_dict)

        return torch.mean(mean_aqk_task, dim=0), choosing_class, class_features, train_loader, label_mapping, prompt_weights_container
        
    def train_class(self, train_dataset, c, t, model_c, global_class_stats, class_train_wrong, args):
        f = 0
        
        if f == 0:
            #print(indices)
            choosing_class = {}
            category_prompt_weights = {}
            classes = self.class_distribution[c][t]
            classes_real = self.class_distribution_real[c][t]
            # print("c", c)
            # print("t", t)
            # print("class_distribution_real", self.class_distribution_real)
            classes_proportion = self.class_distribution_proportion[c][t]
            if self.class_distribution_client_di is not None:
                class_distribution_client_di = self.class_distribution_client_di[c][t]
            else:
                class_distribution_client_di = None
            
            label_mapping = dict(zip(classes, classes_real))
            
            
            for i in range(len(classes)):
                optimizer = torch.optim.Adam(model_c.prompt_module.parameters(), lr=5e-6)
                scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-7)
                #get_linear_schedule_with_warmup
              
             
                # scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
                
                loss_values = []
                num_continue = 0
                if args.dataset == 'ImageNet_R':
                    epoch_class = 20
                else:
                    epoch_class = 25
                for epoch in range(epoch_class):#imagenet-r is 20
                    loss_c = torch.tensor(0.0, requires_grad=True).cuda(self.device)
                    train_dataset.getTrainData([classes[i]], [], [], c, classes_real=[classes_real[i]], classes_proportion=classes_proportion, class_distribution_client_di=class_distribution_client_di)
                    train_loader = DataLoader(dataset=train_dataset,
                                        shuffle=True,
                                        batch_size=self.args.batch_size,
                                        num_workers=8,
                                        pin_memory=True)
                
                    # print("i", i)
                    for step, (indexs, images, target) in enumerate(train_loader):
                        if isinstance(self.device, int):
                            images, target = images.cuda(self.device), target.cuda(self.device)
                        else:
                            images, target = images.cuda(), target.cuda()

                        y = torch.tensor([label_mapping[label.item()] for label in target], dtype=torch.long)
                        # y = torch.tensor(label_mapping[int(target[0])])
                        # print("y", y)
                        class_id = label_mapping[int(target[0])]
                        # print("class_id", class_id)
                        # with torch.no_grad():
                        output = model_c(images, class_id, c, t)
                        output = output[:, 0, :]
                        # print("output", output)
                        g_class_stats = global_class_stats[class_id]
                        # print("g_class_stats", g_class_stats)
                        mean = g_class_stats["mean"]
                        std = g_class_stats["std"]
                        # loss_c = model_c.simple_l2_loss(output, mean)
                        # loss_c = model_c.calculate_loss(output, global_class_stats, y)
                        # loss_c = loss_c / len(target)
                        
                        loss_c = model_c.contrastive_loss(output, global_class_stats, y)

                            
                        # print("loss_c:", loss_c.item())

                        loss_values.append(loss_c.item())
                
                        
                        # num_continue += 1
                        # break    
                    optimizer.zero_grad()
                    loss_c.backward()

                    # torch.nn.utils.clip_grad_norm_(model_c.prompt_module.parameters(), max_norm=1.0)

                    optimizer.step()
                    
                    scheduler.step()

                
                if loss_values[0] <= loss_values[epoch_class-1]:
                    
                    print("class_id_loss", class_id)
                    # model_c.prompt_module.prompt_pool.data[class_id] = 0.0
                    class_train_wrong.append(class_id)
                    # class_train_wrong[c][t].append(class_id)
                # print("class_id", class_id)    
                # if class_id == 129 or class_id == 6 :
                
            # print("num_continue", num_continue)
            class_train_wrong = list(set(class_train_wrong))
           
        return class_train_wrong


    
    def updateweight_with_promptchoosing(self, clients_index, clients_index_push, old_client_0, train_dataset, new_task, task_id, models, global_trained_task_id, choosing, choosing_class, finished_task, finished_task_forchoosing, finished_class, global_task_id_real, class_real, args, ep_g, model_c, label_mapping, global_class_stats, class_stats, prompt_weights_choosing, class_train_wrong, class_stats_to_collect):
        #print(class_real)
        
        trained_task_id_previous = copy.deepcopy(global_trained_task_id)
        trained_task_id_current = copy.deepcopy(global_trained_task_id)
        class_features = defaultdict(list)
        
        if new_task:
            if task_id > 0:
                for c in clients_index:
                    if c in old_client_0:
                        
                        global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                        
                        trained_task_id_previous = sorted(list(set(trained_task_id_previous) - set([global_task_id])))
                    else:
                        global_task_id = task_id * self.prompt.num_clients + c
                        
                        trained_task_id_current = sorted(list(trained_task_id_current + [global_task_id]))
            else:
                for c in clients_index:
                    if c in old_client_0:
                        
                        global_task_id = 0 * self.prompt.num_clients + c
                        
                        trained_task_id_previous = sorted(list(set(trained_task_id_previous) - set([global_task_id])))
                    else:
                        global_task_id = 0 * self.prompt.num_clients + c
                        
                        trained_task_id_current = sorted(list(trained_task_id_current + [global_task_id]))
        
        new_task_id = []
        

        for c in clients_index:
            if new_task:

                if c in old_client_0:
                    current_client_id = c
                    current_task_id = models[c].real_task_id
                    global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                    
                    finished_task[global_task_id] = trained_task_id_current
                    #finished_task_forchoosing[global_task_id] = trained_task_id_current
                    new_task_id.append(global_task_id)
               
                else:
                    if task_id > 0:
                        previous_client_id = c
                        previous_task_id = models[c].real_task_id
                        
                        previous_global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                        
                        finished_task[previous_global_task_id] = trained_task_id_previous
                        #finished_task_forchoosing[previous_global_task_id] = global_trained_task_id
                        

                    current_client_id = c
                    current_task_id = task_id
                   
                    global_task_id = task_id * self.prompt.num_clients + c
                    
                    finished_task[global_task_id] = trained_task_id_current
                    #finished_task_forchoosing[global_task_id] = trained_task_id_current
                    new_task_id.append(global_task_id)
                    
            else:
                current_client_id = c
                current_task_id = models[c].real_task_id
                
                global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                
                finished_task[global_task_id] = global_trained_task_id
                #finished_task_forchoosing[global_task_id] = global_trained_task_id
                new_task_id.append(global_task_id)

        #print(trained_task_id_current)
        
        for c in clients_index:
            if new_task:
                if c in old_client_0:
                    current_client_id = c
                    current_task_id = models[c].real_task_id
                    
                    global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                    
                    if c in clients_index_push:
                        choosing_, choosing_class_, class_features, train_loader, label_mapping, prompt_weights_container = self.calculate_prompt_choosing(train_dataset, current_client_id, current_task_id, trained_task_id_current, trained_task_id_current, finished_task=finished_task, class_features=class_features, label_mapping=label_mapping)
                    else:
                        choosing_, choosing_class_, class_features, train_loader, label_mapping, prompt_weights_container = models[c].model.calculate_prompt_choosing(train_dataset, current_client_id, current_task_id, trained_task_id_current, trained_task_id_current, finished_task=finished_task, class_features=class_features, label_mapping=label_mapping)
                    choosing[global_task_id] = choosing_.detach().cpu()
                    prompt_weights_choosing[global_task_id] = prompt_weights_container
                    for cl in self.class_distribution[c][models[c].real_task_id]:
                        choosing_class[cl] = choosing_class_[cl].detach().cpu()
                        finished_class[cl] = trained_task_id_current
                    
                else:
                    if task_id > 0:
                        previous_client_id = c
                        previous_task_id = models[c].real_task_id
                        
                        previous_global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                        
                        
                        if c in clients_index_push:
                            previous_choosing_, previous_choosing_class_, class_features, train_loader, label_mapping, previous_prompt_weights_container = self.calculate_prompt_choosing(train_dataset, previous_client_id, previous_task_id, global_trained_task_id, trained_task_id_previous, finished_task=finished_task, class_features=class_features, label_mapping=label_mapping)
                        else:
                            previous_choosing_, previous_choosing_class_, class_features, train_loader, label_mapping, previous_prompt_weights_container = models[c].model.calculate_prompt_choosing(train_dataset, previous_client_id, previous_task_id, global_trained_task_id, trained_task_id_previous, finished_task=finished_task, class_features=class_features, label_mapping=label_mapping)
                        choosing[previous_global_task_id] = previous_choosing_.detach().cpu()
                        prompt_weights_choosing[previous_global_task_id] = previous_prompt_weights_container
                        for cl in self.class_distribution[c][models[c].real_task_id]:
                            choosing_class[cl] = previous_choosing_class_[cl].detach().cpu()
                            finished_class[cl] = trained_task_id_previous
                        

                    current_client_id = c
                    current_task_id = task_id
                    global_task_id = task_id * self.prompt.num_clients + c
                    
                    if c in clients_index_push:
                        choosing_, choosing_class_, class_features, train_loader, label_mapping, prompt_weights_container = self.calculate_prompt_choosing(train_dataset, current_client_id, current_task_id, trained_task_id_current, trained_task_id_current, finished_task=finished_task, class_features=class_features, label_mapping=label_mapping)
                    else:
                        choosing_, choosing_class_, class_features, train_loader, label_mapping, prompt_weights_container = models[c].model.calculate_prompt_choosing(train_dataset, current_client_id, current_task_id, trained_task_id_current, trained_task_id_current, finished_task=finished_task, class_features=class_features, label_mapping=label_mapping)
                    choosing[global_task_id] = choosing_.detach().cpu()
                    prompt_weights_choosing[global_task_id] = prompt_weights_container

                    for cl in self.class_distribution[c][task_id]:
                        choosing_class[cl] = choosing_class_[cl].detach().cpu()
                        finished_class[cl] = trained_task_id_current
           
        #if True:
        if ep_g % 3 == 0:
            class_train_wrong = []
            # class_train_wrong = defaultdict(lambda: defaultdict(list))
            for class_id, client_dict in class_features.items():  
                if class_id not in class_stats:
                    class_stats[class_id] = {}

                # class_stats[class_id] = {}
                
                for c, features_list in client_dict.items(): 
                    # print("features_list", features_list)
                    features_array = np.array(features_list) 
                    if len(features_array.shape) == 3 and features_array.shape[0] == 1:
                        features_array = np.squeeze(features_array, axis=0)
                    mean = np.mean(features_array, axis=0)  
                    # print("mean", mean)
                    # print("mean.shape", mean.shape)#(768)
                    std = np.std(features_array, axis=0)    
                   
                    if c in class_stats[class_id]:
                        if not np.array_equal(mean, class_stats[class_id][c]['mean']) or not np.array_equal(std, class_stats[class_id][c]['std']):
           
                            current_max_id = max(class_stats[class_id].keys())
                            new_client_id = current_max_id + 10  
                            class_stats[class_id][new_client_id] = {'mean': mean, 'std': std}
                    else:
                        
                        class_stats[class_id][c] = {'mean': mean, 'std': std}

            class_stats_to_collect = []
            for class_id, client_dict in class_stats.items():
                
                feat_mean_list = []
                feat_std_list = []

                for c_id, stats in client_dict.items():
                    
                    feat_mean_list.append(stats['mean'])  # tensor of shape [feature_dim]
                    feat_std_list.append(stats['std'])    # tensor of shape [feature_dim]
                    if len(client_dict) < 2:

                        client_ids = list(client_dict.keys())
                        class_stats_to_collect.append((class_id, client_ids[0]))


                feat_mean_list = [torch.tensor(x) if isinstance(x, np.ndarray) else x for x in feat_mean_list]
                feat_std_list  = [torch.tensor(x) if isinstance(x, np.ndarray) else x for x in feat_std_list]
                # feat_mean_list = [torch.as_tensor(x, dtype=torch.float32) for x in feat_mean_list]
                # feat_std_list  = [torch.as_tensor(x, dtype=torch.float32) for x in feat_std_list]

                feat_mean_tensor = torch.stack(feat_mean_list, dim=0)  
                feat_std_tensor  = torch.stack(feat_std_list, dim=0)   

                mean = feat_mean_tensor.mean(dim=0)  # [feature_dim]

                var = (feat_std_tensor**2 + feat_mean_tensor**2).mean(dim=0) - mean**2
                
                global_class_stats[class_id] = {
                    'mean': torch.as_tensor(mean, dtype=torch.float32),
                    'std': torch.as_tensor(std, dtype=torch.float32)
                }
        
        


        if ep_g >= 0:
            with torch.random.fork_rng(devices=[]):
                # class_train_wrong = []
                for c in clients_index:
                    if new_task:
                        if c in old_client_0:
                            current_client_id = c
                            current_task_id = models[c].real_task_id
                    
                            global_task_id = models[c].real_task_id * self.prompt.num_clients + c 

                            
                            class_train_wrong = self.train_class(train_dataset, current_client_id, current_task_id, model_c, global_class_stats, class_train_wrong, args)
                                
                    
                                
                        else:
                            if task_id > 0:
                                previous_client_id = c
                                previous_task_id = models[c].real_task_id
                            
                                previous_global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                                    
                                class_train_wrong = self.train_class(train_dataset, previous_client_id, previous_task_id, model_c, global_class_stats, class_train_wrong, args)
                                

                            current_client_id = c
                            current_task_id = task_id
                            class_train_wrong = self.train_class(train_dataset, current_client_id, current_task_id, model_c, global_class_stats, class_train_wrong, args)


        if ep_g % 1 == 0:
            weight = None
            for t_1 in choosing.keys():
                weight_line = None
                for t_2 in choosing.keys():
                    
                    s1 = 0
                    # print("prompt_weights_choosing", prompt_weights_choosing)
                    
                    if len(prompt_weights_choosing[t_1]) > len(prompt_weights_choosing[t_2]):
                        # print("len2")
                        len1 = len(prompt_weights_choosing[t_2])
                    else:
                        # print("len1")
                        len1 = len(prompt_weights_choosing[t_1])
                    for i in range(len1):
                     
                        common_classes_1 = set(prompt_weights_choosing[t_1][i].keys())
                        common_classes_2 = set(prompt_weights_choosing[t_2][i].keys())
                        
                        weights1 = torch.tensor([prompt_weights_choosing[t_1][i][class_id] for class_id in common_classes_1])  
                        weights2 = torch.tensor([prompt_weights_choosing[t_2][i][class_id] for class_id in common_classes_2])  
                         # 3 * 20
                        weights1 = weights1.T
                        weights2 = weights2.T
                        
                    
                        weights1 = nn.functional.normalize(weights1, dim=1)
                        weights2 = nn.functional.normalize(weights2, dim=1)
                    
                        similarity1 = torch.einsum('ij,ij->i', weights1, weights2)
                        s1 += similarity1
                          
                 
                    # print("s1", s1 / len1)
                    similarity = s1 / len1


                    # similarity2 = torch.einsum('bd,bd->b', prompt_choosing_1, prompt_choosing_2)
                    # print("similarity", similarity2)
                    weight_point = torch.mean(similarity, dim=0).unsqueeze(0)
                    
                 
                    if args.dataset == 'ImageNet_R':
                        t = 1 if ep_g % 3 == 0 else 0.9 if ep_g in {2, 5, 8, 11, 14} else 0.95 if ep_g <= 14 else None
                    else:
                        t = 1

                    # print('weight_point', weight_point)


                    print("t_1", t_1)
                    print("t_2", t_2)
                    weight_point = weight_point ** t
                    print('weight_point', weight_point)
                    
                    
                    if weight_line is None:
                        weight_line = weight_point
                    else:
                        weight_line = torch.cat((weight_line, weight_point), dim=0)
                
                #topk_for_task = self.params['topk_for_task']
                topk_for_task = len(trained_task_id_current)
                if topk_for_task > weight_line.shape[0]:
                    topk_for_task = weight_line.shape[0]
                _, idx = weight_line.topk(topk_for_task)
      
                line_choose = torch.ones(weight_line.shape)
                line_choose[idx] = 0
                weight_line = weight_line.masked_fill(line_choose.bool(), 0)
                #print(weight_line)
                weight_line = weight_line / weight_line.sum()
                if weight is None:
                    weight = weight_line.unsqueeze(0)
                else:
                    weight = torch.cat((weight, weight_line.unsqueeze(0)), dim=0)
            if "full" not in self.args.method:
                fc_weight = None
                for c_1 in choosing_class.keys():
                    fc_weight_line = None
                    for c_2 in choosing_class.keys():
                        prompt_choosing_1 = choosing_class[c_1]
                        prompt_choosing_2 = choosing_class[c_2]
                        finished_task_1 = finished_class[c_1]
                        finished_task_2 = finished_class[c_2]
                        #print(finished_task_1)
                        #print(finished_task_2)
                        if len(finished_task_1) > len(finished_task_2):
                            indices = []
                            for i in finished_task_2:
                                indices.append(finished_task_1.index(i))
                            prompt_choosing_1 = prompt_choosing_1.reshape(prompt_choosing_1.shape[0], len(finished_task_1), -1)
                            prompt_choosing_1 = prompt_choosing_1[:, indices, :]
                            prompt_choosing_1 = prompt_choosing_1.reshape(prompt_choosing_1.shape[0], -1)
                        else:
                            indices = []
                            for i in finished_task_1:
                                indices.append(finished_task_2.index(i))
                            prompt_choosing_2 = prompt_choosing_2.reshape(prompt_choosing_2.shape[0], len(finished_task_2), -1)
                            prompt_choosing_2 = prompt_choosing_2[:, indices, :]
                            prompt_choosing_2 = prompt_choosing_2.reshape(prompt_choosing_2.shape[0], -1)
                        
                        prompt_choosing_1 = nn.functional.normalize(prompt_choosing_1, dim=1)
                        prompt_choosing_2 = nn.functional.normalize(prompt_choosing_2, dim=1)
                        similarity = torch.einsum('bd,bd->b', prompt_choosing_1, prompt_choosing_2)
                        
                        if int(c_1 // self.args.class_per_task) == int(c_2 // self.args.class_per_task) and c_1 != c_2: 
                        #if int(c_1 // 20) == int(c_2 // 20):
                            fc_weight_point = torch.zeros(1)
                        else:
                            fc_weight_point = torch.mean(similarity, dim=0).unsqueeze(0)

                            
                            
                            fc_weight_point = fc_weight_point**self.params['class_index']
                            
                        if fc_weight_line is None:
                            fc_weight_line = fc_weight_point
                        else:
                            fc_weight_line = torch.cat((fc_weight_line, fc_weight_point), dim=0)
                    
                    _, idx = fc_weight_line.topk(self.params['topk_for_class'])
                    #print("class id:")
                    #print(c_1)
                    #print(list(choosing_class.keys())[idx[0]])
                    #print(list(choosing_class.keys())[idx[1]])
                    line_choose = torch.ones(fc_weight_line.shape)
                    line_choose[idx] = 0
                    fc_weight_line = fc_weight_line.masked_fill(line_choose.bool(), 0)
                    
                    #print(fc_weight_line)
                    fc_weight_line = fc_weight_line / fc_weight_line.sum()
                    if fc_weight is None:
                        fc_weight = fc_weight_line.unsqueeze(0)
                    else:
                        fc_weight = torch.cat((fc_weight, fc_weight_line.unsqueeze(0)), dim=0)
            #print(weight.size())
            #print(self.prompt.weight[list(choosing.keys())][:, list(choosing.keys())].size())
            #weight_in_prompt = torch.zeros((50, 50), device=self.device)
            #fc_weight_in_prompt = torch.zeros((10, 10), device=self.device)
            #print(fc_weight)
            for i in range(len(list(choosing.keys()))):
                self.prompt.weight[list(choosing.keys()), list(choosing.keys())[i]] = torch.tensor(weight[:, i], device=self.device)
            #self.prompt.weight[list(choosing.keys()), :][:, list(choosing.keys())] = weight
            self.prompt.weight = torch.tensor(self.prompt.weight, device=self.device)
            self.prompt.weight_c[new_task_id] = self.prompt.weight.clone()[new_task_id]
            if "full" not in self.args.method:
                for i in range(len(list(choosing_class.keys()))):
                    self.prompt.fc_weight[list(choosing_class.keys()), list(choosing_class.keys())[i]] = torch.tensor(fc_weight[:, i], device=self.device)
                self.prompt.fc_weight = torch.tensor(self.prompt.fc_weight, device=self.device)
                #print(self.prompt.fc_weight[list(choosing_class.keys()), list(choosing_class.keys())])
        #print(self.prompt.fc_weight)
            
            for c in clients_index:
                if args.dataset == 'ImageNet_R':
                    top_w = (ep_g // 3) + 5
                else:
                    top_w = 4 if ep_g < 6 else 2




                if c in old_client_0:
                    current_client_id = c
                    current_task_id = models[c].real_task_id
                    if "full" not in self.args.method:
                        global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                    else:
                        if models[c].real_task_id == 0:
                            global_task_id = c
                        else:
                            global_task_id = models[c].real_task_id + 49
                    # _, idx = self.prompt.weight[global_task_id].topk(self.params['topk_for_task_selection'])
                    _, idx = self.prompt.weight[global_task_id].topk(top_w)
                    finished_task_forchoosing[global_task_id] = idx
                else:
                    current_client_id = c
                    current_task_id = task_id
                    if "full" not in self.args.method:
                        global_task_id = task_id * self.prompt.num_clients + c
                    else:
                        if task_id == 0:
                            global_task_id = c
                        else:
                            global_task_id = task_id + 49
                    # _, idx = self.prompt.weight[global_task_id].topk(self.params['topk_for_task_selection'])
                    _, idx = self.prompt.weight[global_task_id].topk(top_w)
                    finished_task_forchoosing[global_task_id] = idx

        print(class_real)
        return choosing, choosing_class, finished_task, finished_task_forchoosing, finished_class, global_task_id_real, class_real, label_mapping, model_c, global_class_stats, class_stats, prompt_weights_choosing, top_w, class_train_wrong, class_stats_to_collect


    def Incremental_learning(self, task_id):
        
        self.task_id = task_id
        self.prompt.task_id = self.task_id
        if "noortho" in self.args.method:
            pass
        else:
            self.prompt.process_frequency()

    def set_global_class_min_output(self, global_class_output, global_class_output_now):
        self.global_class_min_output = []
        self.global_class_min_output_contain_previous = []
        self.global_class_max_output_previous = self.global_class_max_output
        self.global_class_max_output = global_class_output
         
        for i in range(self.numclass):
            if i in global_class_output:
                continue
            else:
                self.global_class_min_output.append(i) 
        for i in range(self.numclass):
            if i in global_class_output_now:
                continue
            else:
                self.global_class_min_output_contain_previous.append(i)
        self.fc.global_class_min_output = self.global_class_min_output

    def set_client_class_min_output(self):
        client_class_output = self.current_class
        self.client_class_min_output = []
        self.client_class_min_output_not_contain_previous = []
        self.unlearned_classes = []
        self.client_class_max_output = client_class_output
        for i in range(self.numclass):
            if i in client_class_output:
               continue
            else:
                self.client_class_min_output.append(i)
        for i in range(self.numclass):
            if (i in client_class_output) or (i in self.global_class_max_output_previous):
               continue
            else:
                self.client_class_min_output_not_contain_previous.append(i)
        for i in range(self.numclass):
            if i in self.learned_classes:
               continue
            else:
                self.unlearned_classes.append(i)

           

    def forward(self, x, label=None, client_ids=None, task_ids=None, pen=False, train=False, prompt_pool=None, class_train_wrong=None, select_prompt_class=False, label_mapping=None, aq_k=None, device=0, ova='none', client_learned_task_id=None, labels=None, args=None):
        
        #torch.autograd.set_detect_anomaly(True)
        if self.prompt is not None:
            
            with torch.no_grad():
                q, _, _, q_map = self.feat(x)
                q = q[:,0,:]
            
            
            if train and select_prompt_class:  
                    
                out, prompt_loss, prompt_client, indices_taskchoosing, mean_aqk_list, out_map = self.feat(x, label, client_ids, task_ids, prompt=self.prompt, prompt_class=prompt_pool, class_train_wrong=class_train_wrong, label_mapping=label_mapping, select_prompt_class=select_prompt_class, q=q, train=train, task_id=self.task_id, aq_k=aq_k, ep_g=self.ep_g, client_index=self.prompt.client_index, args=args)
            elif train:  
                out, prompt_loss, prompt_client, indices_taskchoosing, mean_aqk_list, out_map = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id, aq_k=aq_k, ep_g=self.ep_g, client_index=self.prompt.client_index)
            else:
                    
                out, prompt_loss, prompt_client, indices_taskchoosing, mean_aqk_list, out_map = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id, aq_k=aq_k)
        
            
            out = out[:,3 * self.prompt.e_p_length,:]
                
            
        else:
            out, _ = self.feat(x)
            out = out[:,0,:]
        
        #out, _, _ = self.feat(x)
        #out = out[:,0,:]
            
        out = out.view(out.size(0), -1)
        pre_logits = out # for fedmoon
        
        
        weight = self.prompt.fc_weight.detach().clone()
        
        if not pen:
            #if ova == 'ova':
                #return self.fc.forward_for_ova(pre_logits, client_learned_task_id)
            if self.client_index == -1:
                out = self.fc(out, self.global_class_min_output, self.global_class_max_output)
            else:
                out = self.fc(out, self.client_class_min_output, self.client_class_max_output)
      
                
            control_loss = 0
            if self.client_index == -1 and not train:
           
                out[:,self.global_class_min_output] = -float('inf')
                #print(out_ova[0])
                #out[:,self.global_class_min_output] = out[:,self.global_class_min_output]
                #print(out[0])
            elif not train:
               
                out[:,self.client_class_min_output] = -float('inf')
            else:
                out[:,self.client_class_min_output] = -float('inf')
                #out[:,self.client_class_min_output] = out[:,self.client_class_min_output]
                #print(self.client_class_min_output) 
        
        if self.prompt is not None and train:
            return out, prompt_loss, pre_logits, prompt_client, control_loss, mean_aqk_list, q_map, out_map
        else:
            return out
        
        
    
    
    def feature_extractor(self, inputs):
        feature, _ = self.feat(inputs)
        return feature[:,0,:]
    
    def feature_extractor_withprompt(self, inputs):
        with torch.no_grad():
            q, _, _, _ = self.feat(inputs)
            q = q[:,0,:]  
        if "classincremental" in self.args.method:
            #feature, _, _, _ = self.feat(inputs, prompt=self.prompt, q=q, train=True, task_id=self.task_id, aq_k=None)
            feature, _, _, _, _, _, _ = self.feat(inputs, prompt=self.prompt, q=q, train=True, task_id=self.task_id, aq_k=None, ep_g=None, client_index=self.prompt.client_index)
        else:
            feature, _, _, _, _, _ = self.feat(inputs, prompt=self.prompt, q=q, train=True, task_id=self.task_id, aq_k=None, ep_g=None, client_index=self.prompt.client_index)
        if "v2" in self.args.method:
            feature = feature[:,3 * self.prompt.e_p_length,:]
        else:
            feature = feature[:,0,:]
        return feature
    
    def get_K_penalty(self, task):
        K_penalty = self.prompt.get_K_penalty(task)
        return K_penalty
    
    def get_A_penalty(self, task):
        A_penalty = self.prompt.get_A_penalty(task)
        return A_penalty
    
    def get_P_penalty(self, task):
        P_penalty = self.prompt.get_P_penalty(task)
        return P_penalty
    
    def getAttention(self, x, task):
        with torch.no_grad():
            q, _, _ = self.feat(x)
            q = q[:,0,:]
         
        attention = self.feat.getAttention(x, prompt=self.prompt, q=q, task=task)
        return attention
    
    def getPrompt(self, i=0):
        prompts, classes = self.prompt.getPrompt(i)
        return prompts, classes
    
    def getK(self):
        Ks, classes = self.prompt.getK()
        return Ks, classes
    
    def getA(self):
        As, classes = self.prompt.getA()
        return As, classes
            
def get_one_hot(target, num_class, device):
    if isinstance(device, int):
        one_hot=torch.zeros(target.shape[0],num_class).cuda(device)
    else:
        one_hot=torch.zeros(target.shape[0],num_class).cuda()
    
    one_hot=one_hot.scatter(dim=1,index=target.long(),value=1.)
    return one_hot

def vit_pt_imnet_hard(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None, task_size=10, device='cuda:0', local_clients = 10, num_clients=10, class_distribution=None, tasks_global=3, class_distribution_real=None, class_distribution_proportion=None, class_distribution_client_di=None, params=None, args=None):
    return ResNetZoo_hard(num_classes=out_dim, pt=True, mode=0, prompt_flag=prompt_flag, prompt_param=prompt_param, task_size=task_size, device=device, local_clients=local_clients, num_clients=num_clients, class_distribution=class_distribution, tasks_global=tasks_global, class_distribution_real=class_distribution_real, class_distribution_proportion=class_distribution_proportion, class_distribution_client_di=class_distribution_client_di, params=params, args=args)