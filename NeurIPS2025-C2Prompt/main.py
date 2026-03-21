from ResNet import resnet18_cbam
import torch
import copy
import random
import sys
import os.path as osp
import os
from Fed_utils import * 
from iDOMAINNET import *
from option import args_parser
from models_Cprompt.vit_coda_p import vit_pt_imnet
from models_Cprompt.vit_coda_p_hard import vit_pt_imnet_hard
from prompt_hard import DualPrompt_hard
from models_Cprompt.vision_transformer import VisionTransformer
import dataloaders
from dataloaders.utils import *
import nni
# import vit_class
from vit_class import CombinedViT
from collections import defaultdict


params = {'task_index':30, 'topk_for_task':3, 'topk_for_task_selection':4,
          'class_index':30, 'topk_for_class':3,
          'learning_rate':0.001, 'epochs_local': 2, 'batch_size': 32, 'seed': 0}
optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

args = args_parser()
setup_seed(args.seed)
if len(args.device) == 1:
    args.device = args.device[0]
else:
    torch.cuda.set_device(args.device[0])

class_distribution_client_di = None


if "cprompt" in args.method:
    if args.dataset == 'ImageNet_R':
        class_distribution_client = {}
        class_distribution_client_real = {}
        class_distribution_client_proportion = {}
        for i in range(args.num_clients):
            task_list = []
            for j in range(int(args.epochs_global / args.tasks_global)):
                task_list.append(list(range((i * int(args.epochs_global / args.tasks_global) + j) * 20, (i * int(args.epochs_global / args.tasks_global) + j + 1) * 20)))
            class_distribution_client[i] = task_list

        for i in range(args.num_clients):
            task_list = []
            for j in range(int(args.epochs_global / args.tasks_global)):
                if i * int(args.epochs_global / args.tasks_global) + j > 0:
                    similar_global_task_id = random.sample(list(range(i * int(args.epochs_global / args.tasks_global) + j)), 1)[0]
                    similar_level = random.sample(list(range(args.sim, 21)), 1)[0]
                    if similar_level != 0:
                        similar_client_id = int(similar_global_task_id // int(args.epochs_global / args.tasks_global))
                        similar_task_id = int(similar_global_task_id % int(args.epochs_global / args.tasks_global))
                        if similar_client_id == i:
                            task_list.append(list(random.sample(task_list[similar_task_id], similar_level)) + list(random.sample(list(set(list(range(200)))-set(task_list[similar_task_id])), 20-similar_level)))
                        else:
                            task_list.append(list(random.sample(class_distribution_client_real[similar_client_id][similar_task_id], similar_level)) + list(random.sample(list(set(list(range(200)))-set(class_distribution_client_real[similar_client_id][similar_task_id])), 20-similar_level)))                    
                    else:

                        task_list.append(list(random.sample(list(range(200)), 20))) 
                else:
                    
                    task_list.append(list(random.sample(list(range(200)), 20))) 
            class_distribution_client_real[i] = task_list

        for i in range(args.num_clients):
            task_list = []
            for j in range(int(args.epochs_global / args.tasks_global)):
                proportion_list = [0, 0.2, 0.4, 0.6, 0.8]
                start = proportion_list[int(i * int(args.epochs_global / args.tasks_global) + j) % 5]
                task_list.append([start, start + 0.2])
                
            class_distribution_client_proportion[i] = task_list
    elif args.dataset == 'DomainNet':
        class_distribution_client = {}
        class_distribution_client_real = {}
        class_distribution_client_proportion = {}
        for i in range(args.num_clients):
            task_list = []
            for j in range(int(args.epochs_global / args.tasks_global)):
                task_list.append(list(range((i * int(args.epochs_global / args.tasks_global) + j) * 35, (i * int(args.epochs_global / args.tasks_global) + j + 1) * 35)))
            class_distribution_client[i] = task_list
        for i in range(args.num_clients):
            task_list = []
            for j in range(int(args.epochs_global / args.tasks_global)):
                if i * int(args.epochs_global / args.tasks_global) + j > 0:
                    similar_global_task_id = random.sample(list(range(i * int(args.epochs_global / args.tasks_global) + j)), 1)[0]
                    similar_level = random.sample(list(range(args.sim, 36)), 1)[0]
                    if similar_level != 0:
                        similar_client_id = int(similar_global_task_id // int(args.epochs_global / args.tasks_global))
                        similar_task_id = int(similar_global_task_id % int(args.epochs_global / args.tasks_global))
                        if similar_client_id == i:
                            task_list.append(list(random.sample(task_list[similar_task_id], similar_level)) + list(random.sample(list(set(list(range(345)))-set(task_list[similar_task_id])), 35-similar_level)))
                        else:
                            task_list.append(list(random.sample(class_distribution_client_real[similar_client_id][similar_task_id], similar_level)) + list(random.sample(list(set(list(range(345)))-set(class_distribution_client_real[similar_client_id][similar_task_id])), 35-similar_level)))
                    else:
                        
                        task_list.append(list(random.sample(list(range(345)), 35))) 
                else:
                   
                    task_list.append(list(random.sample(list(range(345)), 35))) 
            class_distribution_client_real[i] = task_list
        for i in range(args.num_clients):
            task_list = []
            for j in range(int(args.epochs_global / args.tasks_global)):
                proportion_list = [i * 0.02 for i in range(50)]
                start = proportion_list[int(i * int(args.epochs_global / args.tasks_global) + j)]
                task_list.append([start, start + 0.02])
            class_distribution_client_proportion[i] = task_list


print("********* FULL TEST **********")
print(class_distribution_client)
print(class_distribution_client_real)
print(class_distribution_client_proportion)
print(class_distribution_client_di)

global_class_output = []
global_trained_task_id = []
global_trained_task_id_nosame = []
global_not_trained_task_id = []

feature_extractor = None


num_clients = args.num_clients
old_client_0 = []
old_client_0_review = []
old_client_1 = [i for i in range(args.num_clients)]
new_client = []
models = []
pre_model_trainer = None

model_g = vit_pt_imnet_hard(out_dim=args.numclass, prompt_flag=args.prompt_flag, prompt_param=args.prompt_param, task_size=args.task_size, device=args.device, local_clients=args.local_clients, num_clients=args.num_clients, class_distribution=class_distribution_client, tasks_global=args.tasks_global, class_distribution_real=class_distribution_client_real, class_distribution_proportion=class_distribution_client_proportion, class_distribution_client_di=class_distribution_client_di, params=params, args=args)
model_g = model_to_device(model_g, False, args.device)
model_old = None

train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=True, resize_imnet=True)
test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=True, resize_imnet=True)

model_g.prompt.topk_com = params['topk_for_task_selection']

if args.dataset == 'CIFAR100':
    train_dataset = iCIFAR100('dataset', transform=train_transform, download=True)
    test_dataset = iCIFAR100('dataset', test_transform=test_transform, train=False, download=True)

elif args.dataset == 'DomainNet':
    train_dataset = iDOMAIN_NET(args.dataroot, train=True, transform=train_transform, download_flag=True, seed=args.seed, validation=args.validation, domain=args.easy)
    test_dataset = iDOMAIN_NET(args.dataroot, train=False, transform=test_transform, download_flag=False, seed=args.seed, validation=args.validation)
elif args.dataset == 'ImageNet_R':
    train_dataset = iIMAGENET_R(args.dataroot, train=True, transform=train_transform, download_flag=True, seed=args.seed, validation=args.validation)
    test_dataset = iIMAGENET_R(args.dataroot, train=False, transform=test_transform, download_flag=False, seed=args.seed, validation=args.validation)

encode_model = None

if "cprompt" in args.method:
    for i in range(args.num_clients):
        model_temp = DualPrompt_hard(args.numclass, args.prompt_flag, args.prompt_param, args.task_size, 
                                args.batch_size, args.device, args.epochs_local, args.learning_rate, train_dataset, model_g, args.imbalance)
        models.append(model_temp)

proxy_server = None
## training log
output_dir = osp.join('./training_log', args.method, 'seed' + str(args.seed))
if not osp.exists(output_dir):
    os.system('mkdir -p ' + output_dir)
if not osp.exists(output_dir):
    os.mkdir(output_dir)

out_file = open(osp.join(output_dir, 'log_train_' + str(args.learning_rate) + '.txt'), 'w')
log_str = 'method_{}, task_size_{}, learning_rate_{}'.format(args.method, args.task_size, args.learning_rate)
out_file.write(log_str + '\n')
out_file.flush()

old_task_id = -1

choosing = {}
finished_task = {}
finished_task_forchoosing = {}
choosing_class = {}
finished_class = {}
prompt_weights_choosing = {}

client_finish_task_num = {}
for c in range(args.num_clients):
    client_finish_task_num[c] = 0

unpush_correlation = {}
for c in range(args.num_clients):
    unpush_correlation[c] = 0

unpull_correlation = {}
for c in range(args.num_clients):
    unpull_correlation[c] = 0

global_task_id_real = {}
for i in range(800):
    global_task_id_real[i] = i

model_g.prompt.global_task_id_real = global_task_id_real

class_real = {}
for i in range(args.numclass):
    class_real[i] = i

clients_index_pull = list(range(num_clients))
clients_index_push = list(range(num_clients))
acc_global_list = []
old_client_1_temp = []
label_mapping = {}
global_class_stats = {}
class_stats = {}
num_classes = args.num_classes
with torch.random.fork_rng(devices=[3]):
    model_c = CombinedViT(img_size=224, patch_size=16, in_channels=3, num_classes=num_classes,
                                embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., dropout_rate=0.1, num_prompts=3, args=args)
    model_c = model_to_device(model_c, False, args.device)


class_train_wrong = []
# class_train_wrong = defaultdict(lambda: defaultdict(list))
class_stats_to_collect = []
for ep_g in range(args.epochs_global):
   
    
    pool_grad = []
    num_samples_list = []

    
    task_id = ep_g // args.tasks_global
    new_task = (task_id != old_task_id)
    print(new_task)
    
    if task_id != old_task_id and old_task_id != -1 and "extension" not in args.method and "extencl" not in args.method:
        overall_client = len(old_client_0) + len(old_client_1) + len(new_client)
        new_client = []
        if "full" in args.method:
            old_client_1 = [0]
        else:
            old_client_1 = random.sample([i for i in range(overall_client)], int(overall_client * 0.5))
            # print("old_client_1", old_client_1)
            # Maintain the task order consistent with other comparison methods such as Powder.
            #You can also set task order by using seeds.
            if args.dataset == 'ImageNet_R':
                if ep_g // 3 == 1:
                    old_client_1 = [1, 2]
                elif ep_g // 3 == 2:
                    old_client_1 = [0, 4]
                elif ep_g // 3 == 3:
                    old_client_1 = [1, 3]
                elif ep_g // 3 == 4:
                    old_client_1 = [0, 4]

            # print("old_client_1_change", old_client_1)
        old_client_0 = [i for i in range(overall_client) if i not in old_client_1]
        num_clients = len(new_client) + len(old_client_1) + len(old_client_0)
        print("old_client_0", old_client_0)
        for c in old_client_1:
            client_finish_task_num[c] = client_finish_task_num[c] + 1
    elif task_id != old_task_id and old_task_id != -1 and ("extension" in args.method or "extencl" in args.method):
        overall_client = overall_client_temp
        new_client = new_client_temp
        old_client_1 = old_client_1_temp
        old_client_0 = old_client_0_temp
        num_clients = num_clients_temp
        print("old_client_0", old_client_0)
        for c in old_client_1:
            client_finish_task_num[c] = client_finish_task_num[c] + 1

    if task_id != old_task_id:
        model_g.Incremental_learning(task_id)

    if task_id != old_task_id and old_task_id != -1:   
        model_g = model_to_device(model_g, False, args.device)
        model_old = copy.deepcopy(model_g)
        
    
    print('federated global round: {}, task_id: {}'.format(ep_g, task_id))

    w_local = []
    m_local = []
    taskid_local = []
    clients_learned_task_id = []
    clients_learned_class = []

    idxs = None
    clients_index = random.sample(range(num_clients), args.local_clients)
    print('select part of clients to conduct local training')
    print("clients_index", clients_index)
    
    local_client_index = 0
    
    choosing, choosing_class, finished_task, finished_task_forchoosing, finished_class, global_task_id_real, class_real, label_mapping, model_c, global_class_stats, class_stats, prompt_weights_choosing, top_w, class_train_wrong, class_stats_to_collect = model_g.updateweight_with_promptchoosing(clients_index, clients_index_push, old_client_0, train_dataset, new_task, task_id, models, global_trained_task_id_nosame, choosing, choosing_class, finished_task, finished_task_forchoosing, finished_class, global_task_id_real, class_real, args, ep_g, model_c, label_mapping, global_class_stats, class_stats, prompt_weights_choosing, class_train_wrong, class_stats_to_collect)

    

    model_g.prompt.topk_com = top_w
    model_g.prompt.global_task_id_real = global_task_id_real
    
    
    prompt_pool = model_c.prompt_module.prompt_pool 


    print("prompt_pool", prompt_pool.shape)

    w_g_last = copy.deepcopy(model_g.state_dict())

    for c in clients_index:
        if c in old_client_0:
            continue
        else:
            new_classes = []
            for i in class_distribution_client[c][task_id]:
                new_classes.append(class_real[i])
            global_class_output.extend(new_classes)
            global_trained_task_id.append(global_task_id_real[task_id * args.num_clients + c])
            global_trained_task_id_nosame.append(task_id * args.num_clients + c)

    global_class_output = sorted(list(set(global_class_output)))
    global_trained_task_id = sorted(list(set(global_trained_task_id)))
    global_trained_task_id_nosame = sorted(list(set(global_trained_task_id_nosame)))
    
    for t in range(len(global_trained_task_id)):
        global_trained_task_id[t] = global_task_id_real[global_trained_task_id[t]]
    global_trained_task_id = sorted(list(set(global_trained_task_id)))
    global_class_output_now = []
    for c in clients_index:
        if c in old_client_0:
            continue
        else:
            new_classes = []
            for i in class_distribution_client[c][task_id]:
                new_classes.append(class_real[i])
            global_class_output_now.extend(new_classes)
    global_class_output_now = sorted(list(set(global_class_output_now)))

    print(global_class_output)
    print(global_class_output_now)
    model_g.prompt.trained_task_id_forchoosing = global_trained_task_id
    model_g.prompt.trained_task_id = global_trained_task_id

    model_g.set_global_class_min_output(global_class_output, global_class_output_now)

    
    
    w_g_not_trained = copy.deepcopy(model_g.state_dict())
    for c in clients_index:
        
        if "cprompt" in args.method:
            if (ep_g + 1) % args.tasks_global == 0 and ("extension" in args.method or "extencl" in args.method) and c in old_client_1_temp and client_finish_task_num[c] >= 3 and args.prompt_flag == 'codap_2d_v2':
                local_model, proto_grad, num_samples, local_optimizer, local_lr_schedule, current_classes, idx, client_learned_task_id, global_task_id_real, global_trained_task_id = local_train_cprompt(clients_index_push, models, c, model_g, task_id, model_old, ep_g, old_client_0, global_task_id_real=global_task_id_real, class_real=class_real, consolidation=True)
            else:
                local_model, proto_grad, num_samples, local_optimizer, local_lr_schedule, current_classes, idx, client_learned_task_id, global_task_id_real, global_trained_task_id = local_train_cprompt(clients_index_push, models, c, model_g, task_id, model_old, ep_g, old_client_0, global_task_id_real=global_task_id_real, class_real=class_real, consolidation=False, prompt_pool=prompt_pool, label_mapping=label_mapping, class_train_wrong=class_train_wrong)
                # local_model, proto_grad, num_samples, local_optimizer, local_lr_schedule, current_classes, idx, client_learned_task_id, global_task_id_real, global_trained_task_id = local_train_cprompt(clients_index_push, models, c, model_g, task_id, model_old, ep_g, old_client_0, global_task_id_real=global_task_id_real, class_real=class_real, consolidation=False)
            taskid_local.append(models[c].model.prompt.task_id)
            clients_learned_task_id.append(client_learned_task_id)
            clients_learned_class.append(models[c].model.learned_classes)
        
        if ((ep_g + 1) % args.tasks_global == 0 and ("direct" in args.method or "FLorigin" in args.method or "notran" in args.method)):
            acc_global, accs_global = model_global_eval_hard(models[c].model, test_dataset, task_id, args.task_size, args.device, args.method, int(args.epochs_global / args.tasks_global), models[c].current_class, models[c].current_class_real)
            log_str = 'Client: {}, Task: {}, Round: {} Accuracy = {:.2f}% = Accuracys = {}'\
                                        .format(c, models[c].model.task_id, ep_g, acc_global, accs_global)
            out_file.write(log_str + '\n')
            out_file.flush()
        
        w_local.append(local_model)
        if num_samples != None:
            num_samples_list.append(num_samples)
        local_client_index += 1

    
    clients_index_pull = clients_index
    clients_index_push = clients_index
    
    print("clients_index_pull")
    print(clients_index_pull)
    print("clients_index_push")
    print(clients_index_push)
    print('every participant start updating their exemplar set and old model...')

    participant_exemplar_storing_cprompt(models, num_clients, model_g, old_client_0, task_id, clients_index)
    
    
    print('updating finishes')
    print('federated aggregation...')
    
    
        
    w_g_new = FedAvg_our_v1(w_local, num_samples_list, clients_index, class_distribution_client, task_id, taskid_local, old_client_0, args.num_clients, copy.deepcopy(model_g), args.global_update_lr, args.device, idxs, clients_learned_task_id, clients_learned_class, global_task_id_real, class_real, global_trained_task_id, global_class_output, models, clients_index_pull, w_g_last)
        
    model_g.load_state_dict(w_g_new)
    
    global_optimizer = 0
    global_lr_schedule = 0
    
    model_g.prompt.global_task_id_real = global_task_id_real
    for c in models:
        c.model.prompt.global_task_id_real = global_task_id_real
    


    if (ep_g + 1) % args.tasks_global == 0 and "direct" not in args.method and "notran" not in args.method:
        acc_global_list = []
        model_for_eval = None
        for i in global_trained_task_id_nosame:
            current_class = class_distribution_client[int(i % args.num_clients)][int(i // args.num_clients)]
            
            classes_list = []
            for j in current_class:
                classes_list.append(class_real[j])
            current_class = classes_list
            current_class_real = class_distribution_client_real[int(i % args.num_clients)][int(i // args.num_clients)]
            client_index = int(i % args.num_clients)
            model_task_id = int(i // args.num_clients)
            
            client_class_min_output = sorted(list(set(list(range(args.numclass)))-set(current_class)))
            client_class_max_output = current_class
           
            if client_index in clients_index_push:
                model_for_eval = copy.deepcopy(model_g)
            else:
                model_for_eval = copy.deepcopy(models[client_index].model)
            model_for_eval.client_index = client_index
            model_for_eval.prompt.client_index = client_index
            model_for_eval.prompt.task_id = model_task_id
            model_for_eval.prompt.trained_task_id_forchoosing = finished_task_forchoosing[i]
            model_for_eval.prompt.trained_task_id = global_trained_task_id
            
            model_for_eval.client_class_min_output = client_class_min_output
            model_for_eval.client_class_max_output = client_class_max_output
            model_for_eval.prompt.client_learned_global_task_id = models[client_index].model.prompt.client_learned_global_task_id
            
            
            acc_global, accs_global = model_global_eval_hard(model_for_eval, test_dataset, model_task_id, args.task_size, args.device, args.method, int(args.epochs_global / args.tasks_global), current_class, current_class_real)
            acc_global_list.append(acc_global)
            log_str = 'Client: {}, Task: {}, Round: {} Accuracy = {:.2f}% = Accuracys = {}'\
                                                .format(client_index, model_task_id, ep_g, acc_global, accs_global)
            out_file.write(log_str + '\n')
            out_file.flush()
            print('Client: {}, Task: {}, Round: {} Accuracy = {:.2f}% = Accuracys = {}'.format(client_index, model_task_id, ep_g, acc_global, accs_global))
        del model_for_eval
        nni.report_intermediate_result({'default': sum(acc_global_list) / len(acc_global_list)})
    
    for c in clients_index_push:
        client_learned_global_task_id_saved = models[c].model.prompt.client_learned_global_task_id
        models[c].model = copy.deepcopy(model_g)
        models[c].model.prompt.client_learned_global_task_id = client_learned_global_task_id_saved
    
    old_task_id = task_id
    if "notran" in args.method or "direct" in args.method:
        model_g.load_state_dict(w_g_not_trained)


res=f"Finisned Training! \n Avg={float(sum(acc_global_list) / len(acc_global_list))} \n"
print(res)
out_file.write(res)
out_file.flush()
# print(type(sum(acc_global_list) / len(acc_global_list)))
nni.report_final_result(float(sum(acc_global_list) / len(acc_global_list)))