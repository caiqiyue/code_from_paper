import torch.nn as nn
import torch
import copy
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
# from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def model_to_device(model, parallel, device):
    
    if isinstance(device, int):
        card = torch.device("cuda:{}".format(device))
        model.to(card)
    else:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=device, output_device=device[0])
    return model



def participant_exemplar_storing_fcil(clients, num, model_g, old_client, task_id, clients_index):
    for index in range(num):
        #clients[index].model = copy.deepcopy(model_g)
        if index not in clients_index:
            if index in old_client:
                clients[index].beforeTrain(task_id, 0, index)
            else:
                clients[index].beforeTrain(task_id, 1, index)
            clients[index].update_new_set(task_id, index)


def participant_exemplar_storing_cprompt(clients, num, model_g, old_client, task_id, clients_index):
    for index in range(num):
        #clients[index].model = copy.deepcopy(model_g)
        if index not in clients_index:
            if index in old_client:
                clients[index].beforeTrain(task_id, 0, None, index)
            else:
                clients[index].beforeTrain(task_id, 1, None, index)

def participant_exemplar_storing_dwl(clients, num, model_g, old_client, task_id, clients_index):
    for index in range(num):
        #clients[index].model = copy.deepcopy(model_g)
        if index not in clients_index:
            if index in old_client:
                clients[index].beforeTrain(task_id, 0)
            else:
                clients[index].beforeTrain(task_id, 1)

def participant_exemplar_storing_cfed(clients, num, model_g, old_client, task_id, clients_index):
    for index in range(num):
        #clients[index].model = copy.deepcopy(model_g)
        if index not in clients_index:
            if index in old_client:
                clients[index].beforeTrain(task_id, 0, index)
            else:
                clients[index].beforeTrain(task_id, 1, index)

def participant_exemplar_storing_fedspace(clients, num, model_g, old_client, task_id, clients_index, proto_global, radius_global):
    for index in range(num):
        #clients[index].model = copy.deepcopy(model_g)
        if index not in clients_index:
            if index in old_client:
                clients[index].beforeTrain(task_id, 0, proto_global, radius_global, index)
            else:
                clients[index].beforeTrain(task_id, 1, proto_global, radius_global, index)


   


def local_train_cprompt(clients_index_push, clients, index, model_g, task_id, model_old, ep_g, old_client, classes=None, global_task_id_real=None, class_real=None, consolidation=False, prompt_pool=None, label_mapping=None, class_train_wrong=None):
    if index in clients_index_push:
        if model_g.args.prompt_flag == 'codap_2d_v2':
            client_learned_global_task_id_saved = clients[index].model.prompt.client_learned_global_task_id
        clients[index].model = copy.deepcopy(model_g)
        if model_g.args.prompt_flag == 'codap_2d_v2':
            clients[index].model.prompt.client_learned_global_task_id = client_learned_global_task_id_saved
    else:
        if model_g.args.prompt_flag == 'codap_2d_v2':
            client_learned_global_task_id_saved = clients[index].model.prompt.client_learned_global_task_id
        temp_model = copy.deepcopy(model_g)
        temp_state_dict = copy.deepcopy(clients[index].model.state_dict())
        temp_model.load_state_dict(temp_state_dict)
        clients[index].model = temp_model
        if model_g.args.prompt_flag == 'codap_2d_v2':
            clients[index].model.prompt.client_learned_global_task_id = client_learned_global_task_id_saved

    if index in old_client:
        clients[index].beforeTrain(task_id, 0, classes, index, global_task_id_real, class_real)
    else:
        clients[index].beforeTrain(task_id, 1, classes, index, global_task_id_real, class_real)
    if "extension" in model_g.args.method or "extencl" in model_g.args.method:
        clients[index].update_new_set(task_id, index, ep_g)
    if ep_g < 0:   
        
        with torch.no_grad():
            clients[index].model.fc.fc.weight.data[clients[index].current_class] = 0 * clients[index].model.fc.fc.weight.data[clients[index].current_class]

    if consolidation:
        print("save consolidation")
        clients[index].model.prompt.save_extra_prompt()

    if index in old_client:
        num_samples, local_optimizer, local_lr_schedule, current_class = clients[index].train(ep_g, model_old, model_g, prompt_pool, label_mapping, class_train_wrong, index, task_id, model_g.args, 0)
    else:
        num_samples, local_optimizer, local_lr_schedule, current_class = clients[index].train(ep_g, model_old, model_g, prompt_pool, label_mapping, class_train_wrong, index, task_id, model_g.args)
    #prompt_importance = clients[index].compute_prompt_importance()
    #clients[index].model.eval()
    #idx = clients[index].reorder_prompt(prompt_importance)
    if consolidation:
        print("begin consolidation")
        target_task_id_data_list, origin_task_id, target_class_min_output_list, target_class_max_output_list = clients[index].generate_consolidation_dataset(class_real)
        clients[index].model.prompt.load_extra_prompt()
        clients[index].model.prompt.delete_extra_prompt()
        for t in target_task_id_data_list:
            global_task_id_real[t] = origin_task_id
        for t in range(len(clients[index].model.prompt.client_learned_global_task_id)):
            clients[index].model.prompt.client_learned_global_task_id[t] = global_task_id_real[clients[index].model.prompt.client_learned_global_task_id[t]]
        clients[index].model.prompt.client_learned_global_task_id = sorted(list(set(clients[index].model.prompt.client_learned_global_task_id)))
        clients[index].client_learned_global_task_id = clients[index].model.prompt.client_learned_global_task_id
        for t in range(len(clients[index].model.prompt.trained_task_id)):
            clients[index].model.prompt.trained_task_id[t] = global_task_id_real[clients[index].model.prompt.trained_task_id[t]]   
        clients[index].model.prompt.trained_task_id = sorted(list(set(clients[index].model.prompt.trained_task_id)))
        clients[index].consolidation_train(target_class_min_output_list, target_class_max_output_list)
    local_model = clients[index].model.state_dict()
    
    proto_grad = None

    print('*' * 60)
    return local_model, proto_grad, num_samples, local_optimizer, local_lr_schedule, current_class, None, clients[index].client_learned_task_id, global_task_id_real, clients[index].model.prompt.trained_task_id


def centralized_fractual_pretraining(pre_model_trainer):
    pre_model_trainer.fractal_pretrain()


def federated_fractual_pretraining():
    return


def FedAvg_our_v1(models, num_samples_list, client_index, class_distribution_client, newest_task_id, taskid_local, old_client_0, num_clients, model_g, global_update_lr, device, idxs, clients_learned_task_id, clients_learned_class, global_task_id_real, class_real, global_trained_task_id, global_class_output, models_model, clients_index_pull, w_g_last):
    #print(taskid_local)
    task_frequency = {}
    class_frequency = {}
    for m in models_model:
        if m.model.prompt.client_index in clients_index_pull:
            if "full" not in m.model.prompt.args.method:
                global_task_id = global_task_id_real[m.model.prompt.task_id * num_clients + m.model.prompt.client_index]
            else:
                if m.model.prompt.task_id == 0:
                    global_task_id = m.model.prompt.client_index
                else:
                    global_task_id = m.model.prompt.task_id + 49
            if global_task_id not in task_frequency.keys():
                task_frequency[global_task_id] = 1
            else:
                task_frequency[global_task_id] = task_frequency[global_task_id] + 1
            current_classes = m.model.current_class
            print(current_classes)
            for c in current_classes:
                if c not in class_frequency.keys():
                    class_frequency[c] = 1
                else:
                    class_frequency[c] = class_frequency[c] + 1
    
    summation = sum([num_samples_list[i] for i in range(len(num_samples_list)) if client_index[i] in clients_index_pull])
    if "extension" not in models_model[0].model.prompt.args.method:
        w_avg = copy.deepcopy(models[0])
    else:
        w_avg = copy.deepcopy(w_g_last)
    for key in w_avg.keys():
        weighted_sum = None
        for i in range(len(num_samples_list)):
            weight = num_samples_list[i] / summation
            if "prompt" in key:
                if client_index[i] in clients_index_pull:
                    if client_index[i] != 10000:
                        if "full" not in models_model[0].model.prompt.args.method:
                            global_task_id = global_task_id_real[taskid_local[i] * num_clients + client_index[i]]
                        else:
                            if taskid_local[i] == 0:
                                global_task_id = client_index[i]
                            else:
                                global_task_id = taskid_local[i] + 49
                        if weighted_sum is None:
                            weighted_sum = copy.deepcopy(models[i][key])
                            weighted_sum[list(task_frequency.keys())] = 0 * models[i][key][list(task_frequency.keys())]
                            weighted_sum[global_task_id] = models[i][key][global_task_id] / task_frequency[global_task_id]
                        else:
                            weighted_sum[global_task_id] += models[i][key][global_task_id] / task_frequency[global_task_id]
                    else:
                        pass
            #elif key == "fc.fc.weight" or key == "fc.fc.bias" or key == "fc.fc_ova.weight" or key == "fc.fc_ova.bias":
            elif key == "fc.fc.weight" or key == "fc.fc.bias":    
                #print(key)
                #print(list(class_frequency.keys()))
                if client_index[i] in clients_index_pull:
                    if weighted_sum is None:
                        weighted_sum = copy.deepcopy(models[i][key])
                        weighted_sum[list(class_frequency.keys())] = 0 * models[i][key][list(class_frequency.keys())]
                        #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                        for c in models_model[client_index[i]].model.current_class:
                            weighted_sum[c] = models[i][key][c] / class_frequency[c]
                    else:
                        #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                        for c in models_model[client_index[i]].model.current_class:
                            weighted_sum[c] += models[i][key][c] / class_frequency[c]
                        #weighted_sum[clients_learned_class[i]] = models[i][key][clients_learned_class[i]]
            else:
                if client_index[i] in clients_index_pull:
                    if weighted_sum is None:
                        weighted_sum = weight * models[i][key]
                    else:
                        weighted_sum += weight * models[i][key]
            
            
            
        if weighted_sum is not None:
            w_avg[key] = weighted_sum 
    model_g_before = copy.deepcopy(model_g)   
    model_g.load_state_dict(w_avg)
    print("*************update global**************")
    #update_global(model_g_before, model_g, models, taskid_local, client_index, num_clients, global_update_lr, device, idxs)
    return model_g.state_dict()


def update_global(model_g_before, model_g, models, taskid_local, client_index, num_clients, global_update_lr, device, idxs):
    #diff_K = [, forclient]
    P = model_g_before.prompt.get_previous(taskid_local, client_index, idxs)
    P_now = model_g.prompt.get_previous(taskid_local, client_index, None, pre=model_g_before.prompt)
    P1 = []
    P2 = []
    P3 = []
    diff_P1 = []
    diff_P2 = []
    diff_P3 = []
    i = 0
    for j in range(len(models)):
        client = models[j]
        #local_task_id = taskid_local[j]
        #local_client_index = client_index[j]
        for key in client.keys():
            if "k_share" in key:
                #print(P[i][0][0])
                #print(client[key][local_task_id * num_clients + local_client_index][0][0])
                P1.append(P[i])
                diff_P1.append(P_now[i].clone().detach() - P[i].clone().detach())
                i = i + 1
        for key in client.keys():
            if "a_share" in key:
                P2.append(P[i])
                diff_P2.append(P_now[i].clone().detach() - P[i].clone().detach())
                i = i + 1
        for key in client.keys():
            if "p_share" in key:
                P3.append(P[i])
                diff_P3.append(P_now[i].clone().detach() - P[i].clone().detach())
                i = i + 1
    
    P1 = torch.cat([item.unsqueeze(0) for item in P1], dim=0)
    P2 = torch.cat([item.unsqueeze(0) for item in P2], dim=0)
    P3 = torch.cat([item.unsqueeze(0) for item in P3], dim=0)
    diff_P1 = torch.cat([item.unsqueeze(0) for item in diff_P1], dim=0)
    diff_P2 = torch.cat([item.unsqueeze(0) for item in diff_P2], dim=0)
    diff_P3 = torch.cat([item.unsqueeze(0) for item in diff_P3], dim=0)
    #P1 = torch.FloatTensor(np.array([item.cpu().detach().numpy() for item in P1])).cuda(device)
    #P2 = torch.FloatTensor(np.array([item.cpu().detach().numpy() for item in P2])).cuda(device)
    #P3 = torch.FloatTensor(np.array([item.cpu().detach().numpy() for item in P3])).cuda(device)
    #diff_P1 = torch.FloatTensor(np.array([item.cpu().detach().numpy() for item in diff_P1])).cuda(device)
    #diff_P2 = torch.FloatTensor(np.array([item.cpu().detach().numpy() for item in diff_P2])).cuda(device)
    #diff_P3 = torch.FloatTensor(np.array([item.cpu().detach().numpy() for item in diff_P3])).cuda(device)
    print(model_g.prompt.weight.requires_grad)
    grads1 = torch.autograd.grad(
        outputs = P1,
        inputs=model_g_before.prompt.weight,
        grad_outputs=diff_P1,
        allow_unused=True,
        retain_graph=True
    )
    grads2 = torch.autograd.grad(
        outputs = P2,
        inputs=model_g_before.prompt.weight,
        grad_outputs=diff_P2,
        allow_unused=True,
        retain_graph=True
    )
    grads3 = torch.autograd.grad(
        outputs = P3,
        inputs=model_g_before.prompt.weight,
        grad_outputs=diff_P3,
        allow_unused=True
    )

    #print(grads1[0].size())
    #print(len(grads1))
    
    if (grads1 is not None) and (grads2 is not None) and (grads3 is not None):
        weight = nn.functional.normalize(copy.deepcopy(model_g.prompt.weight), dim=1)
        weight = torch.mm(weight, weight.T)
        print(weight)
        model_g.prompt.weight.data -= global_update_lr * (grads1[0] + grads2[0] + grads3[0])
        weight = nn.functional.normalize(copy.deepcopy(model_g.prompt.weight), dim=1)
        weight = torch.mm(weight, weight.T)
        print(weight)


def aggregate_proto_by_class(proto_locals, proto_global_old, feature_size, ema_global):
    global_classes = set()

    for client in proto_locals.keys():
        global_classes = set.union(global_classes, set(proto_locals[client]["prototype"].keys()))
    global_classes = list(global_classes)
    proto_global = {k: np.zeros(feature_size) for k in global_classes}

    weights_sums = {k: 0 for k in global_classes}

    for client in proto_locals.keys():
        local_proto = proto_locals[client]['prototype']
        for j in global_classes:
            if j in local_proto.keys() and not np.all(local_proto[j] == 0):
                w = proto_locals[client]["num_samples_class"][j]
                proto_global[j] += local_proto[j] * w
                weights_sums[j] += w

    for j in global_classes:
        if 0 < weights_sums[j] < 1:
            proto_global[j] /= weights_sums[j]

    if proto_global_old is not None:
        for k in proto_global_old.keys():
            if k in proto_global.keys():
                proto_global[k] = proto_global[k] * ema_global + proto_global_old[k] * (
                        1 - ema_global)
            else:
                proto_global[k] = proto_global_old[k]

    return proto_global

def aggregate_radius(radius_locals):
    radius_global = 0
    training_num = 0
    for client in radius_locals.keys():
        training_num += radius_locals[client]['sample_num']

    for client in radius_locals.keys():
        local_sample_number = radius_locals[client]['sample_num']
        local_radius = radius_locals[client]['radius']
        w = local_sample_number / training_num
        radius_global += local_radius * w

    return radius_global

def model_global_eval_proto(model_g, proto_global, radius_global, task_id, task_size, device, method):
    model_to_device(model_g, False, device)
    model_g.eval()
    proto_aug = []
    proto_aug_label = []
    index = [k for k, v in proto_global.items()]
    if index:
        for i in range(128):
            np.random.shuffle(index)
            #temp = prototype[index[0]] + np.random.normal(0, 1, prototype[index[0]].shape[0]) * radius
            temp = proto_global[index[0]]
            #print(i)
            proto_aug.append(temp)
            proto_aug_label.append(index[0])
            #proto_aug_label.append(4 * index[0])
        proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(device)
        proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).long()
        soft_feat_aug = model_g.predict(proto_aug)
        predicts = torch.max(soft_feat_aug, dim=1)[1]
        correct = (predicts.cpu() == proto_aug_label).sum()
        #total += len(labels)
        accuracy = (100 * correct / 128)


    model_g.train()
    return accuracy
        



def model_global_eval(model_g, test_dataset, task_id, task_size, device, method, task_num):
    model_to_device(model_g, False, device)
    model_g.eval()
    test_dataset.getTestData([0, task_size * (task_id + 1)])
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=64, num_workers=2, pin_memory=True)
    correct, total = 0, 0
    for step, (indexs, imgs, labels) in enumerate(test_loader):
        if isinstance(device, int):
            imgs, labels = imgs.cuda(device), labels.cuda(device)
        else:
            imgs, labels = imgs.cuda(), labels.cuda()
        with torch.no_grad():
            
            outputs = model_g(imgs)
                
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        if step == 1:
            predicts_1 = predicts.cpu()
            labels_1 = labels.cpu()
        total += len(labels)
    accuracy = 100 * correct / total

    accuracys = []
    for i in range(task_num):
        test_dataset.getTestData([task_size * i, task_size * (i + 1)])
        test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=128, num_workers=2, pin_memory=True)
        correct, total = 0, 0
        for setp, (indexs, imgs, labels) in enumerate(test_loader):
            if isinstance(device, int):
                imgs, labels = imgs.cuda(device), labels.cuda(device)
            else:
                imgs, labels = imgs.cuda(), labels.cuda()
            with torch.no_grad():
                
                outputs = model_g(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracys.append(100 * correct / total)


    model_g.train()
    return accuracy, accuracys, predicts_1, labels_1

def model_global_eval_hard(model_g, test_dataset, task_id, task_size, device, method, task_num, global_class_output, global_class_output_real):
    model_to_device(model_g, False, device)
    model_g.eval()
    #print(model_g.global_class_min_output)
    #print(model_g.client_class_min_output)
    #print(model_g.client_index)
    #print(global_class_output)
    test_dataset.getTestData_hard(global_class_output, global_class_output_real)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=32, num_workers=8, pin_memory=True)
    correct, total = 0, 0
    for step, (indexs, imgs, labels) in enumerate(test_loader):
        if isinstance(device, int):
            imgs, labels = imgs.cuda(device), labels.cuda(device)
        else:
            imgs, labels = imgs.cuda(), labels.cuda()
            
        with torch.no_grad():
            # print("labels", labels)
            if "fcil" in method:
                outputs = model_g(imgs)
            elif "cprompt" in method:
                outputs = model_g(imgs, device=device)
                #print(outputs[0])
            elif "cfed" in method:
                outputs = model_g(imgs)
            elif "fedspace" in method:
                outputs = model_g(imgs)
                #print(outputs[0])
                #print(labels[0])
                #outputs = (outputs[:, 0::4] + outputs[:, 1::4] + outputs[:, 2::4] + outputs[:, 3::4]) / 4
                #outputs = outputs[:, ::4]
            else:
                outputs = model_g(imgs)
                
        predicts = torch.max(outputs, dim=1)[1]

        correct += (predicts.cpu() == labels.cpu()).sum()
        if step == 1:
            predicts_1 = predicts.cpu()
            labels_1 = labels.cpu()
        total += len(labels)
    accuracy = 100 * correct / total

    accuracys = []
    '''
    for i in global_class_output_real:
        test_dataset.getTestData_hard([global_class_output[global_class_output_real.index(i)]], [i])
        test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=128, num_workers=8, pin_memory=True)
        correct, total = 0, 0
        for setp, (indexs, imgs, labels) in enumerate(test_loader):
            if isinstance(device, int):
                imgs, labels = imgs.cuda(device), labels.cuda(device)
            else:
                imgs, labels = imgs.cuda(), labels.cuda()
            with torch.no_grad():
                if "fcil" in method:
                    outputs = model_g(imgs)
                elif "cprompt" in method:
                    outputs = model_g(imgs, device=device)
                elif "cfed" in method:
                    outputs = model_g(imgs)
                elif "fedspace" in method:
                    outputs = model_g(imgs)
                    #outputs = (outputs[:, 0::4] + outputs[:, 1::4] + outputs[:, 2::4] + outputs[:, 3::4]) / 4
                else:
                    outputs = model_g(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracys.append(100 * correct / total)
    '''

    model_g.train()
    return accuracy, accuracys


def cal_entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def cal_modified_entropy(input_, label_, num_classes=500):
    nt_positions = torch.arange(0, num_classes).to(input_.device)
    nt_positions = nt_positions.repeat(input_.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != label_.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)
    logits = torch.gather(input_, 1, nt_positions)

    nt_positions_true = torch.arange(0, num_classes).to(input_.device)
    nt_positions_true = nt_positions_true.repeat(input_.size(0), 1)
    nt_positions_true = nt_positions_true[nt_positions_true[:, :] == label_.view(-1, 1)]
    nt_positions_true = nt_positions_true.view(-1, 1)
    logits_true = torch.gather(input_, 1, nt_positions_true)

    term_1 = -logits * torch.log(1 - logits + 1e-5)
    term_2 = -(1 - logits_true) * torch.log(logits_true + 1e-5)
    term_1 = term_1.sum()
    term_2 = term_2.sum()
    return term_1 + term_2
