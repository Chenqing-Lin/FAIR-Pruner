def hello():
    print('FAIR Pruner is working well.')


#############################################################
from scipy.stats import wasserstein_distance
import torch
import itertools
import ot
import pickle
import os
import torch.nn as nn
################################################################

def get_prunedata(prune_datasetloader,batch_size,class_num,pruning_samples_num):
    #
    # prune_datasetloader : is （from torch.utils.data import DataLoader）.
    # class_num : is the number of categories in the dataset.
    # pruning_samples_num : is the upper limit of the sample size for each category used to calculate the distance.
    #
    class_data = {}
    for type in range(class_num):
        class_data[f'{type}'] = []
    n=[0]*class_num
    for inputs, targets in prune_datasetloader:
        if sum(1 for num in n if num > pruning_samples_num)==class_num:
            break
        class_idx = {}
        for type in range(class_num):
            if n[type]<=pruning_samples_num:
                class_idx[f'{type}'] = torch.where(targets == type)[0]
                class_data[f'{type}'].append(torch.index_select(inputs, 0, class_idx[f'{type}']))
                n[type]+=len(class_idx[f'{type}'])
    for type in range(class_num):
        class_data[f'{type}'] = torch.cat(class_data[f'{type}'], dim = 0)
    prune_data = {}
    for type in range(class_num):
        bnum = class_data[f'{type}'].shape[0]//batch_size
        prune_data[f'{type}']=[]
        for i in range(bnum):
            prune_data[f'{type}'].append(class_data[f'{type}'][(i * batch_size):((i + 1) * batch_size), :])
    print('The pruning data is collated')
    return prune_data


features = {}
# 定义一个钩子函数
def get_features(name):
    def hook(model, input, output):
        if  isinstance(model, nn.GRU):
            features[name] = output[0]
        else:
            features[name] = output
    return hook


def get_Distance(model,prunedata,layer_num,device):
    #
    # model: is the model we want to prune.
    # prunedata :is the output from get_prunedata.
    # layer_num : it the aim layer we want to compute Distance.
    # device: torch.device('cpu') or torch.device('cuda') .
    #
    model.to(device)
    # 注册钩子到每一层
    for i,(name, layer) in enumerate(model.named_modules()):
        if i == layer_num:
            handle = layer.register_forward_hook(get_features(f'{i}'))
            break
##############################以下是收集输出########################################
    model.eval()
    with torch.no_grad():
        output_res = {}
        for type in range(len(prunedata)):
            output_res[f'{type}'] = []
            for num in range(len(prunedata[f'{type}'])):
                model(prunedata[f'{type}'][num].to(device))
                if features[f'{layer_num}'].dim() == 3:
                    output_res[f'{type}'].append(features[f'{layer_num}'])
                else:
                    output_res[f'{type}'].append(features[f'{layer_num}'].view(features[f'{layer_num}'].size(0), features[f'{layer_num}'].size(1), -1).contiguous())
    #############################以下是计算距离########################################

        if features[f'{layer_num}'].dim() == 4:
            channel_num = features[f'{layer_num}'].shape[1]
            all_distance = [0]*channel_num
            for combo in itertools.combinations(range(len(prunedata)), 2):
                for channel in range(channel_num):
                    xjbg0 = []
                    xjbg1 = []
                    for i in range(len(output_res[f'{combo[0]}'])):
                        xjbg0.append(output_res[f'{combo[0]}'][i][:,channel,:].contiguous())
                    for i in range(len(output_res[f'{combo[1]}'])):
                        xjbg1.append(output_res[f'{combo[1]}'][i][:,channel,:].contiguous())
                    xjbg0 = torch.cat(xjbg0, dim=0)
                    xjbg1 = torch.cat(xjbg1, dim=0)
                    distance = ot.sliced.sliced_wasserstein_distance(xjbg0.cpu().detach().numpy(),
                                                                     xjbg1.cpu().detach().numpy(),
                                                                     n_projections =50)
                    if distance > all_distance[channel]:
                        all_distance[channel] = distance
        elif features[f'{layer_num}'].dim() == 2:
            neuron_num = features[f'{layer_num}'].shape[1]
            all_distance = [0] *neuron_num
            for combo in itertools.combinations(range(len(output_res)), 2):
                for neuron in range(neuron_num):
                    xjbg0 = []
                    xjbg1 = []
                    for i in range(len(output_res[f'{combo[0]}'])):
                        xjbg0.append(output_res[f'{combo[0]}'][i][:,neuron,0].contiguous())
                    for i in range(len(output_res[f'{combo[1]}'])):
                        xjbg1.append(output_res[f'{combo[1]}'][i][:,neuron, 0].contiguous())
                    xjbg0 = torch.cat(xjbg0, dim=0)
                    xjbg1 = torch.cat(xjbg1, dim=0)
                    distance = wasserstein_distance(xjbg0.cpu().detach().numpy(), xjbg1.cpu().detach().numpy())
                    if distance > all_distance[neuron]:
                        all_distance[neuron] = distance
        elif features[f'{layer_num}'].dim() == 3:
            hidden_num = features[f'{layer_num}'].shape[2]
            all_distance = [0] * hidden_num
            for combo in itertools.combinations(range(len(prunedata)), 2):
                for hidden in range(hidden_num):
                    xjbg0 = []
                    xjbg1 = []
                    for i in range(len(output_res[f'{combo[0]}'])):
                        xjbg0.append(output_res[f'{combo[0]}'][i][:, :, hidden].contiguous())
                    for i in range(len(output_res[f'{combo[1]}'])):
                        xjbg1.append(output_res[f'{combo[1]}'][i][:, :, hidden].contiguous())
                    xjbg0 = torch.cat(xjbg0, dim=0)
                    xjbg1 = torch.cat(xjbg1, dim=0)
                    distance = ot.sliced.sliced_wasserstein_distance(xjbg0.cpu().detach().numpy(),
                                                                     xjbg1.cpu().detach().numpy(),
                                                                     n_projections=50)
                    if distance > all_distance[hidden]:
                        all_distance[hidden] = distance

    print(f'The Distance of the {layer_num}th layer is calculated.')
    all_distance = torch.tensor(all_distance)
    features.clear()
    handle.remove()

    return all_distance


###########################################################################################################
###########################################################################################################

#v2：This version is the same as the previous version
# （get_Distance）, and in order to reduce the pressure
# on the video memory and memory, the data is saved on
# the hard disk. Although it slows down the running
# speed, it can be adopted in the graphics card and
# insufficient memory space.

def get_Distance2(model,prunedata,layer_num,device,path):
    #
    # model: is the model we want to prune
    # prunedata :is the output from get_prunedata
    # layer_num : it the aim layer we want to compute Distance
    # device: torch.device('cpu') or 'cuda'
    # path: A path to save the temporary file
    #
    model.to(device)
    # 注册钩子到每一层
    for i,(name, layer) in enumerate(model.named_modules()):
        if i == layer_num:
            handle = layer.register_forward_hook(get_features(f'{i}'))
            break
##############################以下是收集输出########################################
    model.eval()
    with torch.no_grad():
        for type in range(len(prunedata)):
            output_res = []
            for num in range(len(prunedata[f'{type}'])):
                model(prunedata[f'{type}'][num].to(device))
                output_res.append(features[f'{layer_num}'].view(features[f'{layer_num}'].size(0), features[f'{layer_num}'].size(1), -1).to(device))
            with open(path+f'/layer{layer_num}_output_type{type}.pkl', 'wb') as file:
                pickle.dump(output_res, file)
    #############################以下是计算距离########################################
        if features[f'{layer_num}'].dim() == 4:
            channel_num = features[f'{layer_num}'].shape[1]
            all_distance = [0]*channel_num
            for combo in itertools.combinations(range(len(prunedata)), 2):
                with open(path+f'/layer{layer_num}_output_type{combo[0]}.pkl', 'rb') as file:
                    output_res0 = pickle.load(file)
                with open(path+f'/layer{layer_num}_output_type{combo[1]}.pkl', 'rb') as file:
                    output_res1 = pickle.load(file)
                for channel in range(channel_num):
                    xjbg0 = []
                    xjbg1 = []
                    for i in range(len(output_res0)):
                        xjbg0.append(output_res0[i][:,channel,:])
                    for i in range(len(output_res1)):
                        xjbg1.append(output_res1[i][:,channel,:])
                    xjbg0 = torch.cat(xjbg0, dim=0)
                    xjbg1 = torch.cat(xjbg1, dim=0)
                    distance = ot.sliced.sliced_wasserstein_distance(xjbg0.cpu().detach().numpy(),
                                                                     xjbg1.cpu().detach().numpy(),
                                                                     n_projections =50)
                    if distance > all_distance[channel]:
                        all_distance[channel] = distance

        elif features[f'{layer_num}'].dim() == 2:
            neuron_num = features[f'{layer_num}'].shape[1]
            all_distance = [0] *neuron_num
            for combo in itertools.combinations(range(len(output_res)), 2):
                with open(path+f'/layer{layer_num}_output_type{combo[0]}.pkl','rb') as file:
                    output_res0 = pickle.load(file)
                with open(path+f'/layer{layer_num}_output_type{combo[1]}.pkl','rb') as file:
                    output_res1 = pickle.load(file)
                for neuron in range(neuron_num):
                    # print(f'第{neuron}个神经元正在计算距离')
                    xjbg0 = []
                    xjbg1 = []
                    for i in range(len(output_res0)):
                        xjbg0.append(output_res0[i][:,neuron,0])
                    for i in range(len(output_res1)):
                        xjbg1.append(output_res1[i][:,neuron,0])
                    xjbg0 = torch.cat(xjbg0, dim=0)
                    xjbg1 = torch.cat(xjbg1, dim=0)
                    distance = wasserstein_distance(xjbg0.cpu().detach().numpy(),
                                                    xjbg1.cpu().detach().numpy())
                    if distance > all_distance[neuron]:
                        all_distance[neuron] = distance
        elif features[f'{layer_num}'].dim() == 3:
            hidden_num = features[f'{layer_num}'].shape[2]
            all_distance = [0] * hidden_num
            for combo in itertools.combinations(range(len(prunedata)), 2):
                with open(path+f'/layer{layer_num}_output_type{combo[0]}.pkl','rb') as file:
                    output_res0 = pickle.load(file)
                with open(path+f'/layer{layer_num}_output_type{combo[1]}.pkl','rb') as file:
                    output_res1 = pickle.load(file)
                for hidden in range(hidden_num):
                    xjbg0 = []
                    xjbg1 = []
                    for i in range(len(output_res0)):
                        xjbg0.append(output_res0[i][:, :, hidden])
                    for i in range(len(output_res0)):
                        xjbg1.append(output_res0[i][:, :, hidden])
                    xjbg0 = torch.cat(xjbg0, dim=0)
                    xjbg1 = torch.cat(xjbg1, dim=0)
                    distance = ot.sliced.sliced_wasserstein_distance(xjbg0.cpu().detach().numpy(),
                                                                     xjbg1.cpu().detach().numpy(),
                                                                     n_projections=50)
                    if distance > all_distance[hidden]:
                        all_distance[hidden] = distance
    print(f'The Distance of the {layer_num}th layer is calculated.')
    all_distance = torch.tensor(all_distance)
    features.clear()
    handle.remove()
    for type in range(len(prunedata)):
        os.remove(path+f'/layer{layer_num}_output_type{type}.pkl')

    return all_distance

###############################################################################################################


# 创建一个字典来存储每一层的梯度
gradients = {}
# 定义一个钩子函数来获取梯度
def get_grad(name):
    def hook(module, grad_input, grad_output):
        # grad_input是输入的梯度，grad_output是输出的梯度
        gradients[name] = grad_output[0]
    return hook

# 定义一个钩子函数来捕获GRU model权重的梯度
gru1_weight_ih_grad = None
gru1_weight_hh_grad = None

def hook_gru1_weight_ih(grad):
    global gru1_weight_ih_grad
    gru1_weight_ih_grad = grad

def hook_gru1_weight_hh(grad):
    global gru1_weight_hh_grad
    gru1_weight_hh_grad = grad

def get_ReconstructionError(model,prune_datasetloader,layer_num,device,loss_function):
    #
    # model: is the model we want to prune.
    # prune_datasetloader : is （from torch.utils.data import DataLoader）.
    # layer_num : it the aim layer we want to compute Distance.
    # device: torch.device('cpu') or torch.device('cuda') .
    # loss_function: The loss function used for model training.
    #

    model.to(device)
    model.train()
    if isinstance(list(model.named_modules())[layer_num][1],nn.GRU):
        the_wih = list(model.named_modules())[layer_num][1].weight_ih_l0.data
        the_whh = list(model.named_modules())[layer_num][1].weight_hh_l0.data
        the_Bias_ih = list(model.named_modules())[layer_num][1].bias_ih_l0.data
        the_Bias_hh = list(model.named_modules())[layer_num][1].bias_hh_l0.data
        loss_fun = loss_function
        loss_fun.to(device)
        gradient = torch.zeros(the_wih.shape[0]).to(device)  # shape[0]
        hinden_num = the_wih.shape[0]
        for inputs, targets in prune_datasetloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fun(output, targets)
            loss.backward()
            gradient += torch.sum(list(model.named_modules())[layer_num][1].weight_ih_l0.grad * the_wih,dim=1)
            gradient += torch.sum(list(model.named_modules())[layer_num][1].weight_hh_l0.grad * the_whh,dim=1)
            gradient += list(model.named_modules())[layer_num][1].bias_ih_l0.grad * the_Bias_ih
            gradient += list(model.named_modules())[layer_num][1].bias_hh_l0.grad * the_Bias_hh
        gradient = gradient[0:int(hinden_num/3)] + gradient[int(hinden_num/3):int(hinden_num/3*2)] + gradient[int(hinden_num/3*2):]
    else:
        the_weight = list(model.named_modules())[layer_num][1].weight.data
        dim = the_weight.dim()
        the_bias = list(model.named_modules())[layer_num][1].bias.data.view(the_weight.shape[0], *([1] * (dim - 1)))
        # print(the_weight.shape)
        loss_fun = loss_function
        loss_fun.to(device)
        gradient = torch.zeros(the_weight.shape).to(device)#shape[0]
        # print(gradients[f'{layer_num}'].shape)
        for inputs, targets in prune_datasetloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fun(output, targets)
            loss.backward()
            gradient += list(model.named_modules())[layer_num][1].weight.grad * the_weight
            gradient += list(model.named_modules())[layer_num][1].bias.grad.view(the_weight.shape[0], *([1] * (dim - 1))) * the_bias
        gradient = torch.sum(gradient,dim = list(range(1,gradient.dim())))
    print(f'The Reconstruction Error of the {layer_num}th layer is calculated.')
    gradients.clear()

    return gradient
################################################
################################################
# The only difference between the two versions is
# whether the gradient is zeroed out on each
# calculation, and empirically the first version
#（get_ReconstructionError）works better
def get_ReconstructionError2(model,prune_datasetloader,layer_num,device,loss_function):
    #
    # model: is the model we want to prune.
    # prune_datasetloader : is （from torch.utils.data import DataLoader）.
    # layer_num : it the aim layer we want to compute Distance.
    # device: torch.device('cpu') or torch.device('cuda') .
    # loss_function: The loss function used for model training.
    #
    model.to(device)
    model.train()
    if isinstance(list(model.named_modules())[layer_num][1],nn.GRU):
        the_wih = list(model.named_modules())[layer_num][1].weight_ih_l0.data
        the_whh = list(model.named_modules())[layer_num][1].weight_hh_l0.data
        the_Bias_ih = list(model.named_modules())[layer_num][1].bias_ih_l0.data
        the_Bias_hh = list(model.named_modules())[layer_num][1].bias_hh_l0.data
        loss_fun = loss_function
        loss_fun.to(device)
        gradient = torch.zeros(the_wih.shape[0]).to(device)  # shape[0]
        hinden_num = the_wih.shape[0]
        for inputs, targets in prune_datasetloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fun(output, targets)
            loss.backward()
            gradient += torch.sum(list(model.named_modules())[layer_num][1].weight_ih_l0.grad * the_wih,dim=1)
            gradient += torch.sum(list(model.named_modules())[layer_num][1].weight_hh_l0.grad * the_whh,dim=1)
            gradient += list(model.named_modules())[layer_num][1].bias_ih_l0.grad * the_Bias_ih
            gradient += list(model.named_modules())[layer_num][1].bias_hh_l0.grad * the_Bias_hh
        gradient = gradient[0:int(hinden_num/3)] + gradient[int(hinden_num/3):int(hinden_num/3*2)] + gradient[int(hinden_num/3*2):]
    else:
        the_weight = list(model.named_modules())[layer_num][1].weight.data
        dim = the_weight.dim()
        the_bias = list(model.named_modules())[layer_num][1].bias.data.view(the_weight.shape[0], *([1] * (dim- 1)))
        loss_fun = nn.CrossEntropyLoss()
        loss_fun.to(device)
        model.zero_grad()
        for inputs, targets in prune_datasetloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fun(output, targets)
            loss.backward()
        gradient = (list(model.named_modules())[layer_num][1].weight.grad * the_weight +
                    list(model.named_modules())[layer_num][1].bias.grad.view(the_weight.shape[0],*([1] * (dim - 1))) * the_bias)
        gradient = torch.sum(gradient,dim = list(range(1,gradient.dim())))
    print(f'The Reconstruction Error of the {layer_num}th layer is calculated.')
    gradients.clear()

    return gradient

########################################################################################################################

def get_k_list(results,the_list_of_layers_to_prune,FDR_level):
    #
    # results: resluts save from main
    # the_list_of_layers_to_compute_Distance: a list of the layer number which we want to get Distance
    # the_list_of_layers_to_prune: a list of the layer number which we want to get ReconstructionError
    # FDR_level

    results[f'D_{len(the_list_of_layers_to_prune) - 1}_s2b_idx'] = list(range(1000))
    k_list = []
    for j in range(len(the_list_of_layers_to_prune)):
        if (j + 1) == len(the_list_of_layers_to_prune):
            k_list.append(0)
            break
        neuron_number = len(results[f'D_{j}_s2b_idx'])
        # m = int((1-2*FDR_level)*neuron_number)
        for i in range(int(0.99 * neuron_number)):
            k = int(0.99 * neuron_number) - i
            intersection = list(
                set(results[f'D_{j}_s2b_idx'][:k]) & set(results[f'RE_{j}_hat_s2b_idx'][int(neuron_number - k):]))
            # print(len(intersection)/k)
            if len(intersection) / k <= FDR_level:
                print(f'The prunable set size for layer {the_list_of_layers_to_prune[j]}th is {k}')
                k_list.append(k)
                break
            if i == (int(0.99 * neuron_number) - 1):
                print(f'The {the_list_of_layers_to_prune[j]}th layer is inspected and does not need to be pruned')
                k_list.append(0)
                break
    return k_list


#######################################################################################################################

def FAIR_Pruner_get_results(model_path,data_path,results_save_path,the_list_of_layers_to_prune,the_list_of_layers_to_compute_Distance,loss_function,device,class_num,the_batch_for_compute_distance=16,max_sample_for_compute_distance=1e+10):
    with open(data_path, 'rb') as f:
        prune_datasetloader = pickle.load(f)
    model = torch.load(model_path)
    prunedata = get_prunedata(prune_datasetloader, the_batch_for_compute_distance, class_num, max_sample_for_compute_distance)  ######!!!!!!!!!!!!!!!!!!!!
    results = {}
    for layer_num in range(len(the_list_of_layers_to_compute_Distance)):
        results[f'RE_{layer_num}_hat'] = get_ReconstructionError(model, prune_datasetloader,
                                                                layer_num=the_list_of_layers_to_prune[layer_num],
                                                                device=device, loss_function=loss_function)
        results[f'RE_{layer_num}_hat_s2b_idx'] = torch.argsort(results[f'RE_{layer_num}_hat']).tolist()
        torch.cuda.empty_cache()
        results[f'D_{layer_num}'] = get_Distance(model, prunedata, layer_num=the_list_of_layers_to_compute_Distance[layer_num],
                                                 device=device)
        results[f'D_{layer_num}_s2b_idx'] = torch.argsort(results[f'D_{layer_num}']).tolist()
        torch.cuda.empty_cache()
    with open(results_save_path, 'wb') as file:
        pickle.dump(results, file)
    return results

def Generate_model_after_pruning(tiny_model,original_model_path,tiny_model_save_path,results,k_list,the_list_of_layers_to_prune):
    original_model = torch.load(original_model_path)
    model_after_pruning_layername = {}
    for i, (name, layer) in enumerate(tiny_model.named_modules()):
        model_after_pruning_layername[f'{i}'] = layer
    layername = {}
    layer_idx = []
    covn2linear = 0
    for i, (name, layer) in enumerate(original_model.named_modules()):
        layername[f'{i}'] = layer
        if hasattr(layer, 'weight') and isinstance(layer.weight, torch.nn.Parameter):
            layer_idx.append(i)
        if isinstance(layer, nn.Linear) and covn2linear == 0:
            covn2linear = 1
            layer_num_conv2linear = layer_idx[-1]
    with torch.no_grad():
        for j, i in enumerate(the_list_of_layers_to_prune):
            position = results[f'D_{j}_s2b_idx']
            position = sorted(position[k_list[j]:])  # No pruning position
            # print(position)
            if j==0:
                if model_after_pruning_layername[f'{i}'].weight.dim() == 4:
                    model_after_pruning_layername[f'{i}'].weight = nn.Parameter(layername[f'{i}'].weight[position, :, :, :])
                    model_after_pruning_layername[f'{i}'].bias = nn.Parameter(layername[f'{i}'].bias[position])
                if layername[f'{i}'].weight.dim() == 2:
                    model_after_pruning_layername[f'{i}'].weight = nn.Parameter(layername[f'{i}'].weight[position, :])
                    model_after_pruning_layername[f'{i}'].bias = nn.Parameter(layername[f'{i}'].bias[position])
                old_position = position
            elif i==layer_num_conv2linear:
                xjbg = layername[f'{i}'].weight[position, :]
                old_position =  [ele for subele in [list(range(i1*layername[f'{int(i-2)}'].output_size[0]*layername[f'{int(i-2)}'].output_size[1],
                                                               (i1+1)*layername[f'{int(i-2)}'].output_size[0]*layername[f'{int(i-2)}'].output_size[1])) for i1 in old_position] for ele in subele]
                model_after_pruning_layername[f'{i}'].weight = nn.Parameter(xjbg[:, old_position])
                model_after_pruning_layername[f'{i}'].bias = nn.Parameter(layername[f'{i}'].bias[position])
                old_position = position
            else:
                if model_after_pruning_layername[f'{i}'].weight.dim() == 4:
                    xjbg = layername[f'{i}'].weight[position, :, :, :]
                    model_after_pruning_layername[f'{i}'].weight = nn.Parameter(xjbg[:, old_position, :, :])
                    model_after_pruning_layername[f'{i}'].bias = nn.Parameter(layername[f'{i}'].bias[position])
                if layername[f'{i}'].weight.dim() == 2:
                    xjbg = layername[f'{i}'].weight[position, :]
                    model_after_pruning_layername[f'{i}'].weight = nn.Parameter(xjbg[:, old_position])
                    model_after_pruning_layername[f'{i}'].bias = nn.Parameter(layername[f'{i}'].bias[position])
                old_position = position
    print(f'parameters number: {sum(p.numel() for p in tiny_model.parameters() if p.requires_grad)}')
    print(f'pruning rate: {1 - sum(p.numel() for p in tiny_model.parameters() if p.requires_grad) / sum(p.numel() for p in original_model.parameters() if p.requires_grad)}')
    torch.save(tiny_model,tiny_model_save_path)

    return tiny_model

if __name__ == '__main__':
    model_path = r'../CIFAR10_vgg16.pht'
    data_path =  r'C:\Users\Administrator\PycharmProjects\lcq\Data\cifar10_train_dataloader.pkl'
    results_save_path = 'test_res.pkl'
    the_list_of_layers_to_prune = [2,4,7,9,12,14,16,19,21,23,26,28,30,35,38,41]
    the_list_of_layers_to_compute_Distance = [3,5,8,10,13,15,17,20,22,24,27,29,31,36,39]
    loss_function = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_num = 10
    results = FAIR_Pruner_get_results(model_path, data_path, results_save_path, the_list_of_layers_to_prune,
                the_list_of_layers_to_compute_Distance, loss_function, device, class_num,
                the_batch_for_compute_distance=32, max_sample_for_compute_distance=1e+10)
    k_list = get_k_list(results,   the_list_of_layers_to_prune,0.05)
    class Tiny_model_class(nn.Module):
        def __init__(self):
            super(Tiny_model_class, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64 - k_list[0], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64 - k_list[0], 64 - k_list[1], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64 - k_list[1], 128 - k_list[2], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128 - k_list[2], 128 - k_list[3], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(128 - k_list[3], 256 - k_list[4], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256 - k_list[4], 256 - k_list[5], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256 - k_list[5], 256 - k_list[6], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(256 - k_list[6], 512 - k_list[7], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512 - k_list[7], 512 - k_list[8], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512 - k_list[8], 512 - k_list[9], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(512 - k_list[9], 512 - k_list[10], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512 - k_list[10], 512 - k_list[11], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512 - k_list[11], 512 - k_list[12], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

            self.classifier = nn.Sequential(
                nn.Linear((512 - k_list[12]) * 7 * 7, 4096 - k_list[13]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096 - k_list[13], 4096 - k_list[14]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096 - k_list[14], 1000 - k_list[15])
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    tiny_model = Tiny_model_class()
    tiny_model = Generate_model_after_pruning(tiny_model,model_path,
                                 'test_tiny_model.pht',
                                 results,k_list,
                                 the_list_of_layers_to_prune)
