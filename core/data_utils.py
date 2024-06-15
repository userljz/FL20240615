import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.models as models
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10, CIFAR100
import logging
from logging import debug, info
from addict import Dict
import random
import math



def _convert_image_to_rgb(image):
    return image.convert("RGB")




class dsDict:
    def __init__(self, ds, mean, std):
        self.dataset = ds
        self.mean = mean
        self.std = std
        
        self.train_transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])


# cifar10_dict = dsDict(CIFAR10, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
# cifar100_dict = dsDict(CIFAR100, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    n_classes = train_labels.max() + 1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    
    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1] * len(k_idcs)).
                                                  astype(int))):
            client_idcs[i] += [idcs]
    
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    
    # 后处理:防止某个Client一个样本也没有分到, 保证每个Client至少分到一个Sample
    sorted_client_idcs = sorted(client_idcs, key=len)
    
    client_idcs_without_0 = []
    used_data_from_other_client = 0
    for idx, client_i in enumerate(sorted_client_idcs[:-1]):
        if len(client_i) == 0:
            _client_i = np.array([sorted_client_idcs[-1][used_data_from_other_client]])
            used_data_from_other_client += 1
            client_idcs_without_0.append(_client_i)
        else:
            client_idcs_without_0.append(client_i)
    client_idcs_without_0.append(sorted_client_idcs[-1][used_data_from_other_client:])
    return client_idcs_without_0




# 20240103 Currently Use
def load_dataloader_from_generate(dataset_name, model_name, batch_size, dirichlet_alpha, dataloader_num=1, dataset_root="/home/ljz/dataset"):
    if dataset_name == 'cifar10':
        train_img = torch.load(f'{dataset_root}/cifar10_generated/cifar10Train_RN50_imgembV1.pth')
        train_label = torch.load(f'{dataset_root}/cifar10_generated/cifar10Train_labelsV1.pth')
        train_img = train_img.float()
        train_img_label_list = [(train_img[i], train_label[i]) for i in range(len(train_label))]
        
        test_img = torch.load(f'{dataset_root}/cifar10_generated/cifar10Test_RN50_imgembV1.pth')
        test_label = torch.load(f'{dataset_root}/cifar10_generated/cifar10Test_labelsV1.pth')
        test_img = test_img.float()
        test_img_label_list = [(test_img[i], test_label[i]) for i in range(len(test_label))]
    
    elif dataset_name == 'PathMNIST' or dataset_name == 'OrganAMNIST' or dataset_name == 'emnist62':
        if model_name == 'ViT-B/32':
            train_img = torch.load(
                f'{dataset_root}/{dataset_name}_generated_vitb32/{dataset_name}_vitb32Train_imgemb.pth')
            train_label = torch.load(
                f'{dataset_root}/{dataset_name}_generated_vitb32/{dataset_name}_vitb32Train_labels.pth')
            train_img = train_img.float()
            train_img_label_list = [(train_img[i], train_label[i]) for i in range(len(train_label))]
            
            test_img = torch.load(
                f'{dataset_root}/{dataset_name}_generated_vitb32/{dataset_name}_vitb32Test_imgemb.pth')
            test_label = torch.load(
                f'{dataset_root}/{dataset_name}_generated_vitb32/{dataset_name}_vitb32Test_labels.pth')
            test_img = test_img.float()
            test_img_label_list = [(test_img[i], test_label[i]) for i in range(len(test_label))]

    elif dataset_name == 'cifar100':
        # ============================================================================ #
        # CLIP RN50 Embedding
        if model_name == 'RN50':
            train_img = torch.load(f'{dataset_root}/cifar100_generated/cifar100Train_RN50_imgembV1.pth')
            train_label = torch.load(f'{dataset_root}/cifar100_generated/cifar100Train_labelsV1.pth')
            train_img = train_img.float()
            train_img_label_list = [(train_img[i], train_label[i]) for i in range(len(train_label))]
            
            test_img = torch.load(f'{dataset_root}/cifar100_generated/cifar100Test_RN50_imgembV1.pth')
            test_label = torch.load(f'{dataset_root}/cifar100_generated/cifar100Test_labelsV1.pth')
            test_img = test_img.float()
            test_img_label_list = [(test_img[i], test_label[i]) for i in range(len(test_label))]
        
        # ============================================================================ #
        # CLIP ViT-B/32 Embedding
        elif model_name == 'ViT-B/32':
            train_img = torch.load(f'{dataset_root}/cifar100_generated_vitb32/cifar100_vitb32Train_imgemb.pth')
            train_label = torch.load(f'{dataset_root}/cifar100_generated_vitb32/cifar100_vitb32Train_labels.pth')
            train_img = train_img.float()
            train_img_label_list = [(train_img[i], train_label[i]) for i in range(len(train_label))]
            
            test_img = torch.load(f'{dataset_root}/cifar100_generated_vitb32/cifar100_vitb32Test_imgemb.pth')
            test_label = torch.load(f'{dataset_root}/cifar100_generated_vitb32/cifar100_vitb32Test_labels.pth')
            test_img = test_img.float()
            test_img_label_list = [(test_img[i], test_label[i]) for i in range(len(test_label))]
        
        elif model_name == 'ViT-B32-timm':
            train_img = torch.load(
                f'{dataset_root}/cifar100_generated_vitb32/cifar100_{model_name}Train_imgemb.pth')
            train_label = torch.load(
                f'{dataset_root}/cifar100_generated_vitb32/cifar100_{model_name}Train_labels.pth')
            train_img = train_img.float()
            train_img_label_list = [(train_img[i], train_label[i]) for i in range(len(train_label))]
            
            test_img = torch.load(
                f'{dataset_root}/cifar100_generated_vitb32/cifar100_{model_name}Test_imgemb.pth')
            test_label = torch.load(
                f'{dataset_root}/cifar100_generated_vitb32/cifar100_{model_name}Test_labels.pth')
            test_img = test_img.float()
            test_img_label_list = [(test_img[i], test_label[i]) for i in range(len(test_label))]
        
        
       
    else:
        print('Please specify the dataset')
    
    
    
    if dataloader_num == 1:
        val_size = int(len(train_img_label_list) * 0.2) + 1
        train_size = len(train_img_label_list) - val_size
        train_subset, val_subset = torch.utils.data.random_split(train_img_label_list, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=batch_size, shuffle=False, num_workers=7, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            test_img_label_list,
            batch_size=batch_size, shuffle=False, num_workers=7, pin_memory=True)
        
        return [train_loader], [val_loader], test_loader
    
    elif dataloader_num > 1:
        test_loader = torch.utils.data.DataLoader(
            test_img_label_list,
            batch_size=batch_size, shuffle=False, num_workers=7, pin_memory=True)
        
        # return non-iid multi-clients trainloaders
        labels = np.array([i[1] for i in train_img_label_list])
        # labels = np.array(train_label)
        client_idcs = dirichlet_split_noniid(labels, dirichlet_alpha, dataloader_num)
        client_trainsets = []
        for client_i in client_idcs:
            client_trainsets.append(Subset(train_img_label_list, client_i))
        
        train_loaders = []
        val_loaders = []
        for trainset in client_trainsets:
            # Calculate the number of samples for validation set (20% of the total)
            val_size = int(len(trainset) * 0.2) + 1
            train_size = len(trainset) - val_size
            
            train_subset, val_subset = torch.utils.data.random_split(trainset, [train_size, val_size])
            
            # Create DataLoader for training and validation sets
            train_loaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True))
            val_loaders.append(DataLoader(val_subset, batch_size=batch_size, shuffle=False))
        
        
        return train_loaders, val_loaders, test_loader


def label_collect(test_loader):
    label_list = []
    for img, label in test_loader:
        label_list.append(label)
    label_list = torch.cat(label_list)
    label_list = [i.item() for i in label_list]
    label_list = set(label_list)
    return label_list


if __name__ == '__main__':
    args = Dict()
    args.cfg.dirichlet_alpha = 0.1
    args.cfg.num_clients = 100
    batch_size = 32
    dataset_name = 'cifar100'
    model_name = 'ViT-B/32'
    train_loader_list, test_loader = load_dataloader_from_generate(args, dataset_name, dataloader_num=10)
    for index, train_loader_i in enumerate(train_loader_list):
        label = label_collect(train_loader_i)
        print(f'[{index}]({len(label)}){label=}')
    
    #     # print(f'{len(train_loader_i)=}')
    #     for i in train_loader_i:
    #         # print(len(i))
    
    # print(f'{len(test_loader)=}')