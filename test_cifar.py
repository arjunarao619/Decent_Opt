import os
import shutil
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
from prettytable import PrettyTable

import pickle
import matplotlib.pyplot as plt

##Test loss attempts

import pcode.models as models
import argparse
parser = argparse.ArgumentParser(description="PyTorch Training for ConvNet")
parser.add_argument("--data", default="cifar10", type=str)
parser.add_argument("--mlp_num_layers", default=1, type=int)
parser.add_argument("--mlp_hidden_size", default=64, type=int)
parser.add_argument("--drop_rate", default=0.0, type=float)
parser.add_argument("--weight_path", default='', type=str)
conf = parser.parse_args()


x = PrettyTable()

x.field_names = ["Units", "Acc", "Loss"]




units = [2048,1024,512,256,128,2048,1024,512,256,128,2048,1024,512]
accs = []
losses = []
for i in range(len(paths_topk)):
    conf.weight_path = paths_topk[i] + 'checkpoint_epoch_200.pth.tar'
    conf.mlp_hidden_size = units[i]
    conf.data='cifar10'
    conf.mlp_num_layers = 1
    conf.drop_rate = 0.0
    model = models.__dict__['mlp'](conf)

    ep_50 = torch.load(conf.weight_path)
    #from IPython import embed;embed()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = datasets.CIFAR10(
        root='./data1', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=True, num_workers=1)


    model.load_state_dict(ep_50['state_dict'])


    model.eval()
    test_loss = []
    test_acc = []
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss().to('cpu')
    with torch.no_grad():
        for batch,(input,target) in enumerate(testloader):
            input = input.to('cpu')
            target = target.to('cpu')
            output = model(input)
            _, predicted = output.max(1)
            loss = criterion(output,target)
            test_loss.append(loss.item()) 
            total+= target.size(0)
            correct += predicted.eq(target).sum().item()
            percentage = 100.*correct/total 
            test_acc.append(percentage)
        
        print("Accuracy: {} Loss {}".format(np.mean(test_acc),np.mean(test_loss)))
        x.add_row([units[i], np.mean(test_acc), np.mean(test_loss)])
        accs.append(np.mean(test_acc))
        losses.append(np.mean(test_loss))
print(x)
print(accs)
print(losses)
