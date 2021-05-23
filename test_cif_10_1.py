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
from load_cif_10_1 import CIFAR10_1 as cif10p1
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

paths=[
    '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621320878_l2-0.0001_lr-0.1_epochs-300_batchsize-128_basebatchsize-64_num_mpi_process_8_n_sub_process-1_topology-ring_optim-parallel_choco_v-stepsize-0.00625_comm_info-compress_random_k-0.99_warmup_epochs-0_unit-2048/0/',
    '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621330047_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.98__unit-1024/0/',
    '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621334515_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.96__unit-512/0/',
    '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621338248_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.92__unit-256/0/',
    '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621340709_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.84__unit-128/0/',

    '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621371383_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.95__dataset-cifar10_unit-2048/0/',
    '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621382215_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.9__dataset-cifar10_unit-1024/0/',
    '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621388862_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.8__dataset-cifar10_unit-512/0/',
    '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621393365_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.6__dataset-cifar10_unit-256/0/',
    '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621564632_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.2__dataset-cifar10_unit-128/0/',

    '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621567221_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.9__dataset-cifar10_unit-2048/0/',
    '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621577705_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.8__dataset-cifar10_unit-1024/0/',
    '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621585960_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.6__dataset-cifar10_unit-512/0/'
   '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621590420_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_random_k-0.2__dataset-cifar10_unit-256/0/'
]

paths_topk = [
    # '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621665989_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.99__dataset-cifar10_unit-2048/0/',
    # '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621672959_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.98__dataset-cifar10_unit-1024/0/',
    # '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621678553_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.96__dataset-cifar10_unit-512/0/',
    # '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621682047_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.92__dataset-cifar10_unit-256/0/',
    # '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621684227_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.84__dataset-cifar10_unit-128/0/',
    # '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621685931_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.95__dataset-cifar10_unit-2048/0/',
    # '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621693377_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9__dataset-cifar10_unit-1024/0/',
    # '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621699886_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-512/0/',
    # '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621703643_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.6__dataset-cifar10_unit-256/0/',
    # '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621706432_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.2__dataset-cifar10_unit-128/0/'
   #'/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621708530_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9__dataset-cifar10_unit-2048/0/'
    # '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621716284_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-1024/0/',
    # '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621722482_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.6__dataset-cifar10_unit-512/0/',
    # '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621726195_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.2__dataset-cifar10_unit-256/0/',
'/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621757828_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9375__dataset-cifar10_unit-2048/0/',
'/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621765281_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.875__dataset-cifar10_unit-1024/0/',
'/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621772113_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.75__dataset-cifar10_unit-512/0/',
'/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621776031_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.5__dataset-cifar10_unit-256/0/',
'/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/1621779092_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.0__dataset-cifar10_unit-128/0/',
]


units = [2048,1024,512,256,128,2048,1024,512,256,128,2048,1024,512]
u2 = [2048,1024,512,256,128]
accs = []
losses = []
for i in range(len(paths_topk)):
    conf.weight_path = paths_topk[i] + 'checkpoint_epoch_190.pth.tar'
    conf.mlp_hidden_size = u2[i]
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

    testset = cif10p1(
        root='./data1', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=True, num_workers=1)


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
        x.add_row([u2[i], np.mean(test_acc), np.mean(test_loss)])
        accs.append(np.mean(test_acc))
        losses.append(np.mean(test_loss))
print(x)
print(accs)
print(losses)
