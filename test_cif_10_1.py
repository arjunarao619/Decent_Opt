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

sp_0p99_2048 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621665989_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.99__dataset-cifar10_unit-2048',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622444914_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.99__dataset-cifar10_unit-2048',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622464263_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.99__dataset-cifar10_unit-2048',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622483664_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.99__dataset-cifar10_unit-2048',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622503697_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.99__dataset-cifar10_unit-2048'
                ]

sp_0p98_1024 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621672959_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.98__dataset-cifar10_unit-1024',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622451375_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.98__dataset-cifar10_unit-1024',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622470746_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.98__dataset-cifar10_unit-1024',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622490229_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.98__dataset-cifar10_unit-1024',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622510587_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.98__dataset-cifar10_unit-1024'
                ]

sp_0p96_512 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621678553_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.96__dataset-cifar10_unit-512',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622456946_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.96__dataset-cifar10_unit-512',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622476344_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.96__dataset-cifar10_unit-512',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622495814_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.96__dataset-cifar10_unit-512',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622516149_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.96__dataset-cifar10_unit-512'
                ]

sp_0p92_256 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621682047_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.92__dataset-cifar10_unit-256',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622460320_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.92__dataset-cifar10_unit-256',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622479709_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.92__dataset-cifar10_unit-256',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622499429_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.92__dataset-cifar10_unit-256',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622519517_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.92__dataset-cifar10_unit-256'
              ]

sp_0p84_128 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621684227_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.84__dataset-cifar10_unit-128',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622462507_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.84__dataset-cifar10_unit-128',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622481914_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.84__dataset-cifar10_unit-128',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622501884_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.84__dataset-cifar10_unit-128',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622521708_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.84__dataset-cifar10_unit-128'
                ]


#95% sparsification
sp_0p95_2048 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621685931_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.95__dataset-cifar10_unit-2048',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622350932_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.95__dataset-cifar10_unit-2048',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622373673_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.95__dataset-cifar10_unit-2048',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622396352_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.95__dataset-cifar10_unit-2048',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622419884_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.95__dataset-cifar10_unit-2048']

sp_0p90_1024 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621693377_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9__dataset-cifar10_unit-1024',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622358335_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9__dataset-cifar10_unit-1024',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622381037_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9__dataset-cifar10_unit-1024',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622403927_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9__dataset-cifar10_unit-1024',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622427396_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9__dataset-cifar10_unit-1024']

sp_0p80_512 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621699886_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-512',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622364921_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-512',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622387623_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-512',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622410934_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-512',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622436132_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-512']

sp_0p60_256 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621703643_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.6__dataset-cifar10_unit-256',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622368712_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.6__dataset-cifar10_unit-256',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622391410_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.6__dataset-cifar10_unit-256',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622414922_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.6__dataset-cifar10_unit-256',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622439989_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.6__dataset-cifar10_unit-256']

sp_0p20_128 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621706432_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.2__dataset-cifar10_unit-128',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622371545_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.2__dataset-cifar10_unit-128',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622394233_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.2__dataset-cifar10_unit-128',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622417739_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.2__dataset-cifar10_unit-128',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1622442795_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.2__dataset-cifar10_unit-128']

#93.75% sparsification
sp_0p93_2048 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622523687_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9375__dataset-cifar10_unit-2048',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622547514_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9375__dataset-cifar10_unit-2048',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622571389_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9375__dataset-cifar10_unit-2048',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622595434_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9375__dataset-cifar10_unit-2048',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621757828_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9375__dataset-cifar10_unit-2048']

sp_0p87_1024 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621765281_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.875__dataset-cifar10_unit-1024',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622555154_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.875__dataset-cifar10_unit-1024',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622531242_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.875__dataset-cifar10_unit-1024',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622578975_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.875__dataset-cifar10_unit-1024',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622603537_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.875__dataset-cifar10_unit-1024'
               ]

sp_0p75_512 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622538194_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.75__dataset-cifar10_unit-512',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622562059_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.75__dataset-cifar10_unit-512',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622585877_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.75__dataset-cifar10_unit-512',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621772113_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.75__dataset-cifar10_unit-512',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622610476_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.75__dataset-cifar10_unit-512']

sp_0p50_256 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622542038_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.5__dataset-cifar10_unit-256',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622565914_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.5__dataset-cifar10_unit-256',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622589771_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.5__dataset-cifar10_unit-256',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621776031_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.5__dataset-cifar10_unit-256',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622614435_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.5__dataset-cifar10_unit-256']

sp_0p0_128 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622545155_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.0__dataset-cifar10_unit-128',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622569006_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.0__dataset-cifar10_unit-128',
             '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622592901_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.0__dataset-cifar10_unit-128',
             '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621779092_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.0__dataset-cifar10_unit-128',
             '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622617524_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.0__dataset-cifar10_unit-128']

#90% sparsification

sp_0p90_2048 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622619918_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9__dataset-cifar10_unit-2048',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622640423_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9__dataset-cifar10_unit-2048',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622660673_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9__dataset-cifar10_unit-2048',
               '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621708530_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9__dataset-cifar10_unit-2048',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622681783_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.9__dataset-cifar10_unit-2048'
               ]
sp_0p80_1024 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621716284_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-1024',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622627711_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-1024',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622648122_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-1024',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622668547_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-1024',
                '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622689575_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.8__dataset-cifar10_unit-1024'
               ]

sp_0p60_512 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621722482_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.6__dataset-cifar10_unit-512',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622633911_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.6__dataset-cifar10_unit-512',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622654244_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.6__dataset-cifar10_unit-512',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622674796_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.6__dataset-cifar10_unit-512',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622695761_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.6__dataset-cifar10_unit-512']

sp_0p20_256 = ['/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/equal_data_0p99_0p95/top_k/1621726195_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.2__dataset-cifar10_unit-256',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622637689_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.2__dataset-cifar10_unit-256',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622657954_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.2__dataset-cifar10_unit-256',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622678812_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.2__dataset-cifar10_unit-256',
              '/home/aarao8/choco_2/ChocoSGD/dl_code/data/checkpoint/cifar10/mlp/test/1622699459_lr-0.1_epochs-300_batchsize-128_num_mpi_process_8_topology-ring_comm_info-compress_top_k-0.2__dataset-cifar10_unit-256']

count = -1
u2 = [2048,2048,2048,2048,2048,1024,1024,1024,1024,1024,512,512,512,512,512,256,256,256,256,256,128,128,128,128,128,
2048,2048,2048,2048,2048,1024,1024,1024,1024,1024,512,512,512,512,512,256,256,256,256,256,128,128,128,128,128,
2048,2048,2048,2048,2048,1024,1024,1024,1024,1024,512,512,512,512,512,256,256,256,256,256,128,128,128,128,128,
2048,2048,2048,2048,2048,1024,1024,1024,1024,1024,512,512,512,512,512,256,256,256,256,256
]
u2 = [2048,1024,512,256,128, 2048,1024,512,256,128, 2048,1024,512,256,128, 2048,1024,512,256]
accs = []
losses = []
# for mod in [sp_0p99_2048,sp_0p98_1024,sp_0p96_512,sp_0p92_256,sp_0p84_128,sp_0p95_2048,
#             sp_0p90_1024,sp_0p80_512,sp_0p60_256,sp_0p20_128, sp_0p93_2048, sp_0p87_1024, 
#             sp_0p75_512, sp_0p50_256, sp_0p0_128, sp_0p90_2048, sp_0p80_1024, sp_0p60_512, 
#             sp_0p20_256]:
mod = sp_0p93_2048
unit = 2048

for i in range(len(mod)):
    
    conf.weight_path = mod[i] + '/0/' +  'checkpoint.pth.tar'
    conf.mlp_hidden_size = unit
    conf.data='cifar10'
    conf.mlp_num_layers = 1
    conf.drop_rate = 0.0
    model = models.__dict__['mlp'](conf)

    ep_50 = torch.load(conf.weight_path)
    #from IPython import embed;embed()

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # testset = cif10p1(
    #     root='./data1', download=True, transform=transform_test)
    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=128, shuffle=True, num_workers=1)

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
        
        #print("Accuracy: {} Loss {} Epochs {}".format(np.mean(test_acc),np.mean(test_loss), ep_50['current_epoch']))
        x.add_row([unit, np.mean(test_acc), np.mean(test_loss)])
        accs.append(np.mean(test_acc))
        losses.append(np.mean(test_loss))
#print(x)
print("{} \pm {}".format(round(np.mean(accs),3),round(np.std(accs),3)))
# print(np.mean(accs))
# print(np.std(accs))
# print(np.mean(losses))
# print(np.std(losses))