import gc
import os
from argparse import ArgumentParser

import cox
import numpy.random

from Utils.datasets import DATASETS
from Utils import helpers
import time

from Utils.helpers import write_batch
from models import enc_model
from train_model import train_model
import numpy as np
import torch as ch
import csv

parser = ArgumentParser()
parser.add_argument('--dataset', choices=['cifar', 'mnist', 'fashion', 'income', 'activity', 'letters', 'imagenet', 'cifar100'], default='mnist')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--attack_layer', type=int, default=0)
parser.add_argument('--attack_type', choices=['none', 'inversion', 'denoiser', 'Bayes'], default='none')
parser.add_argument('--atk_model_knowledge', choices=['none', 'pattern', 'exact'], default='exact')
parser.add_argument('--noise_knowledge', choices=['none', 'exist', 'pattern', 'exact'], default='exact')
parser.add_argument('--noise_type', choices=['none', 'phoni', 'gau', 'uni', 'lap'], default='none')
parser.add_argument('--noise_a', type=float, default=-1)
parser.add_argument('--noise_b', type=float, default=1)
parser.add_argument('--alpha', type=float, default=1.5)
parser.add_argument('--beta', type=float, default=0.005)
parser.add_argument('--lam', type=float, default=0.01)
parser.add_argument('--tol', type=float, default=0.15)
parser.add_argument('--noise_scale', type=float, default=10.0)
parser.add_argument('--data_aug', type=bool, default=False)
parser.add_argument('--save_images', type=bool, default=False)
parser.add_argument('--pretrain', type=bool, default=False)
parser.add_argument('--MI', choices=['DP', 'MI'], default='DP') #Not working for some reason, Have to manully toggle
parser.add_argument('--phoni_num', type=int, default=1)
parser.add_argument('--phoni_size', type=int, default=100)
parser.add_argument('--phoni_epoch', type=int, default=1000)
parser.add_argument('--atk_itr', type=int, default=10)
parser.add_argument('--atk_lr', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--noise_structure', default=[1000, 32, 32, 32])
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--num_attacked', type=int, default=10)
parser.add_argument('--attack_epoch', type=int, default=1000)
parser.add_argument('--image_names', default='default')
parser.add_argument('--multi_target', default='false')
parser.add_argument('--atk_sample', type=float, default=1)

args = parser.parse_args()


def arg_helper(args):
    if args.noise_type != 'phoni':
        args.phoni_num=1
    if args.dataset in ['cifar', 'cifar100']:
        if args.dataset == 'cifar':
            args.num_class = 10
        else:
            args.num_classes = 100
        if args.attack_layer == 0 or args.attack_layer == 1:
            args.noise_structure = [1000, 32, 32, 32]
        elif args.attack_layer == 2 or args.attack_layer == 5:
            args.noise_structure = [1000, 64, 16, 16]
        elif args.attack_layer == 3 or args.attack_layer == 4:
            args.noise_structure = [1000, 128, 16, 16]
        elif args.attack_layer == 6:
            args.noise_structure = [1000, 65536]
        elif args.attack_layer == 7:
            args.noise_structure = [1000, 1024]
    if args.dataset == "mnist" or args.dataset == "fashion":
        args.num_class = 10
        if args.attack_layer == 0 or args.attack_layer == 1:
            args.noise_structure = [1000, 8, 28, 28]
        elif args.attack_layer == 2:
            args.noise_structure = [1000, 3, 28, 28]
        elif args.attack_layer == 3:
            args.noise_structure = [1000, 4*28*28]
        elif args.attack_layer == 4:
            args.noise_structure = [1000, 1024]
        elif args.attack_layer == 5:
            args.noise_structure = [1000, 64]
    if args.dataset == "letters":
        args.num_class = 26
    if args.dataset == "income":
        args.num_class = 2
        args.noise_structure = [1000, 32]
    if args.dataset == "activity":
        args.num_class = 6
        args.noise_structure = [1000, 32]
    return args


def main(args):
    numpy.random.RandomState(42)
    print(args)

    # Use for batch runs for experiment
    MSEs = []
    SSIMs = []
    PSNRs = []
    Losses = []
    Accs = []
    runs = 1
    for i in range(runs):

        print('\nRunning Batch: ', i+1, ' of ', runs)
        if (args.dataset in ['cifar', 'mnist', 'fashion', 'letters', 'cifar100']):
            data_path = os.path.expandvars(args.dataset)
            dataset = DATASETS[args.dataset](data_path)
            train_loader, val_loader = dataset.make_loaders(8, args.batch_size, args, data_aug=args.data_aug)
            train_loader = helpers.DataPrefetcher(train_loader)
            val_loader = helpers.DataPrefetcher(val_loader)
        else:
            dataset, train_loader, val_loader = helpers.get_other_data(args)
        loaders = (train_loader, val_loader)
        enc_Model = enc_model(args).to(args.device)
        starting_time = time.time()
        MSE, SSIM, PSNR, Loss, Acc = train_model(enc_Model, loaders, args, dataset)
        MSEs.append(MSE)
        SSIMs.append(SSIM)
        PSNRs.append(PSNR)
        Losses.append(Loss)
        Accs.append(Acc)
        end_time = time.time()
        total_time = end_time - starting_time
        print('Total Time: ', total_time)
        del dataset, train_loader, val_loader, loaders, enc_Model, MSE, SSIM, PSNR, Loss, Acc, starting_time, end_time, total_time
        gc.collect()
    write_batch(runs, MSEs, SSIMs, PSNRs, Losses, Accs, args)

if __name__ == '__main__':
    args = cox.utils.Parameters(args.__dict__)
    args = arg_helper(args)
    main(args)
