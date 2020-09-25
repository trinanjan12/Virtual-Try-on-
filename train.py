import datetime
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn  # TODO ??
from config import Config  # TODO Add later # DONE
from torch.utils.data import DataLoader
from torchvision import transforms

import torch.nn as nn
from data_process.train_dataset import RegularDatasetDensepose
from lib.geometric_matching_multi_gpu import GMM
from models.generation_model_new_joint_fixed import GenerationModel
from models.models import create_model  # TODO Remove
from torchvision import utils

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ['CUDA_VISIBLE_DEVCIES'] = '0, 1, 2, 3'
gpu_ids = len(os.environ['CUDA_VISIBLE_DEVCIES'].split(','))


def train(opt):
    augment = {}
    augment['1'] = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=10, translate=(
                0.1, 0.1), scale=(0.8, 1.2), shear=20),
            transforms.ToTensor()])  # change to [C, H, W]

    augment['2'] = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))])  # change to [C, H, W]

    augment['3'] = transforms.Compose(
        [
            transforms.ToTensor()])  # change to [C, H, W]

    train_dataset = RegularDatasetDensepose('./datasets/zalando/zolando_top_train.txt', 'train',
                                            'dataset', augment)
    val_dataset = RegularDatasetDensepose('./datasets/zalando/zolando_top_val.txt', 'val', 'dataset',
                                          augment)
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=opt.num_workers,
                                  batch_size=opt.batch_size_t,
                                  pin_memory=True)

    val_dataloader = DataLoader(val_dataset,
                                shuffle=True,
                                drop_last=False,
                                num_workers=opt.num_workers,
                                batch_size=opt.batch_size_v,
                                pin_memory=True)

    model = GenerationModel(opt)

    print('the length of dataset is %d' % len(train_dataset))
    for epoch in range(opt.start_epoch, opt.epoch):

        print('current G learning_rate is : ',
              model.get_learning_rate(model.optimizer_G))
        if opt.train_mode != 'gmm':
            print('current D learning_rate is : ',
                  model.get_learning_rate(model.optimizer_D))

        for i, data in enumerate(train_dataloader):

            # SET THE INPUT ACCORDING TO TRAIN MODE
            model.set_input(opt, data)

            # TRAIN THE NETWORKS ACCORDING TO TRAIN MODE
            model.optimize_parameters(opt)

            if i % opt.print_freq == 0:
                model.print_current_errors(opt, epoch, i,  epoch *
                                  len(train_dataloader) + i)
            if i % opt.val_freq == 0:
                model.save_result(val_dataloader, opt, epoch, epoch *
                                  len(train_dataloader) + i)
            model.update_learning_rate(opt, model.optimizer_G, epoch)
            if opt.train_mode != 'gmm':
                model.update_learning_rate(opt, model.optimizer_D, epoch)

        if epoch % opt.save_epoch_freq == 0:
            model.save_model(opt, epoch)


if __name__ == '__main__':
    opt = Config().parse()
    train(opt)
