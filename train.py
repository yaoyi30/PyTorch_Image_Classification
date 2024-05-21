#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

import os
import torch
from torchvision import transforms, datasets
import torch.nn as nn
from models.Simplify_Net import Simplify_Net
from utils import train_and_val,plot_acc,plot_loss,plot_lr
import argparse
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('Image Classification Train', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,help='Batch size for training')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--input_size', default=[224,224],nargs='+',type=int,help='images input size')
    parser.add_argument('--data_path', default='./datasets/', type=str,help='dataset path')

    parser.add_argument('--init_lr', default=1e-5, type=float,help='intial lr')
    parser.add_argument('--max_lr', default=1e-3, type=float,help='max lr')
    parser.add_argument('--weight_decay', default=1e-5, type=float,help='weight decay')

    parser.add_argument('--nb_classes', default=10, type=int,help='number of the classification types')
    parser.add_argument('--output_dir', default='./output_dir',help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)

    return parser



def main(args):

    device = torch.device(args.device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_transform =  transforms.Compose([
                                       transforms.Resize(args.input_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

    val_transform =  transforms.Compose([
                                       transforms.Resize(args.input_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

    train_dataset = datasets.ImageFolder(os.path.join(args.data_path,'train'), transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)
    len_train = len(train_dataset)

    val_dataset = datasets.ImageFolder(os.path.join(args.data_path,'val'), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)
    len_val = len(val_dataset)

    model = Simplify_Net(args.nb_classes)
    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr, total_steps=args.epochs, verbose=True)

    history = train_and_val(args.epochs, model, train_loader, len_train,val_loader, len_val,loss_function, optimizer,scheduler,args.output_dir,device)

    plot_loss(np.arange(0,args.epochs),args.output_dir, history)
    plot_acc(np.arange(0,args.epochs),args.output_dir, history)
    plot_lr(np.arange(0,args.epochs),args.output_dir, history)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
