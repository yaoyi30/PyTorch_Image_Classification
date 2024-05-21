#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

import argparse
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from torchvision import transforms, datasets
import torch
import os
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from models.Simplify_Net import Simplify_Net
import matplotlib.pyplot as plt
import seaborn as sns

def get_args_parser():
    parser = argparse.ArgumentParser('Eval Model', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,help='Batch size for training')
    parser.add_argument('--input_size', default=[224,224],nargs='+',type=int,help='images input size')
    parser.add_argument('--data_path', default='./datasets/', type=str,help='dataset path')
    parser.add_argument('--weights', default='./output_dir/best.pth', type=str,help='dataset path')
    parser.add_argument('--nb_classes', default=10, type=int,help='number of the classification types')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)

    return parser


def main(args):

    device = torch.device(args.device)

    val_transform =  transforms.Compose([
                                       transforms.Resize(args.input_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

    val_dataset = datasets.ImageFolder(os.path.join(args.data_path,'val'), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)

    model = Simplify_Net(args.nb_classes)

    checkpoint = torch.load(args.weights, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=True)
    print(msg)

    model.to(device)
    model.eval()

    classes = val_dataset.classes

    act = nn.Softmax(dim=1)

    y_true, y_pred = [], []
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as pbar:
            for images, labels in val_loader:
                outputs = act(model(images.to(device)))
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.cpu()
                y_pred.extend(predicted.numpy())
                y_true.extend(labels.cpu().numpy())
                pbar.update(1)

    ac = accuracy_score(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

    df = pd.DataFrame(cr).transpose()
    df.to_csv("result.csv", index=True)
    print("Accuracy is :", ac)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes, cmap='Blues', fmt="d")

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.clf()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
