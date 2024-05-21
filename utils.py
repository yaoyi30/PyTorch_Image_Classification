#####################################################
# Copyright(C) @ 2024.                              #
# Authored by 太阳的小哥(bilibili)                    #
# Email: 1198017347@qq.com                          #
# CSDN: https://blog.csdn.net/qq_38412266?type=blog #
#####################################################

import os
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_and_val(epochs, model, train_loader, len_train,val_loader, len_val,criterion, optimizer,scheduler,output_dir,device):

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    learning_rate = []
    best_acc = 0

    model.to(device)

    fit_time = time.time()
    for e in range(epochs):

        torch.cuda.empty_cache()

        since = time.time()
        training_loss = 0
        training_acc = 0
        model.train()
        with tqdm(total=len(train_loader)) as pbar:
            for image, label in train_loader:

                image = image.to(device)
                label = label.to(device)

                output = model(image)
                loss = criterion(output, label)
                _,predicted = torch.max(output, dim=1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                training_acc += torch.eq(predicted, label).sum().item()
                pbar.update(1)

        model.eval()
        validation_loss = 0
        validation_acc = 0

        with torch.no_grad():
            with tqdm(total=len(val_loader)) as pb:
                for image, label in val_loader:
                    image = image.to(device)
                    label = label.to(device)
                    output = model(image)

                    # loss
                    loss = criterion(output, label)
                    _, predicted = torch.max(output, dim=1)

                    validation_loss += loss.item()
                    validation_acc += torch.eq(predicted, label).sum().item()
                    pb.update(1)

        train_loss.append(training_loss / len(train_loader))
        val_loss.append(validation_loss / len(val_loader))

        train_acc.append(training_acc / len_train)
        val_acc.append(validation_acc / len_val)

        learning_rate.append(scheduler.get_last_lr())

        torch.save(model.state_dict(), os.path.join(output_dir,'last.pth'))
        if best_acc <(validation_acc / len_val):
            torch.save(model.state_dict(), os.path.join(output_dir,'best.pth'))


        print("Epoch:{}/{}..".format(e + 1, epochs),
              "Train Acc: {:.3f}..".format(training_acc / len_train),
              "Val Acc: {:.3f}..".format(validation_acc / len_val),
              "Train Loss: {:.3f}..".format(training_loss / len(train_loader)),
              "Val Loss: {:.3f}..".format(validation_loss / len(val_loader)),
              "Time: {:.2f}s".format((time.time() - since)))

        scheduler.step()

    history = {'train_loss': train_loss, 'val_loss': val_loss ,'train_acc': train_acc, 'val_acc': val_acc,'lr':learning_rate}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))

    return history


def plot_loss(x,output_dir, history):
    plt.plot(x, history['val_loss'], label='val', marker='o')
    plt.plot(x, history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig(os.path.join(output_dir,'loss.png'))
    plt.clf()

def plot_acc(x,output_dir, history):
    plt.plot(x, history['train_acc'], label='train_acc', marker='x')
    plt.plot(x, history['val_acc'], label='val_acc', marker='x')
    plt.title('Acc per epoch')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig(os.path.join(output_dir,'acc.png'))
    plt.clf()

def plot_lr(x,output_dir,  history):
    plt.plot(x, history['lr'], label='learning_rate', marker='x')
    plt.title('learning rate per epoch')
    plt.ylabel('Learning_rate')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig(os.path.join(output_dir,'learning_rate.png'))
    plt.clf()