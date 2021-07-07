# IMPORT LIBRARIES

import torch
import pandas as pd
import numpy as np
import os
from PIL import Image

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt


# IMPORT HELPERS

from models import model


# CREATE TRANSFORMATIONS

'''
First we create de transforamtions to the raw images. In the train dataset we are going to perform
data augmentation, so we include extra  geometric transformations(RandonFlip, RandomRotation,
RandomResizeCrop) as well as colour transformations to represent camera usage in different conditions,
and for the validation dataset we just resize and transform to tensor
'''

train_transform = transforms.Compose([transforms.Resize([300,300]),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(), 
                                      transforms.RandomRotation(5,fill=(66,48,35)),
                                      transforms.CenterCrop(250),
                                      transforms.RandomResizedCrop([250,250],scale=(0.5,1)), 
                                      transforms.ColorJitter(brightness=0.7, 
                                                             contrast=0.5, 
                                                             saturation=0.5, 
                                                             hue=0.05), 
                                      transforms.ToTensor()])

val_transform = transforms.Compose([transforms.Resize([250,250]), 
                                    transforms.CenterCrop(250), 
                                    transforms.ToTensor()])


									
# CREATE THE DATASETS AND DATALOADERS

train_dir = '/train'
val_dir = '/validation'

batch_size = 32

augmented_dataset = ImageFolder(train_dir, transform=train_transform)
augmented_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)
val_dataset = ImageFolder(val_dir, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# SPECIFY THE DEVICE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# CREATE MODEL

model = model()
model.to(device)


# TRAIN FUNCTION

# DEFINE THE TRAINING FUNCTIONS

# We will use this helper class to store the mean value of the metrics

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_model(model, optimizer, criterion, train_loader, val_loader, epochs):

    train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []
    val_loss = AverageMeter()
    val_accuracy = AverageMeter()
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()

    for epoch in range(epochs):
        # train
        model.train()
        train_loss.reset()
        train_accuracy.reset()
        train_loop = tqdm(train_loader, unit="batches")  # Printing the progress bar
        for data, target in train_loop:
            train_loop.set_description('[TRAIN] Epoch {}/{}'.format(epoch + 1, epochs))
            data, target = data.float().to(device), target.float().to(device)
            target = target.unsqueeze(-1) # (batch_size)--> (batch_size,1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), n=len(target))
            pred = output.round()  # get the prediction
            acc = pred.eq(target.view_as(pred)).sum().item()/len(target)
            train_accuracy.update(acc, n=len(target))
            train_loop.set_postfix(loss=train_loss.avg, accuracy=train_accuracy.avg)

        train_losses.append(train_loss.avg)
        train_accuracies.append(train_accuracy.avg)

        # validation
        model.eval()
        val_loss.reset()
        val_accuracy.reset()
        val_loop = tqdm(val_loader, unit=" batches")  # For printing the progress bar
        with torch.no_grad():
            for data, target in val_loop:
                val_loop.set_description('[VAL] Epoch {}/{}'.format(epoch + 1, epochs))
                data, target = data.float().to(device), target.float().to(device)
                target = target.unsqueeze(-1)
                output = model(data)
                loss = criterion(output, target)
                val_loss.update(loss.item(), n=len(target))
                pred = output.round()  # get the prediction
                acc = pred.eq(target.view_as(pred)).sum().item()/len(target)
                val_accuracy.update(acc, n=len(target))
                val_loop.set_postfix(loss=val_loss.avg, accuracy=val_accuracy.avg)

        val_losses.append(val_loss.avg)
        val_accuracies.append(val_accuracy.avg)
        
    return train_accuracies, train_losses, val_accuracies, val_losses

# TRAIN

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
epochs = 10
train_accuracies, train_losses, val_accuracies, val_losses = train_model(model, optimizer, criterion, augmented_loader, val_loader, epochs)


