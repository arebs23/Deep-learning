import enum
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.optim import SGD

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_folder = '../data/FMNIST'
fmnist = datasets.FashionMNIST(data_folder, download=True)
tr_images = fmnist.data
tr_targets = fmnist.targets

unique_values = tr_targets.unique()
# print(f'tr_images & tr_targets:\n\tX -{tr_images.shape}\n\tY \
# -{tr_targets.shape}\n\tY-Unique Values : {unique_values}')
# print(f'TASK:\n\t{len(unique_values)} class Classification')
# print(f'UNIQUE CLASSES:\n\t{fmnist.classes}')

R, C = len(tr_targets.unique()), 10

fig, ax = plt.subplots(R, C, figsize=(10, 10))
for label_class, plot_row in enumerate(ax):
    label_x_rows = np.where(tr_targets == label_class)[0]
    for plot_cell in plot_row:
        plot_cell.grid(False); plot_cell.axis('off')
        ix = np.random.choice(label_x_rows)
        x, y = tr_images[ix], tr_targets[ix]
        plot_cell.imshow(x, cmap='gray')
plt.tight_layout()


class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        x = x.float()
        x = x.view(-1, 28*28)
        self.x, self.y = x, y

    
    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        return x.to(device), y.to(device)

    
    def __len__(self):
        return len(self.x)



def get_dataset():
    train = FMNISTDataset(tr_images, tr_targets)
    trn_dl = DataLoader(train, batch_size=32, shuffle=True)
    return trn_dl

def get_model():
    model = nn.Sequential(
                nn.Linear(28*28, 1000),
                nn.ReLU(),
                nn.Linear(1000,10)).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr = 1e-2)
    return model, loss_fn, optimizer


def train_batch(x, y, model, opt, loss_fn):
    model.train() #

    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    opt.step()
    opt.zero_grad()
    return batch_loss.item()


@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)

    # prediction = torch.argmax(prediction, dim=1)
    # return (prediction == y).float().mean().item()
    _, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()
