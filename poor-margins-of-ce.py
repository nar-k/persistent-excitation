"""
The main file that trains a linear classifier with the cross-entropy loss
using the gradient descent algorithm.
"""

import argparse

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data.dataset import Dataset

# for printing progress
from tqdm import tqdm

# for numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# to save to a folder
import os
import sys


parser = argparse.ArgumentParser(description='Comparison of cross-entropy loss and squared-error loss.')

parser.add_argument(
    '--name', default='', type=str,
    help='name of the model to be saved')
parser.add_argument(
    '--epoch', default=100000, type=int,
    help='number of epochs to run (default: 100000)')
parser.add_argument(
    '--learning-rate', default=1e-2, type=float,
    help='learning rate (default: 1e-2)')
parser.add_argument(
    '--momentum', default=0.9, type=float,
    help='momentum (default: 0.9)')
parser.add_argument(
    '--scaling', default=5, type=int,
    help='scaling factor to scale down the output of the network (default: 5)')
parser.add_argument(
    '--seed', default=1, type=int,
    help='seed for randomness (default: 1)')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


class TwoClassDataset(Dataset):
    """Combines the data from two classes, assigns the labels for each point."""
    
    def __init__(self, dataset_a, dataset_b):
        """
        Args:
            dataset_a (Dataset): dataset_a
            dataset_b (Dataset): dataset_b
        """
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.data_a_len = len(self.dataset_a)
        self.data_b_len = len(self.dataset_b)
        self.data_len = self.data_a_len + self.data_b_len

    def __getitem__(self, index):
        """
        If the index is less than the length of the dataset of A
        then return the sample from dataset A otherwise return
        the sample from dataset B.

        The label of dataset A is 0.
        The label of dataset B is 1.
        """
        if index < self.data_a_len:
            u = self.dataset_a[index][0]
            y = 0
        else:
            u = self.dataset_b[index-self.data_a_len][0]
            y = 1
        
        return (u,y)

    def __len__(self):
        return self.data_len


def train_ce(args, model, train_loader, optimizer, epoch, scaling=1.0, use_tqdm=True):
    """Function for training with the cross-entropy loss."""

    model.train()
    
    if use_tqdm:
        pbar = tqdm(iter(train_loader), leave=True)
    else:
        pbar = iter(train_loader)
    
    c = 0
    acc_loss = 0
    
    for (data, target) in pbar:
        target = target.type(torch.FloatTensor)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output.squeeze()/scaling, target)
        loss.backward()
        optimizer.step()
        
        acc_loss += loss.item() 
        c += 1
        
        if use_tqdm:
            batch_results = dict()
            batch_results['epoch'] = epoch
            batch_results['loss'] = loss.item()
            pbar.set_postfix(batch_results)
        
    return acc_loss/c

# ------------------------------------------------------------------------------
# Create the first data set ----------------------------------------------------
# ------------------------------------------------------------------------------

N = 20

y_center = 35
x_center = -35

theta_off_1 = -np.pi/8
theta_off_2 = -np.pi/2.3

f_1_x = lambda x: x_center + 58*np.cos(x)
f_1_y = lambda x: x_center + 58*np.sin(x + theta_off_1) 
f_2_x = lambda x: y_center + 35*np.cos(x)
f_2_y = lambda x: y_center + 35*np.sin(x + theta_off_2)

theta_vals = torch.FloatTensor(np.arange(N)*(2*np.pi)/N)

y_vals = torch.FloatTensor(list(map(f_1_x,theta_vals)))
y_f = torch.FloatTensor(list(map(f_1_y,theta_vals)))

x_vals = torch.FloatTensor(list(map(f_2_x,theta_vals)))
x_f = torch.FloatTensor(list(map(f_2_y,theta_vals)))

Y = torch.stack((y_vals, y_f),dim=1)
X = torch.stack((x_vals, x_f),dim=1)

train_dataset_a = torch.utils.data.TensorDataset(X)
train_dataset_b = torch.utils.data.TensorDataset(Y)
train_dataset_1 = TwoClassDataset(train_dataset_a,train_dataset_b)

print('First dataset created.')


# Train with the cross-entropy loss ------------------------------------------
train_loader_1 = torch.utils.data.DataLoader(train_dataset_1,
                                                batch_size=2*N,
                                                shuffle=True)

# Create the linear model
model_1 = nn.Linear(2, 1, bias=True)

# Set the optimizer
optimizer_1 = optim.SGD(model_1.parameters(), lr=args.learning_rate, momentum=args.momentum)

print('Training the model on the first data set with the cross-entropy loss.')
ebar = tqdm(range(1, args.epoch + 1))
for epoch in ebar:
    loss = train_ce(args, model_1, train_loader_1, optimizer_1, 
                 epoch, use_tqdm=False, scaling=args.scaling)
    ebar.set_postfix({'loss': loss})

# Create a folder to save the figures
if not os.path.exists('Poor-margins-of-ce'):
    os.makedirs('Poor-margins-of-ce')

# Plot the decision boundary ----------------------------------------------

L = 2500
X_Mi = -200
X_Ma = 200
Y_Mi = -200
Y_Ma = 200
xv, yv = torch.meshgrid([torch.linspace(X_Mi,X_Ma,L), torch.linspace(Y_Mi,Y_Ma,L)])
xy = torch.stack((xv.flatten(),yv.flatten()),dim=1)

model_1.eval()
z = model_1(xy)

a = xy[:,0].data.numpy()
b = xy[:,1].data.numpy()
c = z.squeeze().data.numpy()

plt.figure(figsize=(3.6,3.6),dpi=120)

ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
ax.yaxis.set_major_locator(ticker.MultipleLocator(100))

contour_stan = plt.contour(a.reshape(L,L),b.reshape(L,L),np.sign(c.reshape(L,L)),(0,),
            antialiased=True, colors='maroon', linewidths=2,linestyles='solid')
    
plt.scatter(X[:,0].data.numpy(),X[:,1].data.numpy(),s=20)
plt.scatter(Y[:,0].data.numpy(),Y[:,1].data.numpy(),s=20)

lines = [contour_stan.collections[0]]

plt.gca().set_ylim(bottom=-150)
plt.gca().set_ylim(top=150)
plt.gca().set_xlim(left=-150)
plt.gca().set_xlim(right=150)

plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')

# Name to be used for saving the figure
figure1_name = '{}'.format(args.name) + 'poor-margin-of-ce-data1.pdf'

# Save the first figure -------------------------------------------------
figure1_fout = os.path.join('Poor-margins-of-ce', figure1_name)
plt.savefig(figure1_fout)

print('First figure saved successfully.')

# ------------------------------------------------------------------------
# Create the second dataset ----------------------------------------------
# ------------------------------------------------------------------------

f = lambda x: 10 + 20*x
N = 20
stdev = 0.8

y_center = 2
x_center = 4

y_vals = torch.FloatTensor(y_center - np.random.rand(N)*stdev)
y_f = torch.FloatTensor(list(map(f,y_vals)))
x_vals = torch.FloatTensor(x_center - np.random.rand(N)*stdev)
x_f = torch.FloatTensor(list(map(f,x_vals)))

Y = torch.stack((y_vals, y_f),dim=1)
X = torch.stack((x_vals, x_f),dim=1)

train_dataset_a = torch.utils.data.TensorDataset(X)
train_dataset_b = torch.utils.data.TensorDataset(Y)
train_dataset_2 = TwoClassDataset(train_dataset_a,train_dataset_b)
print('Second dataset created.')

train_loader_2 = torch.utils.data.DataLoader(train_dataset_2,
                                                batch_size=2*N,
                                                shuffle=True)

# Create the second linear model
model_2 = nn.Linear(2, 1, bias=True)

optimizer_2 = optim.SGD(model_2.parameters(), lr=args.learning_rate, momentum=args.momentum)

# Train the second model
ebar = tqdm(range(1, args.epoch + 1))
for epoch in ebar:
    loss = train_ce(args, model_2, train_loader_2, optimizer_2, 
                 epoch, use_tqdm=False, scaling=args.scaling)
    ebar.set_postfix({'loss': loss})

# Plot the decision boundary
L = 2500
X_Mi = -50
X_Ma = 50
Y_Mi = 10
Y_Ma = 110
xv, yv = torch.meshgrid([torch.linspace(X_Mi,X_Ma,L), torch.linspace(Y_Mi,Y_Ma,L)])
xy = torch.stack((xv.flatten(),yv.flatten()),dim=1)

model_2.eval()
z = model_2(xy)

a = xy[:,0].data.numpy()
b = xy[:,1].data.numpy()
c = z.squeeze().data.numpy()

plt.figure(figsize=(3.6,3.6),dpi=120)

ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
ax.yaxis.set_major_locator(ticker.MultipleLocator(30))

contour_stan = plt.contour(a.reshape(L,L),b.reshape(L,L),np.sign(c.reshape(L,L)),(0,),
            antialiased=True, colors='maroon', linewidths=2,linestyles='solid')
    
plt.scatter(X[:,0].data.numpy(),X[:,1].data.numpy(),s=20)
plt.scatter(Y[:,0].data.numpy(),Y[:,1].data.numpy(),s=20)

lines = [contour_stan.collections[0]]
labels = ['cross-entropy min.']


plt.gca().set_ylim(bottom=25)
plt.gca().set_ylim(top=95)
plt.gca().set_xlim(left=-35)
plt.gca().set_xlim(right=35)

plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')

# Name to be used for saving the second figure
figure2_name = '{}'.format(args.name) + 'poor-margin-of-ce-data2.pdf'

# Save the second figure -------------------------------------------------
figure2_fout = os.path.join('Poor-margins-of-ce', figure2_name)
plt.savefig(figure2_fout)

print('Second figure saved successfully.')

