"""
The main file that trains a two-layer network with cross-entropy loss
and squared-error loss for a binary classification task and draws the
decision boundary.
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


parser = argparse.ArgumentParser(description='Comparison of cross-entropy loss and squared-error loss for a two-layer neural network.')

parser.add_argument(
    '--name', default='', type=str,
    help='name of the model to be saved')
parser.add_argument(
    '--epoch', default=60000, type=int,
    help='number of epochs to run (default: 60000)')
parser.add_argument(
    '--learning-rate', default=1e-2, type=float,
    help='learning rate (default: 1e-2)')
parser.add_argument(
    '--momentum', default=0.9, type=float,
    help='momentum (default: 0.9)')
parser.add_argument(
    '--num-hidden-layers', default=1, type=int,
    help='number of hidden layers (defualt:1)')
parser.add_argument(
    '--hidden-dim', default=20, type=int,
    help='number of nodes in each hidden layer (default: 20)')
parser.add_argument(
    '--scaling', default=5, type=int,
    help='scaling factor to scale down the output of the network (default: 5)')
parser.add_argument(
    '--seed', default=1, type=int,
    help='seed for randomness (default: 1)')

args = parser.parse_args()


torch.manual_seed(args.seed)
np.random.seed(args.seed)


class Net(nn.Module):
    def __init__(self,
                 output_dim,
                 num_hidden_layers,
                 hidden_dim):
        super(Net, self).__init__()
        
        if num_hidden_layers == 0:
            self.blocks = lambda x: x
            hidden_dim = 2
        else:
            layers = nn.ModuleList()
            last_hidden_dim = 2
            for i in range(0,num_hidden_layers):
                layers.append(nn.Linear(last_hidden_dim, hidden_dim, bias=True) )
                layers.append(nn.ReLU())
                last_hidden_dim = hidden_dim
            self.blocks = nn.Sequential(*layers)
        
        self.fc_last = nn.Linear(hidden_dim, output_dim, bias=True)
        
    def featurize(self,x):
        x = x.unsqueeze(1)
        return self.blocks(x)
    
    def forward_once(self,x):
        output1 = self.fc_last(self.featurize(x))
        return output1

    def forward(self, input1):
        return self.forward_once(input1)


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


# Create the data set ----------------------------------------------------

f = lambda x: 5 + 2*x
N = 8
stdev = 0.5

mu_Y1 = torch.FloatTensor([[0,f(0)]])
mu_Y2 = torch.FloatTensor([[11,f(11)]])
mu_X = torch.FloatTensor([[7,f(7)]])

X = stdev*torch.randn(N,2)+mu_X
Y1 = stdev*torch.randn(N//2,2)+mu_Y1
Y2 = stdev*torch.randn(N//2,2)+mu_Y2
Y = torch.cat([Y1,Y2],dim=0)

train_dataset_a = torch.utils.data.TensorDataset(X)
train_dataset_b = torch.utils.data.TensorDataset(Y)

train_dataset = TwoClassDataset(train_dataset_a,train_dataset_b)

print('Dataset created.')


# Train with the cross-entropy loss ------------------------------------------

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=2*N,
                                           shuffle=True)

model_ce = Net(output_dim=1,
               num_hidden_layers=args.num_hidden_layers,
               hidden_dim=args.hidden_dim)

mult_fac = 10.0
for k in range(0,args.num_hidden_layers,2):
    model_ce.blocks[k].bias.data = mult_fac * model_ce.blocks[k].bias.data

optimizer_ce = optim.SGD(model_ce.parameters(), lr=args.learning_rate, momentum=args.momentum)

def train_ce(args, model, train_loader, optimizer, epoch, scaling=1.0,
    use_tqdm=True):
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

print('Training with the cross-entropy loss.')
ebar = tqdm(range(1, args.epoch + 1))
for epoch in ebar:
    loss = train_ce(args, model_ce, train_loader, optimizer_ce, 
                 epoch, use_tqdm=False, scaling=args.scaling)
    ebar.set_postfix({'loss': loss})


# Train with the squared-error loss -------------------------------------

model_sq = Net(output_dim=1,
                 num_hidden_layers=args.num_hidden_layers,
                 hidden_dim=args.hidden_dim)

mult_fac = 10.0
for k in range(0,args.num_hidden_layers,2):
    model_sq.blocks[k].bias.data = mult_fac * model_sq.blocks[k].bias.data

def train_sq(args, model, train_loader, optimizer, epoch, scaling=1.0,
    use_tqdm=True):
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
        loss = torch.nn.functional.mse_loss(output.squeeze()/scaling, target - 0.5)
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

optimizer_stan = optim.SGD(model_sq.parameters(), lr=args.learning_rate, momentum=args.momentum)

print('Training with the squared-error loss.')
ebar = tqdm(range(1, args.epoch + 1))
for epoch in ebar:
    loss = train_sq(args, model_sq, train_loader, optimizer_stan, epoch, use_tqdm=False, scaling=args.scaling)
    ebar.set_postfix({'loss': loss})


# Plot the decision boundaries ----------------------------------------------

L = 2500
X_Mi = -5
X_Ma = 70
Y_Mi = -5
Y_Ma = 70
xv, yv = torch.meshgrid([torch.linspace(X_Mi,X_Ma,L), torch.linspace(Y_Mi,Y_Ma,L)])
xy = torch.stack((xv.flatten(),yv.flatten()),dim=1)


model_ce.eval()
z = model_ce.forward_once(xy)
a_ce = xy[:,0].data.numpy()
b_ce = xy[:,1].data.numpy()
c_ce = z.squeeze().data.numpy()

model_sq.eval()
z = model_sq(xy)
a_sq = xy[:,0].data.numpy()
b_sq = xy[:,1].data.numpy()
c_sq = z.squeeze().data.numpy()


plt.figure(figsize=(3.6,3.6),dpi=120)

ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
ax.yaxis.set_major_locator(ticker.MultipleLocator(15))

contour_ce = plt.contour(a_ce.reshape(L,L),b_ce.reshape(L,L),np.sign(c_ce.reshape(L,L)),(0,),
            antialiased=True, colors='maroon', linewidths=2,linestyles='solid')
contour_sq = plt.contour(a_sq.reshape(L,L),b_sq.reshape(L,L),np.sign(c_sq.reshape(L,L)),(0,),
            antialiased=True, colors='maroon', linewidths=2,linestyles='dotted')
    
plt.scatter(X[:,0].data.numpy(),X[:,1].data.numpy(), s=20)
plt.scatter(Y[:,0].data.numpy(),Y[:,1].data.numpy(), s=20)

# plt.xlabel('x')
# plt.ylabel('y')
lines = [contour_ce.collections[0], contour_sq.collections[0]]
labels = ['cross-entropy loss','squared-error loss']

plt.gca().set_ylim(bottom=-5)
plt.gca().set_ylim(top=35)
plt.gca().set_xlim(left=-5)
plt.gca().set_xlim(right=35)

plt.legend(lines, labels, loc='lower right')
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')

# Make sure the relevant folders have been created -------------
if not os.path.exists('Figure-1-ce-vs-sq'):
    os.makedirs('Figure-1-ce-vs-sq')

# name to be used for saving the figure
figure_name = '{}'.format(args.name) + 'ce-vs-sq-s{}-epoch{}k'.format(args.seed, args.epoch/1000) + '.pdf'

# save the figure -------------------------------------------------
figure_fout = os.path.join('Figure-1-ce-vs-sq', figure_name)
plt.savefig(figure_fout)

print('Figure saved successfully.')