"""Trains a network with the cross-entropy loss with no persistent excitation."""

import argparse

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data.dataset import Dataset
import foolbox
import pickle

# torchvision for dataset transformations
from torchvision import datasets, transforms

# for printing progress
from tqdm import tqdm

# for numpy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import copy # for deepcopy

# model
from ourmodel import *

# to save to a folder
import os
import sys


parser = argparse.ArgumentParser(description='Training a neural net with cross-entropy loss')

parser.add_argument(
    '--resume', type=str, default='',
    help='if available, enter the filename with the network weights to resume')
parser.add_argument(
    '--name', default='', type=str,
    help='name of the model for saving purposes')
parser.add_argument(
    '--epoch', default=40, type=int,
    help='number of epochs to run (default: 40)')
parser.add_argument(
    '--net-size', default=2, type=float,
    help='multiplier to increase the network size (default:2)')
parser.add_argument(
    '--batch-size', default=50, type=int,
    help='mini-batch size (default: 50)')
parser.add_argument(
    '--learning-rate', default=1e-3, type=float,
    help='learning rate (default: 1e-3)')
parser.add_argument(
    '--momentum', default=0.9, type=float,
    help='momentum (default: 0.9)')
parser.add_argument(
    '--seed', default=1, type=int,
    help='seed for randomness (default: 1)')

args = parser.parse_args()

torch.manual_seed(args.seed)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_dataset = datasets.CIFAR10(root='../../data',
                                 train=True,
                                 download=True,
                                 transform=transform)

test_dataset = datasets.CIFAR10(root='../../data',
                                train=False,
                                download=True,
                                transform=transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def find_class_idx(dataset, class_label):
    num_samples = len(dataset)
    indices = []
    for i in range(num_samples):
        _, label = dataset[i]
        if label == class_label:
            indices.append(i)
    return indices

class_a_label = 0
class_b_label = 7

print('Subsetting the dataset to two classes...')
idx_class_a = find_class_idx(train_dataset, class_a_label)
idx_class_b = find_class_idx(train_dataset, class_b_label)
train_dataset_a = torch.utils.data.Subset(train_dataset, idx_class_a)
train_dataset_b = torch.utils.data.Subset(train_dataset, idx_class_b)

test_idx_class_a = find_class_idx(test_dataset, class_a_label)
test_idx_class_b = find_class_idx(test_dataset, class_b_label)
test_dataset_a = torch.utils.data.Subset(test_dataset, test_idx_class_a)
test_dataset_b = torch.utils.data.Subset(test_dataset, test_idx_class_b)
print('Subsetting done.')

class TwoClassDataset(Dataset):
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

train_dataset = TwoClassDataset(train_dataset_a, train_dataset_b)
test_dataset = TwoClassDataset(test_dataset_a, test_dataset_b)

# data loaders
train_loader_a = torch.utils.data.DataLoader(train_dataset_a,
                                             batch_size=args.batch_size,
                                             shuffle=True)

train_loader_b = torch.utils.data.DataLoader(train_dataset_b,
                                             batch_size=args.batch_size,
                                             shuffle=True)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=1,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True)

def train_ce(args,
             model,
             train_loader,
             optimizer,
             epoch,
             use_tqdm=True):
    
    # make the model trainable
    model.train()

    # progress bars or not
    if use_tqdm:
        pbar = tqdm(iter(train_loader), leave=True)
    else:
        pbar = iter(train_loader)

    c = 0
    acc_loss = 0
        
    for data1,label1 in pbar:
        
        optimizer.zero_grad()
        output = model(data1)
        
        loss = F.cross_entropy(output, label1)

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

# Create the network.
model = None
model = Net(output_dim=2, net_size=args.net_size)


# If network weights are available, load them.
if args.resume:
    print('Resuming from checkpoint: {}'.format(args.resume))
    model.load_state_dict(torch.load(args.resume))
# Otherwise, train the network.
else:
    print('Training the model.')

    optimizer = optim.SGD(model.params_regular,
                          lr=args.learning_rate,
                          momentum=args.momentum)

    for epoch in range(1, args.epoch + 1):
        train_ce(args,
                 model,
                 train_loader,
                 optimizer,
                 epoch,
                 use_tqdm=True)

def compute_accuracy(args, model, loader, threshold=0.5):
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for data, target in loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target).sum().item()

    print('Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct,
        len(loader.dataset),
        100. * correct / len(loader.dataset)))

    return correct/len(loader.dataset)


print('On training dataset:')
trn_acc = compute_accuracy(args, model, train_loader)

print('On test dataset:')
tst_acc = compute_accuracy(args, model, test_loader)


# Make sure the relevant folders have been created -------------

if not os.path.exists('persistently-excited-disturbance-fig'):
    os.makedirs('persistently-excited-disturbance-fig')

if not os.path.exists('persistently-excited-disturbance-data'):
    os.makedirs('persistently-excited-disturbance-data')

if not os.path.exists('persistently-excited-weights'):
    os.makedirs('persistently-excited-weights')

# -----------------------------------------------------------------

# name to be used for saving the network weights and perturbation data
out_filename = '{}'.format(args.name) + 'cross_ent_bs{}_lr{}_mom{}_ep{}_ns{}_trn{}_tst{}'.format(
    args.batch_size, args.learning_rate, args.momentum,
    args.epoch, args.net_size, trn_acc, tst_acc)

if not args.resume:
    fout_weights = os.path.join('persistently-excited-weights',out_filename + '.pt')

    torch.save(model.state_dict(),fout_weights)
    print('Written to: {}'.format(fout_weights))


model.eval()
foolbox_model = foolbox.models.PyTorchModel(model=model,
                                            num_classes=2,
                                            bounds=(-1, 1))
criterion = foolbox.criteria.Misclassification()
attack = foolbox.attacks.PGD(foolbox_model, criterion, distance=foolbox.distances.Linfinity)

def get_adv_disturbance_mag(attack,
                            dataset,
                            true_label_for_dataset,
                            order):
    list_of_mag = []
    num_samples_in_dataset = len(dataset)
    
    for i in tqdm(range(num_samples_in_dataset)):
        img, _ = dataset[i]
        
        try:
            adversarial_image = attack(img.numpy(), label=true_label_for_dataset)
        
            difference_image = adversarial_image - img.data.numpy() 
            norm_of_difference_image = np.linalg.norm(difference_image.flatten(), ord=order)
            list_of_mag.append(norm_of_difference_image)  
        except Exception:
            pass
        
    return list_of_mag

print('Finding the disturbance needed to flip each image.')

# # TO FIND ADVERSARIAL EXAMPLES ONLY ON SUBSET ----------------------------------
# test_dataset_a = torch.utils.data.Subset(test_dataset_a, test_idx_class_a[0:10])
# test_dataset_b = torch.utils.data.Subset(test_dataset_b, test_idx_class_b[0:10])
# train_dataset_a = torch.utils.data.Subset(train_dataset_a, idx_class_a[0:10])
# train_dataset_b = torch.utils.data.Subset(train_dataset_b, idx_class_b[0:10])
# # ------------------------------------------------------------------------------


print('Running on the test data of the first class.')
testa2b = get_adv_disturbance_mag(attack=attack,
                            dataset=test_dataset_a,
                            true_label_for_dataset=0,
                            order=np.inf)

print('Running on the test data of the second class.')
testb2a = get_adv_disturbance_mag(attack=attack,
                            dataset=test_dataset_b,
                            true_label_for_dataset=1,
                            order=np.inf)

print('Running on the training data of the first class.')
traina2b = get_adv_disturbance_mag(attack=attack,
                            dataset=train_dataset_a,
                            true_label_for_dataset=0,
                            order=np.inf)

print('Running on the training data of the second class.')
trainb2a = get_adv_disturbance_mag(attack=attack,
                            dataset=train_dataset_b,
                            true_label_for_dataset=1,
                            order=np.inf)


# save the output ------------------------------------------------
# plot the figure -------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3))
fig.dpi = 120

bins = np.arange(0, 1.0, 1e-3)

values_d, base_d = np.histogram(testa2b + testb2a, bins=bins)
# #evaluate the cumulative
cumulative_d = np.cumsum(values_d)
# # plot the cumulative function
plt.plot(base_d[:-1], cumulative_d)

plt.axhline(cumulative_d[-1], linestyle='--')
    
plt.xlabel('$\ell_{\infty}$ norm of the disturbance applied')
plt.ylabel('Percentage of points misclassified')
plt.tight_layout()

figure_test_filename = 'dist-on-test-' + out_filename + '.pdf'
figure_test_fout = os.path.join('persistently-excited-disturbance-fig',figure_test_filename)
plt.savefig(figure_test_fout)

# plot the figure for the train data
fig, ax = plt.subplots(figsize=(5, 3))
fig.dpi = 120

bins = np.arange(0, 1.0, 1e-3)

values_d, base_d = np.histogram(traina2b + trainb2a, bins=bins)
# #evaluate the cumulative
cumulative_d = np.cumsum(values_d)
# # plot the cumulative function
plt.plot(base_d[:-1], cumulative_d)

plt.axhline(cumulative_d[-1], linestyle='--')
    
plt.xlabel('$\ell_{\infty}$ norm of the disturbance applied')
plt.ylabel('Percentage of points misclassified')
plt.tight_layout()

figure_train_filename = 'dist-on-train-' + out_filename  + '.pdf'
figure_train_fout = os.path.join('persistently-excited-disturbance-fig',figure_train_filename)
plt.savefig(figure_train_fout)


# save the pickles ---------------------------------------------------
results = {}
results['test-dist'] = testa2b + testb2a
results['train-dist'] = traina2b + trainb2a

data_filename = 'dist-data-' + out_filename + '.p'
fout = os.path.join('persistently-excited-disturbance-data',data_filename)
pickle.dump(results, open(fout, "wb"))
print('written to: {}'.format(fout))
