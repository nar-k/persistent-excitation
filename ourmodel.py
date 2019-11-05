"""The network architecture used in all CIFAR-10 experiments."""

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,
                 output_dim,
                 net_size=2,
                 perturb_only_first_layer=False):
        super(Net, self).__init__()

        self.net_size = net_size
        self.perturb_only_first_layer = perturb_only_first_layer
        
        self.perturb_conv1 = nn.Parameter(torch.zeros(3,32,32))
        self.conv1 = nn.Conv2d(3, round(6 * net_size), 5)
        
        self.perturb_conv2 = nn.Parameter(torch.zeros(round(6 * net_size) ,14 ,14))
        self.conv2 = nn.Conv2d(round(6 * net_size), round(16 * net_size), 5)
        
        self.perturb_fc1 = nn.Parameter(torch.zeros(round(16 * net_size) * 5 * 5))
        self.fc1 = nn.Linear(round(16 * net_size) * 5 * 5, round(120 * net_size))
        
        self.perturb_fc2 = nn.Parameter(torch.zeros(round(120 * net_size)))
        self.fc2 = nn.Linear(round(120 * net_size), round(96 * net_size))
        
        self.perturb_fc3 = nn.Parameter(torch.zeros(round(96 * net_size)))
        self.fc3 = nn.Linear(round(96 * net_size), output_dim)
        
        self.params_regular = nn.ParameterList([self.conv1.weight, self.conv1.bias,
                                               self.conv2.weight, self.conv2.bias,
                                               self.fc1.weight, self.fc1.bias,
                                               self.fc2.weight, self.fc2.bias,
                                               self.fc3.weight, self.fc3.bias])
        
        if self.perturb_only_first_layer:
            self.params_perturb = nn.ParameterList([self.perturb_conv1])
        else:
            self.params_perturb = nn.ParameterList([self.perturb_conv1,
                                                 self.perturb_conv2,
                                                 self.perturb_fc1,
                                                 self.perturb_fc2,
                                                 self.perturb_fc3])
        
    def featurize(self,x):
        if self.perturb_only_first_layer:
            x = x + self.perturb_conv1
            x = F.max_pool2d(F.leaky_relu(self.conv1(x)),2)
            x = F.max_pool2d(F.leaky_relu(self.conv2(x)),2)
            x = x.view(-1, round(16 * self.net_size) * 5 * 5)
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
        else:
            x = x + self.perturb_conv1
            x = F.max_pool2d(F.leaky_relu(self.conv1(x)),2)
            x = x + self.perturb_conv2
            x = F.max_pool2d(F.leaky_relu(self.conv2(x)),2)
            x = x.view(-1, round(16 * self.net_size) * 5 * 5)
            x = x + self.perturb_fc1
            x = F.leaky_relu(self.fc1(x))
            x = x + self.perturb_fc2
            x = F.leaky_relu(self.fc2(x))
        return x
    
    def forward_once(self,x):
        x = self.featurize(x)
        if not self.perturb_only_first_layer:
            x = x + self.perturb_fc3
        output1 = self.fc3(x)
        return output1

    def forward(self, input1):
        return self.forward_once(input1)