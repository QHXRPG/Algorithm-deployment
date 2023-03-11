import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import torch.utils.data as Data
import numpy as np
import math
from copy import deepcopy
from torch.nn.parameter import Parameter
from matplotlib import pyplot as plt

def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return x.clone().detach().requires_grad_(requires_grad)


class Mask_linear(nn.Linear):
    def __init__(self,in_dim, out_dim,is_mask=False):
        super(Mask_linear, self).__init__(in_dim,out_dim)
        self.mask = None
        self.is_mask = is_mask
    def do_mask(self,mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask
        self.is_mask = True
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.is_mask:
            return F.linear(input,self.weight * self.mask)
        else:
            return F.linear(input,self.weight)

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.l1 = Mask_linear(28*28,1000)
        self.relu = nn.ReLU()
        self.l2 = Mask_linear(1000,100)
        self.l3 = Mask_linear(100,10)
    def do_mask(self, mask):
        self.l1.do_mask(mask[0])
        self.l2.do_mask(mask[1])
        self.l3.do_mask(mask[2])
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        return self.l3(out)

def make_mask(model:Linear, rate):
    mask = []
    threshold_all = []
    i=0
    for p in model.parameters():
        if len(p.data.size()) != 1:  #不对偏置做卷积
            weight = p.data.abs().numpy().flatten()
            threshold = np.percentile(weight, rate*100)
            threshold_all.append(threshold)
    for p in model.parameters():
        if len(p.data.size()) != 1:
            print(i)
            mask.append((p.data.abs() > threshold_all[i]).float())
            i+=1
    return mask
def plot_weights(model):
    modules = [module for module in model.modules()]
    num_sub_plot = 0
    for i, layer in enumerate(modules):
        if hasattr(layer, 'weight'):
            plt.subplot(131+num_sub_plot)
            w = layer.weight.data
            w_one_dim = w.cpu().numpy().flatten()
            plt.hist(w_one_dim[w_one_dim!=0], bins=50)
            num_sub_plot += 1
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Linear().to(device)
optimizer = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()
train_data = MNIST("/Users/qiuhaoxuan/PycharmProjects/深度学习视觉实战/部署/",
                   train=True,download=True,
                   transform=torchvision.transforms.ToTensor())
data_loader = Data.DataLoader(train_data,batch_size=600,num_workers=0)

for i in range(50):
    for j,(x,y) in enumerate(data_loader):
        y_p = model(x)
        l = loss(y_p, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    if i%100 ==0 :
        print(l.item())

plot_weights(model)

mask = make_mask(model, 0.6)
pruned_model = deepcopy(model)
pruned_model.do_mask(mask)
plot_weights(pruned_model)
