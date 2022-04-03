import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
# root parameter: define where to save the data
# train parameter: we are initializing the MNIST training dataset.
# download parameter: our data folder dose not already downloaded
# transform parameter: we want to apply any image manipulation transforms

mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
print(len(mnist_trainset))
print(len(mnist_testset))