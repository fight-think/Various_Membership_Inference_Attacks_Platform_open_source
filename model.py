import torch
import torch.nn.functional as F
import torch.nn as nn
from models import DenseNet121,ResNet18,VGG,ResNet,BasicBlock,EfficientNet

cfg = {
        'num_blocks': [1, 2, 2, 3, 3, 4, 1],
        'expansion': [1, 6, 6, 6, 6, 6, 6],
        'out_channels': [16, 24, 40, 80, 112, 192, 320],
        'kernel_size': [3, 3, 5, 3, 5, 5, 3],
        'stride': [1, 2, 2, 2, 1, 2, 1],
        'dropout_rate': 0.2,
        'drop_connect_rate': 0.2,
    }

#2024/8/20 current models
#mnist_CNN
#cifar_LetNet
#Purchase_shallow_MLP


#model for training target and shadow models of the MNIST
class model_mnist_CNN(nn.Module):
  def __init__(self):
    super(model_mnist_CNN, self).__init__()      
    self.cc1=torch.nn.Conv2d(1,4,2)
    self.cc2=torch.nn.Conv2d(4,16,2)
    self.maxpool=torch.nn.MaxPool2d(2)
    self.dropout1=torch.nn.Dropout(p=0.5)
    self.linear2 = torch.nn.Linear(2704, 1024)
    self.linear3 = torch.nn.Linear(1024, 10)
  def forward(self, x): 
    bath_size=x.size(0)
    x=self.cc1(x)  
    x=F.relu(x)
    x=self.cc2(x)  
    x=F.relu(x)
    x=self.maxpool(x)
    x = x.view(bath_size, -1)
    x=self.linear2(x)
    x=self.dropout1(x)
    x=self.linear3(x)
    h_f=x
    x=F.softmax(x,dim=1)
    return x,h_f

#model for training target and shadow models of the MNIST
class model_mnist_ResNet18(ResNet):
  def __init__(self):
    super().__init__(BasicBlock, [2, 2, 2, 2])
    self.conv1 = nn.Conv2d(1, 64, kernel_size=3,stride=1, padding=1, bias=False)
  def forward(self, x):
    x=ResNet.forward(self,x)
    h_f=x
    x=F.softmax(x,dim=1)
    return x,h_f

class model_cifar_LetNet(nn.Module):
  def __init__(self):
    super(model_cifar_LetNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
    self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
    self.fc1 = nn.Linear(16*5*5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2)
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = nn.Dropout(0.5)(x)
    x = F.relu(self.fc2(x))
    x = nn.Dropout(0.5)(x)
    x = self.fc3(x)
    h_f=x
    x=F.softmax(x,dim=1) #to make it consistent, add softmax
    return x,h_f

class model_cifar_ResNet18(ResNet): 
  def __init__(self):
    super().__init__(BasicBlock, [2, 2, 2, 2])

  def forward(self, x):
    x=ResNet.forward(self,x)
    h_f=x
    x=F.softmax(x,dim=1)
    return x,h_f

class model_purchase_Shallow_MLP(nn.Module):
  def __init__(self):
    super(model_purchase_Shallow_MLP, self).__init__()
    self.linear1 = torch.nn.Linear(600, 128)
    self.linear2 = torch.nn.Linear(128, 100)

  def forward(self, x): 
    x=self.linear1(x)
    x=torch.tanh(x)
    x=self.linear2(x)
    x=torch.tanh(x)
    h_f=x
    x=F.softmax(x,dim=1)  
    return x,h_f

class model_purchase_Deeper_MLP(nn.Module):
  def __init__(self):
    super(model_purchase_Deeper_MLP, self).__init__()
    self.linear1 = torch.nn.Linear(600, 512)
    self.linear2 = torch.nn.Linear(512, 256)
    self.linear4 = torch.nn.Linear(256, 128)
    self.linear6 = torch.nn.Linear(128, 100)
    self.dropout = nn.Dropout(0.3)

  def forward(self, x): 
    x=self.linear1(x)
    x=torch.tanh(x)
    x=self.dropout(x)
    x=self.linear2(x)
    x=torch.tanh(x)
    x=self.dropout(x)
    x=self.linear4(x)
    x=torch.tanh(x)
    x=self.dropout(x)
    x=self.linear6(x)
    x=torch.tanh(x)
    h_f=x
    x=F.softmax(x,dim=1)
    return x,h_f

#use fully connected neural network for training attack model
class model_attack(nn.Module):
  def __init__(self, input_n, hidden_n, output_n):
    super(model_attack, self).__init__()               
    self.linear1 = torch.nn.Linear(input_n, hidden_n)
    self.linear2 = torch.nn.Linear(hidden_n, output_n)

  def forward(self, x): 
    x=self.linear1(x)
    x=self.linear2(x)
    x=torch.nn.Sigmoid()(x)
    
    return x