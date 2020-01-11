import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rl_utils
from time import time

class DCNN1(nn.Module):
    def __init__(self, h, c):
        super(DCNN1, self).__init__()
        self.c = c
        self.h = h

        self.m1 = nn.Conv2d(c,c,3,padding=1)
        self.m2 = nn.ConvTranspose2d(c,c,4,stride=2,padding=1)
        self.m3 = nn.Conv2d(c,c,3,padding=1)
        self.m4 = nn.ConvTranspose2d(c,c,4,stride=2,padding=1)


    def forward(self, x):
        x = x.view(x.size(0), self.c, self.h, self.h)
        print(x.shape)
        x =  F.relu(self.m1(x))
        print(x.shape)
        x =  F.relu(self.m2(x))
        print(x.shape)
        x =  F.relu(self.m3(x))
        print(x.shape)
        x =  F.relu(self.m4(x))
        return x

class DCNN_layer(nn.Module):
    def __init__(self, h, c, f):
        super(DCNN_layer, self).__init__()
        self.c = c
        self.h = h

        self.m1 = nn.Conv2d(c,f,3,padding=1)
        self.m2 = nn.ConvTranspose2d(f,f,4,stride=2,padding=1)

    def unflatten(self, x):
        return x.view(x.size(0), self.c, self.h, self.h)

    def forward(self, x):
        print(x.shape)
        x =  F.relu(self.m1(x))
        x =  F.relu(self.m2(x))
        return x

class DCNN_layer2(nn.Module):
    def __init__(self, h, c, f):
        super(DCNN_layer2, self).__init__()
        self.c = c
        self.h = h
        self.f = f
        self.m1 = nn.Conv2d(c,f,3,padding=1)
        self.m2 = nn.ConvTranspose2d(f,f,3,stride=2,padding=1)

    def unflatten(self, x):
        return x.view(x.size(0), self.c, self.h, self.h)

    def forward(self, x):
        print(x.shape)
        x =  F.relu(self.m1(x))
        x =  F.relu(self.m2(x))
        return x

    def size(self):
        s =  int(self.f*(self.h//2)*(self.h//2))
        return s

class CNN_layer(nn.Module):
    def __init__(self, h, c, f) :
        super(CNN_layer, self).__init__()
        self.c = c
        self.f = f
        self.h = h
        self.m1 = nn.Conv2d(c,f,3,padding=1)
        self.m2 = nn.Conv2d(f,f,4,stride=2,padding=1)

    def size(self):
        s =  int(self.f*(self.h//2)*(self.h//2))
        return s

    def flatten(self, x):
        s =  int(self.f*(self.h//2)*(self.h//2))
        return x.view(-1,s)
    
    def forward(self,x):
        x = F.relu(self.m1(x))
        x = F.relu(self.m2(x))
        return x

class CNN_layer_foo(nn.Module):
    def __init__(self, h, c, f, fs=3, p=1) :
        super(CNN_layer_foo, self).__init__()
        self.c = c
        self.f = f
        self.h = h
        self.m1 = nn.Conv2d(c,f,fs,padding=p)
        self.m2 = nn.Conv2d(f,f,fs,padding=p)
        self.m3 = nn.Conv2d(f,f,fs,stride=2,padding=p)

    def size(self):
        s =  int(self.f*(self.h//2)*(self.h//2))
        return s

    def flatten(self, x):
        s =  int(self.f*(self.h//2)*(self.h//2))
        return x.view(-1,s)
   
    def forward(self,x):
        x = F.relu(self.m1(x))
        x = F.relu(self.m2(x))
        x = F.relu(self.m3(x))
        return x

class CNN_layer_foo2(nn.Module):
    def __init__(self, h, c, f, fs=3, p=1) :
        super(CNN_layer_foo2, self).__init__()
        self.c = c
        self.f = f
        self.h = h
        self.m1 = nn.Conv2d(c,f,fs,padding=p)
        self.m2 = nn.Conv2d(f,f,fs,stride=2,padding=p)

    def size(self):
        s =  int(self.f*(self.h//2)*(self.h//2))
        return s

    def flatten(self, x):
        s =  int(self.f*(self.h//2)*(self.h//2))
        return x.view(-1,s)

    def forward(self,x):
        x = F.relu(self.m1(x))
        x = F.relu(self.m2(x))
        return x

class CNN_layer2X(nn.Module):
    def __init__(self, cnn1, cnn2) :
        super(CNN_layer2X, self).__init__()
        self.cnn1 = cnn1
        self.cnn2 = cnn2
   
    def forward(self,x):
        x = self.cnn1.forward(x)
        x = self.cnn2.forward(x)
        return x

    def size(self):
        return self.cnn2.size()

    def flatten(self, x):
        return self.cnn2.flatten(x)


class CNN_layer1(nn.Module):
    def __init__(self, h, c, f) :
        super(CNN_layer1, self).__init__()
        self.c = c
        self.f = f
        self.h = h
        self.m1 = nn.Conv2d(c,f,3,padding=1)

    def size(self):
        s =  int(self.f*(self.h)*(self.h))
        return s

    def flatten(self, x):
        s =  self.size()
        return x.view(-1,s)

    def forward(self,x):
        x = F.relu(self.m1(x))
        return x

class CNN_layer1a(nn.Module):
    def __init__(self, h, c, f) :
        super(CNN_layer1a, self).__init__()
        self.c = c
        self.f = f
        self.h = h
        self.m1 = nn.Conv2d(c,f,3,padding=1)
        self.m2 = nn.Conv2d(f,f,3,padding=1)

    def size(self):
        s =  int(self.f*(self.h)*(self.h))
        return s

    def flatten(self, x):
        s =  self.size()
        return x.view(-1,s)

    def forward(self,x):
        x = F.relu(self.m1(x))
        x = F.relu(self.m2(x))
        return x


class CNN_layer2(nn.Module):
    def __init__(self, h, c, f, fs=3) :
        super(CNN_layer2, self).__init__()
        self.c = c
        self.f = f
        self.h = h
        self.m1 = nn.Conv2d(c,f,fs,padding=1)
        self.m2 = nn.Conv2d(f,f,4,stride=2,padding=1)

    def size(self):
        s =  int(self.f*(self.h//2)*(self.h//2))
        return s

    def flatten(self, x):
        s =  int(self.f*(self.h//2)*(self.h//2))
        return x.view(-1,s)

    def forward(self,x):
        x = F.relu(self.m1(x))
        x = F.relu(self.m2(x))
        return x


class CNN_layer3(nn.Module):
    def __init__(self, h, c, f, fs=3) :
        super(CNN_layer3, self).__init__()
        self.c = c
        self.f = f
        self.h = h
        self.m1 = nn.Conv2d(c,f,fs,padding=1)
        self.m2 = nn.MaxPool2d(2,stride=2)

    def size(self):
        s =  int(self.f*(self.h//2)*(self.h//2))
        return s

    def flatten(self, x):
        s =  int(self.f*(self.h//2)*(self.h//2))
        return x.view(-1,s)

    def forward(self,x):
        x = F.relu(self.m1(x))
        x = self.m2(x)
        return x


class CNN1(nn.Module):
    def __init__(self, h, c):
        super(CNN1, self).__init__()
        self.c = c
        self.h = h

        self.m1 = nn.Conv2d(c,c,3,padding=1)
        self.m2 = nn.Conv2d(c,c,4,stride=2,padding=1)
        self.m3 = nn.Conv2d(c,2*c,3,padding=1)
        self.m4 = nn.Conv2d(2*c,2*c,4,stride=2,padding=1)

    def size(self):
        s =  2*self.c*(self.h//4)*(self.h//4)
        #print(s, type(s))
        return s

    def forward(self,x):
        x = F.relu(self.m1(x))
        print(x.shape)
        x = F.relu(self.m2(x))
        print(x.shape)
        x = F.relu(self.m3(x))
        print(x.shape)
        x = F.relu(self.m4(x))
        print(x.shape)
        x = x.view(x.size(0) , self.size())
        return x

class CNN2(nn.Module):
    def __init__(self, h, c):
        super(CNN2, self).__init__()
        self.c = c
        self.h = h

        self.m1 = nn.Conv2d(c,c,3,padding=1)
        self.m2 = nn.Conv2d(c,c,4,stride=2,padding=1)
        self.m3 = nn.Conv2d(c,2*c,3,padding=1)
        self.m4 = nn.Conv2d(2*c,2*c,4,stride=2,padding=1)
        self.m5 = nn.Conv2d(2*c,2*c,3,padding=1)
        self.m6 = nn.Conv2d(2*c,2*c,4,stride=2,padding=1)

    def size(self):
        s =  2*self.c*(self.h//8)*(self.h//8)
        #print(s, type(s))
        return s

    def forward(self,x):
        x = F.relu(self.m1(x))
        x = F.relu(self.m2(x))
        x = F.relu(self.m3(x))
        x = F.relu(self.m4(x))
        x = F.relu(self.m5(x))
        x = F.relu(self.m6(x))

        #print(x.shape)
        x = x.view(x.size(0) , self.size())
        return x

class CNN3(nn.Module):
    def __init__(self, h, c):
        super(CNN3, self).__init__()
        self.c = c
        self.h = h

        self.m1 = nn.Conv2d(c,c,4,stride=2,padding=1)
        self.m2 = nn.Conv2d(c,2*c,3,padding=1)
        self.m3 = nn.Conv2d(2*c,2*c,4,stride=2,padding=1)
        self.m4 = nn.Conv2d(2*c,4*c,3,padding=1)
        self.m5 = nn.Conv2d(4*c,4*c,4,stride=2,padding=1)

    def size(self):
        s =  4*self.c*(self.h//8)*(self.h//8)

        #print(s, type(s))
        return s

    def forward(self,x):
        x = F.relu(self.m1(x))
        x = F.relu(self.m2(x))
        x = F.relu(self.m3(x))
        x = F.relu(self.m4(x))
        x = F.relu(self.m5(x))

        #print(x.shape)
        x = x.view(x.size(0) , self.size())
        return x

