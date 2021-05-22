import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConvBlock(nn.Module):
    def __init__(self, input, output):
        super(DoubleConvBlock,self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(input,output,3,1,1, bias=False), #same convolution, False cause using batchnorm
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
            nn.Conv2d(input,output,3,1,1, bias=False), #same convolution, False cause using batchnorm
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
          )
    def forward( self, x):
      return self.block(x)
