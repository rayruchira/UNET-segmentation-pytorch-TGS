import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

#double convolution used in unet steps
class DoubleConvBlock(nn.Module):
    def __init__(self, input, output):
        super(DoubleConvBlock,self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(input,output,3,1,1, bias=False), #same convolution, False cause using batchnorm
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
            nn.Conv2d(output,output,3,1,1, bias=False), #same convolution, False cause using batchnorm
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
          )
    def forward( self, x):
      return self.block(x)

#Unet model
class Unet(nn.Module):
  def __init__(self, input=3, output=1, features= [64,128,256,512]):
    super(Unet, self).__init__()
    self.down=nn.ModuleList()
    self.up=nn.ModuleList()

    self.pool=nn.MaxPool2d(2,2)

    #going down
    for f in features:
      self.down.append(DoubleConvBlock(input, f)) #mapping
      input=f

    #going up
    for f in features[::-1]:
      self.up.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2,)) #doubles height and width of image
      self.up.append(DoubleConvBlock(f*2, f))

    self.middle=DoubleConvBlock(features[-1], features[-1]*2)

    #the final conv
    self.finalConv=nn.Conv2d(features[0], output, 1)

  def forward(self, x):
    skipConnections=[]

    #save the skips when going down
    for downconv in self.down:
      x= downconv(x)
      skipConnections.append(x)
      x=self.pool(x)
    
    #the bottleneck transition
    x=self.middle(x)

    #reverse the skip connections 
    skipConnections.reverse()
   
    #going up +add skips
    for upconv in range(0, len(self.up) , 2 ): # step to 2 to add skip connection in between
      # break up the list to add skip in between
      x= self.up[upconv](x)
      skip =skipConnections[upconv//2]

      #incase dimension not divisible by 16
      if x.shape != skip.shape:
        x = TF.resize(x, size=skip.shape[2:])

      # print(x.shape)
      # print(skipConnections.shape)
      concatSkips= torch.cat((skip, x), dim=1) #batch , channel, height , width :: we need to add along channel
      x= self.up[upconv+1](concatSkips)


    return self.finalConv(x)


#testing dimensions
def test():
  x=torch.randn((3, 3, 160, 160))
  model= Unet(input= 3, output=3)
  p=model(x)
  # print(p.shape, x.shape)
  assert p.shape==x.shape


if __name__ == "__main__":
    test()
