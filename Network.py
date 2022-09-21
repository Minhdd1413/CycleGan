import torch.nn as nn
import torch.nn.functional as F
import torch

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride = 2, padding = 1), # channels: 3 -> 64
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(64, 128, 4, stride = 2, padding = 1), # channels: 64 -> 128
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(128, 256, 4, stride = 2, padding = 1), # channels: 128 -> 256
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(256, 512, 4, stride = 2, padding = 1), # channels: 256 -> 512
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),      
            
            nn.Conv2d(512, 1, 4, padding = 1) # channels: 512 -> 1
        )

    def forward(self, x):
        x = self.main(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x 

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # setup khối conv    
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # downsampling
            nn.Conv2d(64, 128, 4, stride = 2, padding = 1), # channels: 64 -> 128
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride = 2, padding = 1), # channels: 128 -> 256
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(inplace=True),           
            
            # khối transform 6 block 
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),
            Residual(256),

            # upsampling
            nn.Conv2d(256, 128, 4, stride = 2, padding = 1), # channels: 256 -> 128
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 64, 4, stride = 2, padding = 1), # channels: 256 -> 64
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(inplace=True), 
            
            # output
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7), # 64 -> 3 = input channels
            nn.Tanh()                       
        )
    
    def forward(self, x):
        return self.main(x)
        
class Residual(nn.Module):
    def __init__(self, input_channel) -> None:
        super(Residual, self).__init__()
        
        self.res = nn.Sequential(nn.ReflectionPad2d(1),
                                 nn.Conv2d(input_channel, input_channel, 4,),
                                 nn.InstanceNorm2d(input_channel),
                                 nn.ReLU(inplace=True),
                                 nn.ReflectionPad2d(1),
                                 nn.Conv2d(input_channel, input_channel, 4,),
                                 nn.InstanceNorm2d(input_channel),)
        
    def forward(self, x):
        return x + self.res