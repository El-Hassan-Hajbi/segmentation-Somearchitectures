# import the necessary packages
from . import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
import torch

        
class Block(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        return self.conv2(self.relu(self.conv1(x)))


class SimpleConv(Module):
    def __init__(self, enc_channels=(3, 16, 32, 64),
                 dec_channels=(64, 32, 16, 1)):
        super().__init__()

        # store the encoder blocks and maxpooling layer
        self.enc_blocks = ModuleList([Block(enc_channels[i], enc_channels[i+1])
                                     for i in range(len(enc_channels) - 1)])
        # TODO: prepare pooling

        # initialize decoder # channels, upsampler blocks, and decoder blocks
        self.channels = dec_channels
        self.dec_blocks = ModuleList(
            [Block(dec_channels[i], dec_channels[i + 1])
             for i in range(len(dec_channels) - 1)])
        # TODO: prepare upconvolutions
    
    def forward(self, x):
        # loop through the encoder blocks
        for block in self.enc_blocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            # TODO: pooling

        # decoder: loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            # TODO: upconvolutions
            x = self.dec_blocks[i](x)
        #
        # return the final decoder output
        return x


class UNet(Module):
    def __init__(self, enc_channels=(3, 16, 32, 64),
                 dec_channels=(64, 32, 16, 1)):
        super().__init__()

        # store the encoder blocks and maxpooling layer
        self.enc_blocks = ModuleList([Block(enc_channels[i], enc_channels[i+1])
                                     for i in range(len(enc_channels) - 1)])
        # prepare pooling
        self.pooling = MaxPool2d(kernel_size=2, stride=2)

        # Couche linéaire pour obtenir le latent space
        self.fc1 = torch.nn.Linear(16*16*64, 128) # last pooling to latent space 

        # initialize decoder # channels, upsampler blocks, and decoder blocks
        self.channels = dec_channels
        self.dec_blocks = ModuleList(
            [Block(2*dec_channels[i], 2*dec_channels[i + 1])
             for i in range(len(dec_channels) - 1)])
        # TODO: prepare upconvolutions
        self.upconv = ModuleList([ConvTranspose2d(dec_channels[0], dec_channels[0], kernel_size=3, stride=2, padding=1, output_padding=1)]+[ConvTranspose2d(2*dec_channels[i], 2*dec_channels[i], kernel_size=3, stride=2, padding=1, output_padding=1) for i in range(1, len(dec_channels)-1)])
        self.outconv = ConvTranspose2d(dec_channels[-1], 1, kernel_size=3, stride=2, padding=1, output_padding=1)
    def forward(self, x):
        memory=[]
        # loop through the encoder blocks
        for block in self.enc_blocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            c = block(x)
            memory.append(c)
            # pooling
            x = (self.pooling)(c)

         # decoder: loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            # upconvolutions
            x = self.upconv[i](x)
            #x = self.dec_blocks[i](x+memory[-1-i])
            concat = torch.cat((memory[-1-i], x), 1)
                
            x = self.dec_blocks[i](concat)
        
        x = (self.outconv)(x)
        return x
        


class HourGlass(Module):
    def __init__(self, enc_channels=(3, 16, 32, 64),
                 dec_channels=(64, 32, 16, 1)):
        super().__init__()

        # store the encoder blocks and maxpooling layer
        self.enc_blocks = ModuleList([Block(enc_channels[i], enc_channels[i+1])
                                     for i in range(len(enc_channels) - 1)])
        # prepare pooling
        self.pooling = MaxPool2d(kernel_size=2, stride=2)

        # Couche linéaire pour obtenir le latent space
        self.fc1 = torch.nn.Linear(16*16*64, 128) # last pooling to latent space 

        # initialize decoder # channels, upsampler blocks, and decoder blocks
        self.channels = dec_channels
        self.dec_blocks = ModuleList(
            [Block(dec_channels[i], dec_channels[i + 1])
             for i in range(len(dec_channels) - 1)])
        # TODO: prepare upconvolutions
        self.upconv = ModuleList([ConvTranspose2d(dec_channels[i], dec_channels[i], kernel_size=3, stride=2, padding=1, output_padding=1) for i in range(1, len(dec_channels))])

    
    def forward(self, x):
        # loop through the encoder blocks
        for block in self.enc_blocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            # pooling
            x = (self.pooling)(x)
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten spatial dimensions
        x = (self.fc1)(x)
        # Reshape back to spatial dimensions
        x = x.view(batch_size, 128, 1, 1)"""
        #x = (ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1))(x)
        # decoder: loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            # upconvolutions
            x = self.dec_blocks[i](x)
            x = self.upconv[i](x)
        # return the final decoder output
        #x = Conv2d(3, 3, 3, padding=1)(x)
        return x
