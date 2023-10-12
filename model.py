import torch
import torch.nn as nn
import torch.nn.functional as F
import torch




def conv_block(in_chan,out_chan,stride=1):
    return nn.Sequential(
      nn.Conv3d(in_chan, out_chan, kernel_size = 3, padding =1, stride=stride),
      nn.InstanceNorm3d(out_chan),
      nn.LeakyReLU(0.2,inplace=True),
     
    )
    
class BFR_gen(nn.Module):
  
  def __init__(self):
    super().__init__()
    
    self.enc1 = conv_block(1,16)
    self.enc1_2 = conv_block(16,16)
    self.enc2 = conv_block(16,32)
    self.enc2_2 = conv_block(32,32)
    self.enc3 = conv_block(32,64)
    self.enc3_2 = conv_block(64,64)
    self.enc4 = conv_block(64,128)
    self.enc4_2 = conv_block(128,128)
    self.enc5 = conv_block(128,256)
    self.pool = nn.MaxPool3d(2,2)
    
    self.dec4 = conv_block(384,192)
    self.dec3 = conv_block(256,128)
    self.dec2 = conv_block(160,80)
    self.dec1 = conv_block(96,48)
    
    self.fc_1 = nn.Conv3d(48,16,1)
    self.fc_2 = nn.Conv3d(16,1,1)
    
  def forward(self,x):
        
        enc1 = self.enc1(x)
        enc1_2 = self.enc1_2(enc1)
        enc2 = self.enc2(self.pool(enc1_2))
        enc2_2 = self.enc2_2(enc2)
        enc3 = self.enc3(self.pool(enc2_2))
        enc3_2 = self.enc3_2(enc3)
        enc4 = self.enc4(self.pool(enc3_2))
        enc4_2 = self.enc4_2(enc4)
        enc5 = self.enc5(self.pool(enc4_2))
        
        dec4 = self.dec4(torch.cat((enc4,F.upsample(enc5,enc4.size()[2:], mode = 'trilinear')),1))
        dec3 = self.dec3(torch.cat((enc3,F.upsample(dec4,enc3.size()[2:], mode = 'trilinear')),1))
        dec2 = self.dec2(torch.cat((enc2,F.upsample(dec3,enc2.size()[2:], mode = 'trilinear')),1))
        dec1 = self.dec1(torch.cat((enc1,F.upsample(dec2,enc1.size()[2:], mode = 'trilinear')),1))
        
        fc_1 = self.fc_1(dec1)
        out = self.fc_2(fc_1)
        
        return out


class BFR_dis(nn.Module):
    def __init__(self, in_channels=1, dim=64, out_conv_channels=64):
        super(BFR_dis, self).__init__()
        conv1_channels = int(out_conv_channels / 8) 
        conv2_channels = int(out_conv_channels / 4) 
        conv3_channels = int(out_conv_channels / 2) 
        self.out_conv_channels = out_conv_channels 
        self.out_dim = int(dim / 16) 

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels, out_channels=conv1_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv1_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv2_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv3_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv3_channels, out_channels=out_conv_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(out_conv_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(out_conv_channels * self.out_dim * self.out_dim * self.out_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and apply linear + sigmoid
        x = x.view(-1, self.out_conv_channels * self.out_dim * self.out_dim * self.out_dim)
        x = self.out(x)
        return x


