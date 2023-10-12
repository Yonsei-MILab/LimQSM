import torch
import scipy.io
import numpy as np
import h5py
import time
import os
from tqdm import tqdm
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch_loss_values_g = []
bbb = (1,1,64,64,64)
zzz = np.zeros(bbb)
zz = torch.Tensor(zzz)
devicec = torch.device('cpu')
device = torch.deivce('gpu')


# Load the train patch .mat file
data = scipy.io.loadmat('file_train.mat')
patches = data['patches']
patches_mask = data['patches_mask']
patches_maskmimi = data['patches_maskmimi']
patches_label = data['patches_label']

# Load the valid patch .mat file
data = scipy.io.loadmat('file_valid.mat')
patches_valid = data['patches']
patches_valid_mask = data['patches_mask']
patches_valid_maskmimi = data['patches_maskmimi']
patches_valid_label = data['patches_label']

train_dataset, valid_dataset = create_datasets(patches, patches_mask, patches_maskmimi, patches_label,
                    patches_valid, patches_valid_mask, patches_valid_maskmimi, patches_valid_label)

train_loader = get_data_train(train_dataset, bs='')  # Modify batch size as needed
valid_loader = get_data_valid(valid_dataset, bsv='')  # Modify batch size as needed

gen_coarse = BFR_gen().to(device)
gen_loss = l1_loss.to(device)
optimizer_gen_coarse = torch.optim.AdamW(gen_coarse.parameters(), lr=0.001)

def train_model_coarse(num_epoch):
    for epoch in range(num_epoch):
        # Initialize the losses
        traing_loss = 0.0
        iteration = 0

        # Fetch the training data
        A = get_data_train(train_dataset)

        # Train the generator and discriminator
        traing_loss, fine_pred = train_generator(A[0].to(device),A[1].to(device), A[2].to(device), A[3].to(device), traing_loss, iteration)
       

        # Save model every 100 epochs
        if (epoch % 100 == 0):
            savePath = "../BFR_extension_gen.pth.pth"
            torch.save(gen_coarse.state_dict(),savePath)

        print('Epoch: {} \tGenerator Loss: {:.8f} \tDiscriminator Loss: {:.8f}'.format(epoch+1, traing_loss, traind_loss))

# Function to train generator
def train_generator(input, Mask, Maskmimi, label, traing_loss, iteration):
    # Coarse generator
    coarse_pred = gen_coarse(input).to(device)

    # Loss calculations and backpropagation
    finalg_loss = calculate_gen_loss(coarse_pred, Mask, Maskmimi, label)
    traing_loss += finalg_loss.item()
    finalg_loss.backward()
    optimizer_gen_coarse.step()

    iteration +=1
    traing_loss = traing_loss/iteration
    epoch_loss_values_g.append(traing_loss)

    return traing_loss, coarse_pred

def calculate_gen_loss(fine_pred, Mask, Maskmimi, label):
    # Recon loss
    recon_loss_L1 = gen_loss(fine_pred*(Mask), label*(Mask)) 
 

    # Laplacian loss
    preds = fine_pred.to(devicec)
    Maskmimis = Maskmimi.to(devicec)
    lp_final = laplacian2(preds,Maskmimis)

    lapla_loss = gen_loss(zz,Maskmimis*lp_final)

    # Total loss
    finalg_loss = recon_loss_L1 + (0.1*lapla_loss)
    
    return finalg_loss

# Call the main function
train_model_coarse(num_epoch)
