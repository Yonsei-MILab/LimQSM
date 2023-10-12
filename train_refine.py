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
from torch.autograd import Variable

from utils import *
from model import *


num_epoch = ''
use_cuda = torch.cuda.is_available()
epoch_loss_values_g = []
epoch_loss_values_d = []
bbb = (1,1,64,64,64)
zzz = np.zeros(bbb)
zz = torch.Tensor(zzz)
devicec = torch.device('cpu')
device = torch.deivce('gpu')
global M


FloatTensor = torch.cuda.FloatTensor if torch.cuda else torch.FloatTensor
valid = Variable(FloatTensor(1, 1).fill_(1.0), requires_grad=False)
fake = Variable(FloatTensor(1, 1).fill_(0.0), requires_grad=False)

gen_coarse = BFR_gen().to(device)
gen_coarse.load_state_dict(torch.load(os.path.join("../BFR_extension_gen.pth")))
gen_refine = BFR_gen().to(device)
dis = BFR_dis().to(device)

optimizer_gen = torch.optim.AdamW(gen_refine.parameters(), lr=lr)
optimizer_dis = torch.optim.AdamW(dis.parameters(), lr=lr)
gen_loss = l1_loss.to(device)
dis_loss = l1_loss.to(device)

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

# Main function for training
def train_model_refine(num_epoch):
    for epoch in range(num_epoch):
        # Initialize the losses
        traing_loss = 0.0
        traind_loss = 0.0
        iteration = 0

        # Fetch the training data
        A = get_data_train(train_dataset)

        # Train the generator and discriminator
        traing_loss, fine_pred = train_generator(A[0].to(device),A[1].to(device), A[2].to(device), A[3].to(device), traing_loss, iteration)
        traind_loss = train_discriminator(A[0].to(device), A[1].to(device), A[2].to(device), A[3].to(device), fine_pred, traind_loss, iteration)

        # Save model every 100 epochs
        if (epoch % 100 == 0):
            savePath = "../BFR_extension_GAN.pth"
            torch.save(gen_refine.state_dict(),savePath)

        print('Epoch: {} \tGenerator Loss: {:.8f} \tDiscriminator Loss: {:.8f}'.format(epoch+1, traing_loss, traind_loss))

# Function to train generator
def train_generator(input, Mask, Maskmimi, label, traing_loss, iteration):
    # Coarse to FINE
    gen_coarse.eval()
    coarse_pred = gen_coarse(input).to(device)
    coarse_to_fine = (coarse_pred * Maskmimi) + (input * Mask) ## Fine generator input
    coarse_to_fine = coarse_to_fine.to(device)

    # Fine Generator
    gen_refine.train()
    optimizer_gen.zero_grad()

    fine_pred = gen_refine(coarse_to_fine).to(device)   # Fine generator output

    # Loss calculations and backpropagation
    finalg_loss = calculate_gen_loss(fine_pred, Mask, Maskmimi, label)
    traing_loss += finalg_loss.item()
    finalg_loss.backward()
    optimizer_gen.step()

    iteration +=1
    traing_loss = traing_loss/iteration
    epoch_loss_values_g.append(traing_loss)

    return traing_loss, fine_pred

# Function to train discriminator
def train_discriminator(input, Mask, Maskmimi, label, fine_pred, traind_loss, iteration):
    dis.train()
    optimizer_dis.zero_grad()

    real_loss = dis_loss(dis(label*Maskmimi + input*Mask),valid)
    fake_loss = dis_loss(dis(fine_pred.detach()),fake)
    ad_loss = (real_loss - fake_loss)

    penalty_loss = calc_gradient_penalty(dis,(label*Maskmimi+input*Mask),fine_pred.detach()); 

    finald_loss = ad_loss + 10*penalty_loss
    traind_loss += finald_loss.item()
    finald_loss.backward()
    optimizer_dis.step()

    traind_loss = traind_loss/iteration
    epoch_loss_values_d.append(traind_loss)

    return traind_loss

# Function to calculate generator loss
def calculate_gen_loss(fine_pred, Mask, Maskmimi, label):
    # Recon loss
    recon_loss_L1 = gen_loss(fine_pred*(Mask), label*(Mask)) 


    # Laplacian loss
    preds = fine_pred.to(devicec)
    Maskmimis = Maskmimi.to(devicec)
    lp_final = laplacian2(preds,Maskmimis)

    lapla_loss = gen_loss(zz,Maskmimis*lp_final)

    # Adversarial loss
    fakeg_loss = dis_loss(dis(fine_pred),valid)

    # Total loss
    finalg_loss = recon_loss_L1  + ( 0.01* fakeg_loss) + (0.1*lapla_loss)
    
    return finalg_loss

# Call the main function
train_model_refine(num_epoch)
