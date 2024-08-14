#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:07:15 2024

@author: andreasaspe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from LoadData import *
from Network import Simple3DCNN
from torch.utils.data import DataLoader
from tqdm import tqdm

# data_dir = '/Volumes/T9/OutlierChallenge2024/challenge_data/train'
data_dir = '/work3/rapa/challenge_data/train'
checkpoint_dir = "./checkpoints"
batch_size = 4
#Load data
Data_train = CTScanDataset(data_dir=data_dir,split='train')
train_loader = DataLoader(Data_train, batch_size=batch_size,
                        shuffle=True, num_workers=4, persistent_workers=True) #SET TO True!
Data_val = CTScanDataset(data_dir=data_dir,split='val')
val_loader = DataLoader(Data_val, batch_size=batch_size, num_workers=0) #SET TO True!

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

# Create the model
model = Simple3DCNN().double().to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    print("Epoch {}/{}".format(epoch+1,num_epochs))
    print("Training...")
    for inputs, targets in tqdm(train_loader):
        # Assumed: Inputs are (batch_size, channels, depth, height, width)
        # Note: Adjust targets based on your actual implementation
        # inputs = inputs[0] # Placeholder for targets
        
        # show_slices_dim1(inputs[1].numpy(),'hej')
        
        optimizer.zero_grad()
        
        outputs = model(inputs.to(device))
        # print(outputs.size(), targets.size())
        loss = criterion(outputs, targets.to(device).unsqueeze(1).double())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    
    
    running_loss = 0
    print("Validating...")
    for inputs, targets in tqdm(val_loader):
        # inputs = inputs[0] # Placeholder for targets
        
        with torch.no_grad():
            outputs = model(inputs.to(device))
            # print(outputs.size(), targets.size())
            loss = criterion(outputs, targets.to(device).unsqueeze(1).double())
        
        running_loss += loss.item()

    val_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.5f} Val Loss: {val_loss:.5f}')

        
    if epoch%10 == 0:
        #Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }

        torch.save(checkpoint, f"{checkpoint_dir}/simplecnn_{epoch}_{val_loss:.5f}.cpt")
    
print('Training complete!')


# #Load data
# VerSe_train = LoadData(img_dir=img_dir_training, heatmap_dir=heatmap_dir_training, msk_dir = msk_dir_training,transform=transform)
# train_loader = DataLoader(VerSe_train, batch_size=batch_size,
#                         shuffle=True, num_workers=0) #SET TO True!