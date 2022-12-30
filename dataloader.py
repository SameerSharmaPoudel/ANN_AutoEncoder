import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random

class my_dataset(): 
    
    def __init__(self, config):

        self.config = config
        self.F = h5py.File(self.config['Dataset']['train_data_file_path'],'r')
                   
    def __getitem__(self, index):
               
        input_path = self.config['Dataset']['input_path']
        input = self.F[input_path.format(index+1)][:]
        
        target_path = self.config['Dataset']['target_path']
        target = self.F[target_path.format(index+1)][:]
        
        return (input,target)
    
    def __len__(self):

        return len(self.F['input_data'])

def return_dataloader(config,*arg):
           
    dataset = my_dataset(config)
    test_percentage, val_percentage = config['Dataset']['test_percentage'], config['Dataset']['val_percentage'] 
    test_amount  = int(len(dataset) * test_percentage)
    val_amount = int(len(dataset) * val_percentage)                 
    train_amount = int(len(dataset) - (test_amount + val_amount))
    
    train_set, val_set, test_set = random_split(dataset, [train_amount, val_amount,	test_amount])	
    
    train_batch_size, val_batch_size, test_batch_size = config['Hyperparameters']['train_batch_size'], config['Hyperparameters']['val_batch_size'], config['Hyperparameters']['test_batch_size']
    if arg == 'postprocess':
        train_batch_size, val_batch_size, test_batch_size = train_amount, val_amount, test_amount
    
    #Dataloader is inbuilt function in pytorch, eachtime to get a new batch
    train_dataloader = DataLoader(train_set, train_batch_size, shuffle=True, pin_memory=False, drop_last=False)
    val_dataloader = DataLoader(val_set, val_batch_size, pin_memory=False, drop_last=False)
    test_dataloader = DataLoader(test_set, test_batch_size, shuffle= False, pin_memory=False, drop_last=False)
    
    return dataset, train_dataloader, val_dataloader, test_dataloader


    