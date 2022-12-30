import h5py
import numpy as np
from sklearn.utils.extmath import randomized_svd
import sys
import os
sys.path.append('models/')
from models import model_list, model_name

def log_recon_loss_and_Rsquare_score(training_loss, validation_loss, 
                                 training_accuracy, validation_accuracy, 
                                 training_mape, validation_mape, path):
    """
    stores training and validation losses, and r-squared score training and validation accuracies as lists in hdf5 format 
    """
    filename = os.path.join(path, 'error_and_accuracy.h5')
    F = h5py.File(filename, 'a')
    	
    group1 = F.require_group('training_loss')    
    group1.require_dataset('training_loss', data= training_loss, shape=np.shape(training_loss), dtype=np.float64, compression='gzip')
    
    group2 = F.require_group('validation_loss')    
    group2.require_dataset('validation_loss', data= validation_loss, shape=np.shape(validation_loss), dtype=np.float64, compression='gzip')
    
    group3 = F.require_group('training_r_square_score')    
    group3.require_dataset('training_r_square_score', data= training_accuracy, shape=np.shape(training_accuracy), dtype=np.float64, compression='gzip')
    
    group4 = F.require_group('validation_r_square_score')    
    group4.require_dataset('validation_r_square_score', data= validation_accuracy, shape=np.shape(validation_accuracy), dtype=np.float64, compression='gzip')
    
    group5 = F.require_group('training_mape')    
    group5.require_dataset('training_mape', data= training_mape, shape=np.shape(training_mape), dtype=np.float64, compression='gzip')
    
    group6 = F.require_group('validation_mape')    
    group6.require_dataset('validation_mape', data= validation_mape, shape=np.shape(validation_mape), dtype=np.float64, compression='gzip')
    									  
    F.close()	
    print('Loss and Accuracy Measures Stored!!!')