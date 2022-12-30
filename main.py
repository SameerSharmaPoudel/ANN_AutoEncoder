import torch
from torch.nn import MSELoss
from torch.optim import Adam
from model_utils import feed_model_to_device, load_config
from train_and_test import train_and_return_best_model, reload_and_test_model
import csv
import time
from datetime import date
import sys
import os
sys.path.append('models/')
from models import model_list, model_name
#from generate_data import save_data, create_frame, widgets
from dataloader import return_dataloader
from post_process_utils import plot_loss_and_accuracy, plot_goodness_of_fit, plot_correlation_coefficient, plot_reconstructed_vs_input_stress

#Data Generation
#widgets()
#create_frame()
#train_data_file_path, ready_to_train = save_data()

config = load_config('config1.yaml') 
dataset, train_dataloader, val_dataloader, test_dataloader = return_dataloader(config)

parent_dir = 'result_data' 
#autoencoder = config['Model']['autoencoder']
#the list of models in 'models.py' should be consistent with the training type (for.eg. autoencoder='True' or 'False')
autoencoder = 'True'

for model_idx in range(len(model_list)): #loops through all models in the list
    for no_of_exp in range(1): #loops 5 times for 5 experiment runs
           
        ########## CREATE DIRECTORY TO SAVE RESULTS DURING TRAINING ###########
        
        #directory = model_name[model_idx] + '_' + str(config['Hyperparameters']['train_batch_size']) + '_' + str(config['Hyperparameters']['lr']) + '_' + str(config['Data_log']['exp_no']) 
        directory = model_name[model_idx]+'_'+str(no_of_exp+1)
        path = os.path.join(parent_dir, directory)
        try:
            os.makedirs(path, exist_ok = True)
            print("Directory '%s' created successfully" %directory)
        except OSError as error:
            print("Directory '%s' can not be created")
        #######################################################################
        
        
        ##########   TRAIN, SAVE RESULTS AND TEST THE MODEL####################
        start_time = time.time()
        
        model = model_list[model_idx]  
        model, device = feed_model_to_device(model, config)  #user defined function
        loss = MSELoss() #pytorch inbuilt
        optimizer = Adam(model.parameters(), config['Hyperparameters']['lr']) #pytorch inbuilt
        PATH_best_model, training_loss, validation_loss, training_accuracy, validation_accuracy, training_mape, validation_mape = train_and_return_best_model(model, device, config, loss, optimizer, train_dataloader, val_dataloader, path, autoencoder)    
        test_r_square, test_loss, test_mape = reload_and_test_model(model, device, config, PATH_best_model, loss, optimizer, test_dataloader, autoencoder)    
        
        end_time = time.time()
        train_time = (end_time-start_time)/60
        print(f'Training time: {train_time} minutes')
        #######################################################################
        
        
        ################ POST PROCESS THE RESULTS #############################
        
        best_validation_loss = min(validation_loss)
        epoch_best_validation_loss = validation_loss.index(best_validation_loss)       
        best_mape = min(validation_mape)
        epoch_best_mape = validation_mape.index(best_mape)   
        best_accuracy = max(validation_accuracy)
        epoch_best_accuracy = validation_accuracy.index(best_accuracy)  
        
        # can be plotted using the stored data as well
        plot_loss_and_accuracy(training_loss, validation_loss, training_mape, 
                               validation_mape, best_validation_loss,
                               epoch_best_validation_loss, best_mape, epoch_best_mape,
                               save_path=os.path.join(path, 'MAE_and_MAPE_loss.png'))
               
        if autoencoder=='False':
            _, train_dataloader, val_dataloader, test_dataloader = return_dataloader(config, 'postprocess')
            plot_goodness_of_fit(train_dataloader, val_dataloader, test_dataloader, model, device, path)
        
        data_save_path = plot_reconstructed_vs_input_stress(dataset, model, device, path, config)
        
        plot_correlation_coefficient(data_save_path, path)
        
        header = ['Model','Run','Best Validation Loss','Best Epoch', 'Best MAPE', 'Best Epoch',
                  'Best Validation Accuracy','Best_Epoch', 'Test Loss', 'Test MAPE','Test Accuracy', 
                  'Training Time(mins)']
        data = [[model_name[model_idx], no_of_exp+1, best_validation_loss, epoch_best_validation_loss,best_mape,
                 epoch_best_mape, best_accuracy, epoch_best_accuracy, test_loss, test_mape, test_r_square, train_time]]
        date = date.today()
        filename = os.path.join(parent_dir, f'Training_Summary_{date}.csv')
        with open(filename, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            if no_of_exp+1 == 1:
                writer.writerow(header)
            writer.writerows(data)        
        print('Summary printed in CSV file!')
        #######################################################################
        
        break
    
    
   
    
