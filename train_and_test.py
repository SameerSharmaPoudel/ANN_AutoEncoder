import torch
from train_aux import training, validation, test
from log_training_data import log_recon_loss_and_Rsquare_score
import os

def train_and_return_best_model(model, device, config, loss, optimizer, train_dataloader, val_dataloader, path, autoencoder):
    
    """
        This function performs the training of the selected model in the selected 
        device following early stop algorithm, saves the model at various training epochs,
        saves the loss and accuracy measures in hdf file and returns the best model.
    """
    
    stopping_delay = config['Hyperparameters']['stopping_delay']
    best_val_loss = config['Hyperparameters']['best_val_loss']
       
    #error and accuracy measures    
    training_loss = []
    validation_loss = [] 
    training_accuracy = []
    validation_accuracy = []
    training_mape = []
    validation_mape = []
        
    for epoch in range(config['Hyperparameters']['num_epochs']):
            
        train_r_square, train_loss, train_mape = training(epoch, model, device, loss, optimizer, train_dataloader, autoencoder)
        val_r_square, val_loss, val_mape  = validation(epoch, model, device, loss, optimizer, val_dataloader, autoencoder)
                
        training_loss.append(train_loss) 
        validation_loss.append(val_loss)        
        training_accuracy.append(train_r_square) 
        validation_accuracy.append(val_r_square) 
        training_mape.append(train_mape)
        validation_mape.append(val_mape)  
        
        ###################   EARLY STOPPING   ################################
        ### if the validation loss doesn't decrease for the prespecified ######
        ##### no of epochs (stagnation delay), the training terminates.  ######
        
        if val_loss < best_val_loss :                      
            best_epoch = epoch
            best_val_loss  = val_loss
            stagnation = 0
            
            ##################### SAVING THE MODEL ###########################
            ###   if the validation loss decreases at certain epoch,     ######
            #####         the model at that epoch is saved.              ######
            checkpoint = {
                'epoch':best_epoch,
                'model_state': model.state_dict(),
                'optim_state':optimizer.state_dict(),
                }    
            PATH_model = os.path.join(path, f'model_at_epoch_{best_epoch}.pth')       
            torch.save(checkpoint, PATH_model)
       ########################################################################
            
        else:
            stagnation += 1
            
        if stagnation > stopping_delay:                  
            break       
    
    #########   so, the best epoch has the lowest validation loss   ###########
    #########  and the model saved at that epoch is returned        ###########
    print( 'Trained for {} epochs. Best epoch at {}.'.format( epoch, best_epoch))   
    PATH_best_model = os.path.join(path, f'model_at_epoch_{best_epoch}.pth')
    
    #saves loss and accuracy measures in hierarchial data format (hdf)
    log_recon_loss_and_Rsquare_score(training_loss, validation_loss, 
                                     training_accuracy, validation_accuracy, 
                                     training_mape, validation_mape, path)

    return PATH_best_model, training_loss, validation_loss, training_accuracy, validation_accuracy, training_mape, validation_mape
    

def reload_and_test_model(model, device, config, PATH_best_model, loss, optimizer, test_dataloader, autoencoder):
    
    """
         This function loads the checkpoint of the best model for Test
         and returns loss and accuracy measure.
    """
                                                            
    loaded_checkpoint = torch.load(PATH_best_model)
    epoch = loaded_checkpoint['epoch']
    model.load_state_dict(loaded_checkpoint['model_state'])
    optimizer.load_state_dict(loaded_checkpoint['optim_state'])
        
    test_r_square, test_loss, test_mape = test(model, device, loss, test_dataloader, autoencoder)
    print(f'Test Loss:{test_loss:.4f}, Test Accuracy:{test_r_square:.4f}, Test MAPE:{test_mape:.4f} ')
    
    return test_r_square, test_loss, test_mape
        
    
    







