import torch
from eval_metrics import compute_r_square, compute_error_metrics

def training(epoch, model, device, loss, optimizer, train_dataloader, autoencoder):
    
    """
         This function trains the selected model during which the parameters are updated.
    """
       
    running_train_loss = 0.0
    running_r_square = 0.0
    running_mape = 0.0
    model.train() 

    for data in train_dataloader: #loops through the data in the train dataset rach time picking a batch     
                                  #the model sees all data in train dataset in each epoch
                               
        input, target = data[0], data[1]
        input, target = input.to(device), target.to(device) #data is fed into the right device for doing the necessary calculations    
        if autoencoder == 'True':
            _, output = model(input)  # autoencoder model has two outputs #the not used output is the latent space feature
            loss_mse = loss(input, output) #the loss between output and target is computed #for autoencoder, the input acts as target(or ground truth or label)
        else:
            output = model(input)
            loss_mse = loss(target, output) 
        running_train_loss += loss_mse.item() 
        loss_mse.backward() #the gradients of the loss wrt parameters are calculated following the back propagation
        optimizer.step() #the parameters are updated with the help of gradients
        optimizer.zero_grad() #the parameters are set to zero so they don't accumulate 
                              #and new set of gradients are computed in the next epoch
        
        running_r_square += compute_r_square(input, output) #user defined function
        _,_,mape = compute_error_metrics(input, output)  #user defined function
        running_mape += mape 
   
    training_r_square =  running_r_square/ len(train_dataloader)   
    training_loss = running_train_loss / len(train_dataloader)
    training_mape = running_mape / len(train_dataloader)
    print(f'Epoch:{epoch}, Training Loss:{training_loss:.4f}')  
    
    return training_r_square, training_loss, training_mape


def validation(epoch, model, device, loss, optimizer, val_dataloader, autoencoder):
    
    """
         This function evaluates the selected model on validation dataset
         simulaneously during training. The parameters, however, are not updated.
    """
         
    with torch.no_grad():
        
        running_val_loss = 0.0
        running_r_square = 0.0
        running_mape = 0.0
        model.eval()
        
        for data in val_dataloader:    

            input, target = data[0], data[1]
            input, target = input.to(device), target.to(device)
            _, output = model(input) 
            if autoencoder == 'True':
                _, output = model(input)
                loss_mse = loss(input, output) #the loss between output and target is computed
            else:
                output = model(input)
                loss_mse = loss(target, output) 
            running_val_loss += loss_mse.item()
            
            running_r_square += compute_r_square(input, output)
            _,_,mape = compute_error_metrics(input, output) 
            running_mape += mape 
            
        validation_r_square =  running_r_square/len(val_dataloader)            
        validation_loss = running_val_loss / len(val_dataloader)
        validation_mape = running_mape / len(val_dataloader)
        print(f'Epoch:{epoch}, Validation Loss:{validation_loss:.4f}')
     
        return validation_r_square, validation_loss, validation_mape


def test(model, device, loss, test_dataloader, autoencoder):
    
    """
         This function evaluates the selected model on test dataset after the 
         whole training process is completed.
    """
    
    with torch.no_grad():
        
        running_test_loss = 0.0
        running_r_square = 0.0
        running_mape = 0.0
        model.eval()
        
        for data in test_dataloader:                
        
             input, target = data[0], data[1]
             input, target = input.to(device), target.to(device)
             if autoencoder == 'True':
                 _, output = model(input)
                 loss_mse = loss(input, output) #the loss between output and target is computed
             else:
                 output = model(input)
                 loss_mse = loss(target, output)   
             running_test_loss += loss_mse.item()
             
             running_r_square += compute_r_square(input, output)
             _,_,mape = compute_error_metrics(input, output) 
             running_mape += mape 
             
        test_r_square =  running_r_square/len(test_dataloader)            
        test_loss = running_test_loss / len(test_dataloader)
        test_mape = running_mape / len(test_dataloader)
        
        return test_r_square, test_loss, test_mape
