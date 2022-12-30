import torch
import os
import yaml

# folder to load config file
config_path = 'config_files/'


def load_config(config_name):
    
    """
       This function loads yaml configuration file.
    """
    
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)
    return config


def feed_model_to_device(model, config):
    
    """
    This function selects the available gpu or cpu.
    """
    if config['Device']['select_GPU'] == 1 and torch.cuda.is_available():

            device = torch.device('cuda') #pytorch convention
            model.to(device)             #pytorch convention
            print('Running on the GPU')
        
    else:
        device = torch.device('cpu')
        model.to(device)    
        print('Running on the CPU')
        
    return model, device


def load_saved_model(model, optimizer, PATH):
    
    loaded_checkpoint = torch.load(PATH)
    epoch = loaded_checkpoint['epoch']
    model.load_state_dict(loaded_checkpoint['model_state'])
    optimizer.load_state_dict(loaded_checkpoint['optim_state'])
   
    return model
                     
