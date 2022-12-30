import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('result_data/')
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math
import torch
from eval_metrics import compute_r_square, compute_error_metrics
plt.rcParams['text.usetex'] = True
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_loss_and_accuracy(training_loss, validation_loss, training_mape, validation_mape, 
                           best_validation_loss, epoch_best_validation_loss,
                           best_mape, epoch_best_mape, save_path):
    
    """
       This function plots MSE loss and MAPE evolution during the training process
       and shows the lowest values for both along with the epoch at which those
       values were achieved.
    """
        
    fig, axs = plt.subplots(1, 2)
    
    axs[0].semilogy(training_loss, label='Training') 
    axs[0].semilogy(validation_loss, label='Validation') 
    axs[0].set_xlabel('No of Epoch [-]', fontsize=13)
    axs[0].set_ylabel('MSE Loss [-]', fontsize=13)
    axs[0].axvline(x=epoch_best_validation_loss, color='#FF3339', label=f'best epoch =  {epoch_best_validation_loss}', ls='--')
    axs[0].axhline(y=best_validation_loss, color='#F615DE', label=f'best val loss = {best_validation_loss:.4f}', ls='--')
    axs[0].legend(loc='upper right', fontsize=10)
    axs[0].grid()
    
    axs[1].plot(training_mape, label='Training') 
    axs[1].plot(validation_mape, label='Validation') 
    axs[1].set_xlabel('No of Epoch [-]', fontsize=13)
    #axs[1].set_ylabel(r'$ R^{2} $ Score [-]', fontsize=13)
    axs[1].set_ylabel('MAPE [-]', fontsize=13)
    axs[1].axvline(x=epoch_best_mape, color='#FF3339', label=f'best epoch =  {epoch_best_mape}', ls='--')
    axs[1].axhline(y=best_mape, color='#F615DE', label=f'best mape = {best_mape:.4f}', ls='--')
    axs[1].legend(loc='upper right', fontsize=10)
    #axs[1].set_ylim(0.5, 1)
    axs[1].grid()
    #axs[i,j].tick_params (labelsize=22) 
    
    fig.tight_layout()        
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)        
    plt.savefig(save_path)
    plt.show()
    
    print('MSE and MAPE Evolution Plotted!')
    
    
def plot_goodness_of_fit(train_dataloader, val_dataloader, test_dataloader, model, device, path):
    
    """
       This function plots goodness of fit separately for train, validation and test dataset.
    """
    
    dataloader_list = [train_dataloader, val_dataloader, test_dataloader]
    dataloader_name_list = ['train', 'val', 'test']    
       
    for loader, dataloader_name in zip(dataloader_list, dataloader_name_list):
                
        ground_truth = []
        prediction = []    
        for data in loader:               
            input, target = data[0], data[1]
            input, target = input.to(device), target.to(device)
            output = model(input)
            
            target, output = target.cpu().detach().numpy(), output.cpu().detach().numpy()
            ground_truth.append(target[0])
            prediction.append(output[0])
              
        ground_truth, prediction = np.array(ground_truth), np.array(prediction)

        for p in range(8):
        
            fig, axs = plt.subplots(1, 1)
            a, b = np.polyfit(ground_truth[:,p], prediction[:,p], 1)
            a=1
            b=0
            axs.scatter(ground_truth[:,p], prediction[:,p], label=f'prediction_{dataloader_name}')   
            axs.plot(ground_truth[:,p], a* ground_truth[:,p] +b, color='red', label='linear fit line', linestyle='-', linewidth=2)
            axs.legend(loc='upper left', fontsize=8)
            axs.grid()
                               
            fig.tight_layout()        
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)      
            image_filename = os.path.join(path, f'goodness_of_fit_line_{dataloader_name}_{p+1}.png') 
            plt.savefig(image_filename)
            plt.show()
            
            print('Goodness of fit line plotted!')

            
def plot_reconstructed_vs_input_stress(dataset, model, device, path, config):
    
    """
       This function plots reconstructed vs input stresses in two subplots.
       The first one plots the stresses in rolling plane while the second one 
       plots the stresses in yield locus.
    """
    incre1 = autoencoder = config['Model']['angle_incre1_for_stress_in_yield_locus']
    incre2 = autoencoder = config['Model']['angle_incre2_for_stress_in_roll_plane']
    
    mae_whole_dataset = []
    predicted_whole_dataset = []
    input_whole_dataset = []
    latent_whole_dataset = []
    target_para_whole_dataset = []      
    
    for i in range(len(dataset)): #looping through whole dataset individually instead in a batch
                                  #and saving them in separate lists
            input, target = dataset[i]
            input = torch.from_numpy(input)
            input = input.to(device)
            latent_feature, output = model(input) 
            _,mae,_ = compute_error_metrics(input, output) 
            mae_whole_dataset.append(mae)
            input, output, latent_feature = input.cpu().detach().numpy(), output.cpu().detach().numpy(), latent_feature.cpu().detach().numpy()  #convert pytorch tensor to numpy array
            predicted_whole_dataset.append(output)
            input_whole_dataset.append(input)
            latent_whole_dataset.append(latent_feature)
            target_para_whole_dataset.append(target)
    
    #the input stress and the corresponding model parameters (target_parameter) 
    #along with the predicted stress, mae, latent feature are saved in hdf file
    
    data_save_path=os.path.join(path, 'input_latent_output_targetpara_mae.h5')
    F = h5py.File(data_save_path, 'a')
    group1 = F.require_group('input_stress')  
    group2 = F.require_group('predicted_stress') 
    group3 = F.require_group('mae')
    group4 = F.require_group('latent_feature')
    group5 = F.require_group('target_parameter')   #alphas and exponent
    group1.require_dataset('input_stress', data= input_whole_dataset, shape=np.shape(input_whole_dataset), dtype=np.float64, compression='gzip')
    group2.require_dataset('predicted_stress', data= predicted_whole_dataset, shape=np.shape(predicted_whole_dataset), dtype=np.float64, compression='gzip')
    group3.require_dataset('mae', data= mae_whole_dataset, shape=np.shape(mae_whole_dataset), dtype=np.float64, compression='gzip')
    group4.require_dataset('latent_feature', data= latent_whole_dataset, shape=np.shape(latent_whole_dataset), dtype=np.float64, compression='gzip')
    group5.require_dataset('target_parameter', data= target_para_whole_dataset, shape=np.shape(target_para_whole_dataset), dtype=np.float64, compression='gzip')
    F.close()
    print('Input, Predicted, Target, Latent Feature and MAE stored!')

    angle_yield_loci = [x for x in range(315,360,incre1)]
    angle_yield_loci.extend([x for x in range(0,138,incre1)])
    
    angle_rolling_dir = [x for x in range(1,90,incre2)]
        
    for i in range(1,6): #looping for five worst cases #or sorting can be used 
        
        max_mae = max(mae_whole_dataset)
        print('Worst MAE on whole dataset:',max_mae)
        idx = mae_whole_dataset.index(max_mae)
        
        ######################################################################
        ## Calculating x and y components from stress points in yield locus ###
        #######################################################################
        input_x = []
        input_y = []
        output_x = []
        output_y = []
        for deg, sigma_input, sigma_output in zip(angle_yield_loci, input_whole_dataset[idx], predicted_whole_dataset[idx]):
            rad = deg/180 * math.pi
            input_x.append(sigma_input * math.cos(rad))
            output_x.append(sigma_output * math.cos(rad))
            input_y.append(sigma_input * math.sin(rad))
            output_y.append(sigma_output * math.sin(rad))
       ########################################################################             
        
        fig = plt.figure()
        fig.set_figheight(7)
        fig.set_figwidth(15)
        
        ax1 = plt.subplot2grid(shape=(5, 10), loc=(0, 0), colspan=5, rowspan=3) 
        ax1.scatter(angle_rolling_dir, input_whole_dataset[idx][61:], label='Input')
        ax1.scatter(angle_rolling_dir, predicted_whole_dataset[idx][61:], label='Prediction')                         
        ax1.set_xlabel('angle (in degree) wrt RD', fontsize=13)
        ax1.set_ylabel('Stress', fontsize=13)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid()
        
        ax2 = plt.subplot2grid(shape=(5, 10), loc=(1, 5), colspan=4, rowspan=4) 
        ax2.plot(input_x, input_y, label='Input', linewidth=4) 
        ax2.plot(output_x, output_y, label='Prediction', linewidth=4)
        ax2.set_xlabel(r'$ sigma_xx $', fontsize=13)
        ax2.set_ylabel(r'$ sigma_yy $', fontsize=13)
        ax2.legend(loc='lower left', fontsize=10)
        ax2.grid()
        
        fig.tight_layout()        
        #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)      
        image_filename = os.path.join(path, f'input_vs_prediction_{i}.png') 
        plt.savefig(image_filename)
        plt.show()
                
        mae_whole_dataset.pop(idx)
        predicted_whole_dataset.pop(idx)
        input_whole_dataset.pop(idx)
        
    print('Input vs Predicted Stresses Plotted!')
    return data_save_path
        
        
def plot_correlation_coefficient(data_save_path, path):
    
    """
    Computes Pearson Correlation Coefficients between the latent space features, 
    between the latent space features and model parameters, and between the model 
    parameters. And show them in single plot.
    
    Formula: (ref:https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) 
    """
    
    latent_feature = h5py.File(data_save_path, 'r')['latent_feature/latent_feature'][:]
    target_parameter = h5py.File(data_save_path, 'r')['target_parameter/target_parameter'][:]
    
    p_corr_coeff_matrix = np.corrcoef(latent_feature, target_parameter, rowvar=False)
        
    fig, axs = plt.subplots(1, 1, figsize=(10,10))
        
    im = axs.imshow(p_corr_coeff_matrix)
    axs.set_title('Pearson Corr. Coeff.')
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="7%", pad=0.07)
    cb = plt.colorbar(im, cax=cax)
    #cb.ax.tick_params(labelsize=20)
    fig.tight_layout()
    
    image_filename = os.path.join(path, 'pearson_correlation_plot.png') 
    plt.savefig(image_filename)
    