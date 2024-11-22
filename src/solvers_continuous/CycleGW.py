import torch
from tqdm.auto import trange
from tqdm import tqdm_notebook as tqdm
from src.metrics import compute_metrics
import wandb
from src.utils import computePotGrad

import numpy as np
import sys
import torch
import matplotlib.pyplot as plt

from functools import partial
from geomloss import SamplesLoss
def norm_square(X1, X2):
    batch_size_1 = X1.size(0)
    batch_size_2 = X2.size(0)

    expand_x1 = X1.expand(batch_size_2,batch_size_1,-1)
    expand_x1 = torch.transpose(expand_x1,0,1)
    expand_x2 = X2.expand(batch_size_1,batch_size_2,-1)
    square = torch.sum(torch.square(expand_x1-expand_x2),2)

    return square

def gaussian_kernel(sigmas, X1, X2):
    square = norm_square(X1,X2)
    #val=torch.exp(-square/sigma) + torch.exp(-square/(sigma*0.25))  + torch.exp(-square/(sigma*0.05))  + torch.exp(-square/(sigma*4))  + torch.exp(-square/(sigma*20))  + torch.exp(-square/(sigma*0.01)) + torch.exp(-square/(sigma*0.001))+torch.exp(-square/(sigma*0.0001))+ torch.exp(-square/(sigma*100))+ torch.exp(-square/(sigma*1000))
    val = 0
    for sigma in sigmas:
        val += torch.exp(-square/sigma)
    return val

def rational_kernel(a, X1, X2):
    square = norm_square(X1,X2)
    val = torch.pow(1+square/(2*a), -a)
    return val


def distortion(X, FX, Y, GY):
    delta1 = torch.mean( torch.abs(  norm_square(X,X) -  norm_square(FX,FX)  )  )
    delta2 = torch.mean( torch.abs(  norm_square(Y,Y) -  norm_square(GY,GY)  )  )
    delta3 = torch.mean( torch.abs(  norm_square(X,GY) -  norm_square(FX,Y)  )  )
    return (delta1 + delta2 + 2*delta3)

class CycleGW:
    def __init__(self, models, optimizers, eps, sigmas, reg, kernel_type='energy_guided', take_median=True):
        self.F_model = models['F_model']
        self.G_model = models['G_model']
        
        self.F_optimizer = optimizers['F_model']
        self.G_optimizer = optimizers['G_model']

        self.eps = eps
        self.sigmas = sigmas
        self.reg = reg
        self.take_median = take_median

        self.kernel_type = kernel_type
        
    def train_epoch(self, sampler_source, sampler_target, n_samples, n_iters, epoch, wandb_report):

        sigmas_x = self.sigmas#10
        sigmas_y = self.sigmas#10
        reg_x = self.reg#1
        reg_y = self.reg#1
        
        self.F_model.train()
        self.G_model.train()
        
        train_tqdm = tqdm(range(len(sampler_source)), leave=False, desc="Train")        

        for ix in train_tqdm:
            self.F_optimizer.zero_grad()
            self.G_optimizer.zero_grad()
            
            x_train, _ = sampler_source.sample(n_samples)
            y_train, _ = sampler_target.sample(n_samples)

            if self.kernel_type == 'gaussian' and self.take_median is True:
                median_x = torch.median(norm_square(x_train, x_train))
                median_y = torch.median(norm_square(y_train, y_train))
                
                sigmas_x =  [sigma_x * median_x for sigma_x in sigmas_x]#* sigmas_x
                sigmas_y =  [sigma_y * median_y for sigma_y in sigmas_y]#* sigmas_y

            if self.kernel_type == 'sinkhorn':
                kernel_x = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
                kernel_y = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
        
            #for X_batch, Y_batch in zip(x_train_dataloader,y_train_dataloader):
            FX = self.F_model(x_train)
            GY = self.G_model(y_train)
    
            if self.kernel_type in ['energy_guided', 'gaussian']:
                loss_x = torch.mean(kernel_x(X1=x_train, X2=x_train)) + torch.mean(kernel_x(X1=GY, X2=GY)) - 2*torch.mean(kernel_x(X1=GY, X2=x_train))
                loss_y = torch.mean(kernel_y(X1=y_train, X2=y_train)) + torch.mean(kernel_y(X1=FX, X2=FX)) - 2*torch.mean(kernel_y(X1=FX, X2=y_train))
    
            if self.kernel_type == 'sinkhorn':
                loss_x = torch.mean(kernel_x(GY, x_train))#torch.mean(kernel_x(x_train, x_train)) + torch.mean(kernel_x(GY, GY)) - 2*torch.mean(kernel_x(GY, x_train))
                loss_y = torch.mean(kernel_y(FX, y_train))#torch.mean(kernel_y(y_train, y_train)) + torch.mean(kernel_y(FX, FX)) - 2*torch.mean(kernel_y(FX, y_train))
                
            loss_function = reg_x*loss_x + reg_y*loss_y + self.eps*distortion(x_train, FX, y_train, GY)
            
            
            
            loss_function.backward()
            self.F_optimizer.step()
            self.G_optimizer.step()
                    
            if wandb_report:
                loss_metrics = {"train/loss": loss_function.item(),
                                "train/step": epoch + ix/len(sampler_source)}

                wandb.log(loss_metrics)
                     
    def train_epoch_toy(self, sampler_source, sampler_target, n_samples, n_iters, epoch, wandb_report):

        sigmas_x = self.sigmas#10
        sigmas_y = self.sigmas#10
        reg_x = self.reg#1
        reg_y = self.reg#1
        
        self.F_model.train()
        self.G_model.train()
            
        x_train = sampler_source.sample(n_samples)
        y_train = sampler_target.sample(n_samples)
        
        if self.kernel_type == 'gaussian':
                kernel_x = partial(gaussian_kernel, sigmas=sigmas_x)
                kernel_y = partial(gaussian_kernel, sigmas=sigmas_y)

        if self.kernel_type == 'energy_guided':
            kernel_x = lambda X1, X2: torch.norm(X1 - X2)
            kernel_y = lambda X1, X2: torch.norm(X1 - X2)

        if self.kernel_type == 'sinkhorn':
            kernel_x = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
            kernel_y = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
        
        #for X_batch, Y_batch in zip(x_train_dataloader,y_train_dataloader):
        FX = self.F_model(x_train)
        GY = self.G_model(y_train)

        if self.kernel_type in ['energy_guided', 'gaussian']:
            loss_x = torch.mean(kernel_x(X1=x_train, X2=x_train)) + torch.mean(kernel_x(X1=GY, X2=GY)) - 2*torch.mean(kernel_x(X1=GY, X2=x_train))
            loss_y = torch.mean(kernel_y(X1=y_train, X2=y_train)) + torch.mean(kernel_y(X1=FX, X2=FX)) - 2*torch.mean(kernel_y(X1=FX, X2=y_train))

        if self.kernel_type == 'sinkhorn':
            loss_x = torch.mean(kernel_x(GY, x_train))#torch.mean(kernel_x(x_train, x_train)) + torch.mean(kernel_x(GY, GY)) - 2*torch.mean(kernel_x(GY, x_train))
            loss_y = torch.mean(kernel_y(FX, y_train))#torch.mean(kernel_y(y_train, y_train)) + torch.mean(kernel_y(FX, FX)) - 2*torch.mean(kernel_y(FX, y_train))
            
        loss_function = reg_x*loss_x + reg_y*loss_y + self.eps*distortion(x_train, FX, y_train, GY)
        
        self.F_optimizer.zero_grad()
        self.G_optimizer.zero_grad()
        
        loss_function.backward()
        self.F_optimizer.step()
        self.G_optimizer.step()
                    
            #if ix % 1000 == 0:
            #    self.F_model.eval()
            #    self.G_model.eval()
            #    
            #    with torch.no_grad():
            #        y_sampled_F = self.F_model(X_test).cpu().numpy()#(T @ y_train_np)/T.sum(axis=1, keepdims=True)
            #        y_sampled_G = self.G_model(Y_test).cpu().numpy()
            #            
            #        fig = plt.figure(figsize=(11, 11))
            #        
            #        if TOY_TYPE == 'toy_2d_3d':
            #            ax1 = fig.add_subplot(projection='3d')
            #            ax2 = fig.add_subplot(projection='3d')
            #           
            #        if TOY_TYPE == 'toy_3d_2d':
            #            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            #            ax2 = fig.add_subplot(2, 2, 2, projection=None)
            #            ax3 = fig.add_subplot(2, 2, 3, projection=None)
            #            ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        #
            #        ax1.scatter(*x_test.cpu().T, c=labels_x_test.cpu().numpy(), cmap="Spectral", alpha=.8)
            #        ax1.set_title('Source distribution (X)', fontsize=14)
            #        
            #        ax2.scatter(*y_test.cpu().T, c=labels_y_test.cpu().numpy(), cmap="Spectral", alpha=.8)
            #        ax2.set_title('Target distribution (Y)', fontsize=14)
            #        
            #        ax3.scatter(*y_sampled_F.T, c=labels_x_test.cpu().numpy(),  cmap="Spectral", alpha=.8)
            #        ax3.set_title('F(X)')
        #
            #        ax4.scatter(*y_sampled_G.T, c=labels_y_test.cpu().numpy(),  cmap="Spectral", alpha=.8)
            #        ax4.set_title('G(Y)')
            #        
            #        plt.show()
        return None 
        
    def valid_step(self, sampler_source, sampler_target, n_samples, metric_names, source_vectors, target_vectors, n_eval):
        
        self.F_model.eval()
        self.G_model.eval()
        
        metrics_dict_F = {metric_name:[] for metric_name in metric_names}
        metrics_dict_G = {metric_name:[] for metric_name in metric_names}
        
        
        with torch.no_grad():
        
            
            for _ in trange(n_eval, leave=False, desc="Evaluation"):
                
                if sampler_target is None:
                    x, y, labels = sampler_source.sample(n_samples)
                    sampled_F_x = self.F_model(x)
                    sampled_G_y = self.G_model(y)
                    
                    metrics_dict_F = compute_metrics(x, y, sampled_F_x, labels, target_vectors, metrics_dict_F)
                    metrics_dict_G = compute_metrics(y, x, sampled_G_y, labels, source_vectors, metrics_dict_G)
                    
                else:
                    x, labels_x = sampler_source.sample(n_samples)
                    y, labels_y = sampler_target.sample(n_samples)
                    
                    sampled_F_x = self.F_model(x)
                    sampled_G_y = self.G_model(y)
                    
                    metrics_dict_F = compute_metrics(x, y, sampled_F_x, labels_x, target_vectors, metrics_dict_F)
                    metrics_dict_G = compute_metrics(y, x, sampled_G_y, labels_y, source_vectors, metrics_dict_G)
                
            return metrics_dict_F, metrics_dict_G
        
