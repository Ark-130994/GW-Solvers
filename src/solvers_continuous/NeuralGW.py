import torch
from tqdm.auto import trange
from tqdm import tqdm_notebook as tqdm
from src.metrics import compute_metrics
import wandb
from src.utils import computePotGrad

class NeuralGW:
    def __init__(self, models, optimizers, reg=None):
        self.cost_model = models['cost']
        self.critic_model = models['critic']
        self.mover_model = models['mover']
        
        self.cost_optimizer = optimizers['cost']
        self.critic_optimizer = optimizers['critic']
        self.mover_optimizer = optimizers['mover']
        
        self.reg = reg
        
    def train_epoch(self, sampler, n_samples, n_iters, epoch, wandb_report):
        
        
        self.critic_model.train()
        self.mover_model.train()
        if self.cost_optimizer is not None: 
            self.cost_model.train()
                
        cost_iters, mover_iters, critic_iters = n_iters['cost'], n_iters['mover'], n_iters['critic']
        
        train_tqdm = tqdm(range(len(sampler)), leave=False, desc="Train")        

        for ix in train_tqdm:
            
            x_train, y_train, _ = sampler.sample(n_samples)

            for i in range(cost_iters):
                
                self.cost_optimizer.zero_grad()
                h_x = self.mover_model(x_train)
                cost_loss = self.cost_model(x_train, h_x).mean()
                cost_loss.backward()
                self.cost_optimizer.step()

                for j in range(critic_iters):
                    
                    for k in range(mover_iters):

                        self.mover_optimizer.zero_grad()
                        h_x = self.mover_model(x_train)
                        mover_loss = self.cost_model(x_train, h_x).mean() - self.critic_model(h_x).mean()
                        mover_loss.backward()
                        self.mover_optimizer.step()
                        
                        
                    h_x = self.mover_model(x_train)
                    self.critic_optimizer.zero_grad()
                    
                    if self.reg is not None:
                        y_train.requires_grad_(True)
                        critic_out = self.critic_model(y_train)
                        gradients = computePotGrad(y_train, critic_out)
                        grad_penalty = (gradients.view(gradients.size(0), -1).norm(2, dim=1) ** 2).mean()
                        critic_loss = self.critic_model(h_x).mean() - self.critic_model(y_train).mean() + self.reg * grad_penalty
                    else:
                        critic_loss = self.critic_model(h_x).mean() - self.critic_model(y_train).mean()
                        
                    critic_loss.backward()
                    self.critic_optimizer.step()
                    
            if wandb_report:
                loss_metrics = {"train/cost_loss": cost_loss.item(),
                                "train/critic_loss": critic_loss.item(),
                                "train/mover_loss": mover_loss.item(),
                                "train/step": epoch + ix/len(sampler)}

                wandb.log(loss_metrics)
                    

        P = self.cost_model.matrix  

        return P
    
    def valid_step(self, sampler, n_samples, metric_names, target_vectors, n_eval, entropic_pred=True):
        
        self.critic_model.eval()
        self.mover_model.eval()
        if self.cost_optimizer is not None: 
            self.cost_model.eval()
        metrics_dict = {metric_name:[] for metric_name in metric_names}
        
        with torch.no_grad():
        
            sampler.reset_sampler()

            for _ in trange(n_eval, leave=False, desc="Evaluation"):
                x, y, labels = sampler.sample(n_samples)
                y_sampled = self.mover_model(x)
                    
                metrics_dict = compute_metrics(x, y, y_sampled, labels, target_vectors, metrics_dict)
            
            return metrics_dict
        
