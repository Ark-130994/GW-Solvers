from types import SimpleNamespace 
import torch
from src.models.simple import mlp
from src.models.costs import InnerGW_linear
from src.utils import pca_plot
from src.solvers_continuous.NeuralGW import NeuralGW
from tqdm.auto import trange
import numpy as np
import matplotlib.pyplot as plt
import wandb
from src.utils import fig2img


def report_wandb_fn(metrics_dict, metrics_names, epoch, fig):
    
    for key in metrics_dict.keys():
        for metric_name in metrics_names:
            wandb.log({f'{key}/{metric_name}':metrics_dict[key][-1][metric_name]['mean'],
                       f'{key}/step':epoch})
    if fig is not None:     
        wandb.log({'test/Plot source->target' : [wandb.Image(fig2img(fig))], 'test/step':epoch})

def train_continuous(train_sampler, test_sampler, 
                     metrics_names, target_vectors,
                     config, ckpt_name=None, 
                     pca_models=None,
                     wandb_report=False,
                     axis_lims=None, report_every=5):
    
    max_accuracy = -0.1
    var_sp = SimpleNamespace(**config['training'])
    n_eval = var_sp.N_EVAL
    DEVICE           = config['dataset']['DEVICE']
    BATCH_SIZE_TRAIN = config['dataset']['BATCH_SIZE_TRAIN']
    BATCH_SIZE_TEST  = config['dataset']['BATCH_SIZE_TEST']
    
    SOURCE_DIM       = config['dataset']['SOURCE_DIM']
    TARGET_DIM       = config['dataset']['TARGET_DIM']
    
    critic_model = mlp(TARGET_DIM, hidden_size=var_sp.HIDDEN_SIZE_MLP, num_layers=var_sp.N_LAYERS_MLP).to(DEVICE)
    mover_model  = mlp(SOURCE_DIM, TARGET_DIM, hidden_size=var_sp.HIDDEN_SIZE_MLP, num_layers=var_sp.N_LAYERS_MLP).to(DEVICE)
    cost_model   = InnerGW_linear(SOURCE_DIM, TARGET_DIM, device=DEVICE)

    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=var_sp.CRITIC_LR)
    mover_optimizer  = torch.optim.Adam(mover_model.parameters(), lr=var_sp.MOVER_LR)
    cost_optimizer   = torch.optim.Adam(cost_model.parameters(), lr=var_sp.COST_LR) 

    models     = {'cost':cost_model, 'critic':critic_model, 'mover':mover_model}
    optimizers = {'cost':cost_optimizer, 'critic':critic_optimizer, 'mover':mover_optimizer}
    n_iters    = {'cost':var_sp.COST_ITERS,'critic':var_sp.CRITIC_ITERS,'mover':var_sp.MOVER_ITERS}

    reg = var_sp.REG_CRITIC
    model_class = NeuralGW(models, optimizers, reg)
          
    cost = models['cost']
    n_samples_plot = var_sp.N_SAMPLES_PLOT

    with torch.no_grad():
        
        x_plot, y_plot, labels_plot = train_sampler.sample(n_samples_plot)
        Px_plot_init = x_plot @ cost.matrix
        
    
    report_keys = ['train'] if test_sampler is None else ['train', 'test']
    metrics_dict = {key:[] for key in report_keys}
    
    if wandb_report:
        for key in report_keys:
            wandb.define_metric(f"{key}/step")
            wandb.define_metric(f"{key}/*", step_metric=f"{key}/step")
    
    try:
        for epoch in trange(var_sp.N_EPOCHS, leave=False, desc="Epoch"):
            
            P_trained = model_class.train_epoch(train_sampler, BATCH_SIZE_TRAIN, n_iters, epoch, wandb_report=wandb_report)  
            
            if epoch % report_every == 0:
                
                metrics_train_dict = model_class.valid_step(train_sampler, BATCH_SIZE_TRAIN, metrics_names, target_vectors, n_eval)
                metrics_dict['train'].append({key1:{'mean':np.mean(metrics_train_dict[key1]), 
                                                    'std':np.std(metrics_train_dict[key1])} for key1 in metrics_names})
                
                if test_sampler is not None:
                    metrics_test_dict = model_class.valid_step(test_sampler, BATCH_SIZE_TEST, metrics_names, target_vectors, n_eval)
                    metrics_dict['test'].append({key1:{'mean':np.mean(metrics_test_dict[key1]), 
                                                       'std':np.std(metrics_test_dict[key1])} for key1 in metrics_names})
                    
                if pca_models is not None:
                    fig, axis_lims = pca_plot(x_plot, y_plot, labels_plot, model_class.mover_model, P_trained, pca_models,                                                               axis_lims=None, figsize=(20, 8))
                    
                if wandb_report:
                    report_wandb_fn(metrics_dict, metrics_names, epoch, None)
                    
            plt.close()
            
        metrics_dict_out = {'train':metrics_dict['train'][-1], 'test':metrics_dict['test'][-1]}    
        
    except KeyboardInterrupt:
        print('Interrumpting by keyboard...')
        return model_class, metrics_dict_out

        if wandb_report:
            wandb.finish()
    
    return model_class, metrics_dict_out