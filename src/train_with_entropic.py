from types import SimpleNamespace 
import torch
from src.models.simple import mlp, mlp_jax
from src.models.costs import InnerGW_linear, CostModel
from src.utils import pca_plot
from src.solvers_continuous.NeuralGW import NeuralGW
from src.solvers_continuous.RegGW_mb import RegGW
from ott.neural.networks.velocity_field import VelocityField

from src.solvers_continuous.NeuralGW_entropic import NeuralGW_entropic

from tqdm.auto import trange
import numpy as np
import matplotlib.pyplot as plt
import wandb
from src.utils import fig2img

from src.solvers_discrete.AlignGW import AlignGW
from src.solvers_discrete.StructuredGW import StructuredGW
from src.solvers_discrete.FlowGW import FlowGW

from src.solvers_discrete.AlignGW_mb import AlignGW_mb
from src.solvers_discrete.StructuredGW_mb import StructuredGW_mb
from src.solvers_discrete.FlowGW_mb import FlowGW_mb

from src.solvers_discrete.DiscreteSolver import DiscreteSolver
from sklearn.neural_network import MLPRegressor

from ott.neural.networks.velocity_field import VelocityField
from ott.neural.networks.layers import time_encoder
from ott.solvers import utils as solver_utils
import optax
import jax
import functools
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type, Union
import jax
import jax.numpy as jnp
from ott.neural.networks.layers import time_encoder


from ott.geometry import costs

import flax.linen as ln

def _multivariate_normal(rng: jax.Array, shape: Tuple[int, ...], dim: int, mean: float = 0.0, cov: float = 1.0) -> jnp.ndarray:
    mean = jnp.full(dim, fill_value=mean)
    cov = jnp.diag(jnp.full(dim, fill_value=cov))
    return jax.random.multivariate_normal(rng, mean=mean, cov=cov, shape=shape)

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
                     wandb_report=False,
                     axis_lims=None, report_every=5):
    
    
    var_sp = SimpleNamespace(**config['training'])
    n_eval           = config['dataset']['N_EVAL']
    DEVICE           = config['dataset']['DEVICE']
    BATCH_SIZE_TRAIN = config['dataset']['BATCH_SIZE_TRAIN']
    BATCH_SIZE_TEST  = config['dataset']['BATCH_SIZE_TEST']
    METHOD_NAME      = config['training']['METHOD_NAME']
    
    SOURCE_DIM       = config['dataset']['SOURCE_DIM']
    TARGET_DIM       = config['dataset']['TARGET_DIM']

    SEED             = config['dataset']['SEED']

    if METHOD_NAME == 'NeuralGW':
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

    if METHOD_NAME == 'RegGW':
        rng = jax.random.PRNGKey(SEED)
        mover_model = mlp_jax(hidden_dims=var_sp.HIDDEN_SIZES_MLP, out_dim=TARGET_DIM, act_fn=ln.relu)
        mover_optimizer = optax.adam(learning_rate=var_sp.MOVER_LR)
        n_iters = None
        cost_fn = costs.Cosine()
        model_class = RegGW(mover_model, mover_optimizer, SOURCE_DIM, cost_fn, var_sp.EPS_FIT, var_sp.EPS_REG, var_sp.LAMBDA)

    if METHOD_NAME == 'FlowGW':
        embed_dim = var_sp.HIDDEN_SIZES_MLP[0]
        n_layers = len(var_sp.HIDDEN_SIZES_MLP)
        
        mover_model = VelocityField(hidden_dims=[embed_dim]*n_layers,
                                    time_dims=[embed_dim, embed_dim],
                                    output_dims=[embed_dim, embed_dim, embed_dim] + [TARGET_DIM],
                                    condition_dims=[embed_dim, embed_dim, embed_dim],
                                    time_encoder=functools.partial(time_encoder.cyclical_time_encoder, n_freqs=self.n_freq),
            )
        n_iters = None
        cost_fn = costs.Cosine()
        eps = var_sp.EPS
        model_class = FlowGW_mb(mover_model, SOURCE_DIM, TARGET_DIM, eps, cost_fn, seed=SEED)
        
    report_keys = ['train'] if test_sampler is None else ['train', 'test']
    metrics_dict = {key:[] for key in report_keys}
    
    if wandb_report:
        for key in report_keys:
            wandb.define_metric(f"{key}/step")
            wandb.define_metric(f"{key}/*", step_metric=f"{key}/step")
    
    try:
        for epoch in trange(var_sp.N_EPOCHS, leave=False, desc="Epoch"):
            
            model_class.train_epoch(train_sampler, BATCH_SIZE_TRAIN, n_iters, epoch, wandb_report=wandb_report)  
            
            if epoch % report_every == 0:
                
                metrics_train_dict = model_class.valid_step(train_sampler, BATCH_SIZE_TRAIN, metrics_names, target_vectors, n_eval)
                metrics_dict['train'].append({key1:{'mean':np.mean(metrics_train_dict[key1]), 
                                                    'std':np.std(metrics_train_dict[key1])} for key1 in metrics_names})
                
                if test_sampler is not None:
                    metrics_test_dict = model_class.valid_step(test_sampler, BATCH_SIZE_TEST, metrics_names, target_vectors, n_eval)
                    metrics_dict['test'].append({key1:{'mean':np.mean(metrics_test_dict[key1]), 
                                                       'std':np.std(metrics_test_dict[key1])} for key1 in metrics_names})
                    
                    
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

def train_continuous_old(train_sampler, test_sampler, 
                         metrics_names, target_vectors,
                         config, ckpt_name=None, 
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

def train_continuous2(train_sampler, test_sampler, 
                     metrics_names, target_vectors,
                     config, ckpt_name=None, 
                     pca_models=None,
                     wandb_report=False,
                     axis_lims=None, report_every=5):
    
    var_sp = SimpleNamespace(**config['training'])
    DEVICE           = config['dataset']['DEVICE']
    BATCH_SIZE_TRAIN = config['dataset']['BATCH_SIZE_TRAIN']
    
    SOURCE_DIM       = config['dataset']['SOURCE_DIM']
    TARGET_DIM       = config['dataset']['TARGET_DIM']

    critic_iters  = var_sp.CRITIC_ITERS
    cost_iters    = var_sp.COST_ITERS
    
    critic_model   = mlp(SOURCE_DIM, hidden_size=var_sp.HIDDEN_SIZE_MLP, num_layers=var_sp.N_LAYERS_MLP).to(DEVICE)
    cost_model     = CostModel(SOURCE_DIM, TARGET_DIM).to(DEVICE)

    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=var_sp.CRITIC_LR)
    cost_optimizer   = torch.optim.Adam(critic_model.parameters(), lr=var_sp.COST_LR)
    
    discrete_model = MLPRegressor(hidden_layer_sizes=256, random_state=1, max_iter=500)    

    model_class = NeuralGW_entropic(critic_model, cost_model, discrete_model, critic_optimizer, cost_optimizer, epsilon=var_sp.EPSILON)
                  
    
    report_keys = ['train'] if test_sampler is None else ['train', 'test']
    metrics_dict = {key:[] for key in report_keys}
    
    if wandb_report:
        for key in report_keys:
            wandb.define_metric(f"{key}/step")
            wandb.define_metric(f"{key}/*", step_metric=f"{key}/step")
    
    try:
        for epoch in trange(var_sp.N_EPOCHS, leave=False, desc="Epoch"):
            
            A, plan = model_class.train_epoch_glove(train_sampler, BATCH_SIZE_TRAIN, critic_iters, cost_iters, epoch, wandb_report=wandb_report)  
            
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


def train_discrete(train_sampler, test_sampler, 
                   metrics_names, target_vectors,
                   config,
                   pca_models=None,
                   wandb_report=False,
                   axis_lims=None, report_every=10):
    
    
    var_sp            = SimpleNamespace(**config['training'])
    N_EVAL            = config['dataset']['N_EVAL']
    DEVICE            = config['dataset']['DEVICE']
    FUSED_DIM         = config['dataset']['FUSED_DIM']
    METHOD_NAME       = var_sp.METHOD_NAME

    if FUSED_DIM > 0 and METHOD_NAME != 'FlowGW':
        return ValueError(f'Fused is not implemented for {METHOD_NAME}.')

    MAX_SAMPLES_TRAIN = train_sampler.loader.batch_size

    if test_sampler is not None:
        MAX_SAMPLES_TEST = test_sampler.loader.batch_size
    
    if METHOD_NAME == 'AlignGW':
        metric_name = var_sp.COST_DISCRETE
        normalize_dists = 'mean'
        eps = var_sp.EPSILON
        tol = 1e-8
        method_class = AlignGW(metric=metric_name, normalize_dists=normalize_dists,
                               loss_fun='square_loss', eps=eps, tol=tol)
        
    if METHOD_NAME == 'StructuredGW':
        M_init = None
        method_M = 'exact'
        eps = var_sp.EPSILON
        tol = 1e-3
        method_class = StructuredGW(M_init, method_M, eps=eps, tol=tol)

    if METHOD_NAME == 'FlowGW':
        eps = var_sp.EPSILON
        embed_dim = 1024
        n_freq = 128
        n_layers = 4
        
        if var_sp.COST_DISCRETE == 'cosine':
            cost_fn = costs.Cosine()
        if var_sp.COST_DISCRETE == 'euclidean':
            cost_fn = costs.SqEuclidean()   
        
        method_class = FlowGW(eps=eps, embed_dim=embed_dim, n_freq=n_freq, n_layers=n_layers, cost_fn=cost_fn)
        
    report_keys = ['train'] if test_sampler is None else ['train', 'test']
    metrics_dict = {key:[] for key in report_keys}

    x_dict, y_dict, labels_dict = {}, {}, {}
    
    try:
            
        x, y, labels = train_sampler.sample(MAX_SAMPLES_TRAIN)
        x_dict['train'], y_dict['train'], labels_dict['train'] = x.cpu(), y.cpu(), labels.cpu()
        
        if test_sampler is not None:
            x, y, labels = test_sampler.sample(MAX_SAMPLES_TEST)
            x_dict['test'], y_dict['test'], labels_dict['test'] = x.cpu(), y.cpu(), labels.cpu()
        else:
            x_dict['test'], y_dict['test'], labels_dict['test'] = None, None, None
        
        if METHOD_NAME == 'FlowGW':   
            method_class.fit(x_dict, y_dict, labels_dict, target_vectors, FUSED_DIM, wandb_report, var_sp.MAX_ITERS, report_every=report_every) 

        else:
            method_class.fit(x_dict, y_dict, labels_dict, target_vectors, wandb_report, var_sp.MAX_ITERS, report_every=report_every) 

        metrics_train_dict = method_class.valid_step(train_sampler, MAX_SAMPLES_TRAIN, metrics_names, target_vectors, N_EVAL)
        metrics_dict['train'].append({key1:{'mean':np.mean(metrics_train_dict[key1]), 
                                            'std':np.std(metrics_train_dict[key1])} for key1 in metrics_names})

        if test_sampler is not None:
            metrics_test_dict = method_class.valid_step(test_sampler, MAX_SAMPLES_TEST, metrics_names, target_vectors, N_EVAL)
            metrics_dict['test'].append({key1:{'mean':np.mean(metrics_test_dict[key1]), 
                                               'std':np.std(metrics_test_dict[key1])} for key1 in metrics_names})

            
    except KeyboardInterrupt:
        print('Interrumpting by keyboard...')
        if wandb_report:
            wandb.finish()
            
        return method_class, metrics_dict

    return method_class, metrics_dict


def train_discrete_mb(train_sampler, test_sampler, 
                      metrics_names, target_vectors,
                      config,
                      pca_models=None,
                      wandb_report=False,
                      axis_lims=None, report_every=10):
    
    
    var_sp = SimpleNamespace(**config['training'])
    N_EVAL = var_sp.N_EVAL
    DEVICE                    = config['dataset']['DEVICE']
    SEED                      = config['dataset']['SEED']
    
    SOURCE_DIM                = config['dataset']['SOURCE_DIM']
    TARGET_DIM                = config['dataset']['TARGET_DIM']
    CONT_SOLVER_TYPE          = var_sp.CONT_SOLVER_TYPE
    
    MAX_SAMPLES_TRAIN         = train_sampler.loader.batch_size#config['dataset']['BATCH_SIZE_TRAIN']
    
    if CONT_SOLVER_TYPE == 'mlp':
        continuous_solver = mlp(SOURCE_DIM, TARGET_DIM, hidden_size=256, num_layers=4).to(DEVICE)
        optimizer         = torch.optim.Adam(continuous_solver.parameters(), lr=1e-3)
        loss_fn           = torch.nn.MSELoss()
        
    if CONT_SOLVER_TYPE == 'norm_flow':
        rng = jax.random.PRNGKey(SEED)
        neural_net = VelocityField(hidden_dims=[1024]*4,
                                          time_dims=[1024, 1024],
                                          output_dims=[1024, 1024, 1024] + [TARGET_DIM],
                                          condition_dims=[1024, 1024, 1024],
                                          time_encoder=functools.partial(time_encoder.cyclical_time_encoder, n_freqs=128),
                                          )
        optimizer = optax.adamw(learning_rate=1e-4, weight_decay=1e-10)
        state_neural_net = neural_net.create_train_state(rng, optimizer, TARGET_DIM, SOURCE_DIM)
        time_sampler = solver_utils.uniform_sampler
        latent_fn = functools.partial(_multivariate_normal, dim=TARGET_DIM)

        loss_fn = None
        continuous_solver = {'rng':rng, 'time_sampler':time_sampler, 'state_neural_net':state_neural_net, 'latent_fn':latent_fn}
    
    if test_sampler is not None:
        MAX_SAMPLES_TEST          = test_sampler.loader.batch_size#config['dataset']['BATCH_SIZE_TRAIN']
    
    #solver = DiscreteSolver(var_sp.METHOD_NAME)
   
    if var_sp.METHOD_NAME == 'AlignGW':
        metric_name = var_sp.COST_DISCRETE
        normalize_dists = 'mean'
        eps = var_sp.EPSILON
        tol = 1e-8
        method_class = AlignGW_mb(CONT_SOLVER_TYPE, continuous_solver, optimizer, loss_fn, metric=metric_name, normalize_dists=normalize_dists, loss_fun='square_loss', eps=eps, tol=tol)

        
    if var_sp.METHOD_NAME == 'StructuredGW':
        M_init = None
        method_M = 'exact'
        eps = var_sp.EPSILON
        tol = 1e-3
        method_class = StructuredGW_mb(continuous_solver, M_init, method_M, eps=eps, tol=tol)
        
    report_keys = ['train'] if test_sampler is None else ['train', 'test']
    metrics_dict = {key:[] for key in report_keys}

    x_dict, y_dict, labels_dict = {}, {}, {}
    
    try:

        method_class.fit(train_sampler, test_sampler, target_vectors, DEVICE, wandb_report, var_sp.MAX_ITERS, report_every=report_every) 

        metrics_train_dict = method_class.valid_step(train_sampler, MAX_SAMPLES_TRAIN, metrics_names, target_vectors, N_EVAL)
        metrics_dict['train'].append({key1:{'mean':np.mean(metrics_train_dict[key1]), 
                                            'std':np.std(metrics_train_dict[key1])} for key1 in metrics_names})
#
        if test_sampler is not None:
            metrics_test_dict = method_class.valid_step(test_sampler, MAX_SAMPLES_TEST, metrics_names, target_vectors, N_EVAL)
            metrics_dict['test'].append({key1:{'mean':np.mean(metrics_test_dict[key1]), 
                                                   'std':np.std(metrics_test_dict[key1])} for key1 in metrics_names})

        plt.close()
            
    except KeyboardInterrupt:
        print('Interrumpting by keyboard...')
        if wandb_report:
            wandb.finish()
            
        return method_class, metrics_dict

    return method_class, metrics_dict


def train_toy(train_sampler, 
              config,
              axis_lims=None, report_every=10):
    
    
    var_sp = SimpleNamespace(**config['training'])
    DEVICE = config['dataset']['DEVICE']
    N_SAMPLES = config['dataset']['N_SAMPLES']
    toy_type = config['dataset']['DATASET_NAME']
        
    if var_sp.METHOD_NAME == 'AlignGW':
        metric_name = var_sp.COST_DISCRETE
        normalize_dists = 'mean'
        eps = var_sp.EPSILON
        tol = 1e-8
        method_class = AlignGW(metric=metric_name, normalize_dists=normalize_dists,
                                      loss_fun='square_loss', eps=eps, tol=tol, toy_type=toy_type)

    if var_sp.METHOD_NAME == 'StructuredGW':
        M_init = None
        method_M = 'exact'
        eps = var_sp.EPSILON
        tol = 1e-3
        method_class = StructuredGW(M_init, method_M, eps=eps, tol=tol, toy_type=toy_type)

    if var_sp.METHOD_NAME == 'FlowGW':
        eps = 1e-4
        embed_dim = 1024
        n_freq = 128
        n_layers = 4
        cost_fn = costs.Cosine()
        toy_type = toy_type
        method_class = FlowGW(eps=eps, embed_dim=embed_dim, n_freq=n_freq, n_layers=n_layers, cost_fn=cost_fn, toy_type=toy_type)
        

    x_dict, y_dict, labels_dict = {}, {}, {}
    
    try:
 
        x, y, labels = train_sampler.sample(N_SAMPLES)
        x_dict['train'], y_dict['train'], labels_dict['train'] = x.cpu(), y.cpu(), labels.cpu()
        
        x_dict['test'], y_dict['test'], labels_dict['test'] = None, None, None


        method_class.fit(x_dict, y_dict, labels_dict, None, False, var_sp.MAX_ITERS, report_every=report_every) 

        plt.close()
            
    except KeyboardInterrupt:
        print('Interrumpting by keyboard...')
            
        return method_class

    return method_class

def train_toy_continuous(train_sampler, 
                         config,
                         axis_lims=None, report_every=10):
    
    
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
        
    
    try:
        for epoch in trange(var_sp.N_EPOCHS, leave=False, desc="Epoch"):
            
            P_trained = model_class.train_epoch(train_sampler, BATCH_SIZE_TRAIN, n_iters, epoch, wandb_report=False)  
            
            if epoch % report_every == 0:
                mover_model_pred = model_class.mover_model
                mover_model_pred.eval()
                
                with torch.no_grad():
                    y_sampled_np = mover_model(x_plot).detach().cpu().numpy()
                    fig = plt.figure(figsize=(8, 8))
                    
                    if toy_type == 'toy_2d_3d':
                        ax = fig.add_subplot(projection='3d')
                       
                    if toy_type == 'toy_3d_2d':
                        ax = fig.add_subplot(projection=None)

                    ax.scatter(*y_sampled_np.T, c=labels_plot.cpu().numpy(),  cmap="Spectral")
                    plt.show()
                    
            
    except KeyboardInterrupt:
        print('Interrumpting by keyboard...')
        return model_class

        if wandb_report:
            wandb.finish()
    
    return model_class

def train_toy_continuous2(source_sampler, target_sampler,
                          config,
                          axis_lims=None, report_every=10):

    var_sp = SimpleNamespace(**config['training'])
    DEVICE           = config['dataset']['DEVICE']
    BATCH_SIZE_TRAIN = config['dataset']['BATCH_SIZE_TRAIN']
    TOY_TYPE         = config['dataset']['TOY_TYPE']
    
    SOURCE_DIM       = config['dataset']['SOURCE_DIM']
    TARGET_DIM       = config['dataset']['TARGET_DIM']
    
    critic_model   = mlp(TARGET_DIM, hidden_size=var_sp.HIDDEN_SIZE_MLP, num_layers=var_sp.N_LAYERS_MLP).to(DEVICE)
    cost_model     = CostModel(SOURCE_DIM, TARGET_DIM).to(DEVICE)

    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=var_sp.CRITIC_LR)
    cost_optimizer   = torch.optim.Adam(critic_model.parameters(), lr=var_sp.COST_LR)
    
    discrete_model = MLPRegressor(hidden_layer_sizes=256, random_state=1, max_iter=500)    

    model_class = NeuralGW_entropic(critic_model, cost_model, discrete_model, critic_optimizer, cost_optimizer, epsilon=var_sp.EPSILON)
          
    n_samples_plot = var_sp.N_SAMPLES_PLOT

    with torch.no_grad():
        
        x_plot, labels_plot = source_sampler.sample_with_labels((n_samples_plot,))
        y_plot = target_sampler.sample((n_samples_plot,))
    
    try:
        for epoch in trange(var_sp.N_EPOCHS, leave=False, desc="Epoch"):
           
            A, plan = model_class.train_epoch(source_sampler, target_sampler, BATCH_SIZE_TRAIN, var_sp.CRITIC_ITERS, var_sp.COST_ITERS, epoch, wandb_report=None)  
            
            if epoch % report_every == 0 and epoch > 50:
                
                with torch.no_grad():
                    y_sampled_np = model_class.discrete_model.predict(x_plot.detach().cpu().numpy())
                    figure = plt.figure(figsize=(10, 10))

                    if TOY_TYPE == 'toy_2d_2d':
                        
                        ax1 = figure.add_subplot(2, 2, 1, projection=None)
                        ax2 = figure.add_subplot(2, 2, 2, projection=None)
                        ax3 = figure.add_subplot(2, 2, 3, projection=None)
                        ax4 = figure.add_subplot(2, 2, 4, projection=None)
                        
                        
                    if TOY_TYPE == 'toy_2d_3d':
                        ax1 = figure.add_subplot(1, 3, 1, projection=None)
                        ax2 = figure.add_subplot(1, 3, 2, projection='3d')
                        ax3 = figure.add_subplot(1, 3, 3, projection='3d')
                        
                       
                    if TOY_TYPE == 'toy_3d_2d':
                        ax1 = figure.add_subplot(1, 3, 1, projection='3d')
                        ax2 = figure.add_subplot(1, 3, 2, projection=None)
                        ax3 = figure.add_subplot(1, 3, 3, projection=None)

                    ax1.scatter(*x_plot.cpu().numpy().T, c=labels_plot.cpu().numpy(),  cmap="Spectral")
                    ax2.scatter(*y_plot.cpu().numpy().T, c='black',  cmap="Spectral")
                    ax3.scatter(*(x_plot@A).cpu().T, c=labels_plot.cpu().numpy(),  cmap="Spectral")
                    ax4.scatter(*y_sampled_np.T, c=labels_plot.cpu().numpy(),  cmap="Spectral")
                        
                    plt.show()

                    wandb.log({'Plot source->target' : [wandb.Image(fig2img(figure))]}, step=epoch)
            
    except KeyboardInterrupt:
        print('Interrumpting by keyboard...')
        return model_class

        if wandb_report:
            wandb.finish()
    
    return model_class