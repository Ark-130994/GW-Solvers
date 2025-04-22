from types import SimpleNamespace 
import torch
from src.models.simple import mlp, mlp_jax, fcnn
from src.models.costs import InnerGW_linear, CostModel, SquaredGW_cost, EmbeddedGW_linear, CayleyCost
from src.models.LightSB import LightSB

from src.utils import pca_plot
from src.solvers_continuous.NeuralGW import NeuralGW
from src.solvers_continuous.SquareGW import SquareGW

from src.solvers_continuous.SimpleGW import SimpleGW
from src.solvers_continuous.LightGW import LightGW

#from src.solvers_continuous.EntropicGW import EntropicGW

from src.solvers_continuous.CycleGW import CycleGW

from src.solvers_continuous.RegGW_mb import RegGW_mb
from src.solvers_continuous.FlowGW_mb import FlowGW_mb

from src.solvers_discrete.RegGW import RegGW
from src.solvers_discrete.FlowGW import FlowGW


from ott.neural.networks.velocity_field import VelocityField
from src.metrics import compute_distortion
from tqdm.auto import trange
import numpy as np
import matplotlib.pyplot as plt
import wandb
from src.utils import fig2img

from src.solvers_discrete.AlignGW import AlignGW
from src.solvers_discrete.StructuredGW import StructuredGW
from src.solvers_discrete.FlowGW import FlowGW

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
import torch.nn as nn
import torch.nn.functional as Fn
import flax.linen as ln

def report_wandb_fn(metrics_dict, metrics_names, epoch, fig):
    
    for key in metrics_dict.keys():
        for metric_name in metrics_names:
            wandb.log({f'{key}/{metric_name}':metrics_dict[key][-1][metric_name]['mean'],
                       f'{key}/step':epoch})
    if fig is not None:     
        wandb.log({'test/Plot source->target' : [wandb.Image(fig2img(fig))], 'test/step':epoch})
        
class generator_x_y(nn.Module):
    def __init__(self, dimension_x, hidden_layer, dimension_y):
        super(generator_x_y, self).__init__()
        
        self.lin1 = nn.Linear(dimension_x, hidden_layer)
        self.lin2 = nn.Linear(hidden_layer, hidden_layer)
        self.lin3 = nn.Linear(hidden_layer, hidden_layer)

        self.lin_end = nn.Linear(hidden_layer, dimension_y)
        
    def forward(self, input):
        y = Fn.leaky_relu(self.lin1(input))
        y = Fn.leaky_relu(self.lin2(y))
        y = Fn.leaky_relu(self.lin3(y))
        y = self.lin_end(y)
        
        return y

class generator_y_x(nn.Module):
    def __init__(self, dimension_x, hidden_layer, dimension_y):
        super(generator_y_x, self).__init__()
               
        self.lin1 = nn.Linear(dimension_y, hidden_layer)
        self.lin2 = nn.Linear(hidden_layer, hidden_layer)
        self.lin3 = nn.Linear(hidden_layer, hidden_layer)

        self.lin_end = nn.Linear(hidden_layer, dimension_x)
        
    def forward(self, input):
        x = Fn.leaky_relu(self.lin1(input))
        x = Fn.leaky_relu(self.lin2(x))
        x = Fn.leaky_relu(self.lin3(x))
        x = self.lin_end(x)
        
        return x
def train_continuous(train_source_sampler, train_target_sampler, 
                     test_sampler, 
                     metrics_names, target_vectors,
                     config,
                     wandb_report=False,
                     axis_lims=None, report_every=10, source_vectors=None):
    
    space_dataset     = SimpleNamespace(**config['dataset'])
    space_training    = SimpleNamespace(**config['training'])
    space_model       = SimpleNamespace(**config['model_specific'])
    
    N_EVAL           = space_dataset.N_EVAL
    DEVICE           = space_dataset.DEVICE
    BATCH_SIZE_TRAIN = space_dataset.BATCH_SIZE_TRAIN
    BATCH_SIZE_TEST  = space_dataset.BATCH_SIZE_TEST
    METHOD_NAME      = space_training.METHOD_NAME
    
    SOURCE_DIM       = space_dataset.SOURCE_DIM
    TARGET_DIM       = space_dataset.TARGET_DIM

    SEED             = space_dataset.SEED

    if METHOD_NAME in ['NeuralGW', 'SquareGW', 'EmbeddedGW', 'CayleyGW']:
        
        if METHOD_NAME == 'NeuralGW':
            cost_model   = InnerGW_linear(SOURCE_DIM, TARGET_DIM, device=DEVICE)
        if METHOD_NAME == 'CayleyGW':
            NEG_EIGVALS = space_model.NEG_EIGVALS
            cost_model   = CayleyCost(SOURCE_DIM, TARGET_DIM, NEG_EIGVALS, device=DEVICE)
        if METHOD_NAME == 'SquareGW':
            cost_model   = SquaredGW_cost(SOURCE_DIM, TARGET_DIM, device=DEVICE)
        if METHOD_NAME == 'EmbeddedGW':
            cost_model   = EmbeddedGW_linear(SOURCE_DIM, TARGET_DIM, device=DEVICE)
            
        critic_model = mlp(TARGET_DIM, hidden_sizes=space_model.HIDDEN_SIZES_MLP).to(DEVICE)
        mover_model  = mlp(SOURCE_DIM, TARGET_DIM, hidden_sizes=space_model.HIDDEN_SIZES_MLP).to(DEVICE)
    
        critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=space_model.CRITIC_LR)
        mover_optimizer  = torch.optim.Adam(mover_model.parameters(), lr=space_model.MOVER_LR)
        cost_optimizer   = torch.optim.Adam(cost_model.parameters(), lr=space_model.COST_LR) 
    
        models     = {'cost':cost_model, 'critic':critic_model, 'mover':mover_model}
        optimizers = {'cost':cost_optimizer, 'critic':critic_optimizer, 'mover':mover_optimizer}
        n_iters    = {'cost':space_model.COST_ITERS,'critic':space_model.CRITIC_ITERS,'mover':space_model.MOVER_ITERS}
    
        reg         = space_model.REG_CRITIC
        
        if METHOD_NAME in ['NeuralGW', 'CayleyGW']:
            model_class = NeuralGW(models, optimizers, reg)
        if METHOD_NAME == 'SquareGW':
            model_class = SquareGW(models, optimizers, reg)

    if METHOD_NAME == 'SimpleGW':
        if space_model.USE_P is None:
            P_matrix = None
            mover_model  = mlp(SOURCE_DIM, TARGET_DIM, hidden_sizes=space_model.HIDDEN_SIZES_MLP).to(DEVICE)
        else:
            P_matrix = utils.generate_random_matrix(n_rows, n_cols, matrix_type=space_model.USE_P)
            mover_model  = mlp(TARGET_DIM, TARGET_DIM, hidden_sizes=space_model.HIDDEN_SIZES_MLP).to(DEVICE)
            
        mover_optimizer  = torch.optim.Adam(mover_model.parameters(), lr=space_model.MOVER_LR)

        n_iters    = space_model.MOVER_ITERS

        model_class = SimpleGW(mover_model, mover_optimizer, P_matrix)
        
    if METHOD_NAME == 'RegGW':
        rng             = jax.random.PRNGKey(SEED)
        mover_model     = mlp_jax(hidden_dims=space_model.HIDDEN_SIZES_MLP, out_dim=TARGET_DIM, act_fn=ln.relu)
        mover_optimizer = optax.adam(learning_rate=space_model.MOVER_LR)
        n_iters         = None
        
        if space_training.COST_DISCRETE == 'cosine':
            cost_fn = costs.Cosine()
        if space_training.COST_DISCRETE == 'euclidean':
            cost_fn = costs.SqEuclidean()
            
        model_class = RegGW_mb(mover_model, mover_optimizer, SOURCE_DIM, cost_fn, space_model.EPS_FIT, space_model.EPS_REG, space_model.LAMBDA)

    if METHOD_NAME == 'FlowGW':
        embed_dim   = space_model.HIDDEN_SIZES_MLP[0]
        n_layers    = len(space_model.HIDDEN_SIZES_MLP)
        
        mover_model = VelocityField(hidden_dims=[embed_dim]*n_layers,
                                    time_dims=[embed_dim, embed_dim],
                                    output_dims=[embed_dim, embed_dim, embed_dim] + [TARGET_DIM],
                                    condition_dims=[embed_dim, embed_dim, embed_dim],
                                    time_encoder=functools.partial(time_encoder.cyclical_time_encoder, n_freqs=space_model.N_FREQ),
                                    )
        
        n_iters     = None
        cost_fn     = costs.Cosine()
        eps         = space_model.EPS
        model_class = FlowGW_mb(mover_model, SOURCE_DIM, TARGET_DIM, eps, cost_fn, seed=SEED)

    if METHOD_NAME == 'CycleGW':
        F_model = fcnn(SOURCE_DIM, TARGET_DIM, hidden_sizes=space_model.HIDDEN_SIZES_MLP).to(DEVICE)#fcnn(SOURCE_DIM, hidden_dim=space_model.HIDDEN_SIZES_MLP).to(DEVICE)
        G_model = fcnn(TARGET_DIM, SOURCE_DIM, hidden_sizes=space_model.HIDDEN_SIZES_MLP).to(DEVICE)

        #F_model =  generator_x_y(SOURCE_DIM, space_model.HIDDEN_SIZES_MLP[0], TARGET_DIM).to(torch.float32).to(DEVICE)
        #G_model =  generator_y_x(SOURCE_DIM, space_model.HIDDEN_SIZES_MLP[0], TARGET_DIM).to(torch.float32).to(DEVICE)
        F_optimizer = torch.optim.Adam(F_model.parameters(), lr=space_model.F_LR)
        G_optimizer = torch.optim.Adam(G_model.parameters(), lr=space_model.G_LR)
    
        models     = {'F_model':F_model, 'G_model':G_model}
        optimizers = {'F_model':F_optimizer, 'G_model':G_optimizer}
        n_iters     = None
    
        model_class = CycleGW(models, optimizers, space_model.EPS, space_model.SIGMAS, space_model.REG, space_model.KERNEL_TYPE, space_model.TAKE_MEDIAN)
        
    if METHOD_NAME == 'LightGW':
        critic_model = LightSB(dim=TARGET_DIM, n_potentials=space_model.N_POTENTIALS, epsilon=space_model.EPSILON,
                       sampling_batch_size=BATCH_SIZE_TRAIN, S_diagonal_init=0.1, 
                       is_diagonal=True).to(DEVICE)
        cost_model   = InnerGW_linear(SOURCE_DIM, TARGET_DIM, device=DEVICE)
    
        critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=space_model.CRITIC_LR)
        cost_optimizer   = torch.optim.Adam(cost_model.parameters(), lr=space_model.COST_LR) 
    
        models     = {'cost':cost_model, 'critic':critic_model}
        optimizers = {'cost':cost_optimizer, 'critic':critic_optimizer}
        n_iters    = {'cost':space_model.COST_ITERS,'critic':space_model.CRITIC_ITERS}
    
        model_class = LightGW(models, optimizers)
        
    if METHOD_NAME == 'EntropicGW':
        critic_model = mlp(SOURCE_DIM, hidden_sizes=space_model.HIDDEN_SIZES_MLP).to(DEVICE)
        critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=space_model.CRITIC_LR)

        continuous_model = MLPRegressor(hidden_layer_sizes=256, random_state=1, max_iter=500)
        
        if space_model.COST_LR is not None:
            cost_model   = InnerGW_linear(SOURCE_DIM, TARGET_DIM, device=DEVICE)
            cost_optimizer   = torch.optim.Adam(cost_model.parameters(), lr=space_model.COST_LR)

        else:
            cost_model = None
            cost_optimizer = None
        
        models     = {'critic':critic_model, 'cost':cost_model, 'continuous':continuous_model}
        optimizers = {'critic':critic_optimizer, 'cost':cost_optimizer}
        n_iters    = {'critic':space_model.CRITIC_ITERS, 'cost':space_model.COST_ITERS}
    
        model_class = EntropicGW(models, optimizers, space_model.EPS)

    if METHOD_NAME != 'CycleGW':
        report_keys = ['train', 'test'] 
        metrics_dict = {key:[] for key in report_keys}
        
    else:
        report_keys = ['train_F', 'train_G', 'test_F', 'test_G'] 
        metrics_dict = {key:[] for key in report_keys}
    
    if wandb_report:
        for key in report_keys:
            wandb.define_metric(f"{key}/step")
            wandb.define_metric(f"{key}/*", step_metric=f"{key}/step")
    
    try:
        for epoch in trange(space_training.N_EPOCHS, leave=False, desc="Epoch"):

            model_class.train_epoch(train_source_sampler, train_target_sampler, BATCH_SIZE_TRAIN, n_iters, epoch, wandb_report)  
            
            if (epoch % report_every == 0 and epoch != 0) or epoch==space_training.N_EPOCHS-1:
                if METHOD_NAME != 'CycleGW':
                    metrics_train_dict = model_class.valid_step(train_source_sampler, train_target_sampler, BATCH_SIZE_TRAIN, metrics_names, target_vectors, N_EVAL)
                    metrics_dict['train'].append({key1:{'mean':np.mean(metrics_train_dict[key1]), 
                                                        'std':np.std(metrics_train_dict[key1])} for key1 in metrics_names})
                    
                    metrics_test_dict = model_class.valid_step(test_sampler, None, BATCH_SIZE_TEST, metrics_names, target_vectors, N_EVAL)
                    metrics_dict['test'].append({key1:{'mean':np.mean(metrics_test_dict[key1]), 
                                                       'std':np.std(metrics_test_dict[key1])} for key1 in metrics_names})

                    metrics_dict_out = {'train':metrics_dict['train'][-1], 'test':metrics_dict['test'][-1]}
                    
                    
                else:
                    metrics_train_dict_F,  metrics_train_dict_G = model_class.valid_step(train_source_sampler, train_target_sampler, BATCH_SIZE_TRAIN, metrics_names, source_vectors, target_vectors, N_EVAL)
                    metrics_dict['train_F'].append({key1:{'mean':np.mean(metrics_train_dict_F[key1]), 
                                                          'std':np.std(metrics_train_dict_F[key1])} for key1 in metrics_names})
                    metrics_dict['train_G'].append({key1:{'mean':np.mean(metrics_train_dict_G[key1]), 
                                                          'std':np.std(metrics_train_dict_G[key1])} for key1 in metrics_names})
                    
                    metrics_test_dict_F, metrics_test_dict_G = model_class.valid_step(test_sampler, None, BATCH_SIZE_TEST, metrics_names, source_vectors, target_vectors, N_EVAL)
                    metrics_dict['test_F'].append({key1:{'mean':np.mean(metrics_test_dict_F[key1]), 
                                                       'std':np.std(metrics_test_dict_F[key1])} for key1 in metrics_names})
                    metrics_dict['test_G'].append({key1:{'mean':np.mean(metrics_test_dict_G[key1]), 
                                                       'std':np.std(metrics_test_dict_G[key1])} for key1 in metrics_names})

                    metrics_dict_out = {'train_F':metrics_dict['train_F'][-1], 'test_F':metrics_dict['test_F'][-1],
                                        'train_G':metrics_dict['train_G'][-1], 'test_G':metrics_dict['test_G'][-1]}  

                    
                if wandb_report:
                    report_wandb_fn(metrics_dict, metrics_names, epoch, None)
                    
            plt.close()
            
          
        
    except KeyboardInterrupt:
        print('Interrumpting by keyboard...')
        return model_class, metrics_dict_out

        if wandb_report:
            wandb.finish()
        assert False
    
    return model_class, metrics_dict_out


def train_discrete_old(
                   x, y, labels,
                   test_sampler, 
                   metrics_names, target_vectors,
                   config,
                   wandb_report=False,
                   axis_lims=None, report_every=10):
    

    space_dataset     = SimpleNamespace(**config['dataset'])
    space_training    = SimpleNamespace(**config['training'])
    space_model       = SimpleNamespace(**config['model_specific'])
    
    N_EVAL            = space_dataset.N_EVAL
    DEVICE            = space_dataset.DEVICE
    SOURCE_DIM        = space_dataset.SOURCE_DIM
    TARGET_DIM        = space_dataset.TARGET_DIM
    METHOD_NAME       = space_training.METHOD_NAME
    SEED              = space_dataset.SEED
    MAX_ITERS         = space_training.MAX_ITERS

    #MAX_SAMPLES_TRAIN = train_source_sampler.loader.batch_size
    MAX_SAMPLES_TEST  = test_sampler.loader.batch_size
    
    if METHOD_NAME == 'AlignGW':
        metric_name     = space_training.COST_DISCRETE
        normalize_dists = 'mean'
        eps             = space_model.EPS
        tol             = 1e-8
        method_class    = AlignGW(metric=metric_name, normalize_dists=normalize_dists,
                                  loss_fun='square_loss', eps=eps, tol=tol)
        
    if METHOD_NAME == 'StructuredGW':
        M_init       = None
        method_M     = 'exact'
        eps          = space_model.EPS #1e-4 (for muse)
        tol          = 1e-3
        method_class = StructuredGW(M_init, method_M, eps=eps, tol=tol)

    if METHOD_NAME == 'FlowGW':
        eps       = space_model.EPS #1e-3
        embed_dim = space_model.HIDDEN_SIZES_MLP[0] #1024
        n_layers  = len(space_model.HIDDEN_SIZES_MLP) #4
        n_freq    = space_model.N_FREQ
        mover_lr  = space_model.MOVER_LR
        
        if space_training.COST_DISCRETE == 'cosine':
            cost_fn = costs.Cosine()
        if space_training.COST_DISCRETE == 'euclidean':
            cost_fn = costs.SqEuclidean()   
        
        method_class = FlowGW(eps=eps, embed_dim=embed_dim, n_freq=n_freq, n_layers=n_layers, cost_fn=cost_fn, lr=mover_lr)

    if METHOD_NAME == 'RegGW':
        rng             = jax.random.PRNGKey(SEED)
        #HIDDEN_SIZE_MLP = [512, 256, 256]
        lamb            = space_model.LAMBDA
        eps_reg         = space_model.EPS_REG
        eps_fit         = space_model.EPS_FIT
        
        mover_model     = mlp_jax(hidden_dims=space_model.HIDDEN_SIZES_MLP, out_dim=TARGET_DIM, act_fn=ln.relu)
        mover_optimizer = optax.adam(learning_rate=space_model.MOVER_LR)
        
        if space_training.COST_DISCRETE == 'cosine':
            cost_fn = costs.Cosine()
        if space_training.COST_DISCRETE == 'euclidean':
            cost_fn = costs.SqEuclidean() 
            
        method_class    = RegGW(mover_model, mover_optimizer, SOURCE_DIM, cost_fn, eps_fit, eps_reg, lamb, rng_seed=space_dataset.SEED)
        
    report_keys = ['train', 'test']
    metrics_dict = {key:[] for key in report_keys}

    x_dict, y_dict, labels_dict = {}, {}, {}
    
    try:
            
        #x, labels = train_source_sampler.sample(MAX_SAMPLES_TRAIN)
        #y, _      = train_target_sampler.sample(MAX_SAMPLES_TRAIN)
        

        if 'distortion' in metrics_names:
            distortion_gt = compute_distortion(x, y)
            print('GT distortion:', distortion_gt.item())

            y = y[torch.randperm(y.shape[0])]
            
            distortion_random = compute_distortion(x, y)
            print(f'Random distortion (seed={SEED}):', distortion_random.item())
            
            if wandb_report:
                wandb.log({'distortion_gt':distortion_gt.item(), 'distortion_random':distortion_random.item()}, 
                           step=0)

            #print('x_train_init:', x)
            
            #print('y_train_init:', y)
        
        x_dict['train'], y_dict['train'], labels_dict['train'] = x.cpu(), y.cpu(), labels.cpu()

        x, y, labels = test_sampler.sample(MAX_SAMPLES_TEST)
        x_dict['test'], y_dict['test'], labels_dict['test'] = x.cpu(), y.cpu(), labels.cpu()
        
        method_class.fit(x_dict, y_dict, labels_dict, target_vectors, wandb_report, MAX_ITERS, report_every=report_every) 

        #metrics_train_dict = method_class.valid_step_2(train_source_sampler, train_target_sampler, MAX_SAMPLES_TRAIN, metrics_names, target_vectors, N_EVAL)
        #metrics_dict['train'].append({key1:{'mean':np.mean(metrics_train_dict[key1]), 
        #                                    'std':np.std(metrics_train_dict[key1])} for key1 in metrics_names})
        #metrics_dict['train'].append({'distortion_gt':{'mean':distortion_gt.item(), 'std':0}})
        #metrics_dict['train'].append({'distortion_random':{'mean':distortion_random.item(), 'std':0}})
        
        #metrics_test_dict = method_class.valid_step_2(test_sampler, None,  MAX_SAMPLES_TEST, metrics_names, target_vectors, N_EVAL)
        #metrics_dict['test'].append({key1:{'mean':np.mean(metrics_test_dict[key1]), 
        #                                   'std':np.std(metrics_test_dict[key1])} for key1 in metrics_names})

            
    except KeyboardInterrupt:
        print('Interrumpting by keyboard...')
        if wandb_report:
            wandb.finish()
        assert False
            
        return method_class, metrics_dict

    return method_class, metrics_dict

def train_discrete(train_source_sampler, train_target_sampler, 
                   test_sampler, 
                   metrics_names, target_vectors,
                   config,
                   wandb_report=False,
                   axis_lims=None, report_every=10):
    

    space_dataset     = SimpleNamespace(**config['dataset'])
    space_training    = SimpleNamespace(**config['training'])
    space_model       = SimpleNamespace(**config['model_specific'])
    
    N_EVAL            = space_dataset.N_EVAL
    DEVICE            = space_dataset.DEVICE
    SOURCE_DIM        = space_dataset.SOURCE_DIM
    TARGET_DIM        = space_dataset.TARGET_DIM
    METHOD_NAME       = space_training.METHOD_NAME
    SEED              = space_dataset.SEED
    MAX_ITERS         = space_training.MAX_ITERS

    MAX_SAMPLES_TRAIN = train_source_sampler.loader.batch_size
    MAX_SAMPLES_TEST  = test_sampler.loader.batch_size
    
    if METHOD_NAME == 'AlignGW':
        metric_name     = space_training.COST_DISCRETE
        normalize_dists = 'mean'
        eps             = space_model.EPS
        tol             = 1e-8
        method_class    = AlignGW(metric=metric_name, normalize_dists=normalize_dists,
                                  loss_fun='square_loss', eps=eps, tol=tol)
        
    if METHOD_NAME == 'StructuredGW':
        M_init       = None
        method_M     = 'exact'
        eps          = space_model.EPS #1e-4 (for muse)
        tol          = 1e-3
        method_class = StructuredGW(M_init, method_M, eps=eps, tol=tol)

    if METHOD_NAME == 'FlowGW':
        eps       = space_model.EPS #1e-3
        embed_dim = space_model.HIDDEN_SIZES_MLP[0] #1024
        n_layers  = len(space_model.HIDDEN_SIZES_MLP) #4
        n_freq    = space_model.N_FREQ
        mover_lr  = space_model.MOVER_LR
        
        if space_training.COST_DISCRETE == 'cosine':
            cost_fn = costs.Cosine()
        if space_training.COST_DISCRETE == 'euclidean':
            cost_fn = costs.SqEuclidean()   
        
        method_class = FlowGW(eps=eps, embed_dim=embed_dim, n_freq=n_freq, n_layers=n_layers, cost_fn=cost_fn, lr=mover_lr)

    if METHOD_NAME == 'RegGW':
        rng             = jax.random.PRNGKey(SEED)
        #HIDDEN_SIZE_MLP = [512, 256, 256]
        lamb            = space_model.LAMBDA
        eps_reg         = space_model.EPS_REG
        eps_fit         = space_model.EPS_FIT
        
        mover_model     = mlp_jax(hidden_dims=space_model.HIDDEN_SIZES_MLP, out_dim=TARGET_DIM, act_fn=ln.relu)
        mover_optimizer = optax.adam(learning_rate=space_model.MOVER_LR)
        
        if space_training.COST_DISCRETE == 'cosine':
            cost_fn = costs.Cosine()
        if space_training.COST_DISCRETE == 'euclidean':
            cost_fn = costs.SqEuclidean() 
            
        method_class    = RegGW(mover_model, mover_optimizer, SOURCE_DIM, cost_fn, eps_fit, eps_reg, lamb, rng_seed=space_dataset.SEED)
        
    report_keys = ['train', 'test']
    metrics_dict = {key:[] for key in report_keys}

    x_dict, y_dict, labels_dict = {}, {}, {}
    
    try:
            
        x, labels = train_source_sampler.sample(MAX_SAMPLES_TRAIN)
        y, _      = train_target_sampler.sample(MAX_SAMPLES_TRAIN)
        #print('source: ', x.norm(dim=1).max())
        ##print('target: ', y.norm(dim=1).max())
        

        if 'distortion' in metrics_names:
            distortion_gt = compute_distortion(x, y)
            print('GT distortion:', distortion_gt.item())

            y_original = torch.clone(y)
            y = y[torch.randperm(y.shape[0])]
            
            distortion_random = compute_distortion(x, y)
            print(f'Random distortion (seed={SEED}):', distortion_random.item())
            
            if wandb_report:
                wandb.log({'distortion_gt':distortion_gt.item(), 'distortion_random':distortion_random.item()}, 
                           step=0)

            #print('x_train_init:', x)
            
            #print('y_train_init:', y)
        
        x_dict['train'], y_dict['train'], labels_dict['train'] = x.cpu(), y.cpu(), labels.cpu()

        x, y, labels = test_sampler.sample(MAX_SAMPLES_TEST)
        x_dict['test'], y_dict['test'], labels_dict['test'] = x.cpu(), y.cpu(), labels.cpu()
        
        method_class.fit(x_dict, y_dict, labels_dict, target_vectors, wandb_report, MAX_ITERS, report_every=report_every) 

        metrics_train_dict = method_class.valid_step(train_source_sampler, train_target_sampler, MAX_SAMPLES_TRAIN, metrics_names, target_vectors, N_EVAL)
        metrics_dict['train'].append({key1:{'mean':np.mean(metrics_train_dict[key1]), 
                                            'std':np.std(metrics_train_dict[key1])} for key1 in metrics_names})

        metrics_test_dict = method_class.valid_step(test_sampler, None,  MAX_SAMPLES_TEST, metrics_names, target_vectors, N_EVAL)
        metrics_dict['test'].append({key1:{'mean':np.mean(metrics_test_dict[key1]), 
                                           'std':np.std(metrics_test_dict[key1])} for key1 in metrics_names})

            
    except KeyboardInterrupt:
        print('Interrumpting by keyboard...')
        if wandb_report:
            wandb.finish()
        assert False
            
        return method_class, metrics_dict

    return method_class, metrics_dict

def train_toy_discrete(source_vectors, target_vectors, labels, 
                       config,
                       axis_lims=None, report_every=10):


    space_dataset     = SimpleNamespace(**config['dataset'])
    space_training    = SimpleNamespace(**config['training'])
    space_model       = SimpleNamespace(**config['model_specific'])
    
    DEVICE     = space_dataset.DEVICE
    N_SAMPLES  = space_dataset.N_SAMPLES
    toy_type   = space_dataset.TOY_TYPE
    TARGET_DIM = space_dataset.TARGET_DIM
    SOURCE_DIM = space_dataset.SOURCE_DIM
    
        
    if space_training.METHOD_NAME == 'AlignGW':
        metric_name = space_training.COST_DISCRETE
        normalize_dists = 'mean'
        eps = space_model.EPS
        tol = 1e-8
        method_class = AlignGW(metric=metric_name, normalize_dists=normalize_dists,
                                      loss_fun='square_loss', eps=eps, tol=tol, toy_type=toy_type)

    if space_training.METHOD_NAME == 'StructuredGW':
        M_init = None
        method_M = 'exact'
        eps = space_model.EPS
        tol = 1e-3
        method_class = StructuredGW(M_init, method_M, eps=eps, tol=tol, toy_type=toy_type)

    if space_training.METHOD_NAME == 'FlowGW':
        eps = 1e-4
        embed_dim = 1024
        n_freq = 128
        n_layers = 4
        cost_fn = costs.Cosine()
        toy_type = toy_type
        method_class = FlowGW(eps=eps, embed_dim=embed_dim, n_freq=n_freq, n_layers=n_layers, cost_fn=cost_fn, lr=1e-4, toy_type=toy_type)

    if space_training.METHOD_NAME == 'RegGW':
        rng = jax.random.PRNGKey(config['dataset']['SEED'])
        #HIDDEN_SIZE_MLP = [512, 256, 256]
        mover_model = mlp_jax(hidden_dims=[128, 64, 64], out_dim=TARGET_DIM, act_fn=ln.relu)
        mover_optimizer = optax.adam(learning_rate=space_model.MOVER_LR)
        
        if space_training.COST_DISCRETE == 'cosine':
            cost_fn = costs.Cosine()
        if space_training.COST_DISCRETE == 'euclidean':
            cost_fn = costs.SqEuclidean() 
            
        method_class = RegGW(mover_model, mover_optimizer, SOURCE_DIM, cost_fn, space_training.EPS_FIT, space_training.EPS_REG, space_training.LAMBDA, toy_type=toy_type)
        

    x_dict, y_dict, labels_dict = {}, {}, {}
    
    try:
 
        #x, labels = source_sampler.sample_with_labels(N_SAMPLES)
        #y = target_sampler.sample(N_SAMPLES)
        
        x_dict['train'], y_dict['train'], labels_dict['train'] = source_vectors.cpu(), target_vectors.cpu(), labels.cpu()
        
        x_dict['test'], y_dict['test'], labels_dict['test'] = None, None, None


        method_class.fit(x_dict, y_dict, labels_dict, None, False, space_training.MAX_ITERS, report_every=report_every) 

        plt.close()
            
    except KeyboardInterrupt:
        print('Interrumpting by keyboard...')
        assert False
            
        return method_class

    return method_class

def train_toy_continuous(source_sampler, target_sampler, 
                         config,
                         axis_lims=None, report_every=10):
    
    space_dataset     = SimpleNamespace(**config['dataset'])
    space_training    = SimpleNamespace(**config['training'])
    space_model       = SimpleNamespace(**config['model_specific'])
    
    DEVICE           = space_dataset.DEVICE
    BATCH_SIZE_TRAIN = space_dataset.BATCH_SIZE_TRAIN
    METHOD_NAME      = space_training.METHOD_NAME
    
    SOURCE_DIM       = space_dataset.SOURCE_DIM
    TARGET_DIM       = space_dataset.TARGET_DIM

    SEED             = space_dataset.SEED
    toy_type         = space_dataset.TOY_TYPE
    n_samples_plot   = space_training.N_SAMPLES_PLOT

    if METHOD_NAME == 'NeuralGW':
        critic_model = mlp(TARGET_DIM, hidden_sizes=space_model.HIDDEN_SIZES_MLP).to(DEVICE)
        mover_model  = mlp(SOURCE_DIM, TARGET_DIM, hidden_sizes=space_model.HIDDEN_SIZES_MLP).to(DEVICE)
        cost_model   = InnerGW_linear(SOURCE_DIM, TARGET_DIM, device=DEVICE)
    
        critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=space_model.CRITIC_LR)
        mover_optimizer  = torch.optim.Adam(mover_model.parameters(), lr=space_model.MOVER_LR)
        cost_optimizer   = torch.optim.Adam(cost_model.parameters(), lr=space_model.COST_LR) 
    
        models     = {'cost':cost_model, 'critic':critic_model, 'mover':mover_model}
        optimizers = {'cost':cost_optimizer, 'critic':critic_optimizer, 'mover':mover_optimizer}
        n_iters    = {'cost':space_model.COST_ITERS,'critic':space_model.CRITIC_ITERS,'mover':space_model.MOVER_ITERS}
    
        reg = space_model.REG_CRITIC
        model_class = NeuralGW(models, optimizers, reg)

    if METHOD_NAME == 'RegGW':
        rng = jax.random.PRNGKey(config['dataset']['SEED'])
        mover_model = mlp_jax(hidden_dims=[512, 256, 256], out_dim=TARGET_DIM, act_fn=ln.relu)
        mover_optimizer = optax.adam(learning_rate=var_sp.MOVER_LR)
        n_iters = None
        
        if space_training.COST_DISCRETE == 'cosine':
            cost_fn = costs.Cosine()
        if space_training.COST_DISCRETE == 'euclidean':
            cost_fn = costs.SqEuclidean() 
            
        model_class = RegGW_mb(mover_model, mover_optimizer, SOURCE_DIM, cost_fn, space_model.EPS_FIT, space_model.EPS_REG, space_model.LAMBDA, toy_type=toy_type)

    if METHOD_NAME == 'FlowGW':
        embed_dim = space_model.HIDDEN_SIZES_MLP[0]
        n_layers = len(space_model.HIDDEN_SIZES_MLP)
        
        mover_model = VelocityField(hidden_dims=[embed_dim]*n_layers,
                                    time_dims=[embed_dim, embed_dim],
                                    output_dims=[embed_dim, embed_dim, embed_dim] + [TARGET_DIM],
                                    condition_dims=[embed_dim, embed_dim, embed_dim],
                                    time_encoder=functools.partial(time_encoder.cyclical_time_encoder, n_freqs=space_model.N_FREQ),
            )
        n_iters = None
        cost_fn = costs.Cosine()
        eps = space_model.EPS
        model_class = FlowGW_mb(mover_model, SOURCE_DIM, TARGET_DIM, eps, cost_fn, seed=SEED)


    if METHOD_NAME == 'CycleGW':
        F_model = fcnn(SOURCE_DIM, TARGET_DIM, hidden_sizes=space_model.HIDDEN_SIZES_MLP).to(DEVICE)#fcnn(SOURCE_DIM, hidden_dim=space_model.HIDDEN_SIZES_MLP).to(DEVICE)
        G_model = fcnn(TARGET_DIM, SOURCE_DIM, hidden_sizes=space_model.HIDDEN_SIZES_MLP).to(DEVICE)

        #F_model =  generator_x_y(SOURCE_DIM, space_model.HIDDEN_SIZES_MLP[0], TARGET_DIM).to(torch.float32).to(DEVICE)
        #G_model =  generator_y_x(SOURCE_DIM, space_model.HIDDEN_SIZES_MLP[0], TARGET_DIM).to(torch.float32).to(DEVICE)
        F_optimizer = torch.optim.Adam(F_model.parameters(), lr=space_model.F_LR)
        G_optimizer = torch.optim.Adam(G_model.parameters(), lr=space_model.G_LR)
    
        models     = {'F_model':F_model, 'G_model':G_model}
        optimizers = {'F_model':F_optimizer, 'G_model':G_optimizer}
        n_iters     = None
    
        model_class = CycleGW(models, optimizers, space_model.EPS, space_model.SIGMAS, space_model.REG, space_model.KERNEL_TYPE, space_model.TAKE_MEDIAN)

    if METHOD_NAME == 'EntropicGW':
        critic_model = mlp(SOURCE_DIM, hidden_sizes=space_model.HIDDEN_SIZES_MLP).to(DEVICE)
        critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=space_model.CRITIC_LR)

        continuous_model = MLPRegressor(hidden_layer_sizes=256, random_state=1, max_iter=500)
        
        if space_model.COST_LR is not None:
            cost_model   = CostModel(SOURCE_DIM, TARGET_DIM).to(DEVICE)
            cost_optimizer   = torch.optim.Adam(cost_model.parameters(), lr=space_model.COST_LR)

        else:
            cost_model = torch.eye(SOURCE_DIM, TARGET_DIM).to(DEVICE)
            cost_optimizer = None
        
        models     = {'critic':critic_model, 'cost':cost_model, 'continuous':continuous_model}
        optimizers = {'critic':critic_optimizer, 'cost':cost_optimizer}
        n_iters    = {'critic':space_model.CRITIC_ITERS, 'cost':space_model.COST_ITERS}
    
        model_class = EntropicGW(models, optimizers, space_model.EPS)

    if METHOD_NAME == 'LightGW':
        critic_model = LightSB(dim=TARGET_DIM, n_potentials=space_model.N_POTENTIALS, epsilon=space_model.EPSILON,
                       sampling_batch_size=BATCH_SIZE_TRAIN, S_diagonal_init=0.1, 
                       is_diagonal=True).to(DEVICE)
        cost_model   = InnerGW_linear(SOURCE_DIM, TARGET_DIM, device=DEVICE)
    
        critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=space_model.CRITIC_LR)
        cost_optimizer   = torch.optim.Adam(cost_model.parameters(), lr=space_model.COST_LR) 
    
        models     = {'cost':cost_model, 'critic':critic_model}
        optimizers = {'cost':cost_optimizer, 'critic':critic_optimizer}
        n_iters    = {'cost':space_model.COST_ITERS,'critic':space_model.CRITIC_ITERS}
    
        model_class = LightGW(models, optimizers)

    if METHOD_NAME == 'TRIP_GW':
        critic_model = TRIP_SB(latent_dim=TARGET_DIM, n_components=space_model.N_COMPONENTS, m1=space_model.M1, eps=self.EPSILON, distr_init=self.D_INIT).to(DEVICE)
        cost_model   = InnerGW_linear(SOURCE_DIM, TARGET_DIM, device=DEVICE)
    
        critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=space_model.CRITIC_LR)
        cost_optimizer   = torch.optim.Adam(cost_model.parameters(), lr=space_model.COST_LR) 
    
        models     = {'cost':cost_model, 'critic':critic_model}
        optimizers = {'cost':cost_optimizer, 'critic':critic_optimizer}
        n_iters    = {'cost':space_model.COST_ITERS,'critic':space_model.CRITIC_ITERS}
    
        model_class = LightGW(models, optimizers)

    if METHOD_NAME == 'SimpleGW':
        mover_model  = mlp(SOURCE_DIM, TARGET_DIM, hidden_sizes=space_model.HIDDEN_SIZES_MLP).to(DEVICE)
    
        mover_optimizer  = torch.optim.Adam(mover_model.parameters(), lr=space_model.MOVER_LR)

        n_iters    = space_model.MOVER_ITERS

        model_class = SimpleGW(mover_model, mover_optimizer)
        
    with torch.no_grad():
        
        x_plot, labels_x_plot = source_sampler.sample_with_labels(n_samples_plot)
        #Px_plot_init = x_plot @ cost.matrix
        y_plot, labels_y_plot = target_sampler.sample_with_labels(n_samples_plot)
        
    try:
        for epoch in trange(space_training.N_EPOCHS, leave=False, desc="Epoch"):
            
            P_trained = model_class.train_epoch_toy(source_sampler, target_sampler, BATCH_SIZE_TRAIN, n_iters, epoch, wandb_report=False)  
            
            if epoch % report_every == 0 and epoch != 0:
                if METHOD_NAME in ['NeuralGW', 'SimpleGW']:
                    mover_model_pred = model_class.mover_model
                    
                    mover_model_pred.eval()
                
                    with torch.no_grad():
                        y_sampled = mover_model_pred(x_plot).detach().cpu().numpy()

                if METHOD_NAME == 'RegGW':
                     y_sampled = model_class.state_neural_net.apply_fn({"params":model_class.state_neural_net.params}, x_plot)
                     y_sampled = np.asarray(y_sampled_np)

                if METHOD_NAME == 'CycleGW':
                    with torch.no_grad():
                        y_sampled_F = model_class.F_model(x_plot).cpu().numpy()
                        y_sampled_G = model_class.G_model(y_plot).cpu().numpy()

                if METHOD_NAME == 'EntropicGW':
                    with torch.no_grad():
                        y_sampled = model_class.continuous_model.predict(x_plot.cpu().numpy())

                if METHOD_NAME == 'LightGW':
                    with torch.no_grad():
                        P = model_class.cost_model.matrix.detach()
                        y_sampled = model_class.critic_model(x_plot, P).cpu()
                
                fig = plt.figure(figsize=(11, 11))

                if toy_type == 'toy_3d_2d':
                    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
                    ax2 = fig.add_subplot(2, 2, 2, projection=None)
                    ax3 = fig.add_subplot(2, 2, 3, projection=None)
                    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
            
                ax1.scatter(*x_plot.cpu().T, c=labels_x_plot.cpu().numpy(), cmap="Spectral", alpha=.8)
                ax1.set_title('Source distribution (X)', fontsize=14)

                if METHOD_NAME == 'CycleGW': 
                    ax2.scatter(*y_plot.cpu().T, c=labels_y_plot.cpu().numpy(), cmap="Spectral", alpha=.8)
                    ax2.set_title('Target distribution (Y)', fontsize=14)
                    
                    ax3.scatter(*y_sampled_F.T, c=labels_x_plot.cpu().numpy(),  cmap="Spectral", alpha=.8)
                    ax3.set_title('F(X)')
            
                    ax4.scatter(*y_sampled_G.T, c=labels_y_plot.cpu().numpy(),  cmap="Spectral", alpha=.8)
                    ax4.set_title('G(Y)')
                    
                else:
                    ax2.scatter(*y_plot.cpu().T, c='black', alpha=.8)
                    ax2.set_title('Target distribution', fontsize=14)

                    ax3.scatter(*y_sampled.T, c=labels_x_plot.cpu().numpy(),  cmap="Spectral", alpha=.8)
                    ax3.set_title('Predicted samples')

                    
                 
                 
                 
                plt.show()
                #fig = plt.figure(figsize=(8, 8))
                #
                #if toy_type == 'toy_2d_3d':
                #    ax = fig.add_subplot(projection='3d')
                #   
                #if toy_type == 'toy_3d_2d':
                #    ax = fig.add_subplot(projection=None)
#
                #ax.scatter(*y_sampled_np.T, c=labels_plot.cpu().numpy(),  cmap="Spectral")
                #plt.show()
                        
                    
            
    except KeyboardInterrupt:
        print('Interrumpting by keyboard...')
        if wandb_report:
            wandb.finish()
        assert False
        
        return model_class
    
    return model_class