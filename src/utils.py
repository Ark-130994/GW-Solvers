import itertools
import gensim
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import collections
import matplotlib.pyplot as plt
from scipy import linalg, stats
import os
import pickle
from PIL import Image
import geotorch
import math
import random
from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll
from sklearn.cluster import AgglomerativeClustering
import torch.autograd as autograd

import wandb

def computePotGrad(input, output, create_graph=True, retain_graph=True):
    '''
    :Parameters:
    input : tensor (bs, *shape)
    output: tensor (bs, 1) , NN(input)
    :Returns:
    gradient of output w.r.t. input (in batch manner), shape (bs, *shape)
    '''
    grad = autograd.grad(
        outputs=output, 
        inputs=input,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=retain_graph,
    ) # (bs, *shape) 
    return grad[0]


def save_dict(dictionary, path):
    with open(path, 'wb') as fp:
        pickle.dump(dictionary, fp)
        print('Dictionary saved successfully to file')
        
def load_dict(path):
    with open(path, 'rb') as fp:
        a = pickle.load(fp)
    return a

def nwise(iterable, n=2):
    iters = itertools.tee(iterable, n)
    for i, it in enumerate(iters):
        next(itertools.islice(it, i, i), None)
    return zip(*iters)

def get_dict(dict_path, source, target):

    dictf = open(dict_path, encoding='utf-8', errors='surrogateescape')
    src2trg = collections.defaultdict(set)

    vocab = set()

    for line in dictf:
        splitted = line.split()
        if len(splitted) > 2:
            # Only using first translation if many are provided
            src, trg = splitted[:2]
        elif len(splitted) == 2:
            src, trg = splitted

        src_ind = source.w2i[src]
        trg_ind = target.w2i[trg]
        src2trg[src_ind].add(trg_ind)
        vocab.add(src)
    return vocab, src2trg

def get_gold_dict(n):
    gold_dict = collections.defaultdict(set)

    for i in range(n):
        gold_dict[i].add(i) 
    return gold_dict

def get_samples(iter_dataloader, dataloader, device): 
    try:
        source_data, target_data, label = next(iter_dataloader)
    except StopIteration:
        iter_dataloader = iter(dataloader)
        source_data, target_data, label = next(iter_dataloader)

    source_data = source_data.to(device)
    target_data = target_data.to(device)
    label = label
    
    data = (source_data, target_data, label)

    return data, iter_dataloader

@torch.no_grad()
def plot_toy(source_samples, target_samples, moved_samples, P, Px_init, 
             y_discrete, plot_cfg, *,
             colors=None, axis_lims=None, **figure_kwargs):
    
    plot_cfg_ = plot_cfg[plot_cfg.index('toy_')+4:]

    if plot_cfg_ == '2d_2d': ax_dict={'ax1':None, 'ax2':None, 'ax3':None, 'ax4':None}
    if plot_cfg_ in ['3d_2d', 'S3d_2d']: ax_dict={'ax1':'3d', 'ax2':None, 'ax3':None, 'ax4':None}
    if plot_cfg_ in ['2d_3d', 'S2d_3d']: ax_dict={'ax1':None, 'ax2':'3d', 'ax3':'3d', 'ax4':'3d'}
        
    figure = plt.figure(**figure_kwargs)
    
    ax1 = figure.add_subplot(2, 2, 1, projection=ax_dict['ax1'])
    ax2 = figure.add_subplot(2, 2, 2, projection=ax_dict['ax2'])
    ax3 = figure.add_subplot(2, 2, 3, projection=ax_dict['ax3'])
    ax4 = figure.add_subplot(2, 2, 4, projection=ax_dict['ax4'])    
        #ax2 = figure.add_subplot(2, 2, 2, projection='3d', elev=7, azim=-80)
    
    if plot_cfg_ == 'S3d_2d': 
        ax1 = figure.add_subplot(2, 2, 1, projection=ax_dict['ax1'], elev=7, azim=-80)
    if plot_cfg_ == 'S2d_3d': 
        ax2 = figure.add_subplot(2, 2, 2, projection=ax_dict['ax2'], elev=7, azim=-80)
        ax3 = figure.add_subplot(2, 2, 3, projection=ax_dict['ax3'], elev=7, azim=-80)
        ax4 = figure.add_subplot(2, 2, 4, projection=ax_dict['ax4'], elev=7, azim=-80)  
        
    Px = source_samples @ P
    
    ax1.scatter(*source_samples.cpu().T, c=colors, cmap="Spectral", alpha=.8)
    ax1.set_title('Source distribution')
    
    ax2.scatter(*Px_init.cpu().T, c='black', alpha=.3, label='Init')
    ax2.scatter(*Px.cpu().T, c=colors, cmap="Spectral", alpha=.8)
    ax2.set_title('Px')
    ax2.legend()

    ax3.scatter(*target_samples.cpu().T, c='black',
                label='Target samples', alpha=.8)
    ax3.scatter(*moved_samples.cpu().T, c=colors,
                label='Moved samples', alpha=.8, cmap="Spectral")
    ax3.set_title('Target space')
    ax3.legend()
    
    ax4.scatter(*y_discrete.cpu().T, c=colors, cmap="Spectral")
    ax4.set_title('Discrete GW Solution')
    
    axes1 = [ax1]
    axes2 = [ax2, ax3, ax4]
    
    if axis_lims is None:
        xlim = ax3.get_xlim()
        ylim = ax3.get_ylim()
        
        if ax_dict['ax3'] == '3d': 
            zlim = ax3.get_zlim()
            #axes2.append(ax2)
            plt.setp(axes2, xlim=xlim, ylim=ylim, zlim=zlim)
            axis_lims = (xlim, ylim, zlim)
            
        else:
            plt.setp(axes2, xlim=xlim, ylim=ylim)
            axis_lims = (xlim, ylim)
    else:
        xlim = axis_lims[0]
        ylim = axis_lims[1]
       
        if ax_dict['ax3'] == '3d': 
            #axes2.append(ax2)
            zlim = axis_lims[2]
            plt.setp(axes2, xlim=xlim, ylim=ylim, zlim=zlim)
        else:
            plt.setp(axes2, xlim=xlim, ylim=ylim)

        
    plt.show()
    
    return figure, axis_lims


@torch.no_grad()
def plot_discrete(source_samples, target_samples, discrete_GW, discrete_EGW, 
                  plot_cfg, *,
                  colors=None, **figure_kwargs):
    
    
    if plot_cfg == '2d_2d': ax_dict={'ax1':None, 'ax2':None, 'ax3':None, 'ax4':None}
    if plot_cfg in ['3d_2d', 'S3d_2d']: ax_dict={'ax1':'3d', 'ax2':None, 'ax3':None, 'ax4':None}
    if plot_cfg in ['2d_3d', 'S2d_3d']: ax_dict={'ax1':None, 'ax2':'3d', 'ax3':'3d', 'ax4':'3d'}
        
    figure = plt.figure(**figure_kwargs)
    
    ax1 = figure.add_subplot(2, 2, 1, projection=ax_dict['ax1'])
    ax2 = figure.add_subplot(2, 2, 2, projection=ax_dict['ax2'])
    ax3 = figure.add_subplot(2, 2, 3, projection=ax_dict['ax3'])
    ax4 = figure.add_subplot(2, 2, 4, projection=ax_dict['ax4'])    
        #ax2 = figure.add_subplot(2, 2, 2, projection='3d', elev=7, azim=-80)
    
    if plot_cfg == 'S3d_2d': 
        ax1 = figure.add_subplot(2, 2, 1, projection=ax_dict['ax1'], elev=5, azim=-80)
    if plot_cfg == 'S2d_3d': 
        ax2 = figure.add_subplot(2, 2, 2, projection=ax_dict['ax2'], elev=3, azim=-80)
        ax3 = figure.add_subplot(2, 2, 3, projection=ax_dict['ax3'], elev=3, azim=-80)
        ax4 = figure.add_subplot(2, 2, 4, projection=ax_dict['ax4'], elev=3, azim=-80)  
    
    ax1.scatter(*source_samples.cpu().T, c=colors, cmap="Spectral")
    ax1.set_title('Source distribution')
    
    ax2.scatter(*target_samples.cpu().T, c='gray', cmap="Spectral")
    ax2.set_title('Target distribution')

    ax3.scatter(*discrete_GW.cpu().T, c=colors, cmap="Spectral")
    ax3.set_title('Discrete GW distribution')
    
    ax4.scatter(*discrete_EGW.cpu().T, c=colors, cmap="Spectral")
    ax4.set_title('Discrete Entropic GW distribution')
    
    axes1 = [ax1, ax2]
    axes2 = [ax3, ax4]
    
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    plt.setp(axes1, xlim=xlim, ylim=ylim)
        
    xlim = ax3.get_xlim()
    ylim = ax3.get_ylim()
    if ax_dict['ax3'] == '3d': 
        zlim = ax3.get_zlim()
        axes2.append(ax2)
        plt.setp(axes2, xlim=xlim, ylim=ylim, zlim=zlim)
    else:
        axes2.append(ax2)
        plt.setp(axes2, xlim=xlim, ylim=ylim)

    plt.show()
    
    return figure

@torch.no_grad()
def pca_plot(x, y, labels, sampling_model, P, pca_models, axis_lims=None, **figure_kwargs):
    
    pca_model_X = pca_models['source']
    pca_model_Y = pca_models['target']
        
    y_sampled = sampling_model(x).cpu()
        
    #model_pca = PCA(n_components=2)
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    y_sampled = y_sampled.detach().cpu().numpy()
    P = P.detach().cpu().numpy()
    
    x_reduced = pca_model_X.transform(x)
    y_reduced = pca_model_Y.transform(y)
    y_sampled_reduced = pca_model_Y.transform(y_sampled)
    
    fig = plt.figure(**figure_kwargs)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.scatter(*x_reduced.T, label='Source samples')
    ax1.set_title('Source space')
    ax1.legend()
    
    Px = x @ P
    Px_reduced = pca_model_Y.transform(Px)
    ax2.scatter(*Px_reduced.T)
    ax2.set_title('Px')

    ax3.scatter(*y_reduced.T, color='blue', label='y_target')
    ax3.scatter(*y_sampled_reduced.T, color='red', label='y_sampled')
    ax3.set_title('Target distribution')
    ax3.legend()

    if axis_lims is None:
        xlim2, ylim2 = ax2.get_xlim(), ax2.get_ylim()
        xlim3, ylim3 = ax3.get_xlim(), ax3.get_ylim()

        axis_lims = ((xlim2, ylim2), (xlim3, ylim3))
        
           
    plt.setp(ax2, xlim=axis_lims[0][0], ylim=axis_lims[0][1])
    plt.setp(ax2, xlim=axis_lims[1][0], ylim=axis_lims[1][1])
    
    
    plt.show()
    #fig_pca = utils.pca_plot(x_reduced, y_reduced, y_sampled_reduced, P, figsize=(15, 5))
    
    return fig, axis_lims

@torch.no_grad()
def get_target_dict(target, n_samples=2048):
    target_samples = target.sample((n_samples,))
    target_mean = torch.mean(target_samples, dim=0)
    target_cov = torch.cov(target_samples.T)
    target_var = torch.trace(target_cov)
    target_dict = {'mean':target_mean, 'cov':target_cov, 'var':target_var}
    return target_dict


@torch.no_grad()
def get_target_dict_nosampler(target):
    target_samples = (target)
    target_mean = torch.mean(target_samples, dim=0)
    target_cov = torch.cov(target_samples.T)
    target_var = torch.trace(target_cov)
    target_dict = {'mean':target_mean, 'cov':target_cov, 'var':target_var}
    return target_dict

@torch.no_grad()
def get_pca_models(source_vectors, target_vectors, n_samples=4096):
    
    source_samples = source_vectors[:n_samples]
    target_samples = target_vectors[:n_samples]

    pca_model_source = PCA(n_components=2)
    pca_model_source.fit(source_samples)

    pca_model_target = PCA(n_components=2)
    pca_model_target.fit(target_samples)
    pca_models = {'source':pca_model_source, 'target':pca_model_target}
    
    return pca_models

@torch.no_grad()
def pca_plot_gene(X_pca, y_pca):

    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].scatter(X_pca[:,0], X_pca[:,1], c="k", s=15, label="Gene Expression (mover)")
    axes[0].scatter(y_pca[:,0], y_pca[:,1], c="r", s=15, label="DNA Methylation (target)")
    axes[0].legend()
    axes[0].set_title("Colored based on domains (Aligned Domains)")

    cellTypes_rna=np.loadtxt("../datasets/scgem/scGEM_typeExpression.txt")
    cellTypes_methyl=np.loadtxt("../datasets/scgem/scGEM_typeMethylation.txt")

    colormap = plt.get_cmap('rainbow', 5) 
    plt.scatter(X_pca[:,0], X_pca[:,1], c=cellTypes_rna, s=15, cmap=colormap)
    plt.scatter(y_pca[:,0], y_pca[:,1], c=cellTypes_methyl, s=15, cmap=colormap)
    cbar=plt.colorbar()

    # approximately center the colors on the colorbar when adding cell type labels
    tick_locs = (np.arange(0,6)+0.8) *4/5 
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(["BJ", "d8", "d16T+", "d24T+", "iPS"]) #cell-type labels
    plt.title("Colored based on cell type identity")
    plt.show()
    
    return fig

def random_cov_matrix(
    dim,
    eigval_range=(.5, 2.),
    eigval_dist="log"):
    if eigval_dist == "uniform":
        diag = np.diag(np.random.uniform(*eigval_range, size=dim))
    elif eigval_dist == "log":
        diag = np.diag(np.exp(np.random.uniform(*np.log(eigval_range), size=dim)))
    rotation = stats.ortho_group.rvs(dim)
    return torch.tensor(rotation @ diag @ rotation.T).float()

def seed_everything(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def fig2img ( fig ):
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf



class Config():
    @staticmethod
    def fromdict(config_dict):
        config = Config()
        for name, val in config_dict.items():
            setattr(config, name, val)
        return config
    
    @staticmethod
    def load(path):
        os.makedirs(os.path.join(*("#" + path).split('/')[:-1])[1:], exist_ok=True)
        with open(path, 'rb') as handle:
            config_dict = pickle.load(handle)
        return Config.fromdict(config_dict)

    def store(self, path):
        os.makedirs(os.path.join(*("#" + path).split('/')[:-1])[1:], exist_ok=True)
        with open(path, 'wb') as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def set_attributes(
            self, 
            attributes_dict, 
            require_presence : bool = True,
            keys_upper: bool = True
        ) -> int:
        _n_set = 0
        for attr, val in attributes_dict.items():
            if keys_upper:
                attr = attr.upper()
            set_this_attribute = True
            if require_presence:
                if not attr in self.__dict__.keys():
                    set_this_attribute = False
            if set_this_attribute:
                if isinstance(val, list):
                    val = tuple(val)
                setattr(self, attr, val)
                _n_set += 1
        return _n_set
