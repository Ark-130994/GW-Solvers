import torch
import itertools
from scipy.stats import ortho_group

import numpy as np

from gensim.downloader import load
import os
from gensim.models import KeyedVectors
import random
from torch.utils.data import TensorDataset, DataLoader
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type, Union

from src import utils
from types import SimpleNamespace 
import moscot
import sklearn.preprocessing as pp
import scanpy as sc
from src import distributions
from src.utils import fig2img
import jax

class Sampler:
    def __init__(
        self, device='cuda',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass
    
class LoaderSampler(Sampler):
    def __init__(self, loader, device='cuda'):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
    def __len__(self):
        return len(self.loader)
    
    def reset_sampler(self):
        self.it = iter(self.loader)
        
    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            source_batch, target_batch, labels = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(source_batch) < size:
            return self.sample(size)
            
        return source_batch[:size].to(self.device), target_batch[:size].to(self.device), labels[:size].to(self.device)

class LoaderSampler2(Sampler):
    def __init__(self, loader, device='cuda'):
        super(LoaderSampler2, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
    def __len__(self):
        return len(self.loader)
    
    def reset_sampler(self):
        self.it = iter(self.loader)
        
    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            source_batch, reference_batch, target_batch, labels = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(source_batch) < size:
            return self.sample(size)
            
        return source_batch[:size].to(self.device), reference_batch[:size].to(self.device), target_batch[:size].to(self.device), labels[:size].to(self.device)

def get_indices(data_path, config):
    
    var_sp = SimpleNamespace(**config['dataset']) 
    N_EVAL          = var_sp.N_EVAL
    N_MAX_SAMPLES   = var_sp.N_MAX_SAMPLES
    N_TRAIN_SAMPLES = var_sp.N_TRAIN_SAMPLES
    N_TEST_SAMPLES  = var_sp.N_TEST_SAMPLES * N_EVAL
    DATASET_NAME    = var_sp.DATASET_NAME
    
    random.seed(var_sp.SEED)

    assert N_MAX_SAMPLES - N_TEST_SAMPLES >= N_TRAIN_SAMPLES, 'Reduce the number of train or test samples.'

    random_indices       = random.sample(range(0, N_MAX_SAMPLES - N_TEST_SAMPLES), N_TRAIN_SAMPLES)            
    indices_test  = list(range(N_MAX_SAMPLES - N_TEST_SAMPLES, N_MAX_SAMPLES))

    random_indices.extend(indices_test)
    
    indices_train = list(range(0, N_TRAIN_SAMPLES))
    indices_test  = list(range(N_TRAIN_SAMPLES, N_TRAIN_SAMPLES + N_TEST_SAMPLES))
    
    if DATASET_NAME in ['wiki-gigaword', 'twitter']:
        source_model, target_model = load_glove(DATASET_NAME, data_path, var_sp.SOURCE_DIM, var_sp.TARGET_DIM)
        
        source_class = embeddings(source_model)
        target_class = embeddings(target_model)
        
        random_words = [source_class.i2w[ix] for ix in random_indices]
        
        source_class.restrict(random_words)
        target_class.restrict(random_words)
        
        return source_class.vectors, target_class.vectors, indices_train, indices_test
        
    if var_sp.DATASET_NAME in ['bone_marrow']:
        
        source, target, labels = load_biodata(data_path + '/', config)
                
        return source, target, indices_train, indices_test
        
    
def get_samplers(source_vectors, target_vectors, random_indices_train, random_indices_test, config):
    
    var_sp = SimpleNamespace(**config['dataset']) 
                                   
    random_indices_train_source = random_indices_train[:int(len(random_indices_train) * (0.5))]
    random_indices_train_target = random_indices_train[int(len(random_indices_train) * (0.5-var_sp.ALPHA*0.5)): int(len(random_indices_train) * (0.5 + 0.5 -var_sp.ALPHA*0.5))]

    random_indices_train_source = torch.tensor(random_indices_train_source).to(torch.int32)
    random_indices_train_target = torch.tensor(random_indices_train_target).to(torch.int32)

    random_indices_test = torch.tensor(random_indices_test).to(torch.int32)
    
    print('Source pairs...')
    print(random_indices_train_source)
    print(len(random_indices_train_source))
        
    print('Target pairs...')
    print(random_indices_train_target)
    print(len(random_indices_train_target))
    
    source_len = len(random_indices_train_source)
    target_len = len(random_indices_train_target)

    if source_len > target_len:
        random_indices_train_source = random_indices_train_source[source_len-target_len:]

    if source_len < target_len:
        random_indices_train_target = random_indices_train_target[:source_len]

    intersected_indices = list(set(random_indices_train_source.numpy()).intersection(random_indices_train_target.numpy()))
        
    if var_sp.TRAIN_TYPE == 'continuous':
        batch_size_train = var_sp.BATCH_SIZE_TRAIN
        batch_size_test = var_sp.BATCH_SIZE_TEST
        
    if var_sp.TRAIN_TYPE == 'discrete':
        batch_size_train = source_len
        batch_size_test = var_sp.N_TEST_SAMPLES
    
    assert np.isclose(len(intersected_indices)/(var_sp.N_TRAIN_SAMPLES/2), var_sp.ALPHA, rtol=1e-2)
        
    if var_sp.NORMALIZE_VECS is True:
        print('Normalizing source and target spaces...')
        source_vectors -= source_vectors.mean(axis=0)
        target_vectors -= target_vectors.mean(axis=0)
        
        source_vectors /= np.linalg.norm(source_vectors, axis=1)[:,None]
        target_vectors /= np.linalg.norm(target_vectors, axis=1)[:,None]
        
    trainset = TensorDataset(source_vectors[random_indices_train_source], target_vectors[random_indices_train_target], random_indices_train_source)
    trainloader = DataLoader(trainset, batch_size=batch_size_train)
    train_sampler = LoaderSampler(trainloader, device=var_sp.DEVICE)
    
    if len(random_indices_test) != 0 or random_indices_test is None:
        testset = TensorDataset(source_vectors[random_indices_test], target_vectors[random_indices_test], random_indices_test)
        testloader = DataLoader(testset, batch_size=batch_size_test)
        test_sampler = LoaderSampler(testloader, device=var_sp.DEVICE)
    else: 
        test_sampler = None
    
    return source_vectors, target_vectors, train_sampler, test_sampler

def get_samplers_toy(config):
    
    var_sp = SimpleNamespace(**config['dataset']) 
    TOTAL_SAMPLES = var_sp.N_SAMPLES
    DEVICE = var_sp.DEVICE
    N_CLUSTERS = var_sp.N_CLUSTERS
    utils.seed_everything(var_sp.SEED)
    
    
    if var_sp.TOY_TYPE == 'toy_3d_2d':
        
        locs_src = distributions.fibonacci_sphere(N_CLUSTERS)
        scales_src = torch.full_like(locs_src, .1)
        source = distributions.GaussianMixture(locs_src, scales_src, device=DEVICE)

        locs_tgt = distributions.uniform_circle(N_CLUSTERS)
        scales_tgt = torch.full_like(locs_tgt, .1)
        target = distributions.GaussianMixture(locs_tgt, scales_tgt, device=DEVICE)
        
        source_vectors, labels = source.sample_with_labels((TOTAL_SAMPLES, ))
        target_vectors = target.sample((TOTAL_SAMPLES, ))


        
    if var_sp.TOY_TYPE == 'toy_2d_3d':
        
        locs_src = distributions.uniform_circle(N_CLUSTERS)
        scales_src = torch.full_like(locs_src, .1)
        source = distributions.GaussianMixture(locs_src, scales_src, device=DEVICE)

        locs_tgt = distributions.fibonacci_sphere(N_CLUSTERS) 
        scales_tgt = torch.full_like(locs_tgt, .1)
        target = distributions.GaussianMixture(locs_tgt, scales_tgt, device=DEVICE)
        
        source_vectors, labels = source.sample_with_labels((TOTAL_SAMPLES, ))
        target_vectors = target.sample((TOTAL_SAMPLES, ))

    if var_sp.TOY_TYPE == 'toy_3d_3d_iso':
        
        locs_src = distributions.fibonacci_sphere(N_CLUSTERS)
        scales_src = torch.full_like(locs_src, .1)
        source = distributions.GaussianMixture(locs_src, scales_src, device=DEVICE)
        
        R1 = torch.tensor(ortho_group.rvs(3)).to(torch.float32).to(DEVICE)
        #t1 = 2 * torch.rand(3) - 1
        
        source_vectors, labels = source.sample_with_labels((TOTAL_SAMPLES, ))
        reference_vectors = source_vectors @ R1 #+ t1
        reference_vectors = reference_vectors.to(torch.float32)

        R2 = torch.tensor([[1, 0.7, 1.2],[0, 1, 0],[0, 0, 1]]).to(torch.float32).to(DEVICE)
        
        target_vectors = reference_vectors @ R2

        batch_size_train = var_sp.N_SAMPLES
        trainset = TensorDataset(source_vectors, reference_vectors, target_vectors, labels)
        trainloader = DataLoader(trainset, batch_size=batch_size_train)
        train_sampler = LoaderSampler2(trainloader, device=var_sp.DEVICE)

        return train_sampler
        
    if var_sp.NORMALIZE_VECS is True:
        print('Normalizing source and target spaces...')
        source_vectors -= source_vectors.mean(axis=0)
        target_vectors -= target_vectors.mean(axis=0)
        
        source_vectors /= np.linalg.norm(source_vectors, axis=1)[:,None]
        target_vectors /= np.linalg.norm(target_vectors, axis=1)[:,None]
        
    batch_size_train = var_sp.N_SAMPLES
    trainset = TensorDataset(source_vectors, target_vectors, labels)
    trainloader = DataLoader(trainset, batch_size=batch_size_train)
    train_sampler = LoaderSampler(trainloader, device=var_sp.DEVICE)
    
    return train_sampler

def get_samplers_continuous_toy(config):
    
    var_sp = SimpleNamespace(**config['dataset']) 
    DEVICE = var_sp.DEVICE
    N_CLUSTERS_SOURCE = var_sp.N_CLUSTERS_SOURCE
    N_CLUSTERS_TARGET = var_sp.N_CLUSTERS_TARGET
    
    utils.seed_everything(var_sp.SEED)
    
    
    if var_sp.TOY_TYPE == 'toy_3d_2d':
        
        locs_src = distributions.fibonacci_sphere(N_CLUSTERS_SOURCE)
        scales_src = torch.full_like(locs_src, .1)
        source_sampler = distributions.GaussianMixture(locs_src, scales_src, device=DEVICE)

        locs_tgt = distributions.uniform_circle(N_CLUSTERS_TARGET)
        scales_tgt = torch.full_like(locs_tgt, .1)
        target_sampler = distributions.GaussianMixture(locs_tgt, scales_tgt, device=DEVICE)

    if var_sp.TOY_TYPE == 'toy_3d_2d_circle':
        
        locs_src = distributions.fibonacci_sphere(N_CLUSTERS_SOURCE)
        scales_src = torch.full_like(locs_src, .1)
        source_sampler = distributions.GaussianMixture(locs_src, scales_src, device=DEVICE)
        target_sampler = distributions.circle_distribution(device=DEVICE)
        
    if var_sp.TOY_TYPE == 'toy_2d_2d':
        
        locs_src = distributions.uniform_circle(N_CLUSTERS_SOURCE)
        scales_src = torch.full_like(locs_src, .1)
        source_sampler = distributions.GaussianMixture(locs_src, scales_src, device=DEVICE)

        locs_trg = distributions.uniform_circle(N_CLUSTERS_TARGET)
        scales_trg = torch.full_like(locs_trg, .1)
        target_sampler = distributions.GaussianMixture(locs_trg, scales_trg, device=DEVICE)
        
    if var_sp.TOY_TYPE == 'toy_2d_3d':
        
        locs_src = distributions.uniform_circle(N_CLUSTERS_SOURCE)
        scales_src = torch.full_like(locs_src, .1)
        source_sampler = distributions.GaussianMixture(locs_src, scales_src, device=DEVICE)

        locs_tgt = distributions.fibonacci_sphere(N_CLUSTERS_TARGET) 
        scales_tgt = torch.full_like(locs_tgt, .1)
        target_sampler = distributions.GaussianMixture(locs_tgt, scales_tgt, device=DEVICE)
        
        #source_vectors, labels = source.sample_with_labels((TOTAL_SAMPLES, ))
        #target_vectors = target.sample((TOTAL_SAMPLES, ))

    if var_sp.TOY_TYPE == 'toy_3d_3d_iso':
        
        locs_src = distributions.fibonacci_sphere(N_CLUSTERS)
        scales_src = torch.full_like(locs_src, .1)
        source = distributions.GaussianMixture(locs_src, scales_src, device=DEVICE)
        
        R1 = torch.tensor(ortho_group.rvs(3)).to(torch.float32).to(DEVICE)
        #t1 = 2 * torch.rand(3) - 1
        
        source_vectors, labels = source.sample_with_labels((TOTAL_SAMPLES, ))
        reference_vectors = source_vectors @ R1 #+ t1
        reference_vectors = reference_vectors.to(torch.float32)

        R2 = torch.tensor([[1, 0.7, 1.2],[0, 1, 0],[0, 0, 1]]).to(torch.float32).to(DEVICE)
        
        target_vectors = reference_vectors @ R2

        batch_size_train = var_sp.N_SAMPLES
        trainset = TensorDataset(source_vectors, reference_vectors, target_vectors, labels)
        trainloader = DataLoader(trainset, batch_size=batch_size_train)
        train_sampler = LoaderSampler2(trainloader, device=var_sp.DEVICE)

        return train_sampler
        
    
    return source_sampler, target_sampler

def load_model(name, emb_dim):
    full_name = 'glove-' + name + '-' + str(emb_dim)
    model = load(full_name)
    return model

def load_glove(dataset_name, data_path, SOURCE_DIM, TARGET_DIM):
    
    if f'{dataset_name}_{SOURCE_DIM}.d2v' not in os.listdir(data_path):
        print(f'Downloading model {dataset_name}_{SOURCE_DIM} to {data_path}')
        source_model = load_model(dataset_name, SOURCE_DIM)
        source_model.save(f'{data_path}/{dataset_name}_{SOURCE_DIM}.d2v')
    else:
        print(f'Loading model {dataset_name}_{SOURCE_DIM} to source...')
        source_model = KeyedVectors.load(f'{data_path}/{dataset_name}_{SOURCE_DIM}.d2v')
        
    if f'{dataset_name}_{TARGET_DIM}.d2v' not in os.listdir(data_path):
        print(f'Downloading model {dataset_name}_{TARGET_DIM} to {data_path}')
        target_model = load_model(dataset_name, TARGET_DIM)
        target_model.save(f'{data_path}/{dataset_name}_{TARGET_DIM}.d2v')
    else:
        print(f'Loading model {dataset_name}_{TARGET_DIM} to target...')
        target_model = KeyedVectors.load(f'{data_path}/{dataset_name}_{TARGET_DIM}.d2v')
    
    return source_model, target_model

    
class embeddings: 

    def __init__(self, model, dataset='glove'):
        
        if dataset == 'glove':
            self.vectors = torch.FloatTensor(model.vectors)
            self.i2w = model.index_to_key
            self.w2i = model.key_to_index

    def restrict(self, words):
        i2w = [w for w, _ in itertools.groupby(words)]
        ix = torch.tensor([self.w2i[word] for word in i2w if word in self.w2i])
        
        vectors = self.vectors[ix]
       
        w2i = {w: i for i, w in enumerate(i2w)}
        self.vectors = vectors
        self.i2w = i2w
        self.w2i = w2i

from sklearn.decomposition import PCA

def load_biodata(data_path, config):

    fused_dim       = config['dataset']['FUSED_DIM']
    n_train_samples = config['dataset']['N_TRAIN_SAMPLES']
    n_test_samples  = config['dataset']['N_TEST_SAMPLES'] * config['dataset']['N_EVAL']
    seed            = config['dataset']['SEED']
    
    if fused_dim is None:
        fused_dim = 0
        
    adata_atac = moscot.datasets.bone_marrow(path=data_path, rna=False)
    adata_rna = moscot.datasets.bone_marrow(path=data_path, rna=True)
    
    adata_source = adata_atac.copy()
    adata_target = adata_rna.copy()
    
    n_cells_source = len(adata_atac)
    
    inds_train = np.asarray(jax.random.choice(jax.random.PRNGKey(seed), n_cells_source, (n_train_samples,), replace=False))
    inds_test = list(set(list(range(n_cells_source))) - set(np.asarray(inds_train)))[:n_test_samples]
    
    adata_source_train = adata_source[inds_train, :]
    adata_target_train = adata_target[inds_train, :]

    adata_source_test = adata_source[inds_test, :]
    adata_target_test = adata_target[inds_test, :]
    
    if fused_dim > 0:
        # This should be done only with train data, for test data just apply the same pca.
        # Use only fused, regular optimal transport. Set fused_dim to a high number.
        
        fused_train = np.concatenate((adata_source_train.obsm["geneactivity_scvi"], adata_target_train.obsm["geneactivity_scvi"]), axis=0)
        
        pca = PCA(n_components=fused_dim)
        pca.fit(fused_train)
        fused_train =  pca.transform(fused_train) 
        
        source_fused_train = fused_train[:len(adata_source_train), :]
        target_fused_train = fused_train[len(adata_target_train):, :]

        fused_test = np.concatenate((adata_source_test.obsm["geneactivity_scvi"], adata_target_test.obsm["geneactivity_scvi"]), axis=0)
        fused_test = pca.transform(fused_test)

        source_fused_test = fused_test[:len(adata_source_test), :]
        target_fused_test = fused_test[len(adata_target_test):, :]        

    labels_dict = {CT:ix for ix, CT in enumerate(set(adata_atac.obs['cell_type'].values))}
    labels = [labels_dict[val] for val in adata_atac.obs['cell_type'].values]
    
    if fused_dim > 0 and fused_dim <= 25:
        source_q = pp.normalize(adata_source.obsm["ATAC_lsi_red"], norm="l2") 
        target_q = adata_target.obsm["GEX_X_pca"]
    
        source_q_train = source_q[inds_train, :]
        target_q_train = target_q[inds_train, :]
        
        source_q_test = source_q[inds_test, :]
        target_q_test = target_q[inds_test, :]
        
        source_train = np.concatenate((source_fused_train, source_q_train), axis=1)
        target_train = np.concatenate((target_fused_train, target_q_train), axis=1)

        source_test = np.concatenate((source_fused_test, source_q_test), axis=1)
        target_test = np.concatenate((target_fused_test, target_q_test), axis=1)
        
    elif fused_dim == 0:
        source_q = pp.normalize(adata_source.obsm["ATAC_lsi_red"], norm="l2") 
        target_q = adata_target.obsm["GEX_X_pca"]
    
        source_q_train = source_q[inds_train, :]
        target_q_train = target_q[inds_train, :]
        
        source_q_test = source_q[inds_test, :]
        target_q_test = target_q[inds_test, :]
        
        source_train = np.copy(source_q_train)
        target_train = np.copy(target_q_train)

        source_test = np.copy(source_q_test)
        target_test = np.copy(target_q_test)

    elif fused_dim > 25:
        source_train = np.copy(source_fused_train)
        target_train = np.copy(target_fused_train)

        source_test = np.copy(source_fused_test)
        target_test = np.copy(target_fused_test)

    source = np.concatenate((source_train, source_test), axis=0)
    target = np.concatenate((target_train, target_test), axis=0)
    
    source = torch.tensor(source).to(torch.float32)
    target = torch.tensor(target).to(torch.float32)
    labels = torch.tensor(labels).to(torch.int32)

    #del source_train, target_train, source_test, target_test
    #del source_q_train, target_q_train, source_q_test, target_q_test
    #del source_fused_train, target_fused_train, source_fused_test, target_fused_test
    #del adata_atac, adata_rna, adata_source, adata_target
    
    return source, target, labels
