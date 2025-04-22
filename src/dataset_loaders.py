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

class LoaderSamplerTrain(Sampler):
    def __init__(self, loader, device='cuda'):
        super(LoaderSamplerTrain, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
    def __len__(self):
        return len(self.loader)
    
    def reset_sampler(self):
        self.it = iter(self.loader)
        
    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch, labels = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)
            
        return batch[:size].to(self.device), labels[:size].to(self.device)
        
class LoaderSamplerTest(Sampler):
    def __init__(self, loader, device='cuda'):
        super(LoaderSamplerTest, self).__init__(device)
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


        
def load_vectors(data_path, config):
    
    space_dataset   = SimpleNamespace(**config['dataset']) 
    DATASET_NAME    = space_dataset.DATASET_NAME
    N_MAX_SAMPLES   = space_dataset.N_MAX_SAMPLES
    
    EMB_TYPE_SOURCE = space_dataset.EMB_TYPE_SOURCE
    EMB_TYPE_TARGET = space_dataset.EMB_TYPE_TARGET

    SOURCE_DIM      = space_dataset.SOURCE_DIM
    TARGET_DIM      = space_dataset.TARGET_DIM


    if DATASET_NAME in ['muse_multi', 'muse']:
        SOURCE_LANG = space_dataset.SOURCE_LANG
        TARGET_LANG = space_dataset.TARGET_LANG
    
    if DATASET_NAME in ['wiki-gigaword', 'twitter']:
        if EMB_TYPE_SOURCE == 'BP':
            VS = space_dataset.VS
        
            data_path_source = f'../datasets/{DATASET_NAME}_{EMB_TYPE_SOURCE}_{SOURCE_DIM}_{VS//1000}K.d2v'
        else:
            data_path_source = f'../datasets/{DATASET_NAME}_{EMB_TYPE_SOURCE}_{SOURCE_DIM}.d2v'
            
        if EMB_TYPE_TARGET == 'BP':
            VS = space_dataset.VS
            data_path_target = f'../datasets/{DATASET_NAME}_{EMB_TYPE_TARGET}_{TARGET_DIM}_{VS//1000}K.d2v'

        else:
            data_path_target = f'../datasets/{DATASET_NAME}_{EMB_TYPE_TARGET}_{TARGET_DIM}.d2v'
            
        print(f'Loading {DATASET_NAME}_{EMB_TYPE_SOURCE}_{SOURCE_DIM} to source...')
        source_model = KeyedVectors.load(data_path_source)

        print(f'Loading {DATASET_NAME}_{EMB_TYPE_TARGET}_{TARGET_DIM} to target...')
        target_model = KeyedVectors.load(data_path_target)

        source_vectors = source_model.vectors[:]
        target_vectors = target_model.vectors[:]

    if DATASET_NAME == 'muse':
        VS = space_dataset.VS
        
        
        data_path_source = f'../datasets/{DATASET_NAME}_{SOURCE_LANG}_{EMB_TYPE_SOURCE}_{SOURCE_DIM}_{VS//1000}K.d2v'
        data_path_target = f'../datasets/{DATASET_NAME}_{TARGET_LANG}_{EMB_TYPE_TARGET}_{TARGET_DIM}_{VS//1000}K.d2v'

        print(f'Loading {DATASET_NAME}_{SOURCE_LANG}_{EMB_TYPE_SOURCE}_{SOURCE_DIM} to source...')
        source_model = KeyedVectors.load(data_path_source)
        
        print(f'Loading {DATASET_NAME}_{TARGET_LANG}_{EMB_TYPE_TARGET}_{TARGET_DIM} to target...')
        target_model = KeyedVectors.load(data_path_target)

        source_vectors = source_model.vectors[:]
        target_vectors = target_model.vectors[:]
        
    if DATASET_NAME == 'muse_multi':

        if EMB_TYPE_SOURCE == 'BP':
            VS = space_dataset.VS
            data_path = f'../datasets/muse_{SOURCE_LANG}({EMB_TYPE_SOURCE})({SOURCE_DIM})_{TARGET_LANG}({EMB_TYPE_TARGET})({TARGET_DIM})_{VS//1000}K.d2v'
        else:
            data_path = f'../datasets/muse_{SOURCE_LANG}({EMB_TYPE_SOURCE})({SOURCE_DIM})_{TARGET_LANG}({EMB_TYPE_TARGET})({TARGET_DIM}).d2v'
            
        source_target_model = KeyedVectors.load(data_path)

        source_model = KeyedVectors(vector_size=SOURCE_DIM)
        target_model = KeyedVectors(vector_size=TARGET_DIM)

        full_len = len(source_target_model.vectors)

        source_model.vectors = source_target_model.vectors[:full_len//2]
        target_model.vectors = source_target_model.vectors[full_len//2:]

        #source_model.index_to_key = source_target_model.index_to_key[:full_len//2]
        #target_model.index_to_key = source_target_model.index_to_key[full_len//2:]

        #for ix, key in enumerate(source_target_model.key_to_index.keys()):
        #    if ix < full_len//2:
        #        source_model.key_to_index[key] = ix

        #    else:
        #        target_model.key_to_index[key] = ix-full_len//2

        source_vectors = source_model.vectors[:]
        target_vectors = target_model.vectors[:]
        
    if DATASET_NAME in ['bone_marrow']:
        
        source_vectors, target_vectors, labels = load_biodata(data_path + '/', config)

    if EMB_TYPE_SOURCE != EMB_TYPE_TARGET:
        #words_source = source_model.index_to_key[:]
        #words_target = target_model.index_to_key[:]
#
        #common_words = set(words_source).intersection(words_target)
#
        #indices1 = [words_source.index(word) for word in common_words]
        #indices2 = [words_target.index(word) for word in common_words]
        #
        #source_vectors = source_vectors[indices1]
        #target_vectors = target_vectors[indices2]
        common_words = set(source_model.index_to_key).intersection(target_model.index_to_key)
        print(len(common_words))
        # Step 2: Get indices directly from KeyedVectors
        source_indices = [source_model.key_to_index[word] for word in common_words]
        target_indices = [target_model.key_to_index[word] for word in common_words]
        
        # Step 3: Select vectors using numpy indexing for faster extraction
        source_vectors = source_model.vectors[source_indices]
        target_vectors = target_model.vectors[target_indices]

        assert len(source_vectors) == len(target_vectors)
        
        
    if len(source_vectors) < N_MAX_SAMPLES:
            print('Setting the number of samples to the maximum possible equals to:', len(source_vectors))
            N_MAX_SAMPLES = len(source_vectors)
            config['dataset']['N_MAX_SAMPLES'] = N_MAX_SAMPLES

            
    source_vectors = source_vectors[:N_MAX_SAMPLES]
    target_vectors = target_vectors[:N_MAX_SAMPLES]
    
    source_vectors = torch.tensor(source_vectors).to(torch.float32)
    target_vectors = torch.tensor(target_vectors).to(torch.float32)
    
    return source_vectors, target_vectors
        
    
def get_samplers(config, source_vectors, target_vectors):

    space_dataset   = SimpleNamespace(**config['dataset']) 
    N_EVAL          = space_dataset.N_EVAL
    N_MAX_SAMPLES   = space_dataset.N_MAX_SAMPLES
    N_TRAIN_SAMPLES = space_dataset.N_TRAIN_SAMPLES
    N_TEST_SAMPLES  = space_dataset.N_TEST_SAMPLES * N_EVAL
    ALPHA           = space_dataset.ALPHA
    DEVICE          = space_dataset.DEVICE
    SHUFFLE         = space_dataset.SHUFFLE

    TRAIN_TYPE      = config['training']['TRAIN_TYPE']

    assert N_MAX_SAMPLES - N_TEST_SAMPLES >= N_TRAIN_SAMPLES, 'Reduce the number of train or test samples.'   
    
    random.seed(space_dataset.SEED)

    indices_train = random.sample(range(0, N_MAX_SAMPLES - N_TEST_SAMPLES), N_TRAIN_SAMPLES) 
    indices_test  = list(range(N_MAX_SAMPLES - N_TEST_SAMPLES, N_MAX_SAMPLES))

    indices_train_source = indices_train[:int(len(indices_train) * (0.5))]
    indices_train_target = indices_train[int(len(indices_train) * (0.5-ALPHA*0.5)): int(len(indices_train) * (1.0 - ALPHA*0.5))]

    indices_train_source = torch.tensor(indices_train_source).to(torch.int32)
    indices_train_target = torch.tensor(indices_train_target).to(torch.int32)
    
    indices_test = torch.tensor(indices_test).to(torch.int32)

    source_len = len(indices_train_source)
    target_len = len(indices_train_target)
    
    if source_len > target_len:
        indices_train_source = indices_train_source[source_len-target_len:]

    if source_len < target_len:
        indices_train_target = indices_train_target[:source_len]

    print('Source pairs...')
    print(len(indices_train_source[int(len(indices_train) * (0.5-ALPHA*0.5)):]))
    print(indices_train_source[int(len(indices_train) * (0.5-ALPHA*0.5)):])
        
    print('Target pairs...')
    print(len(indices_train_target[:int(len(indices_train) * ALPHA * 0.5)]))
    print(indices_train_target[:int(len(indices_train) * ALPHA * 0.5)])
    
    intersected_indices = list(set(indices_train_source.numpy()).intersection(indices_train_target.numpy()))
        
    if TRAIN_TYPE == 'continuous':
        batch_size_train = space_dataset.BATCH_SIZE_TRAIN
        batch_size_test = space_dataset.BATCH_SIZE_TEST
        
    if TRAIN_TYPE == 'discrete':
        batch_size_train = source_len
        batch_size_test = space_dataset.N_TEST_SAMPLES
    
    assert np.isclose(len(intersected_indices)/(N_TRAIN_SAMPLES/2), ALPHA, rtol=1e-2)
        
    #if var_sp.NORMALIZE_VECS is True:
    #    print('Normalizing source and target spaces...')
    #    source_vectors -= source_vectors.mean(axis=0)
    #    target_vectors -= target_vectors.mean(axis=0)
    #    
    #    source_vectors /= np.linalg.norm(source_vectors, axis=1)[:,None]
    #    target_vectors /= np.linalg.norm(target_vectors, axis=1)[:,None]

    train_source_dataset = TensorDataset(source_vectors[indices_train_source], indices_train_source)
    train_target_dataset = TensorDataset(target_vectors[indices_train_target], indices_train_target)

    train_source_loader = DataLoader(train_source_dataset, batch_size=batch_size_train, shuffle=SHUFFLE)
    train_target_loader = DataLoader(train_target_dataset, batch_size=batch_size_train, shuffle=SHUFFLE)
    
    train_source_sampler = LoaderSamplerTrain(train_source_loader, device=DEVICE)
    train_target_sampler = LoaderSamplerTrain(train_target_loader, device=DEVICE)
    
    testset      = TensorDataset(source_vectors[indices_test], target_vectors[indices_test], indices_test)
    testloader   = DataLoader(testset, batch_size=batch_size_test)
    test_sampler = LoaderSamplerTest(testloader, device=DEVICE)
    
    return source_vectors, target_vectors, train_source_sampler, train_target_sampler, test_sampler

def get_samplers_paired(config, source_vectors, target_vectors):

    space_dataset   = SimpleNamespace(**config['dataset']) 
    N_EVAL          = space_dataset.N_EVAL
    N_MAX_SAMPLES   = space_dataset.N_MAX_SAMPLES
    N_TRAIN_SAMPLES = space_dataset.N_TRAIN_SAMPLES
    N_TEST_SAMPLES  = space_dataset.N_TEST_SAMPLES * N_EVAL
    ALPHA           = space_dataset.ALPHA
    DEVICE          = space_dataset.DEVICE
    SHUFFLE         = space_dataset.SHUFFLE

    TRAIN_TYPE      = config['training']['TRAIN_TYPE']

    assert N_MAX_SAMPLES - N_TEST_SAMPLES >= N_TRAIN_SAMPLES, 'Reduce the number of train or test samples.'   
    
    random.seed(space_dataset.SEED)
    indices_train = random.sample(range(0, N_TRAIN_SAMPLES), N_TRAIN_SAMPLES) 
    indices_test  = range(N_MAX_SAMPLES - N_TEST_SAMPLES, N_MAX_SAMPLES)
    indices_test = torch.tensor(indices_test).to(torch.int32)
    
    start_shared_target = int(len(indices_train) * (0.5 - ALPHA*0.5))
    end_shared_target   = int(len(indices_train) * (1.0 - ALPHA*0.5))
    
    paired_indices          = indices_train[start_shared_target:int(len(indices_train) * (0.5))]
    unpaired_indices_source = indices_train[:start_shared_target]
    unpaired_indices_target = indices_train[start_shared_target:end_shared_target]
    
    indices_train_source = paired_indices + unpaired_indices_source
    indices_train_target = paired_indices + unpaired_indices_target
    
    source_len = len(indices_train_source)
    target_len = len(indices_train_target)
    
    if source_len > target_len:
        indices_train_source = indices_train_source[source_len-target_len:]
    if source_len < target_len:
        indices_train_target = indices_train_target[:source_len]
    
    intersected_indices = list(set(indices_train_source).intersection(indices_train_target))
    
    assert np.isclose(len(intersected_indices)/(N_TRAIN_SAMPLES/2), ALPHA, rtol=1e-2)
    
    
    
    #intersected_indices = list(set(indices_train_source.numpy()).intersection(indices_train_target.numpy()))
        
    if TRAIN_TYPE == 'continuous':
        batch_size_train = space_dataset.BATCH_SIZE_TRAIN
        batch_size_test = space_dataset.BATCH_SIZE_TEST
        
    if TRAIN_TYPE == 'discrete':
        batch_size_train = source_len
        batch_size_test = space_dataset.N_TEST_SAMPLES

    bs = batch_size_train
    n_batches = N_TRAIN_SAMPLES / bs / 2
    n_paired  = int(ALPHA * bs)
    n_unpaired = int((1 - ALPHA) * bs)
    if n_paired + n_unpaired < bs:
        n_unpaired += bs - (n_paired + n_unpaired)
    source_new = []
    target_new = []
    
    for i in range(int(n_batches)):
        new_s = paired_indices[i*n_paired:(i+1)*n_paired] + unpaired_indices_source[i*n_unpaired:(i+1)*n_unpaired]
        new_t = paired_indices[i*n_paired:(i+1)*n_paired] + unpaired_indices_target[i*n_unpaired:(i+1)*n_unpaired]
        new_t = list(np.random.permutation(new_t))
        intersected_indices = list(set(new_s).intersection(new_t))
        #print(len(intersected_indices)/(bs))
        source_new = source_new + new_s
        target_new = target_new + new_t

    for i in range(int(n_batches)):
        #intersected_indices = list(set(source_new[i*bs:(i+1)*bs]).intersection(target_new[i*bs:(i+1)*bs]))
        #print(source_new[i*bs:(i+1)*bs])
        #print(target_new[i*bs:(i+1)*bs])
        
        intersected_indices = list(set(source_new[i*bs:(i+1)*bs]).intersection(target_new[i*bs:(i+1)*bs]))
        
        #print(len(intersected_indices)/(bs))
        #print(ALPHA)
        assert np.isclose(len(intersected_indices)/(bs), ALPHA, rtol=1e-1)
        

    indices_train_source = torch.tensor(source_new[:]).to(torch.int32)
    indices_train_target = torch.tensor(target_new[:]).to(torch.int32)

    print('Source pairs...')
    print(len(indices_train_source[int(len(indices_train) * (0.5-ALPHA*0.5)):]))
    print(indices_train_source[int(len(indices_train) * (0.5-ALPHA*0.5)):])
        
    print('Target pairs...')
    print(len(indices_train_target[:int(len(indices_train) * ALPHA * 0.5)]))
    print(indices_train_target[:int(len(indices_train) * ALPHA * 0.5)])   
    
    train_source_dataset = TensorDataset(source_vectors[indices_train_source], indices_train_source)
    train_target_dataset = TensorDataset(target_vectors[indices_train_target], indices_train_target)

    train_source_loader = DataLoader(train_source_dataset, batch_size=batch_size_train, shuffle=SHUFFLE)
    train_target_loader = DataLoader(train_target_dataset, batch_size=batch_size_train, shuffle=SHUFFLE)
    
    train_source_sampler = LoaderSamplerTrain(train_source_loader, device=DEVICE)
    train_target_sampler = LoaderSamplerTrain(train_target_loader, device=DEVICE)
    
    testset      = TensorDataset(source_vectors[indices_test], target_vectors[indices_test], indices_test)
    testloader   = DataLoader(testset, batch_size=batch_size_test)
    test_sampler = LoaderSamplerTest(testloader, device=DEVICE)
    
    return source_vectors, target_vectors, train_source_sampler, train_target_sampler, test_sampler

def get_vectors_discrete_toy(config):
    
    var_sp = SimpleNamespace(**config['dataset']) 
    TOTAL_SAMPLES = var_sp.N_SAMPLES
    DEVICE = var_sp.DEVICE
    N_CLUSTERS_SOURCE = var_sp.N_CLUSTERS_SOURCE
    N_CLUSTERS_TARGET = var_sp.N_CLUSTERS_TARGET
    utils.seed_everything(var_sp.SEED)
    
    
    if var_sp.TOY_TYPE == 'toy_3d_2d':
        
        locs_src = distributions.fibonacci_sphere(N_CLUSTERS_SOURCE)
        scales_src = torch.full_like(locs_src, .1)
        source = distributions.GaussianMixture(locs_src, scales_src, device=DEVICE)

        locs_tgt = distributions.uniform_circle(N_CLUSTERS_TARGET)
        scales_tgt = torch.full_like(locs_tgt, .1)
        target = distributions.GaussianMixture(locs_tgt, scales_tgt, device=DEVICE)
        
        source_vectors, labels = source.sample_with_labels(TOTAL_SAMPLES)
        target_vectors = target.sample(TOTAL_SAMPLES)

    if var_sp.TOY_TYPE == 'toy_2d_3d':
        
        locs_src = distributions.uniform_circle(N_CLUSTERS_SOURCE)
        scales_src = torch.full_like(locs_src, .1)
        source = distributions.GaussianMixture(locs_src, scales_src, device=DEVICE)

        locs_tgt = distributions.fibonacci_sphere(N_CLUSTERS_TARGET) 
        scales_tgt = torch.full_like(locs_tgt, .1)
        target = distributions.GaussianMixture(locs_tgt, scales_tgt, device=DEVICE)
        
        source_vectors, labels = source.sample_with_labels(TOTAL_SAMPLES)
        target_vectors = target.sample(TOTAL_SAMPLES)
        
    if var_sp.NORMALIZE_VECS is True:
        print('Normalizing source and target spaces...')
        source_vectors -= source_vectors.mean(axis=0)
        target_vectors -= target_vectors.mean(axis=0)
        
        source_vectors /= np.linalg.norm(source_vectors, axis=1)[:,None]
        target_vectors /= np.linalg.norm(target_vectors, axis=1)[:,None]
    
    return source_vectors, target_vectors, labels

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

    labels = torch.tensor(labels).to(torch.int32)
    
    return source, target, labels
