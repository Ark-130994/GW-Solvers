import torch
import itertools

import numpy as np

from gensim.downloader import load
import os
from gensim.models import KeyedVectors
import random
from torch.utils.data import TensorDataset, DataLoader
from src import utils
from types import SimpleNamespace 
import moscot
import sklearn.preprocessing as pp
import scanpy as sc


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

def get_indices(data_path, config):
    
    var_sp = SimpleNamespace(**config['dataset']) 
    N_EVAL = config['training']['N_EVAL']
    random.seed(var_sp.SEED)
    
    if var_sp.DATASET_NAME in ['wiki-gigaword', 'twitter']:
        dataset_name = var_sp.DATASET_NAME#[6:]
        source_model, target_model = load_glove(dataset_name, data_path, var_sp.SOURCE_DIM, var_sp.TARGET_DIM)
        
        source_class = embeddings(source_model)
        target_class = embeddings(target_model)
        
        if var_sp.TRAIN_TYPE == 'continuous':
            train_words = int(var_sp.MAX_WORDS * var_sp.TEST_SPLIT)
            random_indices = random.sample(range(0, 400000), var_sp.MAX_WORDS)
            random_indices_train = torch.arange(0, train_words)
            random_indices_test = torch.arange(train_words, var_sp.MAX_WORDS)
            
        if var_sp.TRAIN_TYPE == 'discrete' and var_sp.VARIANT != 'v3':
            TOTAL_SAMPLES = var_sp.N_SAMPLES_DISCRETE + var_sp.N_SAMPLES_CONTINUOUS * N_EVAL
            random_indices = random.sample(range(0, 400000), TOTAL_SAMPLES)
            random_indices_train = torch.arange(0, var_sp.N_SAMPLES_DISCRETE)
            random_indices_test = torch.arange(var_sp.N_SAMPLES_DISCRETE, TOTAL_SAMPLES)
            
        if var_sp.TRAIN_TYPE == 'discrete' and var_sp.VARIANT == 'v3':
            TOTAL_SAMPLES = 300000 + var_sp.N_SAMPLES_CONTINUOUS * N_EVAL
            random_indices = random.sample(range(0, 400000), TOTAL_SAMPLES)
            random_indices_train = torch.arange(0, 300000)
            random_indices_test = torch.arange(300000, TOTAL_SAMPLES)
        
        random_words = [source_class.i2w[ix] for ix in random_indices]
        
        source_class.restrict(random_words)
        target_class.restrict(random_words)

        return source_class.vectors, target_class.vectors, random_indices_train, random_indices_test
        
    if var_sp.DATASET_NAME in ['bone_marrow']:
        
        source, target, labels = load_biodata(data_path+'/')

        if var_sp.TRAIN_TYPE == 'continuous':
            total_samples = len(source)
            train_samples = int(var_sp.TEST_SPLIT * total_samples)
            random_indices = random.sample(range(0, total_samples), train_samples)
            random_indices_train = torch.arange(0, train_samples)
            random_indices_test = torch.arange(train_samples, total_samples)
            
        if var_sp.TRAIN_TYPE == 'discrete':
            TOTAL_SAMPLES = var_sp.N_SAMPLES_DISCRETE + var_sp.N_SAMPLES_CONTINUOUS*N_EVAL
            random_indices = random.sample(range(0, len(source)), TOTAL_SAMPLES)
            random_indices_train = torch.arange(0, var_sp.N_SAMPLES_DISCRETE)
            random_indices_test = torch.arange(var_sp.N_SAMPLES_DISCRETE, TOTAL_SAMPLES)
        
        return source, target, random_indices_train, random_indices_test
        
    
def get_samplers(source_vectors, target_vectors, random_indices_train, random_indices_test, config):
    
    var_sp = SimpleNamespace(**config['dataset']) 

                                       
    random_indices_train_source = random_indices_train[:int(len(random_indices_train) * (0.5))]
    random_indices_train_target = random_indices_train[int(len(random_indices_train) * (0.5-var_sp.ALPHA*0.5)): int(len(random_indices_train) * (0.5 + 0.5 - var_sp.ALPHA*0.5))]
    
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

        train_words = int(len(source_vectors) * var_sp.TEST_SPLIT)/2
        assert np.isclose(len(intersected_indices)/(train_words), var_sp.ALPHA, rtol=1e-2)
        
        
    if var_sp.TRAIN_TYPE == 'discrete':
        batch_size_train = source_len
        batch_size_test = var_sp.N_SAMPLES_CONTINUOUS#len(random_indices_test)

        assert np.isclose(len(intersected_indices)/(var_sp.N_SAMPLES_DISCRETE/2), var_sp.ALPHA, rtol=1e-2)
        
    
    trainset = TensorDataset(source_vectors[random_indices_train_source], target_vectors[random_indices_train_target], random_indices_train_source)
    trainloader = DataLoader(trainset, batch_size=batch_size_train)
    train_sampler = LoaderSampler(trainloader, device=var_sp.DEVICE)
    
    if len(random_indices_test) != 0:
        testset = TensorDataset(source_vectors[random_indices_test], target_vectors[random_indices_test], random_indices_test)
        testloader = DataLoader(testset, batch_size=batch_size_test)
        test_sampler = LoaderSampler(testloader, device=var_sp.DEVICE)
    else: 
        test_sampler = None
    
    return source_vectors, target_vectors, train_sampler, test_sampler
    

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

def load_biodata(data_path):
    
    adata_atac = moscot.datasets.bone_marrow(path=data_path, rna=False)
    adata_rna = moscot.datasets.bone_marrow(path=data_path, rna=True)

    adata_source = adata_atac.copy()
    adata_target = adata_rna.copy()
    
    n_cells_source = len(adata_atac)

    n_samples_train = n_cells_source

    source_q = pp.normalize(adata_source.obsm["ATAC_lsi_red"], norm="l2")
    target_q = adata_target.obsm["GEX_X_pca"]

    labels_dict = {CT:ix for ix, CT in enumerate(set(adata_atac.obs['cell_type'].values))}
    labels = [labels_dict[val] for val in adata_atac.obs['cell_type'].values]
    labels = torch.tensor(labels)
    
    source = np.copy(source_q)
    target = np.copy(target_q)

    source = torch.tensor(source).to(torch.float32)
    target = torch.tensor(target).to(torch.float32)
    labels = torch.tensor(labels).to(torch.int32)

    return source, target, labels
