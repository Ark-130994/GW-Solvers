
import os
import random
from types import SimpleNamespace

import jax
import moscot
import numpy as np
import sklearn.preprocessing as pp
import torch

from gensim.downloader import load
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from src import distributions, utils


class Sampler:
    def __init__(
        self, device='cuda',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass

class LoaderSamplerTrain(Sampler):
    def __init__(self, loader, device='cuda', train_type='continuous'):
        super(LoaderSamplerTrain, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
        self.train_type = train_type
        
    def __len__(self):
        return len(self.loader)
    
    def reset_sampler(self):
        self.it = iter(self.loader)
        
    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            if self.train_type == 'continuous':
                batch, labels = next(self.it)
            if self.train_type == 'discrete':
                batch, labels, labels_red = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)

        if self.train_type == 'continuous':
            return batch[:size].to(self.device), labels[:size].to(self.device)
        if self.train_type == 'discrete':
            return batch[:size].to(self.device), labels[:size].to(self.device), labels_red[:size].to(self.device)
        
class LoaderSamplerTest(Sampler):
    def __init__(self, loader, device='cuda', train_type='continuous'):
        super(LoaderSamplerTest, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
        self.train_type = train_type
        
    def __len__(self):
        return len(self.loader)
    
    def reset_sampler(self):
        self.it = iter(self.loader)
        
    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            if self.train_type == 'continuous':
                source_batch, target_batch, labels = next(self.it)
            if self.train_type == 'discrete':
                source_batch, target_batch, labels, labels_red = next(self.it)
                
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(source_batch) < size:
            return self.sample(size)
            
        if self.train_type == 'continuous':
            return source_batch[:size].to(self.device), target_batch[:size].to(self.device), labels[:size].to(self.device)

        if self.train_type == 'discrete':
            return source_batch[:size].to(self.device), target_batch[:size].to(self.device), labels[:size].to(self.device), labels_red[:size].to(self.device)

def load_vectors(config):

    dataset_cfg     = config['dataset']
    solver_type     = config['solver_type']

    data_path       = dataset_cfg['data_path']
    dataset_name    = dataset_cfg['dataset_name']
    n_max_samples   = dataset_cfg['n_max_samples']
    emb_type_source = dataset_cfg['emb_type_source']
    emb_type_target = dataset_cfg['emb_type_target']
    source_dim      = dataset_cfg['source_dim']
    target_dim      = dataset_cfg['target_dim']
    test_seed       = dataset_cfg['test_seed']

    if dataset_name in ['muse_multi', 'muse']:
        source_lang = dataset_cfg['source_lang']
        target_lang = dataset_cfg['target_lang']

    if dataset_name in ['wiki-gigaword', 'twitter']:
        if emb_type_source == 'BP':
            vs = dataset_cfg['vs']
            source_file = f'{dataset_name}_{emb_type_source}_{source_dim}_{vs//1000}K.d2v'
        else:
            source_file = f'{dataset_name}_{emb_type_source}_{source_dim}.d2v'

        if emb_type_target == 'BP':
            vs = dataset_cfg['vs']
            target_file = f'{dataset_name}_{emb_type_target}_{target_dim}_{vs//1000}K.d2v'
        else:
            target_file = f'{dataset_name}_{emb_type_target}_{target_dim}.d2v'

        data_path_source = os.path.join(data_path, source_file)
        data_path_target = os.path.join(data_path, target_file)

        print(f'Loading {dataset_name}_{emb_type_source}_{source_dim} to source...')
        source_model = KeyedVectors.load(data_path_source)

        print(f'Loading {dataset_name}_{emb_type_target}_{target_dim} to target...')
        target_model = KeyedVectors.load(data_path_target)

        source_vectors = source_model.vectors[:]
        target_vectors = target_model.vectors[:]

    if dataset_name == 'muse':
        vs = dataset_cfg['vs']

        source_file = f'{dataset_name}_{source_lang}_{emb_type_source}_{source_dim}_{vs//1000}K.d2v'
        target_file = f'{dataset_name}_{target_lang}_{emb_type_target}_{target_dim}_{vs//1000}K.d2v'

        data_path_source = os.path.join(data_path, source_file)
        data_path_target = os.path.join(data_path, target_file)

        print(f'Loading {dataset_name}_{source_lang}_{emb_type_source}_{source_dim} to source...')
        source_model = KeyedVectors.load(data_path_source)

        print(f'Loading {dataset_name}_{target_lang}_{emb_type_target}_{target_dim} to target...')
        target_model = KeyedVectors.load(data_path_target)

        source_vectors = source_model.vectors[:]
        target_vectors = target_model.vectors[:]

    if dataset_name == 'bone_marrow':
        source_vectors, target_vectors, labels = load_biodata(os.path.join(data_path, ''), config)

    if dataset_name == 'voxceleb':
        data_path_source = os.path.join(data_path, f'{dataset_name}_{emb_type_source}_dim_{source_dim}_148K.pt')
        data_path_target = os.path.join(data_path, f'{dataset_name}_{emb_type_target}_dim_{target_dim}_148K.pt')

        source_vectors = torch.load(data_path_source)
        target_vectors = torch.load(data_path_target)

    if dataset_name == 'voxceleb2':
        data_path_source = os.path.join(data_path, f'{dataset_name}_{emb_type_source}_dim_{source_dim}_600K.pt')
        data_path_target = os.path.join(data_path, f'{dataset_name}_{emb_type_target}_dim_{target_dim}_600K.pt')

        source_vectors = torch.load(data_path_source)
        target_vectors = torch.load(data_path_target)

    if emb_type_source != emb_type_target and dataset_name in ['wiki-gigaword', 'twitter', 'muse']:
        common_words = sorted(set(source_model.index_to_key).intersection(target_model.index_to_key))
        print(len(common_words))

        source_indices = [source_model.key_to_index[word] for word in common_words]
        target_indices = [target_model.key_to_index[word] for word in common_words]

        source_vectors = source_model.vectors[source_indices]
        target_vectors = target_model.vectors[target_indices]

        assert len(source_vectors) == len(target_vectors)

    if len(source_vectors) < n_max_samples:
        print('Setting the number of samples to the maximum possible equals to:', len(source_vectors))
        n_max_samples = len(source_vectors)
        dataset_cfg['n_max_samples'] = n_max_samples

    print('Original:', source_vectors.shape[0])

    indices_init   = np.random.default_rng(test_seed).permutation(source_vectors.shape[0])
    source_vectors = source_vectors[indices_init]
    target_vectors = target_vectors[indices_init]

    source_vectors = source_vectors[:n_max_samples]
    target_vectors = target_vectors[:n_max_samples]

    print('Final:', source_vectors.shape[0])

    source_vectors = torch.as_tensor(source_vectors, dtype=torch.float32)
    target_vectors = torch.as_tensor(target_vectors, dtype=torch.float32)

    return source_vectors, target_vectors


def indices_slicer(indices_train, alpha, n_train_samples):
    indices_train_source = indices_train[:int(len(indices_train) * (0.5))]
    indices_train_target = indices_train[int(len(indices_train) * (0.5-alpha*0.5)): int(len(indices_train) * (1.0 - alpha*0.5))]

    indices_train_source = torch.tensor(indices_train_source).to(torch.int32)
    indices_train_target = torch.tensor(indices_train_target).to(torch.int32)
    
    source_len = len(indices_train_source)
    target_len = len(indices_train_target)
    
    if source_len > target_len:
        indices_train_source = indices_train_source[source_len-target_len:]

    if source_len < target_len:
        indices_train_target = indices_train_target[:source_len]

    print('Source pairs...')
    print(len(indices_train_source[int(len(indices_train) * (0.5-alpha*0.5)):]))
    print(indices_train_source[int(len(indices_train) * (0.5-alpha*0.5)):])
        
    print('Target pairs...')
    print(len(indices_train_target[:int(len(indices_train) * alpha * 0.5)]))
    print(indices_train_target[:int(len(indices_train) * alpha * 0.5)])
    
    intersected_indices = list(set(indices_train_source.numpy()).intersection(indices_train_target.numpy()))

    assert np.isclose(len(intersected_indices)/(n_train_samples/2), alpha, rtol=1e-2)
    return indices_train_source, indices_train_target
    


def get_samplers_embeddings_discrete(config, source_vectors, target_vectors):

    dataset_cfg     = config['dataset']
    solver_type     = config['solver_type']
    device          = config['device']


    n_eval          = dataset_cfg['n_eval']
    n_max_samples   = dataset_cfg['n_max_samples']
    n_train_samples = dataset_cfg['n_train_samples']
    n_test_samples  = dataset_cfg['n_test_samples'] * n_eval
    alpha           = dataset_cfg['alpha']
    shuffle         = dataset_cfg['shuffle']
    seed            = dataset_cfg['seed']

    assert n_max_samples - n_test_samples >= n_train_samples, 'Reduce the number of train or test samples.'

    rng_train = random.Random(seed)

    indices_train = rng_train.sample(range(0, n_max_samples - n_test_samples), n_train_samples)
    indices_test  = list(range(n_max_samples - n_test_samples, n_max_samples))

    indices_train_red = list(range(0, n_train_samples))
    indices_test_red  = list(range(n_train_samples, n_train_samples + n_test_samples))

    indices_train_source_red, indices_train_target_red = indices_slicer(indices_train_red, alpha, n_train_samples)
    indices_test_red                                   = torch.tensor(indices_test_red).to(torch.int32)

    indices_train_source, indices_train_target = indices_slicer(indices_train, alpha, n_train_samples)
    indices_test                               = torch.tensor(indices_test).to(torch.int32)

    source_len = indices_train_source.shape[0]

    batch_size_train = source_len
    batch_size_test  = dataset_cfg['n_test_samples']

    source_vectors_red = source_vectors[indices_train]
    target_vectors_red = target_vectors[indices_train]

    source_vectors_train_red = source_vectors[indices_train_source]
    target_vectors_train_red = target_vectors[indices_train_target]

    source_vectors_test = source_vectors[indices_test]
    target_vectors_test = target_vectors[indices_test]

    train_source_dataset = TensorDataset(source_vectors_train_red, indices_train_source, indices_train_source_red)
    train_target_dataset = TensorDataset(target_vectors_train_red, indices_train_target, indices_train_target_red)

    testset = TensorDataset(source_vectors_test, target_vectors_test, indices_test, indices_test_red)

    source_vectors_out = torch.cat([source_vectors_red, source_vectors_test], dim=0)
    target_vectors_out = torch.cat([target_vectors_red, target_vectors_test], dim=0)

    assert source_vectors_out.shape[0] == n_train_samples + n_test_samples

    train_source_loader = DataLoader(train_source_dataset, batch_size=batch_size_train, shuffle=shuffle)
    train_target_loader = DataLoader(train_target_dataset, batch_size=batch_size_train, shuffle=shuffle)

    train_source_sampler = LoaderSamplerTrain(train_source_loader, device, solver_type)
    train_target_sampler = LoaderSamplerTrain(train_target_loader, device, solver_type)

    testloader   = DataLoader(testset, batch_size=batch_size_test)
    test_sampler = LoaderSamplerTest(testloader, device, solver_type)

    return source_vectors, target_vectors, source_vectors_out, target_vectors_out, train_source_sampler, train_target_sampler, test_sampler


def get_samplers_embeddings_continuous(config, source_vectors, target_vectors, paired=False):

    dataset_cfg     = config['dataset']
    solver_type     = config['solver_type']
    device          = config['device']


    n_eval          = dataset_cfg['n_eval']
    n_max_samples   = dataset_cfg['n_max_samples']
    n_train_samples = dataset_cfg['n_train_samples']
    n_test_samples  = dataset_cfg['n_test_samples'] * n_eval
    alpha           = dataset_cfg['alpha']
    shuffle         = dataset_cfg['shuffle']
    seed            = dataset_cfg['seed']

    if paired == True and shuffle == True:
        raise ValueError('Shuffling DataLoaders is not available for paired minibatches...')

    assert n_max_samples - n_test_samples >= n_train_samples, 'Reduce the number of train or test samples.'

    rng_train = random.Random(seed)
    indices_train = rng_train.sample(range(0, n_max_samples - n_test_samples), n_train_samples)
    indices_test  = list(range(n_max_samples - n_test_samples, n_max_samples))
    indices_test  = torch.tensor(indices_test).to(torch.int32)

    if paired:
        start_shared_target = int(len(indices_train) * (0.5 - alpha*0.5))
        end_shared_target   = int(len(indices_train) * (1.0 - alpha*0.5))

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

        assert np.isclose(len(intersected_indices)/(n_train_samples/2), alpha, rtol=1e-2)

        batch_size_train = dataset_cfg['batch_size_train']
        batch_size_test  = dataset_cfg['batch_size_test']

        bs = batch_size_train
        n_batches = n_train_samples / bs / 2
        n_paired  = int(alpha * bs)
        n_unpaired = int((1 - alpha) * bs)

        if n_paired + n_unpaired < bs:
            n_unpaired += bs - (n_paired + n_unpaired)

        source_new = []
        target_new = []

        for i in range(int(n_batches)):
            new_s = paired_indices[i*n_paired:(i+1)*n_paired] + unpaired_indices_source[i*n_unpaired:(i+1)*n_unpaired]
            new_t = paired_indices[i*n_paired:(i+1)*n_paired] + unpaired_indices_target[i*n_unpaired:(i+1)*n_unpaired]
            new_t = list(np.random.permutation(new_t))
            intersected_indices = list(set(new_s).intersection(new_t))
            source_new = source_new + new_s
            target_new = target_new + new_t

        for i in range(int(n_batches)):
            intersected_indices = list(set(source_new[i*bs:(i+1)*bs]).intersection(target_new[i*bs:(i+1)*bs]))
            assert np.isclose(len(intersected_indices)/(bs), alpha, rtol=1e-1), f'{len(intersected_indices)/(bs)}'

        indices_train_source = torch.tensor(source_new[:]).to(torch.int32)
        indices_train_target = torch.tensor(target_new[:]).to(torch.int32)

    else:
        indices_train_source, indices_train_target = indices_slicer(indices_train, alpha, n_train_samples)
        source_len = indices_train_source.shape[0]

        batch_size_train = dataset_cfg['batch_size_train']
        batch_size_test  = dataset_cfg['batch_size_test']

    source_vectors_out = torch.clone(source_vectors)
    target_vectors_out = torch.clone(target_vectors)

    train_source_dataset = TensorDataset(source_vectors[indices_train_source], indices_train_source)
    train_target_dataset = TensorDataset(target_vectors[indices_train_target], indices_train_target)
    testset              = TensorDataset(source_vectors[indices_test], target_vectors[indices_test], indices_test)

    train_source_loader = DataLoader(train_source_dataset, batch_size=batch_size_train, shuffle=shuffle)
    train_target_loader = DataLoader(train_target_dataset, batch_size=batch_size_train, shuffle=shuffle)

    train_source_sampler = LoaderSamplerTrain(train_source_loader, device, solver_type)
    train_target_sampler = LoaderSamplerTrain(train_target_loader, device, solver_type)

    testloader   = DataLoader(testset, batch_size=batch_size_test)
    test_sampler = LoaderSamplerTest(testloader, device, solver_type)

    return source_vectors_out, target_vectors_out, train_source_sampler, train_target_sampler, test_sampler


def get_samplers_toy(config):

    dataset_cfg     = config['dataset']
    device          = config['device']

    torch.manual_seed(dataset_cfg['seed'])

    # Source distribution
    toy_type          = dataset_cfg['toy_type']
    n_clusters_source = dataset_cfg['n_clusters_source']
    n_clusters_target = dataset_cfg['n_clusters_target']

    if toy_type == 'toy_3d_2d':
        locs_src = distributions.fibonacci_sphere(n_clusters_source)
    elif toy_type in ['toy_2d_2d', 'toy_2d_3d']:
        locs_src = distributions.uniform_circle(n_clusters_source)
    else:
        raise ValueError(f'Unknown toy type: {toy_type}')

    scales_src = torch.full_like(locs_src, .1)
    source_sampler = distributions.GaussianMixture(locs_src, scales_src, device=device)

    # Target distribution
    if toy_type == 'toy_2d_3d':
        locs_tgt = distributions.fibonacci_sphere(n_clusters_target)
    else:
        locs_tgt = distributions.uniform_circle(n_clusters_target)

    scales_tgt = torch.full_like(locs_tgt, .1)
    target_sampler = distributions.GaussianMixture(locs_tgt, scales_tgt, device=device)

    return source_sampler, target_sampler



def load_model(name, emb_dim):
    full_name = 'glove-' + name + '-' + str(emb_dim)
    model = load(full_name)
    return model


from sklearn.decomposition import PCA

def load_biodata(data_path, config):

    fused_dim       = config['dataset_cfg']['FUSED_DIM']
    n_train_samples = config['dataset_cfg']['N_TRAIN_SAMPLES']
    n_test_samples  = config['dataset_cfg']['N_TEST_SAMPLES'] * config['dataset_cfg']['N_EVAL']
    seed            = config['dataset_cfg']['SEED']
    
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
