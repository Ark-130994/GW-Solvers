import itertools
import gensim
import torch
import torch.nn.functional as F
import numpy as np
import collections
from scipy import linalg, stats
from src import utils
import faiss
import wandb
import scipy

from functools import partial

from sklearn.neighbors import KNeighborsClassifier

def get_accuracy_plan(plan, test_dict, top_n=[1, 5, 10]):
    accs = {}
    for k in top_n:
        correct = 0
        n,m = plan.shape 
        for src_idx, tgt_idx in test_dict.items():

            knn = np.argpartition(plan[src_idx,:], -k)[-k:]
            knn_sort = knn[np.argsort(-plan[src_idx,knn])] 

            if set(knn_sort.numpy()).intersection(tgt_idx):
                correct +=1
                
        accs[f'Top@{k}'] = correct / len(test_dict)
    
    return accs

def get_close_words_plan(source_i2w, target_i2w, plan, k, verbose):
    plan.max(0)
    n_s, n_t = plan.shape
    best_match_src = plan.argmax(1) 
    best_match_tgt = plan.argmax(0)

    paired = []
    for i in range(n_s):
        m = best_match_src[i]
        if verbose:
            topk_idx = np.argpartition(plan[i,:], -k)[-k:]
            topk_idx_sort = topk_idx[np.argsort(-plan[i,topk_idx])] 
            print('{:20s} -> {}'.format(source_i2w[i],','.join([target_i2w[m] for m in topk_idx_sort])))
        if best_match_tgt[m] == i:
            paired.append((i,m))

    paired_toks = []
    if source_i2w and target_i2w:
        paired_toks = [(source_i2w[i],target_i2w[j]) for (i,j) in paired]
    else:
        paired_toks = paired
    paired_toks
    
    return paired_toks

def calculate_frechet_distance(mu1, sigma1,
                               mu2, sigma2,
                               eps=1e-6):

    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

@torch.no_grad()
def calc_frac_idx(x1_mat, x2_mat):
    """
    Returns fraction closer than true match for each sample (as an array)
    """
    fracs = []
    x = []
    nsamp = x1_mat.shape[0]
    rank=0
    for row_idx in range(nsamp):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(x1_mat[row_idx,:], x2_mat)), axis=1))
        true_nbr = euc_dist[row_idx]
        sort_euc_dist = sorted(euc_dist)
        rank =sort_euc_dist.index(true_nbr)
        frac = float(rank)/(nsamp -1)

        fracs.append(frac)
        x.append(row_idx+1)

    return fracs,x
    
@torch.no_grad()
def kernel_2(x, y, outer=False):
    x, y = x.flatten(1), y.flatten(1)
    if outer or x.size(0) != y.size(0):
        x, y = x[:, None], y[None]
    return torch.einsum("...i,...i->...", x, y)

@torch.no_grad()
def bw_uvp(y_sampled, target_dict):
    """
    Calculate the BW distance between an empirical distribution and a Gaussian.
    """

    target_cov = target_dict['cov']
    target_mean = target_dict['mean']
    
    moved_mean = y_sampled.mean(0).cpu().numpy()
    moved_cov = torch.cov(y_sampled.T).cpu().numpy()
    target_var = torch.trace(target_cov).cpu().item()
    
    bw_uvp_val = 2 * 100 * calculate_frechet_distance(
        moved_mean, moved_cov,
        target_mean.cpu().numpy(), target_cov.cpu().numpy()
    ) / target_var 
    return bw_uvp_val.item()

@torch.no_grad()
def inner_gw(x, y_sampled, kernel=kernel_2):

    return F.mse_loss(kernel(x.chunk(2)[0], x.chunk(2)[1], outer=True),
                      kernel(y_sampled.chunk(2)[0], y_sampled.chunk(2)[1], outer=True)).item()

@torch.no_grad()
def top_accuracies(y_sampled, target_vectors, labels, top_n=(1, 5, 10)): 
    top_n = np.array(top_n)

    L = y_sampled.shape[0]
    assert y_sampled.shape[0] == labels.shape[0]

    #most_similar_vals, most_similar_ix = target.most_similar_ix(predictions, top_n.max(), batch_size)
    vectors_space = target_vectors.cpu().numpy().copy()
    y_sampled_np = y_sampled.cpu().numpy()

    index = faiss.IndexFlatIP(y_sampled.shape[1])
    faiss.normalize_L2(vectors_space)
    faiss.normalize_L2(y_sampled_np)
    index.add(vectors_space) 

    k = max(top_n)   

    most_similar_vals, most_similar_ix = index.search(y_sampled_np, 10)

    out = np.full_like(labels, -1)
    idx, vals = np.where(most_similar_ix == labels.cpu().numpy()[:,None])
    out[idx] = vals

    v = dict()
    for p in top_n:
        v[f"Top@{p}"] = sum(i <= p-1 and i != -1 for i in out)

    top_accs_dict = {k:v[k]/L for k in v.keys()}

    return most_similar_vals[:,0].mean().item(), top_accs_dict

@torch.no_grad()
# This metric only works if it is paired data since it computes the index based on the index in the 
# dataset, so it is not suitable if we do not have paired data
def foscttm(y, y_sampled) -> float:
    d = scipy.spatial.distance_matrix(y, y_sampled)
    foscttm_x = (d < np.expand_dims(np.diag(d), axis=1)).mean(axis=1)
    foscttm_y = (d < np.expand_dims(np.diag(d), axis=0)).mean(axis=0)
    fracs = []
    for i in range(len(foscttm_x)):
        fracs.append((foscttm_x[i] + foscttm_y[i]) / 2)
    return np.mean(fracs).round(4)

@torch.no_grad()
def foscttm2(y, y_sampled):
    """
    Outputs average FOSCTTM measure (averaged over both domains)
    Get the fraction matched for all data points in both directions
    Averages the fractions in both directions for each data point
    """
    fracs1,xs = calc_frac_idx(y, y_sampled)
    fracs2,xs = calc_frac_idx(y_sampled, y)
    fracs = []
    for i in range(len(fracs1)):
        fracs.append((fracs1[i]+fracs2[i])/2)  
    return fracs

@torch.no_grad()
def cosine_similarity(y, y_sampled):
    return torch.cosine_similarity(y, y_sampled).mean().item()

#@torch.no_grad()
#def SVD_distance(self, x, y, y_sampled, labels, cost_model=None):
#    P_SVD_ = P_SVD(self.x_fixed, self.y_fixed).cpu().T
#    P = cost_model.compute_P(self.x_fixed, y_sampled).cpu()
#    return torch.norm(P_SVD_ - P, p='fro').item()

@torch.no_grad()
def label_transfer(y, y_sampled, labels):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(y, labels)
    y_pred = knn.predict(y_sampled)
    #np.savetxt("type1_predict.txt", type1_predict)
    count = 0
    for label1, label2 in zip(y_pred, labels):

        
        if label1 == label2:
            count += 1
    return count / len(y_sampled)

@torch.no_grad()
def compute_metrics(x, y, y_sampled, labels, target_vectors, metrics_dict):   

    x, y, y_sampled, labels = x.cpu(), y.cpu(), y_sampled.cpu(), labels.cpu()
    
    cossim_vals, top_accuracies_vals = top_accuracies(y_sampled, target_vectors, labels, top_n=(1, 5, 10))
    cossim_gt = cosine_similarity(y, y_sampled)
    inner_gw_val = inner_gw(x, y_sampled)
    foscttm_val   = foscttm(y, y_sampled)
    
    metrics_dict['Top@1'].append(top_accuracies_vals['Top@1'])
    metrics_dict['Top@5'].append(top_accuracies_vals['Top@5'])
    metrics_dict['Top@10'].append(top_accuracies_vals['Top@10'])
    
    metrics_dict['cossim_gt'].append(cossim_gt)
    metrics_dict['inner_gw'].append(inner_gw_val)
    metrics_dict['foscttm'].append(foscttm_val)
    
    return metrics_dict

@torch.no_grad()

def calc_frac_idx(x1_mat,x2_mat):
    """
    Returns fraction closer than true match for each sample (as an array)
    """
    fracs = []
    x = []
    nsamp = x1_mat.shape[0]
    rank=0
    for row_idx in range(nsamp):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(x1_mat[row_idx,:], x2_mat)), axis=1))
        true_nbr = euc_dist[row_idx]
        sort_euc_dist = sorted(euc_dist)
        rank =sort_euc_dist.index(true_nbr)
        frac = float(rank)/(nsamp -1)

        fracs.append(frac)
        x.append(row_idx+1)

    return fracs,x

@torch.no_grad()
def calc_domainAveraged_FOSCTTM(x1_mat, x2_mat):
    """
    Outputs average FOSCTTM measure (averaged over both domains)
    Get the fraction matched for all data points in both directions
    Averages the fractions in both directions for each data point
    """
    fracs1,xs = calc_frac_idx(x1_mat, x2_mat)
    fracs2,xs = calc_frac_idx(x2_mat, x1_mat)
    fracs = []
    for i in range(len(fracs1)):
        fracs.append((fracs1[i]+fracs2[i])/2)  
    return fracs
