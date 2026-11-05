
import faiss
import numpy as np
import torch
import torch.nn.functional as F

from scipy.linalg import sqrtm
from geomloss import SamplesLoss


def to_tensor(x):
    return x.detach().cpu().float() if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)


def normalize(x):
    return F.normalize(to_tensor(x), dim=1)


def build_index(vectors):
    vectors = normalize(vectors).contiguous().numpy()
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


@torch.no_grad()
def top_accuracies(prediction, labels, index, top_n=(1, 5, 10)):
    prediction = normalize(prediction).contiguous().numpy()
    labels = torch.as_tensor(labels, dtype=torch.long).cpu().numpy()

    if len(prediction) != len(labels) or len(prediction) == 0:
        raise ValueError('Prediction and labels must have equal nonzero lengths.')
    if np.any(labels < 0) or np.any(labels >= index.ntotal):
        raise ValueError('Ground-truth labels are outside the reference index.')

    k = min(max(top_n), index.ntotal)
    _, indices = index.search(prediction, k)

    return {f'Top@{n}': float(np.any(indices[:, :min(n, k)] == labels[:, None], axis=1).mean()) for n in top_n}


@torch.no_grad()
def cosine_similarity_gt(target, prediction):
    return F.cosine_similarity(target, prediction, dim=1).mean().item()


@torch.no_grad()
def foscttm(target, prediction):
    target, prediction = to_tensor(target), to_tensor(prediction)

    if target.shape != prediction.shape:
        raise ValueError('FOSCTTM requires paired target and prediction tensors.')

    distances = torch.cdist(target, prediction)
    diagonal  = distances.diagonal()
    forward   = (distances < diagonal[:, None]).float().mean()
    backward  = (distances < diagonal[None, :]).float().mean()

    return ((forward + backward) / 2).item()


@torch.no_grad()
def mmd(target, prediction, sigma=10.0):
    gamma = 1 / (2 * sigma**2)

    k_xx = torch.exp(-gamma * torch.cdist(target, target).square()).mean()
    k_yy = torch.exp(-gamma * torch.cdist(prediction, prediction).square()).mean()
    k_xy = torch.exp(-gamma * torch.cdist(target, prediction).square()).mean()

    return (k_xx + k_yy - 2 * k_xy).item()


@torch.no_grad()
def bw_uvp(target, prediction, eps=1e-6):
    mu1, mu2 = target.mean(0).numpy(), prediction.mean(0).numpy()
    cov1, cov2 = torch.cov(target.T).numpy(), torch.cov(prediction.T).numpy()

    covmean = sqrtm(cov1 @ cov2)

    if not np.isfinite(covmean).all():
        offset = np.eye(cov1.shape[0]) * eps
        covmean = sqrtm((cov1 + offset) @ (cov2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diag(covmean).imag, 0, atol=1e-3):
            raise ValueError('Significant imaginary component in covariance square root.')
        covmean = covmean.real

    distance = np.sum((mu1 - mu2)**2) + np.trace(cov1) + np.trace(cov2) - 2 * np.trace(covmean)
    return float(200 * distance / max(np.trace(cov1), eps))


@torch.no_grad()
def distortion(source, prediction, cost='cosine'):
    if cost == 'cosine':
        source_cost = 1 - source @ source.T
        target_cost = 1 - prediction @ prediction.T
        return F.mse_loss(source_cost, target_cost).item()

    elif cost == 'inner':
        x1, x2 = source.chunk(2)
        y1, y2 = prediction.chunk(2)
        return F.mse_loss(x1 @ x2.T, y1 @ y2.T).item()

    raise ValueError(f'Unknown distortion cost: {cost}')


@torch.no_grad()
def sinkhorn_divergence(target, prediction, blur=0.01):
    loss = SamplesLoss(loss='sinkhorn', p=2, blur=blur)
    return loss(target, prediction).item()


@torch.no_grad()
def compute_metrics(source, target, prediction, labels, labels_red, index_full, index_red, split='test',
                    top_n=(1, 5, 10), mmd_sigma=10.0, sinkhorn_blur=0.01, distortion_cost='cosine'):

    if split not in ['train', 'test']:
        raise ValueError(f'Unknown evaluation split: {split}')

    source     = to_tensor(source)
    target     = to_tensor(target)
    prediction = to_tensor(prediction)

    source_norm     = normalize(source)
    target_norm     = normalize(target)
    prediction_norm = normalize(prediction)

    if len(source_norm) != len(prediction_norm):
        raise ValueError('Source and prediction must have equal sample counts.')

    metrics = top_accuracies(prediction_norm, labels, index_full, top_n)
    metrics.update({f'{key}_red': value for key, value in top_accuracies(prediction_norm, labels_red, index_red, top_n).items()})

    metrics['mmd']                 = mmd(target_norm, prediction_norm, mmd_sigma)
    metrics['bw_uvp']              = bw_uvp(target_norm, prediction_norm)
    metrics['distortion']          = distortion(source_norm, prediction_norm, distortion_cost)
    metrics['sinkhorn_divergence'] = sinkhorn_divergence(target_norm, prediction_norm, sinkhorn_blur)

    if split == 'test':
        metrics['cossim_gt']      = cosine_similarity_gt(target_norm, prediction_norm)
        metrics['foscttm_norm']   = foscttm(target_norm, prediction_norm)
        metrics['foscttm_unnorm'] = foscttm(target, prediction)

    return metrics
