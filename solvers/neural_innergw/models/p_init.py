
import torch
import torch.nn.functional as F


def constant_cost(x, y, P):
    Px = x @ P.T
    return F.mse_loss(Px, y) - (Px ** 2).mean() - (y ** 2).mean()


def estimate_gaussian_params(sampler, n_samples, batch_size):

    samples = []

    for i in range(0, n_samples, batch_size):
        n = min(batch_size, n_samples - i)
        sample = sampler.sample(n)

        if isinstance(sample, (tuple, list)):
            sample = sample[0]

        samples.append(sample)

    samples = torch.cat(samples)
    mean = samples.mean(dim=0)
    centered = samples - mean
    cov = centered.T @ centered / (len(samples) - 1)

    return mean, cov


def compute_optimal_P(m_s, m_t, Sigma_s, Sigma_t, p, q, lr=0.1, n_iter=200):

    if p < q:
        raise ValueError('Gaussian P initialization requires source_dim >= target_dim.')

    eig_s, U_s = torch.linalg.eigh(Sigma_s)
    eig_t, U_t = torch.linalg.eigh(Sigma_t)

    idx_s = torch.argsort(eig_s, descending=True)
    idx_t = torch.argsort(eig_t, descending=True)

    U_s = U_s[:, idx_s]
    U_t = U_t[:, idx_t]
    eig_s = eig_s[idx_s]
    eig_t = eig_t[idx_t]

    A = U_s @ torch.diag(torch.sqrt(eig_s.clamp_min(0))) @ U_s.T
    B = U_t @ torch.diag(torch.sqrt(eig_t.clamp_min(0))) @ U_t.T
    M = torch.outer(m_s, m_t)

    K = U_s[:, :q] @ U_t.T

    for _ in range(n_iter):

        grad_E = 2 * A.T @ (A @ K @ B + M) @ B.T

        sym = (K.T @ grad_E + (K.T @ grad_E).T) / 2
        grad_R = grad_E - K @ sym

        K_new = K + lr * grad_R
        K, _ = torch.linalg.qr(K_new)

    C = A @ K @ B + M
    P_opt = (p ** 0.5) * C.T / torch.norm(C, p='fro')

    return P_opt
