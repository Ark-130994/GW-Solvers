import torch
import numpy as np

from collections import Counter
from typing import List, Any

from torch import Tensor, Size
from torch.types import Device, _size

import sklearn.datasets as datasets
from sklearn.datasets import make_blobs

from scipy.stats import special_ortho_group
from sklearn.datasets import make_swiss_roll

from scipy.stats import beta


def to_size(shape) -> Size:
    if isinstance(shape, Size):
        return shape
    return Size(shape)

def to_tensor(data: Any, device: Device) -> Tensor:
    if isinstance(data, Tensor):
        return data.clone().detach().to(device)
    return torch.tensor(data, device=device)

class Distribution:
    def __init__(self, event_shape: _size, device: Device = None):
        self.event_shape = to_size(event_shape)
        self.device = device

    def to(self, device: Device):
        self.device = device

    def sample(self, sample_shape: _size) -> Tensor:
        raise NotImplementedError()

    def sample_with_labels(self, sample_shape: _size):
        return self.sample(sample_shape), None

    @property
    def mean(self) -> Tensor:
        raise NotImplementedError()


class CompositeDistribution(Distribution):
    def __init__(self, event_shape: _size, labels: Any,
                 device: Device = None):
        self.labels = to_tensor(labels, device)
        self._component_ix = {
            c.item(): ix for ix, c in enumerate(self.labels)
        }
        super().__init__(event_shape, device)

    def to(self, device: Device):
        super().to(device)
        self.labels.to(device)

    def sample(self, sample_shape: _size = ()):
        samples, _ = self.sample_with_labels(sample_shape)
        return samples

    def sample_with_labels(self, sample_shape: _size):
        raise NotImplementedError()

    def sample_from_components(self,
                               sample_shape: _size,
                               components: Any = None):
        sample_shape = to_size(sample_shape)
        if components is None:
            components = self.labels.tolist()
        elif not components:
            return self.sample(sample_shape)
        return self._sample_from_nonempty_components(sample_shape, components)

    def _sample_from_nonempty_components(self, sample_shape: _size,
                                         components: Any):
        raise NotImplementedError()


class Uniform(Distribution):
    def __init__(self, low: Any, high: Any, *, device: Device = None):
        self.low = to_tensor(low, device)
        self.high = to_tensor(high, device)
        super().__init__(self.low.size(), device)

    def to(self, device: Device):
        super().to(device)
        self.low.to(device)
        self.high.to(device)

    @torch.no_grad()
    def sample(self, sample_shape: _size = ()):
        sample_shape = to_size(sample_shape)
        random = torch.rand(sample_shape + self.event_shape,
                            dtype=self.low.dtype, device=self.device)
        return self.low + random * (self.high - self.low)

    @property
    def mean(self):
        return (self.high - self.low) / 2


class Normal(Distribution):
    def __init__(self, loc: Any, scale: Any, *, device: Device = None):
        self.loc = to_tensor(loc, device)
        self.scale = to_tensor(scale, device)
        super().__init__(self.loc.size(), device)

    def to(self, device: Device):
        super().to(device)
        self.loc.to(device)
        self.scale.to(device)

    @torch.no_grad()
    def sample(self, sample_shape: _size = ()):
        sample_shape = to_size(sample_shape)
        eps = torch.randn(sample_shape + self.event_shape,
                          dtype=self.loc.dtype, device=self.device)
        return self.loc + eps * self.scale

    @property
    def mean(self):
        return self.loc


class MultivariateNormal(Normal):
    def __init__(self, loc: Any, covariance: Any, *,
                 device: Device = None):
        self.covariance = to_tensor(covariance, device)
        scale = torch.linalg.cholesky(self.covariance)
        super().__init__(loc, scale, device=device)

    @torch.no_grad()
    def sample(self, sample_shape: _size = ()):
        sample_shape = to_size(sample_shape)
        eps = torch.randn(sample_shape + self.event_shape,
                          dtype=self.loc.dtype, device=self.device)
        return self.loc + (self.scale @ eps.unsqueeze(-1)).squeeze(-1)


class DiscreteMixture(CompositeDistribution):
    def __init__(self,
                 components: List[Distribution],
                 probs: Any,
                 labels: Any,
                 *,
                 device: Device = None):
        self.components = components
        for component in self.components:
            component.to(device)
        self.probs = to_tensor(probs, device)
        event_shape = components[0].event_shape
        super().__init__(event_shape, labels, device)

    def to(self, device: Device):
        super().to(device)
        for component in self.components:
            component.to(device)
        self.probs.to(device)

    def sample_with_labels(self, sample_shape: _size = ()):
        #print(sample_shape)
        sample_shape = (sample_shape,)
        sample_shape = to_size(sample_shape)
        #print(sample_shape)
        
        
        indices = torch.multinomial(self.probs, sample_shape.numel(),
                                    replacement=True)
        samples = torch.empty(sample_shape.numel(), *self.event_shape,
                              device=self.device)
        for index, count in Counter(indices.tolist()).items():
            samples[indices == index] = self.components[index].sample((count,))
        samples = samples.view(sample_shape + self.event_shape)

        labels = self.labels[indices].view(sample_shape)
        return samples, labels

    def _sample_from_nonempty_components(self, sample_shape: _size,
                                         components: Any):
        return torch.stack([
            self.components[self._component_ix[label]].sample(sample_shape)
            for label in components
        ])

    @property
    def mean(self):
        means = torch.stack([c.mean for c in self.components], -1)
        return torch.sum(self.probs * means, -1)




class GaussianMixture(DiscreteMixture):
    def __init__(self, locs: Any, scales: Any, *,
                 probs: Any = None,
                 device: Device = None):
        components = [Normal(loc, scale, device=device)
                      for loc, scale in zip(locs, scales)]
        labels = torch.arange(0, locs.size(0), device=device)
        if probs is None:
            probs = torch.ones_like(labels) / locs.size(0)
        super().__init__(components, probs, labels, device=device)


def to_composite(dist: Distribution) -> CompositeDistribution:
    return DiscreteMixture([dist], [1.], [0], device=dist.device)

def fibonacci_sphere(n_samples: int) -> Tensor:
    phi = torch.pi * (3. - np.sqrt(5.))  # golden angle in radians
    y = 1 - 2 * torch.arange(n_samples) / (n_samples - 1)
    theta = phi * torch.arange(n_samples)  # golden angle increment
    radius = torch.sqrt(1 - y * y)
    return torch.column_stack([theta.cos() * radius, y, theta.sin() * radius])

def uniform_circle(n_samples):
    theta = 2 * torch.pi * torch.arange(n_samples) / n_samples
    return torch.column_stack([theta.cos(), theta.sin()])

class MoonsDistribution(CompositeDistribution):
    def __init__(self, noise: float = .1, *, device: Device = None):
        self.noise = noise
        super().__init__((2,), (0, 1), device)

    def sample_with_labels(self, sample_shape: _size = ()):
        sample_shape = to_size(sample_shape)
        samples, labels = datasets.make_moons(sample_shape.numel(), noise=self.noise)
        samples = torch.tensor(samples,
                               device=self.device,
                               dtype=torch.float32).view(sample_shape + self.event_shape)
        labels = torch.tensor(labels, device=self.device).view(sample_shape)

        return samples, labels

    def _sample_from_nonempty_components(self, sample_shape: _size, components):
        sample_shape = to_size(sample_shape)
        n_samples = torch.zeros_like(self.labels)
        n_samples[to_tensor(components, self.device)] = sample_shape.numel()
        samples, _ = datasets.make_moons(n_samples.numpy(), noise=self.noise)
        return torch.tensor(samples, dtype=torch.float32)
    
class SwissRollSampler():
    def __init__(self, dim, noise, device):
        self.dim = dim
        self.noise = noise
        self.device = device
        
    def sample(self, n_samples):
        n_samples = n_samples[0]
        if self.dim == 3:
            X, _ = make_swiss_roll(n_samples, noise=self.noise)
            X = X.astype('float32')
            X[:, 1] *= 0.05
            
        elif self.dim ==2:
            X, _ = make_swiss_roll(n_samples, noise=self.noise)
            X = X.astype('float32')[:, [0, 2]]
        
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        return X
    
    def sample_with_labels(self, n_samples):
        n_samples = n_samples[0]
        if self.dim == 3:
            X, labels = make_swiss_roll(n_samples, noise=self.noise)
            X = X.astype('float32')
            X[:, 1] *= 0.05
        elif self.dim ==2:
            X, labels = make_swiss_roll(n_samples, noise=self.noise)
            X = X.astype('float32')[:, [0, 2]]
        
        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        labels = torch.tensor(labels, device='cpu')#.reshape(-1)
        return X, labels   
    
def trim(mask: torch.Tensor):
    i, j = torch.where(mask)
    return mask[i.min():i.max(), j.min():j.max()]

def image_rbf(image: Any, scale: Any, *,
              center: Any = None,
              sigma: float = .01,
              num_atoms: int = 1000,
              device: Device = None):
    image_tensor = trim(to_tensor(image, device)).flip(0)
    density = image_tensor / image_tensor.sum()
    nonzero = torch.stack(torch.where(density), dim=-1)
    scale = to_tensor(scale, device)
    size = to_tensor(image_tensor.size(), device)

    if center is None:
        center = torch.zeros(2)
    center = to_tensor(center, device)

    num_atoms = min(num_atoms, torch.count_nonzero(density))

    ix = torch.multinomial(density[density != 0], num_atoms)

    locs = center + scale * (2 * nonzero[ix] / size - 1).flip(1)
    scales = torch.empty_like(locs, device=device).fill_(sigma)
    probs = density[density != 0][ix]

    return GaussianMixture(locs, scales, probs=probs, device=device)

def make_spiral(n_samples, noise=.5):
    n = np.sqrt(np.random.rand(n_samples,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_samples,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_samples,1) * noise
    return np.array(np.hstack((d1x,d1y)))

def create_ds(n_samples,spiral=False):
    if spiral:
        centers = [[5, 5], [-1, -1], [1, -1]]
        X, labels_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.2,
                                    random_state=0)
        Xs = X[labels_true==0,:]
        Xs = make_spiral(n_samples=n_samples, noise=1)
        
        centers = [[2, 2], [0, 0], [1, -1]]
        X, labels_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.2,
                                    random_state=0)
        Xt = X[labels_true==0,:]
        Xt = make_spiral(n_samples=n_samples, noise=1)
        
        A = special_ortho_group.rvs(2,random_state=0)
        
        get_rot= lambda theta : np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    
        A = get_rot(np.pi)
    
        Xt = np.dot(Xs,A)*.5+10*np.random.rand(1,2)+10
    else:
        Xs=(10-3)*np.random.rand(n_samples,2)+3
        Xt=(8-1)*np.random.rand(n_samples,2)+1
    
    return Xs,Xt

class circle_distribution:
    def __init__(self, alpha=7.0, beta_param=5.0, seed=None, device='cpu'):
        self.alpha = alpha
        self.beta_param = beta_param
        self.seed = seed
        self.device = device
            
    def sample(self, n_samples):
        
        if self.seed is not None:
            np.random.seed(self.seed)
            
        r = beta.rvs(self.alpha, self.beta_param, size=n_samples, random_state=self.seed)
        circle = np.random.uniform(0, 1, n_samples)
        
        theta = 2.0 * np.pi * circle
        
        a = r * np.cos(theta)
        b = r * np.sin(theta)

        a = torch.tensor(a).to(torch.float32).to(self.device)
        b = torch.tensor(b).to(torch.float32).to(self.device)
        
        return torch.vstack([a, b]).T
