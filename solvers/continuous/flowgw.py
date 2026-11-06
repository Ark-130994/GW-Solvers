
import jax
import jax.numpy as jnp
import numpy as np
import torch


from solvers.discrete.flowgw import FlowGW as DiscreteFlowGW


class FlowGW(DiscreteFlowGW):

    def __init__(self, n_samples=256, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples

    def setup(self, source_sampler, target_sampler):
        self.source_sampler = source_sampler
        self.target_sampler = target_sampler

        source = source_sampler.sample(1)
        target = target_sampler.sample(1)

        if isinstance(source, (tuple, list)):
            source = source[0]
        if isinstance(target, (tuple, list)):
            target = target[0]

        super().setup(source, target)

    def train(self):
        source = self.source_sampler.sample(self.n_samples)
        target = self.target_sampler.sample(self.n_samples)

        if isinstance(source, (tuple, list)):
            source = source[0]
        if isinstance(target, (tuple, list)):
            target = target[0]

        x = source.detach().cpu().numpy() if torch.is_tensor(source) else np.asarray(source)
        y = target.detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target)

        with jax.default_device(self.device):
            self.x_train = jax.device_put(jnp.asarray(x), self.device)
            self.y_train = jax.device_put(jnp.asarray(y), self.device)

        return super().solve()
