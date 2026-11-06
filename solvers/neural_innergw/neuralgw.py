
import functools
import torch

from solvers.neural_innergw.models.p_init import constant_cost


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)


def computePotGrad(x, out):
    return torch.autograd.grad(out, x, grad_outputs=torch.ones_like(out), create_graph=True, retain_graph=True)[0]


class NeuralGW:

    def __init__(self, models, optimizers, n_samples, n_iters, reg=None, lambda_distortion=1e-3):

        self.cost_model   = models['cost']
        self.critic_model = models['critic']
        self.mover_model  = models['mover']

        self.cost_optimizer   = optimizers['cost']
        self.critic_optimizer = optimizers['critic']
        self.mover_optimizer  = optimizers['mover']

        self.n_samples         = n_samples
        self.n_iters           = n_iters
        self.reg               = reg
        self.lambda_distortion = lambda_distortion
        self.fixed_P           = None

    def setup(self, source_sampler, target_sampler):
        self.source_sampler = source_sampler
        self.target_sampler = target_sampler

    def sample(self, sampler):
        out = sampler.sample(self.n_samples)
        x = out[0] if isinstance(out, (tuple, list)) else out
        return x.to(next(self.mover_model.parameters()).device)

    def train(self):

        self.critic_model.train()
        self.mover_model.train()

        if self.cost_optimizer is not None:
            self.cost_model.train()

        cost_iters   = self.n_iters['cost']
        mover_iters  = self.n_iters['mover']
        critic_iters = self.n_iters['critic']

        cost_loss = torch.zeros((), device=next(self.mover_model.parameters()).device)
        
        for _ in range(len(self.source_sampler)):
            for i in range(cost_iters):

                if self.cost_optimizer is not None:
                    x_train = self.sample(self.source_sampler)
                    y_train = self.sample(self.target_sampler)

                    self.cost_optimizer.zero_grad()
                    h_x = self.mover_model(x_train)
                    cost_loss = self.cost_model(x_train, h_x).mean()
                    cost_loss.backward()
                    self.cost_optimizer.step()

                for j in range(critic_iters):

                    unfreeze(self.mover_model)
                    freeze(self.critic_model)

                    for k in range(mover_iters):
                        x_train = self.sample(self.source_sampler)
                        y_train = self.sample(self.target_sampler)

                        self.mover_optimizer.zero_grad()
                        h_x = self.mover_model(x_train)
                        mover_loss = self.cost_model(x_train, h_x).mean() - self.critic_model(h_x).mean()

                        mover_loss.backward()
                        self.mover_optimizer.step()

                    x_train = self.sample(self.source_sampler)
                    y_train = self.sample(self.target_sampler)

                    freeze(self.mover_model)
                    unfreeze(self.critic_model)

                    with torch.no_grad():
                        h_x = self.mover_model(x_train)

                    self.critic_optimizer.zero_grad()

                    if self.reg is not None:
                        y_train.requires_grad_(True)
                        critic_out = self.critic_model(y_train)
                        gradients = computePotGrad(y_train, critic_out)
                        grad_penalty = (gradients.view(gradients.size(0), -1).norm(2, dim=1) ** 2).mean()
                        critic_loss = self.critic_model(h_x).mean() - self.critic_model(y_train).mean() + self.reg * grad_penalty
                    else:
                        critic_loss = self.critic_model(h_x).mean() - self.critic_model(y_train).mean()

                    critic_loss.backward()
                    self.critic_optimizer.step()

        return {
            'cost_loss': cost_loss.detach(),
            'mover_loss': mover_loss.detach(),
            'critic_loss': critic_loss.detach(),
        }

    def set_fixed_P(self, P):
        device = next(self.mover_model.parameters()).device
        self.fixed_P = P.detach().to(device).clone()
        self.cost_model = functools.partial(constant_cost, P=self.fixed_P)
        self.cost_optimizer = None

    def set_trainable_cost(self, cost_model, cost_optimizer):
        self.cost_model = cost_model
        self.cost_optimizer = cost_optimizer
        self.fixed_P = None

    def predict(self, source):
        self.mover_model.eval()
        with torch.no_grad():
            device = next(self.mover_model.parameters()).device
            return self.mover_model(source.to(device))

    def get_checkpoint(self):
        return {
            'mover': self.mover_model.state_dict(),
            'critic': self.critic_model.state_dict(),
            'cost': self.cost_model.state_dict() if self.cost_optimizer is not None else None,
            'fixed_P': self.fixed_P,
            'mover_optimizer': self.mover_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'cost_optimizer': self.cost_optimizer.state_dict() if self.cost_optimizer is not None else None,
        }
