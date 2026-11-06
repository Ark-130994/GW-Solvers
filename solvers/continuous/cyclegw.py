
import torch

from functools import partial
from geomloss import SamplesLoss


def norm_square(X1, X2):
    batch_size_1 = X1.size(0)
    batch_size_2 = X2.size(0)

    expand_x1 = X1.expand(batch_size_2, batch_size_1, -1)
    expand_x1 = torch.transpose(expand_x1, 0, 1)
    expand_x2 = X2.expand(batch_size_1, batch_size_2, -1)
    square = torch.sum(torch.square(expand_x1 - expand_x2), 2)

    return square


def gaussian_kernel(sigmas, X1, X2):
    square = norm_square(X1, X2)
    val = 0
    for sigma in sigmas:
        val += torch.exp(-square / sigma)
    return val


def distortion(X, FX, Y, GY):
    delta1 = torch.mean(torch.abs(norm_square(X, X) - norm_square(FX, FX)))
    delta2 = torch.mean(torch.abs(norm_square(Y, Y) - norm_square(GY, GY)))
    delta3 = torch.mean(torch.abs(norm_square(X, GY) - norm_square(FX, Y)))
    return delta1 + delta2 + 2 * delta3


class CycleGW:

    def __init__(self, models, optimizers, eps, sigmas, reg, kernel_type='energy_guided', take_median=True, n_samples=256):

        self.F_model = models['F_model']
        self.G_model = models['G_model']

        self.F_optimizer = optimizers['F_model']
        self.G_optimizer = optimizers['G_model']

        self.eps         = eps
        self.sigmas      = sigmas
        self.reg         = reg
        self.kernel_type = kernel_type
        self.take_median = take_median
        self.n_samples   = n_samples

    def setup(self, source_sampler, target_sampler):
        self.source_sampler = source_sampler
        self.target_sampler = target_sampler
        self.is_embeddings = hasattr(source_sampler, 'loader')

    
    def train(self):

        sigmas_x = self.sigmas
        sigmas_y = self.sigmas
        reg_x = self.reg
        reg_y = self.reg

        self.F_model.train()
        self.G_model.train()

        x_train = self.source_sampler.sample(self.n_samples)
        y_train = self.target_sampler.sample(self.n_samples)

        if isinstance(x_train, (tuple, list)):
            x_train = x_train[0]
        if isinstance(y_train, (tuple, list)):
            y_train = y_train[0]

        if self.kernel_type == 'gaussian':
            if self.take_median and self.is_embeddings:
                median_x = torch.median(norm_square(x_train, x_train))
                median_y = torch.median(norm_square(y_train, y_train))

                sigmas_x = [sigma_x * median_x for sigma_x in sigmas_x]
                sigmas_y = [sigma_y * median_y for sigma_y in sigmas_y]

            kernel_x = partial(gaussian_kernel, sigmas=sigmas_x)
            kernel_y = partial(gaussian_kernel, sigmas=sigmas_y)

        elif self.kernel_type == 'energy_guided':
            kernel_x = lambda X1, X2: torch.norm(X1 - X2)
            kernel_y = lambda X1, X2: torch.norm(X1 - X2)

        elif self.kernel_type == 'sinkhorn':
            kernel_x = SamplesLoss(loss='sinkhorn', p=2, blur=0.01)
            kernel_y = SamplesLoss(loss='sinkhorn', p=2, blur=0.01)

        else:
            raise ValueError(f'Unknown kernel type: {self.kernel_type}')

        FX = self.F_model(x_train)
        GY = self.G_model(y_train)

        if self.kernel_type in ['energy_guided', 'gaussian']:
            loss_x = torch.mean(kernel_x(X1=x_train, X2=x_train)) + torch.mean(kernel_x(X1=GY, X2=GY)) - 2 * torch.mean(kernel_x(X1=GY, X2=x_train))
            loss_y = torch.mean(kernel_y(X1=y_train, X2=y_train)) + torch.mean(kernel_y(X1=FX, X2=FX)) - 2 * torch.mean(kernel_y(X1=FX, X2=y_train))

        if self.kernel_type == 'sinkhorn':
            loss_x = torch.mean(kernel_x(GY, x_train))
            loss_y = torch.mean(kernel_y(FX, y_train))

        loss_function = reg_x * loss_x + reg_y * loss_y + self.eps * distortion(x_train, FX, y_train, GY)

        self.F_optimizer.zero_grad()
        self.G_optimizer.zero_grad()

        loss_function.backward()
        self.F_optimizer.step()
        self.G_optimizer.step()

        return {'loss': loss_function.detach()}


    def predict(self, source):
        self.F_model.eval()
        with torch.no_grad():
            device = next(self.F_model.parameters()).device
            return self.F_model(source.to(device))

    def predict_inverse(self, target):
        self.G_model.eval()
        with torch.no_grad():
            device = next(self.G_model.parameters()).device
            return self.G_model(target.to(device))

    def get_checkpoint(self):
        return {
            'F_model': self.F_model.state_dict(),
            'G_model': self.G_model.state_dict(),
            'F_optimizer': self.F_optimizer.state_dict(),
            'G_optimizer': self.G_optimizer.state_dict(),
        }
