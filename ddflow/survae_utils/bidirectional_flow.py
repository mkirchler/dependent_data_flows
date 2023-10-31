from survae.flows import Flow

import numpy as np

import torch
from survae.transforms.bijections import Bijection


class BidirectionalFlow(Flow):
    """
    hacky extension of Flow that allows to:
        1) map noise -> data; and data -> noise in one class
        2) re-implements the slice to be bijective

    differences in use:
        1) use a StandardNormal (or any other univariate) distribution instead of conv distribution
            this also means: currently no parameter in the base_distribution!
        2) pass the final_shape, ie, what shape data would have after passing through all transforms
        3) data2noise returns flattened and noise2data accepts flattened; instead of the channel-dims
    """

    def __init__(self, base_dist, transforms, final_shape):
        super(BidirectionalFlow, self).__init__(
            base_dist=base_dist, transforms=transforms
        )
        self.final_shape = final_shape
        self.static_z = None

    def static_seed_sample(self, num_samples):
        if self.static_z is None:
            del self.static_z
            self.register_buffer("static_z", self.base_dist.sample(num_samples))
        assert (
            len(self.static_z) == num_samples
        ), f"called with different num_samples: {num_samples} vs {len(self.static_z)}"
        data = self.noise2data(self.static_z)
        return data

    def sample(self, num_samples):
        z = self.base_dist.sample(num_samples)
        data = self.noise2data(z)
        return data

    def print_shapes(self):
        z = self.base_dist.sample(1)
        last_d = np.prod(self.final_shape)
        factored = z[:, :-last_d]
        z = z[:, -last_d:].view(-1, *self.final_shape)
        for i, transform in reversed(list(enumerate(self.transforms))):
            print(
                f"layer {i}, {type(transform).__name__} \t\t - z-shape: {z.shape[1:]} \t factored out: {factored.shape[1:]}"
            )
            if isinstance(transform, BijectiveSlice):
                next_d = np.prod(transform.noise_shape)
                next_z = factored[:, -next_d:]
                factored = factored[:, :-next_d]
                z = torch.cat([z, next_z.view(-1, *transform.noise_shape)], dim=1)
            z = transform.inverse(z)
        print(f"output - z-shape: {z.shape[1:]} - factored out: {factored.shape[1:]}")

    def log_prob(self, x):
        log_prob = torch.zeros(x.shape[0], device=x.device)
        factored_out = []
        for transform in self.transforms:
            x, ldj = transform(x)
            # check for BijectiveSlice
            if isinstance(x, (tuple, list)):
                x, factored = x
                factored_out.append(factored)
            log_prob += ldj
        factored_out.append(x.flatten(1))
        z = torch.hstack(factored_out)
        log_prob += self.base_dist.log_prob(z)
        return log_prob

    def noise2data(self, z):
        last_d = np.prod(self.final_shape)
        factored = z[:, :-last_d]
        z = z[:, -last_d:].view(-1, *self.final_shape)
        for transform in reversed(self.transforms):
            if isinstance(transform, BijectiveSlice):
                next_d = np.prod(transform.noise_shape)
                next_z = factored[:, -next_d:]
                factored = factored[:, :-next_d]
                z = torch.cat([z, next_z.view(-1, *transform.noise_shape)], dim=1)
            z = transform.inverse(z)
        return z

    def data2noise(self, x, with_ldj=False):
        if with_ldj:
            log_prob = torch.zeros(x.shape[0], device=x.device)
        factored_out = []
        for transform in self.transforms:
            x, ldj = transform(x)
            # check for BijectiveSlice
            if isinstance(x, (tuple, list)):
                x, factored = x
                factored_out.append(factored)
            if with_ldj:
                log_prob += ldj
        factored_out.append(x.flatten(1))
        z = torch.hstack(factored_out)
        if with_ldj:
            return z, log_prob
        else:
            return z


class BijectiveSlice(Bijection):
    def __init__(self, noise_shape, num_keep, dim=1):
        super(BijectiveSlice, self).__init__()
        assert dim >= 1
        self.dim = dim
        self.num_keep = num_keep
        self.noise_shape = noise_shape

    def split_input(self, input):
        split_proportions = (self.num_keep, input.shape[self.dim] - self.num_keep)
        return torch.split(input, split_proportions, dim=self.dim)

    def forward(self, x):
        z, x2 = self.split_input(x)
        return (z, x2.flatten(1)), 0

    def inverse(self, x):
        return x
