import torch
from survae.flows import Flow

from ddflow.dependent_distributions import DependentDistribution
from ddflow.survae_utils.bidirectional_flow import BidirectionalFlow


class BidirectionalMixedFlow(BidirectionalFlow):
    def __init__(self, base_dist, transforms, final_shape):
        super(BidirectionalMixedFlow, self).__init__(
            base_dist=base_dist, transforms=transforms, final_shape=final_shape
        )
        assert isinstance(base_dist, DependentDistribution)

    def log_prob_dependent(self, x, sample_indices):
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
        log_prob += self.base_dist.log_prob_mini_batch(z, sample_indices=sample_indices)
        return log_prob


class MixedFlow(Flow):
    def __init__(self, base_dist, transforms):
        super(MixedFlow, self).__init__(base_dist=base_dist, transforms=transforms)
        assert isinstance(base_dist, DependentDistribution)

    def log_prob_dependent(self, x, sample_indices):
        log_prob = torch.zeros(x.shape[0], device=x.device)
        for transform in self.transforms:
            x, ldj = transform(x)
            log_prob += ldj
        log_prob += self.base_dist.log_prob_mini_batch(x, sample_indices=sample_indices)
        return log_prob
