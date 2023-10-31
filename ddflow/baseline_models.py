import torch
from survae.transforms.bijections.coupling import CouplingBijection
from survae.transforms.bijections.functional import splines
from survae.utils import sum_except_batch
from survae.distributions import StandardNormal
from survae.nn.layers import ElementwiseParams, scale_fn
from survae.nn.nets import MLP
from survae.transforms import (
    ActNormBijection,
    ActNormBijection2d,
    AdditiveCouplingBijection,
    AffineCouplingBijection,
    Bijection,
    Conv1x1,
    CubicSplineCouplingBijection,
    LinearSplineCouplingBijection,
    QuadraticSplineCouplingBijection,
    RationalQuadraticSplineCouplingBijection,
    Reverse,
    ScalarAffineBijection,
    Shuffle,
    Squeeze2d,
    UniformDequantization,
    VariationalDequantization,
)
from torch import nn

from ddflow.mixedflows import BidirectionalFlow, BidirectionalMixedFlow
from ddflow.survae_utils.bidirectional_flow import BijectiveSlice
from ddflow.survae_utils.coupling import Coupling
from ddflow.survae_utils.dequantization_flow import DequantizationFlow


def extra_scale_fn(scale_str):
    if scale_str in ["exp", "softplus", "sigmoid", "tanh_exp"]:
        return scale_fn(scale_str)
    elif scale_str == "sigmoid_p05":
        return lambda s: torch.sigmoid(s) + 0.5
    elif scale_str == "tanh_exp_s1.5":
        return lambda s: torch.exp(1.5 * torch.tanh(s / 1.5))
    elif scale_str == "tanh_exp_s07":
        return lambda s: torch.exp(0.7 * torch.tanh(s / 0.7))
    else:
        return ValueError(scale_str)


def get_spline_dependent_flow(
    base_dist,
    D=2,
    num_layers=3,
    num_bins=100,
    hidden_units=[50],
    activation="relu",
    shuffle=True,
    spline_type="rational_quadratic",
    bound=3.0,
    **kwargs,
):
    print(f"ignoring keyword arguments in spline flow: {kwargs}")

    transforms = get_spline_transforms(
        D=D,
        num_layers=num_layers,
        num_bins=num_bins,
        hidden_units=hidden_units,
        activation=activation,
        shuffle=shuffle,
        spline_type=spline_type,
        bound=bound,
    )
    model = BidirectionalMixedFlow(
        base_dist=base_dist, transforms=transforms, final_shape=(D,)
    )
    return model


def get_spline_transforms(
    num_bins=100,
    D=2,
    num_layers=3,
    hidden_units=[50],
    activation="relu",
    shuffle=True,
    spline_type="rational_quadratic",
    bound=4.5,
):
    transforms = []
    if spline_type in ["unconstrained_rational_quadratic", "rational_quadratic"]:
        P = 3 * num_bins + 1
    elif spline_type == "cubic":
        P = 2 * num_bins + 2
    elif spline_type == "quadratic":
        P = 2 * num_bins + 1
    elif spline_type == "linear":
        P = num_bins
    ## -> coupling pushes first ceil(D/2) dimensions through network to condition the last floor(D/2) ones
    d_in = D - (D // 2)
    d_out = D // 2
    for i in range(num_layers):
        net = nn.Sequential(
            MLP(
                d_in,
                P * d_out,
                hidden_units=hidden_units,
                activation=activation,
            ),
            ElementwiseParams(P),
        )
        if spline_type == "rational_quadratic":
            T = RationalQuadraticSplineCouplingBijection(
                net,
                num_bins=num_bins,
            )
        elif spline_type == "unconstrained_rational_quadratic":
            T = UnconstrainedRationalQuadraticSplineCouplingBijection(
                net,
                num_bins=num_bins,
                tail_bound=bound,
            )
        elif spline_type == "cubic":
            T = CubicSplineCouplingBijection(net, num_bins)
        elif spline_type == "quadratic":
            T = QuadraticSplineCouplingBijection(net, num_bins)
        elif spline_type == "linear":
            T = LinearSplineCouplingBijection(net, num_bins)
        transforms.append(T)

        # clip, because due to floating point stuff, splines are
        # sometimes 0-eps, or 1+eps, but need to stay within bounds

        if not spline_type == "unconstrained_rational_quadratic":
            transforms.append(Clip(0.0, 1.0))

        if shuffle:
            transforms.append(Shuffle(D))
        else:
            transforms.append(Reverse(D))
    transforms.pop()

    return transforms


def get_affine_dependent_flow(
    base_dist,
    D=2,
    num_layers=4,
    hidden_units=[50],
    activation="relu",
    scale="exp",
    actnorm=False,
    affine=True,
    shuffle=False,
    **kwargs,
):
    print(f"ignoring keyword arguments in affine flow: {kwargs}")
    transforms = get_affine_transforms(
        D=D,
        num_layers=num_layers,
        hidden_units=hidden_units,
        activation=activation,
        scale=scale,
        actnorm=actnorm,
        affine=affine,
        shuffle=shuffle,
    )
    model = BidirectionalMixedFlow(
        base_dist=base_dist, transforms=transforms, final_shape=(D,)
    )
    return model


def get_affine_independent_flow(
    D=2,
    num_layers=4,
    hidden_units=[50],
    activation="relu",
    scale="exp",
    actnorm=False,
    affine=True,
    shuffle=False,
):
    base_dist = StandardNormal((D,))
    transforms = get_affine_transforms(
        D=D,
        num_layers=num_layers,
        hidden_units=hidden_units,
        activation=activation,
        scale=scale,
        actnorm=actnorm,
        affine=affine,
        shuffle=shuffle,
    )

    model = BidirectionalFlow(
        base_dist=base_dist, transforms=transforms, final_shape=(D,)
    )
    return model


def get_affine_transforms(
    D=2,
    num_layers=4,
    hidden_units=[50],
    activation="relu",
    scale="exp",
    actnorm=False,
    affine=True,
    shuffle=False,
):
    transforms = []
    d_in = D - (D // 2)
    d_out = D // 2

    P = d_out
    if affine:
        P *= 2
    for i in range(num_layers):
        seq = [
            MLP(
                d_in,
                P,
                hidden_units=hidden_units,
                activation=activation,
            )
        ]
        if affine:
            seq.append(ElementwiseParams(2))
        net = nn.Sequential(*seq)
        if affine:
            transforms.append(
                AffineCouplingBijection(net, scale_fn=extra_scale_fn(scale))
            )
        else:
            transforms.append(AdditiveCouplingBijection(net))
        if actnorm:
            transforms.append(ActNormBijection(D))
        if shuffle:
            transforms.append(Shuffle(D))
        else:
            transforms.append(Reverse(D))
    transforms.pop()
    return transforms


def setup_baseline_img_flow(
    base_dist,
    data_shape=(3, 32, 32),
    num_bits=8,
    num_scales=2,
    num_steps=12,
    actnorm=False,
    dequant="none",
    dequant_steps=4,
    dequant_context=32,
    densenet_blocks=1,
    densenet_channels=64,
    densenet_depth=10,
    densenet_growth=64,
    dropout=0.0,
    gated_conv=True,
    set_seed=False,
    is_mixed=False,
):
    """from survae flows"""
    if set_seed:
        torch.manual_seed(555)
    transforms = []
    current_shape = data_shape
    if dequant == "uniform":
        transforms.append(UniformDequantization(num_bits=num_bits))
    elif dequant == "flow":
        dequantize_flow = DequantizationFlow(
            data_shape=data_shape,
            num_bits=num_bits,
            num_steps=dequant_steps,
            num_context=dequant_context,
            num_blocks=densenet_blocks,
            mid_channels=densenet_channels,
            depth=densenet_depth,
            growth=densenet_growth,
            dropout=dropout,
            gated_conv=gated_conv,
        )
        transforms.append(
            VariationalDequantization(encoder=dequantize_flow, num_bits=num_bits)
        )
    else:
        raise NotImplementedError(dequant)

    # Change range from [0,1]^D to [-0.5, 0.5]^D
    transforms.append(ScalarAffineBijection(shift=-0.5))

    # Initial squeeze
    transforms.append(Squeeze2d())
    current_shape = (current_shape[0] * 4, current_shape[1] // 2, current_shape[2] // 2)

    for scale in range(num_scales):
        for step in range(num_steps):
            if actnorm:
                transforms.append(ActNormBijection2d(current_shape[0]))
            transforms.extend(
                [
                    Conv1x1(current_shape[0]),
                    Coupling(
                        in_channels=current_shape[0],
                        num_blocks=densenet_blocks,
                        mid_channels=densenet_channels,
                        depth=densenet_depth,
                        growth=densenet_growth,
                        dropout=dropout,
                        gated_conv=gated_conv,
                    ),
                ]
            )

        if scale < num_scales - 1:
            noise_shape = (
                current_shape[0] * 3,
                current_shape[1] // 2,
                current_shape[2] // 2,
            )
            transforms.append(Squeeze2d())
            transforms.append(
                # Slice(StandardNormal(noise_shape), num_keep=current_shape[0], dim=1)
                BijectiveSlice(
                    noise_shape=noise_shape, num_keep=current_shape[0], dim=1
                )
            )
            current_shape = (
                current_shape[0],
                current_shape[1] // 2,
                current_shape[2] // 2,
            )
        else:
            if actnorm:
                transforms.append(ActNormBijection2d(current_shape[0]))

    if is_mixed:
        flow = BidirectionalMixedFlow(
            base_dist=base_dist, transforms=transforms, final_shape=current_shape
        )
    else:
        flow = BidirectionalFlow(
            base_dist=base_dist, transforms=transforms, final_shape=current_shape
        )
    return flow


class Clip(Bijection):
    def __init__(self, min=0.0, max=1.0, clip_inverse=True):
        super(Clip, self).__init__()
        self.min = min
        self.max = max
        self.clip_inverse = clip_inverse

    def forward(self, x):
        return (
            x.clip(self.min, self.max),
            torch.zeros(len(x), dtype=x.dtype, device=x.device),
        )

    def inverse(self, z):
        if self.clip_inverse:
            return z.clip(self.min, self.max)
        else:
            return z


class UnconstrainedRationalQuadraticSplineCouplingBijection(CouplingBijection):
    def __init__(
        self, coupling_net, num_bins, split_dim=1, tail_bound=1.0, num_condition=None
    ):
        super(UnconstrainedRationalQuadraticSplineCouplingBijection, self).__init__(
            coupling_net=coupling_net, split_dim=split_dim, num_condition=num_condition
        )
        self.num_bins = num_bins
        self.tail_bound = tail_bound

    def _output_dim_multiplier(self):
        return 3 * self.num_bins + 1

    def _elementwise_forward(self, x, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths = elementwise_params[..., : self.num_bins]
        unnormalized_heights = elementwise_params[
            ..., self.num_bins : 2 * self.num_bins
        ]
        unnormalized_derivatives = elementwise_params[..., 2 * self.num_bins :]
        z, ldj_elementwise = splines.unconstrained_rational_quadratic_spline(
            x,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            tail_bound=self.tail_bound,
            inverse=False,
        )
        ldj = sum_except_batch(ldj_elementwise)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths = elementwise_params[..., : self.num_bins]
        unnormalized_heights = elementwise_params[
            ..., self.num_bins : 2 * self.num_bins
        ]
        unnormalized_derivatives = elementwise_params[..., 2 * self.num_bins :]
        x, _ = splines.unconstrained_rational_quadratic_spline(
            z,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            tail_bound=self.tail_bound,
            inverse=True,
        )
        return x
