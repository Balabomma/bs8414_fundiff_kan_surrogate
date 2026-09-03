"""KANLinear, vendored verbatim for the Part1 KAN architecture variants.

Lines 18-105 of `bs8414_KAN_surrogate/model.py`, copied byte-for-byte rather
than re-implemented. The whole point of a KAN-vs-baseline ablation is that the
B-spline edge block is the SAME block the existing KAN surrogate was measured
with; a re-implementation that differed in an initialisation constant or in the
basis width would silently make the comparison meaningless.

Vendored rather than imported because each project has its own venv and there is
no shared package on the path - the same reason `bs8414_samba_mlp_surrogate`
vendors its Mamba block. Verify with:

    python -c "import kan_layers_part1 as k, hashlib, inspect; \
               print(hashlib.sha256(inspect.getsource(k.KANLinear).encode()).hexdigest()[:16])"

and compare against the same hash taken on `bs8414_KAN_surrogate/model.py`.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """KAN Layer: replaces nn.Linear with learnable B-spline activations on edges.

    Instead of: y = activation(W @ x + b)
    KAN does:   y_j = sum_i(spline_ij(x_i))

    Each connection (i->j) has its own learnable activation function
    parameterized as a linear combination of B-spline basis functions.
    """

    def __init__(self, in_features, out_features, num_knots=8, residual=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_knots = num_knots
        self.residual = residual

        # Learnable B-spline coefficients for each edge (out x in x num_knots)
        self.spline_weights = nn.Parameter(
            torch.randn(out_features, in_features, num_knots) * (1.0 / math.sqrt(in_features * num_knots))
        )

        # Learnable per-feature input scale (helps map inputs into basis support)
        self.input_scale = nn.Parameter(torch.ones(in_features))

        # RBF basis centers in [-1, 1] (inputs are squashed to this range via tanh)
        centers = torch.linspace(-1.0, 1.0, num_knots)
        self.register_buffer('centers', centers)
        # Slightly wider than uniform spacing for smooth overlap
        self.width = 2.0 / (num_knots - 1) * 1.2

        # Residual linear connection (SiLU-weighted)
        if residual:
            self.residual_weight = nn.Parameter(
                torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features))
            )
            self.residual_bias = nn.Parameter(torch.zeros(out_features))

        # LayerNorm for stability (works with small batches and seq data)
        self.ln = nn.LayerNorm(out_features)

    def compute_basis(self, x):
        """Compute RBF basis values. x: (..., in_features) -> (..., in_features, num_knots)

        Inputs are tanh-squashed into [-1, 1] to guarantee basis coverage,
        with a learnable per-feature scale applied before the squash.
        """
        x_scaled = torch.tanh(x * self.input_scale)
        x_expanded = x_scaled.unsqueeze(-1)  # (..., in_features, 1)
        basis = torch.exp(-0.5 * ((x_expanded - self.centers) / self.width) ** 2)
        return basis  # (..., in_features, num_knots)

    def forward(self, x):
        """x: (batch, in_features) or (batch, seq, in_features)"""
        orig_shape = x.shape
        if x.dim() == 3:
            batch, seq, feat = x.shape
            x_flat = x.reshape(-1, feat)
        else:
            x_flat = x
            batch = x.shape[0]

        # Compute basis functions for each input
        basis = self.compute_basis(x_flat)  # (batch*, in_features, num_knots)

        # Spline output: sum over input features and knots
        out = torch.einsum('bik,oik->bo', basis, self.spline_weights)

        # Add residual connection
        if self.residual:
            residual = F.silu(x_flat) @ self.residual_weight.t() + self.residual_bias
            out = out + residual

        # LayerNorm over feature dim (safe for any batch/seq shape)
        out = self.ln(out)

        if len(orig_shape) == 3:
            out = out.reshape(batch, seq, -1)

        return out

    def spline_regularization(self):
        """L2 + smoothness penalty on spline weights, for use in training loss."""
        l2 = self.spline_weights.pow(2).mean()
        # Smoothness: penalize differences between adjacent knot coefficients
        diff = self.spline_weights[..., 1:] - self.spline_weights[..., :-1]
        smooth = diff.pow(2).mean()
        return l2 + smooth


def kan_regularization(model):
    """Sum spline L2 + smoothness penalty over all KANLinear layers in `model`."""
    reg = 0.0
    n = 0
    for m in model.modules():
        if isinstance(m, KANLinear):
            reg = reg + m.spline_regularization()
            n += 1
    if n == 0:
        return torch.tensor(0.0)
    return reg / n
