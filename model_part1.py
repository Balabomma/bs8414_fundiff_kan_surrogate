"""KAN-FunDiff slice surrogate for the BS 8414 Part1 corpus.

Sibling of `bs8414_fundiff_surrogate`. Two-stage generative model: a Function
Autoencoder (Stage 1) and a rectified-flow Diffusion Transformer over the frozen
FAE latents (Stage 2). The corpus, the split and both stage trainers are
unchanged; the single variable under test is that the DiT's conditioning path is
built from `KANLinear` B-spline edges instead of a dense MLP.

WHERE THE KAN GOES, AND WHY NOT IN THE DiT BLOCKS
-------------------------------------------------
Same boundary as `bs8414_physicsnemo_kan_surrogate`, and the same precedent from
`bs8414_KAN_surrogate` (KAN in the parameter encoder and decoders, backbone
untouched):

    KAN-ised   Part1 conditioning -> AdaLN-Zero modulation vector
    UNCHANGED  DiT transformer blocks, AdaLN-Zero mechanism, rectified-flow
               objective, and the Stage-1 FAE

The DiT blocks are left alone for a concrete reason, not conservatism. AdaLN-Zero
works by having the conditioning vector emit per-block scale/shift/gate
parameters that start at zero, so the residual branch is initially an identity.
`KANLinear` ends in a `LayerNorm`, which cannot emit an exact zero and would
break that initialisation - the model would start with live, randomly-scaled
residual branches instead of a clean identity, and any accuracy difference would
be attributable to a broken warm start rather than to B-splines. So the KAN sits
BEFORE the modulation head: it shapes the conditioning vector, and the parent's
zero-initialised projection still produces the modulation itself.

The Stage-1 FAE is untouched because it autoencodes the field and never sees the
parameter vector - there is no conditioning path in it to KAN-ise.

To train this variant, swap the two imports the parent README already documents
(`dataset_part1` for the data layer) and build `Part1DiT(config)` from here.
"""
import torch
import torch.nn as nn

from part1_conditioning import Part1Conditioner
from kan_layers_part1 import KANLinear, kan_regularization
import config as parent_config
from model.dit import DiT, rectified_flow_loss, sample_latent, count_parameters

import os as _os
# Spline knots per KAN edge. Env-overridable so the capacity-match experiment
# of the thesis can vary it without editing the tracked default.
NUM_KNOTS = int(_os.environ.get("FUNDIFF_NUM_KNOTS",
                                getattr(parent_config, "NUM_KNOTS", 8)))


class Part1KANParamConditioner(nn.Module):
    """Part1 conditioning + slice embedding -> (B, D), through B-spline edges.

    Drop-in for the parent's `ParamConditioner`, and a drop-in for the non-KAN
    Part1 conditioner in `bs8414_fundiff_surrogate` - same signature, same output
    width, so the only thing that changes downstream is how the vector was made.
    """

    def __init__(self, n_slices, slice_embed_dim, out_dim, num_knots=NUM_KNOTS):
        super().__init__()
        self.conditioner = Part1Conditioner()
        self.slice_embed = nn.Embedding(n_slices, slice_embed_dim)
        d_in = Part1Conditioner.OUT_DIM + slice_embed_dim
        self.kan1 = KANLinear(d_in, out_dim, num_knots=num_knots)
        self.kan2 = KANLinear(out_dim, out_dim, num_knots=num_knots)

    def forward(self, params, slice_ids):
        x = torch.cat([self.conditioner(params), self.slice_embed(slice_ids)],
                      dim=-1)
        return self.kan2(self.kan1(x))


class Part1KANDiT(DiT):
    """Parent DiT with the KAN Part1 conditioner swapped in.

    Subclassed rather than copied so the transformer blocks and the
    rectified-flow head stay provably identical to the parent's - the same
    argument `bs8414_fundiff_surrogate/model_part1.py` makes for its own swap.
    """

    def __init__(self, config=parent_config):
        # Default so the shared trainer's `Part1Surrogate()` call works; the
        # explicit form `Part1Surrogate(cfg)` is unchanged.
        super().__init__(config)
        self.cond = Part1KANParamConditioner(
            config.N_SLICES, config.SLICE_EMBED_DIM, config.DIT_EMBED_DIM)


# ── uniform interface (see bs8414_KAN_surrogate/model_part1.py) ───────────
MODEL_NAME = "KAN-FunDiff DiT (Part1)"
Part1Surrogate = Part1KANDiT

# Matches the thermocouple KAN and the KAN-PhysicsNeMo variant so the spline
# penalty is not a second uncontrolled difference across the KAN studies.
LAMBDA_REG = 2e-3


def regularization(model):
    """Architecture-specific weight penalty, added to the training loss."""
    return kan_regularization(model)


if __name__ == "__main__":
    from config_part1 import (N_INPUT_PARAMS, N_CLADDING, N_INSULATION,
                              N_GEOMETRY, COL_CLADDING, COL_INSULATION, COL_GEOM)

    model = Part1Surrogate(parent_config)
    print(f"{MODEL_NAME}: {count_parameters(model):,} parameters")

    B = 4
    p = torch.rand(B, N_INPUT_PARAMS)
    p[:, COL_CLADDING] = torch.randint(0, N_CLADDING, (B,)).float()
    p[:, COL_INSULATION] = torch.randint(0, N_INSULATION, (B,)).float()
    p[:, COL_GEOM] = torch.randint(0, N_GEOMETRY, (B,)).float()
    slice_ids = torch.randint(0, parent_config.N_SLICES, (B,))

    z = torch.randn(B, parent_config.N_LATENT_TOKENS, parent_config.EMBED_DIM)
    t = torch.rand(B)
    v = model(z, t, p, slice_ids)
    print(f"  latent {tuple(z.shape)} -> velocity {tuple(v.shape)}")

    loss = rectified_flow_loss(model, z, p, slice_ids)
    loss = loss[0] if isinstance(loss, tuple) else loss
    print(f"  rectified-flow loss on random latents: {float(loss.detach()):.4f}")
    print(f"  spline regularisation: {float(regularization(model).detach()):.6f}")
