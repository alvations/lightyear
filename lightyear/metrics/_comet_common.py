"""Shared building blocks for COMET-family scorers (reference, referenceless,
unified, and src-only / difficulty)."""
import os

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


# Architecture families (distinguished by `estimator.ff.0.weight.shape[1]`
# relative to the encoder hidden dim).
ARCH_REGRESSION = "regression"        # wmt*-comet-da: 6 features, avg pool, separate encodings
ARCH_REFERENCELESS = "referenceless"  # old referenceless QE: 4 features, avg pool, separate encodings
ARCH_UNIFIED = "unified"              # wmt22-cometkiwi-da, XCOMET: joint (mt, src) input, CLS pool
ARCH_SRC_ONLY = "src_only"            # sentinel-src, precomet-*: single src input, avg/CLS pool

_ARCH_N_FEATS = {
    ARCH_REGRESSION: 6,
    ARCH_REFERENCELESS: 4,
    ARCH_UNIFIED: 1,
    ARCH_SRC_ONLY: 1,
}


def sparsemax(z: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Native sparsemax (Martins & Astudillo 2016) — pure torch, differentiable."""
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    cumsum = torch.cumsum(z_sorted, dim=dim)
    shape = [1] * z.ndim
    shape[dim] = -1
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype).view(shape)
    support = ((1 + k * z_sorted) > cumsum).to(z.dtype)
    k_max = support.sum(dim=dim, keepdim=True)
    tau = (torch.gather(cumsum, dim, (k_max.long() - 1).clamp(min=0)) - 1) / k_max.clamp(min=1)
    return torch.clamp(z - tau, min=0)


class LayerwiseAttention(nn.Module):
    """Scalar-mix of encoder layers; `layer_transformation` picks softmax vs sparsemax."""

    def __init__(self, num_layers: int = 25, layer_transformation: str = "sparsemax"):
        super().__init__()
        self.num_layers = num_layers
        self.layer_transformation = layer_transformation
        if layer_transformation == "sparsemax":
            self._transform = sparsemax
        elif layer_transformation == "softmax":
            self._transform = lambda z, dim=0: F.softmax(z, dim=dim)
        else:
            raise ValueError(f"Unknown layer_transformation: {layer_transformation!r}")
        self.scalar_parameters = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(num_layers)]
        )
        self.gamma = nn.Parameter(torch.ones(1))
        self.register_buffer("dropout_mask", torch.zeros(num_layers))
        self.register_buffer("dropout_fill", torch.full((num_layers,), -1e20))

    def forward(self, layers):
        weights = torch.cat([p for p in self.scalar_parameters])
        normed = self._transform(weights, dim=0)
        stacked = torch.stack(list(layers), dim=0)
        mixed = (normed.view(-1, 1, 1, 1) * stacked).sum(dim=0)
        return self.gamma * mixed


class Estimator(nn.Module):
    """FF head: Linear → Tanh → Dropout repeated; final Linear → 1."""

    def __init__(self, in_dim: int, hidden_sizes=(3072, 1024), dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.Tanh(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.ff = nn.Sequential(*layers)

    def forward(self, x):
        return self.ff(x)


class Encoder(nn.Module):
    """Thin wrapper so state_dict keys line up as `encoder.model.*` (matches
    Unbabel/Sapienza PL checkpoint naming)."""

    def __init__(self, pretrained: str = "xlm-roberta-large"):
        super().__init__()
        try:
            self.model = AutoModel.from_pretrained(pretrained, add_pooling_layer=False)
        except TypeError:
            self.model = AutoModel.from_pretrained(pretrained)


def avg_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def resolve_checkpoint(model_name: str, checkpoint_path: str = None,
                       filename: str = "checkpoints/model.ckpt") -> str:
    """Find a PL-style checkpoint locally (lightyear repo cache) or via transformers."""
    if checkpoint_path and os.path.isfile(checkpoint_path):
        return checkpoint_path
    local_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        f"models--{model_name.replace('/', '--')}",
    )
    snaps = os.path.join(local_root, "snapshots")
    if os.path.isdir(snaps):
        for snap in sorted(os.listdir(snaps)):
            candidate = os.path.join(snaps, snap, filename)
            if os.path.isfile(candidate):
                return candidate
    from transformers.utils import cached_file
    return cached_file(model_name, filename)


def detect_hidden_sizes(state_dict) -> tuple:
    """Read ff.X.weight shapes to infer hidden layer sizes."""
    sizes = []
    i = 0
    while f"estimator.ff.{i}.weight" in state_dict:
        out_dim, _in = state_dict[f"estimator.ff.{i}.weight"].shape
        if out_dim == 1:
            break
        sizes.append(out_dim)
        i += 3
    return tuple(sizes) if sizes else (3072, 1024)


def detect_architecture(state_dict, hidden_dim: int, prefer_src_only: bool = False) -> str:
    """Auto-detect COMET-family architecture from the FF input dimension.

    in_dim == hidden_dim       → unified OR src_only (ambiguous without more info)
    in_dim == hidden_dim * 4   → referenceless QE
    in_dim == hidden_dim * 6   → regression (ref + src + mt)
    """
    in_dim = state_dict["estimator.ff.0.weight"].shape[1]
    if in_dim == hidden_dim * 6:
        return ARCH_REGRESSION
    if in_dim == hidden_dim * 4:
        return ARCH_REFERENCELESS
    if in_dim == hidden_dim:
        return ARCH_SRC_ONLY if prefer_src_only else ARCH_UNIFIED
    raise ValueError(
        f"Unrecognized COMET-family architecture: ff in_dim={in_dim}, hidden_dim={hidden_dim}"
    )
