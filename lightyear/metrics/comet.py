import os

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig


def _sparsemax(z: torch.Tensor, dim: int = 0) -> torch.Tensor:
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
    def __init__(self, num_layers: int = 25, layer_transformation: str = "sparsemax"):
        super().__init__()
        self.num_layers = num_layers
        self.layer_transformation = layer_transformation
        if layer_transformation == "sparsemax":
            self._transform = _sparsemax
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


class _Estimator(nn.Module):
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


class _Encoder(nn.Module):
    def __init__(self, pretrained: str = "xlm-roberta-large"):
        super().__init__()
        try:
            self.model = AutoModel.from_pretrained(pretrained, add_pooling_layer=False)
        except TypeError:
            self.model = AutoModel.from_pretrained(pretrained)


def _avg_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


# COMET architecture family constants
ARCH_REGRESSION = "regression"        # wmt*-comet-da: 6 features, avg pool, separate encodings
ARCH_REFERENCELESS = "referenceless"  # old referenceless QE: 4 features, avg pool, separate encodings
ARCH_UNIFIED = "unified"              # wmt22-cometkiwi-da, XCOMET: joint (mt, src) input, CLS pool


_ARCH_N_FEATS = {
    ARCH_REGRESSION: 6,
    ARCH_REFERENCELESS: 4,
    ARCH_UNIFIED: 1,
}


class COMETRegressionModel(nn.Module):
    """Unified COMET scoring backbone supporting all 3 common architectures."""

    def __init__(self, pretrained: str = "xlm-roberta-large",
                 hidden_sizes=(3072, 1024), dropout: float = 0.1,
                 architecture: str = ARCH_REGRESSION,
                 layer_transformation: str = "sparsemax"):
        super().__init__()
        if architecture not in _ARCH_N_FEATS:
            raise ValueError(f"Unknown architecture: {architecture!r}")
        self.architecture = architecture
        self.encoder = _Encoder(pretrained)
        hidden_dim = self.encoder.model.config.hidden_size
        num_layers = self.encoder.model.config.num_hidden_layers + 1
        self.layerwise_attention = LayerwiseAttention(
            num_layers=num_layers, layer_transformation=layer_transformation)
        n_feats = _ARCH_N_FEATS[architecture]
        self.estimator = _Estimator(in_dim=hidden_dim * n_feats,
                                    hidden_sizes=hidden_sizes, dropout=dropout)

    def _encode_and_mix(self, input_ids, attention_mask):
        out = self.encoder.model(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=True, return_dict=True)
        return self.layerwise_attention(out.hidden_states)

    def forward_unified(self, input_ids, attention_mask):
        """Joint-encoding forward for CometKiwi-style models."""
        mixed = self._encode_and_mix(input_ids, attention_mask)
        cls = mixed[:, 0, :]
        return self.estimator(cls).squeeze(-1)

    def forward_separate(self, src_ids, src_mask, mt_ids, mt_mask,
                         ref_ids=None, ref_mask=None):
        """Separate-encoding forward for regression / referenceless models."""
        def sentemb(ids, mask):
            return _avg_pool(self._encode_and_mix(ids, mask), mask)
        s = sentemb(src_ids, src_mask)
        m = sentemb(mt_ids, mt_mask)
        if self.architecture == ARCH_REFERENCELESS:
            feats = torch.cat([m, s, m * s, (m - s).abs()], dim=1)
        else:  # ARCH_REGRESSION
            r = sentemb(ref_ids, ref_mask)
            feats = torch.cat([m, r, m * r, (m - r).abs(), m * s, (m - s).abs()], dim=1)
        return self.estimator(feats).squeeze(-1)


def _resolve_checkpoint(model_name: str, checkpoint_path: str = None) -> str:
    if checkpoint_path and os.path.isfile(checkpoint_path):
        return checkpoint_path
    local_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        f"models--{model_name.replace('/', '--')}",
    )
    snaps = os.path.join(local_root, "snapshots")
    if os.path.isdir(snaps):
        for snap in sorted(os.listdir(snaps)):
            ckpt = os.path.join(snaps, snap, "checkpoints", "model.ckpt")
            if os.path.isfile(ckpt):
                return ckpt
    from transformers.utils import cached_file
    return cached_file(model_name, "checkpoints/model.ckpt")


def _detect_hidden_sizes(state_dict) -> tuple:
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


def _detect_architecture(state_dict, hidden_dim: int) -> str:
    in_dim = state_dict["estimator.ff.0.weight"].shape[1]
    if in_dim == hidden_dim * 6:
        return ARCH_REGRESSION
    if in_dim == hidden_dim * 4:
        return ARCH_REFERENCELESS
    if in_dim == hidden_dim:
        return ARCH_UNIFIED
    raise ValueError(
        f"Unrecognized COMET architecture: ff in_dim={in_dim}, hidden_dim={hidden_dim}"
    )


class COMETScore:
    """COMET-family scorer with `qe` toggle and `model_name` override.

    qe=False → reference-based, Unbabel/wmt22-comet-da (feature-cat regression).
    qe=True  → referenceless, Unbabel/wmt22-cometkiwi-da (joint/unified).

    Auto-detects the underlying architecture (regression / referenceless /
    unified) from the checkpoint's FF input dim, so swapping `model_name` to
    any compatible Unbabel/* ckpt works without flag changes.
    """

    REF_MODEL = "Unbabel/wmt22-comet-da"
    QE_MODEL = "Unbabel/wmt22-cometkiwi-da"

    # Checkpoint-specific defaults. Any `Unbabel/*-cometkiwi*` was trained with
    # microsoft/infoxlm-large + softmax layer transform (hparams says
    # `sparsemax_patch`, which the upstream comet lib silently treats as
    # softmax). Reference COMETs use xlm-roberta-large + sparsemax.
    def __init__(self, qe: bool = False, model_name: str = None,
                 pretrained: str = None,
                 hidden_sizes=None,
                 layer_transformation: str = None,
                 checkpoint_path: str = None, cuda: bool = True,
                 dropout: float = 0.1, score_key: str = None):
        self.qe = qe
        self._model_name = model_name or (self.QE_MODEL if qe else self.REF_MODEL)
        if pretrained is None:
            pretrained = ("microsoft/infoxlm-large" if "cometkiwi" in self._model_name.lower()
                          else "xlm-roberta-large")
        if layer_transformation is None:
            layer_transformation = ("softmax" if "cometkiwi" in self._model_name.lower()
                                    else "sparsemax")
        self._score_key = score_key or ("cometkiwi_score" if qe else "comet_score")
        self.device = "cuda" if cuda and torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True)
        ckpt_path = _resolve_checkpoint(self._model_name, checkpoint_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        hidden_dim = AutoConfig.from_pretrained(pretrained).hidden_size
        self.architecture = _detect_architecture(state_dict, hidden_dim)
        if hidden_sizes is None:
            hidden_sizes = _detect_hidden_sizes(state_dict)

        self.model = COMETRegressionModel(
            pretrained=pretrained,
            hidden_sizes=tuple(hidden_sizes),
            dropout=dropout,
            architecture=self.architecture,
            layer_transformation=layer_transformation,
        )
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

    def _tokenize(self, texts):
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    def _tokenize_pair(self, texts_a, texts_b):
        enc = self.tokenizer(texts_a, texts_b, padding=True, truncation=True,
                             return_tensors="pt")
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    def score(self, hyp: str, ref: str = None, src: str = None,
              normalize: bool = True, *args, **kwargs):
        if self.architecture == ARCH_UNIFIED:
            if src is None:
                raise ValueError("Unified/CometKiwi model requires a source")
            with torch.no_grad():
                ids, mask = self._tokenize_pair([hyp], [src])
                raw = self.model.forward_unified(ids, mask).item()
        elif self.architecture == ARCH_REFERENCELESS:
            if src is None:
                raise ValueError("Referenceless COMET requires a source")
            with torch.no_grad():
                src_ids, src_mask = self._tokenize([src])
                mt_ids, mt_mask = self._tokenize([hyp])
                raw = self.model.forward_separate(src_ids, src_mask, mt_ids, mt_mask).item()
        else:  # ARCH_REGRESSION
            if ref is None:
                raise ValueError("Reference-based COMET requires a reference")
            if src is None:
                src = ref
            with torch.no_grad():
                src_ids, src_mask = self._tokenize([src])
                mt_ids, mt_mask = self._tokenize([hyp])
                ref_ids, ref_mask = self._tokenize([ref])
                raw = self.model.forward_separate(
                    src_ids, src_mask, mt_ids, mt_mask, ref_ids, ref_mask).item()
        score = raw * 100 if normalize else raw
        return {self._score_key: {"score": score}}

    def score_batch(self, hyps, refs=None, srcs=None, batch_size: int = 8,
                    normalize: bool = True, *args, **kwargs):
        if self.architecture == ARCH_UNIFIED:
            if srcs is None:
                raise ValueError("Unified/CometKiwi requires sources")
        elif self.architecture == ARCH_REFERENCELESS:
            if srcs is None:
                raise ValueError("Referenceless COMET requires sources")
        else:
            if refs is None:
                raise ValueError("Reference-based COMET requires references")
            if srcs is None:
                srcs = refs
        scores = []
        with torch.no_grad():
            for i in range(0, len(hyps), batch_size):
                h = hyps[i:i + batch_size]
                s = srcs[i:i + batch_size] if srcs is not None else None
                if self.architecture == ARCH_UNIFIED:
                    ids, mask = self._tokenize_pair(h, s)
                    batch = self.model.forward_unified(ids, mask)
                elif self.architecture == ARCH_REFERENCELESS:
                    src_ids, src_mask = self._tokenize(s)
                    mt_ids, mt_mask = self._tokenize(h)
                    batch = self.model.forward_separate(src_ids, src_mask, mt_ids, mt_mask)
                else:
                    r = refs[i:i + batch_size]
                    src_ids, src_mask = self._tokenize(s)
                    mt_ids, mt_mask = self._tokenize(h)
                    ref_ids, ref_mask = self._tokenize(r)
                    batch = self.model.forward_separate(
                        src_ids, src_mask, mt_ids, mt_mask, ref_ids, ref_mask)
                scores.extend(batch.tolist())
        scores = [s * 100 for s in scores] if normalize else scores
        return {self._score_key: {"score": scores}}


class CometKiwiScore(COMETScore):
    """Thin alias for `COMETScore(qe=True)`."""
    def __init__(self, **kwargs):
        kwargs.setdefault("qe", True)
        super().__init__(**kwargs)
