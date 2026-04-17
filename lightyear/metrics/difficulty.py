"""Source-only difficulty / quality-estimation metrics.

`DifficultyScore` (default PreCOMET-diff) and `SentinelSrcScore` (default
Prosho/sentinel-src-25). Both predict a scalar from the source sentence
alone — no hypothesis, no reference.

Architecture:
  - PreCOMET:     encoder → layerwise-attention(25, sparsemax) → avg pool → FF(hidden_dim → ... → 1)
  - Sentinel-src: encoder → last_hidden_state               → avg pool → FF(hidden_dim → ... → 1)

The presence of layerwise_attention weights and hidden-sizes are
auto-detected from the checkpoint, so any compatible
`hypothesisless_regression_metric` / `sentinel_regression_metric` model
loads through the same class."""

import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig

from ._comet_common import (
    Encoder, Estimator, LayerwiseAttention, avg_pool,
    resolve_checkpoint, detect_hidden_sizes,
)


class _DifficultyModel(nn.Module):
    def __init__(self, pretrained: str = "xlm-roberta-large",
                 hidden_sizes=(2048, 1024), dropout: float = 0.1,
                 use_layerwise: bool = True,
                 layer_transformation: str = "sparsemax"):
        super().__init__()
        self.encoder = Encoder(pretrained)
        hidden_dim = self.encoder.model.config.hidden_size
        self.use_layerwise = use_layerwise
        if use_layerwise:
            num_layers = self.encoder.model.config.num_hidden_layers + 1
            self.layerwise_attention = LayerwiseAttention(
                num_layers=num_layers, layer_transformation=layer_transformation)
        self.estimator = Estimator(in_dim=hidden_dim,
                                   hidden_sizes=hidden_sizes, dropout=dropout)

    def forward(self, input_ids, attention_mask):
        if self.use_layerwise:
            out = self.encoder.model(input_ids=input_ids, attention_mask=attention_mask,
                                     output_hidden_states=True, return_dict=True)
            hidden = self.layerwise_attention(out.hidden_states)
        else:
            out = self.encoder.model(input_ids=input_ids, attention_mask=attention_mask,
                                     return_dict=True)
            hidden = out.last_hidden_state
        pooled = avg_pool(hidden, attention_mask)
        return self.estimator(pooled).squeeze(-1)


class DifficultyScore:
    """Source-only difficulty scorer. Default: `zouharvi/precomet-diff`.

    Pass `model_name` to load any `hypothesisless_regression_metric` or
    `sentinel_regression_metric` checkpoint (e.g. `zouharvi/precomet-avg`,
    `zouharvi/precomet-var`, `zouharvi/precomet-diversity`,
    `Prosho/sentinel-src-25`, `sapienzanlp/sentinel-src-mqm`).
    """

    DEFAULT_MODEL = "zouharvi/precomet-diff"

    def __init__(self, model_name: str = None,
                 pretrained: str = "xlm-roberta-large",
                 hidden_sizes=None,
                 use_layerwise: bool = None,
                 layer_transformation: str = "sparsemax",
                 cuda: bool = True, dropout: float = 0.1,
                 score_key: str = "difficulty_score"):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._score_key = score_key
        self.device = "cuda" if cuda and torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True)
        ckpt_path = resolve_checkpoint(self._model_name)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        if use_layerwise is None:
            use_layerwise = "layerwise_attention.gamma" in state_dict
        if hidden_sizes is None:
            hidden_sizes = detect_hidden_sizes(state_dict)

        self.model = _DifficultyModel(
            pretrained=pretrained, hidden_sizes=tuple(hidden_sizes),
            dropout=dropout, use_layerwise=use_layerwise,
            layer_transformation=layer_transformation,
        )
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

    def _tokenize(self, texts):
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    def score(self, hyp=None, ref=None, src=None, normalize: bool = True,
              *args, **kwargs):
        # Src-only metric: prefer src, fall back to hyp so it also works when
        # called via the Buzz(hyp, ref) signature.
        text = src if src is not None else hyp
        if text is None:
            raise ValueError("DifficultyScore needs a source (src=... or hyp=...)")
        with torch.no_grad():
            input_ids, attn = self._tokenize([text])
            raw = self.model(input_ids, attn).item()
        score = raw * 100 if normalize else raw
        return {self._score_key: {"score": score}}

    def score_batch(self, hyps=None, refs=None, srcs=None, batch_size: int = 8,
                    normalize: bool = True, *args, **kwargs):
        texts = srcs if srcs is not None else hyps
        if texts is None:
            raise ValueError("DifficultyScore needs sources (srcs=... or hyps=...)")
        scores = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                input_ids, attn = self._tokenize(texts[i:i + batch_size])
                scores.extend(self.model(input_ids, attn).tolist())
        scores = [s * 100 for s in scores] if normalize else scores
        return {self._score_key: {"score": scores}}


class SentinelSrcScore(DifficultyScore):
    """Sentinel-src difficulty scorer. Default: `Prosho/sentinel-src-25`."""

    DEFAULT_MODEL = "Prosho/sentinel-src-25"

    def __init__(self, model_name: str = None,
                 score_key: str = "sentinel_src_score", **kwargs):
        super().__init__(
            model_name=model_name or self.DEFAULT_MODEL,
            score_key=score_key, **kwargs,
        )
