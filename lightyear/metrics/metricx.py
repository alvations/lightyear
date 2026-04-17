import copy
import dataclasses
import os
from typing import Optional, Tuple, Union

# mT5 tokenizers pull in sentencepiece's generated protobuf bindings, which
# collide with newer protobuf installs. Pure-python protobuf avoids the clash.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from torch import nn
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.models.mt5.modeling_mt5 import MT5Config, MT5PreTrainedModel, MT5Stack


@dataclasses.dataclass
class MT5ForRegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    predictions: torch.FloatTensor = None


class MT5ForRegression(MT5PreTrainedModel):
    """MT5 encoder-decoder with a regression head reading logit at position
    (0, 250089) — the <extra_id_10> token. Shared between MetricX-23 and
    MetricX-24; the only differences are checkpoint weights and input format."""

    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.post_init()
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], MT5ForRegressionOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        batch_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)
        decoder_input_ids = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=hidden_states.device
        )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)
        predictions = lm_logits[:, 0, 250089]
        predictions = torch.clamp(predictions, 0, 25)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            labels = labels.to(predictions.device)
            loss = loss_fct(predictions.view(-1), labels.view(-1))

        return MT5ForRegressionOutput(loss=loss, predictions=predictions)


class MetricXScore:
    """MetricX scorer with `version` and `qe` toggles.

    Args:
        version: '23' or '24' (default '24').
        qe: If True, score in QE mode (no reference).
        size: 'large' (default), 'xl', or 'xxl'.
        model_name: Override auto-selected HF model id.
        tokenizer_name: Override auto-selected mT5 tokenizer id.

    Defaults (size='large'):
        version='24', qe=False → google/metricx-24-hybrid-large-v2p6  (ref+src+hyp)
        version='24', qe=True  → google/metricx-24-hybrid-large-v2p6  (src+hyp)
        version='23', qe=False → google/metricx-23-large-v2p0         (ref+hyp)
        version='23', qe=True  → google/metricx-23-qe-large-v2p0      (src+hyp)

    Score output is also exposed under `raw_error` (the native 0–25
    error-score, lower is better). The `score` key is the normalized
    0–100 version (higher is better) when `normalize=True`.
    """

    _DEFAULTS = {
        ("24", False): "google/metricx-24-hybrid-{size}-v2p6",
        ("24", True):  "google/metricx-24-hybrid-{size}-v2p6",
        ("23", False): "google/metricx-23-{size}-v2p0",
        ("23", True):  "google/metricx-23-qe-{size}-v2p0",
    }

    def __init__(self, version: str = "24", qe: bool = False, size: str = "large",
                 model_name: str = None, tokenizer_name: str = None,
                 max_input_length: int = 1536, cuda: bool = True,
                 score_key: str = None):
        self.version = str(version)
        self.qe = qe
        if (self.version, self.qe) not in self._DEFAULTS:
            raise ValueError(f"Unsupported (version, qe) = ({self.version}, {self.qe})")

        self._model_name = model_name or self._DEFAULTS[(self.version, self.qe)].format(size=size)
        # mT5 SentencePiece is identical across all sizes, so `mt5-large` is a
        # safe default tokenizer regardless of the MetricX backbone size.
        self._tokenizer_name = tokenizer_name or "google/mt5-large"
        self.max_input_length = max_input_length
        self.device = "cuda" if cuda and torch.cuda.is_available() else "cpu"
        self.score_key = score_key or ("metricxqe_score" if qe else "metricx_score")

        self.tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
        self.model = MT5ForRegression.from_pretrained(self._model_name, torch_dtype="auto")
        self.model.to(self.device).eval()

    def _build_input(self, hyp: str, ref: str = None, src: str = None) -> str:
        if self.version == "24":
            if self.qe:
                if src is None:
                    raise ValueError("MetricX-24 QE requires a source")
            else:
                if ref is None:
                    raise ValueError("MetricX-24 ref-mode requires a reference")
                # Mirror COMET's convention: if no source, stand in with ref.
                if src is None:
                    src = ref
            parts = [f"source: {src}", f"candidate: {hyp}"]
            if not self.qe:
                parts.append(f"reference: {ref}")
            return " ".join(parts)
        else:
            if self.qe:
                if src is None:
                    raise ValueError("MetricX-23 QE requires a source")
                return f"candidate: {hyp} source: {src}"
            if ref is None:
                raise ValueError("MetricX-23 ref-mode requires a reference")
            return f"candidate: {hyp} reference: {ref}"

    def _tokenize(self, texts):
        # Tokenize each text without padding so we can strip EOS per-row
        # (mirrors the official metricx `_remove_eos` then pad pipeline).
        encs = [self.tokenizer(t, max_length=self.max_input_length,
                               truncation=True, padding=False) for t in texts]
        for e in encs:
            e["input_ids"] = e["input_ids"][:-1]
            e["attention_mask"] = e["attention_mask"][:-1]
        padded = self.tokenizer.pad(encs, padding=True, return_tensors="pt")
        return padded.input_ids.to(self.device), padded.attention_mask.to(self.device)

    def score(self, hyp: str, ref: str = None, src: str = None,
              normalize: bool = True, *args, **kwargs):
        text = self._build_input(hyp, ref=ref, src=src)
        input_ids, attn = self._tokenize([text])
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attn)
        raw = float(out.predictions[0])
        score = (1 - raw / 25) * 100 if normalize else raw
        return {self.score_key: {"score": score, "raw_error": raw}}

    def score_batch(self, hyps, refs=None, srcs=None, batch_size: int = 8,
                    normalize: bool = True, *args, **kwargs):
        n = len(hyps)
        refs = refs if refs is not None else [None] * n
        srcs = srcs if srcs is not None else [None] * n
        raws = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                texts = [self._build_input(h, ref=r, src=s)
                         for h, r, s in zip(hyps[i:i + batch_size],
                                            refs[i:i + batch_size],
                                            srcs[i:i + batch_size])]
                input_ids, attn = self._tokenize(texts)
                out = self.model(input_ids=input_ids, attention_mask=attn)
                raws.extend(out.predictions.tolist())
        scores = [(1 - r / 25) * 100 for r in raws] if normalize else raws
        return {self.score_key: {"score": scores, "raw_error": raws}}
