# lightyear

To infinity and beyond...

A unified MT-evaluation toolbox. Wraps **sacrebleu** (BLEU / CHRF / TER) and
re-implements **BERTScore**, **COMET**, **CometKiwi**, **MetricX-23**,
**MetricX-24**, and **SentenceBERTScore** on top of plain
`transformers` + `torch` + `sentence-transformers` — no
`pytorch-lightning`, no `unbabel-comet`, no `bert_score` dependency.

All neural metrics are **bit-exact parity-verified** against their original
libraries on sample inputs (see parity notes below).

Install
====

```
pip install -U lightyear
```

Dependencies (all via pip):
`sacrebleu`, `torch`, `transformers`, `datasets`, `sentencepiece`,
`sentence-transformers`.

Usage
====

## Typical Hyp + Ref setup

```python
from lightyear import Buzz

hyp = 'The dog bit the man.'
ref = 'The dog had bit the man.'

buzz = Buzz()
buzz.score(hyp, ref)
```

```
{'bert_score': 93.52701902389526,
 'bleu_score': 51.15078115793242,
 'comet_score': 94.05231475830078,
 'metricx_score': 82.4525089263916,
 'sentbert_score': 96.90018892288208,
 'chrf_score': 67.64711450656517,
 'ter_score': 16.666666666666664}
```

## Hyp + Ref + Src setup

```python
from lightyear import Buzz

src = 'The dog bit the man.'
hyp = 'Hund Mann gebissen.'
ref = 'Der Hund hatte den Mann gebissen.'

buzz = Buzz(metrics='all', trg_lang='de', lowercase=True)
print(buzz.score(hyp, ref, src))
```

```
{'bert_score': 71.62808179855347,
 'bleu_score': 30.18153515504547,
 'comet_score': 67.40456819534302,
 'metricx_score': 94.57738256454468,
 'sentbert_score': 98.21405410766602,
 'chrf_score': 51.512405379593574,
 'ter_score': 50.0}
```

## Presets

- `Buzz('fast')` — sacrebleu only (BLEU / CHRF / TER). No model downloads.
- `Buzz('neural')` — BERTScore + COMET + MetricX-24 + SentenceBERTScore.
- `Buzz('all')` — everything above + BLEU / CHRF / TER.
- `Buzz(metrics={...})` — bring your own dict of scorers (see below).

```python
buzz = Buzz(metrics='fast', lowercase=True)
with open('hyp.txt') as hfin, open('ref.txt') as rfin:
    for hyp, ref in zip(hfin, rfin):
        print(buzz.score(hyp, ref))
```

```
{'bleu_score': 11.631736348831648, 'chrf_score': 25.66796545720479, 'ter_score': 100.0}
{'bleu_score': 21.3643503198117,   'chrf_score': 38.2883972133884, 'ter_score': 75.0}
```

## Quality Estimation (QE, no reference)

Both COMET and MetricX expose a `qe=True` toggle. In QE mode only the
source and hypothesis are used — no reference needed.

```python
from lightyear.metrics import COMETScore, MetricXScore

# Reference-based (default)
comet = COMETScore()                       # Unbabel/wmt22-comet-da
mx24  = MetricXScore(version='24')         # google/metricx-24-hybrid-large-v2p6

# Quality-estimation (no reference)
cometkiwi = COMETScore(qe=True)            # Unbabel/wmt22-cometkiwi-da
mx24qe    = MetricXScore(version='24', qe=True)
mx23      = MetricXScore(version='23')             # google/metricx-23-large-v2p0
mx23qe    = MetricXScore(version='23', qe=True)    # google/metricx-23-qe-large-v2p0

src = 'Der Hund hat den Mann gebissen.'
hyp = 'The dog bit the man.'

print(cometkiwi.score(hyp, src=src))
# {'cometkiwi_score': {'score': 87.19...}}

print(mx24qe.score(hyp, src=src))
# {'metricxqe_score': {'score': 94.34..., 'raw_error': 1.41...}}
```

Use a custom mix via `Buzz(metrics={...})`:

```python
buzz_qe = Buzz(metrics={
    'cometkiwi_score': COMETScore(qe=True),
    'metricxqe_score': MetricXScore(version='24', qe=True),
})
print(buzz_qe.score(hyp, src=src))
```

## Swapping in other COMET or MetricX checkpoints

Any COMET-family checkpoint (`wmt20-comet-da`, `wmt23-cometkiwi-da-xl`,
`XCOMET-XL`, …) works via `model_name`. The architecture
(regression / referenceless / unified) is auto-detected from the
checkpoint's FF shape.

```python
COMETScore(model_name='Unbabel/wmt20-comet-da',
           layer_transformation='softmax')   # older COMETs used softmax

COMETScore(qe=True,
           model_name='Unbabel/wmt23-cometkiwi-da-xl',
           pretrained='facebook/xlm-roberta-xl')
```

MetricX exposes `version` ('23' / '24') and `size` ('large' / 'xl' / 'xxl'):

```python
MetricXScore(version='24', size='xxl')                   # hybrid-xxl-v2p6
MetricXScore(version='24', size='large', qe=True)        # hybrid-large-v2p6, QE mode
MetricXScore(model_name='google/metricx-24-hybrid-xxl-v2p6-bfloat16')
```

## Using Machine Translators

```python
from lightyear.translators import HelsinkiMarianTranslator, M2MTranslator, NLLBTranslator

hema = HelsinkiMarianTranslator()
print(hema.translate('en', 'de', 'I am pregnant'))
print(hema.translate('de', 'zh', 'Ich bin schwanger'))

m2m = M2MTranslator()
print(m2m.translate('en', 'de', 'I am pregnant'))
print(m2m.translate('de', 'zh', 'Ich bin schwanger'))

nllb = NLLBTranslator()
print(nllb.translate('eng_Latn', 'deu_Latn', 'I am pregnant'))
print(nllb.translate('deu_Latn', 'zho_Hans', 'Ich bin schwanger'))
```

```
Ich bin schwanger
我怀孕了 我怀孕了

Ich bin schwanger.
我怀孕了

Ich bin schwanger.
我怀孕了
```

## Parity verification

The neural metrics were checked bit-exact against the upstream libraries
on sample inputs before the rewrite landed:

| Metric | Reference implementation | Diff |
|---|---|---|
| BERTScore (mBERT) | `bert_score.BERTScorer` | 0 (to 4dp) |
| COMET (wmt22-comet-da) | `comet.load_from_checkpoint` | 0 (to 6dp) |
| CometKiwi QE (wmt22-cometkiwi-da) | `comet.load_from_checkpoint` | 0 (exact) |
| MetricX-24 ref + QE | `metricx24.MT5ForRegression` | 0 (to 6dp) |
| MetricX-23 ref + QE | `metricx23.MT5ForRegression` | 0 (to 6dp) |

BLEU / CHRF / TER call `sacrebleu` directly, and `SentenceBERTScore` calls
`sentence-transformers` directly, so parity is automatic.

Gotchas found during the port:
- wmt22-comet-da uses `layer_transformation: sparsemax` (pure-torch
  sparsemax now shipped — no `entmax` dep).
- wmt22-cometkiwi-da uses a **different architecture** ("unified": joint
  `(mt, src)` encoding, CLS pooling) on a `microsoft/infoxlm-large`
  backbone. Upstream's `sparsemax_patch` hparam silently falls through to
  softmax in the library; we replicate that.
- MetricX's EOS-strip needs to happen per-row before padding, not via a
  global column slice.
