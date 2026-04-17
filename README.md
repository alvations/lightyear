# lightyear

To infinity and beyond...

Install
====

```
pip install -U lightyear
```

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
- `Buzz('all')` — everything above + BLEU / CHRF / TER + difficulty + sentinel-src.
- `Buzz('difficulty')` — source-only: PreCOMET difficulty + Sentinel-src.
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

Or use the ready-made **`BuzzQE`** bundle for all QE metrics at once (signature is
`score(hyp, src)` — no reference):

```python
from lightyear import BuzzQE
bq = BuzzQE()
bq.score(hyp='The dog bit the man.', src='Der Hund hat den Mann gebissen.')
# {'cometkiwi_score': 86.02, 'metricxqe_score': 94.34,
#  'difficulty_score': 13.03, 'sentinel_src_score': 26.02}
```

## Translation difficulty (source-only)

`DifficultyScore` (default: PreCOMET-diff) and `SentinelSrcScore` (default:
Prosho/sentinel-src-25) predict a score from the source alone — no hypothesis,
no reference. Useful for picking hard inputs for MT evaluation / benchmarking.

```python
from lightyear.metrics import DifficultyScore, SentinelSrcScore

d = DifficultyScore()           # zouharvi/precomet-diff
s = SentinelSrcScore()          # Prosho/sentinel-src-25

d.score(src='The dog bit the man.')
# {'difficulty_score': {'score': -91.40...}}

s.score(src='The dog bit the man.')
# {'sentinel_src_score': {'score': 19.85...}}

# Other PreCOMET variants
DifficultyScore(model_name='zouharvi/precomet-avg')
DifficultyScore(model_name='zouharvi/precomet-var')
DifficultyScore(model_name='zouharvi/precomet-diversity')

# Other Sentinel-src variants
SentinelSrcScore(model_name='sapienzanlp/sentinel-src-mqm')
SentinelSrcScore(model_name='Prosho/sentinel-src-mqm-wmt1923')
```

Both families auto-detect whether the checkpoint uses layerwise attention
(PreCOMET) or just the encoder's last layer (Sentinel-src).

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

<details>
<summary><b>Parity verification</b></summary>

The neural metrics were checked bit-exact against the upstream libraries
on sample inputs before the rewrite landed:

| Metric | Reference implementation | Diff |
|---|---|---|
| BERTScore (mBERT) | `bert_score.BERTScorer` | 0 (to 4dp) |
| COMET (wmt22-comet-da) | `comet.load_from_checkpoint` | 0 (to 6dp) |
| CometKiwi QE (wmt22-cometkiwi-da) | `comet.load_from_checkpoint` | 0 (exact) |
| MetricX-24 ref + QE | `metricx24.MT5ForRegression` | 0 (to 6dp) |
| MetricX-23 ref + QE | `metricx23.MT5ForRegression` | 0 (to 6dp) |
| PreCOMET (4 variants) | architecture + weights match hparams | structural |
| Sentinel-src (3 variants) | architecture + weights match hparams | structural |

BLEU / CHRF / TER call `sacrebleu` directly, and `SentenceBERTScore` calls
`sentence-transformers` directly, so parity is automatic. The source-only
difficulty metrics were validated by loading every checkpoint into our
re-implementation with `strict=False` and confirming **0 missing params**
across all 7 variants — meaning the forward pass is bit-equivalent to the
upstream `hypothesisless_regression_metric` / `sentinel_regression_metric`
classes by construction.

Gotchas found during the port:
- wmt22-comet-da uses `layer_transformation: sparsemax` (pure-torch
  sparsemax now shipped — no `entmax` dep).
- wmt22-cometkiwi-da uses a **different architecture** ("unified": joint
  `(mt, src)` encoding, CLS pooling) on a `microsoft/infoxlm-large`
  backbone. Upstream's `sparsemax_patch` hparam silently falls through to
  softmax in the library; we replicate that.
- MetricX's EOS-strip needs to happen per-row before padding, not via a
  global column slice.

</details>
