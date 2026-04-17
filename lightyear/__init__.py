from . import metrics as _metrics


def __getattr__(name):
    if name in _metrics.__all__:
        return getattr(_metrics, name)
    raise AttributeError(f"module 'lightyear' has no attribute {name!r}")


class Buzz:
    """Reference-based metric bundle. `score(hyp, ref, src=None)`.

    Presets:
      - 'fast'       — sacrebleu only (BLEU/CHRF/TER)
      - 'neural'     — BERTScore + COMET + MetricX-24 + SentenceBERTScore
      - 'all'        — 'neural' + sacrebleu + difficulty_score (PreCOMET) + sentinel_src_score
      - 'difficulty' — src-only: DifficultyScore (PreCOMET) + SentinelSrcScore

    Or pass a custom dict via `metrics={...}`.
    """

    def __init__(self, metrics='all', trg_lang='en', lowercase=False):
        if metrics == 'all':
            self.metrics = {
                'bert_score':        _metrics.BERTScore(),
                'bleu_score':        _metrics.BLEUScore(trg_lang=trg_lang, lowercase=lowercase),
                'comet_score':       _metrics.COMETScore(),
                'metricx_score':     _metrics.MetricXScore(),
                'sentbert_score':    _metrics.SentenceBERTScore(),
                'chrf_score':        _metrics.CHRFScore(lowercase=lowercase),
                'ter_score':         _metrics.TERScore(lowercase=lowercase),
                'difficulty_score':  _metrics.DifficultyScore(),
                'sentinel_src_score': _metrics.SentinelSrcScore(),
            }
        elif metrics == 'fast':
            self.metrics = {
                'bleu_score':   _metrics.BLEUScore(trg_lang=trg_lang, lowercase=lowercase),
                'chrf_score':   _metrics.CHRFScore(lowercase=lowercase),
                'ter_score':    _metrics.TERScore(lowercase=lowercase),
            }
        elif metrics == 'neural':
            self.metrics = {
                'bert_score':     _metrics.BERTScore(),
                'comet_score':    _metrics.COMETScore(),
                'metricx_score':  _metrics.MetricXScore(),
                'sentbert_score': _metrics.SentenceBERTScore(),
            }
        elif metrics == 'difficulty':
            self.metrics = {
                'difficulty_score':   _metrics.DifficultyScore(),
                'sentinel_src_score': _metrics.SentinelSrcScore(),
            }
        elif isinstance(metrics, dict):
            self.metrics = metrics
        else:
            raise ValueError(f"Unknown metrics preset: {metrics!r}")

    # Keys whose scores are naturally bounded in [0, 100]. Difficulty metrics
    # are regression outputs and can legitimately go negative / above 100.
    _BOUNDED_KEYS = {
        'bert_score', 'bleu_score', 'comet_score', 'metricx_score',
        'sentbert_score', 'chrf_score', 'ter_score',
        'cometkiwi_score', 'metricxqe_score',
    }

    def score(self, hyp, ref, src=None, normalize=True, clip=True):
        hyp, ref = hyp.strip(), ref.strip()
        src = src.strip() if src else None
        scores = {}
        for n, m in self.metrics.items():
            _score = m.score(hyp, ref, src, normalize=normalize)[n]['score']
            if clip and n in self._BOUNDED_KEYS:
                _score = max(0, min(_score, 100))
            scores[n] = _score
        return scores


class BuzzQE:
    """Quality-estimation bundle. `score(hyp, src)` — no reference needed.

    Default metrics: CometKiwi, MetricX-24 QE, DifficultyScore (PreCOMET),
    SentinelSrcScore. All accept `(hyp, src)`; the src-only ones ignore `hyp`.
    """

    _BOUNDED_KEYS = Buzz._BOUNDED_KEYS

    def __init__(self, metrics=None):
        if metrics is None:
            self.metrics = {
                'cometkiwi_score':    _metrics.COMETScore(qe=True),
                'metricxqe_score':    _metrics.MetricXScore(qe=True),
                'difficulty_score':   _metrics.DifficultyScore(),
                'sentinel_src_score': _metrics.SentinelSrcScore(),
            }
        elif isinstance(metrics, dict):
            self.metrics = metrics
        else:
            raise ValueError(f"BuzzQE expects a dict or None, got {metrics!r}")

    def score(self, hyp, src, normalize=True, clip=True):
        hyp, src = hyp.strip(), src.strip()
        scores = {}
        for n, m in self.metrics.items():
            _score = m.score(hyp=hyp, ref=None, src=src, normalize=normalize)[n]['score']
            if clip and n in self._BOUNDED_KEYS:
                _score = max(0, min(_score, 100))
            scores[n] = _score
        return scores
