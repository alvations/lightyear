from . import metrics as _metrics


def __getattr__(name):
    if name in _metrics.__all__:
        return getattr(_metrics, name)
    raise AttributeError(f"module 'lightyear' has no attribute {name!r}")


class Buzz:
    def __init__(self, metrics='all', trg_lang='en', lowercase=False):
        if metrics == 'all':
            self.metrics = {
                'bert_score': _metrics.BERTScore(),
                'bleu_score': _metrics.BLEUScore(trg_lang=trg_lang, lowercase=lowercase),
                'comet_score': _metrics.COMETScore(),
                'metricx_score': _metrics.MetricXScore(),
                'sentbert_score': _metrics.SentenceBERTScore(),
                'chrf_score': _metrics.CHRFScore(lowercase=lowercase),
                'ter_score': _metrics.TERScore(lowercase=lowercase),
            }
        elif metrics == 'fast':
            self.metrics = {
                'bleu_score': _metrics.BLEUScore(trg_lang=trg_lang, lowercase=lowercase),
                'chrf_score': _metrics.CHRFScore(lowercase=lowercase),
                'ter_score': _metrics.TERScore(lowercase=lowercase),
            }
        elif metrics == 'neural':
            self.metrics = {
                'bert_score': _metrics.BERTScore(),
                'comet_score': _metrics.COMETScore(),
                'metricx_score': _metrics.MetricXScore(),
                'sentbert_score': _metrics.SentenceBERTScore(),
            }
        elif isinstance(metrics, dict):
            self.metrics = metrics
        else:
            raise ValueError(f"Unknown metrics preset: {metrics!r}")

    def score(self, hyp, ref, src=None, normalize=True, clip=True):
        hyp, ref = hyp.strip(), ref.strip()
        src = src.strip() if src else None

        scores = {}
        for n, m in self.metrics.items():
            _score = m.score(hyp, ref, src, normalize=normalize)[n]['score']
            if clip:
                _score = max(0, min(_score, 100))
            scores[n] = _score
        return scores
