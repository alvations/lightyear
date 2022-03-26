
from .metrics import *

class Buzz:
    def __init__(self, metrics='all', trg_lang='en', lowercase=False):
        if metrics == 'all':
            self.metrics = {
                'bert_score': BERTScore(),
                'bleu_score': BLEUScore(trg_lang=trg_lang, lowercase=lowercase),
                'comet_score': COMETScore(),
                'chrf_score': CHRFScore(lowercase=lowercase),
                'ter_score': TERScore(lowercase=lowercase)
                }
        elif metrics == 'fast':
            self.metrics = {
                'bleu_score': BLEUScore(trg_lang=trg_lang, lowercase=lowercase),
                'chrf_score': CHRFScore(lowercase=lowercase),
                'ter_score': TERScore(lowercase=lowercase)
                }


    def score(self, hyp, ref, src=None, normalize=True, clip=True):
        scores = {}
        for n, m in self.metrics.items():
            hyp, ref = hyp.strip(), ref.strip()
            src = src.strip() if src else None
            _score = m.score(hyp, ref, src, normalize=normalize)[n]['score']
            if clip:
                _score = max (0, min(_score, 100))
            scores[n] = _score
        return scores
