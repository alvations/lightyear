
from lightyear.metrics import *


class Buzz:
    def __init__(self, metrics='all', lowercase=False):
        if metrics == 'all':
            self.metrics = {
                'bert_score': BERTScore(),
                'bleu_score': BLEUScore(lowercase=lowercase),
                'comet_score': COMETScore(),
                'chrf_score': CHRFScore(lowercase=lowercase),
                'ter_score': TERScore(lowercase=lowercase)
                }
        elif metrics == 'fast':
            self.metrics = {
                'bleu_score': BLEUScore(lowercase=lowercase),
                'chrf_score': CHRFScore(lowercase=lowercase),
                'ter_score': TERScore(lowercase=lowercase)
                }
            

    def score(self, hyp, ref, src=None, normalize=True, clip=True):
        scores = {}
        for n, m in self.metrics.items():
            _score = m.score(hyp.strip(), ref.strip(), src.strip(), normalize=normalize)[n]['score']
            if clip:
                _score = max (0, min(_score, 100))
            scores[n] = _score
        return scores
