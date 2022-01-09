
from lightyear.metrics import *


class Buzz:
    def __init__(self):
        self.metrics = {
            'bert_score': BERTScore(),
            'bleu_score': BLEUScore(),
            'comet_score': COMETScore()
            }

    def score(self, hyp, ref, normalize=True, clip=True):
        scores = {}
        for n, m in self.metrics.items():
            _score = m.score(hyp, ref, normalize=normalize)[n]['score']
            if clip:
                _score = min(_score, 100)
            scores[n] = _score
        return scores
