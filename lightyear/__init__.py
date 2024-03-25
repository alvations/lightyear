

from .metrics import *

class Buzz:
    def __init__(self, metrics='all', trg_lang='en', lowercase=False):
        """
        metrics = [default:all, fast, chrf, bleu, ter, sentbert, bert, comet]
        lowercase  = [default:False, True]
        """
        if metrics == 'all':
            self.metrics = {
                'bert_score': BERTScore(),
                'bleu_score': BLEUScore(trg_lang=trg_lang, lowercase=lowercase),
                'comet_score': COMETScore(),
                'sentbert_score': SentenceBERTScore(),
                'chrf_score': CHRFScore(lowercase=lowercase),
                'ter_score': TERScore(lowercase=lowercase)
                }
        elif metrics == 'fast':
            self.metrics = {
                'bleu_score': BLEUScore(trg_lang=trg_lang, lowercase=lowercase),
                'chrf_score': CHRFScore(lowercase=lowercase),
                'ter_score': TERScore(lowercase=lowercase)
                }
        elif metrics == "chrf":
            self.metrics = {
                'chrf_score': CHRFScore(lowercase=lowercase),
            }
        elif metrics == "bleu":
            self.metrics = {
                'bleu_score': BLEUScore(trg_lang=trg_lang, lowercase=lowercase),
            }
        elif metrics == "ter":
            self.metrics = {
                'ter_score': TERScore(lowercase=lowercase)
            }
        elif metrics == "sentbert":
            self.metrics = {
                'sentbert_score': SentenceBERTScore(),
            }
        elif metrics == "bert":
            self.metrics = {
                'bert_score': BERTScore(),
            }
        elif metrics == "comet":
            self.metrics = {
                'comet+_score': COMETScore(),
            }
    def score(self, hyp, ref, src=None, normalize=True, clip=True):
        # Processing Inputs
        hyp, ref = hyp.strip(), ref.strip()
        src = src.strip() if src else None

        scores = {}
        for n, m in self.metrics.items():
            _score = m.score(hyp, ref, src, normalize=normalize)[n]['score']
            if clip:
                _score = max(0, min(_score, 100))
            scores[n] = _score
        return scores

    def score_parallel(self, hyp, ref, src=None, normalize=True, clip=True):
        raise NotImplementedError
        """
        Somehow this took longer?!
        """
        from functools import partial
        from multiprocessing import Process, Manager
        
        # Processing Inputs
        hyp, ref = hyp.strip(), ref.strip()
        src = src.strip() if src else None

        # Multiprocessing stuff.
        manager = Manager()
        outputs = manager.dict()
        jobs = []

        self.metrics_parallel = { n:partial(m.score, hyp, ref, src, normalize=normalize)
            for n, m in self.metrics.items() } 
        
        # Gets scores for individual metrics.
        for n, m in self.metrics_parallel.items():
            p = Process(target=m)
            p.start()
            jobs.append((n, p))
        for n, p in jobs:
            p.join()

        return outputs
            
