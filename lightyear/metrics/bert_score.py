import torch
from torch import nn
from torch.nn.utils import prune

from bert_score import BERTScorer

from ..utils import prune_model, quantize_model


class BERTScore:
    def __init__(self, modelname="bert-base-multilingual-cased", cuda=True):
        _cuda = "cuda" if cuda and torch.cuda.is_available() else "cpu"
        self.model = BERTScorer(model_type=modelname, use_fast_tokenizer=True,
                                device=_cuda)
        self.model._model = prune_model(quantize_model(self.model._model))


    def score(self, hyp: str, ref: str, src=None, normalize=True, *args, **kwargs):
        with torch.no_grad():
            precision, recall, f1 = map(float, self.model.score([hyp], [[ref]]))
        score = f1 * 100 if normalize else f1
        return {'bert_score':{'precision': precision,
                'recall': recall, 'f1':f1,
                'score': score}}

    def score_batch(self, hyps, refs, srcs=None, normalize=True, *args, **kwargs):
        with torch.no_grad():
            pred_scores = self.model.score(hyps, [[r] for r in refs])
        precisions, recalls, f1s = [list(map(float, s)) for s in pred_scores]
        scores = [f1*100 for f1 in f1s] if normalize else f1s
        return {'bert_score':{'precision': precisions,
                'recall': recalls, 'f1':f1s,
                'score': scores}}