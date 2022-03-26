import torch
from torch import nn
from torch.nn.utils import prune

from bert_score import BERTScorer


class BERTScore:
    def __init__(self, modelname="bert-base-multilingual-cased"):
        self.model = BERTScorer(model_type=modelname, use_fast_tokenizer=True)

        self.model._model = torch.quantization.quantize_dynamic(
            self.model._model,
            {nn.LayerNorm, nn.Linear, nn.Dropout},
            dtype=torch.qint8)

        for module in self.model._model.modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, 'weight', amount=0.2)
                prune.remove(module, 'weight')

    def score(self, hyp: str, ref: str, src=None, normalize=True, *args, **kwargs):
        precision, recall, f1 = map(float, self.model.score([hyp], [[ref]]))
        score = f1 * 100 if normalize else f1
        return {'bert_score':{'precision': precision,
                'recall': recall, 'f1':f1,
                'score': score}}
