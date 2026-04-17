import torch
from sentence_transformers import SentenceTransformer, util


class SentenceBERTScore:
    def __init__(self, modelname='sentence-transformers/stsb-xlm-r-multilingual', cuda=True):
        device = "cuda" if cuda and torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(modelname, device=device)

    def score(self, hyp: str, ref: str, src: str = None, normalize=True, *args, **kwargs):
        emb = self.model.encode([hyp, ref], convert_to_tensor=True, normalize_embeddings=True)
        score = float(util.cos_sim(emb[0], emb[1]))
        return {'sentbert_score': {'score': score * 100 if normalize else score}}

    def score_batch(self, hyps, refs, srcs=None, normalize=True, *args, **kwargs):
        h = self.model.encode(hyps, convert_to_tensor=True, normalize_embeddings=True)
        r = self.model.encode(refs, convert_to_tensor=True, normalize_embeddings=True)
        scores = (h * r).sum(dim=1).tolist()
        scores = [s * 100 for s in scores] if normalize else scores
        return {'sentbert_score': {'score': scores}}
