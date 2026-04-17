import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


DEFAULT_LAYERS = {
    "bert-base-uncased": 8,
    "bert-large-uncased": 18,
    "bert-base-cased": 10,
    "bert-large-cased": 17,
    "bert-base-multilingual-cased": 9,
    "roberta-base": 10,
    "roberta-large": 17,
    "xlm-roberta-base": 9,
    "xlm-roberta-large": 17,
    "distilbert-base-uncased": 5,
    "distilbert-base-multilingual-cased": 5,
}


class BERTScore:
    def __init__(self, modelname="bert-base-multilingual-cased", layer=None, cuda=True):
        self.device = "cuda" if cuda and torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=True)
        self.model = AutoModel.from_pretrained(modelname, output_hidden_states=True).to(self.device)
        self.model.eval()
        self.layer = layer if layer is not None else DEFAULT_LAYERS.get(modelname, -1)

    def _embed(self, texts):
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        hidden = out.hidden_states[self.layer]
        hidden = F.normalize(hidden, p=2, dim=-1)
        mask = enc.attention_mask.clone()
        mask[:, 0] = 0
        idx = mask.sum(dim=1) + (enc.attention_mask[:, 0].long())
        for i, j in enumerate((idx - 1).tolist()):
            if j >= 0:
                mask[i, j] = 0
        return hidden, mask

    def _prf(self, h_emb, h_mask, r_emb, r_mask):
        sim = torch.bmm(h_emb, r_emb.transpose(1, 2))
        r_mask_b = r_mask.unsqueeze(1).bool()
        h_mask_b = h_mask.unsqueeze(2).bool()

        sim_p = sim.masked_fill(~r_mask_b, -1.0)
        prec_per_tok = sim_p.max(dim=2).values
        prec_per_tok = prec_per_tok.masked_fill(~h_mask.bool(), 0.0)
        P = prec_per_tok.sum(dim=1) / h_mask.sum(dim=1).float().clamp(min=1)

        sim_r = sim.masked_fill(~h_mask_b, -1.0)
        rec_per_tok = sim_r.max(dim=1).values
        rec_per_tok = rec_per_tok.masked_fill(~r_mask.bool(), 0.0)
        R = rec_per_tok.sum(dim=1) / r_mask.sum(dim=1).float().clamp(min=1)

        F1 = 2 * P * R / (P + R + 1e-12)
        return P, R, F1

    def score(self, hyp: str, ref: str, src=None, normalize=True, *args, **kwargs):
        h_emb, h_mask = self._embed([hyp])
        r_emb, r_mask = self._embed([ref])
        P, R, F1 = self._prf(h_emb, h_mask, r_emb, r_mask)
        p, r, f = float(P[0]), float(R[0]), float(F1[0])
        return {'bert_score': {'precision': p, 'recall': r, 'f1': f,
                               'score': f * 100 if normalize else f}}

    def score_batch(self, hyps, refs, srcs=None, normalize=True, *args, **kwargs):
        h_emb, h_mask = self._embed(hyps)
        r_emb, r_mask = self._embed(refs)
        P, R, F1 = self._prf(h_emb, h_mask, r_emb, r_mask)
        ps, rs, fs = P.tolist(), R.tolist(), F1.tolist()
        scores = [f * 100 for f in fs] if normalize else fs
        return {'bert_score': {'precision': ps, 'recall': rs, 'f1': fs, 'score': scores}}
