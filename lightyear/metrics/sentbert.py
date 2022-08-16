
import torch
from transformers import AutoTokenizer, AutoModel

from ..utils import prune_model, quantize_model


class SentenceBERTScore:
    def __init__(self, modelname='sentence-transformers/stsb-xlm-r-multilingual', cuda=True):
        _cuda = "cuda" if cuda and torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.model = prune_model(quantize_model(
            AutoModel.from_pretrained(modelname)
        )).to(_cuda)

        self.cosine = torch.nn.CosineSimilarity(dim=0)

    def score(self, hyp: str, ref: str, src: str=None, normalize=True, *args, **kwargs):
        # Process inputs.       
        encoded_input = self.tokenizer([hyp, ref], 
            padding=True, truncation=True, return_tensors='pt'
        )

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, mean pooling.
        s1, s2 = self.mean_pooling(model_output, encoded_input['attention_mask'])
        score = float(self.cosine(s1, s2))
        score = score * 100 if normalize else score
        return {'sentbert_score':{'score': score}}

    def mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging"""
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
