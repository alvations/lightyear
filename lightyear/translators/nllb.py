
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..utils import prune_model, quantize_model


class NLLBTranslator:
    def __init__(self, quantize=True, prune=True, prune_amount=0.2):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        if quantize:
            self.model = quantize_model(self.model)
        if prune:
            self.prune = prune_model(self.model, prune_amount)

    def translate(self, src_lang, trg_lang, text, max_length=1000, **kwargs):
        self.tokenizer.src_lang = src_lang
        tokenized_src = self.tokenizer(text, return_tensors="pt")

        generated_tokens = self.model.generate(**tokenized_src, 
        	forced_bos_token_id=self.tokenizer.lang_code_to_id[trg_lang], max_length=max_length, **kwargs)

        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


    def languages(self):
        return set(self.tokenizer.lang_code_to_id.keys())

