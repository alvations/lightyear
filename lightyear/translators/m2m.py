from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from ..utils import prune_model, quantize_model

class M2MTranslator:
    def __init__(self, quantize=True, prune=True, prune_amount=0.2):
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        if quantize:
            self.model = quantize_model(self.model)
        if prune:
            self.prune = prune_model(self.model, prune_amount)

    def translate(self, src_lang, trg_lang, text, max_length=max_length, **kwargs):
        self.tokenizer.src_lang = src_lang
        tokenized_src = self.tokenizer(text, return_tensors="pt")

        generated_tokens = self.model.generate(**tokenized_src,
            forced_bos_token_id=self.tokenizer.get_lang_id(trg_lang), max_length=max_length, **kwargs)
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    def languages(self):
        return set(tokenizer.lang_code_to_id.keys())
