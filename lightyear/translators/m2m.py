from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from torch import nn
from torch.nn.utils import prune


class M2MTranslator:
    def __init__(self, quantize=True, prune=True, prune_amount=0.2):
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        if quantize:
            self.model = torch.quantization.quantize_dynamic(model,
                {nn.LayerNorm, nn.Linear}, dtype=torch.qint8)
        if prune:
            for module in model_quantized.modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, 'weight', amount=prune_amount)
                    prune.remove(module, 'weight')

    def translate(self, src_lang, trg_lang, text):
        tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        tokenizer.src_lang = src_lang
        tokenized_src = tokenizer(text, return_tensors="pt")

        generated_tokens = self.model.generate(**tokenized_src,
                            forced_bos_token_id=tokenizer.get_lang_id(trg_lang))
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    def languages(self):
        return set(tokenizer.lang_code_to_id.keys())
