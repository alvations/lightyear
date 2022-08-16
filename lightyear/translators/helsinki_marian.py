from itertools import chain

from transformers import MarianMTModel, MarianTokenizer

from ..utils import prune_model, quantize_model


supported_langs = {
    'af': {'sv', 'fi', 'ru', 'nl', 'de', 'es', 'fr', 'en'},
    'am': {'sv'},
    'ar': {'he', 'ru', 'de', 'it', 'el', 'pl', 'tr', 'es', 'fr', 'en'},
    'az': {'es', 'tr', 'en'},
    'be': {'es'},
    'bg': {'uk', 'sv', 'fi', 'ru', 'de', 'it', 'tr', 'es', 'fr', 'en'},
    'bn': {'en'},
    'ca': {'uk', 'nl', 'pt', 'it', 'de', 'es', 'fr', 'en'},
    'ceb': {'sv', 'fi', 'es', 'fr', 'en'},
    'cs': {'uk', 'sv', 'fi', 'de', 'fr', 'en'},
    'cy': {'en'},
    'da': {'fi', 'ru', 'de', 'es', 'no', 'fr', 'en'},
    'de': {'af', 'ar', 'bg', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es',
            'et', 'fi', 'fr', 'ha', 'he', 'hr', 'ht', 'hu', 'ig', 'ilo',
            'is', 'it', 'ln', 'lt', 'ms', 'nl', 'no', 'pl', 'tl', 'uk',
            'vi'},
    'el': {'sv', 'fi', 'fr', 'ar'},
    'en': {'af', 'ar', 'az', 'bg', 'ca', 'ceb', 'cs', 'cy', 'da', 'de',
            'el', 'es', 'et', 'fi', 'fr', 'ga', 'gl', 'ha', 'he', 'hi',
            'ht', 'hu', 'hy', 'id', 'ig', 'ilo', 'is', 'it', 'lg', 'ln',
            'mg', 'mk', 'ml', 'mr', 'nl', 'ro', 'ru', 'sk', 'sq', 'ss',
            'sv', 'sw', 'tl', 'tn', 'uk', 'ur', 'vi', 'xh', 'zh'},
    'es': {'af', 'ar', 'bg', 'ca', 'ceb', 'cs', 'da', 'de', 'el', 'en',
            'es', 'et', 'fi', 'fr', 'gl', 'ha', 'he', 'hr', 'ht', 'id',
            'ig', 'ilo', 'is', 'it', 'ln', 'lt', 'mk', 'nl', 'no', 'pl',
            'ro', 'ru', 'sl', 'tl', 'tn', 'uk', 'vi', 'xh', 'yo'},
    'et': {'sv', 'fi', 'ru', 'de', 'es', 'fr', 'en'},
    'fi': {'af', 'bg', 'ceb', 'cs', 'de', 'el', 'en', 'es', 'et', 'fi',
            'fr', 'ha', 'he', 'hr', 'ht', 'hu', 'id', 'ig', 'ilo', 'is',
            'it', 'lg', 'ln', 'lv', 'mg', 'mk', 'nl', 'no', 'ro', 'ru',
            'sk', 'sl', 'sq', 'sv', 'sw', 'tn', 'tr', 'uk', 'xh', 'yo'},
    'fr': {'af', 'ar', 'bg', 'ca', 'ceb', 'de', 'el', 'en', 'es', 'ha',
            'he', 'hr', 'ht', 'hu', 'id', 'ig', 'ilo', 'lg', 'ln', 'ms',
            'no', 'pl', 'ro', 'ru', 'sk', 'sl', 'sv', 'tl', 'tn', 'uk',
            'vi', 'xh', 'yo'},
    'ga': {'en'},
    'gl': {'es', 'pt', 'en'},
    'ha': {'sv', 'fi', 'es', 'fr', 'en'},
    'he': {'uk', 'sv', 'fi', 'ru', 'it', 'de', 'ar', 'es'},
    'hi': {'ur', 'en'},
    'hr': {'fi', 'sv', 'es', 'fr'},
    'ht': {'sv', 'fi', 'es', 'fr', 'en'},
    'hu': {'uk', 'sv', 'fi', 'de', 'fr', 'en'},
    'hy': {'ru', 'en'},
    'id': {'sv', 'fi', 'es', 'fr', 'en'},
    'ig': {'sv', 'fi', 'de', 'es', 'fr', 'en'},
    'ilo': {'sv', 'fi', 'de', 'es', 'en'},
    'is': {'sv', 'fi', 'de', 'it', 'es', 'fr', 'en'},
    'it': {'ar', 'bg', 'ca', 'de', 'en', 'es', 'fr', 'is', 'lt', 'ms',
            'sv', 'uk', 'vi'},
    'ja': {'ar', 'bg', 'da', 'de', 'en', 'es', 'fi', 'fr', 'he', 'hu',
            'it', 'ms', 'nl', 'pl', 'pt', 'ru', 'sv', 'tr', 'vi'},
    'ka': {'ru', 'en'},
    'ko': {'sv', 'fi', 'ru', 'de', 'hu', 'es', 'fr', 'en'},
    'lg': {'sv', 'fi', 'es', 'fr', 'en'},
    'ln': {'fr', 'es', 'de', 'en'},
    'lt': {'sv', 'ru', 'it', 'de', 'pl', 'tr', 'es', 'fr'},
    'lv': {'sv', 'fi', 'ru', 'es', 'fr', 'en'},
    'mg': {'es', 'en'},
    'mk': {'fi', 'es', 'fr', 'en'},
    'ml': {'en'},
    'mr': {'en'},
    'ms': {'fr', 'it', 'ms', 'de'},
    'nl': {'uk', 'sv', 'fi', 'af', 'es', 'no', 'ca', 'fr', 'en'},
    'no': {'da', 'de', 'es', 'fi', 'fr', 'nl', 'no', 'pl', 'ru', 'sv',
            'uk'},
    'pa': {'en'},
    'pl': {'uk', 'sv', 'de', 'ar', 'es', 'no', 'lt', 'fr', 'en'},
    'pt': {'ca', 'uk', 'tl', 'gl'},
    'ro': {'sv', 'fi', 'fr'},
    'ru': {'af', 'ar', 'bg', 'da', 'en', 'es', 'et', 'fi', 'fr', 'he',
            'hy', 'lt', 'lv', 'no', 'sl', 'sv', 'uk', 'vi'},
    'sk': {'sv', 'fi', 'es', 'fr', 'en'},
    'sl': {'uk', 'sv', 'fi', 'ru', 'es', 'fr'},
    'sq': {'sv', 'es', 'en'},
    'ss': {'en'},
    'sv': {'af', 'bg', 'ceb', 'cs', 'el', 'en', 'es', 'et', 'fi', 'fr',
            'ha', 'he', 'hr', 'ht', 'hu', 'id', 'ig', 'ilo', 'is', 'lg',
            'ln', 'lv', 'nl', 'no', 'ro', 'ru', 'sk', 'sl', 'sq', 'sv',
            'th', 'tn', 'uk', 'xh', 'yo'},
    'th': {'fr', 'en'},
    'tl': {'pt', 'es', 'de', 'en'},
    'tn': {'sv', 'es', 'fr', 'en'},
    'tr': {'uk', 'az', 'sv', 'ar', 'es', 'lt', 'fr', 'en'},
    'uk': {'bg', 'ca', 'cs', 'de', 'en', 'es', 'fi', 'fr', 'he', 'hu',
            'it', 'nl', 'no', 'pl', 'pt', 'ru', 'sl', 'sv', 'tr'},
    'ur': {'en'},
    'vi': {'ru', 'it', 'de', 'es', 'fr', 'en'},
    'xh': {'sv', 'es', 'fr', 'en'},
    'yo': {'sv', 'fi', 'es', 'fr', 'en'},
    'zh': {'bg', 'de', 'en', 'fi', 'he', 'it', 'ms', 'nl', 'sv', 'uk',
            'vi'}
}


class HelsinkiMarianTranslator:
    def __init__(self):
        self._languages = set(supported_langs.keys()) | set(chain(*supported_langs.values()))
        self._models = {}

    def load_model(self, src_lang, trg_lang,
                   quantize=True, prune=True, prune_amount=0.2):
        # Check if language pair is pre-loaded.
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{trg_lang}"
        if model_name in self._models:
            tokenizer, model = self._models[model_name]
        else:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            if quantize:
                self.model = quantize_model(self.model)
            if prune:
                self.prune = prune_model(self.model, prune_amount)
            self._models[model_name] = tokenizer, model
        return tokenizer, model

    def translate(self, src_lang, trg_lang, text, pivot=True, max_length=max_length, **kwargs):
        processes = []
        if src_lang in supported_langs and trg_lang in supported_langs[src_lang]:
            processes.append(self.load_model(src_lang, trg_lang))
        elif src_lang in self._languages and trg_lang in self._languages and pivot:
            processes.append(self.load_model(src_lang, 'en'))
            processes.append(self.load_model('en', trg_lang))

        translated_text = text
        for tokenizer, model in processes:
            translated = model.generate(**tokenizer([translated_text], return_tensors="pt", padding=True),
                 max_length=max_length, **kwargs)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text

    def languages():
        return self._languages
