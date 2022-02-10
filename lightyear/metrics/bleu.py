import sacrebleu

class BLEUScore:
    def __init__(self, trg_lang='en', lowercase=True):
        self.scorer = sacrebleu.BLEU(trg_lang=trg_lang, effective_order=True, lowercase=lowercase)
    def score(self,  hyp: str, ref: str, src=None, normalize=True, *args, **kwargs):
        this_score = self.scorer.sentence_score(hyp, [ref])
        return {'bleu_score': {'1gram_precision': this_score.precisions[0],
                '2gram_precision': this_score.precisions[1],
                '3gram_precision': this_score.precisions[2],
                '4gram_precision': this_score.precisions[3],
                'brevity_penalty': this_score.bp,
                'score': this_score.score }}

    
class CHRFScore:
    def __init__(self, lowercase=True):
        self.scorer = sacrebleu.CHRF(lowercase=lowercase)
    def score(self, hyp: str, ref: str, src=None, *args, **kwargs):
        this_score = self.scorer.sentence_score(hyp, [ref])
        return {'chrf_score': {this_score.name: this_score.score,
                'score': this_score.score}}
                
                
class TERScore:
    def __init__(self, lowercase=True):
        self.scorer = sacrebleu.TER()
        self.lowercase = lowercase
    def score(self, hyp: str, ref: str, src=None, *args, **kwargs):
        if self.lowercase:
            hyp, ref = hyp.lower(), ref.lower()
        this_score = self.scorer.sentence_score(hyp, [ref])
        return {'ter_score': {'score': this_score.score}}
