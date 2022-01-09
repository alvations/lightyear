import sacrebleu

class BLEUScore:
    def __init__(self):
        self.scorer = sacrebleu.BLEU(effective_order=True)
    def score(self,  hyp: str, ref: str, normalize=True):
        this_score = self.scorer.sentence_score(hyp, [ref])
        return {'bleu_score': {'1gram_precision': this_score.precisions[0],
                '2gram_precision': this_score.precisions[1],
                '3gram_precision': this_score.precisions[2],
                '4gram_precision': this_score.precisions[3],
                'brevity_penalty': this_score.bp,
                'score': this_score.score }}
