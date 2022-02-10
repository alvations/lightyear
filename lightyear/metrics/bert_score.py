import bert_score


class BERTScore:
    def score(self, hyp: str, ref: str, src=None, normalize=True, *args, **kwargs):
        precision, recall, f1 = map(float, bert_score.score([hyp], [[ref]], lang='en'))
        score = f1 * 100 if normalize else f1
        return {'bert_score':{'precision': precision,
                'recall': recall, 'f1':f1,
                'score': score}}
