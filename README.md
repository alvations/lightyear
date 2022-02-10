# lightyear

To infinity and beyond...

Install
====

```
pip install -U lightyear
```

Usage
====

[in]:

```python
from lightyear import Buzz

hyp = 'The dog bit the man.'
ref = 'The dog had bit the man.'

buzz = Buzz()
buzz.score(hyp, ref)
```

[out]:

```
{'bert_score': 98.19023013114929,
 'bleu_score': 51.15078115793242,
 'comet_score': 100,
 'chrf_score': 67.64711450656517,
 'ter_score': 16.666666666666664}
```


[in]:

```
from lightyear import Buzz

buzz = Buzz(metrics='fast', lowercase=True)

with open('hyp.txt') as hfin, open('ref.txt') as rfin:
    for hyp, ref in zip(hfin, rfin):
        print(buzz.score(hyp, ref))

```


```
{'bleu_score': 11.631736348831648, 'chrf_score': 25.66796545720479, 'ter_score': 100.0}
{'bleu_score': 21.3643503198117, 'chrf_score': 38.2883972133884, 'ter_score': 75.0}
```

