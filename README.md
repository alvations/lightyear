# lightyear

To infinity and beyond...

Usage
====

[in]:

```
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
 'comet_score': 100}
```
