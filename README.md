# lightyear

To infinity and beyond...

Install
====

```
pip install -U lightyear
```

Usage
====

## Typical Hyp + Ref Setup

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

## Hyp + Ref + Src setup

[in]:

```python
from lightyear import Buzz

src = 'The dog bit the man.'
hyp = 'Hund Mann gebissen.'
ref = 'Der Hund hatte den Mann gebissen.'

buzz = Buzz(metrics='all', trg_lang='de', lowercase=True)

print(buzz.score(hyp, ref))
print(buzz.score(hyp, ref, src))
```

[out]:

```
{'bert_score': 89.7886335849762, 'bleu_score': 30.18153515504547, 'comet_score': 17.846399545669556, 'chrf_score': 51.512405379593574, 'ter_score': 50.0}
{'bert_score': 89.7886335849762, 'bleu_score': 30.18153515504547, 'comet_score': 6.618038564920425, 'chrf_score': 51.512405379593574, 'ter_score': 50.0}
```

## Only fast metrics


[in]:

```python
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

## Using Machine Translators

[in]:

```python
from lightyear.translators import HelsinkiMarianTranslator
from lightyear.translators import M2MTranslator

hema = HelsinkiMarianTranslator()
print(hema.translate('en', 'de', 'I am pregnant'))
print(hema.translate('de', 'zh', 'Ich bin schwanger'))

print()

m2m = M2MTranslator()
print(m2m.translate('en', 'de', 'I am pregnant'))
print(m2m.translate('de', 'zh', 'Ich bin schwanger'))
```

[out]:

```
Ich bin schwanger
我怀孕了 我怀孕了

Ich bin schwanger.
我怀孕了
```
