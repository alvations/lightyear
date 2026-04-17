import pathlib
from setuptools import setup, find_packages

long_description = (pathlib.Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
  name = 'lightyear',
  packages = find_packages(),
  version = '0.0.18',
  description = 'Unified MT evaluation toolbox: BLEU, CHRF, TER, BERTScore, SentenceBERTScore, COMET, CometKiwi, MetricX-23/24, PreCOMET, Sentinel-src — built on transformers + torch + sacrebleu.',
  long_description = long_description,
  long_description_content_type = 'text/markdown',
  author = '',
  url = 'https://github.com/alvations/lightyear',
  keywords = ['machine-translation', 'evaluation', 'metrics', 'bleu', 'comet', 'bertscore', 'metricx', 'quality-estimation', 'difficulty-estimation'],
  install_requires = [
    'sacrebleu',
    'torch',
    'transformers',
    'datasets',
    'sentencepiece',
    'sentence-transformers',
  ],
  python_requires = '>=3.10,<3.15',
  classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Text Processing :: Linguistic",
    "Intended Audience :: Science/Research",
  ]
)
