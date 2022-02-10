from setuptools import setup

setup(
  name = 'lightyear',
  packages = ['lightyear'],
  version = '0.0.3',
  description = 'lightyear',
  long_description = '',
  author = '',
  url = 'https://github.com/alvations/lightyear',
  keywords = [],
  install_requires = ['bert_score', 'sacrebleu', 'unbabel-comet', 'pytorch_lightning'],
  classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ]
)
