from setuptools import setup, find_packages

setup(
  name = 'lightyear',
  packages = find_packages(),
  version = '0.0.14',
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
