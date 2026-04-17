from setuptools import setup, find_packages

setup(
  name = 'lightyear',
  packages = find_packages(),
  version = '0.0.17',
  description = 'lightyear',
  long_description = '',
  author = '',
  url = 'https://github.com/alvations/lightyear',
  keywords = [],
  install_requires = [
    'sacrebleu',
    'torch',
    'transformers',
    'datasets',
    'sentencepiece',
    'sentence-transformers',
  ],
  classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ]
)
