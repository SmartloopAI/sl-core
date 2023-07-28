import sys
from setuptools import setup, find_packages

install_requires=[
    'nltk==3.7',
    'joblib==1.1.1',
    'PyYAML==6.0',
    'scikit-learn~=1.2.2'
]

setup(
    name='sl_core',
    description='Natural language processing framework for text processing',
    version='1.0.1',
    author_email='mehfuz@smartloop.ai',
    author='Smartloop Inc.',
    keywords=['NLP', 'framework', 'tensorflow'],
    packages=find_packages(exclude=['tests*']),
    license='LICENSE.txt',
    install_requires=install_requires
)
