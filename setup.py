from setuptools import setup, find_packages
from smartloop import __version__
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

install_requires = [
    'nltk==3.7',
    'joblib==1.1.1',
    'PyYAML==6.0',
    'scikit-learn~=1.2.2',
    'numpy~=1.23.5'
]

setup(
    name='smartloop',
    description='Natural language processing framework for text processing',
    version=__version__,
    author_email='mehfuz@smartloop.ai',
    author='Smartloop Inc.',
    url='https://github.com/SmartloopAI/sl-core',
    download_url='https://github.com/SmartloopAI/sl-core/archive/refs/tags/1.0.1.tar.gz',
    keywords=['NLP', 'framework', 'tensorflow', 'smartloop'],
    packages=find_packages(exclude=['tests*']),
    license='LICENSE.txt',
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        "Topic :: Software Development :: Libraries",
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
)
