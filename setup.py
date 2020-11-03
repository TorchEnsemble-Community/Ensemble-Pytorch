from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from README.md
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]

cmdclass = {}

setup(
    name='Ensemble-PyTorch',
    version='0.0.1',
    author='AaronX121',
    
    description=('Implementations of scikit-learn like ensemble methods in'
                 ' Pytorch'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AaronX121/Ensemble-Pytorch',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    keywords='Ensemble Learning',
    
    packages=find_packages(),
    cmdclass=cmdclass,
    install_requires=install_requires,
)