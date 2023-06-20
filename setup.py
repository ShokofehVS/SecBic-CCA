from setuptools import setup, find_packages
from setuptools import find_namespace_packages, setup, find_packages

setup(    name='SecBiclib',
    version='0.1',
    description='Library of biclustering algorithms, evaluation measures and dataset',
    url='https://github.com/ShokofehVS/SecBic-CCA',
    author='...',
    author_email='...',
    license='MIT',
    packages=find_namespace_packages(include=['SecBiclib.*']),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'numpy>=1.16.5',
        'pandas>=0.25.3',
        'scikit-learn>=0.22.2.post1',
        'matplotlib>=3.1.1'
    ])