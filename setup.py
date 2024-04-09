from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys, os, setuptools

from setuptools import setup, Extension

path_to_risk_c= 'optuna/samplers/_hyperjump/'

requires = ['torch','GPy','transformers','mpmath','gpytorch','scikit-learn']  

# Define custom install command to run gcc command
class CustomInstallCommand(install):
    def run(self):
        # Run gcc command or any other custom commands
        #subprocess.call(['gcc', '-o', 'output_file', 'input_file.c'])
        subprocess.run('gcc -shared -fPIC -o ' + path_to_risk_c + '/func.so ' + path_to_risk_c + '/func.c', shell=True,)
        # Continue with regular installation
        install.run(self)


setuptools.setup(name='optuna',
                version='3.6.0.dev0',
                author='Pedro Mendes',
                author_email='pgmendes@andrew.cmu.edu',
                description='Optuna with HyperJump',
                keywords='Machine Learining, Bayesian Optimization, HyperBand, Hyper-parameter tuning',
                packages=setuptools.find_packages(),
                install_requires=requires,
                ext_modules=[Extension('optuna.hyperjump', sources=[path_to_risk_c + '/func.c'])],
                cmdclass={'install': CustomInstallCommand},)



