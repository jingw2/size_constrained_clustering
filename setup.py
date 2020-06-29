from setuptools import setup, find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='psoco',
      description='partical swarm optimization constraint optimization solver',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/jingw2/solver/tree/master/psoco',
      author='Jing Wang',
      author_email='jingw2@foxmail.com',
      license='MIT',
      packages=find_packages(),
      python_requires='>=3.6')