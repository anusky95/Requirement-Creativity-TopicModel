# Requirement Creativity TopicModel
This is an implementation of Topic Modeling using Latent Dirichlet Allocation, Word2Vector and Part of Speech Tagging
in order to cluster the data set into different topics based on the content of the data.

# Getting Started
1.The python version that is used in this project is 3.6.1.
2.Check if PIP is installed.Run the following command in the command line if not installed.
  
##### python -m pip install -U pip setuptools

3.Upon installing PIP we can install the other packages using the PIP command.
4.Before installation of every package upgrade the PIP using the following two commands

##### sudo apt-get install build-essential gfortran libatlas-base-dev python-pip python-dev
##### sudo pip install --upgrade pip

5.The packages used in the project are
  
  Numpy
  ##### sudo pip install numpy

  Scipy
  ##### sudo pip install scipy

  Scikit-learn
  ##### pip install -U scikit-learn

  Gensim
  ##### pip install --upgrade gensim

  NLTK
  ##### sudo pip install -U nltk

6. Import all the packages mentioned above along with 
##### sys  
##### argparse

### Input
The input for TopicModeling.py can be arguments as File path or string literal.
Use W2V as an argument for the word2vector implementation.
Use File Argument for LDA implementation.
The input for Verb_Noun_Extracter.py are passed as arguments(Path to a file).

