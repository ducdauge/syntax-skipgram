# syntax-skipgram

Tensorflow implementation of **syntax-based Word2Vec**. Parts of the code are sourced from the [TensorFlow word2vec example](https://www.tensorflow.org/tutorials/word2vec). In spirit the algorithm is similar to [Word2VecF](https://bitbucket.org/yoavgo/word2vecf), but enables people fluent in the TF library to customise it more easily.

The contexts taken into account during training are the parents and children of words in dependency trees. The implementation is optimized for speed through *asynchronouns* processes to load the data and perform forward/back-propagation.

## Requirements

* tensorflow >= 1.4.1
* threading

## Setup

Edit the main file or add command-line flags to specify the treebank file and the hyperparameters. Then the execution is as simple as:

```
python skipgram_tf_asynch.py
```
