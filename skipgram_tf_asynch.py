#################################################################
# Parts of the code are sourced from the TF word2vec example:   #
# https://www.tensorflow.org/tutorials/word2vec                 #
#################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy
import os

import tensorflow as tf
from utils import TreebankReader, cw_Tokenizer
import logging
import threading

FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

flags = tf.flags
flags.DEFINE_string ('corpus_path', 'file.conll', 'full path for conll file')
flags.DEFINE_integer('embedding_size', 300, 'embedding size')
flags.DEFINE_integer('min_freq', 100, 'minimun word frequency (discarded if lower)')
flags.DEFINE_integer('negative_samples', 100, 'number of negative samples for contrastive loss')
flags.DEFINE_integer('epochs', 10, 'number of epochs')
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_float('learning_rate', 0.2, 'learning rate')
flags.DEFINE_boolean('labeled', False, 'concatenate dependency relations to WEs?')
FLAGS = flags.FLAGS

class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """
    def __init__(self, batch_size, iterator):
        self.batch_size= batch_size
        self.iterator = iterator
        self.words_ph = tf.placeholder(tf.int32, shape=[None,])
        self.contexts_ph = tf.placeholder(tf.int32, shape=[None,])
        # The actual queue of data.
        capacity = batch_size * 3
        self.queue = tf.RandomShuffleQueue(shapes=[[], []],
                                           dtypes=[tf.int32, tf.int32],
                                           capacity=capacity,
                                           min_after_dequeue=batch_size)

        self.enqueue_op = self.queue.enqueue_many([self.words_ph, self.contexts_ph])

    def get_inputs(self):
        """
        Return's tensors containing a batch of words and contexts
        """
        words_batch, contexts_batch = self.queue.dequeue_many(self.batch_size)
        return words_batch, contexts_batch

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for i, (w, c) in enumerate(self.iterator):
            sess.run(self.enqueue_op, feed_dict={self.words_ph:w, self.contexts_ph:c})

    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads


class SkipGram(object):
    """ WordContextModel implements a word2vec-like model trained with
    negative sampling. The word indices (of both the target word and
    its context) are passed to two distinct networks and the goal is to
    learn to discriminate between `true' and `false' contexts
    Parameters:
    corpus_path (str): the path to a text file
    embedding_size (int): the size of the neural embeddings
    labeled (boolean): whether including dependency labels in contexts
    """

    def __init__(self, corpus_path, embedding_size=300, min_freq=100,
                 negative_samples=100, epochs=10, batch_size=16, learning_rate=0.2,
		 labeled=False, *args, **kwargs):
        self.corpus_path = corpus_path
        self.embedding_size = embedding_size
        self.min_freq = min_freq
        self.negative_samples=negative_samples
        self.epochs=epochs
        self.batch_size = batch_size
        self.learning_rate = 0.2
        self.tokenizer = None
        self.corpus = None
        self.model = None
        self.labeled = labeled
        super(SkipGram, self).__init__(*args, **kwargs)

    def tokenize_corpus(self):
        """ Tokenize the corpus using Keras helper classes"""
        self.corpus = TreebankReader(self.corpus_path, labeled=self.labeled)
        logging.info('Tokenizing the corpus')
        self.tokenizer = cw_Tokenizer(min_freq=self.min_freq)
        for words in self.corpus.generate(single=True):
            self.tokenizer.fit_on_texts(words)
        self.tokenizer.finalize_dict(self.corpus._wordcounts)

    def skipgrams_generator(self, sampling_table):
        words = []
        contexts = []
        for s in self.corpus.generate():
            for w, c in s:
                wi = self.tokenizer.word_index.get(w)
                if not wi:
                    continue
                ci = self.tokenizer.context_index.get(c, 0)
                if sampling_table is not None:
                    if sampling_table[ci] < random.random():
                        continue
                words.append(wi)
                contexts.append(ci)
                if len(words) == self.batch_size:
                    yield (numpy.array(words, dtype=numpy.int32),
                           numpy.array(contexts, dtype=numpy.int32))
                    words = []
                    contexts = []

    @staticmethod
    def _make_sampling_table(size, sampling_factor=1e-5):
        gamma = 0.577
        rank = numpy.arange(size)
        rank[0] = 1
        inv_fq = rank * (numpy.log(rank) + gamma) + 0.5 - 1. / (12. * rank)
        f = sampling_factor * inv_fq

        return numpy.minimum(1., f / numpy.sqrt(f))

    def prepare_model(self):
        vocab_length = len(self.tokenizer.word_index) + 1
        context_frequencies = list(self.tokenizer.context_frequencies)
        graph = tf.Graph()

        with graph.as_default():

          # Input data.
          sampling_table = self._make_sampling_table(vocab_length)
          self.custom_runner = CustomRunner(self.batch_size, self.skipgrams_generator(sampling_table))
          words_ph, contexts_ph = self.custom_runner.get_inputs()

          # Ops and variables pinned to the CPU because of missing GPU implementation
          with tf.device('/cpu:0'):
            # Declare all variables we need.
            # Embedding: [vocab_size, emb_dim]
            init_width = 0.5 / self.embedding_size
            emb = tf.Variable(
                tf.random_uniform(
                    [vocab_length, self.embedding_size], -init_width, init_width),
                name="emb")
            self._emb = emb

            # Softmax weight: [vocab_size, emb_dim]. Transposed.
            sm_w_t = tf.Variable(
                tf.zeros([vocab_length, self.embedding_size]),
                name="sm_w_t")

            # Softmax bias: [vocab_size].
            sm_b = tf.Variable(tf.zeros([vocab_length]), name="sm_b")

            # Global step: scalar, i.e., shape [].
            self.global_step = tf.Variable(0, name="global_step")

            # Nodes to compute the nce loss w/ candidate sampling.
            contexts_matrix = tf.reshape(
                tf.cast(contexts_ph,
                        dtype=tf.int64),
                [self.batch_size, 1])

            # Negative sampling.
            sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=contexts_matrix,
                num_true=1,
                num_sampled=self.negative_samples,
                unique=True,
                range_max=vocab_length,
                distortion=0.75,
                unigrams= [max(context_frequencies)] + context_frequencies))

            # Embeddings for examples: [batch_size, emb_dim]
            example_emb = tf.nn.embedding_lookup(emb, words_ph)

            # Weights for labels: [batch_size, emb_dim]
            true_w = tf.nn.embedding_lookup(sm_w_t, contexts_ph)
            # Biases for labels: [batch_size, 1]
            true_b = tf.nn.embedding_lookup(sm_b, contexts_ph)

            # Weights for sampled ids: [num_sampled, emb_dim]
            sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
            # Biases for sampled ids: [num_sampled, 1]
            sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

            # True logits: [batch_size, 1]
            true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b

            # Sampled logits: [batch_size, num_sampled]
            # We replicate sampled noise labels for all examples in the batch
            # using the matmul.
            sampled_b_vec = tf.reshape(sampled_b, [self.negative_samples])
            sampled_logits = tf.matmul(example_emb,
                                       sampled_w,
                                       transpose_b=True) + sampled_b_vec

            true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(true_logits), logits=true_logits)
            sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

            # NCE-loss is the sum of the true and noise (sampled words)
            # contributions, averaged over the batch.
            loss = (tf.reduce_sum(true_xent) +
                               tf.reduce_sum(sampled_xent)) / self.batch_size
            tf.summary.scalar("NCE_loss", loss)
            self._loss = loss

            decay_steps = sum(numpy.array(context_frequencies) * sampling_table[1:]) * \
                        self.epochs / self.batch_size
            lr = tf.train.polynomial_decay(self.learning_rate, self.global_step,
                                           int(decay_steps), end_learning_rate=0.00002, power=1.0)
            self._lr = lr
            optimizer = tf.train.GradientDescentOptimizer(lr)
            train = optimizer.minimize(loss,
                                       global_step=self.global_step,
                                       gate_gradients=optimizer.GATE_NONE)
            self._optimizer = train
        return graph

    def train(self):

        with tf.Session(graph=self.prepare_model()) as session:
          # We must initialize all variables before we use them.
          tf.global_variables_initializer().run()
          # start the tensorflow QueueRunner's
          coord = tf.train.Coordinator()
          threads = tf.train.start_queue_runners(coord=coord, sess=session)
          # start our custom queue runner's threads
          self.custom_runner.start_threads(session)
          print('Variables initialized and queues running')

          average_loss = 0
          current_epoch = self.corpus._epoch

          while True:
              if self.corpus._epoch > current_epoch:
                  self.save_embeddings("embeddings_{}.txt".format(current_epoch))
                  current_epoch = self.corpus._epoch
              if self.corpus._epoch > self.epochs:
                  break

              # We perform one update step by evaluating the optimizer op (including it
              # in the list of returned values for session.run()
              _, loss_val, lr, step = session.run(
                      [self._optimizer, self._loss, self._lr, self.global_step])
              average_loss += loss_val

              if step % 10000 == 0 and step > 0:
                average_loss /= 10000
                # The average loss is an estimate of the loss over the last 2000 batches.
                logging.info('Epoch {} step {}: lr {} average loss {}'.format(
                        self.corpus._epoch, step, lr, average_loss))
                average_loss = 0
          session.run(self.custom_runner.queue.close(cancel_pending_enqueues=True))
          coord.request_stop()
          coord.join(threads)

    def save_embeddings(self, path):
        logging.info("Saving the embeddings")
        if not os.path.exists("embeddings"):
            os.makedirs("embeddings")
        with open("embeddings/" + path, 'w') as fout:
            embs = self._emb.eval()
            for k in self.tokenizer.word_index.keys():
                i = self.tokenizer.word_index[k]
                vector = embs[i]
                l = " ".join([k] + [str(d) for d in vector])
                fout.write(l + "\n")

if __name__ == "__main__":
    SG = SkipGram(corpus_path=FLAGS.corpus_path, embedding_size=FLAGS.embedding_size,
        min_freq=FLAGS.min_freq, negative_samples=FLAGS.negative_samples, epochs=FLAGS.epochs,
        learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size, labeled=FLAGS.labeled)
    SG.tokenize_corpus()
    SG.train()
