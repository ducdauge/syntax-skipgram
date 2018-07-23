"""
Helper abstract classes.
"""

import logging, codecs
from collections import OrderedDict, defaultdict
# CoNLL-U column names
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

class UDWord(object):
        def __init__(self, columns):
            self.parent = None
            self.children = []
            _, self.form, _, _, _, _, self.head, self.deprel, _, _ = columns

        @property
        def _pf(self):
            if self.parent:
                return [(self.parent.form, self.parent.deprel + "|p")]
            else:
                return [], []
        @property
        def _cf(self):
            if self.children:
                return [(c.form, c.deprel + "|c") for c in self.children]
            else:
                return [], []


class TreebankReader(object):
    """ TreebankReader object. Works as a iterator of dependency trees
    Parameters:
        path (str): The path to a text file
    """

    def __init__(self, path, labeled=False):
        self._path = path
        self._labeled = labeled
        self._epoch = 0
        self._corpus_size = 0
        self._wordcounts = defaultdict(int)

    # UD Error is used when raising exceptions in this module
    class UDError(Exception):
        pass

    def generate(self, single=False):
        while True:
            sentence = []
            word_context = []
            for i, line in enumerate(codecs.open(self._path)):
                line = line.rstrip("\r\n")
                if line.startswith("#"):
                    continue
                if not line:
                    # Add parent and children UDWord links and check there are no cycles
                    def process_word(word):
                        if word[1] == "remapping":
                            raise self.UDError("There is a cycle in a sentence")
                        if word[1] is None:
                            head = int(word[0][HEAD])
                            if head:
                                parent = sentence[head - 1]
                                if single:
                                    self._wordcounts[word[0][FORM]] += 1
                                if self._labeled == True:
                                    word_context.append((word[0][FORM], "|".join([parent[0][FORM], word[0][DEPREL], "p"])))
                                    word_context.append((parent[0][FORM], "|".join([word[0][FORM], word[0][DEPREL], "c"])))
                                else:
                                    word_context.append((word[0][FORM], parent[0][FORM]))
                                    word_context.append((parent[0][FORM], word[0][FORM]))

                                word[1] = "remapping"
                                process_word(parent)
                                word[1] = "done"

                    for word in sentence:
                        process_word(word)

                    # End the sentence
                    self._corpus_size += 1
                    if not self._corpus_size % 100000:
                        logging.info('Done %d sentences', self._corpus_size)
                    yield word_context
                    sentence = []
                    word_context = []
                    continue

                # Read next token/word
                columns = line.split("\t")
                if len(columns) != 10:
                    raise self.UDError("The CoNLL-U line does not contain 10 tab-separated columns: '{}'".format(line))

                # Skip empty nodes
                if "." in columns[ID] or "-" in columns[ID]:
                    continue

                # Basic tokens/words
                else:
                    sentence.append([columns, None])
            self._epoch += 1
            if single:
                self._corpus_size = 0
                break

    def generate_CBOW(self, single=False):
        self.deprel_labels = set()
        while True:
            sentence = []
            for i, line in enumerate(codecs.open(self._path)):
                line = line.rstrip("\r\n")
                if line.startswith("#"):
                    continue
                if not line:
                    # Add parent and children UDWord links and check there are no cycles
                    def process_word(word):
                        if word.parent == "remapping":
                            raise self.UDError("There is a cycle in a sentence")
                        if word.parent is None:
                            head = int(word.head)
                            if head:
                                parent = sentence[head - 1]
                                if single:
                                    self._wordcounts[word.form] += 1

                                word.parent = parent
                                parent.children.append(word)

                                word.head = "remapping"
                                process_word(parent)

                    for word in sentence:
                        process_word(word)

                    # End the sentence
                    self._corpus_size += 1
                    if not self._corpus_size % 100000:
                        logging.info('Done %d sentences', self._corpus_size)
                    for word in sentence:
                        context, deprels = zip(*[word._pf + word._cf])
                        if len(context) > self.max_context:
                            self.max_context = len(context)
                        self.deprel_labels.update(context)
                        yield word.form, context, deprels
                    sentence = []
                    continue

                # Read next token/word
                columns = line.split("\t")
                if len(columns) != 10:
                    raise self.UDError("The CoNLL-U line does not contain 10 tab-separated columns: '{}'".format(line))

                # Skip empty nodes
                if "." in columns[ID] or "-" in columns[ID]:
                    continue

                # Basic tokens/words
                else:
                    sentence.append(UDWord(columns))
            self._epoch += 1
            if single:
                self._corpus_size = 0
                break
    @property
    def corpus_size(self):
        """ Return the corpus size """
        return self._corpus_size

    @property
    def path(self):
        """ Return the corpus path """
        return self._path

class cw_Tokenizer(object):
    """Text tokenization utility class.
    This class allows to vectorize a text corpus
    # Arguments
        num_words: the maximum number of words to keep, based on frequency
    `0` is a reserved index that won't be assigned to any word.
    """

    def __init__(self, min_freq=100,
                 **kwargs):

        self.context_counts = OrderedDict()
        self.min_freq = min_freq
        self.word_index = None
        self.context_index = None

    def fit_on_texts(self, words):
        """Updates internal vocabulary based on a list of texts.
        Required before using `texts_to_sequences` or `texts_to_matrix`.
        # Arguments
            texts: can be a list of strings,
                or a generator of strings (for memory-efficiency)
        """
        for w, c in words:
            self.context_counts[c] = self.context_counts.get(c, 0) + 1

    def finalize_dict(self, wordcounts):
        wcounts = list(wordcounts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc, self.word_frequencies = zip(*[wc for wc in wcounts if wc[1] >= self.min_freq])
        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

        ccounts = list(self.context_counts.items())
        ccounts.sort(key=lambda x: x[1], reverse=True)
        sorted_ctx, self.context_frequencies = zip(*ccounts[: len(sorted_voc) ])
        # note that index 0 is reserved, never assigned to an existing word
        self.context_index = dict(list(zip(sorted_ctx, list(range(1, len(sorted_ctx) + 1)))))

        logging.info("Vocabulary has {} words and contexts".format(len(sorted_voc)))

class cb_Tokenizer(object):
    """Text tokenization utility class.
    This class allows to vectorize a text corpus
    # Arguments
        num_words: the maximum number of words to keep, based on frequency
    `0` is a reserved index that won't be assigned to any word.
    """

    def __init__(self, min_freq=100,
                 **kwargs):

        self.min_freq = min_freq
        self.word_index = None

    def finalize_dict(self, wordcounts):
        wcounts = list(wordcounts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc, self.word_frequencies = zip(*[wc for wc in wcounts if wc[1] >= self.min_freq])
        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))
        len(sorted_voc)
        logging.info("Vocabulary has {} words and contexts".format(len(sorted_voc)))
