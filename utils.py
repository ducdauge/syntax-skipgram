"""
Helper abstract classes for SynNegSamp
"""

import logging, codecs
from collections import OrderedDict

# CoNLL-U column names
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

class TreebankReader(object):
    """ TreebankReader object. Works as a iterator of dependency trees
    Parameters:
        path (str): The path to a text file
    """
    
    def __init__(self, path, labeled=True):
        self._path = path
        self._labeled = labeled
        self._epoch = 0
        self._corpus_size = 0
    
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
                            if head > len(sentence):
                                raise self.UDError("HEAD '{}' points outside of the sentence".format(word.columns[HEAD]))
                            if head:
                                parent = sentence[head - 1]
                                if self._labeled:
                                    word_context.append((word[0][FORM], "|".join([parent[0][FORM], word[0][DEPREL], "p"])))
                                    word_context.append((parent[0][FORM], "|".join([word[0][FORM], word[0][DEPREL], "c"])))
                                    parent_children = [w for w in sentence if w[0][HEAD] == str(head)]
                                    for pc in parent_children:
                                        word_context.append((word[0][FORM], "|".join([pc[0][FORM], pc[0][DEPREL], "pc"])))
                                else:
                                    word_context.append((word[0][FORM], parent[0][FORM]))
                                    word_context.append((parent[0][FORM], word[0][FORM]))
                                    parent_children = [w for w in sentence if w[0][HEAD] == str(head)]
                                    for pc in parent_children:
                                        word_context.append((word[0][FORM], pc[0][FORM]))
                                word[1] = "remapping"
                                process_word(parent)
                                word[1] = "done"
        
                    for word in sentence:
                        process_word(word)
        
                    # End the sentence
                    self._corpus_size += 1
                    if not self._corpus_size % 10000:
                        logging.info('Done %d sequences', self._corpus_size)
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
    @property
    def corpus_size(self):
        """ Return the corpus size """
        return self._corpus_size

    @property
    def path(self):
        """ Return the corpus path """
        return self._path
    
    @property
    def label_set(self):
        """ Return the corpus path """
        return self._label_set

class cw_Tokenizer(object):
    """Text tokenization utility class.
    This class allows to vectorize a text corpus
    # Arguments
        num_words: the maximum number of words to keep, based on frequency
    `0` is a reserved index that won't be assigned to any word.
    """

    def __init__(self, num_words=None,
                 **kwargs):
        
        self.word_counts = OrderedDict()
        self.context_counts = OrderedDict()
        self.num_words = num_words
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
            self.word_counts[w] = self.word_counts.get(w, 0) + 1
            self.context_counts[c] = self.context_counts.get(c, 0) + 1
    
    def finalize_dict(self):
        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        wlimit = min(len(wcounts), self.num_words)
        sorted_voc = [wc[0] for wc in wcounts[: wlimit ]]
        words_per_epoch = sum([wc[1] for wc in wcounts[: wlimit ]])
        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))
        self.words_per_epoch = words_per_epoch
        
	ccounts = list(self.context_counts.items())
        ccounts.sort(key=lambda x: x[1], reverse=True)
        sorted_ctx = [cc[0] for cc in ccounts[: wlimit ]]
        # note that index 0 is reserved, never assigned to an existing word
        self.context_index = dict(list(zip(sorted_ctx, list(range(1, len(sorted_ctx) + 1)))))
