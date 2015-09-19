from __future__ import division  # py3 "true division"

import logging
import numpy
import time

# utility fnc for pickling, common scipy operations etc
from gensim import matutils
import six

logger = logging.getLogger("custom")

def argsort(x, topn=None, reverse=False):
    logger.info("numpy.argpartition? " + str(hasattr(numpy, 'argpartition')))
    x = numpy.asarray(x)  # unify code path for when `x` is not a numpy array (list, tuple...)
    if topn is None:
        topn = x.size
    if topn <= 0:
        return []
    if reverse:
        x = -x
    if topn >= x.size or not hasattr(numpy, 'argpartition'):
        return numpy.argsort(x)[:topn]

    # numpy >= 1.8 has a fast partial argsort, use that!
    a = time.time()
    most_extreme = numpy.argpartition(x, topn)[:topn]
    b = time.time()
    logger.info("numpy.argpartition(): " + str((b - a)*1000) + "ms")

    # resort topn into order
    a = time.time()
    x_extreme = x.take(most_extreme)
    b = time.time()
    logger.info("x.take(): " + str((b - a)*1000) + "ms")

    a = time.time()
    x_extreme_sorted = numpy.argsort(x_extreme)
    b = time.time()
    logger.info("numpy.argsort(): " + str((b - a)*1000) + "ms")

    a = time.time()
    x_extreme_taken = most_extreme.take(x_extreme_sorted)
    b = time.time()
    logger.info("most_extreme.take(): " + str((b - a)*1000) + "ms")

    return x_extreme_taken

matutils.argsort = argsort

def most_similar(self, positive=[], negative=[], topn=10):
    self.init_sims()

    if isinstance(positive, six.string_types) and not negative:
        positive = [positive]


    positive = [(word, 1.0) if isinstance(word, six.string_types + (numpy.ndarray,))
                            else word for word in positive]

    negative = [(word, -1.0) if isinstance(word, six.string_types + (numpy.ndarray,))
                             else word for word in negative]

    all_words, mean = set(), []
    for word, weight in positive + negative:
        if isinstance(word, numpy.ndarray):
            mean.append(weight * word)
        elif word in self.vocab:
            mean.append(weight * self.syn0norm[self.vocab[word].index])
            all_words.add(self.vocab[word].index)
        else:
            raise KeyError("word '%s' not in vocabulary" % word)
    if not mean:
        raise ValueError("cannot compute similarity with no input")
    mean = matutils.unitvec(numpy.array(mean).mean(axis=0)).astype(numpy.float32)

    # SLOW
    a = time.time()
    dists = numpy.dot(self.syn0norm, mean)
    b = time.time()
    logger.info("numpy.dot(): " + str((b - a)*1000) + "ms")



    if not topn:
        return dists

    best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)

    # ignore (don't return) words from the input
    result = []

    a = time.time()
    for sim in best:
        if sim not in all_words:
            r = (self.index2word[sim], float(dists[sim]))
            result.append(r)

    result_l = result[:topn]

    b = time.time()

    logger.info("result[] builder: " + str((b - a)*1000) + "ms")
    return result_l
