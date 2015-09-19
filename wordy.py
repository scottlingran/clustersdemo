import numpy
import math

from sklearn import decomposition, cluster, manifold
from gensim import models

import time
import os

import logging
logging.basicConfig(level=logging.INFO)

CORPUS = os.environ.get("CORPUS") or "corpus/glove.6B.50d.txt"
VEC_SIZE = int(os.environ.get("VEC_SIZE") or 300)
BINARY = bool(os.environ.get("BINARY")) or False

start = time.time()
print(CORPUS + ": loading ...")
v_model = models.Word2Vec.load_word2vec_format(CORPUS, binary=BINARY)
end = time.time()
print(CORPUS + ": done! (" + str(end-start) + "s)")


import overrides
models.Word2Vec.most_similar = overrides.most_similar

def get_similar(args):
    opts = {}
    opts["positive"] = args.getlist("positive") or ["happy"]
    opts["negative"] = args.getlist("negative") or []
    opts["topn"] = args.get("topn") or 25

    check_alt(opts["positive"] + opts["negative"])

    response = []

    v_results = v_model.most_similar(
        positive=opts["positive"],
        negative=opts["negative"],
        topn=int(opts["topn"])
    )

    v_results_first = [el[0] for el in v_results]
    similarity_scores = [el[1] for el in v_results]

    v_processed = process(v_results_first, args)

    for index, p in enumerate(v_processed):
        obj = {}

        obj["word"] = " ".join(v_results_first[index].split("_"))

        obj["x"] = p["low_d_array"][0]
        obj["y"] = p["low_d_array"][1]
        obj["dynamic"] = p["dynamic"]
        obj["fixed"] = p["fixed"]
        word_first = obj["word"].split(" ")[0]
        obj["vocab_index"] = v_model.vocab[v_results_first[index]].index

        obj["similarity"] = similarity_scores[index]

        response.append(obj)

    return response


def get_list(args):
    response = []

    opts = {}
    opts["list"] = args.getlist("list") or ["happy", "sad", "funny", "boring"]
    topic = args.get("topic") or opts["list"][0].split(" ")[0]

    if len(opts["list"]) < 4:
        raise ValueError("need >= 4 list items")

    check(opts["list"] + [topic])

    v_processed = process(opts["list"], args)

    for index, p in enumerate(v_processed):
        obj = {}
        obj["word"] = opts["list"][index]

        obj["x"] = p["low_d_array"][0]
        obj["y"] = p["low_d_array"][1]
        obj["dynamic"] = p["dynamic"]
        obj["fixed"] = p["fixed"]

        word_first = obj["word"].split(" ")[0]
        obj["vocab_index"] = v_model.vocab[word_first].index

        sim = numpy.dot(p["high_d_array"], v_model[topic])
        obj["similarity"] = numpy.float32(sim).item()

        response.append(obj)

    return response


def process(v_results, args):
    opts = {}
    opts["split_type"] = args.get("split_type") or "sum"
    opts["n_clusters"] = args.get("n_clusters") or 3
    opts["n_components"] = args.get("n_components") or 2

    res = [{} for word in v_results]

    high_d_array = numpy.array([numpy.arange(VEC_SIZE)])

    for element in v_results:
        words = element.split(" ")

        if len(words) > 1:
            vec_list = [v_model[word] for word in words]

            if opts["split_type"] == "mean":
                key_vector = numpy.mean(vec_list, axis=0)
            else:
                key_vector = sum(vec_list)
        else:
            key_vector = v_model[words[0]]

        cur_d_array = numpy.array([key_vector])
        high_d_array = numpy.concatenate((high_d_array, cur_d_array))

    high_d_array = numpy.delete(high_d_array, 0, 0)



    # pca = decomposition.PCA(n_components=int(opts["n_components"]))
    # low_d_array = pca.fit_transform(high_d_array)

    # NOTE: All the cool kids are using PCA
    # NOTE: But, it generates the same thing if init=default, random_state=default
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    low_d_array = tsne.fit_transform(high_d_array)

    spec = cluster.SpectralClustering(n_clusters=int(opts["n_clusters"]))
    klasses_spec = spec.fit_predict(high_d_array)

    # NOTE: SLOW! 250ms for 4 items ???
    meanshift = cluster.MeanShift()
    klasses_meanshift = meanshift.fit_predict(low_d_array)


    for i, k in enumerate(klasses_spec):
        res[i]["fixed"] = numpy.int16(k).item()
    for i, k in enumerate(klasses_meanshift):
        res[i]["dynamic"] = numpy.int16(k).item()
    for i, k in enumerate(low_d_array):
        res[i]["low_d_array"] = k
    for i, k in enumerate(high_d_array):
        res[i]["high_d_array"] = k

    return res

def check(word_list):
    missing = []
    for word in word_list:
        try:
            [v_model[s] for s in word.split(" ")]
        except:
            missing.append(word)

    if len(missing) > 0:
        raise ValueError("Words not found: " + ", ".join(missing))

    return True

def check_alt(word_list):
    missing = []
    for word in word_list:
        try:
            v_model[word]
        except:
            missing.append(word)

    if len(missing) > 0:
        raise ValueError("Words not found: " + ", ".join(missing))

    return True
