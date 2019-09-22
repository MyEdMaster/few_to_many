import pandas as pd
import numpy as np
import tensorflow as tf
import ssl
import nltk
import requests
import math
import os

from gensim import models
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


nltk.download('punkt')
nltk.download('stopwords')
STOP = set(nltk.corpus.stopwords.words("english"))

print("sdasdasd")
class Sentence:
    def __init__(self, sentence):
        self.raw = sentence
        normalized_sentence = sentence.replace("‘", "'").replace("’", "'")
        self.tokens = [t.lower() for t in nltk.word_tokenize(normalized_sentence)]
        self.tokens_without_stop = [t for t in self.tokens if t not in STOP]

    def __str__(self):
        return self.raw


def load_sts_dataset(filename):
    # Loads a subset of the STS dataset into a DataFrame. In particular both
    # sentences and their human rated similarity score.
    sent_pairs = []
    with tf.gfile.GFile(filename, "r") as f:
        for line in f:
            ts = line.strip().split("\t")
            sent_pairs.append((ts[5], ts[6], float(ts[4])))
    return pd.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])


def download_and_load_sts_data():
    sts_dataset = tf.keras.utils.get_file(
        fname="Stsbenchmark.tar.gz",
        origin="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz",
        extract=True)

    sts_dev = load_sts_dataset(os.path.join(os.path.dirname(sts_dataset), "stsbenchmark", "sts-dev.csv"))
    sts_test = load_sts_dataset(os.path.join(os.path.dirname(sts_dataset), "stsbenchmark", "sts-test.csv"))

    return sts_dev, sts_test


sts_dev, sts_test = download_and_load_sts_data()


def download_sick(f):
    response = requests.get(f).text

    lines = response.split("\n")[1:]
    lines = [l.split("\t") for l in lines if len(l) > 0]
    lines = [l for l in lines if len(l) == 5]

    df = pd.DataFrame(lines, columns=["idx", "sent_1", "sent_2", "sim", "label"])
    df['sim'] = pd.to_numeric(df['sim'])
    return df


sick_train = download_sick("https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_train.txt")
sick_dev = download_sick("https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_trial.txt")
sick_test = download_sick("https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_test_annotated.txt")
sick_all = sick_train.append(sick_test).append(sick_dev)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

PATH_TO_WORD2VEC = os.path.expanduser(
    "./GoogleNews-vectors-negative300.bin")
word2vec = models.KeyedVectors.load_word2vec_format(PATH_TO_WORD2VEC, binary=True)

print("sdasdasd")


def run_avg_benchmark(sentences1, sentences2, model=None, use_stoplist=False, doc_freqs=None):
    if doc_freqs is not None:
        N = doc_freqs["NUM_DOCS"]

    sims = []
    sent1 = sentences1
    sent2 = sentences2

    tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
    tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

    tokens1 = [token for token in tokens1 if token in model]
    tokens2 = [token for token in tokens2 if token in model]

    tokfreqs1 = Counter(tokens1)
    tokfreqs2 = Counter(tokens2)

    weights1 = [tokfreqs1[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
                for token in tokfreqs1] if doc_freqs else None
    weights2 = [tokfreqs2[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
                for token in tokfreqs2] if doc_freqs else None

    embedding1 = np.average([model[token] for token in tokfreqs1], axis=0, weights=weights1).reshape(1, -1)
    embedding2 = np.average([model[token] for token in tokfreqs2], axis=0, weights=weights2).reshape(1, -1)

    sim = cosine_similarity(embedding1, embedding2)[0][0]
    sims.append(sim)

    return sims


#question_list = []
#question_list = part_database.get_question()
#question_list=["How are u.","Today is sunny!"]


def find_most_similar(sentences="", question_list=[]):
    similar_score = []
    for s in range(len(question_list)):
        ql=Sentence(question_list[s])
        similar_score.append([run_avg_benchmark(Sentence(sentences), ql, word2vec), ql])
    similar_score.sort(reverse=True)
    # for i in similar_score:
    #     print(i[0])
    #     print(i[1])

    if similar_score[0][0][0] > 0.6:
        return similar_score[0]
    else:
        return "no match"


def classify(inputs,question_list):
    tmp = find_most_similar(inputs, question_list)
    question = tmp[1].raw
    return question


#print(classify('how are you.',question_list))
