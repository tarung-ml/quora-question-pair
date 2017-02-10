import pandas as pd, numpy as np
from word2vec.load_pretrained import static_w2v
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
from util.helper import dropPickle
import re

def cleanWord(word):
    return re.sub("\W+", " ", word)


def bowAverage(sentence, w2v):
    bowVec = np.zeros((300,))
    count = 0
    try:
        for word in sentence.split():
            word = cleanWord(word)
            if word in w2v.vocab and (word not in STOPWORDS or word == 'not'):
                bowVec += w2v[(word)]
            count += 1
    except AttributeError: # some bad lines containing floats
        pass
    return bowVec

def prepareMatrix(dataset, w2v):
    X = []
    y = []
    sentences = []
    for i in range(0, dataset.shape[0]):
        try:
            y.append(1 if dataset.loc[i, 'gold_label'][0] == 'e' else 0) # 'entailment' is positive label
            X.append((list(bowAverage(dataset.loc[i, 'sentence1_parse'], w2v)) + list(bowAverage(dataset.loc[i, 'sentence2_parse'], w2v))))
            sentences.append(str(dataset.loc[i, 'sentence1_parse']) + ";" + str(dataset.loc[i, 'sentence2_parse']))
            if i % 100 == 0:
                print("processed: " + str(i))
        except TypeError:
            pass
    X = (pd.DataFrame(data=np.array(X))) #make dataframe
    return (X, y, sentences)



STOPWORDS = set(stopwords.words('english'))

# MAIN

# load w2v
w2v = static_w2v(
    '/Users/tarun/Downloads/GoogleNews-vectors-negative300.bin')  # Word2Vec.load("word2vec/model/word2vec_model") #

# read training data
train = pd.read_csv("data_split/quora_duplicate_questions_train.txt", sep="\t", error_bad_lines=False)
print("Read " + str(train.shape[0]) + " lines of training examples..."); print("Columns : " + str(train.columns))

# read dev data
dev = pd.read_csv("data_split/quora_duplicate_questions_dev.txt", sep="\t", error_bad_lines=False)
print("Read " + str(dev.shape[0]) + " lines of dev examples...")

(X_train, y_train, sentences_train) = (prepareMatrix(train, w2v))
(X_dev, y_dev, sentences_dev) = (prepareMatrix(dev, w2v))

print(len(X_train)); print(len(y_train))

# save pickle files to save time for model building (large files excluded from Git)
dropPickle(X_train, "/Users/tarun/X_train.pkl"); dropPickle(X_dev, "/Users/tarun/X_dev.pkl")
dropPickle(y_train, "/Users/tarun/y_train.pkl"); dropPickle(y_dev, "/Users/tarun/y_dev.pkl")
# save pickle
dropPickle(sentences_train, "/Users/tarun/sentences_train.pkl")
dropPickle(sentences_dev, "/Users/tarun/sentences_dev.pkl")





