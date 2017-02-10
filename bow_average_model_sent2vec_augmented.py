import pandas as pd, numpy as np
from word2vec.load_pretrained import static_w2v
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
from util.helper import dropPickle, loadPickle
import re
import random

AUGMENT_PROB = 0.1
WORD_PER_SEN = 1/3

def cleanWord(word):
    return re.sub("\W+", " ", word)


def bowAverageRand(sentence, w2v):
    bowVec = np.zeros((300,))
    count = 0
    try:
        for word in sentence.split():
            word = cleanWord(word)
            if word in w2v.vocab and (word not in STOPWORDS or word == 'not'):
                if random.random() < WORD_PER_SEN:
                    word = w2v.most_similar(positive=word, topn=1)[0][0]
                bowVec += w2v[(word)]
            count += 1
    except AttributeError: # some bad lines containing floats
        pass
    return bowVec

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
            if i % 20 == 0: # add random examples at 10% (randomize only 1 sentence out of 2)
                y.append(1 if dataset.loc[i, 'gold_label'][0] == 'e' else 0)  # 'entailment' is positive label
                X.append((list(bowAverageRand(dataset.loc[i, 'sentence1_parse'], w2v)) + list(bowAverage(dataset.loc[i, 'sentence2_parse'], w2v))))
                sentences.append(str(dataset.loc[i, 'sentence1_parse']) + ";" + str(dataset.loc[i, 'sentence2_parse']))
            elif i % 10 == 0:
                y.append(1 if dataset.loc[i, 'gold_label'][0] == 'e' else 0)  # 'entailment' is positive label
                X.append((list(bowAverage(dataset.loc[i, 'sentence1_parse'], w2v)) + list(bowAverageRand(dataset.loc[i, 'sentence2_parse'], w2v))))
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

(X_aug, y_aug, sentences_train) = (prepareMatrix(train, w2v))

X_train = loadPickle("/Users/tarun/X_train.pkl");
y_train = loadPickle("/Users/tarun/y_train.pkl");

print(len(X_train)); print(len(y_train))

# append the random augmented examples
X_train_aug = pd.concat([X_train, X_aug])
y_train_aug = y_train + y_aug

# save pickle files to save time for model building (large files excluded from Git)
dropPickle(X_train_aug, "/Users/tarun/X_train_aug.pkl");
dropPickle(y_train_aug, "/Users/tarun/y_train_aug.pkl");
