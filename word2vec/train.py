# LEARN word-vectors
import gensim
from gensim.models.word2vec import Word2Vec
from nltk.corpus import brown
from pandas import DataFrame
import re
import logging
import json

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


MIN_COUNT = 100 # ignore words below this count
# Skip-gram hyperparameters to be tuned
DIMENSION = 100
CONTEXT = 5

# get some dummy data: http://jmcauley.ucsd.edu/data/amazon/
def getSampleData():
    sentences = []
    for line in open("data/reviews_Digital_Music_5.json", "r"):
        review = json.loads(line)
        sentences.append(review['reviewText'].split())
    return sentences

# get Quora data : https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs (NLI problem)
def getQuoraData():
    df = DataFrame.from_csv("data/quora_duplicate_questions.tsv", sep="\t")
    sentences = []
    for i in range(0,df.shape[0]):
        try:
            sentences.append(df.loc[i,'question1'].split())
            sentences.append(df.loc[i, 'question2'].split())
        except AttributeError:
            pass
    return sentences


# Take a list of words as input and clean them
# (currently bit slow, parallelize : http://chriskiehl.com/article/parallelism-in-one-line/)
def cleanSentence(sentence):
    return (" ".join([re.sub('\W+', " ", word).lower() for word in sentence])).split()


if __name__== '__main__':
    # get data
    sentences = getQuoraData();
    print(sentences[0]); print(cleanSentence(sentences[0]))
    sentences = [cleanSentence(s) for s in sentences]
    print("Done cleaning the tokens");
    # train model
    model = Word2Vec(sentences, size=DIMENSION, window = CONTEXT, min_count=MIN_COUNT)
    model.save("model/word2vec_model");
    print(len(model.vocab))
    # train vecs for bi-grams
    bigram_transformer = gensim.models.Phrases(sentences); print("Training bigrams ...")
    bigram_model = Word2Vec(bigram_transformer[sentences], size=DIMENSION, window = CONTEXT, min_count=MIN_COUNT)
    bigram_model.save("model/word2vec_model_bigrams");
    print(len(bigram_model.vocab));


