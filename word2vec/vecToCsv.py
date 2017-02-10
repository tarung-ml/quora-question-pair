# WRITE word-vectors to csv file
from gensim.models.word2vec import Word2Vec
import pandas as pd

if __name__== '__main__':
    model = Word2Vec.load("model/word2vec_model")
    print(len(model.index2word));print(model.syn0.shape)
    df = pd.DataFrame(data = model.syn0, index = model.index2word)
    df.to_csv("model/word2vec_model.csv")

    bigram_model = Word2Vec.load("model/word2vec_model_bigrams")
    print(len(bigram_model.vocab))
    df2 = pd.DataFrame(data = bigram_model.syn0, index = bigram_model.index2word)
    df2.to_csv("model/word2vec_model_bigrams.csv")

