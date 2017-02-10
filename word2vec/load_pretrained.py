from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords

# Load Google's pre-trained Word2Vec model from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
# large file: so store local (exclude from git)
def static_w2v(fileloc):
    model = Word2Vec.load_word2vec_format(fileloc, binary=True)
    print(len(model.vocab)) # 3 Million
    #print(model.vocab) # a lot of weird words but ok.
    #print(model['What']); print(model['what'])
    return model
"""
w2v = static_w2v(
    '/Users/tarun/Downloads/GoogleNews-vectors-negative300.bin')  # Word2Vec.load("word2vec/model/word2vec_model") #
from util.helper import cosine

#print(w2v['dynamo'])
#print(w2v['Dynamo'])
print(cosine(w2v['dynamo'], w2v['Dynamo']))
print(cosine(w2v['invention'], w2v['inventing']))
print(cosine(w2v['discovery'], w2v['discovering']))
print(cosine(w2v['mind'], w2v['mindfulness']))
print(cosine(w2v['Father'], w2v['father']))
print(cosine(w2v['Economics'], w2v['economic']))
print(cosine(w2v['rom'], w2v['ROM']))
print(cosine(w2v['Text'], w2v['text']))
print(cosine(w2v['masturbations'], w2v['masturbating']))
print(cosine(w2v['mean'], w2v['means']))
print(cosine(w2v['disrupt'], w2v['disrupting']))
print('.' in w2v.vocab)
print('?' in w2v.vocab)
print('15k' in w2v.vocab)


print('Text.' in w2v.vocab)


#print(cosine(w2v['heart'], w2v['heartfulness']))

STOPWORDS = set(stopwords.words('english'))
print('is' in STOPWORDS)
print('does' in STOPWORDS)
print('who' in STOPWORDS)
print('not' in STOPWORDS)
print(STOPWORDS)
"""

