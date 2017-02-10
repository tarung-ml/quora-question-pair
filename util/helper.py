from keras.models import model_from_json
import numpy as np

import pickle
def dropPickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def loadPickle(filename):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_model(model):
    json_string = model.to_json()
    open('nn/model.json', 'w').write(json_string)
    model.save_weights('nn/model.h5',overwrite=True)

def load_model():
    model = model_from_json(open('nn/model.json').read())
    model.load_weights('nn/model.h5')
    return model


# Feature engineering
epsilon = 1e-5

def cosine(arr1, arr2):
    return arr1.dot(arr2) / (epsilon + np.sqrt(sum(arr1 ** 2))) / (epsilon + np.sqrt(sum(arr2 ** 2)))

def addInteractions(X):
    X['x1_cosine_x2'] = X.apply(lambda row: cosine(np.array(row[0:299]), np.array(row[300:599])), axis = 1)
    X['eucledian'] = 0 # sum_i[(xi-yi)^2] called eucledian
    X['abs'] = 0 # sum_i[abs(xi-yi)] # called manhattan
    X['minmax_num'] = 0 # sum_i[min(xi-yi)] / sum_i[max(xi-yi)] called minMax distance
    X['minmax_den'] = 0 #
    for i in range(0, 300):
        col_plus = 'plus'+ str(i) # measures sum at index i (a new feature)
        X[col_plus] =  X[i] + X[300 + i]
        col_minus = 'minus' + str(i) # measures diff at index i (a new feature)
        X[col_minus] =  X[i] - X[300 + i]
        # update eucledian
        X['eucledian'] = X['eucledian'] + X[col_minus]**2 #.apply(lambda x: x**2)
        # update abs distance
        absdiff = X[col_minus].abs(); X['abs'] = X['abs'] + absdiff #apply(lambda x: abs(x))
        # update min max distance (to make fast do it without IF) max(a,b) = 1/2*(a + b + |a-b|)
        max_col = 0.5*(X[col_plus] + absdiff)
        X['minmax_num'] = X['minmax_num']  +  max_col
        X['minmax_den'] = X['minmax_den']  +  X[col_plus] - max_col
        if i % 10 == 0:
            print(i)
    # do the final op for min-max distance and cleanup
    X['minmax'] = X.apply(lambda row: row['minmax_num']/(row['minmax_den'] + epsilon), axis =1)
    del X['minmax_num']; del X['minmax_den']
    return X