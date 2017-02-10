# An important observation running bow_average_logistic_regression: traning on full-sample doesn't really help that much. For model selection,
# we can try a smaller sample of data (once we find the optimal hyperparameters we can use them to train a model on full data-set)

from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import svm
from sklearn import grid_search
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics.scorer import make_scorer

#do bayesian optimization for hyper-parameter search
from sklearn.cross_validation import cross_val_score
# clone https://github.com/fmfn/BayesianOptimization (other : http://hyperopt.github.io/hyperopt/)
from  BayesianOptimization.bayes_opt import BayesianOptimization

#DL
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
from cycler import cycler
from matplotlib.pyplot import imshow
import time

from util.helper import dropPickle, loadPickle
import matplotlib.pyplot as plt
plt.interactive(False) # maybe pycharm specific

# add logging

import sys
f = open("tuning_logs/log" + str(time.strftime("%Y%m%d-%H%M%S")) + ".out", 'w')

# ref: http://stackoverflow.com/questions/11325019/output-on-the-console-and-file-using-python
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()


f = open("tuning_logs/log" + str(time.strftime("%Y%m%d-%H%M%S")) + ".out", 'w')
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)

"""import logging

handlers = [logging.FileHandler("tuning_logs/log" + str(time.strftime("%Y%m%d-%H%M%S")) + ".log"), logging.StreamHandler()]
logging.basicConfig(level = logging.info, handlers = handlers)"""


# fix keras bug (compatbility with scikit)
from keras.wrappers.scikit_learn import BaseWrapper
import copy
def custom_get_params(self, **params):
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res
BaseWrapper.get_params = custom_get_params

# custom scorer for cross-validation: current use aucROC
def precisionAtK(y_true, y_pred, k = 0):
    print('.')# indicates progress
    y_pred = y_pred[:,1] # pos class probability
    return metrics.roc_auc_score(y_true, y_pred)


# helper function to print metrics
def printMetrics(y_test, probs):
    #print(metrics.accuracy_score(y_test, predicted))
    auc = (metrics.roc_auc_score(y_test, probs))
    print('auc:' + str(auc))
    #print('conf_mat:' + str(metrics.confusion_matrix(y_test, predicted)))
    #print('met:' + str(metrics.classification_report(y_test, predicted)))
    return auc


#same as OLD
def logisticRegression(X_train, y_train, X_test, state):
    print("training LR... \n");
    # fit model
    #lr_cv = LogisticRegressionCV(penalty=['l1','l2'], class_weight = "balanced", solver='liblinear', Cs=np.linspace(1e-4, 3, 100), refit=True, random_state = state, scoring = make_scorer(precisionAtK), cv = 5)
    Cs = [0.01, 0.1, 1] #np.linspace(1e-5, 3, 100)#
    lr = LogisticRegression(dual = False, class_weight = "balanced", solver='liblinear', random_state = state)
    lr_cv = grid_search.GridSearchCV(estimator = lr, param_grid = dict(C = Cs, penalty = ['l1', 'l2']), scoring = make_scorer(precisionAtK, needs_proba = True),
                                     cv = StratifiedKFold(y_train, n_folds=3, shuffle=True, random_state=state))
    lr_cv = lr_cv.fit(X_train, y_train)
    # check which value of C was selected
    print(lr_cv.best_params_['C'])
    # select the best estimator
    print(lr_cv.best_estimator_)
    lr_cv = lr_cv.best_estimator_.fit(X_train, y_train)
    # check number of coefficients selected
    print(np.count_nonzero(lr_cv.coef_))  #  non-zero coefficients
    # check features and their coeffs.
    print([ (x,y) for (x,y) in zip(X_train.columns[np.nonzero(lr_cv.coef_)[1]], lr_cv.coef_[0][np.nonzero(lr_cv.coef_[0])[0]])])
    # predict class labels for the test set
    predicted = lr_cv.predict(X_test)
    # generate class probabilities
    probs_lr = lr_cv.predict_proba(X_test)
    return predicted, probs_lr, lr_cv


# too slow (avoid hyperparrametert search) . doesn't scale for > 10K examples (param search : 1, 'rbf'
def suppVector(X_train, y_train, X_test, state):
    print("training SVM... \n");
    """Cs = [0.01, 0.1, 1] #np.linspace(1e-4, 10, 50)#np.logspace(0,1,10) for linear [0,1], for rbf [0,100]
    sv = svm.SVC(class_weight= "balanced",  random_state=state, probability = True)
    kernel_choices = ['linear','poly', 'rbf','sigmoid'] #... tuning always picks 'rbf' as the best kernel (make it default to save time)
    sv_cv = grid_search.GridSearchCV(estimator = sv, param_grid = dict(C = Cs, kernel = kernel_choices), scoring = make_scorer(precisionAtK, needs_proba = True),
                                     cv = StratifiedKFold(y_train, n_folds=3, shuffle=True, random_state=state))
    sv_cv = sv_cv.fit(X_train, y_train)
    print(sv_cv.best_params_)
    C_best = sv_cv.best_params_['C']
    print(sv_cv.best_estimator_)
    sv_cv = sv_cv.best_estimator_.fit(X_train, y_train) # or sv.fit(X_train, y_train)"""
    # to avoid the slow search
    sv_cv = svm.SVC(class_weight= "balanced",  random_state=state, probability = True, kernel = 'rbf', C = 1).fit(X_train, y_train)

    # check which value of C was selected
    #print('C:' + str(C_best))
    # predict class labels for the test set
    predicted = sv_cv.predict(X_test)
    # generate class probabilities
    probs_svm = sv_cv.predict_proba(X_test)
    return predicted, probs_svm, sv_cv



# bayesian optimization for hyperparameter-tuning of random forest
# they suggest 64-128 trees https://www.researchgate.net/publication/230766603_How_Many_Trees_in_a_Random_Forest
def randomForest(X_train, y_train, X_test, state):
    print("training randomForest... \n");
    # set state
    np.random.seed(state)
    # define a search space
    space = {'max_depth': (5,10),
             'max_features': (0.1, 0.999),
             'min_samples_split': (1,20),
             'min_samples_leaf': (1,20),
             'n_estimators': (64, 100)
    }
    # minimize the objective over the space
    best = BayesianOptimization (lambda max_depth, min_samples_split, max_features, min_samples_leaf, n_estimators :
                                 np.nanmean(cross_val_score(
            RandomForestClassifier(n_estimators = int(n_estimators),class_weight = "balanced", random_state = state,
                                   max_depth = int(max_depth),
                                   max_features = min(max_features, 0.999),
                                   min_samples_split = int(min_samples_split),
                                   min_samples_leaf = int(min_samples_leaf),
                                   criterion = "gini"),
                                 X_train, y_train, scoring=make_scorer(precisionAtK, needs_proba = True),
                cv=StratifiedKFold(y_train, n_folds=3, shuffle=True, random_state=state))), space) # X_, y_, scoring=make_scorer(precisionAtK), cv=5).mean(), space) #
    best.maximize()
    print(best.res['max']['max_params'])
    # obtain best fit
    best_fit = RandomForestClassifier(n_estimators = int(best.res['max']['max_params']['n_estimators']),class_weight = "balanced", random_state = state,
                                   max_depth = int(best.res['max']['max_params']['max_depth']),
                                   max_features = best.res['max']['max_params']['max_features'],
                                   min_samples_split = int(best.res['max']['max_params']['min_samples_split']),
                                   min_samples_leaf = int(best.res['max']['max_params']['min_samples_leaf']),
                                   criterion = "gini")
    rf_cv  = best_fit.fit(X_train, y_train)
    # predict class labels for the test set
    predicted = rf_cv.predict(X_test)
    # generate class probabilities
    probs_rf = rf_cv.predict_proba(X_test)
    return predicted, probs_rf, rf_cv


# bayesian optimization for hyperparameter-tuning of "extremely randomized trees" P. Geurts, D. Ernst., and L. Wehenkel, “Extremely randomized trees”, Machine Learning, 63(1), 3-42, 2006.
# key = splits are at ranodm (compare to information gain optimization in random-forest) = > less over-fitting
def extremeTrees(X_train, y_train, X_test, state):
    print("training ExtraTrees... \n");
    # set state
    np.random.seed(state)
    # define a search space
    space = {'max_depth': (5, 20),
             'max_features': (0.1, 0.999),
             'min_samples_split': (2,20),
             'min_samples_leaf': (1,20),
             'n_estimators': (64, 128)
    }
    # minimize the objective over the space
    best = BayesianOptimization (lambda max_depth, min_samples_split, max_features, min_samples_leaf, n_estimators :
                                 np.nanmean(cross_val_score(
            ExtraTreesClassifier (n_estimators = int(n_estimators),class_weight = "balanced", random_state = state,
                                   max_depth = int(max_depth),
                                   max_features = min(max_features, 0.999),
                                   min_samples_split = int(min_samples_split),
                                   min_samples_leaf = int(min_samples_leaf),
                                   criterion = "gini"),
                                 X_train, y_train, scoring=make_scorer(precisionAtK, needs_proba = True),
                cv=StratifiedKFold(y_train, n_folds=3, shuffle=True, random_state=state))), space) # X_, y_, scoring=make_scorer(precisionAtK), cv=5).mean(), space) #
    best.maximize()
    print(best.res['max']['max_params'])
    # obtain best fit
    best_fit = ExtraTreesClassifier (n_estimators = int(best.res['max']['max_params']['n_estimators']),class_weight = "balanced", random_state = state,
                                   max_depth = int(best.res['max']['max_params']['max_depth']),
                                   max_features = best.res['max']['max_params']['max_features'],
                                   min_samples_split = int(best.res['max']['max_params']['min_samples_split']),
                                   min_samples_leaf = int(best.res['max']['max_params']['min_samples_leaf']),
                                   criterion = "gini")
    et_cv  = best_fit.fit(X_train, y_train)
    # predict class labels for the test set
    predicted = et_cv.predict(X_test)
    # generate class probabilities
    probs_et = et_cv.predict_proba(X_test)
    return predicted, probs_et, et_cv



# extreme gradient boosting (uses regularization to control over-fitting)
# https://www.quora.com/What-is-the-difference-between-the-R-gbm-gradient-boosting-machine-and-xgboost-extreme-gradient-boosting
# git clone --recursive https://github.com/dmlc/xgboost.git  && cd xgboost && sh build.sh && cd python-package && python3 setup.py install --user
def extremeBoosting(X_train, y_train, X_test, state):
    print("training xgboost... \n");
    # set state
    np.random.seed(state)
    # define a search space
    space = {'max_depth': (5, 15),
              'learning_rate': (2/300, 10/20),
              'n_estimators': (20, 300),
              'gamma': (0, 5),
              'min_child_weight': (1, 20),
              'max_delta_step': (0, 10),
              'subsample': (0.3, 1.0),
              'colsample_bytree' : (max(0.3, 2/X_train.shape[1]), 1.0) if X_train.shape[1] >  10 else (1.0, 1.0)
            }
    # balance classes
    weight = sum(y_train == 0)/ sum(y_train == 1) #https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    # minimize the objective over the space
    best = BayesianOptimization (lambda max_depth,learning_rate,n_estimators,gamma,min_child_weight,max_delta_step,subsample,colsample_bytree,
                  silent =True,
                  nthread = -1,
                  seed = state:
                                 np.nanmean(cross_val_score(
                XGBClassifier(max_depth = int(max_depth),
                             learning_rate = learning_rate,
                             n_estimators = int(n_estimators),
                             silent = silent,
                             nthread = nthread,
                             gamma = gamma,
                             min_child_weight = int(min_child_weight),
                             max_delta_step = max_delta_step,
                             subsample = subsample,
                             colsample_bytree = colsample_bytree,
                             seed = seed,
                             scale_pos_weight = weight,
                             objective = "binary:logistic"),
                             X_train, y_train, scoring=make_scorer(precisionAtK, needs_proba = True),
                cv=StratifiedKFold(y_train, n_folds=3, shuffle=True, random_state=state))), space) # X_, y_, scoring=make_scorer(precisionAtK), cv=5).mean(), space) #
    best.maximize()
    print(best.res['max']['max_params'])
    # obtain best fit
    best_fit = XGBClassifier (n_estimators = int(best.res['max']['max_params']['n_estimators']), silent = True, nthread = -1, seed = state,
                                   max_depth = int(best.res['max']['max_params']['max_depth']),
                                   learning_rate = best.res['max']['max_params']['learning_rate'],
                                   gamma = (best.res['max']['max_params']['gamma']),
                                   min_child_weight = (best.res['max']['max_params']['min_child_weight']),
                                   max_delta_step = (best.res['max']['max_params']['max_delta_step']),
                                   subsample = (best.res['max']['max_params']['subsample']),
                                   colsample_bytree = best.res['max']['max_params']['colsample_bytree'],
                                   scale_pos_weight = weight, objective = "binary:logistic")
    xgb_cv  = best_fit.fit(X_train, y_train)
    # sorted(xgb_cv.booster().get_fscore().items(), key = lambda x: x[1])
    # predict class labels for the test set
    predicted = xgb_cv.predict(X_test)
    # generate class probabilities
    probs_xgb = xgb_cv.predict_proba(X_test)
    return predicted, probs_xgb, xgb_cv




# bayesian optimization for hyperparameter-tuning of one/two layer NN [tune hidden neurons, batch size and epoch number]
# add dropout regularization as per https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
def returnNNModelObjectwDropout(inputSize, nneurons1, nneurons2, dropout_ratio_input, dropout_ratio_hidden, state):
    # set state
    np.random.seed(state)
    model = Sequential()
    #model.add(Dropout(float(dropout_ratio_input), input_shape=(inputSize,)))
    model.add(Dense(nneurons1, input_dim=(inputSize), init='uniform', activation='tanh'))
    model.add(Dropout(float(dropout_ratio_hidden), input_shape=(nneurons1,)))
    model.add(Dense(nneurons2, init='normal', activation='tanh'))
    model.add(Dense(1, init='uniform', activation='sigmoid')) # try dropout layers
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # only 'accuracy' supported currently, categorical_crossentropy() if tertiary
    # print model.summary()
    return model


def buildPipelinewDropout(inputSize, nneurons1, nneurons2, dropout_ratio_input, dropout_ratio_hidden,  epoch, batch, state):
    # set state
    np.random.seed(state)
    # Two-layer NN
    # create model
    model = lambda : returnNNModelObjectwDropout(inputSize, nneurons1, nneurons2, dropout_ratio_input, dropout_ratio_hidden, state) # kerasclassifer takes function as input
    # build pipeline
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=model, nb_epoch=epoch, batch_size=batch, verbose = 0))) #verbose = 0
    pipeline = Pipeline(estimators)
    return pipeline


# bayesian optimization for hyperparameter-tuning of one/two layer NN [tune hidden neurons, batch size and epoch number]
# add dropout regularization as per https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
def layerNNwDropout(X_train, y_train, X_test, state):
    print("training NN... \n");
    # set state
    np.random.seed(state)
    inputDim = X_train.shape[1]
    ##
    #             'batch':(50, 50),
    #             'dropout_ratio_input': (0.0, 0.2),
    # define a search space
    space = {'nneurons1': (min(5, inputDim),inputDim),
             'dropout_ratio_hidden': (0.0, 0.4),
             'nneurons2': (5,700),
             'epoch': (1, 100),
             'batch': (10, 3000)
             }
    # minimize the objective over the space
    best = BayesianOptimization (lambda nneurons1, nneurons2, dropout_ratio_hidden, epoch, batch :
                                 np.nanmean(cross_val_score(
            buildPipelinewDropout(inputDim, int(nneurons1), int(nneurons2), float(0), float(dropout_ratio_hidden), int(epoch), int(batch), state),
                                 X_train, y_train, scoring=make_scorer(precisionAtK, needs_proba = True),
                cv=StratifiedKFold(y_train, n_folds=3, shuffle=True, random_state=state))), space) # X_, y_, scoring=make_scorer(precisionAtK), cv=5).mean(), space) #
    best.maximize()
    print(best.res['max']['max_params'])
    # obtain best fit #float(best.res['max']['max_params']['dropout_ratio_input'])
    best_fit = buildPipelinewDropout(inputDim, int(best.res['max']['max_params']['nneurons1']), int(best.res['max']['max_params']['nneurons2']),
                                     float(0) ,
                             float(best.res['max']['max_params']['dropout_ratio_hidden']), int(best.res['max']['max_params']['epoch']),
                             int(best.res['max']['max_params']['batch']), state)
    nn_cv  = best_fit.fit(X_train, y_train)
    # predict class labels for the test set
    predicted = nn_cv.predict(X_test)
    # generate class probabilities
    probs_nn = nn_cv.predict_proba(X_test)
    return predicted, probs_nn, nn_cv

# No tuning  (just use the big training-set with hard-coded knobs ) p/2 p/3
# add dropout regularization as per https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
def layerNNwDropoutHard(X_train, y_train, X_test, state):
    print("training NN... \n");
    # set state
    np.random.seed(state)
    inputDim = X_train.shape[1]
    # define a search space
    space = {'nneurons1': (min(5, inputDim),inputDim),
             'dropout_ratio_hidden': (0.0, 0.4),
             'epoch': (1, 100),
             'batch':(1, 200),
             'dropout_ratio_input': (0.0, 0.2),
             'nneurons2': (5,500)
             }
    # minimize the objective over the space
    """best = BayesianOptimization (lambda nneurons1, nneurons2, dropout_ratio_input, dropout_ratio_hidden, epoch, batch :
                                 np.nanmean(cross_val_score(
            buildPipelinewDropout(inputDim, int(nneurons1), int(nneurons2), float(dropout_ratio_input), float(dropout_ratio_hidden), int(epoch), int(batch), state),
                                 X_train, y_train, scoring=make_scorer(precisionAtK, needs_proba = True),
                cv=StratifiedKFold(y_train, n_folds=3, shuffle=True, random_state=state))), space) # X_, y_, scoring=make_scorer(precisionAtK), cv=5).mean(), space) #
    best.maximize()
    print(best.res['max']['max_params'])"""
    # obtain best fit #float(best.res['max']['max_params']['dropout_ratio_input'])
    best_fit = buildPipelinewDropout(inputDim, 600, 400,
                                     0 ,
                             0.2, 100,
                             1000, state)
    nn_cv  = best_fit.fit(X_train, y_train)
    # predict class labels for the test set
    predicted = nn_cv.predict(X_test)
    # generate class probabilities
    probs_nn = nn_cv.predict_proba(X_test)
    return predicted, probs_nn, nn_cv


# bayesian optimization for hyperparameter-tuning of random forest
def gradientBoosting(X_train, y_train, X_test, state):
    print("training gradientBoosting... \n");
    # set state
    np.random.seed(state)
    # define a search space
    space = {'max_depth': (5,20),
             'max_features': (0.1, 0.999),
             'min_samples_split': (1,20),
             'min_samples_leaf': (1,20),
             'n_estimators': (64, 128)
    }
    # minimize the objective over the space
    best = BayesianOptimization (lambda max_depth, min_samples_split, max_features, min_samples_leaf, n_estimators :
                                 np.nanmean(cross_val_score(
            GradientBoostingClassifier(n_estimators = int(n_estimators), random_state = state,
                                   max_depth = int(max_depth),
                                   max_features = min(max_features, 0.999),
                                   min_samples_split = int(min_samples_split),
                                   min_samples_leaf = int(min_samples_leaf)),
                                 X_train, y_train, scoring=make_scorer(precisionAtK, needs_proba = True),
                cv=StratifiedKFold(y_train, n_folds=3, shuffle=True, random_state=state))), space) # X_, y_, scoring=make_scorer(precisionAtK), cv=5).mean(), space) #
    best.maximize()
    print(best.res['max']['max_params'])
    # obtain best fit
    best_fit = GradientBoostingClassifier(n_estimators = int(best.res['max']['max_params']['n_estimators']), random_state = state,
                                   max_depth = int(best.res['max']['max_params']['max_depth']),
                                   max_features = best.res['max']['max_params']['max_features'],
                                   min_samples_split = int(best.res['max']['max_params']['min_samples_split']),
                                   min_samples_leaf = int(best.res['max']['max_params']['min_samples_leaf']))
    rf_cv  = best_fit.fit(X_train, y_train)
    # predict class labels for the test set
    predicted = rf_cv.predict(X_test)
    # generate class probabilities
    probs_rf = rf_cv.predict_proba(X_test)
    return predicted, probs_rf, rf_cv


def rankOrdered(array):
    order = array.argsort()
    ranked = np.empty(len(array))
    ranked[order] = np.linspace(0, len(array), len(array)+1)
    return ranked

def ensembleModelCalculation(probs_lr, probs_svm, probs_rf, probs_gb, probs_nn = 0.5, probs_erf = 0.5, probs_egb = 0.5):
    rank_lr = rankOrdered(probs_lr)
    rank_svm = rankOrdered(probs_svm)
    rank_rf = rankOrdered(probs_rf)
    rank_gb = rankOrdered(probs_gb)
    rank_erf = rankOrdered(probs_erf)
    rank_egb = rankOrdered(probs_egb)
    rank_nn = rankOrdered(probs_nn)
    # allocate weights if model is predicting something
    w_lr = 0 if np.all(probs_lr == 0) else 1
    w_svm = 0 if np.all(probs_svm == 0) else 1
    w_rf = 0 if np.all(probs_rf == 0) else 1
    w_gb = 0 if np.all(probs_gb == 0) else 1 # assign equal weights, since rf and gb are similar
    w_erf = 0 if np.all(probs_erf == 0) else 1
    w_egb = 0 if np.all(probs_egb == 0) else 1 # assign equal weights, since rf and gb are similar
    w_nn = 0 if np.all(probs_nn == 0) or np.all(probs_nn == 0) else 1
    w_tot = w_lr + w_svm + w_rf + w_gb + w_nn + w_erf + w_egb
    # average and normalize the ranks to prob_score in [0,1]
    return (w_lr*rank_lr + w_svm*rank_svm + w_rf*rank_rf + w_gb*rank_gb + w_nn*rank_nn + w_erf*rank_erf + w_egb*rank_egb)/(w_tot)/len(rank_lr)



def ensembleModel(X_train, y_train, X_test, state):
    print("training xgboost... \n"); predicted, probs_egb, obj = extremeBoosting(X_train, y_train, X_test, state)
    print("training kNearestNeighbours... \n"); predicted, probs_knn, obj = kNearestNeighbours(X_train, y_train, X_test, state)
    print("training ExtraTrees... \n"); predicted, probs_erf, obj = extremeTrees(X_train, y_train, X_test, state)
    print("training logisticRegression... \n"); predicted, probs_lr, obj = logisticRegression(X_train, y_train, X_test, state)
    print("training randomForest... \n"); predicted, probs_rf, obj = randomForest(X_train, y_train, X_test, state)
    print("training suppVector... \n"); predicted, probs_svm, obj = suppVector(X_train, y_train, X_test, state)
    print("training gradientBoosting... \n"); predicted, probs_gb, obj = gradientBoosting(X_train, y_train, X_test, state)
    print("training layerNNwDropout... \n"); predicted, probs_nn, obj = layerNNwDropout(X_train, y_train, X_test, state)
    #probs_nn = np.zeros([X_test.shape[0],2]) # maybe turn-off NN for now, not performing well. n/p not large enough. Decide to use with or without dropout
    probs_ensemble = ensembleModelCalculation(probs_lr[:,1], probs_svm[:,1], probs_rf[:,1], probs_gb[:,1], probs_nn[:,1], probs_erf[:,1], probs_egb[:,1], probs_knn[:,1])
    result = np.empty(shape = probs_lr.shape)
    result[:,1] = probs_ensemble
    # populate a placeholder for predicted (not used)
    predicted = np.zeros(result.shape[0])
    baseline = sum(y_train)/len(y_train)
    predicted[probs_ensemble > (1-baseline)] = 1 # label top-b % as successes where b is baseline seed->seriesA rate %
    return predicted, result, None


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
        col_prod = 'prod' + str(i) # measures diff at index i (a new feature)
        X[col_prod] =  X[i]*X[300 + i]
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

# MAIN
# get data
print("getting data and adding interactions between s1, s2...")
X_train = loadPickle("/Users/tarun/X_train.pkl"); X_train = addInteractions(X_train); print(("X_shape", X_train.shape))
X_dev = loadPickle("/Users/tarun/X_dev.pkl"); X_dev = addInteractions(X_dev);#print(X_dev[['x1_cosine_x2']].head())
y_train = np.array(loadPickle("/Users/tarun/y_train.pkl")); y_dev = np.array(loadPickle("/Users/tarun/y_dev.pkl"))

TINY_MODEL = True
state = 0

# TINY MODEL!! (train only on a small sample of training data for speed). Use PySpark on full-training set for speed
if TINY_MODEL:
    np.random.seed(state)
    # tinier
    sam_5K = np.random.choice(range(X_train.shape[0]), 10000)
    X_train_5K = X_train.loc[sam_5K,]
    y_train_5K = y_train[sam_5K]

    sam = np.random.choice(range(X_train.shape[0]), 10000)
    X_train = X_train.loc[sam,]
    y_train = y_train[sam]

# "Random Forest": randomForest(X_train, y_train, X_dev, state),"Gradient Boosting": gradientBoosting(X_train, y_train, X_dev, state),
#                "Two-layer NN":layerNNwDropout(X_train, y_train, X_dev, state)
#                 "SVM":suppVector(X_train_5K, y_train_5K, X_dev, state),

classifiers = {
               "Two-layer NN": layerNNwDropout(X_train, y_train, X_dev, state),
               "Regularized Gradient Boosting": extremeBoosting(X_train, y_train, X_dev, state),
               "Logistic regression": logisticRegression(X_train, y_train, X_dev, state),
               "Extremely Randomized Trees": extremeTrees(X_train, y_train, X_dev, state)
               }


probs = {}; predicted = {}
for method, clf in classifiers.items():
    predicted_method, probs_method, fit = clf;
    probs[method] = probs_method[:,1]
    predicted[method] = predicted_method #junk
    dropPickle(probs, "/Users/tarun/probs" + str(time.strftime("%Y%m%d-%H%M%S")) + ".pkl") # save to file


# PLOTTING ROUTINE (also computes ensemble Model)
#probs = loadPickle("/Users/tarun/probs.pkl")

#predicted_method, probs_method, fit = suppVector(X_train_5K, y_train_5K, X_dev, state)
#probs["SVM"] = probs_method[:,1]

#predicted_method, probs_method, fit = layerNNwDropoutHard(X_train, y_train, X_dev, state)
#probs["Two-layer NN"] = probs_method[:,1]

# ENSEMBLE EQUAL WEIGHTs MODEL
probs["Ensemble Model"] = ensembleModelCalculation(probs["Logistic regression"], 0*probs["Logistic regression"], 0*probs["Logistic regression"],0*probs["Logistic regression"],
                                                   probs["Two-layer NN"], probs["Extremely Randomized Trees"], probs["Regularized Gradient Boosting"])  #
# save files
#dropPickle(probs, "/Users/tarun/probs_.pkl") # always save a copy

# if loading from file
#probs = loadPickle("/Users/tarun/probs.pkl")
#y_dev = np.array(loadPickle("/Users/tarun/y_dev.pkl"))


# generate auc values for different models
auc_f = open("results/model_selection_prod_" + "_auc" + str(time.strftime("%Y%m%d-%H%M%S")) + ".txt", "w")
for method in probs.keys():
    # generate evaluation metrics
    auc = printMetrics(y_dev, probs[method])
    auc_f.write(method + str(auc) + "\n")
auc_f.close()


# GENERATE EVALUATION PLOTS -> ROC, Recall-Precision
# ROC plot
roc_filename = "results/model_selection_prod_roc" + str(time.strftime("%Y%m%d-%H%M%S")) + ".pdf"
plt.xlabel("FPR"); plt.ylabel("TPR (or Recall)"); plt.title("ROC Curve")
plt.xlim([0,1]); plt.ylim([0,1])
plt.plot([0,1], [0,1], '--k', label = "Random")

for method in probs.keys():
    fpr, tpr, thresholds = metrics.roc_curve(y_dev, probs[method])
    plt.plot(fpr, tpr, label = method)

plt.grid(True); plt.legend(loc = "lower right"); plt.savefig(roc_filename); plt.clf(); plt.close()

# Recall Precision plot
roc_filename = "results/model_selection_prod_RP" + str(time.strftime("%Y%m%d-%H%M%S")) + ".pdf"
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision vs Recall Curve")
plt.xlim([0,1]); plt.ylim([0,1])
plt.plot([0,len(y_dev)], [(sum(y_dev)/len(y_dev)),(sum(y_dev)/len(y_dev))], '--k', label = "Random")

for method in probs.keys():
    fpr, tpr, thresholds = metrics.precision_recall_curve(y_dev, probs[method])
    plt.plot(tpr, fpr, label = method)

plt.grid(True); plt.legend(loc = "lower right"); plt.savefig(roc_filename); plt.clf(); plt.close()

#close loggin
f.close()
