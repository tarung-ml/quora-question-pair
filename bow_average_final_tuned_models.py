## BEST HYPERPARAMS (FOR TINY_DATA = 10K samples) FROM LOG FILE
import pandas as pd
#ML
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from xgboost.sklearn import XGBClassifier
#DL
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# metrics
from sklearn import metrics
#plots
from util.helper import dropPickle, loadPickle
import matplotlib.pyplot as plt
plt.interactive(False) # maybe pycharm specific
import time

from util.helper import dropPickle, loadPickle, save_model, load_model
import numpy as np

# helper function to print metrics
def printMetrics(y_test, probs):
    #print(metrics.accuracy_score(y_test, predicted))
    auc = (metrics.roc_auc_score(y_test, probs))
    print('auc:' + str(auc))
    #print('conf_mat:' + str(metrics.confusion_matrix(y_test, predicted)))
    #print('met:' + str(metrics.classification_report(y_test, predicted)))
    return auc



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


# given continous class probability, find optimal cutoff and generate predictions
# use cutoff that maximizes (sensitivity (TPR) + specificty (TNR or 1-FPR)) in ROC curve
def probsToPred(probs, y_dev):
    print(len(y_dev))
    fpr, tpr, thresholds = metrics.roc_curve(y_dev, probs)
    obj = tpr + (1-fpr)
    maxIndex =  (obj.argsort()[::-1][0])
    print(obj[maxIndex])
    return (thresholds[maxIndex])


#MAIN

state = 0
probs = {} # dicitonary to hold result for different classifiers
predicted = {}
# get data
X_train = loadPickle("/Users/tarun/X_train.pkl"); X_train = addInteractions(X_train); print(("X_shape", X_train.shape))
y_train = np.array(loadPickle("/Users/tarun/y_train.pkl"));
y_dev= np.array(loadPickle("/Users/tarun/y_dev.pkl"))
X_dev = loadPickle("/Users/tarun/X_dev.pkl"); X_dev = addInteractions(X_dev);



# logistic
C = 0.1
penalty = 'l1'
print("training LR...")
lr = LogisticRegression(dual=False, class_weight="balanced", solver='liblinear', random_state=state, verbose=1, C=C, penalty=penalty)
lr.fit(X_train, y_train)
dropPickle(lr, "/Users/tarun/lr.pkl")
probs_lr = lr.predict_proba(X_dev)
dropPickle(probs_lr, "/Users/tarun/probs_lr.pkl")
probs_lr = loadPickle("/Users/tarun/probs_lr.pkl")
probs["Logistic Regression"] = probs_lr[:,1]
lr = loadPickle("/Users/tarun/lr.pkl")
predicted_lr = lr.predict(X_dev); print(metrics.accuracy_score(y_dev, predicted_lr))
predicted["Logistic Regression"] = predicted_lr

print("training NN...")

nn_params = {'nneurons1': 723.68041461328903, 'dropout_ratio_hidden': 0.080626647957330419,
 'nneurons2': 401.82991897130654, 'epoch': 95.141409242381044, 'batch': 1144.2029579701891}
nn_params = {'nneurons1': 1077.0688601922568, 'dropout_ratio_hidden': 0.21795327319875876,
             'nneurons2': 324.71758596799475, 'epoch': 89.285527077425897, 'batch': 1156.4901412890754}


inputDim = X_train.shape[1]
nn = returnNNModelObjectwDropout(inputDim, int(nn_params['nneurons1']),
                                 int(nn_params['nneurons2']),
                                 float(0),
                                 float(nn_params['dropout_ratio_hidden']),state)
nn.fit(X_train.as_matrix(), (y_train), nb_epoch = int(nn_params['epoch']),
                                batch_size = int(nn_params['batch']))
save_model(nn);
#probs_nn = nn.predict_proba(X_dev)
nn = load_model();
probs_nn = nn.predict(X_dev.as_matrix())
dropPickle(probs_nn, "/Users/tarun/probs_nn.pkl")
probs_nn = loadPickle("/Users/tarun/probs_nn.pkl")
probs["Two-layer NN"] = probs_nn
predicted_nn = nn.predict_classes(X_dev.as_matrix());
predicted["Two-layer NN"] = predicted_nn

print(metrics.accuracy_score(y_dev, predicted_nn));
print(metrics.roc_auc_score(y_dev, probs_nn));
print(metrics.average_precision_score(y_dev, probs_nn))

print("training EGB...")

xgboost_params = {'max_depth': 5.4010507677659927, 'learning_rate': 0.14525482333645176,
 'n_estimators': 214.46942538892159, 'gamma': 0.070027735703458993,
 'min_child_weight': 19.984629266901518, 'max_delta_step': 9.1309588410154987,
 'subsample': 0.62572584626366734, 'colsample_bytree': 0.90655225081438462}
xgboost_params = {'max_depth': 5.2362189654402744, 'learning_rate': 0.42073456089650635,
                  'n_estimators': 298.76580117424407, 'gamma': 0.39598121767507211,
                  'min_child_weight': 19.687102537529192, 'max_delta_step': 9.6235806376445989,
                  'subsample': 0.80393409185325226, 'colsample_bytree': 0.7814669621409378}
# balance classes
weight = sum(y_train == 0) / sum(y_train == 1)  # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
xgb = XGBClassifier(n_estimators=int(xgboost_params['n_estimators']), silent=True, nthread=-1,
                         seed=state,
                         max_depth=int(xgboost_params['max_depth']),
                         learning_rate=xgboost_params['learning_rate'],
                         gamma=(xgboost_params['gamma']),
                         min_child_weight=(xgboost_params['min_child_weight']),
                         max_delta_step=(xgboost_params['max_delta_step']),
                         subsample=(xgboost_params['subsample']),
                         colsample_bytree=xgboost_params['colsample_bytree'],
                         scale_pos_weight=weight, objective="binary:logistic")
xgb.fit(X_train, y_train)
dropPickle(xgb, "/Users/tarun/xgb.pkl")
probs_xgb = xgb.predict_proba(X_dev)
dropPickle(probs_xgb, "/Users/tarun/probs_xgb.pkl")
probs_xgb = loadPickle("/Users/tarun/probs_xgb.pkl")
probs["Regularized Gradient Boosting"] = probs_xgb[:,1]
xgb = loadPickle("/Users/tarun/xgb.pkl")
predicted_xgb = xgb.predict(X_dev); print(metrics.accuracy_score(y_dev, predicted_xgb))
predicted["Regularized Gradient Boosting"] = predicted_xgb

print("training ET...")

et_params = {'max_depth': 18.920408567011741, 'max_features': 0.5228547869126724,
 'min_samples_split': 2.3979081629701948, 'min_samples_leaf': 19.624364841583301,
 'n_estimators': 126.66223678558441}
et_params = {'max_depth': 19.887056359208664, 'max_features': 0.56638799507556647,
             'min_samples_split': 19.54653343959329, 'min_samples_leaf': 1.2071344551101633, 'n_estimators': 89.34618629386847}

et = ExtraTreesClassifier(n_estimators=int(et_params['n_estimators']),
                                class_weight="balanced", random_state=state,
                                max_depth=int(et_params['max_depth']),
                                max_features=et_params['max_features'],
                                min_samples_split=int(et_params['min_samples_split']),
                                min_samples_leaf=int(et_params['min_samples_leaf']),
                                criterion="gini")
et.fit(X_train, y_train)
dropPickle(et, "/Users/tarun/et.pkl")
probs_et = et.predict_proba(X_dev)
dropPickle(probs_et, "/Users/tarun/probs_et.pkl")
probs_et = loadPickle("/Users/tarun/probs_et.pkl")
probs["Extremely Randomized Trees"] = probs_et[:,1]
et = loadPickle("/Users/tarun/et.pkl")
predicted_et = et.predict(X_dev); print(metrics.accuracy_score(y_dev, predicted_et))
predicted["Extremely Randomized Trees"] = predicted_et



# ENSEMBLE EQUAL WEIGHTs MODEL
probs["Ensemble Model"] = ensembleModelCalculation(probs["Logistic Regression"], 0*probs["Two-layer NN"], 0*probs["Two-layer NN"],0*probs["Two-layer NN"],
                                                   probs["Two-layer NN"], probs["Extremely Randomized Trees"], probs["Regularized Gradient Boosting"])  #


# find optimal thresholds for different models
thresh_f= open("results/full_model" + "_thresh" + str(time.strftime("%Y%m%d-%H%M%S")) + ".txt", "w")
for method in probs.keys():
    # generate evaluation metrics
    thresh = probsToPred(probs[method], y_dev)
    thresh_f.write(method + " " + str(thresh)  + "\n")
thresh_f.close()


# generate auc values for different models
auc_f = open("results/full_model" + "_auc" + str(time.strftime("%Y%m%d-%H%M%S")) + ".txt", "w")
for method in probs.keys():
    # generate evaluation metrics
    auc = printMetrics(y_dev, probs[method])
    auc_f.write(method + " " + str(auc) + "\n")
auc_f.close()

# generate acc values for different models
acc_f = open("results/full_model" + "_acc" + str(time.strftime("%Y%m%d-%H%M%S")) + ".txt", "w")
for method in predicted.keys():
    # generate evaluation metrics
    acc = metrics.accuracy_score(y_dev, predicted[method])
    acc_f.write(method + " " + str(acc) + "\n")
acc_f.close()

# generate roc values for different models
ap_f= open("results/full_model" + "_ap" + str(time.strftime("%Y%m%d-%H%M%S")) + ".txt", "w")
for method in probs.keys():
    # generate evaluation metrics
    ap = metrics.average_precision_score(y_dev, probs[method])
    ap_f.write(method + " " + str(ap) + "\n")
ap_f.close()

#


# GENERATE EVALUATION PLOTS -> ROC, Recall-Precision
# ROC plot
roc_filename = "results/full_model_roc" + str(time.strftime("%Y%m%d-%H%M%S")) + ".pdf"
plt.xlabel("FPR"); plt.ylabel("TPR (or Recall)"); plt.title("ROC Curve")
plt.xlim([0,1]); plt.ylim([0,1])
plt.plot([0,1], [0,1], '--k', label = "Random")

for method in probs.keys():
    fpr, tpr, thresholds = metrics.roc_curve(y_dev, probs[method])
    plt.plot(fpr, tpr, label = method)
plt.grid(True); plt.legend(loc = "lower right"); plt.savefig(roc_filename); plt.clf(); plt.close()

# Sensitivty Specificity splot plot
roc_filename = "results/full_model_spc" + str(time.strftime("%Y%m%d-%H%M%S")) + ".pdf"
plt.xlabel("Specificity"); plt.ylabel("Sensitivity"); plt.title("Sensitivty vs/ Specificty")
plt.xlim([0,1]); plt.ylim([0,1])
plt.plot([0,1], [1,0], '--k', label = "Random")

for method in probs.keys():
    fpr, tpr, thresholds = metrics.roc_curve(y_dev, probs[method])
    plt.plot(1 - fpr, tpr, label = method)

plt.grid(True); plt.legend(loc = "lower right"); plt.savefig(roc_filename); plt.clf(); plt.close()

# Recall Precision plot
roc_filename = "results/full_model_RP" + str(time.strftime("%Y%m%d-%H%M%S")) + ".pdf"
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision vs Recall Curve")
plt.xlim([0,1]); plt.ylim([0,1])
plt.plot([0,len(y_dev)], [(sum(y_dev)/len(y_dev)),(sum(y_dev)/len(y_dev))], '--k', label = "Random")

for method in probs.keys():
    fpr, tpr, thresholds = metrics.precision_recall_curve(y_dev, probs[method])
    plt.plot(tpr, fpr, label = method)

plt.grid(True); plt.legend(loc = "lower right"); plt.savefig(roc_filename); plt.clf(); plt.close()
