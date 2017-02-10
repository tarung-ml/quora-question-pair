from sklearn.linear_model import LogisticRegression
from sklearn import grid_search
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from util.helper import dropPickle, loadPickle


epsilon = 1e-5

def cosine(arr1, arr2):
    return arr1.dot(arr2) / (epsilon + np.sqrt(sum(arr1 ** 2))) / (epsilon + np.sqrt(sum(arr2 ** 2)))

def addInteractions(X):
    X['x1_cosine_x2'] = X.apply(lambda row: cosine(np.array(row[0:299]), np.array(row[300:599])), axis = 1)
    for i in range(0, 300):
        col = 'plus'+ str(i)
        X[col] =  X[i] + X[300 + i]
        col = 'minus' + str(i)
        X[col] =  X[i] - X[300 + i]
    return X

import matplotlib.pyplot as plt
plt.interactive(False) # maybe pycharm specific

X_train = loadPickle("/Users/tarun/X_train.pkl"); X_train = addInteractions(X_train);print(X_train.head())
X_dev = loadPickle("/Users/tarun/X_dev.pkl"); X_dev = addInteractions(X_dev);print(X_dev[['x1_cosine_x2']].head())
y_train = np.array(loadPickle("/Users/tarun/y_train.pkl")); y_dev = np.array(loadPickle("/Users/tarun/y_dev.pkl"))

TINY_MODEL = False
state = 0

# TINY MODEL!! (train only on a small sample of training data for speed). Use PySpark on full-training set for speed
if TINY_MODEL:
    np.random.seed(state)
    sam = np.random.choice(range(X_train.shape[0]), 10000)
    X_train = X_train.loc[sam,]
    y_train = y_train[sam]

    # fit a logistic regression and tune the regularization param and other hyper-parameters using cross-val on training-set
    state = 0; Cs = [0.1] #np.linspace(0.01, 5, 10) # gridSearch tries to mindlessly (pure exploration) search the grid
                                             # and narrow down on optimal C. Will try Bayesian Optimization.
    lr = LogisticRegression(dual=False, class_weight="balanced", solver='liblinear', random_state=state, verbose = 1)
    lr_cv = grid_search.GridSearchCV(estimator=lr, param_grid=dict(C=Cs, penalty=['l1', 'l2']),
                                     scoring='roc_auc',
                                     cv=StratifiedKFold(y_train, n_folds=3, shuffle=True, random_state=state))

    lr_cv = lr_cv.fit(X_train, y_train)
    # check which value of C was selected
    print(lr_cv.best_params_['C'])
    # select the best estimator
    print(lr_cv.best_estimator_)
    lr_cv = lr_cv.best_estimator_.fit(X_train, y_train)
else:
    # FOR BIG MODEL ONLY !  (use pre-tuned C, and penalty-type)
    lr_cv = LogisticRegression(dual=False, class_weight="balanced", solver='liblinear', random_state=state, verbose = 1, C = 0.1, penalty = 'l1')
    lr_cv = lr_cv.fit(X_train, y_train)

# check number of coefficients selected
print(np.count_nonzero(lr_cv.coef_))  # non-zero coefficients
# check features and their coeffs.
print([(x, y) for (x, y) in
       zip(X_train.columns[np.nonzero(lr_cv.coef_)[1]], lr_cv.coef_[0][np.nonzero(lr_cv.coef_[0])[0]])])
# predict class labels for the test set
predicted = lr_cv.predict(X_dev)
# generate class probabilities
probs_lr = lr_cv.predict_proba(X_dev)

dropPickle(probs_lr, "/Users/tarun/probs_lr.pkl")

# print evaluation on dev set
print(metrics.roc_auc_score(y_dev, probs_lr[:, 1]))
print(metrics.average_precision_score(y_dev, probs_lr[:, 1]))
with open("results/bow_LR_eval.txt", "w") as f:
    f.write("AUC, AP :" + str(metrics.roc_auc_score(y_dev, probs_lr[:, 1]))
            + ", " + str(metrics.average_precision_score(y_dev, probs_lr[:, 1])))

# GENERATE EVALUATION PLOTS -> ROC, Recall-Precision
# ROC plot
roc_filename = "results/bow_LR_roc.pdf"
plt.xlabel("FPR"); plt.ylabel("TPR (or Recall)"); plt.title("ROC Curve")
plt.xlim([0,1]); plt.ylim([0,1])
plt.plot([0,1], [0,1], '--k', label = "Random")
fpr, tpr, thresholds = metrics.roc_curve(y_dev, probs_lr[:, 1])
plt.plot(fpr, tpr, label = "Logistic Regression")
plt.grid(True); plt.legend(loc = "lower right"); plt.savefig(roc_filename); plt.clf(); plt.close()

# Recall Precision plot
roc_filename = "results/bow_LR_pr.pdf"
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision vs Recall Curve")
plt.xlim([0,1]); plt.ylim([0,1])
plt.plot([0,len(y_dev)], [(sum(y_dev)/len(y_dev)),(sum(y_dev)/len(y_dev))], '--k', label = "Random")
fpr, tpr, thresholds = metrics.precision_recall_curve(y_dev, probs_lr[:, 1])
plt.plot(tpr, fpr, label = "Logistic Regression")
plt.grid(True); plt.legend(loc = "lower right"); plt.savefig(roc_filename); plt.clf(); plt.close()

# An important observation: traning on full-sample doesn't really help that much. For model selection,
# we can try a smaller sample of data.