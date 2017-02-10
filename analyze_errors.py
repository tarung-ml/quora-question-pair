from util.helper import dropPickle, loadPickle, save_model, load_model, addInteractions
import numpy as np, pandas as pd
from sklearn import metrics


"""
y_dev= np.array(loadPickle("/Users/tarun/y_dev.pkl"))
X_dev = loadPickle("/Users/tarun/X_dev.pkl"); X_dev = addInteractions(X_dev);


nn = load_model();
predicted_nn = nn.predict_classes(X_dev.as_matrix());
sentences_dev = loadPickle("/Users/tarun/sentences_dev.pkl")


view = (pd.DataFrame({'sentence_pair': sentences_dev, '_truth': y_dev.tolist(), '_predicted':[x[0] for x in predicted_nn.tolist()]}))
(view[view['_truth'] != view['_predicted']]).to_csv("errors_nn.csv", index=False)

#(view[view['_truth'] == view['_predicted']]).to_csv("not_errors_nn.csv", index=False)

"""

# error across question types
errors_nn = pd.read_csv("errors_nn.csv")
errors_nn['qtype'] = errors_nn['sentence_pair'].apply(lambda x: x.split()[0])
errors_nn['len'] = errors_nn['sentence_pair'].apply(lambda x: len(x.split(";")[0]))
print(errors_nn.groupby(['qtype']).count().sort_values('_predicted', ascending = False)['_predicted'][0:10]/errors_nn.shape[0])
print(errors_nn['len'].describe())
print(errors_nn['len'].median())


errors_nn = pd.read_csv("not_errors_nn.csv")
errors_nn['qtype'] = errors_nn['sentence_pair'].apply(lambda x: x.split()[0])
errors_nn['len'] = errors_nn['sentence_pair'].apply(lambda x: len(x.split(";")[0]))
print(errors_nn.groupby(['qtype']).count().sort_values('_predicted', ascending = False)['_predicted'][0:10]/errors_nn.shape[0])
print(errors_nn['len'].describe())
print(errors_nn['len'].median())

