import pandas as pd
import sklearn.metrics as metrics
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, PredefinedSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC
import myDataloader

random_state = 42
dataloader = myDataloader.MyDataloader(random_state=random_state)
test_X, test_y = dataloader.get_test()

train_X, train_y = dataloader.get_train()

model = Pipeline([("vectorizer", CountVectorizer()), ("scaler", None), ("clf", MultinomialNB(alpha=1.0))])
param_grid = [
    {
        "clf" : [MultinomialNB(), ComplementNB()],
        "scaler" : [MaxAbsScaler(), None],
        "clf__alpha" : [100, 10, 1, 0.1]
    },
    {
        "clf" : [SVC()],
        "scaler" : [StandardScaler(), None],
        "clf__kernel" : ["linear"],
        "clf__C" : [0.001, 0.01, 0.1, 1, 10]
    },
    {
        "clf" : [SVC()],
        "scaler" : [StandardScaler(), None],
        "clf__kernel" : ["rbf"],
        "clf__C" : [0.1, 1, 10, 100],
        "clf__gamma" : [0.1, 1, 10, 100]
    }
]
k_fold = KFold(n_splits = 5)
gs = GridSearchCV(model, param_grid, n_jobs=-1, refit=True, cv=k_fold)
gs.fit(train_X, train_y)
print(gs.best_params_)
print("____________")
print("TRAIN")
print(metrics.accuracy_score(train_y, gs.predict(train_X)))
print(metrics.precision_score(train_y, gs.predict(train_X)))
print(metrics.recall_score(train_y, gs.predict(train_X)))
print(metrics.roc_auc_score(train_y, gs.predict(train_X)))
print(metrics.r2_score(train_y, gs.predict(train_X)))
print("____________")
print("TEST")
print(metrics.accuracy_score(test_y, gs.predict(test_X)))
print(metrics.precision_score(test_y, gs.predict(test_X)))
print(metrics.recall_score(test_y, gs.predict(test_X)))
print(metrics.roc_auc_score(test_y, gs.predict(test_X)))
print(metrics.r2_score(test_y, gs.predict(test_X)))
