from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import myDataloader

def createModel():
    dataloader = myDataloader.MyDataloader()
    X, y = dataloader.get_train()

    #based on GridSearchCV result
    model = Pipeline([("vectorizer", CountVectorizer()), ("scaler", None), ("clf", MultinomialNB(alpha=1.0))])
    model.fit(X, y)
    return model

trainedModel = createModel()

#1 for spam, else 0
def classify(email_content):
    return trainedModel.predict(np.array([email_content]))[0]

print(classify("Click here to win a free car!"))

