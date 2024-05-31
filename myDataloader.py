#splits dataset into train, validate, test
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class MyDataloader:
    def __init__ (self, random_state=42, test_frac=0.3):
        df = pd.read_csv("emails.csv")
        #preprocessing
        
        df["text"]=df["text"].str.replace("Subject:","")
        X = df["text"]
        y = df["spam"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = test_frac, random_state= random_state)
    
    def get_train(self):
        return [self.X_train, self.y_train]
    

    def get_test(self):
        return [self.X_test, self.y_test]



