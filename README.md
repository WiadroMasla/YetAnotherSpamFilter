# YetAnotherSpamFilter
Projekt ma na celu utworzenie modelu pozwalającego na filtrowanie spamu.
Projekt wykorzystuje biblioteki numpy, pandas i sklearn.

## Uruchomienie
Skrypt validationAndTests.py wykorzystuje GridSearchCV w celu walidacji modelu i prezentacji metryk wytrenowanego modelu.
Skrypt main.py udostępnia metodę createModel(), która trenuje i zwraca model na bazie wyników walidacji; oraz przykładową metodę classify(), która prezentuje przykładowe wykorzystanie modelu.

## Model
Model składa się z CountVectorizer i MultinomialNB. CountVectorizer zlicza słowa i zwraca ich wektor krotności. MultinomialNB jest wariantem Bayes'a.
