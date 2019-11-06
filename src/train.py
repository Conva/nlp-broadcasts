from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from joblib import dump, load
import json
import os
dirname = os.path.dirname(__file__)

model = SentenceTransformer('bert-base-nli-mean-tokens')


def train(data: list, classified: list):
    sentence_embeddings = model.encode(data)
    X = pd.DataFrame(np.row_stack(sentence_embeddings))
    Y = np.array(classified)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, Y)
    # Y_pred = clf.predict(X_test)
    # print("Accuracy", metrics.accuracy_score(Y_test, Y_pred))
    return clf


def train_event_categories(name: str):

    with open(os.path.join(dirname, "assets/{0}/train.json".format(name)), "r") as f:
        trainSet = json.load(f)
        data = trainSet["event_categories"]["data"]
        classified = trainSet["event_categories"]["classified"]
        clf = train(data, classified)
        dump(clf, os.path.join(
            dirname, "assets/{0}/event_categories_model.joblib".format(name)))


def train_locations(name: str):

    with open(os.path.join(dirname, "assets/{0}/train.json".format(name)), "r") as f:
        trainSet = json.load(f)
        data = trainSet["locations"]["data"]
        classified = trainSet["locations"]["classified"]
        clf = train(data, classified)
        dump(clf, os.path.join(
            dirname, "assets/{0}/locations.joblib".format(name)))


# train_locations("gatech")
