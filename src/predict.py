from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from joblib import dump, load
from enum import Enum
import os
dirname = os.path.dirname(__file__)

model = SentenceTransformer('bert-base-nli-mean-tokens')


class Category(Enum):
    SPORTS = 0
    WELLBEING = 1
    CULTURAL = 2
    COMMUNITY = 3
    ARTS = 4
    TECHNOLOGY = 5
    GAMES = 6
    CAREER = 7
    EXERCISE = 8
    OTHER = 9


def predict(clf, phrase: str):
    test_title = [phrase]
    test_sentence_embeddings = model.encode(test_title)
    test_X = pd.DataFrame(np.row_stack(test_sentence_embeddings))
    test_Y_pred = clf.predict(test_X)
    index = int(test_Y_pred)
    print(index)
    return index


def predict_event_categories(name: str, phrase: str):
    clf = load(os.path.join(dirname, "./assets/{0}/event_categories_model.joblib".format(name)))
    index = predict(clf, phrase)
    return Category(index).name


def predict_locations(name: str, phrase: str):
    clf = load(os.path.join(dirname, "./assets/{0}/locations.joblib".format(name)))
    index = predict(clf, phrase)
    return index

# predict_event_categories("gatech","yoga")
# predict_locations("gatech","GTRI")