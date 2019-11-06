from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from joblib import dump, load

model = SentenceTransformer('bert-base-nli-mean-tokens')

def createAndRunClassifier():
    categoryList = [7,5,1,2,3,5,3,1,2,1,4,8,9,4,4,1,1,1,1,5,7,1,7,3,2,7,6,7,5,4,1,4,8,4,6,1,1,1,5,4,1,1,4,8,4,1,1,5,1,8,6,1,1,5,5,1,8,4,1,1,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4]
    titleList = ["German Resume and Career Consulting","Data Science at Georgia Tech Meeting","Melody and Meditation","Cultural and Religious/Spiritual Organization Roundtable","Tech's Giving","Ramblin' Rocket Club General Meeting","Students Organizing for Sustainability Fall 2019 Meeting","Heartfulness Meditation","Taste Of Africa 2019","Yoga for the Soul","DramaTech presents Yellow Face by David Henry Hwang","Weekend Group Run (Team Asha - Atlanta)","Design Jam with Wish for WASH (Part 2)","Fire & Ice Masquerade","DramaTech presents Yellow Face by David Henry Hwang","Meditation Retreat","The Heartfulness Way Retreat","Mindfulness and Meditation","Mental Health Student Coalition Open Meeting","Data Science at Georgia Tech Meeting","BHS Student Intern/Co-op Panel","Diabetes 101","Physician Assistant Club 4th General Meeting","Illuminate Tech","Debate - Tawheed or Trinity: Is God One or Three Divine Persons?","Fall 2019 Undergraduate Research Fair","SCPC Presents: Ultimate Smash Ultimate Tournament","GTSWE x Takeda: Pizza Night","Ramblin' Rocket Club General Meeting","DramaTech presents Yellow Face by David Henry Hwang","Yoga for the Soul","DramaTech presents Yellow Face by David Henry Hwang","Weekend Group Run (Team Asha - Atlanta)","DramaTech presents Yellow Face by David Henry Hwang","D&D Club Meeting","Mindfulness and Meditation","Mental Health Student Coalition Open Meeting","Data Science at Georgia Tech Meeting","Ramblin' Rocket Club General Meeting","DramaTech presents Yellow Face by David Henry Hwang","Heartfulness Meditation","Yoga for the Soul","DramaTech presents Yellow Face by David Henry Hwang","Weekend Group Run (Team Asha - Atlanta)","DramaTech presents Yellow Face by David Henry Hwang","Mindfulness and Meditation","Mental Health Student Coalition Open Meeting","Data Science at Georgia Tech Meeting","Yoga for the Soul","Weekend Group Run (Team Asha - Atlanta)","D&D Club Meeting","Mindfulness and Meditation","Mental Health Student Coalition Open Meeting","Data Science at Georgia Tech Meeting","Ramblin' Rocket Club General Meeting","Yoga for the Soul","Weekend Group Run (Team Asha - Atlanta)","The Atlanta Santa Clausterf@%!","Mindfulness and Meditation","Yoga for the Soul","Mindfulness and Meditation","Yoga for the Soul","DramaTech presents Tribes by Nina Raine","DramaTech presents Tribes by Nina Raine","DramaTech presents Tribes by Nina Raine","DramaTech presents Tribes by Nina Raine","DramaTech presents Tribes by Nina Raine","DramaTech presents A New Brain","DramaTech presents A New Brain","DramaTech presents A New Brain","DramaTech presents A New Brain","DramaTech presents A New Brain","DramaTech presents A New Brain","DramaTech presents A New Brain","DramaTech presents A New Brain"]
    sentence_embeddings = model.encode(titleList)
    X = pd.DataFrame(np.row_stack(sentence_embeddings))
    Y = np.array(categoryList)
    X.to_csv("categories.csv", index=False)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,Y_train)
    Y_pred=clf.predict(X_test)
    print("Accuracy", metrics.accuracy_score(Y_test,Y_pred))
    dump(clf, 'filename.joblib') 

clf = load('filename.joblib') 
test_title = ["DramaTech presents A New Brain"]
test_sentence_embeddings = model.encode(test_title)
test_X = pd.DataFrame(np.row_stack(test_sentence_embeddings))
test_Y_pred=clf.predict(test_X)
print(test_Y_pred)
