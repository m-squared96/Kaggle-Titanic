#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


class ModelBuilder():

    def __init__(self):

        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")

        self.data_prepare(train, test)

    def data_prepare(self, train, test):

        for dataset in (train, test):
            dataset = self.flags(dataset)

        self.train = train.select_dtypes(include=np.number).drop("Age", axis=1).fillna(value=0)
        self.test = test.select_dtypes(include=np.number).drop("Age", axis=1).fillna(value=0)

    def log_construct(self):

        logmodel = LogisticModel(self.train, self.test)
        logmodel.model_construct()

    def dtree_construct(self):

        dtree = DTreeModel(self.train, self.test)
        dtree.model_construct()

    def rf_construct(self):

        rf = RFModel(self.train, self.test)
        rf.model_construct()

    def ada_construct(self):

        ada = AdaBoostModel(self.train, self.test)
        ada.model_construct()

    def svm_construct(self):

        svm = SVMModel(self.train, self.test)
        svm.model_construct()

    def submission(self):

        submission_frame = pd.DataFrame([])
        submission_frame["PassengerId"] = self.test["PassengerId"]
        submission_frame["Survived"] = self.logmodel.predict(self.test.drop("PassengerId", axis=1))
        submission_frame.to_csv("titanic_submission_file.csv", index=False)

    def flags(self, data):

        data["CabinFlag"] = data["Cabin"].apply(self.present_flag)
        data["CabinLabel"] = data["Cabin"].apply(self.cabin_label)

        data["AgeFlag"] = data["Age"].apply(self.present_flag)
        data["RevisedAge"] = data[["Age", "Pclass"]].apply(self.approx_age, axis=1)

        data["EmbarkedFlag"] = data["Embarked"].apply(self.embarked)
        data["GenderFlag"] = data["Sex"].apply(self.gender_flag)
        data["DrFlag"] = data["Name"].apply(self.dr_flag)

        return data

    def present_flag(self, x):

        if pd.isna(x):
            return 0
        else:
            return 1

    def cabin_label(self, x):

        if pd.isnull(x):
            return 0

        else:

            label = x[0]

            if label in ("A", "a"):
                return 1

            elif label in ("B", "b"):
                return 2

            elif label in ("C", "c"):
                return 3

            elif label in ("D", "d"):
                return 4

            elif label in ("E", "e"):
                return 5

            elif label in ("F", "f"):
                return 6

            elif label in ("G", "g"):
                return 7

    def approx_age(self, cols):

        Age = cols[0]
        Pclass = cols[1]

        if pd.isnull(Age):
            if Pclass == 1:
                return 38

            if Pclass == 2:
                return 30

            if Pclass == 3:
                return 25
        
        else:
            return Age

    def embarked(self, x):

        if pd.isnull(x):
            return 0

        else:
            if x == "S":
                return 1

            elif x == "C":
                return 2

            elif x == "Q":
                return 3

    def gender_flag(self, x):

        if pd.isnull(x):
            return 0

        else:
            if x == "male":
                return 1

            elif x == "female":
                return 2

    def dr_flag(self, x):

        if pd.isnull(x):
            return 0

        else:
            try:
                x.index("Dr.")
                return 1

            except ValueError:
                return 0

    def fare_plug(self, x):

        if not pd.isnull(x):
            return x


class LogisticModel():

    def __init__(self, train, test):

        self.train = train
        self.test = test

    def model_construct(self):

        x = self.train.drop(["PassengerId", "Survived"], axis=1)
        y = self.train["Survived"]

        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
        #print(self.train.isnull().sum().sum())
        self.logmodel = LogisticRegression()
        self.logmodel.fit(x_train, y_train)
        predictions = self.logmodel.predict(x_test)

        print("\nLogistic Regression Model:")
        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))


class DTreeModel():

    def __init__(self, train, test):

        self.train = train
        self.test = test

    def model_construct(self):

        x = self.train.drop(["PassengerId", "Survived"], axis=1)
        y = self.train["Survived"]                
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

        self.dtreemodel = DecisionTreeClassifier()
        self.dtreemodel.fit(x_train, y_train)
        predictions = self.dtreemodel.predict(x_test)

        print("\nDecision Tree Model:")
        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))


class RFModel():

    def __init__(self, train, test):

        self.train = train
        self.test = test

    def model_construct(self):

        x = self.train.drop(["PassengerId", "Survived"], axis=1)
        y = self.train["Survived"]                
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

        self.rfmodel = RandomForestClassifier(n_estimators=1000)
        self.rfmodel.fit(x_train, y_train)
        predictions = self.rfmodel.predict(x_test)

        print("\nRandom Forest Model:")
        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))


class AdaBoostModel():

    def __init__(self, train, test):

        self.train = train
        self.test = test

    def model_construct(self):

        x = self.train.drop(["PassengerId", "Survived"], axis=1)
        y = self.train["Survived"]                
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

        self.adamodel = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=500) , n_estimators=500)
        self.adamodel.fit(x_train, y_train)
        predictions = self.adamodel.predict(x_test)

        print("\nAdaBoost Model:")
        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))

class SVMModel():

    def __init__(self, train, test):

        self.train = train
        self.test = test

    def model_construct(self):

        x = self.train.drop(["PassengerId", "Survived"], axis=1)
        y = self.train["Survived"]                
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

        self.svcmodel = SVC(C=5.6)
        self.svcmodel.fit(x_train, y_train)
        predictions = self.svcmodel.predict(x_test)

        print("\nSVM Model:")
        print(classification_report(y_test, predictions))
        print(self.svcmodel.score(x_test, y_test))


builder = ModelBuilder()
builder.log_construct()
builder.dtree_construct()
builder.rf_construct()
builder.svm_construct()
#builder.ada_construct()
