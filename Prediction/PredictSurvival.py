#!/usr/bin/python

from ProcessData import ReadCSV
from ProcessData import Graphing
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
import numpy as np


test_path = "../dataset/test.csv"
train_path = "../dataset/train.csv"


def recover_train_test_target(combined):

    train0 = pd.read_csv(train_path)

    targets = train0.Survived
    train = combined.head(891)
    test = combined.iloc[891:]

    return train, test, targets

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)

def process_names(combined):
    # we clean the Name variable
    combined.drop('Name', axis=1, inplace=True)

    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)

    # removing the title variable
    combined.drop('Title', axis=1, inplace=True)
    return combined


def process_fares(combined):
    # there's one missing fare value - replacing it with the mean.
    combined.head(891).Fare.fillna(combined.head(891).Fare.mean(), inplace=True)
    combined.iloc[891:].Fare.fillna(combined.iloc[891:].Fare.mean(), inplace=True)
    return combined


def process_embarked(combined):
    # two missing embarked values - filling them with the most frequent one (S)
    combined.head(891).Embarked.fillna('S', inplace=True)
    combined.iloc[891:].Embarked.fillna('S', inplace=True)

    # dummy encoding
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', axis=1, inplace=True)

    return combined


def process_cabin(combined):

    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U', inplace=True)

    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])

    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')

    combined = pd.concat([combined, cabin_dummies], axis=1)

    combined.drop('Cabin', axis=1, inplace=True)
    return combined



def process_sex(combined):
    # mapping string values to numerical one
    combined['Sex'] = combined['Sex'].map({'male': 1, 'female': 0})
    return combined



def process_pclass(combined):
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")

    # adding dummy variables
    combined = pd.concat([combined, pclass_dummies], axis=1)

    # removing "Pclass"

    combined.drop('Pclass', axis=1, inplace=True)
    return combined



def process_ticket(combined):


    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.', '')
        ticket = ticket.replace('/', '')
        ticket = ticket.split()
        ticket = map(lambda t: t.strip(), ticket)
        ticket = filter(lambda t: not t.isdigit(), ticket)
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'

    # Extracting dummy variables from tickets:

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop('Ticket', inplace=True, axis=1)
    return combined



def process_family(combined):
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    return combined






def get_titles(combined):

    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated titles
    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"

    }

    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)
    return combined



def get_combined_data():

    # reading train data
    train = pd.read_csv(train_path)

    # reading test data
    test = pd.read_csv(test_path)

    # extracting and then removing the targets from the training data
    targets = train.Survived
    train.drop('Survived', 1, inplace=True)

    # merging train data and test data for future feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)

    return combined

def predict():
    features=[5,6]
    from sklearn.naive_bayes import GaussianNB
    from sklearn import tree
    from sklearn.svm import SVC
    import matplotlib.pyplot as plt





    from ProcessData.ProcessData import getCSVData

    #Get Training data
    data = getCSVData(train_path)

    #Describe data

    #print data.describe(include='all')

    #Fix NaN in age
    data['Age'].fillna(data['Age'].median(), inplace=True)

    #visulaize on Gender
    survived_sex = data[data['Survived'] == 1]['Sex'].value_counts()
    dead_sex = data[data['Survived'] == 0]['Sex'].value_counts()
    df = pd.DataFrame([survived_sex, dead_sex])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(15, 8))
    #plt.show()

    #visualize on Age
    figure = plt.figure(figsize=(15, 8))
    plt.hist([data[data['Survived'] == 1]['Age'], data[data['Survived'] == 0]['Age']], stacked=True, color=['g', 'r'],
             bins=30, label=['Survived', 'Dead'])
    plt.xlabel('Age')
    plt.ylabel('Number of passengers')
    plt.legend()
    #plt.show()

    #visualize on Fare
    figure = plt.figure(figsize=(15, 8))
    plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']], stacked=True, color=['g', 'r'],
             bins=30, label=['Survived', 'Dead'])
    plt.xlabel('Fare')
    plt.ylabel('Number of passengers')
    plt.legend()
    #plt.show()

    #all 3 together
    plt.figure(figsize=(15, 8))
    ax = plt.subplot()
    ax.scatter(data[data['Survived'] == 1]['Age'], data[data['Survived'] == 1]['Fare'], c='green', s=40)
    ax.scatter(data[data['Survived'] == 0]['Age'], data[data['Survived'] == 0]['Fare'], c='red', s=40)
    ax.set_xlabel('Age')
    ax.set_ylabel('Fare')
    ax.legend(('survived', 'dead'), scatterpoints=1, loc='upper right', fontsize=15, )
    #plt.show()


    #Combine train and test data
    combined=get_combined_data()

    get_titles(combined)

    #fill age NaN with better data
    combined["Age"] = combined.groupby(['Sex', 'Pclass', 'Title'])['Age'].transform(lambda x: x.fillna(x.median()))

    combined=process_cabin(combined)
    combined =process_embarked(combined)
    combined =process_family(combined)
    combined =process_fares(combined)
    combined =process_names(combined)
    combined =process_pclass(combined)
    combined =process_sex(combined)
    combined = process_ticket(combined)

    train, test, targets = recover_train_test_target(combined)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    clf = clf.fit(train, targets)
    model = SelectFromModel(clf, prefit=True)
    train_reduced = model.transform(train)
    print train_reduced.shape
    test_reduced = model.transform(test)
    print test_reduced.shape

    # turn run_gs to True if you want to run the gridsearch again.
    run_gs = False

    if run_gs:
        parameter_grid = {
            'max_depth': [4, 6, 8],
            'n_estimators': [50, 10],
            'max_features': ['sqrt', 'auto', 'log2'],
            'min_samples_split': [1, 3, 10],
            'min_samples_leaf': [1, 3, 10],
            'bootstrap': [True, False],
        }
        forest = RandomForestClassifier()
        cross_validation = StratifiedKFold(targets, n_folds=5)

        grid_search = GridSearchCV(forest,
                                   scoring='accuracy',
                                   param_grid=parameter_grid,
                                   cv=cross_validation)

        grid_search.fit(train, targets)
        model = grid_search
        parameters = grid_search.best_params_

        print('Best score: {}'.format(grid_search.best_score_))
        print('Best parameters: {}'.format(grid_search.best_params_))
    else:
        parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50,
                      'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}

        model = RandomForestClassifier(**parameters)
        model.fit(train, targets)

    print compute_score(model, train, targets, scoring='accuracy')

    output = model.predict(test).astype(int)
    df_output = pd.DataFrame()
    aux = pd.read_csv(test_path)
    df_output['PassengerId'] = aux['PassengerId']
    df_output['Survived'] = output
    df_output[['PassengerId', 'Survived']].to_csv('../result.csv', index=False)


predict()