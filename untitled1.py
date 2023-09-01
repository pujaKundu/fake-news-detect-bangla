# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 18:55:48 2023

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:59:57 2023

@author: USER
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC


from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

import warnings
warnings.filterwarnings("ignore")

# Set a random seed for reproducibility
random_seed = 42

import sys
sys.path.insert(0, 'F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Results')
from accuracy_plot import accuracy_compare
from show_results import show_result
from show_results import show_plot_confusion_matrix
from show_results import show_roc_plot
from prediction import show_prediction
from counters import punctuation_counter

sys.path.insert(1, 'F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Preprocessing')
from stemming import stemming

"""Data Preprocessing
"""

#loading the dataset to pandas dataframe

real_news = pd.read_csv('F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Dataset/Authentic-48K.csv',nrows=3000)
fake_news = pd.read_csv('F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Dataset/Fake-1K.csv')
new_fake_news = pd.read_csv('F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Dataset/Fake-data-466.csv')
new_fake_news2 = pd.read_csv('F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Dataset/Fake-data-400.csv')
new_fake_news3 = pd.read_csv('F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Dataset/Fake-data-136.csv')

#concat csv files

news_dataset = pd.concat([real_news,fake_news,new_fake_news,new_fake_news2,new_fake_news3])
news_dataset = shuffle(news_dataset)
news_dataset.reset_index(inplace=True, drop=True)

#print(news_dataset.shape)

#counting the number of missing values in the dataset
print('number of null values : ')
print(news_dataset.isnull().sum())

#replacing the null values with empty string
news_dataset = news_dataset.fillna('')

#print(news_dataset['headline'])
#merging the news headline and title
news_dataset['content_data'] =news_dataset['domain']+' '+news_dataset['headline']+' '+news_dataset['content']

#separating the data and label

X=news_dataset.drop(columns='label',axis=1)
Y=news_dataset['label']



"""Stemming:
"""
news_dataset['content_data'] = news_dataset['content_data'].apply(stemming)
#print(news_dataset['content_data'])

#Separating data and label

X=news_dataset['content_data']
Y=news_dataset['label']

v=Y.value_counts()

print(v)

#training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=0)


#TfIDF Vectorizer
vectorizer = TfidfVectorizer()
XV_train = vectorizer.fit_transform(X_train)
XV_test = vectorizer.transform(X_test)

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC  

SVM = SVC(kernel='linear', random_state=random_seed)  
param_grid_SVM = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_SVM = GridSearchCV(SVM, param_grid_SVM, cv=5, scoring='accuracy')
grid_SVM.fit(XV_train, Y_train)
best_SVM_model = grid_SVM.best_estimator_

X_test_prediction_SVM = best_SVM_model.predict(XV_test)

test_data_accuracy_SVM = accuracy_score(X_test_prediction_SVM, Y_test)
test_data_f1_SVM = f1_score(X_test_prediction_SVM, Y_test)
test_data_precision_SVM= precision_score(X_test_prediction_SVM, Y_test)
test_data_recall_SVM = recall_score(X_test_prediction_SVM, Y_test)

show_result('Support Vector Machine', test_data_accuracy_SVM,test_data_f1_SVM,test_data_precision_SVM,test_data_recall_SVM, Y_test, X_test_prediction_SVM)
show_plot_confusion_matrix('Support Vector Machine',Y_test,X_test_prediction_SVM)


#Gradient Boosting Algorithm
from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(random_state=random_seed)

param_grid_GBC = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750]}

grid_GBC = GridSearchCV(estimator =GradientBoostingClassifier(max_depth=4, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=42), 
            param_grid = param_grid_GBC, scoring='accuracy',cv=5)
#grid_GBC = GridSearchCV(GBC, param_grid_GBC, cv=5, scoring='accuracy')
grid_GBC.fit(XV_train, Y_train)
best_GBC_model = grid_GBC.best_estimator_

print("Best Gradient Boosting Model:", best_GBC_model)
X_test_prediction_GBC = best_GBC_model.predict(XV_test)

test_data_accuracy_GBC = accuracy_score(X_test_prediction_GBC, Y_test)
test_data_f1_GBC = f1_score(X_test_prediction_GBC, Y_test)
test_data_precision_GBC = precision_score(X_test_prediction_GBC, Y_test)
test_data_recall_GBC = recall_score(X_test_prediction_GBC, Y_test)

show_result('Gradient Boosting Classifier', test_data_accuracy_GBC,test_data_f1_GBC,test_data_precision_GBC,test_data_recall_GBC, Y_test, X_test_prediction_GBC)
show_plot_confusion_matrix('Gradient Boosting Classifier',Y_test,X_test_prediction_GBC)


#predictive system    


print('According to Gradient Boosting Classifier:\n ')
show_prediction(1, XV_test,GBC)

# ROC-AUC score calculation
def calculate_roc_auc(model, X, y):
    y_probs = model.predict_proba(X)[:, 1]  # Probability of positive class
    return roc_auc_score(y, y_probs)

roc_auc_GBC = calculate_roc_auc(best_GBC_model, XV_test, Y_test)


print("Gradient Boosting:", roc_auc_GBC)


