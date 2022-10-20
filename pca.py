# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 20:45:17 2022

@author: USER
"""

# -*- coding: utf-8 -*-

"""Importing dependencies"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, 'F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Results')
from accuracy_plot import accuracy_compare
from show_results import show_result
from show_results import show_plot_confusion_matrix
from prediction import show_prediction
from counters import punctuation_counter

sys.path.insert(1, 'F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Preprocessing')
from stemming import stemming

import nltk
nltk.download('stopwords')

"""Data Preprocessing
"""

#loading the dataset to pandas dataframe

real_news = pd.read_csv('F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Dataset/Authentic-48K.csv')
fake_news = pd.read_csv('F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Dataset/Fake-1K.csv')

#concat two csv files

news_dataset = pd.concat([real_news,fake_news])
news_dataset = shuffle(news_dataset)
news_dataset.reset_index(inplace=True, drop=True)

#counting the number of missing values in the dataset
news_dataset.isnull().sum()

#replacing the null values with empty string
news_dataset = news_dataset.fillna('')

#merging the news headline and title
news_dataset['data'] =news_dataset['domain']+' '+news_dataset['category']+' '+ news_dataset['headline']+' '+news_dataset['content']

#separating the data and label

X=news_dataset.drop(columns='label',axis=1)
Y=news_dataset['label']

"""Stemming:
"""
news_dataset['data'].apply(stemming)

#Separating data and label

X=news_dataset['data']
Y=news_dataset['label']

#Y.shape
#training and testing data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=0)

#Hashing Vectorizer

from sklearn.feature_extraction.text import HashingVectorizer
vectorizer = HashingVectorizer(n_features=2**4)
XV_train = vectorizer.fit_transform(X_train)
XV_test = vectorizer.fit_transform(X_test)

#Logistic regression model
LR_model = LogisticRegression()
LR_model.fit(XV_train,Y_train)

#accuracy score of training data
X_train_prediction = LR_model.predict(XV_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

#accuracy score of test data
X_test_prediction_LR = LR_model.predict(XV_test)
test_data_accuracy_LR = accuracy_score(X_test_prediction_LR,Y_test)

show_result('Logistic Regression Model', test_data_accuracy_LR, Y_test, X_test_prediction_LR)
show_plot_confusion_matrix('Logistic Regression Model',Y_test,X_test_prediction_LR)

#Random Forest Classifier

RFC = RandomForestClassifier(random_state=0)
RFC.fit(XV_train,Y_train)
RandomForestClassifier(random_state=0)

X_train_prediction_RFC = RFC.predict(XV_train)
X_test_prediction_RFC = RFC.predict(XV_test)

training_data_accuracy_RFC = accuracy_score(X_train_prediction_RFC, Y_train)
test_data_accuracy_RFC = accuracy_score(X_test_prediction_RFC, Y_test)

show_result('Random Forest Classifier', test_data_accuracy_RFC, Y_test, X_test_prediction_RFC)
show_plot_confusion_matrix('Random Forest Classifier',Y_test,X_test_prediction_RFC)

#Naive Bayes Model
'''
from sklearn.naive_bayes import MultinomialNB

NB = MultinomialNB()
NB.fit(XV_train,Y_train)
X_test_prediction_NB = NB.predict(XV_test)

test_data_accuracy_NB = accuracy_score(X_test_prediction_NB,Y_test)

show_result('Naive Bayes Model', test_data_accuracy_NB, Y_test, X_test_prediction_NB)
show_plot_confusion_matrix('Naive Bayes Model',Y_test,X_test_prediction_NB)
'''
#decision tree classifier
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(XV_train,Y_train)

X_test_prediction_DT = DT.predict(XV_test)

test_data_accuracy_DT = accuracy_score(X_test_prediction_DT, Y_test)

show_result('Decision Tree Classifier', test_data_accuracy_DT, Y_test, X_test_prediction_DT)
show_plot_confusion_matrix('Decision Tree Classifier',Y_test,X_test_prediction_DT)

#predictive system    
    
print('According to Logistic Regression Model:\n ')
show_prediction(1, XV_test,LR_model)
print('According to Random Forest Classifier:\n ')
show_prediction(1, XV_test,RFC)

print('According to Decision Tree Classifier:\n ')
show_prediction(1, XV_test,DT)

#compare accuracy
accuracy_compare(test_data_accuracy_LR,test_data_accuracy_RFC,test_data_accuracy_DT)
