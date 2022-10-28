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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,f1_score
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

real_news = pd.read_csv('F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Dataset/Authentic-48K.csv',nrows=3000)
fake_news = pd.read_csv('F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Dataset/Fake-1K.csv')
#fake_news = pd.read_csv('F:\CSE academic\Fake_news_detection\fake_daa/new_fake_data.csv')
new_fake_news = pd.read_csv('F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Dataset/Fake-data-313.csv')
new_fake_news2 = pd.read_csv('F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Dataset/Fake-data-375.csv')
#concat two csv files

news_dataset = pd.concat([real_news,fake_news,new_fake_news,new_fake_news2])
news_dataset = shuffle(news_dataset)
news_dataset.reset_index(inplace=True, drop=True)

#print(news_dataset.shape)

#print first five rows of the dataframe
#news_dataset.head()

#counting the number of missing values in the dataset
news_dataset.isnull().sum()

#replacing the null values with empty string
news_dataset = news_dataset.fillna('')

#merging the news headline and title
news_dataset['content_data'] =news_dataset['headline']+' '+news_dataset['content']

#print(news_dataset['content_data'])

#separating the data and label

X=news_dataset.drop(columns='label',axis=1)
Y=news_dataset['label']

"""Stemming:
"""
news_dataset['content_data'] = news_dataset['content_data'].apply(stemming)

#fake_news['headline'] = fake_news['headline'].apply(stemming)

#print('After applying stemming')
#print(fake_news['headline'])

#Separating data and label

X=news_dataset['content_data']
Y=news_dataset['label']

v=Y.value_counts()

print(v)
#Y.shape
#training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3,random_state=10)


#TfIDF Vectorizer Bi-gram
vectorizer = TfidfVectorizer(ngram_range=(2,2))
XV_train = vectorizer.fit_transform(X_train)
XV_test = vectorizer.transform(X_test)

#Logistic regression model
LR_model = LogisticRegression()
LR_model.fit(XV_train,Y_train)

#accuracy score of test data
X_test_prediction_LR = LR_model.predict(XV_test)

test_data_accuracy_LR = accuracy_score(X_test_prediction_LR,Y_test)
test_data_f1_LR = f1_score(X_test_prediction_LR,Y_test)
test_data_precsion_LR = precision_score(X_test_prediction_LR,Y_test)
test_data_recall_LR = recall_score(X_test_prediction_LR,Y_test)

show_result('Logistic Regression Model', test_data_accuracy_LR,test_data_f1_LR,test_data_precsion_LR,test_data_recall_LR, Y_test, X_test_prediction_LR)
show_plot_confusion_matrix('Logistic Regression Model',Y_test,X_test_prediction_LR)

#Random Forest Classifier

RFC = RandomForestClassifier(random_state=0)
RFC.fit(XV_train,Y_train)
RandomForestClassifier(random_state=0)

X_test_prediction_RFC = RFC.predict(XV_test)

test_data_accuracy_RFC = accuracy_score(X_test_prediction_RFC, Y_test)
test_data_f1_RFC = f1_score(X_test_prediction_RFC, Y_test)
test_data_precision_RFC = precision_score(X_test_prediction_RFC, Y_test)
test_data_recall_RFC = recall_score(X_test_prediction_RFC, Y_test)

show_result('Random Forest Classifier', test_data_accuracy_RFC,test_data_f1_RFC, test_data_precision_RFC,test_data_recall_RFC,Y_test, X_test_prediction_RFC)
show_plot_confusion_matrix('Random Forest Classifier',Y_test,X_test_prediction_RFC)

#Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB

NB = MultinomialNB()
NB.fit(XV_train,Y_train)
X_test_prediction_NB = NB.predict(XV_test)

test_data_accuracy_NB = accuracy_score(X_test_prediction_NB,Y_test)
test_data_f1_NB = f1_score(X_test_prediction_NB,Y_test)
test_data_precision_NB = precision_score(X_test_prediction_NB,Y_test)
test_data_recall_NB = recall_score(X_test_prediction_NB,Y_test)

show_result('Naive Bayes Model', test_data_accuracy_NB,test_data_f1_NB,test_data_precision_NB,test_data_recall_NB, Y_test, X_test_prediction_NB)
show_plot_confusion_matrix('Naive Bayes Model',Y_test,X_test_prediction_NB)

#decision tree classifier
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(XV_train,Y_train)

X_test_prediction_DT = DT.predict(XV_test)

test_data_accuracy_DT = accuracy_score(X_test_prediction_DT, Y_test)
test_data_f1_DT = f1_score(X_test_prediction_DT, Y_test)
test_data_precision_DT = precision_score(X_test_prediction_DT, Y_test)
test_data_recall_DT = recall_score(X_test_prediction_DT, Y_test)

show_result('Decision Tree Classifier', test_data_accuracy_DT,test_data_f1_DT,test_data_precision_DT,test_data_recall_DT, Y_test, X_test_prediction_DT)
show_plot_confusion_matrix('Decision Tree Classifier',Y_test,X_test_prediction_DT)


#Gradient Boosting Algorithm
from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(XV_train,Y_train)

GradientBoostingClassifier(random_state=0)

X_test_prediction_GBC = GBC.predict(XV_test)

test_data_accuracy_GBC = accuracy_score(X_test_prediction_GBC, Y_test)
test_data_f1_GBC = f1_score(X_test_prediction_GBC, Y_test)
test_data_precision_GBC = precision_score(X_test_prediction_GBC, Y_test)
test_data_recall_GBC = recall_score(X_test_prediction_GBC, Y_test)

show_result('Gradient Boosting Classifier', test_data_accuracy_GBC,test_data_f1_GBC,test_data_precision_GBC,test_data_recall_GBC, Y_test, X_test_prediction_GBC)
show_plot_confusion_matrix('Gradient Boosting Classifier',Y_test,X_test_prediction_DT)

#Passive Aggresive Classification
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.datasets import make_classification


PAC = PassiveAggressiveClassifier(max_iter=1000, random_state=0,tol=1e-3)
PAC.fit(XV_train, Y_train)
PassiveAggressiveClassifier(random_state=0)

X_test_prediction_PAC = PAC.predict(XV_test)

test_data_accuracy_PAC = accuracy_score(X_test_prediction_PAC, Y_test)
test_data_f1_PAC = f1_score(X_test_prediction_PAC, Y_test)
test_data_precision_PAC= precision_score(X_test_prediction_PAC, Y_test)
test_data_recall_PAC = recall_score(X_test_prediction_PAC, Y_test)

show_result('Passive Aggressive Classifier', test_data_accuracy_PAC,test_data_f1_PAC,test_data_precision_PAC,test_data_recall_PAC, Y_test, X_test_prediction_PAC)
show_plot_confusion_matrix('Passive Aggressive Classifier',Y_test,X_test_prediction_PAC)

#predictive system    
    
print('According to Logistic Regression Model:\n ')
show_prediction(1, XV_test,LR_model)
print('According to Random Forest Classifier:\n ')
show_prediction(1, XV_test,RFC)
print('According to Naive Bayes:\n ')
show_prediction(1, XV_test,NB)
print('According to Decision Tree Classifier:\n ')
show_prediction(1, XV_test,DT)
print('According to Gradient Boosting Classifier:\n ')
show_prediction(1, XV_test,GBC)
print('According to Passive Aggressive Classifier:\n ')
show_prediction(1, XV_test,PAC)
#compare accuracy
accuracy_compare(test_data_accuracy_LR,test_data_accuracy_RFC,test_data_accuracy_NB,test_data_accuracy_DT,test_data_accuracy_GBC,test_data_accuracy_PAC)
