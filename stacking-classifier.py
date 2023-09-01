# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 15:22:29 2023

@author: USER
"""
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score


from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

import warnings
warnings.filterwarnings("ignore")

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


#Linear Support Vector Machine

from sklearn.svm import SVC  
SVM = SVC(kernel='linear', random_state=0)  
SVM.fit(XV_train, Y_train)  

X_test_prediction_SVM = SVM.predict(XV_test)

test_data_accuracy_SVM = accuracy_score(X_test_prediction_SVM, Y_test)
test_data_f1_SVM = f1_score(X_test_prediction_SVM, Y_test)
test_data_precision_SVM= precision_score(X_test_prediction_SVM, Y_test)
test_data_recall_SVM = recall_score(X_test_prediction_SVM, Y_test)

show_result('Support Vector Machine', test_data_accuracy_SVM,test_data_f1_SVM,test_data_precision_SVM,test_data_recall_SVM, Y_test, X_test_prediction_SVM)
show_plot_confusion_matrix('Support Vector Machine',Y_test,X_test_prediction_SVM)

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
print('According to Support Vector Machine:\n ')
show_prediction(1, XV_test,SVM)


# Create a list of base models
models = [('LR', LR_model), ('RFC', RFC), ('NB', NB), ('DT', DT), ('GBC', GBC), ('PAC', PAC), ('SVM', SVM)]

# Create the Voting Classifier
voting_classifier = VotingClassifier(estimators=models, voting='hard')

# Fit the ensemble model on the training data
voting_classifier.fit(XV_train, Y_train)

# Predict on the test data
voting_predictions = voting_classifier.predict(XV_test)

# Evaluate the ensemble model's performance
ensemble_accuracy = accuracy_score(voting_predictions, Y_test)
ensemble_f1 = f1_score(voting_predictions, Y_test)
ensemble_precision = precision_score(voting_predictions, Y_test)
ensemble_recall = recall_score(voting_predictions, Y_test)

show_result('Ensemble Voting Classifier', ensemble_accuracy, ensemble_f1, ensemble_precision, ensemble_recall, Y_test, voting_predictions)
show_plot_confusion_matrix('Ensemble Voting Classifier', Y_test, voting_predictions)


#stacking classifier
# Create a list of base models
base_models = [('LR', LR_model), ('RFC', RFC), ('NB', NB), ('DT', DT), ('GBC', GBC), ('PAC', PAC), ('SVM', SVM)]

# Create a stacking classifier with a meta-model (you can choose any of your existing models as the meta-model)
stacking_classifier = StackingClassifier(estimators=base_models, final_estimator=RandomForestClassifier(random_state=0), cv=5)

# Fit the stacking model on the training data
stacking_classifier.fit(XV_train, Y_train)

# Predict on the test data
stacking_predictions = stacking_classifier.predict(XV_test)

# Evaluate the stacking model's performance
stacking_accuracy = accuracy_score(stacking_predictions, Y_test)
stacking_f1 = f1_score(stacking_predictions, Y_test)
stacking_precision = precision_score(stacking_predictions, Y_test)
stacking_recall = recall_score(stacking_predictions, Y_test)

show_result('Stacking Classifier', stacking_accuracy, stacking_f1, stacking_precision, stacking_recall, Y_test, stacking_predictions)
show_plot_confusion_matrix('Stacking Classifier', Y_test, stacking_predictions)

#bagging classifier

# Bagging classifier
bagging_classifier = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=0)

# Fit the bagging model on the training data
bagging_classifier.fit(XV_train, Y_train)

# Predict on the test data
bagging_predictions = bagging_classifier.predict(XV_test)

# Evaluate the bagging model's performance
bagging_accuracy = accuracy_score(bagging_predictions, Y_test)
bagging_f1 = f1_score(bagging_predictions, Y_test)
bagging_precision = precision_score(bagging_predictions, Y_test)
bagging_recall = recall_score(bagging_predictions, Y_test)

show_result('Bagging Classifier', bagging_accuracy, bagging_f1, bagging_precision, bagging_recall, Y_test, bagging_predictions)
show_plot_confusion_matrix('Bagging Classifier', Y_test, bagging_predictions)

# AdaBoost classifier
adaboost_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=50, random_state=0)

# Fit the AdaBoost model on the training data
adaboost_classifier.fit(XV_train, Y_train)

# Predict on the test data
adaboost_predictions = adaboost_classifier.predict(XV_test)

# Evaluate the AdaBoost model's performance
adaboost_accuracy = accuracy_score(adaboost_predictions, Y_test)
adaboost_f1 = f1_score(adaboost_predictions, Y_test)
adaboost_precision = precision_score(adaboost_predictions, Y_test)
adaboost_recall = recall_score(adaboost_predictions, Y_test)

show_result('AdaBoost Classifier', adaboost_accuracy, adaboost_f1, adaboost_precision, adaboost_recall, Y_test, adaboost_predictions)
show_plot_confusion_matrix('AdaBoost Classifier', Y_test, adaboost_predictions)

#accuracy graph
accuracy_compare(test_data_accuracy_LR,test_data_accuracy_RFC,test_data_accuracy_NB,test_data_accuracy_DT,test_data_accuracy_GBC,test_data_accuracy_PAC,test_data_accuracy_SVM)
def calculate_roc_auc(model, X, y):
    y_scores = model.decision_function(X)  # Raw decision scores
    y_binary = np.where(y == model.classes_[1], 1, 0)  # Binarize labels
    return roc_auc_score(y_binary, y_scores)

roc_auc_LR = calculate_roc_auc(LR_model, XV_test, Y_test)
roc_auc_RFC = calculate_roc_auc(RFC, XV_test, Y_test)
roc_auc_NB = calculate_roc_auc(NB, XV_test, Y_test)
roc_auc_DTC = calculate_roc_auc(DT, XV_test, Y_test)
y_probs_pac = PAC.decision_function(XV_test)
roc_auc_PAC = calculate_roc_auc(PAC, XV_test, Y_test)
roc_auc_GBC = calculate_roc_auc(GBC, XV_test, Y_test)
roc_auc_SVM = calculate_roc_auc(SVM, XV_test, Y_test)


# Print ROC-AUC scores
print("ROC-AUC Scores:")
print("Logistic Regression:", roc_auc_LR)
print("Random Forest:", roc_auc_RFC)
print("Naive Bayes:",roc_auc_NB)
print("Decision Tree:", roc_auc_DTC)
print("Passive Aggressive:", roc_auc_PAC)
print("Gradient Boosting:", roc_auc_GBC)
print("SVM:",roc_auc_SVM)



