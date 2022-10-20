# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 14:19:27 2022

@author: USER
"""

import pandas as pd
import tensorflow as tf
import os
import re
import string
import numpy as np
from string import punctuation
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,f1_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# importing neural network libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, LSTM, RNN, SpatialDropout1D


#loading the dataset to pandas dataframe

real_news = pd.read_csv('F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Dataset/Authentic-48K.csv',nrows=1300)
fake_news = pd.read_csv('F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Dataset/Fake-1K.csv')

#concat two csv files

news_dataset = pd.concat([real_news,fake_news])
news_dataset = shuffle(news_dataset)
news_dataset.reset_index(inplace=True, drop=True)

#print(news_dataset.shape)

#print first five rows of the dataframe
news_dataset.head()

#counting the number of missing values in the dataset
news_dataset.isnull().sum()

#replacing the null values with empty string
news_dataset = news_dataset.fillna('')

#merging the news headline and title
news_dataset['content_data'] = news_dataset['headline']+' '+news_dataset['content']

#print(news_dataset['content_data'])

#separating the data and label

X=news_dataset.drop(columns='label',axis=1)
Y=news_dataset['label']

def stemming(content):
  content = content.lower()
  content = re.sub('\[.*?\]','',content)
  content = re.sub("\\W"," ",content)
  content = re.sub('https?://\S+|www\.\S+','',content)
  content = re.sub('<.*?>+','',content)
  content = re.sub('[%s]' % re.escape(string.punctuation), '', content)
  content = re.sub('\n','',content)
  content = re.sub('\w*\d\w*','',content)
  return content

news_dataset['content_data'] = news_dataset['content_data'].apply(stemming)

X=news_dataset['content_data']
Y=news_dataset['label']

#print(X)
#print(Y)

Y.shape
#training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=18)

length = []
[length.append(len(str(text))) for text in news_dataset['content_data']]
news_dataset['length'] = length
news_dataset.head()

min(news_dataset['length']), max(news_dataset['length']), round(sum(news_dataset['length'])/len(news_dataset['length']))

len(news_dataset[news_dataset['length'] < 50])

news_dataset['content_data'][news_dataset['length'] < 50]

#remove outliers if any.Mostly empty texts. They can be removed since they will surely guide the neural network in the wrong way

news_dataset = news_dataset.drop(news_dataset['content_data'][news_dataset['length'] < 50].index, axis = 0)

min(news_dataset['length']), max(news_dataset['length']), round(sum(news_dataset['length'])/len(news_dataset['length']))

max_features = 4500

# Tokenizing the text - converting the words, letters into counts or numbers. 
# We dont need to explicitly remove the punctuations. we have an inbuilt option in Tokenizer for this purpose
tokenizer = Tokenizer(num_words = max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
tokenizer.fit_on_texts(texts = news_dataset['content_data'])

X = tokenizer.texts_to_sequences(texts = news_dataset['content_data'])

# now applying padding to make them even shaped.
X = pad_sequences(sequences = X, maxlen = max_features, padding = 'pre')

#print(X.shape)
y = news_dataset['label'].values
#print(y.shape)

# splitting the data training data for training and validation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

# LSTM Neural Network
lstm_model = Sequential(name = 'lstm_nn_model')
lstm_model.add(layer = Embedding(input_dim = max_features, output_dim = 120, name = '1st_layer'))
lstm_model.add(layer = LSTM(units = 120, dropout = 0.2, recurrent_dropout = 0.2, name = '2nd_layer'))
lstm_model.add(layer = Dropout(rate = 0.5, name = '3rd_layer'))
lstm_model.add(layer = Dense(units = 120,  activation = 'relu', name = '4th_layer'))
lstm_model.add(layer = Dropout(rate = 0.5, name = '5th_layer'))
lstm_model.add(layer = Dense(units = len(set(y)),  activation = 'sigmoid', name = 'output_layer'))
# compiling the model
lstm_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#lstm_model.summary()


X_test = news_dataset.copy()
print(X_test.shape)


X_test = news_dataset.fillna(' ')
print(X_test.shape)
X_test.isnull().sum()

lstm_prediction = lstm_model.predict(X_test)

print(lstm_prediction)

accuracy = round((accuracy_score( Y_test,lstm_prediction)*100),2)
precision = round((precision_score( Y_test,lstm_prediction)*100),2)
recall = round((recall_score( Y_test,lstm_prediction)*100),2)
f1score = round((f1_score( Y_test,lstm_prediction)*100),2)
print('Accuracy on testing set:', accuracy,'%')
print('Precision on testing set:', precision,'%')
print('Recall on testing set:', recall,'%')
print('F1-score on testing set:', f1score,'%')


#print('gru pred',gru_prediction)



