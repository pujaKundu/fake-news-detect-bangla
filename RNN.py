# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 10:54:44 2022

@author: USER
"""

import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,f1_score
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns 

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import os
from string import punctuation


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, LSTM, RNN, SpatialDropout1D

import sys
sys.path.insert(0, 'F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Results')
from accuracy_plot import accuracy_compare
from show_results import show_result
from show_results import show_plot_confusion_matrix
from prediction import show_prediction


#loading the dataset to pandas dataframe
real_news = pd.read_csv('F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Dataset/Authentic-48K.csv',nrows=2000)
fake_news = pd.read_csv('F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Dataset/Fake-1K.csv')
#fake_news = pd.read_csv('F:\CSE academic\Fake_news_detection\fake_daa/new_fake_data.csv')
new_fake_news = pd.read_csv('F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Dataset/fake_collection.csv')
new_fake_news2 = pd.read_csv('F:\CSE academic\CSE 4-2\project\Bangla_fake_news_detection\Dataset/Fake-Data-m.csv')
#concat two csv files

news_dataset = pd.concat([real_news,fake_news,new_fake_news,new_fake_news2])

news_dataset = shuffle(news_dataset)
news_dataset.reset_index(inplace=True, drop=True)


plt.figure(figsize=(10, 5))
plt.bar('Fake News', len(fake_news), color='orange')
plt.bar('Real News', len(real_news), color='green')
plt.title('Distribution of Fake News and Real News', size=15)
plt.xlabel('News Type', size=15)
plt.ylabel('# of News Articles', size=15)


total_len = len(fake_news) + len(real_news)
plt.figure(figsize=(10, 5))
plt.bar('Fake News', len(fake_news) / total_len, color='orange')
plt.bar('Real News', len(real_news) / total_len, color='green')
plt.title('Distribution of Fake News and Real News', size=15)
plt.xlabel('News Type', size=15)
plt.ylabel('Proportion of News Articles', size=15)


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

X=news_dataset['content_data']
Y=news_dataset['label']

v=Y.value_counts()

#training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=18)

max_vocab = 10000
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(X_train)

# tokenize the text into vectors 
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', maxlen=256)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', maxlen=256)

#Building RNN

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_vocab, 128),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])



model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=10,validation_split=0.1, batch_size=30, shuffle=True, callbacks=[early_stop])

history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = history.epoch

plt.figure(figsize=(12,9))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss', size=20)
plt.xlabel('Epochs', size=20)
plt.ylabel('Loss', size=20)
plt.legend(prop={'size': 20})
plt.show()

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy', size=20)
plt.xlabel('Epochs', size=20)
plt.ylabel('Accuracy', size=20)
plt.legend(prop={'size': 20})
plt.ylim((0.5,1))
plt.show()

pred = model.predict(X_test)

binary_predictions = []

for i in pred:
    if i >= 0.5:
        binary_predictions.append(1)
    else:
        binary_predictions.append(0) 

model.evaluate(X_test, Y_test)

accuracy = round((accuracy_score( Y_test,binary_predictions)*100),2)
precision = round((precision_score( Y_test,binary_predictions)*100),2)
recall = round((recall_score( Y_test,binary_predictions)*100),2)
f1score = round((f1_score( Y_test,binary_predictions)*100),2)
print('Accuracy on testing set:', accuracy,'%')
print('Precision on testing set:', precision,'%')
print('Recall on testing set:', recall,'%')
print('F1-score on testing set:', f1score,'%')

matrix = confusion_matrix( Y_test,binary_predictions, normalize='all')
print('Confusion Matrix\ntn,fp,fn,tp\n',matrix)
plt.figure(figsize=(16, 10))
ax= plt.subplot()
sns.heatmap(matrix, annot=True, ax = ax)

# labels, title and ticks
ax.set_xlabel('Predicted Labels', size=20)
ax.set_ylabel('True Labels', size=20)
ax.set_title('Confusion Matrix', size=20) 
ax.xaxis.set_ticklabels([0,1], size=15)
ax.yaxis.set_ticklabels([0,1], size=15)

