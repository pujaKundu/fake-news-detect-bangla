# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 20:36:27 2022

@author: USER
"""
import re
import string
from bangla_stemmer.stemmer.stemmer import BanglaStemmer

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')

stop_words = stopwords.words("bengali")
stemmer= BanglaStemmer()

#print(stop_words)
def stemming(content):
  content = content.lower()
  content = ' '.join([stemmer.stem(word) for word in content.split(' ') if word not in stop_words])
  content = re.sub(r'https*\S+', ' ', content)
  content = re.sub(r'@\S+', ' ', content)
  content = re.sub(r'#\S+', ' ', content)
  content = re.sub(r'\'\w+', '', content)
  content = re.sub('[%s]' % re.escape(string.punctuation), ' ', content)
  content = re.sub(r'\w*\d+\w*', '', content)
  content = re.sub(r'\s{2,}', ' ', content)
  content = re.sub('\n','',content)
  content = re.sub('\w*\d\w*','',content)
  return content

'''
content = re.sub('\[.*?\]','',content)
content = re.sub("\\W"," ",content)
content = re.sub('https?://\S+|www\.\S+','',content)
content = re.sub('<.*?>+','',content)
content = re.sub('[%s]' % re.escape(string.punctuation), '', content)
content = re.sub('\n','',content)
content = re.sub('\w*\d\w*','',content)
'''