# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 20:36:27 2022

@author: USER
"""
import re
import string

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