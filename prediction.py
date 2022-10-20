# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:32:59 2022

@author: Puja Kundu
"""

def show_prediction(idx,data,model):
    X_new = data[idx]
    
    prediction = model.predict(X_new)
    if (prediction[0]==0):
      print('The news is Fake\n')
    else:
      print('The news is Real\n')