# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 07:35:40 2022

@author: USER
"""
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

def show_plot_confusion_matrix(model,x_data,y_data):
    print('\n')
    print('Confusion matrix for ',model)
    print('\n')
    CR = confusion_matrix(y_data,x_data )
    print(CR)
    print('\n')
    fig,ax = plot_confusion_matrix(conf_mat=CR,figsize=(10,10),show_absolute=True,show_normed=True,colorbar=True)
    plt.show()
    
def show_result(model,accuracy,f1,precision,recall,y_data,x_pred):
    print('Accuracy for ',model)
    print('\n')
    accuracy_percentage = "{:.4f}".format(accuracy * 100)
    f1_percentage = "{:.4f}".format(f1 * 100)
    precision_percentage = "{:.4f}".format(precision * 100)
    recall_percentage = "{:.4f}".format(recall * 100)
    print('Accuracy score of the test data: ',accuracy_percentage,'%')
    print('\n')
    print('F1 score of the test data: ',f1_percentage,'%')
    print('\n')
    print('Precision score of the test data: ',precision_percentage,'%')
    print('\n')
    print('Recall score of the test data: ',recall_percentage,'%')
    print('\n')
    print('Classification report for ',model)
    print('\n')
    print(classification_report(y_data,x_pred ))
    print('\n')