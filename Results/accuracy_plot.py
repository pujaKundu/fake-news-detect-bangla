# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 18:44:19 2022

@author: Puja Kundu
"""
import matplotlib.pyplot as plt
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])


def accuracy_compare(acc1,acc2,acc3,acc4):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    models = ['LR', 'RFC','MNB','DT']
    accuracy = [acc1,acc2,acc3,acc4]
    ax.bar(models,accuracy, color=['blue','green','red','yellow'])
    addlabels(models,accuracy)
    plt.show()
