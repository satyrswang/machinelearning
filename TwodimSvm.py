#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 19:21:18 2017

@author: wyq
"""
import numpy as np
from matplotlib import pyplot as plt



X = np.array([
[-2,4,-1],
[4,1,-1],
[1, 6, -1],
[2, 4, -1],
[6, 2, -1],])

y = np.array([-1,-1,1,1,1])

enumofx = enumerate(X)


print enumofx

for d, sample in enumofx:
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

plt.plot([-2,6],[6,0.5])



def svm_sgd_plot(X, Y):
    
    w = np.zeros(len(X[0]))

    eta = 1

    epochs = 100000
   
    errors = []

  
    for epoch in range(1,epochs):
        print epoch
        error = 0
        for i, x in enumerate(X):
         
            if (Y[i]*np.dot(X[i], w)) < 1:
               
                w = w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
                error = 1
            else:
              
                w = w + eta * (-2  *(1/epoch)* w)
        errors.append(error)
        

    
    plt.plot(errors, '|')
    plt.ylim(0.5,1.5)
    plt.axes().set_yticklabels([])
    
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()
    
    return w

w = svm_sgd_plot(X,y)

