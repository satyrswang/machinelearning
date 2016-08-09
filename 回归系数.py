# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:30:43 2016

@author: w
"""

def loadDataSet(): ##从dataset里分出data和label
    dataMat =[]
    labelMat = []
    fr = open('test.txt')
    for line in fr.readline():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0],float(lineArr[1])]) #[1.0,  ,  ]
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):##sigmoid函数
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn , classLabels):
    dataMatrix =mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles =500
    weights =ones((n,1))
#——————————————————————————————————————————————————梯度下降公式
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha*dataMatrix.transpose()*error
#————————————————————————————————————————————————————————————
    return weights

#input做成mat格式 label同样mat格式 行列m n  dataMatIn是二维Numpy数组，每列分别代表不同特征

