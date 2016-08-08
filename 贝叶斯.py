# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 11:26:44 2016

@author: w
"""

def createVocabuList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec (vocabList, inputSet): #输入词汇表和文档，返回文档响亮
    returnVec = [0]*len(vocabList) #长度为词汇表全为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] =1
        else:
            print("the word %s is not in my vocabulary" % word)
    return returnVec
    
def trainNB0(trainMatrix,trainCategory):#输入文档矩阵，每篇文档类别标签构成的向量trainCategory
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) /float(numTrainDocs)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p1Denom =0.0
    p1Denom =0.0
    for i in range(numTrainDocs):
        if trainCategory[i] ==1 :
            p1Num+=trainMatrix[i]
            p1Denom +=sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect = p1Num /p1Denom
    p0Vect = p0Num /p0Denom
    return p0Vect,p1Vect,pAbusive
    
    
    