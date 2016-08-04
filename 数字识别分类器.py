# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:59:40 2016

@author: w
"""

#lable,listdir 获取目录内容长度
#矩阵，对目录中每个文件
#从文件名解析分类数字
    
import numpy

def img2vector():
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j  in  range(32):
            returnVect[0,32*i+j] = int(lineStr[j])#文件中01的存储连续
    return returnVect
    
    
def handwritingClassTest():
    hwLabels =[]
    trainingFileList = listdir ('trainingDigits') #目录trainingDigits下文件名列表
    m = len(trainingFileList) #目录下文件数目
    trainingMat = zeros((m,1024)) #m行1024列
    #print(trainingMat)
    for i in range(m):#对于每个文件:
        fileNameStr = trainingFileList[i] #从文件名中记录数字
        fileStr = fileNameStr.split('.')[0] #文件名格式 1_1.abcd
        classNumStr = int(fileStr.split('_')[0]) 
        hwLables.append(classNumStr)
        trainingMat[i,:] =img2vector('trainingDigits/%s' % fileNameStr)#取第i行所有列 
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i] #从文件名中记录数字
        fileStr = fileNameStr.split('.')[0] #文件名格式 1_1.abcd
        classNumStr = int(fileStr.split('_')[0]) 
        vectorUnderTest = img2vector('testDigits/$s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat,hwLabels,3)#参数： 测试集，训练集，label和最近邻居数目为3
        print('the classfier came back with:%d, the real answer is :%d' %(classifierResult, classNumStr))
        if(classifierResult!=classNumStr):
            errorCount +=1.0
    print ('\n the total num of errors is %d' % errorCount)



#==============================================================================
# import os    
# os.listdir(os.getcwd())
# 
# m=5
# s=0
# for i in m:
#     s=s+m;
#     print(s)
# m=4
# s = 'this is a line'
# print(s[2])#2是01234....
# 
# print('whatis s %s %d ' % (s,m))
#==============================================================================
