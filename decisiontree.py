# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 17:21:04 2016

@author: w
"""

from math import log
import operator
#香农，信息熵--数据集的无序程度 创建字典，求对数
def calcShannonEnt(dataSet):
    #计算数据集中实例总数，创建数据字典，键值为最后一列数值，键值记录当前类别淑贤次数，所有类标签的发生频率计算类别出现的概率
    #通过概率计算熵值
    labelCounts={}
    numEntries = len(dataset)
    for featVec in dataSet:
        currentLabel = featVec[-1]#最后一列 dataSet 的格式是[['','',''],['','',''],['','','']]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] =0
            labelCounts[currentLabel]+=1
            
        shannonEnt =0.0
        for key in labelCounts:
            prob = float(labelCounts[key])/numEntries
            shannonEnt -= prob*log(prob,2) #香农公式
        return shannonEnt
        
#通过最大信息增益的方法划分数据集，从一个数据集中随机选取子项，度量其被错误分类到其他分组的概率
#基尼不纯度方法

def splitDataSet(dataSet, axis, value):#划分的数据集，划分数据集的特征，特征的返回值
    retDataSet=[]
    for featVec in dataSet:
        if featVec [axis] == value:  #符合要求
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#python 不考虑内存分配问题 。传递的是对列表的引用。
#声明新的列表对象 避免对列表的修改影响整个生产周期  遍历数据集中元素，将符合要求的值添加到新创建列表中
#选择最好的数据集划分方式--最好的特征--熵增益最大
def chooseBestFeatureToSplit(dataSet):#数据必须是由相同的数据长度列表元素组成的列表 且最后一列为标签 
#先计算最初的无序值，用于与划分后的比较
    numFeature = len(dataSet[0])-1 #除去label的列数
    baseEntropy = calcShannonEnt(dataSet) 
    bestInfoGain = 0.0
    bestFeature =-1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] #遍历每个元素featList每次取一个值--dataSet的列遍历
        uniqueVals = set(featList) #将值放入集合中
        newEntropy =0.0
        for value in uniqueVals: #集合中的元素
            subDataSet =splitDataSet(dataSet,i,value)
            prob= len(subDataSet)/float(len(dataSet)) #对每一个值进行prob计算
            newEntropy+=prob *calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain >bestInfoGain):
            bestInfoGain = infoGain
            bestFeature =i
    return bestFeature
#example 将第0个特征为1的subDataSet一组，为0的另一组。数据一致性如何？
    

#递归构建决策树
#递归 如果数据集已经处理了所有属性，但是类标签依然不是唯一的。此时需要决定如何定义叶子节点--多数表决的方法觉得该叶子节点分类。

def majorityCnt(classList):
    classCount={} #字典，存储每个标签出现的频率 返回出现次数最多的名称分类
    for vote in classList: #对每个标签进行操作 
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1 #如果存在则对每一个类标签出现次数进行统计
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]
#sorted list或者iterable，key和reverse true为降序 排序时进行比较的函数，可指定一个函数或者lambda函数 key= lambda classCount: classCount[2] 此处为list
    
#创建树的函数代码
def createTree(dataSet,labels):#数据集和标签列表 classList包含数据集的所有类标签
#类别完全相同则停止继续划分
#遍历所有特征时返回出现次数最多的
#得到列表包含所有属性值
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): #所有类标签完全相同则返回该类标签
        return classList[0]
    if len(dataSet[0]) == 1: #用完了所有特征，但是数据集无法划分成唯一类别分组，返回多数
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat] #labels包含的是所有特征的标签
 #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++构建树的代码--递归 
   myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals =set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
 #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return myTree
#递归
    


        
        
    






















    

# extend append
# [1,2,3,4,5,6] [1,2,3,[4,5,6]]
m= [1,2,3,4,5,6]
print (m[2:],m[:2])

a = set('boy')
a.add('python')
a.update('python')
a.remove('python')

x= {'title':'python web site','url':'www.i.com'}
a = x.items()
type(x)
type(a)
b=[('title', 'python web site'), ('url', 'www.i.com')]
type(b)# list
c = x.iteritems() #字典项的迭代器

 s0 = set()
 d0 = {}
 s1 = {0}
 s2 = {i % 2 for i in range(10)}
 s = set('hi')
 t = set(['h', 'e', 'l', 'l', 'o'])
 print(s0, s1, s2, s, t, type(d0))

 print(s.intersection(t), s & t)  # 交集  
 print(s.union(t), s | t)   # 并集 
 print(s.difference(t), s - t)  # 差集 
 print(s.symmetric_difference(t), s ^ t) # 对称差集 
 print(s1.issubset(s2), s1 <= s2) # 子集
 print(s1.issuperset(s2), s1 >= s2)      # 包含

 s = {0}
 print(s, len(s))   # 获取集合中的元素的总数
 s.add("x")         # 添加一个元素
 print(s)
 s.update([1,2,3])  # 添加多个元素
 print(s, "x" in s) # 成员资格测试
 s.remove("x")      # 去掉一个元素
 print(s, "x" not in s)  
 s.discard("x")     # 如果集合存在指定元素，则删除该元素
 c = s.copy()       # 复制集合     
 print(s, s.pop())  # 弹出集合中的一个不确定元素，如果原集合为空则引发 KeyError
 s.clear()          # 删除集合中的元素
 print(s, c)



















