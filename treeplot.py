# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 12:31:58 2016

@author: w
"""

import matplotlib.pyplot as plt

#使用文本注释绘制树节点
#定义文本框和箭头格式 绘制带箭头的注解
decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
leafNode = dict(boxstyle = "round4", fc ="0.8")
arrow_args = dict(arrowstyle = "<-")

def plotNode (nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt,xy = parentPt, xycoords ='axes fraction',xytext = centerPt, textcoords ='axes fraction',va ="center", ha="center",bbox=nodeType, arrowprops = arrow_args)

def createPlot():
    fig = plt.figure(1,facecolor = 'white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon = False)
    plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
    
#第一个版本的createPlot()  全局变量ax1定义绘图区
#必须知道多少个叶节点，确定x轴长度；树有多少层，确定y轴高度。定义两个新函数getNumLeafs()和getTreeDepth()
   #如何利用python字典来存储树。 
def getNumLeafs(myTree):
    numLeafs =0
    firstStr = myTree.keys()[0] #keys()列表的第一个 根
    secondDict= myTree[firstStr]#第一个value-- 第二层
    for key in seecondDict.keys():
        if type(secondDict[key])._name_=='dict': #是否还是字典
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs

def getTreeDepth(mytree):
    maxDepth = 0
    firstStr = mytree.keys()[0]
    secondDict = mytree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key])._name_ =='dict':
            thisDepth = 1+getTreeDepth(secondDict[key]) #同上对每个value判断是否还是字典
        else:
            thisDepth =1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees =[{'no surfacing':{0:'no',1:{'flippers':{0:'no' , 1:'yes'}}}},{'no surfacing':{0:'no',1:{'flippers':{0：{'head': {0:'no' , 1:'yes'}},1:'no'}}}}]
    return listOfTrees[i]
    #用于测试，返回预定义的树结构

def plotMidText (cntrPt,parentPt,txtString):#在父子节点间填充文本信息
    xMid = （parentPt[0]-cntrPt[0]）/2.0 +cntrPt[0]
    yMid = （parentPt[1]-cntrPt[1]）/2.0 +cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)

def plotTree(myTree,parentPt,nodeTxt):
    numLeafs =getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff +(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff) #
    plotMidText
    plotNode
    secondDict
    plotTree.yOff
    for key in secondDict.keys():
        if type(secondDict[key])._name_=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            ……
    
def createPlot(inTree):
    fig =plt.figure(1,facecolor ='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon = False,**axprops)
    plotTree.totalW  #树的宽度
    plotTree.totalD  #树的深度
    ……
    plt.show()



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#测试
def classify(inputTree,featLabels,testVec):
    fistStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(fistStr) #根key的value在secondDict里，将标签字符串转换为index索引
    for key in secondDict.keys():
        if testVec[featIndex] ==key:
            if type(secondDict[key])._name_ =='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel
#决策树分类函数
#使用index方法查找当前列表中第一个匹配firstStr变量的元素。代码递归遍历树，比较testVec变量的值与树节点的值 到达叶子节点则返回当前节点的分类标签
    

#构造分类器很耗时。每次执行分类时调用已经构造好的决策树。实用python模块pickle序列化对象。序列化对象可以在磁盘上保存对象，在需要的时候读取。
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

