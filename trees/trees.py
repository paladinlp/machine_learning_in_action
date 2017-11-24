from math import log
# -*- coding: UTF-8 -*-

def creatDataSet():
    dataSet=[
        ['1', '1', 'yes'],
        ['1', '1', 'yes'],
        ['1', '0', 'no'],
        ['0', '1', 'no'],
        ['0', '1', 'no'],
    ]
    labels = ['no surfacing','flippers']
    return dataSet
def cal_ShannonEnt(dataSet):   #计算香农熵
    labelCounts = {}
    item_nums = len(dataSet)
    for item in dataSet:
        itemName = item[-1]
        if itemName not in labelCounts.keys():
            labelCounts[itemName] = 0
        labelCounts[itemName] +=1
    shannonEnt = 0.0
    for item in labelCounts:
        prob = labelCounts[item]/item_nums*1.0
        shannonEnt -= prob*log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet,axis,value):
    retDateSet = []
    for item in dataSet:
        if item[axis] == value:
            reduceFeatVec = item[:axis]
            reduceFeatVec.extend(item[axis+1:])
            retDateSet.append(reduceFeatVec)
    return retDateSet


def chooseBestFeatureToSplit(dataSet):
    #设定一列的初始条件
    numFeatures = len(dataSet[0])-1
    baseEntropy = cal_ShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1 # 这种写法其实很好，设定一个最小值来写，先设定一个基础值，满足条件就去替换
    for i in range(numFeatures):
        valueList = set([item[i] for item in dataSet])
        subShannonEntropy = 0.0
        for value in valueList:
            subDateSet = splitDataSet(dataSet, i, value)
            subShannonEntropy += len(subDateSet) / float(len(dataSet)) * cal_ShannonEnt(subDateSet)
        if (baseEntropy - subShannonEntropy)> bestInfoGain:
            bestInfoGain = baseEntropy - subShannonEntropy
            bestFeature = i
    return bestFeature









if __name__ == '__main__' :
    print(chooseBestFeatureToSplit(creatDataSet()))
