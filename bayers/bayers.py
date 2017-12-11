# -*-coding:utf-8-*-
__author__ = 'paladinlp'
__date__ = '2017/12/10 14:33'


from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWord2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word :%s is not in my Vocabulary!' %  word)
    return returnVec





def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = zeros(numWords); p1Num = zeros(numWords)
    p0Denom = 0.0; p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]  # 最后的结果是每一个词条的计数值
            p1Denom += sum(trainMatrix[i])  #所有词条的计数值,除的是p(w/ci)
        else:
            p0Num += trainMatrix[i]  # 最后的结果是每一个词条的计数值
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive


if __name__ == '__main__':
    listOPost, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPost)
    trainMat = []
    for postinDoc in listOPost:
        trainMat.append(setOfWord2Vec(myVocabList,postinDoc))
    p0v,p1v,pAb = trainNB0(trainMat,listClasses)
    print(p0v,p1v,pAb)