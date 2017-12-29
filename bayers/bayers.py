# -*-coding:utf-8-*-
__author__ = 'paladinlp'
__date__ = '2017/12/10 14:33'


from numpy import *
import re
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

def bagOfWord2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print('the word :%s is not in my Vocabulary!' %  word)
    return returnVec





def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords);  p1Num = ones(numWords)
    p0Denom = 2.0;  p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]  # 最后的结果是每一个词条的计数值
            p1Denom += sum(trainMatrix[i])  #所有词条的计数值,除的是p(w/ci)
        else:
            p0Num += trainMatrix[i]  # 最后的结果是每一个词条的计数值
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec)+log(pClass1)
    p0 = sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():

    docList=[];  classList = []; fullText = []
    for i in range(1, 26):

        wordList = textParse(open('email/spam/%d.txt' % i, ).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, ).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50) ; testSet = []
    for i in range(10):

        randIndex = int(random.uniform(0, len(trainingSet)))
        print(trainingSet[randIndex])
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v, p1v, pSpam = trainNB0(trainMat, array(trainClasses))

    errCount = 0
    for docIndex in testSet:
        wordVector = setOfWord2Vec(vocabList, docList[docIndex])

        if classifyNB(array(wordVector), p0v, p1v, pSpam) != classList[docIndex]:
            errCount += 1
    print('the error rate is :', float(errCount)/len(testSet))




if __name__ == '__main__':
    listOPost, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPost)
    trainMat = []
    for postinDoc in listOPost:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    p0v, p1v, pAb = trainNB0(trainMat, listClasses)

    # testEntry = ['love', 'my', 'dalmation']
    # thisDoc = array(setOfWord2Vec(myVocabList,testEntry))
    # print (testEntry,'classified as :', classifyNB(thisDoc ,p0v, p1v, pAb))
    #
    # testEntry = ['stupid', 'garbage']
    # thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    # print(testEntry, 'classified as :', classifyNB(thisDoc, p0v, p1v, pAb))

    spamTest()
    # for i in range(1, 26):
    #     wordList = textParse(open('email/spam/%d.txt' % i,'r').read())
    #     print(wordList)








