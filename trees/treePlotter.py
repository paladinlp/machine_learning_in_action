# -*-coding:utf-8-*-
__author__ = 'paladinlp'
__date__ = '2017/11/27 18:39'

import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = 'sawtooth',fc = '0.8')

leafNode = dict(boxstyle = 'round4',fc = '0.8')
arrow_args = dict(arrowstyle = '<-')

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt, xy = parentPt,xycoords='axes fraction',xytext = centerPt,textcoords = 'axes fraction',va='center',ha = 'center',bbox = nodeType,arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False)
    plotNode(u'决策节点',(0.5,0.1),(0.1,0.5), decisionNode)
    plotNode(u'叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

# def createPlot():
#     fig = plt.figure(1, facecolor = 'white')
#     fig.clf()
#     createPlot.ax1 = plt.subplot(111, frameon = False)
#     plotNode('nonLeafNode', (0.2, 0.1), (0.4, 0.8), decisionNode)
#     plotNode('LeafNode', (0.8, 0.1), (0.6, 0.8), leafNode)
#     plt.show()

if __name__ == '__main__':
    createPlot()