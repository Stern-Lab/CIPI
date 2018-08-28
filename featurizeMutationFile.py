'''
Created on Feb 16, 2017

@author: Guyling1
'''
import pandas as pd
import itertools
import numpy as np

def generateFeature(k):
    for j in range(1,k+1):
        for feature in itertools.imap(''.join,itertools.combinations('1245',j)):
            yield feature

def functionGenerator(positions,x):
    res=''
    for position in positions:
        res+=x['P{}'.format(position)]
    return res

def createFeaturePrefix(featureList):
    res=''
    for feature in featureList:
        res+="P{}_".format(feature)
    return res

def getDummyVarPositionForK(data,k):
    for feature in generateFeature(k):
        featureList=[int(x) for x in list(feature)]
        concatFeature=data.apply(lambda x: functionGenerator(featureList,x),axis=1)
        dummies=pd.get_dummies(concatFeature,prefix=createFeaturePrefix(featureList))
        data=pd.concat([data,dummies],axis=1)
    return data

def cleanDataMatrix(data,withNull=False):
    for i in range(1,6):
        data=data.drop('P{}'.format(i),axis=1)
    if not(withNull):
        data=data[data['prob']>10**-8]
    return data

def createCovarsAndProbVectorForMutationType(data,mutType):
    data=data[data['mutationType']==mutType]
    probVector=data.prob
    counts=data.counts
    covars=data.drop('prob',axis=1).drop('kmer',axis=1).drop('mutationType',axis=1).drop('indices',axis=1).drop('counts',axis=1)
    return covars,probVector,counts,data





