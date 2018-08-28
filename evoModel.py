'''
Created on Jan 16, 2017

@author: Guyling1
'''
import numpy as np
import itertools as it
import pandas as pd

class evoModel(object):
    '''
    classdocs
    '''


    def __init__(self, modelType,rateParam,space=[0,1,2,3],p=[0.25,0.25,0.25,0.25]):
        '''
        Constructor
        '''
        self.model=modelType
        self.params=rateParam
        self.space=space
        self.piVector=p
    """
    This function creates a probabibilty dictionatry of the different couples above the space. The values in the dictionary aren't numbers but functions 
    """
    def getProb(self,t):
        n=len(self.space)
        probMatrix=np.zeros((n,n))
        pi=self.piVector
        if self.model=='JC':#Jukes cantor model
            for i in range(n):
                for j in range(n):
                    mu=self.params['mu']
                    v=3/float(4)*t*mu
                    if i==j:
                        probMatrix[i,j]=0.25+0.75*np.exp(-(4/float(3)*v))
                    else:
                        probMatrix[i,j]=0.25-0.25*np.exp(-(4/float(3)*v))
        if self.model=='HKY':
            A,C,G,T=pi[0],pi[1],pi[2],pi[3]
            realBeta=self.params['realBeta']
            alpha=self.params['alpha']
            h=T+((T*(A+G))/(T+C))*np.exp(-realBeta*t) +(C/(T+C))*np.exp(-((T+C)*alpha+(A+G)*realBeta)*t)
            i=C+((C*(A+G))/(T+C))*np.exp(-realBeta*t) -(C/(T+C))*np.exp(-((T+C)*alpha+(A+G)*realBeta)*t)
            j=T+((T*(A+G))/(T+C))*np.exp(-realBeta*t) -(T/(T+C))*np.exp(-((T+C)*alpha+(A+G)*realBeta)*t)
            k=C+((C*(A+G))/(T+C))*np.exp(-realBeta*t) +(T/(T+C))*np.exp(-((T+C)*alpha+(A+G)*realBeta)*t)
            l=A+((A*(T+C))/(A+G))*np.exp(-realBeta*t) +(G/(A+G))*np.exp(-((A+G)*alpha+(T+C)*realBeta)*t)
            m=G+((G*(T+C))/(A+G))*np.exp(-realBeta*t) -(G/(A+G))*np.exp(-((A+G)*alpha+(T+C)*realBeta)*t)
            n=A+((A*(T+C))/(A+G))*np.exp(-realBeta*t) -(A/(A+G))*np.exp(-((A+G)*alpha+(T+C)*realBeta)*t)
            o=G+((G*(T+C))/(A+G))*np.exp(-realBeta*t) +(A/(A+G))*np.exp(-((A+G)*alpha+(T+C)*realBeta)*t)
            p=A*(1-(np.exp(-realBeta*t)))
            q=G*(1-(np.exp(-realBeta*t)))
            r=T*(1-(np.exp(-realBeta*t)))
            s=C*(1-(np.exp(-realBeta*t)))
            probMatrix=np.matrix([[h,i,p,q],[j,k,p,q],[r,s,l,m],[r,s,n,o]])
        return probMatrix
    
    def getPiVector(self):
        return self.piVector
    
    def getCharSpace(self):
        return self.space
    
    def getSpaceLen(self):
        return len(self.space)
            