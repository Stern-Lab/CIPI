'''
Created on Feb 27, 2017

@author: Guyling1

'''
import numpy as np
import scipy as sc
import itertools as it
import theano.tensor as T
from theano import function
from theano import shared


def createStochMatrixForNucs():
    indices=[x for x in it.permutations(range(4),2)]
    S=np.zeros((4,len(indices)))
    for k in range(len(indices)):
        i=indices[k][0]
        j=indices[k][1]
        S[i,k]=1
        S[j,k]=-1
    return S
    
def changeRate(x,selection,mutation,S):
    N=sum(x)
    rate=np.zeros(S.shape[1])
    for k in range(S.shape[1]):
        r=S[:,k]
        i=np.where(r==1)[0][0]
        j=np.where(r==-1)[0][0]
        mutSum=x[j]*mutation[j,i]
        rate[k]=x[i]*x[j]/N*(1+selection[i])+mutSum     
    return rate

def ssa(x0,t0,vectFunc,tfinal,S,Nr,selection,mutation,steps=True):
    x=x0# x is a vector of the ammount of individuals in each population 
    t=t0
    reactions=range(Nr)
    if steps:
        steps=[] #steps and times are enabling the plotting of the process and not ony the end state
        times=[]
        steps.append(x)
        times.append(t)
    while t<tfinal:
        v=vectFunc(x,selection,mutation,S)
        a0=float(v.sum())
        if np.isclose(a0,0):
            print "both died" #if the reaction vector is 0 than both 
            t=tfinal
        r1=np.random.uniform()
        tau=(1/a0)*np.log(1/r1)
        
        if t+tau>tfinal:
            t=tfinal
            break
        t=t+tau
        
        if steps:
            times.append(t)
        prob=v/a0
        j=np.random.choice(reactions,p=prob)# defining a prob vetcor for the choosing of the reactions, similiar to r2 in the notes 
        x=x+S[:,j]
    
        if steps:
            steps.append(x)
    if steps:
        return steps,times
    return x
    
    
    
x0=np.array([100000,0,0,0])

N=np.sum(x0)

t0,tfinal=0,14

S=createStochMatrixForNucs()
Nr=12
selection=np.zeros(4)
mutation=np.zeros((4,4))
mutation+=10**-5/3.0
np.fill_diagonal(mutation,1-10**-5)


    