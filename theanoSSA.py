'''
Created on Apr 26, 2017

@author: Guyling1
'''
import theano
import theano.tensor as tt
import theano.scan_module as ts
from theano import shared
import numpy as np
from theano import scan_module
import itertools as it

def changeRate(x,selection,mutation,S):
    N=tt.sum(x)
    rate=tt.zeros(S.shape[1])
    for k in range(S.shape[1]):
        r=S[:,k]
        i=np.where(r==1)[0][0]
        j=np.where(r==-1)[0][0]
        mutSum=x[j]*mutation[j,i]
        rate[k]=x[i]*x[j]/N*(1+selection[i])+mutSum     
    return rate

def ssaStep(vectFunc,x,t,tfinal,selection,mutation,S,reactions):
    vFunc=([x,selection,mutation,S],vectFunc)
    v=vFunc(x,selection,mutation,S)
    a0=tt.sum(v)
    r1=shared(np.random.rand())
    tau=1/a0*tt.log(1/r1)
    prob=v/a0
    j=tt.raw_random.choice(reactions, p=prob)
    x+=S[:,j]
    t=t+tau
    return x,t, scan_module.until(t>tfinal)
    
def createStochMatrixForNucs():
    indices=[x for x in it.permutations(range(4),2)]
    S=np.zeros((4,len(indices)))
    for k in range(len(indices)):
        i=indices[k][0]
        j=indices[k][1]
        S[i,k]=1
        S[j,k]=-1
    return S   

def ssa(x0,t0,vectFunc,tfinal,S,Nr,selection,mutation,steps=True):
    x=tt.ivector("x")
    t=tt.dscalar("time")
    x=x0
    t=t0
    reactions=tt.arange(Nr)
    results,updates=theano.scan(fn=ssaStep,outputs_info=x0,n_steps=2048,non_sequences=(vectFunc,tfinal,selection,mutation,S,reactions))
    res=results[-1]
    ssaApply=theano.function(inputs=[x,t,vectFunc,tfinal,selection,mutation,S,reactions],outputs=res,updates=updates)


x0=np.array([100000,0,0,0])

N=shared(np.sum(x0))
x0=shared(x0)

t0,tfinal=shared(0),shared(14)

S=createStochMatrixForNucs()
S=shared(S)
Nr=shared(12)
selection=shared(np.zeros(4))
mutation=np.zeros((4,4))
mutation+=10**-5/3.0
np.fill_diagonal(mutation,1-10**-5)
mutation=shared(mutation)
ssa(x0, t0, vectFunc, tfinal, S, Nr, selection, mutation)

    
    
    