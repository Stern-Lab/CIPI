#simulate data
from __future__ import division
import numpy as np
import numpy.random as nr
import pymc3 as pm
from theano import shared
from reverseJumpSampler import Metropolis
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import time
#import theano.tensor as T
from theano import as_op
import featurizeMutationFile as f 
import scipy.special as sc
from mcmc import featureListToGraphMatrix
from tqdm import tqdm
from joblib import Memory
import math
from tempfile import mkdtemp
import os

global MT
MT=['AG','CT','GA','TC',]
import pandas as pd
global EPSILON
EPSILON=10**-9


def uniformPrior(x,low,high):
    if x<=high and x>=low:
        return 1./(high-low)
    else:
        return -float('inf')
def bernouliPrior(v,size,vector,reuseVals=None):
    v=np.ones(size)*v
    
    if reuseVals is None:
        ll=np.sum(vector*np.log(v) +(1-vector)*np.log(1-v))
    else:
        logv=reuseVals['logv']
        logvbar=reuseVals['logvbar']
        ll=np.sum(vector*logv +(1-vector)*logvbar)
    
    return ll

def posteriorMean(Matrix):
    return Matrix.mean(axis=0)

def posteriorSD(Matrix):
    return Matrix.std(axis=0)

def posteriorTopPercentile(Matrix,alpha=97.5):
    return np.percentile(Matrix,alpha,axis=0)

def posteriorBottomPercentile(Matrix,alpha=2.5):
    return np.percentile(Matrix,alpha,axis=0)

    



def LDPPPrior(X,v,size,vector,w=100):#http://faculty.chicagobooth.edu/veronika.rockova/determinantal.pdf
    berll=np.sum(vector*np.log(v) +(1-vector)*np.log(1-v))
    indices=np.where(vector==1)[0]
    if len(indices)==0:
        return berll#for null matrix det=1-> log(det)=0
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    xGamma=X[:,indices]
    dot=np.dot(xGamma.transpose(),xGamma)
    dot=dot/(np.diag(dot)[0])#normalizing by the diag. value ensures det in [0,1]
    det=np.abs(np.linalg.det(dot))
    logDet=np.log(det)*w#adding a weight to the penalty
    
    ll=berll+logDet
    if np.isnan(ll):
        print det,logDet
        ll=-np.inf
    return ll

def normalLL(vector,mean,variance):
    n=len(vector)
    return -np.sum((vector-mean)**2/2*variance)

def chooseln(N, k):
    return sc.gammaln(N+1) - sc.gammaln(N-k+1) - sc.gammaln(k+1)

def binomialLL(vector,initVector,p,reuse=None):
    p,vector,initVector= np.squeeze(np.asarray(p)),np.squeeze(np.asarray(vector)),np.squeeze(np.asarray(initVector))
    if reuse is None:
        combln=chooseln(initVector, vector).sum()
    else:
        combln=reuse['combln']
    lnp=np.log(p)
    lnbar=np.log(1-p)
    berln=(vector*lnp).sum()+((initVector-vector)*lnbar).sum()
    return combln+berln
        
    

def correlationStepProbVector(corMatrix,gamma):
    m=len(gamma)
    probVector=np.ones(m)*(1./m)#uniform prior 
    penalty=np.dot(corMatrix,gamma) #matrix multipl. to get how many correlated features are "on"
    indices=np.where(penalty>0)[0]#make sure no negative entries
    penalty[indices]=penalty[indices]-1 #removing the diagnol entry from the penalty (each feature is correalted with self)
    penalty=5-4*np.exp(-penalty)
    probVector=probVector*penalty
    probVector=probVector/probVector.sum()#normalizing to have total 1
    return probVector

def normalPrior(vector,variance=0.1,mean=0):
    return -np.sum((vector-mean)**2/(2*variance))

def multiVarNormalLL(vector,mean,cov):
    inverSigma=np.linalg.inv(cov)
    diff=vector-mean
    expPart=np.dot(np.dot(-diff.transpose(),inverSigma),diff)/2
    ll=expPart
    return ll   

def gammaCov(gamma,C=3):
    m=len(gamma)
    cov=np.eye(m)*0.1
    indices=np.where(gamma==1)[0]
    cov[indices]*=C
    return cov

def featureSelection(X,y,initNum,features,iterations=20000,betaThreshold=0.5,gammaThreshold=0.7):
    n,m=X.shape
    indexList=[]
    for i in range(m):
        Xcur=X[:,i]
        gamma,beta,accept,logPost=MCMC(Xcur, y, initNum, iterations,v=10**-50,determinantel=False)
        precOn=gamma[:,0].mean()
        meanBeta=beta.mean()
        if np.abs(meanBeta)>betaThreshold and precOn>gammaThreshold:
            indexList.append(i)
    return indexList
        
    
def MCMC(X,y,initNum,iterations,start=None,sigmaBoundries=(0.5,3),varianceFactor=4,v=10**-200,reuseDict=True,verbose=False,corMatrix=None,heat=1,determinantel=False):
    tic=time.time()
    n,m=X.shape
    #burnin is 10% of the iteration number
    burnin=int(iterations/10)
    #setting higher v for low complexity models
    #v=max(10**(-40*m),v)
    #creating a dictionary of values for repeating computation
    if reuseDict:
        d={}
        d['logv']=math.log(v)
        d['logvbar']=math.log(1-v)
        d['combln']=chooseln(initNum,y).sum()
    
    #finding empirical mean and SD to initalize the alpha var.
    empiricalMean=sc.logit(y/initNum+EPSILON).mean()
    empiricalVar=sc.logit(y/initNum+EPSILON).mean()**2
    alpha0=empiricalMean
    
    #initilizing the chains to 0 for all vars. 
    
    betaMatrix=np.zeros((iterations,m))
    gammaMatrix=np.zeros((iterations,m))
    acceptVector=np.zeros(iterations)
    logPostVector=np.zeros(iterations,dtype=np.float64)
    
    #initializing the parameters for beta dist. 
    betaMu=np.zeros(m)
    
    accept=0 #accept counter 
    
    #setting random start points for latent vars.
    beta0=nr.normal(size=m)
    if start is None:
        gamma0=np.zeros(m)
    else:
        gamma0=start
    betaOld=beta0
    gammaOld=gamma0
    alphaOld=alpha0

    #calculating old ll
    betaGammaOld=betaOld*gammaOld
    muOld=sc.expit(np.dot(X,betaGammaOld)+alphaOld)
    oldLL=binomialLL(y, initNum, muOld,reuse=d)
    
    #iterating over the chain steps 
    for i in tqdm(range(iterations)):
        
        #suggesting new vars. 
        gammaNew=gammaOld.copy()
        numPositionChangeGamma=nr.poisson(lam=1)#poisson number of changes on the gamma vector
        if not(corMatrix is None):
            probVector=correlationStepProbVector(corMatrix, gammaNew)
        else:
            probVector=np.ones(m)*1./m
            gammaOneIndex=np.where(gammaNew==1)[0]
            probVector[gammaOneIndex]*=10
            probVector/=probVector.sum()
            #print probVector
        gammaIndexSwitch=nr.choice(m,size=numPositionChangeGamma,p=probVector)# ,p=probVector,replace=False
        gammaNew[gammaIndexSwitch]=1-gammaOld[gammaIndexSwitch]
        betaNew=betaOld.copy()
        alphaNew=nr.normal(loc=alphaOld,scale=1)
        betaIndexSwitch=nr.choice(m)
        betaNew[betaIndexSwitch]=nr.normal(loc=betaOld[betaIndexSwitch],scale=1)
        
        #calculating old and new post 
        
        betaGammaNew=betaNew*gammaNew
        muNew=sc.expit(np.dot(X,betaGammaNew)+alphaNew)

        newLL=binomialLL(y,initNum,muNew,reuse=d)
        
        if determinantel:
            PrGnew=LDPPPrior(X, v, m, gammaNew)#
            PrGold=LDPPPrior(X, v, m, gammaOld)#, reuseVals=d
        else:
            PrGnew=bernouliPrior(v, m, gammaNew, reuseVals=d)
            PrGold=bernouliPrior(v, m, gammaOld, reuseVals=d)

        PrAnew=normalPrior(alphaNew, variance=1, mean=empiricalMean)
        PrAOld=normalPrior(alphaOld, variance=1, mean=empiricalMean)

        covOld=gammaCov(gammaOld)
        covNew=gammaCov(gammaNew)
        PrBnew=multiVarNormalLL(betaNew, betaMu, covNew)
        PrBold=multiVarNormalLL(betaOld, betaMu, covOld)
        newPost=PrBnew+PrGnew+newLL+PrAnew
        oldPost=PrBold+PrGold+oldLL+PrAOld
        #MH step in log space
        r=nr.rand()
        MHprob=min(0,(newPost)-(oldPost))#+PrSnew+PrSold
        
        #accept new step
        if r<np.exp(MHprob):

            gammaOld=gammaNew
            betaOld=betaNew
            alphaOld=alphaNew
            muOld=muNew
            oldLL=newLL
            accept+=1
            acceptVector[i]=1
        
        #setting the cur. chain status     
        gammaMatrix[i,:]=gammaOld
        betaMatrix[i,:]=betaOld
        logPostVector[i]=oldPost
        
        #verbose priniting of variables 
        
        if verbose:
            print"_______________"
            print"iteration number :{}".format(i)
            print "gammaIndexSwitch switch "+ str(gammaIndexSwitch)
            print "betaIndexSwitch switch "+ str(betaIndexSwitch)
            print "new gamma "+str(gammaNew)
            print "old gamma "+str(gammaOld)
            print "new alpha"+str(alphaNew)
            print "old alpha"+str(alphaOld)
            print "new beta" + str(betaNew)
            print "old beta" + str(betaOld)
            #print "new sigma"+str(sigmaNew)
            #print "old sigma"+str(sigmaOld)
            #print "newSigma prior"+ str(PrSnew)
            #print "oldSigma prior"+ str(PrSold)
            print "newGammaPrior "+str(PrGnew)
            print "oldGammaPrior "+str(PrGold)
            print "newAlphaPrior "+str(PrAnew)
            print "oldAlphaPrior "+str(PrAOld)
            print "newBeta prior "+str(PrBnew)
            print "oldBeta prior "+str(PrBold)
            print "old number of features "+str(sum(gammaOld))
            print "new number of features "+str(sum(gammaNew))
            print "oldll "+str(oldLL)
            print "newll "+str(newLL)
            print "newPost"+str(newPost)
            print "oldPost"+str(oldPost)
            print "jump prob "+str(np.exp(MHprob))
            print"_______________"
        
    
    tac=time.time()
    # print "elapsed time:" +str(tac-tic)
    # print "accept rate:" +str(float(accept)/iterations)
    
    return gammaMatrix[burnin:,:],betaMatrix[burnin:,:],acceptVector,logPostVector


def multipleChainsAnalysis(X,y,initNum,iterations,chains,features=None,burnin=None,corMatrix=None,sigIndices=[]):
    if len(sigIndices)==0:
        sigIndices=range(X.shape[1])

    X=X[:,sigIndices]
    n,m=X.shape
    if burnin is None:
        burnin=int(iterations/10)
    
    iterAfterBurnin=iterations-burnin
    gammaInterChainVariance=np.zeros((iterations,m))
    gammaBetweenChinVariance=np.zeros((1,m))
    betaInterChainVariance=np.zeros((iterations,m))
    betaBetweenChainVariance=np.zeros((1,m)) 
    gammaMatrix=np.zeros((iterAfterBurnin*chains,m))
    betaMatrix=np.zeros((iterAfterBurnin*chains,m))
    acceptVector=np.zeros(iterations*chains)#accpet has no burn in 
    logPostVector=np.zeros(iterations*chains)
    bestMeanPost=-np.Inf
    bestChain=None
    startParamNumber=nr.poisson(lam=2)
    
    for i in range(chains):
        startGamma=np.zeros(m)
        indices=nr.choice(m,size=startParamNumber)
        startGamma[indices]=1

        gammaVal,betaVal,accept,logPost=MCMC(X, y,initNum,iterations, startGamma,corMatrix=corMatrix,heat=1)
        gammaMatrix[iterAfterBurnin*i:iterAfterBurnin*(i+1)]=gammaVal
        betaMatrix[iterAfterBurnin*i:iterAfterBurnin*(i+1)]=betaVal
        acceptVector[iterations*i:iterations*(i+1)]=accept
        logPostVector[iterations*i:iterations*(i+1)]=logPost
        meanpost=logPost.mean()
        if meanpost>bestMeanPost:
            bestMeanPost=meanpost
            bestChain=i
    gammaMatrixBest,betaMatrixBest=gammaMatrix[iterations*bestChain:iterations*(bestChain+1),:],betaMatrix[iterations*bestChain:iterations*(bestChain+1),:]
    
    allChains=pd.DataFrame(columns=['betaMean','gammaMean','betaSD','gammaSD','topBeta','bottomBeta','topGamma','bottomGamma','name'])
    allChains.name=[features[sigIndices[i]] for i in range(m)]
    allChains.betaMean=posteriorMean(betaMatrix)
    allChains.gammaMean=posteriorMean(gammaMatrix)
    allChains.betaSD=posteriorSD(betaMatrix)
    allChains.gammaSD=posteriorSD(gammaMatrix)
    allChains.topBeta=posteriorTopPercentile(betaMatrix)
    allChains.bottomBeta=posteriorBottomPercentile(betaMatrix)
    allChains.topGamma=posteriorTopPercentile(gammaMatrix)
    allChains.bottomGamma=posteriorBottomPercentile(gammaMatrix)
    
    bestChainSum=pd.DataFrame(columns=['betaMean','gammaMean','betaSD','gammaSD','topBeta','bottomBeta','topGamma','bottomGamma','name'])
    bestChainSum.name=[features[sigIndices[i]] for i in range(m)]
    bestChainSum.betaMean=posteriorMean(betaMatrixBest)
    bestChainSum.gammaMean=posteriorMean(gammaMatrixBest)
    bestChainSum.betaSD=posteriorSD(betaMatrixBest)
    bestChainSum.gammaSD=posteriorSD(gammaMatrixBest)
    bestChainSum.topBeta=posteriorTopPercentile(betaMatrixBest)
    bestChainSum.bottomBeta=posteriorBottomPercentile(betaMatrixBest)
    bestChainSum.topGamma=posteriorTopPercentile(gammaMatrixBest)
    bestChainSum.bottomGamma=posteriorBottomPercentile(gammaMatrixBest)
    return features[sigIndices],gammaMatrix,gammaMatrixBest,betaMatrix,betaMatrixBest,allChains,bestChainSum,logPostVector

def outputAnalysis(name,sigFeatures,gammaMatrix,gammaMatrixBest,betaMatrix,betaMatrixBest,allChains,bestChainSum,logPostVector,plot=False,folder=None):
    m=len(sigFeatures)
    if folder is None:
            folder="./"
    if plot:
        plt.plot(np.log(-logPostVector))
        plt.title("log Posterior")
        plt.show()
        plt.plot(gammaMatrixBest.sum(axis=1)) 
        plt.show()   
        for i in range(m):
            plt.hist(gammaMatrixBest[:,i],normed=True)
            plt.title("gammat coeff of {} feature".format(sigFeatures[i]))
            plt.show()
            plt.plot(betaMatrixBest[:,i])
            plt.title("beta coeff of {} feature".format(sigFeatures[i]))
            plt.show()
            plt.hist(betaMatrixBest[:,i], bins=20,normed=True)
            plt.title("histogram of values")
            plt.show()
    else:
        os.chdir(folder)  
        os.mkdir(os.pardir+"/"+name+"_plots")
        os.chdir(os.pardir+"/"+name+"_plots")
        plt.plot(np.log(-logPostVector))
        plt.title("loglog -Posterior")
        plt.savefig('./{}_log_posterior.png'.format(name))
        plt.clf()
        plt.plot(gammaMatrixBest.sum(axis=1))

        plt.savefig('./{}_numberOfFeatures.png'.format(name)) 
        plt.clf()
        for i in range(m):
            plt.hist(gammaMatrixBest[:,i],normed=True)
            plt.title("gamma coeff of {} feature".format(sigFeatures[i]))
            plt.savefig('./{}_gamma coeff of {} feature.png'.format(name,sigFeatures[i]))
            plt.clf()
            plt.plot(betaMatrixBest[:,i])
            plt.title("beta coeff of {} feature".format(sigFeatures[i]))
            plt.savefig('./{}_beta coeff of {} feature.png'.format(name,sigFeatures[i]))
            plt.clf()
            plt.hist(betaMatrixBest[:,i], bins=20,normed=True)
            plt.title("histogram of values of {} feature".format(sigFeatures[i]))
            plt.savefig('./{}_hist_values of {} feature.png'.format(name,sigFeatures[i]))
            plt.clf()
        os.chdir(folder)
        summaryFolder=folder+"/resultSummary"

        # create the folder if does not exists
        try:
            os.stat(summaryFolder)
        except:
            os.mkdir(summaryFolder)

        # print summaryFolder
        allChains.to_csv(summaryFolder+'/'+name+"_allChainsSummary.csv")
        bestChainSum.to_csv(summaryFolder+'/'+name+"_bestChainsSummary.csv")
        
def loadData(dataPath,mutType):
    data=pd.read_csv(dataPath)
    name=".".join(dataPath.split("/")[-1].split(".")[:-1])+"_"+mutType
    folder="/".join(dataPath.split("/")[:-1])
    data=f.getDummyVarPositionForK(data,2)
    data=f.cleanDataMatrix(data,withNull=True)
    X,y,initNum,D=f.createCovarsAndProbVectorForMutationType(data,mutType)
    features= X.columns[1:]
    X=np.matrix(X)[:,1:]
    n,m=X.shape
    y=np.array((y*initNum),dtype=int).reshape(n,1)
    initNum=np.array(initNum).reshape(n,1)
    return X,y,initNum,features,name,folder

def fullAnalysis(dataPath,mutationType):
    
    X,y,initNum,features,name,folder=loadData(dataPath, mutationType)
    sigIndex=featureSelection(X,y,initNum,features)
    sigFeatures,gammaMatrix,gammaMatrixBest,betaMatrix,betaMatrixBest,allChains,bestChainSum,logPostVector=multipleChainsAnalysis(X, y,initNum,1000000,1,sigIndices=sigIndex,features=features)#,corMatrix=cormatrix
    outputAnalysis(name+"_"+mutationType, sigFeatures, gammaMatrix, gammaMatrixBest, betaMatrix, betaMatrixBest, allChains, bestChainSum, logPostVector, plot=False, folder=folder)

