'''
Created on May 10, 2017

@author: Guyling1
'''
import mcmc

'''
Created on Jan 16, 2017

@author: Guyling1
'''
import evoModel
import simulations
print "ok"
import numpy as np
import numpy.random as nr
from featureExtraction import featureExtraction
global NUCS
import os
import glob
import featurizeMutationFile as f 
import pandas as pd
import mcmc
import time

NUCS=['A','C','G','T']
mod=evoModel.evoModel('JC',rateParam={'mu':5*10**-5})

def vecMotifToMotif(vec,k=5):
    indices=np.where(vec!=-1)[0]
    middle=np.where(indices==k//2)
    indices=np.delete(indices,middle)#removing the middle item because it's not part of the motif
    indices+=1
    pos=["P{}".format(i) for i in indices] #getting the indices of the features
    chars=[NUCS[int(vec[i-1])]  for i in indices] # getting the char in those indices., i-1 is for moving to zero base again
    pos="_".join(pos)
    chars="".join(chars)
    motif=pos+"__"+chars
    print motif
    return motif
        


def createSimulationRandomly(reference, k=5,oldNuc=1,scale=2):
    sim=simulations.EvolustionSimulations(mod,time=15,sampleNum=10000,chroLen=800)
    sim.parseChromosome(reference)
    motifList=[]
    #randomly create motifs and add them to the simulation
    numOfMotifs=nr.randint(low=1,high=4)
    for i in range(numOfMotifs):
        vec=np.ones(k)*-1#setting all positions to be non significant. Null
        vec[k//2]=oldNuc #simulating for oldNuc->X mutations
        newNuc=nr.choice(4)#choosing the new nucleotide type
        motifOrder=nr.choice(range(1,k-1)) #length of motif
        possiblePositions=range(k)
        possiblePositions.remove(k//2) #can't be a part of the motif
        positions=nr.choice(possiblePositions,size=motifOrder)
        vec[positions]=nr.choice(range(len(NUCS)),size=len(positions))
        effectSize= np.random.normal(scale=scale)
        motif=((vec,newNuc),effectSize)
        
        motifList.append(motif)
    #adding the motifs constructed to the simulation
    for m in motifList:
        sim.addMotif(m[0],m[1])
    sim.initializeProb()
    regularizedMotifList=[(vecMotifToMotif(m[0][0], k))   for m in motifList]
    newNucList=[m[0][1] for m in motifList]
    CoeffList=[format(m[1],'.2f') for m in motifList]
    name=["{}_{}_{}".format(regularizedMotifList[i],NUCS[newNucList[i]],CoeffList[i])  for i in range(len(regularizedMotifList))]
    name=":".join(name)
    name=name+"_oldNuc_{}".format(NUCS[oldNuc])
    name=name+":"+time.asctime()
    sim.setName(name)
    moranMatrix=sim.moranModelByPosition()
    sim.toFreqsFile(moranMatrix, r'./{}.freqs'.format(name))
    fc=featureExtraction(r'./{}.freqs'.format(name),[(748,7371)],5)
    fc.createRegressionTable()  
    fc.regTable.to_csv(r'./{}.csv'.format(name))

    return

def analyseSample(location, file,oldNuc=1,newNuc=3):
    data=pd.read_csv(location+file)
    data=f.getDummyVarPositionForK(data,2)
    data=f.cleanDataMatrix(data,withNull=True)
    X,y_,initNum,D=f.createCovarsAndProbVectorForMutationType(data,NUCS[oldNuc]+NUCS[newNuc])
    features=X.columns[1:]

    sigList=mcmc.featureSelectionPairs(X, y_, initNum, features) 
    
    if len(sigList)>0:
        sigFeatures,resultSummary=mcmc.featureSelection(X, y_, initNum, features, sigList,name=file+"_old:{}_new:{}_".format(NUCS[oldNuc],NUCS[newNuc]))
        
        sigFeatures=features[[s[0] for s in sigFeatures]]
        out=open(location+file+"_old:{}_new:{}_sigFeatures.out".format(NUCS[oldNuc],NUCS[newNuc]),'w')
        for s in sigFeatures:
            out.write(str(s)+"\n")
        out.close()
        resultSummary.to_csv(location+file+"_old:{}_new:{}_resultSummary.csv".format(NUCS[oldNuc],NUCS[newNuc]))
    else:
        out=open(location+file+"_old:{}_new:{}_sigFeatures.out".format(NUCS[oldNuc],NUCS[newNuc]),'w')
        out.write("#no sig features in the initial stage")
    pass
    
