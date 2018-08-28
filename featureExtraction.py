'''
Created on Mar 15, 2017

@author: Guyling1
'''

import pandas as pd 
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
import itertools as it
from Bio import SeqIO


class featureExtraction(object):
    '''
    classdocs
    '''
    global MT
    MT=[x for x in it.permutations('ACTG',2)]
    MT=["".join(x) for x in MT]
    #coding region is a list of tuples of (startNuc,endNuc)
    def __init__(self,freqsInput,codingRegions,k):
        '''
        Constructor
        '''
        self.codingRegion=codingRegions
        self.k=k
        self.freqs=freqsInput
        self.data=self.parseFreqs()
        self.chromosome=self.parseChromosome()
        self.regTable=None
        
    def parseFreqs(self):
        try:
            data=pd.read_table(self.freqs,sep='\s+',skiprows=1)
        except:
            data=pd.reed_csv(self.freqs)
        data=data[data['Ref']!='-']
        data=data.set_index([range(1,len(data.index)+1)])
        return data
    
    def parseChromosome(self):
        data=self.data
        chro="".join(list(data[data.index%4==0]['Ref']))
        return chro
    
    def leaveOnlyCoding(self):
        coding=self.codingRegion
        region=[range(code[0],code[1]) for code in coding] #range of coding regions
        region=[item for sublist in region for item in sublist] #flatten list
        index=self.data[[pos in region for pos in self.data['Pos']]].index#getting the index of the coding positions
        codingData=self.data.iloc[index]
        codingData=codingData.set_index([range(1,len(codingData.index)+1)])
        codingData['translationIndex']=(codingData.index-1)/4+1 #which amino acid index in the coding region
        return codingData
    
    def leaveOnlySynMutations(self):
        synList=[]#output  list of syn indices
        data=self.leaveOnlyCoding()
        for i in range(len(data)):
            mutOption=data.iloc[i]
            if mutOption['Base']==mutOption['Ref']: #not a mutation but same base    
                continue
            pos=int(mutOption['Pos']-1)#-1 for 0 base indexing of the position
            #changing the different options depending on the position of the base in the AA
            if mutOption['translationIndex']%3==1:
                originalSeq=self.chromosome[pos:pos+3]
                originalAA=Seq(originalSeq,generic_dna).translate()[0]
                newSeq=originalSeq[:]
                newSeq=list(newSeq)
                newSeq[0]=mutOption['Base']
                newSeq="".join(newSeq)
                newAA=Seq(newSeq,generic_dna).translate()[0]
            
            if mutOption['translationIndex']%3==2:
                originalSeq=self.chromosome[pos-1:pos+2]
                originalAA=Seq(originalSeq,generic_dna).translate()[0]
                newSeq=originalSeq[:]
                newSeq=list(newSeq)
                newSeq[1]=mutOption['Base']
                newSeq="".join(newSeq)
                newAA=Seq(newSeq,generic_dna).translate()[0]
            
            if mutOption['translationIndex']%3==0:
                originalSeq=self.chromosome[pos-2:pos+1]
                originalAA=Seq(originalSeq,generic_dna).translate()[0]
                newSeq=originalSeq[:]
                newSeq=list(newSeq)
                newSeq[2]=mutOption['Base']
                newSeq="".join(newSeq)
                newAA=Seq(newSeq,generic_dna).translate()[0]
            
            if newAA==originalAA:
                synList.append(i)
                continue
        synData=data.iloc[synList]
        synData=synData.set_index([range(1,len(synData.index)+1)])
        return synData
    
    def addContextToData(self,data):
        k=self.k
        chro=self.chromosome
        context=data['Pos'].apply(lambda x:chro[int(x-1)-k//2:int(x-1)+k//2+1])#slicing the part from the chro. after moving to 0-base
        data['context']=context
        data['context'].apply(lambda x:"".join(x))
        return data
    
    def createRegressionTable(self):
        codingData=self.leaveOnlySynMutations()
        data=self.addContextToData(codingData)
        k=self.k
        positions=['P{}'.format(i) for i in range(1,k+1)]
        kmers=[]
        mutTypes=[]
        prob=[]
        counts=[]
        indices=[]
        for mt in MT:
            refNuc=mt[0]
            newNuc=mt[1]
            mutData=data[(data['Base']==newNuc) & (data['Ref']==refNuc)]
            contexts=mutData['context'].unique()
            for c in contexts:
                contextData=mutData[mutData['context']==c]
                cnt=contextData['Read_count'].sum()
                prb=(contextData['Read_count']*contextData['Freq']).sum()/cnt+10**-9
                indx=list(contextData['Pos'].unique())
                #changing the kmer to be with the new nuc in the middle
                c=list(c)
                c[k//2]=newNuc
                c="".join(c)
                kmers.append(c)
                mutTypes.append(mt)
                prob.append(prb)
                counts.append(cnt)
                indices.append(indx)
        
        regTable=pd.DataFrame(columns=['kmer','mutationType']+positions+['prob','indices','counts'])
        regTable['kmer']=kmers
        for i in range(1,k+1):
            regTable['P{}'.format(i)]=[kmer[i-1] for kmer in kmers]
        regTable['mutationType']=mutTypes
        regTable['prob']=prob
        regTable['indices']=indices
        regTable['counts']=counts
        regTable=regTable.sort_values('kmer')
        self.regTable=regTable
        return regTable



        
    
        
            
                
                
                
            
        
    
    
    
    
        
        