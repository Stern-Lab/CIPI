# Context
Detecting context dependent mutation rates from NGS data.<br />
We designed a statistical method that allows for detection and 
evaluation of the effects of different motifs on mutation rates. 

## Reference:
"A Bayesian framework for inferring the influence of sequence context on single base modifications" ,Guy Ling,
Danielle Miller, Rasmus Nielsen, Adi Stern


## Getting started:
A jupyter notebook for running the code in available [here](RunContextAnalysis.ipynb) with two optional modes:
- simulate
- real data inference

Input frequency file can be either a csv file or a tab delimited file (.freq):
*Note* that the simulations result have an additional line :`#moran model results`
```
#moran model results
Pos    Base    Freq    Ref    Read_count    Rank    Prob
1    A    0.9826    A    10000    3    1.0
1    C    0.0012    A    10000    0    1.0
1    G    0.0058    A    10000    1    1.0
1    T    0.0104    A    10000    2    1.0
2    A    0.9852    A    10000    3    1.0
2    C    0.0053    A    10000    1    1.0
2    G    0.0030    A    10000    0    1.0
2    T    0.0065    A    10000    2    1.0
```

Generating frequency files from raw NGS data (fastq files) can be done by our AccuNGS pipeline<br />
"AccuNGS: detecting ultra-rare variants in viruses from clinical samples." (Gelbart, M., et al. 2018).<br />
code available [here](https://github.com/SternLabTAU/AccuNGS)
## 

## Outputs
The script creates a folder named `resultSummary` with two types of files:
* MutationFilename_allChainsSummary.csv - all MCMC chains
* MutationFilename_bestChainsSummary.csv - the final file for analysis
 
 Each file should include all possible motifs and their final beta and gamma:
 
 ```
 betaMean	gammaMean	betaSD	gammaSD	topBeta	bottomBeta	topGamma	bottomGamma	name
-0.035037904	0	0.370295511	0	0.610536338	-0.644909518	0	0	P4__A
0.111123401	0	0.253151736	0	0.376907698	-0.847964432	0	0	P4__C
1.882467488	1	3.86E-11	0	1.882467488	1.882467488	1	1	P4__G
0.048830458	0	0.32543011	0	0.713381789	-0.511792261	0	0	P4__T
0.159739611	0	0.191817238	0	0.505363867	-0.320087558	0	0	P1_P4__AG
-0.11833743	0	0.259224332	0	0.455072615	-0.474370422	0	0	P1_P4__AT
-0.076704198	0	0.190523941	0	0.304878287	-0.463398403	0	0	P1_P4__CG
 ```
 
 Here we can see that the motif `P4__G` is significanct as `gammaMean = 1`.
 
Also, a gamma and beta posterior plots will be saved. For only presenting figures go to `GammaPriorTest.py` and under `outputAnalysis` set `plot=True` 


### Requirements
Python 2.7 with the following packages installed:
* numpy
* pandas
* Bio
* pymc3
* theano
* matplotlib
* seaborn
* tqdm






