'''
Created on Apr 24, 2017

@author: Guyling1
'''
import numpy as np
from pymc3.distributions import Discrete
import theano.tensor as tt
from theano import as_op
from theano import shared,function
import theano
from pymc3.distributions.dist_math import bound
theano.config.compute_test_value = "ignore"

"""
This Gamma prior is a bernouli prior utlizing the "dilute" concept 
as described http://www-stat.wharton.upenn.edu/~edgeorge/Research_papers/ims.pdf P14
and https://projecteuclid.org/download/pdfview_1/euclid.imsc/1288099018 section 4
"""
def likelihoodFunc(v,size,graphMatrix,value):
    v=tt.fill(tt.ones(size),v)#create a vector of the null bernouli probs 
    penalty=tt.dot(graphMatrix,value) #matrix multipl. to get how many correlated features are "on"
    penalty=5-4*tt.exp(-penalty)
    v=v*penalty
    ll=tt.sum(tt.log(tt.pow(v, value)))
    return ll

class GammaaPrior(Discrete):
    """Bernoulli log-likelihood

    The Bernoulli distribution describes the probability of successes
    (x=1) and failures (x=0).

    .. math:: f(x \mid p) = p^{x} (1-p)^{1-x}

    ========  ======================
    Support   :math:`x \in \{0, 1\}`
    Mean      :math:`p`
    Variance  :math:`p (1 - p)`
    ========  ======================

    Parameters
    ----------
    p : float
        Probability of success (0 < p < 1).
    """

    def __init__(self, v,graphMatrix,*args, **kwargs):
        super(GammaaPrior, self).__init__(*args, **kwargs)
        self.v = v
        self.size=graphMatrix.shape[1]
        self.graphMatrix=graphMatrix
        self.mode=np.zeros(self.size,dtype=np.int64)
        self.type=tt.lvector
        self.shape=(self.size,)
        
        


    def logp(self, value):
        ll=likelihoodFunc(shared(self.v), shared(self.size), shared(self.graphMatrix),value)
        return ll
   
