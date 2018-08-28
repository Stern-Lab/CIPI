import numpy as np
import numpy.random as nr
import theano

from pymc3.distributions import draw_values
from pymc3.step_methods.arraystep import ArrayStepShared, ArrayStep, metrop_select, Competence
import pymc3 as pm


__all__ = ['Metropolis', 'BinaryMetropolis', 'BinaryGibbsMetropolis',
           'CategoricalGibbsMetropolis', 'NormalProposal', 'CauchyProposal',
           'LaplaceProposal', 'PoissonProposal', 'MultivariateNormalProposal']

# Available proposal distributions for Metropolis


class Proposal(object):
    def __init__(self, s):
        self.s = s


class NormalProposal(Proposal):
    def __call__(self):
        return nr.normal(scale=self.s)


class CauchyProposal(Proposal):
    def __call__(self):
        return nr.standard_cauchy(size=np.size(self.s)) * self.s


class LaplaceProposal(Proposal):
    def __call__(self):
        size = np.size(self.s)
        return (nr.standard_exponential(size=size) - nr.standard_exponential(size=size)) * self.s


class PoissonProposal(Proposal):
    def __call__(self):
        return nr.poisson(lam=self.s, size=np.size(self.s)) - self.s


class MultivariateNormalProposal(Proposal):
    def __call__(self, num_draws=None):
        return nr.multivariate_normal(mean=np.zeros(self.s.shape[0]), cov=self.s, size=num_draws)


class Metropolis(ArrayStepShared):
    """
    Metropolis-Hastings sampling step

    Parameters
    ----------
    vars : list
        List of variables for sampler
    S : standard deviation or covariance matrix
        Some measure of variance to parameterize proposal distribution
    proposal_dist : function
        Function that returns zero-mean deviates when parameterized with
        S (and n). Defaults to normal.
    scaling : scalar or array
        Initial scale factor for proposal. Defaults to 1.
    tune : bool
        Flag for tuning. Defaults to True.
    tune_interval : int
        The frequency of tuning. Defaults to 100 iterations.
    model : PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).

    """
    default_blocked = False

    def __init__(self, vars=None, S=None,sigmaFactor=1, proposal_dist=NormalProposal, scaling=1.,
                 tune=True, tune_interval=100,varProp=1/30.,model=None, **kwargs):

        model = pm.modelcontext(model)

        if vars is None:
            vars = model.vars
        vars = pm.inputvars(vars)

        if S is None:
            S = np.ones(sum(v.dsize for v in vars))
        self.proposal_dist = proposal_dist(S)
        self.scaling = np.atleast_1d(scaling)
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0
        self.sigmaFactor=sigmaFactor
        self.varProp=varProp

        # Determine type of variables
        self.discrete = np.concatenate(
            [[v.dtype in pm.discrete_types] * (v.dsize or 1) for v in vars])
        self.any_discrete = self.discrete.any()
        self.all_discrete = self.discrete.all()

        shared = pm.make_shared_replacements(vars, model)
        self.delta_logp = delta_logp(model.logpt, vars, shared)
        super(Metropolis, self).__init__(vars, shared)

    def astep(self, q0):
        if not self.steps_until_tune and self.tune:
            # Tune scaling parameter
            self.scaling = tune(
                self.scaling, self.accepted / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        delta = self.proposal_dist() * self.scaling
        if self.any_discrete:
            oneLocations=np.where(q0[self.discrete]==1)[0]#looking for the 1 values in the vector
            zeroLocations=np.where(q0[self.discrete]==0)[0]
            #delta[oneLocations-len(self.discrete)]*=self.sigmaFactor #increasing the variance for 1 value entries. the oneLocations are the betas corresponding to the gamma=1
            #under the assumption that the vars are beta,gamma,...
        if self.any_discrete:
            if self.all_discrete:
                delta = np.round(delta, 0).astype('int64')
                q0 = q0.astype('int64')
                q = (q0 + delta).astype('int64')
            else:
                loc=nr.choice(self.discrete)#pick a location
                q=q0+delta#change all the q by delta 
                q[self.discrete]=q0[self.discrete]#set the discrete values to be q0, meaning disregarding the delta change
                r=nr.rand()#random number
                dif=len(oneLocations)-int(self.varProp*len(self.discrete))
                #http://www-stat.wharton.upenn.edu/~edgeorge/Research_papers/ims.pdf
                if loc in zeroLocations and dif>0:#adding another sig. feature and there are already more than expected
                    if r>np.exp(-dif):
                        loc2=nr.choice(oneLocations)#take an existing feature and reverse it 
                        q[loc2]+=1
                if loc in oneLocations and dif<0:#reducing feature number and there are already too few
                    if r>np.exp(dif):
                        loc2=nr.choice(zeroLocations)
                        q[loc2]+=1
                q[loc]+=1 #change the bit and %2 to stay {0,1}
                q[self.discrete]=q[self.discrete]%2
                
        else:
            q = q0 + delta
            
        q_new = metrop_select(self.delta_logp(q, q0), q, q0)

        if q_new is q:
            self.accepted += 1

        self.steps_until_tune -= 1

        return q_new

    @staticmethod
    def competence(var):
        if var.dtype in pm.discrete_types:
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE


def tune(scale, acc_rate):
    """
    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:

    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10

    """

    # Switch statement
    if acc_rate < 0.001:
        # reduce by 90 percent
        scale *= 0.1
    elif acc_rate < 0.05:
        # reduce by 50 percent
        scale *= 0.5
    elif acc_rate < 0.2:
        # reduce by ten percent
        scale *= 0.9
    elif acc_rate > 0.95:
        # increase by factor of ten
        scale *= 10.0
    elif acc_rate > 0.75:
        # increase by double
        scale *= 2.0
    elif acc_rate > 0.5:
        # increase by ten percent
        scale *= 1.1

    return scale


    """A Metropolis-within-Gibbs step method optimized for categorical variables.
       This step method works for Bernoulli variables as well, but it is not
       optimized for them, like BinaryGibbsMetropolis is. Step method supports
       two types of proposals: A uniform proposal and a proportional proposal,
       which was introduced by Liu in his 1996 technical report
       "Metropolized Gibbs Sampler: An Improvement".
    """

    def astep_unif(self, q0, logp):
        dimcats = self.dimcats
        if self.shuffle_dims:
            nr.shuffle(dimcats)

        q = np.copy(q0)
        logp_curr = logp(q)

        for dim, k in dimcats:
            curr_val, q[dim] = q[dim], sample_except(k, q[dim])
            logp_prop = logp(q)
            q[dim] = metrop_select(logp_prop - logp_curr, q[dim], curr_val)
            if q[dim] != curr_val:
                logp_curr = logp_prop

        return q

    def astep_prop(self, q0, logp):
        dimcats = self.dimcats
        if self.shuffle_dims:
            nr.shuffle(dimcats)

        q = np.copy(q0)
        logp_curr = logp(q)

        for dim, k in dimcats:
            logp_curr = self.metropolis_proportional(q, logp, logp_curr, dim, k)

        return q

    def metropolis_proportional(self, q, logp, logp_curr, dim, k):
        given_cat = int(q[dim])
        log_probs = np.zeros(k)
        log_probs[given_cat] = logp_curr
        candidates = list(range(k))
        for candidate_cat in candidates:
            if candidate_cat != given_cat:
                q[dim] = candidate_cat
                log_probs[candidate_cat] = logp(q)
        probs = softmax(log_probs)
        prob_curr, probs[given_cat] = probs[given_cat], 0.0
        probs /= (1.0 - prob_curr)
        proposed_cat = nr.choice(candidates, p = probs)
        accept_ratio = (1.0 - prob_curr) / (1.0 - probs[proposed_cat])
        if not np.isfinite(accept_ratio) or nr.uniform() >= accept_ratio:
            q[dim] = given_cat
            return logp_curr
        q[dim] = proposed_cat
        return log_probs[proposed_cat]

    @staticmethod
    def competence(var):
        '''
        CategoricalGibbsMetropolis is only suitable for Bernoulli and
        Categorical variables.
        '''
        distribution = getattr(
            var.distribution, 'parent_dist', var.distribution)
        if isinstance(distribution, pm.Categorical):
            if distribution.k > 2:
                return Competence.IDEAL
            return Competence.COMPATIBLE
        elif isinstance(distribution, pm.Bernoulli) or (var.dtype in pm.bool_types):
            return Competence.COMPATIBLE
        return Competence.INCOMPATIBLE

def sample_except(limit, excluded):
    candidate = nr.choice(limit - 1)
    if candidate >= excluded:
        candidate += 1
    return candidate

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis = 0)

def delta_logp(logp, vars, shared):
    [logp0], inarray0 = pm.join_nonshared_inputs([logp], vars, shared)

    tensor_type = inarray0.type
    inarray1 = tensor_type('inarray1')

    logp1 = pm.CallableTensor(logp0)(inarray1)

    f = theano.function([inarray1, inarray0], logp1 - logp0)
    f.trust_input = True
    return f
