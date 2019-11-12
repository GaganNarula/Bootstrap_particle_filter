import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.stats import zscore, entropy
from scipy.stats import multinomial, multivariate_normal, gamma
from scipy.signal import butter, filtfilt
from time import time
import joblib
from modelGradfuncs import *
from joblib import Parallel, delayed

pdict = {'init_mu': 15., 'init_cov': 0.5, 'sigma_x' : .4,
        'sigma_y' : .1, 'dt': 1e-2, 'alpha' : 0.4, 'beta' : 1.1,
         'delta': 0.1, 'theta' : -1/300, 'prior_shape': 1.5, 'prior_scale' : 0.1 ,
         'prior_emissionbias_mu': 0., 'prior_emissionbias_var': 1., 'params_update':['theta','alpha','delta']}

def SGLD_step(LL_grad, prior_grad, eps):
    ''' 
    Stochastic Gradient Lagevin gradient step 
    Params:
    -------
        LL_grad : gradient of loglikelihood function
        prior_grad : gradient of log prior function
        eps : step size
    '''
    return eps*(LL_grad + prior_grad) + np.sqrt(2*eps)*np.random.randn()


class BootstrapPF_withLLgradients(object):
    ''' For Linear and Nonlinear state space models. 
    Inference: bootstrap Particle Filtering with ancestor resampling (Andrieu et al 2010)
    Learning: Stochastic gradient Langevin
    
    Params
    ------
        ndim_state : (int) number of hidden state dimensions
        nparticles : (int) number of particles
        param_dict : (dict) a dictionary with all learnable and fixed model parameters
        proposal_func : (function handle) proposal used for BootStrap Filter
        LL_func : (function handle) Likelihood function for emissions
        LL_grad_func : (function handle) gradient of model parameters under data likelihood and proposal
        prior_grad_func : (function handle) gradient of model parameters under prior probability 
        param_keys_update : (list) parameters to update with SGLD, the str names in the list MUST MATCH
                            the keys in param_dict
        control_keys : (list) control variables that may occur in state space model or emission model
        verbose : creates plots of the posterior mean if True
        verbose_every : plots every 'verbose_every' time steps
    ''' 
    def __init__(self, ndim_state = 1, nparticles = 10, param_dict = pdict,
                 proposal_func = linear_proposal3,
                 LL_func = emission_pdf_gauss, 
                 emission_model = emission_meanfunc_proposal3,
                 LL_grad_func = get_grads_linear_proposal3,
                 prior_grad_func = prior_grads_linearproposal3,
                 param_keys_update = ['alpha', 'beta', 'theta', 'delta'],
                 control_keys = ['u[t]','x_prev','yprev'],
                verbose = False, verbose_every = 10):
        
        self.N = nparticles
        self.args = {}
        for k in control_keys:
            self.args[k] = np.nan 
        self.ndim_state = ndim_state
        self.params_update = param_keys_update
        self.num_params = len(param_keys_update)
        self.pdict = param_dict
        assert [p in pdict.keys() for p in params_update]
        self.proposal_func = proposal_func
        self.emission_model = emission_model
        self.LL_func = LL_func
        self.compute_grads = True
        self.LL_grad_func = LL_grad_func
        self.prior_grad_func = prior_grad_func
        self.verbose = verbose
        self.verbose_every = verbose_every
        
    def initialize_particles(self):
        return self.pdict['init_mu'] + self.pdict['init_cov'] * np.random.randn(self.N, self.ndim_state)
    
    def ancestor_resample(self, weights):
        ''' returns index of ancestors for each particle '''
        # for each particle, sample its ancestors based 
        # on the previous time step weights
        p = multinomial.rvs(n=1, p=weights, size=self.N)
        idx = p.argmax(axis=1)
        return idx
    
    def compute_weight_and_state(self, prev_state, y):
        # sample next state from proposal
        x = self.proposal_func(prev_state, self.pdict, self.args)
        # compute likelhood for the current data point under each particle
        # and get unnormalized particle weights
        unnorm_weight = self.LL_func(y, x, self.emission_model, self.pdict, self.args)
        # compute gradients of parameters 
        grads = self.LL_grad_func(y, x, self.pdict, self.args) 
        return x, unnorm_weight, grads
        
    def inference(self, y, controls):
        ''' Sequential Importance Sampling with ancestor resampling
        '''
        # time steps
        T = len(y)
        # state of each particles [timesteps x Nparticles x state_dimension]
        x = np.zeros((T, self.N, self.ndim_state))
        ancestors = np.zeros((T, self.N), dtype='int')
        weights = np.zeros((T, self.N))
        
        # initialize weights
        weights0 = (1 / self.N) * np.ones(self.N)
        # initialize first state
        x0 = self.initialize_particles()
        # cumulative gradients for parameters
        cumulative_grads = np.zeros((self.N, self.num_params))
        # posterior mean
        x_mu = np.zeros((T,self.ndim_state))
        # posterior variance
        x_var = np.zeros((T, self.ndim_state))
        #running_ll = 0.
        
        yprev = 0.
         
        for t in range(T):
            start = time()
            # resample ancestors
            if t == 0:
                ancestors[t] = self.ancestor_resample(weights0)
            else:
                ancestors[t] = self.ancestor_resample(weights[t-1])
            # get previous state for all particles
            if t == 0:
                x_prev = x0[ancestors[t]]
            else:
                x_prev = x[t-1,ancestors[t]]
            
            # For each particle, get its new state, then compute its weights and gradients
            unnorm_weights = np.zeros(self.N)
            
            for i in range(self.N):
                self.args['yprev'] = yprev
                self.args['u[t]'] = controls['u'][t]
                self.args['x_prev'] = x_prev[i]
                x[t,i], unnorm_weights[i], grads = self.compute_weight_and_state(x_prev[i], y[t])
                for j in range(self.num_params):
                    cumulative_grads[i,j] = cumulative_grads[ancestors[t,i],j] + grads[j]
                        
            # re-normalize weights
            if unnorm_weights.sum()==0:
                pdb.set_trace()
            weights[t] = unnorm_weights / unnorm_weights.sum()
            
            # post. mean
            x_mu[t] = np.dot(weights[t], x[t].squeeze())
            # post.var
            #x_var[t] = (1/self.N-1) * np.dot(weights[t],(x[t].squeeze() - x_mu[t])**2)  
            x_var[t] = (1/self.N-1) * np.sum((x[t].squeeze() - x_mu[t])**2)  
            
            # compute data Likelihood under x_mu
            #ll = np.log(self.LL_func(y[t], x_mu[t], self.emission_model, self.pdict))
            #running_ll += ll
            
            if self.verbose and t%self.verbose_every==self.verbose_every-1:
                plt.figure(figsize=(15,4))
                #plt.bar(x=np.arange(len(weights[t])), height = weights[t], width = 0.8)
                plt.plot(np.arange(t), x_mu[:t])
                plt.fill_between(np.arange(t), y1=x_var[:t].squeeze(), 
                                 y2=-x_var[:t].squeeze(), alpha=0.4)
                plt.show()
                end = time()
                print('Time step %d/%d, running LL : %.4f'%(t,T,running_ll))
                print('Time taken %.2f secs'%(end-start))
            
            yprev = y[t]
            
        # done with iteration over time
        H = np.zeros(self.num_params)
        for j in range(self.num_params):
            H[j] = np.dot(weights[-1], cumulative_grads[:,j].squeeze())
        return x, x_mu, H, weights
    
    def init_learning_rate(self, n_iter, r = 0.01, g = 2., gamm = 0.9):
        ''' 
        Initialzes the stepsize (learning rate) sequence 
        for each SGLD iteration. The function below produces exponentially 
        reducing stepsize.
        Params
        ------
            n_iter : number of SGLD iterations
            r  :  scaling param
            g  :  initial value
            gamm : decay rate
        Returns
        ------
            eps : (np.array) 
        '''
        tim = np.arange(n_iter)
        # eps is learning rate , decays exponentially in time
        eps = r * ((g + tim)**(-gamm))
        return eps
    
    def fit_params(self, y, controls, n_iter = 100, r = 0.005, g = 2., gamm = 0.9):
        ''' Do SGLD as outer loop and Particle Filtering as inner loop '''
        # learning rate decay 
        print('\n Starting SGLD with PF ')
        eps = self.init_learning_rate(n_iter, r, g, gamm)
        print('initial learning rate = %.4f \n'%(eps[0]))
        # state sequences for each particle
        state_seqs = [None for _ in range(n_iter)]
        # particle weighted posterior mean sequence
        state_mean = [None for _ in range(n_iter)]
        for k in range(n_iter):
            # get state sequence + gradients by running particle filter O(N * T * ndim_state**2 time)
            state_seqs[k], state_mean[k], grad, w = self.inference(y, controls)
            e = entropy(w[-1], base=2)
            # get prior gradients
            prior_grad_list = self.prior_grad_func(self.pdict)
            prior_grads = np.array(prior_grad_list)
            all_steps = []
            for j, key in enumerate(self.params_update):
                stp = SGLD_step(grad[j], prior_grads[j], eps[k])
                all_steps.append(stp)
                self.pdict[key] += stp
            print('... Likelihood gradient norm = %.3f .....'%(np.linalg.norm(grad)))
            print('... Prior gradient norm = %.3f .....'%(np.linalg.norm(prior_grads)))
            print('... SGLD gradient step norm = %.3f .... '%(np.linalg.norm(np.array(all_steps))))
            print('... Entropy of weights at last step (bits) = %.3f .....\n'%(e))
        return state_seqs, state_mean
