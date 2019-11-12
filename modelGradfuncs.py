import numpy as np

def ursino_proposal(xprev, u_t, pdict):
    dP = pdict['alpha']*xprev*(u_t - pdict['beta']*xprev)*pdict['dt'] + pdict['sigma_x']*np.random.randn()
    return xprev + dP 

def linear_proposal2(xprev, u_t, pdict):
    dP = pdict['alpha']*u_t - pdict['beta']*xprev + pdict['sigma_x']*np.random.randn()
    return dP

def linear_proposal3(xprev, pdict, control_dict):
    u_t = control_dict['u[t]']
    dP = pdict['alpha']*u_t - xprev + pdict['sigma_x']*np.random.randn()
    return dP*pdict['dt']

def linear_proposal4(xprev, pdict, control_dict):
    u_t = control_dict['u[t]']
    xnew = xprev + (pdict['alpha']*u_t - pdict['beta']*xprev)*pdict['dt'] + \
    (pdict['dt']**0.5)*pdict['sigma_x']*np.random.randn()
    return xnew

#def emission_pdf_gauss(y_t, x_t, emission_meanfunc, pdict):
#    return multivariate_normal(mean = emission_meanfunc(x_t, pdict), cov=pdict['sigma_y']**2).pdf(y_t)

def emission_pdf_gauss(y_t, x_t, emission_meanfunc, pdict, emission_args):
    Z = (1 / (np.sqrt(2 * np.pi)*pdict['sigma_y']))
    e = np.exp(- (1 / (2*pdict['sigma_y']**2)) * (y_t - emission_meanfunc(x_t, 
                                                                          pdict,
                                                                          emission_args))**2)
    return Z * e

def emission_meanfunc_proposal3(x_t, pdict, *args):
    return pdict['theta']*x_t + pdict['delta']

def emission_meanfunc_proposal4(x_t, pdict, arterm):
    yprev = arterm['yprev']
    return (1 - pdict['theta'])*yprev + pdict['theta']*x_t + pdict['delta']



############################# GRADIENT FUNCTIONS ####################################


def get_grads_linear_proposal2(y_t, x_t, x_prev, u_t, pdict):
    grad_theta = (y_t - pdict['theta']*x_t)*x_t / pdict['sigma_y']**2
    grad_alpha = (x_t - (pdict['alpha']*u_t - pdict['beta']*x_prev))*u_t / pdict['sigma_x']**2
    grad_beta = -(x_t - (pdict['alpha']*u_t - pdict['beta']*x_prev))*x_prev / pdict['sigma_x']**2
    return grad_theta, grad_alpha, grad_beta

def get_grads_linear_proposal3(y_t, x_t, pdict, other):
    x_prev = other['x_prev']
    u_t = other['u[t]']
    grad_theta = (y_t - pdict['theta']*x_t - pdict['delta'])*x_t / pdict['sigma_y']**2
    grad_alpha = (x_t - (pdict['alpha']*u_t*pdict['dt'] - x_prev*pdict['dt']) )*u_t*pdict['dt'] / pdict['sigma_x']**2
    grad_delta = (y_t - pdict['theta']*x_t - pdict['delta'])/ pdict['sigma_y']**2
    return grad_theta, grad_alpha, grad_delta


def get_grads_linear_proposal4(y_t, x_t, pdict, other):
    x_prev = other['x_prev']
    u_t = other['u[t]']
    yprev = other['yprev']
    
    inner_term = x_t - (x_prev + (pdict['alpha']*u_t - pdict['beta']*x_prev)*pdict['dt']) 
    grad_alpha = inner_term * u_t * pdict['dt'] / pdict['sigma_x']**2
    grad_beta = -inner_term * x_prev * pdict['dt'] / pdict['sigma_x']**2
    
    inner_term = (y_t - ((1-pdict['theta'])*yprev + pdict['theta']*x_t + pdict['delta']))                    
    grad_theta = inner_term * (x_t - yprev) / pdict['sigma_y']**2
    grad_delta = inner_term / pdict['sigma_y']**2
    return grad_alpha, grad_beta, grad_theta, grad_delta

def get_grads_ursino_proposal(y_t, x_t, x_prev, u_t, pdict):
    # NOT IMPLEMENTED
    return


################################ PRIOR PROBABILITY GRADIENT FUNCTIONS #############################
def loggamma_prior_grad(x, prior_shape, prior_scale):
    return ((prior_shape - 1) / x) - (1 / prior_scale) 

def loggauss_prior_grad(x, prior_mu, prior_var):
    return (-1 / prior_var) * (x - prior_mu)

def prior_grads_linearproposal2(pdict):
    grad_alpha = loggamma_prior_grad(pdict['alpha'], pdict['prior_shape'], pdict['prior_scale'])
    grad_beta = loggamma_prior_grad(pdict['beta'], pdict['prior_shape'], pdict['prior_scale'])
    grad_theta = loggamma_prior_grad(pdict['theta'], pdict['prior_shape'], pdict['prior_scale'])
    return grad_theta, grad_alpha, grad_beta

def prior_grads_linearproposal3(pdict):
    grad_alpha = loggamma_prior_grad(pdict['alpha'], pdict['prior_shape'], pdict['prior_scale'])
    grad_theta = loggamma_prior_grad(pdict['theta'], pdict['prior_shape'], pdict['prior_scale'])
    grad_delta = loggauss_prior_grad(pdict['delta'], pdict['prior_emissionbias_mu'], pdict['prior_emissionbias_var'])
    return grad_theta, grad_alpha, grad_delta

def prior_grads_linearproposal4(pdict):
    grad_alpha = loggamma_prior_grad(pdict['alpha'], pdict['prior_shape'], pdict['prior_scale'])
    grad_beta = loggamma_prior_grad(pdict['beta'], pdict['prior_shape'], pdict['prior_scale'])
    grad_theta = loggamma_prior_grad(pdict['theta'], pdict['prior_shape'], pdict['prior_scale'])
    grad_delta = loggauss_prior_grad(pdict['delta'], pdict['prior_emissionbias_mu'], pdict['prior_emissionbias_var'])
    return grad_alpha, grad_beta, grad_theta, grad_delta