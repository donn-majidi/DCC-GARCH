import numpy as np
from scipy.optimize import minimize
import warnings

from statsmodels.tools.validation import (string_like,
                                          array_like,
                                          bool_like,
                                          float_like,
                                          int_like,
                                          )

class DCC:
    
    def __init__(self, endog: np.ndarray, corr_matrix: np.ndarray | None = None,
                 dist: str | None = 'normal'):
        
        self._endog = array_like(endog, 'endog', ndim=2)
        self._corr_matrix = array_like(corr_matrix, 'corr_matrix', ndim=2, optional=True)
        self._dist = string_like(dist, 'dist')
        self._nobs, self._ndim = self._endog.shape
        
        if dist not in ('Normal', 'normal'):
            raise NotImplementedError('Currently only the Gaussian Likelihood function is implemented.')
            
        if corr_matrix is not None:
            ## Check if the dimension of the input corr_matrix matches ndim
            if self._corr_matrix.shape != (self._ndim, self._ndim):
                raise ValueError("The dimension of the input correlation matrix and endog "
                                 "don't match.")
            ## Check if the input corr_matrix is self-dual and p.d.
            if not np.allclose(self._corr_matrix, self._corr_matrix.conj().T, rtol=1e-5):
                raise ValueError('Input correlation matrix is not self-dual, therefore '
                                 'it does not qualify as a correlation matrix.')
            if not np.all(np.linalg.eigvalsh(self._corr_matrix) > 0):
                raise ValueError('Input correlation matrix is not positive definite.')
  
        else:
            self._set_corr_matrix()
            
        ## Declare class properties
        self.cond_Q = None
        self.cond_R = None
            
    def _set_corr_matrix(self):
            
            endog = self._endog
            nobs = self._nobs
            
            corr_matrix = endog.T @ endog / nobs
            self._corr_matrix = corr_matrix
            
    def get_recursion(self, params: np.ndarray):
            
        params = array_like(params, 'params', ndim=1)
        a,b = params
        endog = self._endog
        corr_matrix = self._corr_matrix
            
        nobs = self._nobs
        ndim = self._ndim
            
        cond_Q = np.zeros(shape=(nobs,ndim,ndim))
        cond_R = np.zeros(shape=(nobs,ndim,ndim))
                
        cond_Q[0] = corr_matrix
        d = 1/np.sqrt(np.diag(cond_Q[0]))
        cond_R[0] = cond_Q[0] * np.outer(d,d)
            
        for i in range(1,nobs):
                
            eps_1 = endog[i-1]
            Q_1 = cond_Q[i-1]
                
            Q_t = (1-a-b) * corr_matrix + a * np.outer(eps_1,eps_1) + b * Q_1
            d = 1/np.sqrt(np.diag(Q_t))
            R_t = Q_t * np.outer(d, d)
                
            cond_Q[i] = Q_t
            cond_R[i] = R_t
                
        return cond_Q, cond_R
        
    def _loglikelihood(self, params):
        
        a,b = params
        if a < 0 or b < 0 or a + b >= 1:
            return 1e10
        endog = self._endog
        nobs = self._nobs
        _, cond_R = self.get_recursion(params)
            
        n_llf = 0
        for i in range(nobs):
            eps_i = endog[i]
            R_i = cond_R[i]
                
            n_llf_i = np.log(np.linalg.det(R_i)) + eps_i.T @ np.linalg.inv(R_i) @ eps_i - eps_i.T @ eps_i
            n_llf += n_llf_i
            
        n_llf /= (nobs * 2)
        return n_llf
    
    def _composite_loglikelihood(self, params):
        
        a,b = params
        if a < 0 or b < 0 or a + b >= 1:
            return 1e10
        
        endog = self._endog
        nobs = self._nobs
        _, cond_R = self.get_recursion(params)
        i,j = self._pair_indx
        
        rho = cond_R[:,i,j]
        eps_i = endog[:,i]
        eps_j = endog[:,j]
        
        if np.any(np.abs(rho) >= 1):
            return 1e10
        
        r_det = 1 - rho **2
        n_llf = np.log(r_det) + (eps_i**2 - 2*rho * eps_i * eps_j + eps_j**2 )/r_det - (eps_i**2 + eps_j**2)
        n_llf = n_llf.sum()
        n_llf /= (2 * nobs * len(i))
        return n_llf
        
        
    def fit(self, starting_values: np.ndarray | None = None, tol: float | None = None,
            likelihood_method: str | None = 'MLE',  options: dict | None = None,
            method: str | None = 'SLSQP'):
            
        endog = self._endog
        nobs = self._nobs
        ndim = self._ndim
            
        starting_values = array_like(starting_values, 'starting_values', optional=True)
        tol = float_like(tol, 'tol', optional=True)
        likelihood_method = string_like(likelihood_method, 'likelihood_method')
        method = string_like(method, 'method')
            
        if starting_values is not None:
            if starting_values.size != 2:
                raise ValueError("Size of 'starting_values' has to be 2.")
            if np.any(starting_values < 0):
                raise ValueError('Starting values cannot be negative.')
            elif np.any(starting_values > 1):
                raise ValueError('Starting values cannot be greater than one.')
            elif np.sum(starting_values) >= 1:
                raise ValueError('The sum of starting values cannot be greater than one.')
        else:  
            starting_values = np.array([0.05,0.9])
                
        sv = starting_values
        
        if likelihood_method == 'mle':  
            func = self._loglikelihood
        elif likelihood_method == 'mcle':
            self._pair_indx = np.triu_indices(ndim, 1)
            func = self._composite_loglikelihood
        elif likelihood_method == 'mscle':
            i = np.arange(ndim -1)
            j = i + 1
            self._pair_indx = (i,j)
            func = self._composite_loglikelihood
        else:
            raise NotImplementedError('Unrecognized likelihood method.')
        
        bounds = [(0,1), (0,1)]
        constraints = {'type': 'ineq', 'fun': lambda p: 1 - p.sum()}
            
        options = {} if options is None else options
            
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "Values in x were outside bounds during a minimize step",
                RuntimeWarning,
                )
            opt = minimize(
                func,
                sv,
                method=method,
                bounds=bounds,
                constraints=constraints,
                tol=tol,
                options=options,
                )
        
        
        self.cond_Q, self.cond_R = self.get_recursion(opt.x)
        return opt
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                