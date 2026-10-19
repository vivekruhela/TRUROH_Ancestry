"""Firth logistic regression with full-design penalized likelihood-ratio tests.

The restricted fit fixes the tested coefficient within the SAME design, including
its Jeffreys information determinant. Confidence intervals are explicitly Wald
approximations, not profile-likelihood intervals. No ordinary-logistic fallback.
"""
from __future__ import annotations
import math,time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2

class FitFailure(RuntimeError):pass

def objective(beta,X,y):
 eta=X@beta;p=expit(eta);w=np.maximum(p*(1-p),1e-14)
 info=X.T@(w[:,None]*X)
 sign,logdet=np.linalg.slogdet(info)
 if sign<=0:raise FitFailure('Non-positive Fisher information determinant')
 inv=np.linalg.inv(info)
 h=w*np.einsum('ij,jk,ik->i',X,inv,X)
 ll=float(np.sum(y*eta-np.logaddexp(0,eta))+0.5*logdet)
 score=X.T@(y-p+h*(0.5-p))
 return ll,score,inv

def fit(X,y,max_iter=300,tol=1e-5,max_seconds=60,test_index=None):
 X=np.asarray(X,float);y=np.asarray(y,float)
 if not np.isfinite(X).all() or not np.isfinite(y).all():raise FitFailure('Nonfinite model input')
 if set(np.unique(y))!={0.,1.}:raise FitFailure('Outcome must contain both 0 and 1')
 if np.linalg.matrix_rank(X)!=X.shape[1]:raise FitFailure('Rank-deficient design')
 begin=time.monotonic();p=X.shape[1]
 def optimize(start,fixed=None):
  free=np.array([j for j in range(p) if fixed is None or j!=fixed],dtype=int)
  base=np.zeros(p);base[:]=start
  if fixed is not None:base[fixed]=0
  def f(v):
   if time.monotonic()-begin>max_seconds:raise FitFailure('Firth fit timeout')
   b=base.copy();b[free]=v;ll,g,_=objective(b,X,y)
   return -ll,-g[free]
  result=minimize(f,base[free],jac=True,method='BFGS',options={'maxiter':max_iter,'gtol':tol/10})
  b=base.copy();b[free]=result.x;ll,g,vcov=objective(b,X,y)
  residual=float(np.max(np.abs(g[free]))) if len(free) else 0
  converged=bool(residual<=tol and np.isfinite(ll) and np.isfinite(b).all())
  if not converged:raise FitFailure(f'Firth did not converge; max_score={residual:.4g}; {result.message}')
  return b,ll,vcov,int(result.nit),residual
 beta,ll,vcov,niter,residual=optimize(np.zeros(p))
 out={'beta':beta,'vcov':vcov,'loglik_penalized':ll,'iterations':niter,'max_score':residual,'converged':True}
 if test_index is not None:
  _,null_ll,_,null_iter,null_residual=optimize(beta,int(test_index))
  lr=2*(ll-null_ll)
  if lr < -1e-6:raise FitFailure('Restricted penalized likelihood exceeds fitted maximum')
  out.update({'LR':max(0.,lr),'p_value':float(chi2.sf(max(0.,lr),1)),'null_iterations':null_iter,'null_max_score':null_residual})
 return out

def safe_exp(x):return float(np.exp(x)) if x<709 else math.inf
