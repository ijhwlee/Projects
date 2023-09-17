import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import tensorflow as tf
from time import time
from numpy import random
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from genetic_algorithm230915 import *

#ij2102 result
# numerically solving H(z), Omega_i(z)
# n = 1, k = 0,+1,-1 case
from scipy.integrate import solve_ivp
from scipy.special import gamma, airy
import matplotlib.pyplot as plt
import numpy as np
import math
from dynesty import NestedSampler
from dynesty import utils as dyfunc

Omega_k_idx = 0
Omega_L_idx = 1
H0_idx = 2
b2_idx = 3
c_idx = 4
n_idx = 5
H_bias_idx = 6

def copy_data(src, dst):
    if len(dst) > 0:
        dst.clear()
    for idx in range(len(src)):
        dst.append(src[idx])
    return

#some global variables
shift_lower = 1.0
shift_upper = 1.5
train_x = []
train_y = []
train_y_shift = []
train_sigma = []
def set_train_data(x, y, z):
    global train_x, train_y, train_y_shift, train_sigma
    train_x = np.array(x)
    train_y = np.array(y)
    train_y_shift = np.array(y)
    train_sigma = np.array(z)
    #print("[DEBUG-hwlee]Bayesian.py: vary_index = {}".format(vary_index))
    return
#minimum logL value for error case
logL_min = -1.0e300

# some global parameters
th_min = []
th_max = []
th_fixed = []
variants = []
vary_index = []
delz = 0.5
global_k = 0
Omega_m_backward = 0
regularization = 0.001 # regularization for Ridge and Lasso
def set_global_params(fixed, indices, varies, beta):
    global th_fixed, vary_index, variants, regularization
    copy_data(fixed, th_fixed)
    copy_data(indices, vary_index)
    copy_data(varies, variants)
    regularization = beta
    #print("[DEBUG-hwlee]Bayesian.py: vary_index = {}".format(vary_index))
    return

# x => Omega_Lambda, y => Omega_k, c
def eos_Lambda(x, y, c):
    #if x < 0 or (1-c*c*y/x) < 0:
    #  print("x = {}, y = {}, c = {}, 1-c*c*y/x = {}".format(x, y,c, 1-c*c*y/x))
    v = -1.0/3.0 -2/(3*c)*math.sqrt(x)*math.sqrt(1-c*c*y/x)
    return v

# x => Omega_Lambda, y => Omega_k, b2, n
def eos_m(x, y, b2, n):
    #if x < 0 or 1+y < 0:
    #  print(" x = {}, y = {}, 1+y = {}".format(x,y, (1+y)))
    if x == 0 :
        return 0 # constant
    v = -b2/math.pow(x, n-1)*math.pow((1+y), n)/(1-x+y)
    return v
#differential equation dOmega[0,1,2]=f[0,1,2](x, Omega[0,1,2]), Omega[0] => Omega_k, Omeag[1]= > Omega_Lambda
#                                                               Omega[2] => H, Omega[3] => b2, Omega[4] => c, Omega[5] => n
def f_forward(x, Omega):
    dOmega0 = Omega[0]*(1+Omega[0])*(1+3*eos_Lambda(Omega[1], Omega[0], Omega[4])) - 3*Omega[0]*(1-Omega[1]+Omega[0])*(eos_Lambda(Omega[1], Omega[0], Omega[4])-eos_m(Omega[1], Omega[0], Omega[3], Omega[5]))
    dOmega1 = Omega[0]*Omega[1]*(1+3*eos_Lambda(Omega[1], Omega[0], Omega[4])) - 3*Omega[1]*(1-Omega[1]+Omega[0])*(eos_Lambda(Omega[1], Omega[0], Omega[4])-eos_m(Omega[1], Omega[0], Omega[3], Omega[5]))
    dOmega2 = Omega[2]*(-1.5*(1+eos_m(Omega[1], Omega[0], Omega[3], Omega[5]))*(1-Omega[1]+Omega[0])
                              -1.5*(1+eos_Lambda(Omega[1], Omega[0], Omega[4]))*Omega[1]+Omega[0])
    dOmega3 = 0
    dOmega4 = 0
    dOmega5 = 0
    return [dOmega0, dOmega1, dOmega2, dOmega3, dOmega4, dOmega5]
def f_backward(x, Omega):
    dOmega0 = Omega[0]*(1+Omega[0])*(1+3*eos_Lambda(Omega[1], Omega[0], Omega[4])) - 3*Omega[0]*(1-Omega[1]+Omega[0])*(eos_Lambda(Omega[1], Omega[0], Omega[4])-eos_m(Omega[1], Omega[0], Omega[3], Omega[5]))
    dOmega1 = Omega[0]*Omega[1]*(1+3*eos_Lambda(Omega[1], Omega[0], Omega[4])) - 3*Omega[1]*(1-Omega[1]+Omega[0])*(eos_Lambda(Omega[1], Omega[0], Omega[4])-eos_m(Omega[1], Omega[0], Omega[3], Omega[5]))
    dOmega2 = Omega[2]*(-1.5*(1+eos_m(Omega[1], Omega[0], Omega[3], Omega[5]))*(1-Omega[1]+Omega[0])
                        -1.5*(1+eos_Lambda(Omega[1], Omega[0], Omega[4]))*Omega[1]+Omega[0])
    dOmega3 = 0
    dOmega4 = 0
    dOmega5 = 0
    return [-dOmega0, -dOmega1, -dOmega2, -dOmega3, -dOmega4, -dOmega5]
# linear fit for the given z value
def find_value(z_value, z_backward, h_of_z):
    for idx in range(len(z_backward)-1):
        z0 = z_backward[idx]
        z1 = z_backward[idx+1]
        if z_value >= z0 and z_value <z1:
            r = (z_value-z0)/(z1-z0)
            #print("z = {0}, idx = {1}, r = {2}, z0 = {3}, z1 = {4}".format(z_value, idx, r, z0, z1))
            h_z = r*(h_of_z[idx+1]-h_of_z[idx])+h_of_z[idx]
            #print("z = {0}, idx = {1}, r = {2}, h0 = {3}, h1 = {4}, hz = {5}".format(z_value, idx, r, h_of_z[idx], h_of_z[idx+1], h_z))
            return h_z
    if z_value == z_backward[len(z_backward)-1]:
      h_z = h_of_z[len(z_backward)-1]
      return h_z
    else: # extrapolate
      z0 = z_backward[len(z_backward)-2]
      z1 = z_backward[len(z_backward)-1]
      r = (z_value-z0)/(z1-z0)
      #print("Extrapolate: z_value = {}, z1 = {}, z0 = {}".format(z_value, z1, z0))
      h_z = r*(h_of_z[len(z_backward)-1]-h_of_z[len(z_backward)-2])+h_of_z[len(z_backward)-2]
      return h_z

def H_of_z(z_values, th0):
    #print("[DEBUG-hwlee]z_values[{0}] : {1}".format(len(z_values),z_values))
    zp1_values = [1.0+z for z in z_values]
    x_values = np.log(zp1_values)
    x_max = np.max(x_values)
    #print("[DEBUG-hwlee]x_values[{0}] : {1}".format(len(x_values), x_values))
    #print("[DEBUG-hwlee]x_max = {}".format(x_max))
    #Omega_0 = [Omega0_0, Omega1_0, Omega2_0, Omega3_0]
    rtol, atol = (1e-8, 1e-8)
    x_span = [0,x_max]
    sol_backward = solve_ivp(f_backward, x_span, th0, rtol=rtol, atol=atol)
    Omega_m_backward = 1.0 - sol_backward.y[1] + sol_backward.y[0]
    z_backward = np.exp(sol_backward.t) -1
    #print("[DEBUG-hwlee]z_backward : {}".format(z_backward))
    #print("[DEBUG-hwlee]sol_backward.t : {}".format(sol_backward.t))
    #print("[DEBUG-hwlee]sol_backward.y[2] : {}".format(sol_backward.y[2]))
    h_of_z = []
    for idx in range(len(z_values)):
        z0 = z_values[idx]
        h_z = find_value(z0, z_backward, sol_backward.y[2])
        h_of_z.append(h_z)
    #print("[DEBUG-hwlee]h_of_z : {}".format(h_of_z))
    return h_of_z

# define error, gradient and Hessian functions to find optimal value with gradient descent method
# 예측함수
#def f(x):
#    return np.dot(x, theta)
#default delta theta value
del_th = 1.0e-3
# 목적함수, x = red shift values, y = H(z) values, z = sigma of H(z), th0 = initial parameters for numerical integration
def E(x, y,z, th0):
    return 0.5 * np.sum((y - H_of_z(x, th0)) ** 2/z**2)
def chi_square(x, y, th0):
    #chi2 = np.sum((y - H_of_z(x, th0)) ** 2/(H_of_z(x, th0)**2))
    chi2 = np.sum((y/H_of_z(x, th0) - 1) ** 2)
    return chi2, chi2/(len(x)-1)
def chi_square_sigma(x, y, z, th0, params):
    #chi2 = np.sum((y - H_of_z(x, th0)) ** 2/(H_of_z(x, th0)**2))
    chi2 = np.sum((y - H_of_z(x, th0)) ** 2/z**2)
    return chi2, chi2/(len(x)-params)
# gradient function, vary = variant control parameters of initial values for machine elarning
def grad_E(x, y, z, th0, vary, k0):
    delE_th = []
    #print("[DEBUG-hwlee]th0 = {0}".format(th0))
    for i in range(len(th0)):
      #print("[DEBUG-hwlee]i = {0}, vary[i] = {1}".format(i, vary[i]))
      if vary[i] == 1:
        th1 = []
        for j in range(len(th0)):
          th1.append(th0[j])
          if j==i:
            th1[j] = th1[j] + del_th
        #Omega_0 = [th1[0], th1[1], th1[2], th1[3], th1[4], th1[5]]
        if i==2: # H0 changed
          th1[0] = 110.8*k0/(th1[2]**2) # change Omega_k also
        E1 = E(x, y, z, th1)
        #print("[DEBUG-hwlee]th1 = {0}, E1 = {1}, H(z) = {2}".format(Omega_0, E1, H_of_z(train_x, th1)))
        #Omega_0 = [th0[0], th0[1], th0[2], th0[3], th0[4], th0[5]]
        E0 = E(x, y, z, th0)
        #print("[DEBUG-hwlee]th0 = {0}, E0 = {1}, H(z) = {2}".format(Omega_0, E0, H_of_z(train_x, th0)))
        delE_th.append((E1-E0)/del_th)
      else:
        delE_th.append(0)
    return np.array(delE_th)
#calculate Hessian, x : observed z-values, y:observed H(z) values, z:observed sigma values, th0:center patameters, k0:spatial curvature
def Hessian_E(x, y, z, th0, k0, del_th, vary):
    hessian_th = []
    std_th = []
    E0 = E(x, y, z, th0)
    #print("th0 = {}".format(th0))
    for i in range(len(th0)):
      #print("i = {0}, vary[i] = {1}".format(i, vary[i]))
      if vary[i] ==1:
        th1 = []
        th2 = []
        for j in range(len(th0)):
          th1.append(th0[j])
          if j==i:  #change only ith parameter
            th1[j] = th1[j] + del_th
          if j==2: # H0 changed
            th1[0] = 110.8*k0/(th1[2]**2) # change Omega_k also
          th2.append(th0[j])
          if j ==i: # change only ith parameter
            th2[j] = th2[j] - del_th
          if j==2: # H0 changed
            th2[0] = 110.8*k0/(th2[2]**2) # change Omega_k also
        #print("th1 = {}".format(th1))
        #print("th2 = {}".format(th2))
        E1 = E(x, y, z, th1)
        E2 = E(x, y, z, th2)
        std_th.append(np.sqrt((del_th*del_th)/(2*(E1+E2-2*E0))))
      else:
        E1 = E0
        E2 = E0
        std_th.append(0)
      print("E0 = {}, E1 = {}, E2 = {}, E1+E2-2*E0 = {}, del_th = {}".format(E0, E1, E2, (E1+E2 -2*E0), del_th))
      hessian_th.append((E1+E2-2*E0)/(del_th*del_th))
    return np.array(hessian_th), np.array(std_th)

#Bayesian sampling assuming uniform prior
#generate a random smaple, th[0] = Omega_k, th[1] = Omega_L, th[2] = H0, th[3] = b2, th[4] = c, th[5] = n
def random_sample(th_min, th_max):
  found = 0
  while( not found):
    th = []
    for idx in range(len(th_min)):
      th0 = th_min[idx] + np.random.rand()*(th_max[idx] - th_min[idx])
      th.append(th0)
    found = check_parameter(th)
  #print("[DEBUG-hwlee]random_sample: value = {}, th = {}".format((1 - th[4]*th[4] * th[0]/th[1]),th))
  return np.array(th)
#check validity of parameters 1 - c^2*Omega_k/Omega_L > 0
def check_parameter(th):
  if th[1] != 0:
    check = (1 - th[4]*th[4] * th[0]/th[1] > 0) and (th[1] <= 1 + th[0]) and np.abs(th[0]) <= 1 and th[1] <= 1
  else:
    check = (th[1] <= 1 + th[0]) and np.abs(th[0]) <= 1 and th[1] <= 1
  return check

#direct sampler N = number of total sample, th_min = lower boundary of th, th_max + upper boundary of th
# accept th if log(r) < -0.5error(th) - 0.5error(th_opt) = -0.5*error(th) -5
# this method is very inefficient, its acceptance rate is less than 0.1%
def direct_sampler(N, th_min, th_max, train_x, train_y, train_sigma, nprint):
  th_list = []
  count = 0
  total = 0
  maxLogL = -5.0 # -0.5 * log(L(th_optimal)) = -0.5*error(th_optimal)
  while(1):
    th1 = random_sample(th_min, th_max)
    total = total + 1
    try:
      logL = -0.5*E(train_x, train_y, train_sigma, th1)
    except:
      #print("[DEBUG-hwlee]direct_sampler: Error for error calculation th = {}".format(th1))
      continue
    r = np.random.rand()
    if np.log(r) < logL - maxLogL:
      th_list.append(th1)
      count = count + 1
      if count%nprint == 0:
        print("Count = {}, Error = {}, th={}".format(count, -2*logL, th1))
      if count >= N:
        break
  print("Acceptance ratio = {}".format(count/total))
  return np.array(th_list)

# sampling using Dynesty sampler
#from dynesty import NestedSampler
#from dynesty import utils as dyfunc
# assuming uniform prior for simplicity, delz is the convergence creterior for Dynesty sampler, n MAX iteration non limit if zero
#th_min = []
#th_max = []
def Dynesty_sampler(logLikelihood, th_min0, th_max0, train_x0, train_y0, train_sigma0, delz0, live_points, N, multiple):
  # Define the dimensionality of our problem.
  global delz, th_min, th_max, train_x, train_y, train_sigma
  copy_data(th_min0, th_min)
  copy_data(th_max0, th_max)
  train_x = np.array(train_x0)
  train_y = np.array(train_y0)
  train_sigma = np.array(train_sigma0)
  ndim = len(th_min0)
  rlist = []
  delz = delz0
  sampler = NestedSampler(logLikelihood, prior_transform, ndim, nlive=live_points, bound='none')
  for iter in range(multiple):
      # start our run
      if N <= 0:
        sampler.run_nested(dlogz=delz)
      else:
        sampler.run_nested(dlogz=delz, maxiter=N)
      res1 = sampler.results
      rlist.append(res1)
      sampler.reset()
    
  #Merge into a single run.
  results = dyfunc.merge_runs(rlist)
  return results

# prior transform function, u is a list of random number between 0 and 1  
def prior_transform(u):
    """Transforms the uniform random variables `u ~ Unif[0., 1.)`
    to the parameters of interest."""

    x = np.array(u)  # copy u
    # uniform prior for each parameters of range [th_min(i), th_max(i)]
    for idx in range(len(u)):
      x[idx] = (th_max[idx] - th_min[idx])*u[idx] + th_min[idx]
    #print("u = {}".format(u))
    #print("x = {}".format(x))
    while(1):
      if check_parameter(x) :
        break
      # uniform prior for each parameters of range [th_min(i), th_max(i)]
      for idx in range(len(u)):
        x[idx] = (th_max[idx] - th_min[idx])*np.random.rand() + th_min[idx]

    return x

# log likelihood function for interacting dark energy model
# log likelihood function, th is the list of starting parameters
# th[0] = Omega_k, th[1] = Omega_L, th[2] = H0, th[3] = b2, th[4] = c, th[5] = n
# train_x, train_y, train_sigma are global variables
def loglike(th):
  try:
    logL = -E(train_x, train_y, train_sigma, th)
  except Exception as e:
    #logL = -np.log(0.01*sys.float_info.max) # this is not valid solution, that is not allowed cosmological evolution
    logL = logL_min*(1 + np.random.rand())
    #print(e)
  return logL

def shift_H(bias):
  global train_x, train_y, train_y_shift, train_sigma, shift_lower, shift_upper
  train_y_shift = np.array(train_y)
  for idx in range(len(train_x)):
    if train_x[idx] >= shift_lower and train_x[idx] <= shift_upper:
      train_y_shift[idx] = (train_y[idx] - bias)
    else:
      train_y_shift[idx] = train_y[idx]
  return

# log likelihood function for LCDM model
# th[0] = Omega_k, th[1] = Omega_L, th[2] = H0, th[3] = b2, th[4] = c, th[5] = n
# th_fixed = [0.0, Omega_L, H0, 0, 1, 1, 0]
def loglikeLCDM(th):
    global train_x, train_y, train_sigma
    return -0.5 * np.sum((train_y - th[2]*((1+th[0]-th[1])*(1+train_x)**3 + th[1])**(0.5)) ** 2/train_sigma**2)

# log likelihood function for friedmann model
# th[0] = Omega_k, th[1] = Omega_L, th[2] = H0, th[3] = b2, th[4] = c, th[5] = n
# Omega_m = 1 + Omega_k - Omega_L
# th_fixed = [-0.7, 0, H0, 0, 1, 1]
def loglikeFriedmann(th):
    global train_x, train_y, train_sigma
    return -0.5 * np.sum((train_y - th[2]*((1+th[0]-th[1])*(1+train_x)**3 - th[0]*(1+train_x)**2)**(0.5)) ** 2/train_sigma**2)

# log likelihood function for wCDM
# th[0] = Omega_k, th[1] = Omega_L, th[2] = H0, th[3] = b2, th[4] = c, th[5] = n
# th[3] = wde in this case
# th_fixed = [0.0, Omega_L, H0, w_de, 1, 1]
def loglikewCDM(th):
    global train_x, train_y, train_sigma
    return -0.5 * np.sum((train_y - th[2]*((1+th[0]-th[1])*(1+train_x)**3 + th[1]*(1+train_x)**(3*(1+th[3])))**(0.5)) ** 2/train_sigma**2)

# log likelihood function for Rh
# th[0] = Omega_k, th[1] = Omega_L, th[2] = H0, th[3] = b2, th[4] = c, th[5] = n
# th_fixed = [0.0, 0.0, H0, 0, 1, 1]
def loglikeRh(th):
    global train_x, train_y, train_sigma
    return -0.5 * np.sum((train_y - th[2]*(1+train_x)) ** 2/train_sigma**2)

#show parameter definitions for models
def show_params():
    param_names = {"Interacting":['Omega_k', 'Omega_L', 'H0', 'b2', 'c', 'n'],
                   "LCDM":['Omega_k', 'Omega_L', 'H0', 'NA', 'NA', 'NA'],
                   "Friedmann":['Omega_k', 'Omega_L', 'H0', 'NA', 'NA', 'NA'],
                   "wCDM":['Omega_k', 'Omega_L', 'H0', 'w_de', 'NA', 'NA'],
                   "Rh":['NA', 'NA', 'H0', 'NA', 'NA', 'NA']
                  }
    print("Parameter definitions = {}".format(param_names))
    return

#H_lacdm=68.2*(0.32*(1+x)**3+0.68)**(0.5)
#H_friedmann=57.4*(0.3*(1+x)**3+0.7*(1+x)**2)**(0.5)
#H_wCDM=70.8*(0.32*(1+x)**3+0.68*(1+x)**(3*-0.25))**(0.5)
#H_Rh = 61.8*(1+x)

#find the maximum likelihood values
# input : th_samples = Dynesty results
def findMaxL(th_samples):
    print("Samples shape = {}".format(th_samples['samples'].shape))
    maxL = np.max(th_samples['logl'])
    #print("maxL = {}".format(maxL))
    maxLindices = np.where(th_samples['logl'] == maxL)
    maxLindex = maxLindices[0]
    #print("Indices = {}, maxIndex = {}".format(maxLindices, maxLindex))
    th_maxL = th_samples['samples'][maxLindex, :][0]
    print("Maximum Likelihood Parameters = {}, maxl = {}".format(th_maxL, maxL))
    print("Error value for maxL parameters = {}".format(E(train_x, train_y, train_sigma, th_maxL)))
    return maxL, maxLindex, th_maxL

# log likelihood function for LCDM model
# th[0] = Omega_k, th[1] = Omega_L, th[2] = H0, th[3] = b2, th[4] = c, th[5] = n, th[6] = shift
# th_fixed = [0.0, Omega_L, H0, 0, 1, 1, 0]
def loglikeLCDM_restricted_shift(th0):
  global train_x, train_y, train_y_shift, train_sigma
  global vary_index, th_fixed
  th = np.copy(th_fixed)
  vary_params = len(th0)
  train_y_shift = np.array(train_y)
  shift_H(th_fixed[H_bias_idx])
  for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
    if vary_params == len(th):
      th[idx] = th0[idx]
      if idx == H_bias_idx:
        shift_H(th0[idx])
    else:
      th[vary_index[idx]] = th0[idx]
      if vary_index[idx] == H_bias_idx:
        shift_H(th0[idx])
  return -0.5 * np.sum((train_y_shift - th[2]*((1+th[0]-th[1])*(1+train_x)**3 - th[0]*(1+train_x)**2 + th[1])**(0.5)) ** 2/train_sigma**2)

# log likelihood function for LCDM model
#H_lacdm=68.2*(0.32*(1+x)**3+0.68)**(0.5)
# th[0] = Omega_k, th[1] = Omega_L, th[2] = H0, th[3] = b2, th[4] = c, th[5] = n
# variants = [1, 1, 1, 0, 0, 0] Omega_L and H0 vary including Omega_k
# th_fixed = [0.0, Omega_L, H0, 0, 1, 1]
def loglikeLCDM_restricted(th0):
  global train_x, train_y, train_sigma
  th = np.copy(th_fixed)
  vary_params = len(th0)
  for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
    if vary_params == len(th):
      th[idx] = th0[idx]
    else:
      th[vary_index[idx]] = th0[idx]
  return -0.5 * np.sum((train_y - th[2]*((1+th[0]-th[1])*(1+train_x)**3 - th[0]*(1+train_x)**2 + th[1])**(0.5)) ** 2/train_sigma**2)

# log likelihood function for ML None model
#H_ML_none = Sum_{i=0}^6 w_i * z^i
# th[0] = Omega_k, th[1] = Omega_L, th[2] = H0, th[3] = b2, th[4] = c, th[5] = n, th[6] = H_bias, 
# th[7] = a, th[8] = b, th[9] = c, th[10]=d, th[11] = H0, th[12] = H1
# th[13] = w0, th[14] = w1, th[15] = w2, th[16] = w3, th[17] = w4, th[18] = w5, th[19] = w6
# variants = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1] w0 ~ w6 are varying
# th_fixed = [0.0, 0, 0, 0, 1, 1, 0, 8.55^2, 19.1, 30.99, 4.88, 79, 92.3, ...] ??hwlee
def loglikeML_LeastSquare_restricted(th0):
  global train_x, train_y, train_sigma, regularization
  th = np.copy(th_fixed)
  vary_params = len(th0)
  for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
    if vary_params == len(th):
      th[idx] = th0[idx]
    else:
      th[vary_index[idx]] = th0[idx]
  return -0.5 * np.sum((train_y - (th[13] + th[14]*train_x + th[15]*train_x**2 + 
        th[16]*train_x**3 + th[17]*train_x**4 + th[18]*train_x**5 + th[19]*train_x**6)) ** 2/train_sigma**2)

def loglikeML_None_restricted(th0):
  global train_x, train_y, train_sigma, regularization
  th = np.copy(th_fixed)
  vary_params = len(th0)
  for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
    if vary_params == len(th):
      th[idx] = th0[idx]
    else:
      th[vary_index[idx]] = th0[idx]
  return -0.5 * np.sum(((train_y - (th[13] + th[14]*train_x + th[15]*train_x**2 + 
        th[16]*train_x**3 + th[17]*train_x**4 + th[18]*train_x**5 + th[19]*train_x**6))/train_sigma) ** 2)

def loglikeML_Ridge_restricted(th0):
  global train_x, train_y, train_sigma, regularization
  th = np.copy(th_fixed)
  vary_params = len(th0)
  for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
    if vary_params == len(th):
      th[idx] = th0[idx]
    else:
      th[vary_index[idx]] = th0[idx]
  penalty = 0
  for idx in range(7):
    penalty += th[13+idx]*th[13+idx]
  penalty = regularization * penalty
  return -0.5 * (np.sum(((train_y - (th[13] + th[14]*train_x + th[15]*train_x**2 + 
       th[16]*train_x**3 + th[17]*train_x**4 + th[18]*train_x**5 + th[19]*train_x**6))/train_sigma) ** 2) + penalty)

def loglikeML_Lasso_restricted(th0):
  global train_x, train_y, train_sigma, regularization
  th = np.copy(th_fixed)
  vary_params = len(th0)
  for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
    if vary_params == len(th):
      th[idx] = th0[idx]
    else:
      th[vary_index[idx]] = th0[idx]
  penalty = 0
  for idx in range(7):
    penalty += np.abs(th[13+idx])
  penalty = regularization * penalty
  #print("[DEBUG-hwlee]Lasso:th0 = {0}".format(th0))
  #print("[DEBUG-hwlee]Lasso:th = {0}".format(th))
  #print("[DEBUG-hwlee]Lasso: regularization = {0}, penalty = {1}".format(regularization, penalty))
  return -0.5 * (np.sum(((train_y - (th[13] + th[14]*train_x + th[15]*train_x**2 + 
        th[16]*train_x**3 + th[17]*train_x**4 + th[18]*train_x**5 + th[19]*train_x**6))/train_sigma) ** 2) + penalty)

# log likelihood function for Power GA Swampland model
#H_ga_swampland= H0 + H1 * z^2/(1+z)
# th[0] = Omega_k, th[1] = Omega_L, th[2] = H0, th[3] = b2, th[4] = c, th[5] = n, th[6] = H_bias, th[7] = a, th[8] = b, th[9] = c, th[10]=d, th[11] = H0, th[12] = H1
# variants = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1, 1] a, b are varying
# th_fixed = [0.0, 0, 0, 0, 1, 1, 0, 8.55^2, 19.1, 30.99, 4.88, 79, 92.3]
def loglikeGA_Swampland_restricted(th0):
  global train_x, train_y, train_sigma
  th = np.copy(th_fixed)
  vary_params = len(th0)
  for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
    if vary_params == len(th):
      th[idx] = th0[idx]
    else:
      th[vary_index[idx]] = th0[idx]
  return -0.5 * np.sum((train_y - (th[11] + th[12]*train_x**2/(1+train_x))) ** 2/train_sigma**2)

# log likelihood function for Power GA model
#H_ga_power=a + b * z + c * z^2 - z^d
# th[0] = Omega_k, th[1] = Omega_L, th[2] = H0, th[3] = b2, th[4] = c, th[5] = n, th[6] = H_bias, th[7] = a, th[8] = b, th[9] = c, th[10]=d
# variants = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1] a, b, c, d are varying
# th_fixed = [0.0, 0, 0, 0, 1, 1, 0, 8.55^2, 19.1, 30.99, 4.88]
def loglikeGA_Power_restricted(th0):
  global train_x, train_y, train_sigma
  th = np.copy(th_fixed)
  vary_params = len(th0)
  for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
    if vary_params == len(th):
      th[idx] = th0[idx]
    else:
      th[vary_index[idx]] = th0[idx]
  return -0.5 * np.sum((train_y - (th[7] + th[8]*train_x + th[9]*train_x**2 - train_x**th[10])) ** 2/train_sigma**2)

# log likelihood function for friedmann model
#H_friedmann=57.4*(0.3*(1+x)**3+0.7*(1+x)**2)**(0.5)
# th[0] = Omega_k, th[1] = Omega_L, th[2] = H0, th[3] = b2, th[4] = c, th[5] = n
# Omega_m = 1 + Omega_k - Omega_L
# variants = [0, 0, 1, 0, 0, 0] H0 vary 
# th_fixed = [-0.7, 0, H0, 0, 1, 1]
def loglikeFriedmann_restricted(th0):
  global train_x, train_y, train_sigma
  th = np.copy(th_fixed)
  #print("[DEBUG-hwlee]th = {0}".format(th))
  vary_params = len(th0)
  #print("[DEBUG-hwlee]vary_params = {0}".format(vary_params))
  for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
    #print("[DEBUG-hwlee]idx = {0}, th0[idx]={1}".format(idx, th0[idx]))
    if vary_params == len(th):
      th[idx] = th0[idx]
    else:
      th[vary_index[idx]] = th0[idx]
  #v = 0.0
  #for idx in range(len(train_x)):
  #  v = v + (train_y[idx] - th[2]*((1+th[0]-th[1])*(1+train_x[idx])**3 - th[0]*(1+train_x[idx])**2)**(0.5)) ** 2/train_sigma[idx]**2
  #return v
  return -0.5 * np.sum((train_y - th[2]*((1+th[0]-th[1])*(1+train_x)**3 - th[0]*(1+train_x)**2)**(0.5)) ** 2/train_sigma**2)

# log likelihood function for wCDM
#H_wCDM=70.8*(0.32*(1+x)**3+0.68*(1+x)**(3*-0.25))**(0.5)
# th[0] = Omega_k, th[1] = Omega_L, th[2] = H0, th[3] = b2, th[4] = c, th[5] = n
# th[3] = wde in this case
# variants = [0, 1, 1, 1, 0, 0] Omega_L, H0 and w_de vary
# th_fixed = [0.0, Omega_L, H0, w_de, 1, 1]
def loglikewCDM_restricted(th0):
  global train_x, train_y, train_sigma
  th = np.copy(th_fixed)
  vary_params = len(th0)
  for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
    if vary_params == len(th):
      th[idx] = th0[idx]
    else:
      th[vary_index[idx]] = th0[idx]
  return -0.5 * np.sum((train_y - th[2]*((1+th[0]-th[1])*(1+train_x)**3 + th[1]*(1+train_x)**(3*(1+th[3])))**(0.5)) ** 2/train_sigma**2)

# log likelihood function for Rh
#H_Rh = 61.8*(1+x)
# th[0] = Omega_k, th[1] = Omega_L, th[2] = H0, th[3] = b2, th[4] = c, th[5] = n
# variants = [0, 0, 1, 0, 0, 0] H0 vary
# th_fixed = [0.0, 0.0, H0, 0, 1, 1]
def loglikeRh_restricted(th0):
  global train_x, train_y, train_sigma
  global vary_index, th_fixed
  th = np.copy(th_fixed)
  vary_params = len(th0)
  for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
    if vary_params == len(th):
      th[idx] = th0[idx]
    else:
      th[vary_index[idx]] = th0[idx]
  return -0.5 * np.sum((train_y - th[2]*(1+train_x)) ** 2/train_sigma**2)


# sampling using Dynesty sampler
# fix some parameters
# th_min, th_max are global variables
# vary_index is global variable which variable is varying vary_index[idx] is actual parameter index
# th_fixed is a global variable which are fixed
#from dynesty import NestedSampler
#from dynesty import utils as dyfunc
#check validity of parameters 1 - c^2*Omega_k/Omega_L > 0
def check_parameter_restricted(th0):
  th = np.copy(th_fixed)
  #print("check_restricted: fixed = {}, th = {}".format(th_fixed, th))
  vary_params = len(th0)
  for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
    th[vary_index[idx]] = th0[idx]
  if th[1] != 0:
    check = (1 - th[4]*th[4] * th[0]/th[1] > 0) and (th[1] <= 1 + th[0]) and np.abs(th[0]) <= 1 and th[1] <= 1
  else:
    check = (th[1] <= 1 + th[0]) and np.abs(th[0]) <= 1 and th[1] <= 1
  #print("check_restricted: check = {}, th = {}".format(check, th))
  return check

# prior transform function, u is a list of random number between 0 and 1  
# dimension of u is smaller than 6, so convert to actual parameters correctly
def prior_transform_restricted(u):
    """Transforms the uniform random variables `u ~ Unif[0., 1.)`
    to the parameters of interest."""
    global vary_index, th_fixed
    global th_min, th_max

    x = np.array(u)  # copy u
    # uniform prior for each parameters of range [th_min(i), th_max(i)]
    #print("th_min = {}".format(th_min))
    #print("th_max = {}".format(th_max))
    #print("vary_index = {}".format(vary_index))
    for idx in range(len(u)):
      x[idx] = (th_max[vary_index[idx]] - th_min[vary_index[idx]])*u[idx] + th_min[vary_index[idx]]
    #print("u = {}".format(u))
    #print("x = {}".format(x))
    while(1):
      #print("prior_transform_restricted: x = {}".format(x))
      if check_parameter_restricted(x) :
        break
      # uniform prior for each parameters of range [th_min(i), th_max(i)]
      for idx in range(len(u)):
          x[idx] = (th_max[vary_index[idx]] - th_min[vary_index[idx]])*np.random.rand() + th_min[vary_index[idx]]

    return x

# assuming uniform prior for simplicity, delz is the convergence creterior for Dynesty sampler, n MAX iteration non limit if zero
# input : variants = list of varying parameters, integer 0 or 1
def Dynesty_sampler_restricted(loglike_restricted, th_min0, th_max0, train_x0, train_y0, train_sigma0, k0, delz0, live_points, N, multiple, variants0):
  # Define the dimensionality of our problem.
  global delz, th_min, th_max, train_x, train_y, train_sigma, global_k
  #print("Dynesty_sampler_restricted : variants = {}".format(variants))
  varying_params = 0
  for idx in range(len(th_min0)):
    if variants0[idx] > 0:
        varying_params = varying_params + 1
  ndim = varying_params
  #print("Dynesty_sampler_restricted : ndim = {}".format(ndim))
  copy_data(th_min0, th_min)
  copy_data(th_max0, th_max)
  copy_data(variants0, variants)
  #print("Dynesty_sampler_restricted : train_x = {}".format(train_x))
  #print("Dynesty_sampler_restricted : train_x0 = {}".format(train_x0))
  train_x = np.array(train_x0)
  #print("Dynesty_sampler_restricted : train_x = {}".format(train_x))
  train_y = np.array(train_y0)
  train_sigma = np.array(train_sigma0)
  delz = delz0
  global_k = k0
  rlist = []
  sampler = NestedSampler(loglike_restricted, prior_transform_restricted, ndim, nlive=live_points, bound='none')
  for idx in range(multiple):
      # start our run
      if N <= 0:
        sampler.run_nested(dlogz=delz)
      else:
        sampler.run_nested(dlogz=delz, maxiter=N)
      res1 = sampler.results
      rlist.append(res1)
      sampler.reset()
    
  #Merge into a single run.
  results = dyfunc.merge_runs(rlist)
  return results


# log likelihood function, th is the list of starting parameters
# train_x, train_y, train_sigma are global variables
# debug correction for fixed Omega_k and varying H0 case at April 1, 2022 by hwlee
# in this case, correct Omega_k as varying H0 using global_k value Omega_k^0 = 110.8*global_k/H0**2
def loglikeInteracting_restricted(th0):
  global train_x, train_y, train_sigma, global_k
  th = np.copy(th_fixed)
  vary_params = len(th0)
  #print("[DEBUG-hwlee]vary_params = {0}, len(th) = {1}".format(vary_params, len(th)))
  if vary_params == len(th):
    for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
      th[idx] = th0[idx]
  else:
    for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
      th[vary_index[idx]] = th0[idx]
      if vary_index[0] == 0 and vary_index[2] == 1 and vary_index[idx] == 2: # if Omega_k is fixed and H0 is varying case
        th[0] = 110.8*global_k/(th[2]**2) # change Omega_k also
  #print("[DEBUG-hwlee]logL : th = {}".format(th))
  #print("[DEBUG-hwlee]logL : train_x[{0}] = {1}".format(len(train_x), train_x))
  try:
    logL = -E(train_x, train_y, train_sigma, th)
  except Exception as e:
    #logL = -np.log(sys.float_info.max) # this is not valid solution, that is not allowed cosmological evolution
    logL = logL_min*(1 + np.random.rand())
    #print("Error while calculating cost function: logL = {}, error = {}".format(logL, e))
  #print("[DEBUG-hwlee]log L : fixed = {}, th = {}, logL = {}".format(th_fixed, th, logL))
  return logL

# actual number of varying parameters ??hwlee
def varying_param_numbers(variants):
  varying_params = 0
  for idx in range(len(variants)):
    if variants[idx] > 0:
        varying_params = varying_params + 1
  return varying_params

# actual number of varying parameters
def get_varying_index(variants):
  varies = []
  for idx in range(len(variants)):
    if variants[idx] > 0:
        varies.append(idx)
  return varies

#define global labes
global_labels=[r"$\Omega_k^0$", r"$\Omega_\Lambda^0$", r"$H_0$", r"$b^2$", r"$c$", r"$n$", r"$S_b$", 
    r"$a$", r"$b$", r"$c$", r"$d$", r"$H_0$", r"$H_1$", 
    r"$\omega_0$", r"$\omega_1$", r"$\omega_2$", r"$\omega_3$", r"$\omega_4$", r"$\omega_5$", r"$\omega_6$"]

# actual labels of varying parameters
def get_varying_labels(variants, labels0):
  varies = []
  for idx in range(len(variants)):
    if variants[idx] > 0:
        varies.append(labels0[idx])
  return varies

#find the maximum likelihood values
# input : th_samples = Dynesty results
def findMaxL_restricted_old(th_samples):
    print("Samples shape = {}".format(th_samples['samples'].shape))
    maxL = np.max(th_samples['logl'])
    maxLindices = np.where(th_samples['logl'] == maxL)
    maxLindex = maxLindices[0]
    th_maxL0 = th_samples['samples'][maxLindex, :][0]
    th_maxL = np.copy(th_fixed)
    vary_params = varying_param_numbers(variants)
    for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
      th_maxL[vary_index[idx]] = th_maxL0[idx]
    print("Maximum Likelihood Parameters = {}, maxl = {}".format(th_maxL, maxL))
    print("Error value for maxL parameters = {}".format(E(train_x, train_y, train_sigma, th_maxL)))
    return maxL, maxLindex, th_maxL

#Dictionary for log Likelihood functions
logLikeFunctions_restricted = {"Interacting":loglikeInteracting_restricted, "LCDM":loglikeLCDM_restricted, "Friedmann":loglikeFriedmann_restricted, "wCDM":loglikewCDM_restricted, "Rh":loglikeRh_restricted}
logLikeFunctions_restricted_shift = {"Interacting":loglikeInteracting_restricted, "LCDM":loglikeLCDM_restricted_shift, "Friedmann":loglikeFriedmann_restricted, "wCDM":loglikewCDM_restricted, "Rh":loglikeRh_restricted}
logLikeFunctions_restricted_GA = {"Interacting":loglikeInteracting_restricted, "LCDM":loglikeLCDM_restricted, "LCDM_Shift":loglikeLCDM_restricted_shift, "Friedmann":loglikeFriedmann_restricted, "wCDM":loglikewCDM_restricted, "Rh":loglikeRh_restricted, 
  "GA_Power":loglikeGA_Power_restricted, "GA_Swampland":loglikeGA_Swampland_restricted, "ML_LeastSquare":loglikeML_LeastSquare_restricted,
  "ML_LCDM":loglikeML_None_restricted, "ML_None":loglikeML_None_restricted,
  "ML_Ridge":loglikeML_Ridge_restricted, "ML_Lasso":loglikeML_Lasso_restricted}
logLikeFunctions = {"Interacting":loglike, "LCDM":loglikeLCDM, "Friedmann":loglikeFriedmann, "wCDM":loglikewCDM, "Rh":loglikeRh}
#  global param_numbers : dictionary for the parameter numbers {"Interacting":6, "LCDM":2, ...}
global_param_numbers = {"Interacting":6, "LCDM":2, "LCDM_Shift":3, "Friedmann":1, "wCDM":3, "Rh":1, 
  "GA_Power":4, "GA_Swampland":2, "ML_LCDM":2, "ML_LeastSquare":7, "ML_None":7, "ML_Ridge":7, "ML_Lasso":7}
#Dictionary for best parameters for each model, these value may change with further analysis
global_best_params = {"Interacting":[0,0,0,0,0,0], "LCDM":[0.673, 67.776], 
  "GA_Power":[68.35986079, 33.96376347, 18.23075075,  0.2040769], "GA_Swampland":[72.36071148, 101.30509381],
  "ML_LCDM":[68.133,   32.704, 24.855, -1.029, -4.040,  2.314,   0.355], 
  "ML_LeastSquare":[55.99787729 , 182.70364549, -487.77741817,  498.40171095,   96.76843638, -301.99012188,   91.96869298], 
  "ML_None":[65.19150947,   96.37304528, -204.84092753,  103.82747624,  331.73884544, -347.95102451,   90.28361852], 
  "ML_Ridge":[72.23508144,  13.38461105,   3.75429337, 106.03935698, -73.41626385,   3.81633195,   4.4323096], 
  "ML_Lasso":[65.07883175,   96.94751382, -205.04598287,  103.12484364,  331.56479242, -347.11871663,   89.99203462]}

#find the maximum likelihood values
# input : th_samples = Dynesty results
def findMaxL_restricted(th_samples, type):
    print("Samples shape = {}".format(th_samples['samples'].shape))
    maxL = np.max(th_samples['logl'])
    maxLindices = np.where(th_samples['logl'] == maxL)
    maxLindex = maxLindices[0]
    th_maxL0 = th_samples['samples'][maxLindex, :][0]
    print("th_maxL0 = {}, maxLIndex = {}, maxL = {}".format(th_maxL0, maxLindex, maxL))
    th_maxL = np.copy(th_fixed)
    vary_params = varying_param_numbers(variants)
    for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
      th_maxL[vary_index[idx]] = th_maxL0[idx]
    print("Maximum Likelihood Parameters = {}, maxl = {}".format(th_maxL, maxL))
    print("Error value for maxL parameters = {}".format(-logLikeFunctions_restricted[type](th_maxL)))
    return maxL, maxLindex, th_maxL

def findMaxL_restricted_shift(th_samples, type):
    print("Samples shape = {}".format(th_samples['samples'].shape))
    maxL = np.max(th_samples['logl'])
    maxLindices = np.where(th_samples['logl'] == maxL)
    maxLindex = maxLindices[0]
    th_maxL0 = th_samples['samples'][maxLindex, :][0]
    print("th_maxL0 = {}, maxLIndex = {}, maxL = {}".format(th_maxL0, maxLindex, maxL))
    th_maxL = np.copy(th_fixed)
    vary_params = varying_param_numbers(variants)
    for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
      th_maxL[vary_index[idx]] = th_maxL0[idx]
    print("Maximum Likelihood Parameters = {}, maxl = {}".format(th_maxL, maxL))
    print("Error value for maxL parameters = {}".format(-logLikeFunctions_restricted_shift[type](th_maxL)))
    return maxL, maxLindex, th_maxL

def findMaxL_restricted_GA(th_samples, type):
    print("Samples shape = {}".format(th_samples['samples'].shape))
    maxL = np.max(th_samples['logl'])
    maxLindices = np.where(th_samples['logl'] == maxL)
    maxLindex = maxLindices[0]
    th_maxL0 = th_samples['samples'][maxLindex, :][0]
    print("th_maxL0 = {}, maxLIndex = {}, maxL = {}".format(th_maxL0, maxLindex, maxL))
    th_maxL = np.copy(th_fixed)
    vary_params = varying_param_numbers(variants)
    for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
      th_maxL[vary_index[idx]] = th_maxL0[idx]
    print("Maximum Likelihood Parameters = {}, maxl = {}".format(th_maxL, maxL))
    print("Error value for maxL parameters = {}".format(-logLikeFunctions_restricted_GA[type](th_maxL)))
    return maxL, maxLindex, th_maxL

def findMaxP_restricted_shift(th_samples, type, num_bins):
    print("Samples shape = {}".format(th_samples['samples'].shape))
    maxPvalues = []
    th_maxP = np.copy(th_fixed)
    vary_params = varying_param_numbers(variants)
    columns = list(zip(*th_samples['samples']))
    for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
      hist, bin_edges = np.histogram(columns[idx], num_bins)
      #print("data = {0}".format(columns[idx]))
      v = np.max(hist)
      maxPvalues.append(v)
      maxPindices = np.where(hist == v)[0]
      #print("max count = {0}, indices = {1}, hist = {2}, bin_edges = {3}".format(v, maxPindices, hist, bin_edges))
      th_maxP[vary_index[idx]] = bin_edges[maxPindices] + (bin_edges[1]-bin_edges[0])/2
    maxP = logLikeFunctions_restricted_shift[type](th_maxP)
    print("Maximum Posterior Parameters = {}, maxl = {}".format(th_maxP, maxP))
    print("Error value for maxP parameters = {}".format(-logLikeFunctions_restricted_shift[type](th_maxP)))
    return maxP, maxPvalues, th_maxP

def findMaxP_restricted_GA(th_samples, type, num_bins):
    print("Samples shape = {}".format(th_samples['samples'].shape))
    maxPvalues = []
    th_maxP = np.copy(th_fixed)
    vary_params = varying_param_numbers(variants)
    columns = list(zip(*th_samples['samples']))
    for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
      hist, bin_edges = np.histogram(columns[idx], num_bins)
      #print("data = {0}".format(columns[idx]))
      v = np.max(hist)
      maxPvalues.append(v)
      maxPindices = np.where(hist == v)[0]
      if len(maxPindices) > 1 :
        maxPindices = maxPindices[0]
      #print("max count = {0}, indices = {1}, hist = {2}, bin_edges = {3}".format(v, maxPindices, hist, bin_edges))
      th_maxP[vary_index[idx]] = bin_edges[maxPindices] + (bin_edges[1]-bin_edges[0])/2
    maxP = logLikeFunctions_restricted_GA[type](th_maxP)
    print("Maximum Posterior Parameters = {}, maxl = {}".format(th_maxP, maxP))
    print("Error value for maxP parameters = {}".format(-logLikeFunctions_restricted_GA[type](th_maxP)))
    return maxP, maxPvalues, th_maxP

def findAvgP_restricted_shift(th_samples, type, num_bins):
    print("Samples shape = {}".format(th_samples['samples'].shape))
    avgPvalues = []
    th_avgP = np.copy(th_fixed)
    vary_params = varying_param_numbers(variants)
    columns = list(zip(*th_samples['samples']))
    for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
      hist, bin_edges = np.histogram(columns[idx], num_bins)
      total = np.sum(hist)
      avg = 0
      for idx1 in range(len(hist)):
        avg = avg + hist[idx1]*(bin_edges[idx1] + (bin_edges[1]-bin_edges[0])/2)
      avg = avg / total
      avgPvalues.append(avg)
      th_avgP[vary_index[idx]] = avg
    avgP = logLikeFunctions_restricted_GA[type](th_avgP)
    print("Average Posterior Parameters = {}, maxl = {}".format(th_avgP, avgP))
    print("Error value for avgP parameters = {}".format(-logLikeFunctions_restricted_shift[type](th_avgP)))
    return avgP, avgPvalues, th_avgP

def findAvgP_restricted_GA(th_samples, type, num_bins):
    print("Samples shape = {}".format(th_samples['samples'].shape))
    avgPvalues = []
    th_avgP = np.copy(th_fixed)
    vary_params = varying_param_numbers(variants)
    columns = list(zip(*th_samples['samples']))
    for idx in range(vary_params): #copy varying parameters to actual 6 dimensional parameters
      hist, bin_edges = np.histogram(columns[idx], num_bins)
      total = np.sum(hist)
      avg = 0
      for idx1 in range(len(hist)):
        avg = avg + hist[idx1]*(bin_edges[idx1] + (bin_edges[1]-bin_edges[0])/2)
      avg = avg / total
      avgPvalues.append(avg)
      th_avgP[vary_index[idx]] = avg
    avgP = logLikeFunctions_restricted_GA[type](th_avgP)
    print("Average Posterior Parameters = {}, maxl = {}".format(th_avgP, avgP))
    print("Error value for avgP parameters = {}".format(-logLikeFunctions_restricted_GA[type](th_avgP)))
    return avgP, avgPvalues, th_avgP

#find the average likelihood values
# input : th_samples = Dynesty results
def findAvgL_restricted(th_samples):
    print("Samples shape = {}".format(th_samples['samples'].shape))
    logL = th_samples['logl']
    avgL = 0.0
    for idx in range(len(logL)):
      avgL = avgL + logL[idx]
    avgL = avgL / len(logL)
    print("Average Likelihood = {}".format(avgL))
    return avgL

def findAvgL_restricted_shift(th_samples):
    return findAvgL_restricted(th_samples)

def findAvgL_restricted_GA(th_samples):
    return findAvgL_restricted(th_samples)

#calculate AIC for various models
# input :
#    th : dictionary for the best parameters {"Interacting":[th[0], ..., th[5]], "LCDM":[...], ...}
#    param_numbers : dictionary for the parameter numbers {"Interacting":6, "LCDM":2, ...}
# returns : dictionary of AIC values {"Interacting":v1, "LCDM":v2, ...}
# train_x, train_y, train_sigma are global variables
def calculateAIC(th, param_numbers):
    results = {}
    results['Interacting'] = -2*loglike(th['Interacting']) + 2*param_numbers['Interacting']
    results['LCDM'] = -2*loglikeLCDM(th['LCDM']) + 2*param_numbers['LCDM']
    results['Friedmann'] = -2*loglikeFriedmann(th['Friedmann']) + 2*param_numbers['Friedmann']
    results['wCDM'] = -2*loglikewCDM(th['wCDM']) + 2*param_numbers['wCDM']
    results['Rh'] = -2*loglikeRh(th['Rh']) + 2*param_numbers['Rh']
    return results
def calculateAIC_single(loglike, th, param_numbers):
    results = -2*loglike(th) + 2 * param_numbers
    return results


#calculate BIC for various models
# input :
#    th : dictionary for the best parameters {"Interacting":[th[0], ..., th[5]], "LCDM":[...], ...}
#    param_numbers : dictionary for the parameter numbers {"Interacting":6, "LCDM":2, ...}
# returns : dictionary of BIC values {"Interacting":v1, "LCDM":v2, ...}
# train_x, train_y, train_sigma are global variables
def calculateBIC(th, param_numbers):
    num_data = len(train_x)
    results = {}
    results['Interacting'] = -2*loglike(th['Interacting']) + param_numbers['Interacting']*np.log(num_data)
    results['LCDM'] = -2*loglikeLCDM(th['LCDM']) + param_numbers['LCDM']*np.log(num_data)
    results['Friedmann'] = -2*loglikeFriedmann(th['Friedmann']) + param_numbers['Friedmann']*np.log(num_data)
    results['wCDM'] = -2*loglikewCDM(th['wCDM']) + param_numbers['wCDM']*np.log(num_data)
    results['Rh'] = -2*loglikeRh(th['Rh']) + param_numbers['Rh']*np.log(num_data)
    return results

def calculateBIC_restricted(th, param_numbers):
    num_data = len(train_x)
    results = {}
    results['Interacting'] = -2*loglike(th['Interacting']) + param_numbers['Interacting']*np.log(num_data)
    results['LCDM'] = -2*loglikeLCDM_restricted(th['LCDM']) + param_numbers['LCDM']*np.log(num_data)
    results['LCDM_Shift'] = -2*loglikeLCDM_restricted_shift(th['LCDM_Shift']) + param_numbers['LCDM_Shift']*np.log(num_data)
    results['Friedmann'] = -2*loglikeFriedmann_restricted(th['Friedmann']) + param_numbers['Friedmann']*np.log(num_data)
    results['wCDM'] = -2*loglikewCDM_restricted(th['wCDM']) + param_numbers['wCDM']*np.log(num_data)
    results['Rh'] = -2*loglikeRh_restricted(th['Rh']) + param_numbers['Rh']*np.log(num_data)
    results['GA_Power'] = -2*loglikeGA_Power_restricted(th['GA_Power']) + param_numbers['GA_Power']*np.log(num_data)
    results['GA_Swampland'] = -2*loglikeGA_Swampland_restricted(th['GA_Swampland']) + param_numbers['GA_Swampland']*np.log(num_data)
    return results

# calculate BIC value for given likelihood function, optimal parameters and number of parameters
# loglike : loglikelihood function
# th : optimal parameters
# param_numbers : the number of model parameters
# num_data : number of data points
def calculateBIC_single(loglike, th, param_numbers, num_data):
    results = -2*loglike(th) + param_numbers*np.log(num_data)
    return results

#calculate DIC for various models
# input :
#    th : dictionary for the best parameters {"Interacting":[th[0], ..., th[5]], "LCDM":[...], ...}
#    avgL : dictionary for the average log likelihood {"Interacting":avgL, "LCDM":avgL, ...}
# returns : dictionary of DIC values {"Interacting":v1, "LCDM":v2, ...}
# train_x, train_y, train_sigma are global variables
def calculateDIC(th, avgL):
    meff = {}
    meff['Interacting'] = -2*avgL['Interacting'] + 2*loglike(th['Interacting'])
    meff['LCDM'] = -2*avgL['LCDM'] + 2*loglikeLCDM(th['LCDM'])
    meff['Friedmann'] = -2*avgL['Friedmann'] + 2*loglikeFriedmann(th['Friedmann'])
    meff['wCDM'] = -2*avgL['wCDM'] + 2*loglikewCDM(th['wCDM'])
    meff['Rh'] = -2*avgL['Rh'] + 2*loglikeRh(th['Rh'])

    results = {}
    results['Interacting'] = -2*loglike(th['Interacting']) + meff['Interacting']
    results['LCDM'] = -2*loglikeLCDM(th['LCDM']) + meff['LCDM']
    results['Friedmann'] = -2*loglikeFriedmann(th['Friedmann']) + meff['Friedmann']
    results['wCDM'] = -2*loglikewCDM(th['wCDM']) + meff['wCDM']
    results['Rh'] = -2*loglikeRh(th['Rh']) + meff['Rh']
    return results

def calculateDIC_restricted(th, avgL):
    meff = {}
    meff['Interacting'] = -2*avgL['Interacting'] + 2*loglike(th['Interacting'])
    meff['LCDM'] = -2*avgL['LCDM'] + 2*loglikeLCDM_restricted(th['LCDM'])
    meff['Friedmann'] = -2*avgL['Friedmann'] + 2*loglikeFriedmann_restricted(th['Friedmann'])
    meff['wCDM'] = -2*avgL['wCDM'] + 2*loglikewCDM_restricted(th['wCDM'])
    meff['Rh'] = -2*avgL['Rh'] + 2*loglikeRh_restricted(th['Rh'])
    meff['GA_Power'] = -2*avgL['GA_Power'] + 2*loglikeRh_restricted(th['GA_Power'])
    meff['GA_Swampland'] = -2*avgL['GA_Swampland'] + 2*loglikeRh_restricted(th['GA_Swampland'])

    results = {}
    results['Interacting'] = -2*loglike(th['Interacting']) + meff['Interacting']
    results['LCDM'] = -2*loglikeLCDM_restricted(th['LCDM']) + meff['LCDM']
    results['LCDM_Shift'] = -2*loglikeLCDM_restricted_shift(th['LCDM_Shift']) + meff['LCDM_Shift']
    results['Friedmann'] = -2*loglikeFriedmann_restricted(th['Friedmann']) + meff['Friedmann']
    results['wCDM'] = -2*loglikewCDM_restricted(th['wCDM']) + meff['wCDM']
    results['Rh'] = -2*loglikeRh_restricted(th['Rh']) + meff['Rh']
    results['GA_Power'] = -2*loglikeGA_Power_restricted(th['GA_Power']) + meff['GA_Power']
    results['GA_Swampland'] = -2*loglikeGA_Swampland_restricted(th['GA_Swampland']) + meff['GA_Swampland']
    return results

# calculate DIC value for given likelihood function, optimal parameters and number of parameters
# loglike : loglikelihood function
# th : optimal parameters
# avgL : average loglikelihood
# num_data : number of data points
def calculateDIC_single(loglike, th, avgL):
    meff = -2*avgL + 2*loglike(th)
    results = -2*loglike(th) + 2*meff
    return results

def plot_sol(Omega_k_0, Omega_L_0, Omega_H_0, time, b, c, n):
    #initial condition
    Omega0_0 = Omega_k_0
    Omega1_0 = Omega_L_0
    Omega2_0 = Omega_H_0
    Omega3_0 = b
    Omega4_0 = c
    Omega5_0 = n
    Omega_0 = [Omega0_0, Omega1_0, Omega2_0, Omega3_0, Omega4_0, Omega5_0]
    rtol, atol = (1e-8, 1e-8)
    x_span = [0,time]
    print('Calling solve_ivp')
    sol_forward = solve_ivp(f_forward, x_span, Omega_0, rtol=rtol, atol=atol)
    
    print("sol_forward.y[2][::6]: {}".format(sol_forward.y[2][0::6]))
    print("sol_forward.y[1][::6]: {}".format(sol_forward.y[1][0::6]))
    print("sol_forward.y[0][::6]: {}".format(sol_forward.y[0][0::6]))
    Omega_m_forward = 1.0 - sol_forward.y[1] - sol_forward.y[0]
    #eos_Lambda_forward=eos_Lambda(sol_forward.y[1], sol_forward.y[0], c)
    #eos_m_forward=eos_m(sol_forward.y[1], sol_forward.y[0], Omega3_0, Omega5_0)

    sol_backward = solve_ivp(f_backward, x_span, Omega_0, rtol=rtol, atol=atol)
    Omega_m_backward = 1.0 - sol_backward.y[1] - sol_backward.y[0]
    #eos_Lambda_backward=eos_Lambda(sol_backward.y[1], sol_backward.y[0], c)
    #eos_m_backward=eos_m(sol_backward.y[1], sol_backward.y[0], Omega3_0, Omega5_0)
    print("sol_backward.y[2][::6]: {}".format(sol_backward.y[2][0::6]))
    print("sol_backward.y[1][::6]: {}".format(sol_backward.y[1][0::6]))
    print("sol_backward.y[0][::6]: {}".format(sol_backward.y[0][0::6]))

    plt.figure(figsize=(20, 6))
    
    plt.subplot(131)
    plt.plot(sol_forward.t, sol_forward.y[1], color="blue", label='$\Omega_\Lambda$')
    plt.plot(-sol_backward.t, sol_backward.y[1], color="blue")
    plt.plot(sol_forward.t, sol_forward.y[0], color="orange", label='$\Omega_k$')
    plt.plot(-sol_backward.t, sol_backward.y[0], color="orange")
    plt.plot(sol_forward.t, Omega_m_forward, color="red", label='$\Omega_m$')
    plt.plot(-sol_backward.t, Omega_m_backward, color="red")

    plt.title("$\Omega^0_k = {0:.3f}$".format(Omega_k_0))
    plt.xlabel("$\ln a$")
    plt.legend()
   
    
    plt.subplot(132)
    plt.plot(sol_forward.t, sol_forward.y[2], color="green", label='H')
    plt.plot(-sol_backward.t, sol_backward.y[2], color="green")
    plt.title("$\Omega^0_k = {0:.3f}$".format(Omega_k_0))
    plt.xlabel("$\ln a$")
    plt.legend()

    plt.subplot(133)
    eos_list = eos_Lambda_list(sol_forward.y[1], sol_forward.y[0], c)
    plt.plot(sol_forward.t, eos_list, color="red", label='eos_Lambda')
    eos_list = eos_Lambda_list(sol_backward.y[1], sol_backward.y[0], c)
    plt.plot(-sol_backward.t, eos_list, color="red")

    eos_list = eos_m_list(sol_forward.y[1], sol_forward.y[0], b, n)
    plt.plot(sol_forward.t, eos_list, color="blue", label='eos_m')
    eos_list = eos_m_list(sol_backward.y[1], sol_backward.y[0], b, n)
    plt.plot(-sol_backward.t, eos_list, color="blue")
    #plt.plot( eos_Lambda(Omega_L_0, Omega_k_0, c), color="red")
    plt.legend()

    #plt.plot(sol_forward.t, eos_m_forward, color="green", label='eos_m')
    #plt.plot(-sol_backward.t, eos_m_backward, color="green")

    
    plt.title("$ EOS_lamda+EOS_m= {0:.3f}$".format(Omega_k_0))
    plt.xlabel("$\ln a$")

    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    z_forward = np.exp(-sol_forward.t) -1
    z_backward = np.exp(sol_backward.t) -1
    #plt.plot(z_forward, sol_forward.y[2], color="green")
    plt.plot(z_backward, sol_backward.y[2], color="green", label="H")
    #plt.xlim(0,2)
    #plt.ylim(0,2*Omega_H_0)
    plt.title("$\Omega^0_k = {0:.3f}$".format(Omega_k_0))
    plt.xlabel("z")
    plt.legend()

    plt.subplot(122)
    #plt.plot(z_forward, sol_forward.y[2], color="green")
    plt.plot(z_backward, sol_backward.y[2], color="green", label="H")
    plt.xlim(0,20)
    plt.ylim(0,100*Omega_H_0)
    plt.title("$\Omega^0_k = {0:.3f}$".format(Omega_k_0))
    plt.xlabel("z")

    plt.legend()
    plt.show()

def plot_sol_single(Omega_k_0, Omega_L_0, Omega_H_0, time, b, c, n):
    #initial condition
    Omega0_0 = Omega_k_0
    Omega1_0 = Omega_L_0
    Omega2_0 = Omega_H_0
    Omega3_0 = b
    Omega4_0 = c
    Omega5_0 = n
    Omega_0 = [Omega0_0, Omega1_0, Omega2_0, Omega3_0, Omega4_0, Omega5_0]
    rtol, atol = (1e-8, 1e-8)
    x_span = [0,time]
    #print('Calling solve_ivp')
    sol_forward = solve_ivp(f_forward, x_span, Omega_0, rtol=rtol, atol=atol)
    
    #print("sol_forward.y[2][::6]: {}".format(sol_forward.y[2][0::6]))
    #print("sol_forward.y[1][::6]: {}".format(sol_forward.y[1][0::6]))
    #print("sol_forward.y[0][::6]: {}".format(sol_forward.y[0][0::6]))
    Omega_m_forward = 1.0 - sol_forward.y[1] - sol_forward.y[0]
    #eos_Lambda_forward=eos_Lambda(sol_forward.y[1], sol_forward.y[0], c)
    #eos_m_forward=eos_m(sol_forward.y[1], sol_forward.y[0], Omega3_0, Omega5_0)

    sol_backward = solve_ivp(f_backward, x_span, Omega_0, rtol=rtol, atol=atol)
    Omega_m_backward = 1.0 - sol_backward.y[1] - sol_backward.y[0]
    #eos_Lambda_backward=eos_Lambda(sol_backward.y[1], sol_backward.y[0], c)
    #eos_m_backward=eos_m(sol_backward.y[1], sol_backward.y[0], Omega3_0, Omega5_0)
    #print("sol_backward.y[2][::6]: {}".format(sol_backward.y[2][0::6]))
    #print("sol_backward.y[1][::6]: {}".format(sol_backward.y[1][0::6]))
    #print("sol_backward.y[0][::6]: {}".format(sol_backward.y[0][0::6]))

    #plt.figure(figsize=(20, 6))
    
    #plt.subplot(131)
    plt.plot(sol_forward.t, sol_forward.y[1], color="blue", label=r'$\Omega_\Lambda$')
    plt.plot(-sol_backward.t, sol_backward.y[1], color="blue")
    plt.plot(sol_forward.t, sol_forward.y[0], color="orange", label=r'$\Omega_k$')
    plt.plot(-sol_backward.t, sol_backward.y[0], color="orange")
    plt.plot(sol_forward.t, Omega_m_forward, color="red", label=r'$\Omega_m$')
    plt.plot(-sol_backward.t, Omega_m_backward, color="red")

    plt.title(r"$\Omega^0_k = {0:.3f}$".format(Omega_k_0))
    plt.xlabel(r"$\ln a$")
    plt.legend()
   
    plt.show()

"""
  finding cosmological parameters of flat LCDM, H0 and Omega_L0 by transforming from the other function
  method : minimizing Sum_i (H(z_i;H0, OL0)**2 - H_GA(z_i)**2)**2
  A = Sum_i [(1+z_i)**3*(1-(1+z_i)**3)]
  B = Sum_i [(1-(1+z_i)**3)**2]
  C = Sum_i [H_GA(z_i)**2*(1-(1+z_i)**3)]
  D = Sum_i [(1+z_i)**6]
  E = Sum_i [H_GA(z_i)**2]
"""
def find_H0_Omega_L0(H_GA, params0, train_x):
  """
    H_GA : function to fit
    params : parameters for H_GA(z) function
    train_x : observed red shift values
  """
  params = params0['samples']
  ndim = len(np.array(params).shape)
  if ndim == 1: # need to consider this case more
    v0 = find_H0_Omega_L0_single(H_GA, params, train_x)
    if v0[2]: #valid case
      return [v0[0], v0[1]], None
    else:
      return None, [v0[0], v0[1]]
  elif ndim > 2:
    return None

  th_samples_new = {}
  th_samples_new['samples'] = []
  th_samples_new['logwt'] = []
  th_samples_new['logz'] = []
  th_samples_new['logl'] = []

  th_samples_ignored = {}
  th_samples_ignored['samples'] = []
  th_samples_ignored['logwt'] = []
  th_samples_ignored['logz'] = []
  th_samples_ignored['logl'] = []

  for idx in range(np.array(params).shape[0]):
    param = params[idx]
    v0 = find_H0_Omega_L0_single(H_GA, param, train_x)
    v = [v0[0], v0[1]]
    #if v[0] >= 0 and v[0] <= 1 : # add only physically acceptabile value for Omega_L0, here we only consider flat case
    if v0[2]: # v0[2] contains valid value, True means that this value is valid physically
      th_samples_new['samples'].append(v)
      th_samples_new['logwt'].append(params0['logwt'][idx])
      th_samples_new['logz'].append(params0['logz'][idx])
      th_samples_new['logl'].append(params0['logl'][idx])
    else:
      th_samples_ignored['samples'].append(v)
      th_samples_ignored['logwt'].append(params0['logwt'][idx])
      th_samples_ignored['logz'].append(params0['logz'][idx])
      th_samples_ignored['logl'].append(params0['logl'][idx])
  th_samples_new['samples'] = np.array(th_samples_new['samples'])
  th_samples_new['logwt'] = np.array(th_samples_new['logwt'])
  th_samples_new['logz'] = np.array(th_samples_new['logz'])
  th_samples_new['logl'] = np.array(th_samples_new['logl'])
  th_samples_ignored['samples'] = np.array(th_samples_ignored['samples'])
  th_samples_ignored['logwt'] = np.array(th_samples_ignored['logwt'])
  th_samples_ignored['logz'] = np.array(th_samples_ignored['logz'])
  th_samples_ignored['logl'] = np.array(th_samples_ignored['logl'])
  return th_samples_new, th_samples_ignored

def find_H0_Omega_L0_single(H_GA, params, train_x):
  """
    H_GA : function to fit
    params : parameters for H_GA(z) function
    train_x : observed red shift values
  """
  y = np.array(H_GA(params, train_x))
  if np.isnan(y).any() or np.isinf(y).any():
    H0 = H_GA(params, 0) # value at z = 0
    H1 = H_GA(params, 1) # value at z = 1
    OL0 = 8/7 - (H1/H0)**2/7 # we will ignore this case
    valid = False
    return [OL0, H0, valid]
  z = np.array(train_x)
  A = np.sum(np.power(1+z, 3)*(1-np.power(1+z, 3)))
  B = np.sum(np.power(1-np.power(1+z, 3), 2))
  C = np.sum(np.power(y, 2)*(1-np.power(1+z,3)))
  D = np.sum(np.power(1+z, 6))
  E = np.sum(np.power(y, 2)*np.power(1+z,3))
  OL0 = (A*E - D*C)/(A*C - B*E)
  H02 = (C/(A+B*OL0))
  valid = True
  if OL0 < 0 or OL0 > 1 or H02 < 0: #this means that it is not possible to find optimal value for this function
    H0 = H_GA(params, 0) # value at z = 0
    H1 = H_GA(params, 1) # value at z = 1
    OL0 = 8/7 - (H1/H0)**2/7 # we will ignore this case
    valid = False
  else:
    H0 = np.sqrt(H02)
  #print(A,B,C,D,E)
  #print("[DEBUG-hwlee]find_H0_Omega_L0_single: H02 = {0}, OL0 = {1}".format(H02, OL0))
  return [OL0, H0, valid]

  """
  finding cosmological parameters of flat LCDM, H0 and Omega_L0 by transforming from the other function
  method : minimizing Sum_i (H(z_i;H0, OL0) - H_GA(z_i))**2
  using scipy curve fitting routine
  """
def find_H0_Omega_L0_scipy(H_GA, params0, train_x, train_sigma, x0):
  """
    H_GA : function to fit
    params : parameters for H_GA(z) function
    train_x : observed red shift values
  """
  params = params0['samples']
  ndim = len(np.array(params).shape)
  if ndim == 1: # need to consider this case more
    v0 = find_H0_Omega_L0_single_scipy(H_GA, params, train_x, train_sigma, x0)
    if v0[2]: #valid case
      return [v0[0], v0[1]], None
    else:
      return None, [v0[0], v0[1]]
  elif ndim > 2:
    return None

  th_samples_new = {}
  th_samples_new['samples'] = []
  th_samples_new['logwt'] = []
  th_samples_new['logz'] = []
  th_samples_new['logl'] = []

  th_samples_ignored = {}
  th_samples_ignored['samples'] = []
  th_samples_ignored['logwt'] = []
  th_samples_ignored['logz'] = []
  th_samples_ignored['logl'] = []

  for idx in range(np.array(params).shape[0]):
    param = params[idx]
    v0 = find_H0_Omega_L0_single_scipy(H_GA, param, train_x, train_sigma, x0)
    v = [v0[0], v0[1]]
    #if v[0] >= 0 and v[0] <= 1 : # add only physically acceptabile value for Omega_L0, here we only consider flat case
    if v0[2]: # v0[2] contains valid value, True means that this value is valid physically
      th_samples_new['samples'].append(v)
      th_samples_new['logwt'].append(params0['logwt'][idx])
      th_samples_new['logz'].append(params0['logz'][idx])
      th_samples_new['logl'].append(params0['logl'][idx])
    else:
      th_samples_ignored['samples'].append(v)
      th_samples_ignored['logwt'].append(params0['logwt'][idx])
      th_samples_ignored['logz'].append(params0['logz'][idx])
      th_samples_ignored['logl'].append(params0['logl'][idx])
  th_samples_new['samples'] = np.array(th_samples_new['samples'])
  th_samples_new['logwt'] = np.array(th_samples_new['logwt'])
  th_samples_new['logz'] = np.array(th_samples_new['logz'])
  th_samples_new['logl'] = np.array(th_samples_new['logl'])
  th_samples_ignored['samples'] = np.array(th_samples_ignored['samples'])
  th_samples_ignored['logwt'] = np.array(th_samples_ignored['logwt'])
  th_samples_ignored['logz'] = np.array(th_samples_ignored['logz'])
  th_samples_ignored['logl'] = np.array(th_samples_ignored['logl'])
  return th_samples_new, th_samples_ignored

def fun_fit(z, H0, OL0):
  v = H0*np.sqrt((1-OL0)*(1+z)**3 + OL0)
  return v

def find_H0_Omega_L0_single_scipy(H_GA, params, train_x, train_sigma, x0):
  """
    H_GA : function to fit
    params : parameters for H_GA(z) function
    train_x : observed red shift values
    train_sigma : observational errors
    x0 : start parameter
  """
  train_y = np.array(H_GA(params, train_x))
  if np.isnan(train_y).any() or np.isinf(train_y).any():
     return [x0[1], x0[0], False]
  valid = True
  popt = [x0[0], x0[1]]
  perr = [np.inf, np.inf]
  try: 
    popt, pcov = curve_fit(fun_fit, train_x, train_y, x0, train_sigma)
    if popt[0] < 0 or popt[1] > 1 or popt[1] < 0:
      valid = False
    perr = np.sqrt(np.diag(pcov))
  except RuntimeError:
    popt[0] = x0[0]
    popt[1] = x0[1]
    valid = False
  #print("[DEBUG-hwlee]find_H0_Omega_L0_single_scipy: H0 = {0}, OL0 = {1}, valid = {2}, errors = {3}".format(popt[0], popt[1], valid, perr))
  return [popt[1], popt[0], valid, perr[1], perr[0]]

"""
  Optimization using Tensorflow
"""
# Define linear regression expression y
def linreg(x, weight):
  y = 0
  for idx in range(len(weight)):
    y = y + weight[idx]*x**idx
  return y

# Define derivative regression expression for no penalty term
def deriv_linreg(x, weight, sigma, reg_factor):
  y = linreg(x, weight)
  dy = []
  for idx in range(len(weight)):
    dy.append(-np.sum((train_y - y)/sigma*x**idx))
  return np.array(dy)
def deriv_linreg_ridge(x, weight, sigma, reg_factor):
  y = linreg(x, weight)
  dy = []
  for idx in range(len(weight)):
    dy.append(-np.sum((train_y - y)/sigma*x**idx) + reg_factor*weight[idx])
  return np.array(dy)
def deriv_linreg_lasso(x, weight, sigma, reg_factor):
  y = linreg(x, weight)
  dy = []
  for idx in range(len(weight)):
    derive = -np.sum((train_y - y)/sigma*x**idx)
    if weight[idx]>0:
      derive = derive + 0.5*reg_factor
    elif weight[idx] < 0:
      derive = derive - 0.5*reg_factor
    dy.append(derive)
  return np.array(dy)

# Define loss function (MSE)
def squared_error(y_pred, train_y, train_sigma, weights, reg_factor):
  return 0.5*np.sum(((y_pred - train_y)/train_sigma)**2)
def squared_error_ridge(y_pred, train_y, train_sigma, weights, reg_factor):
  penalty = reg_factor*np.sum(weights**2)
  return 0.5*np.sum(((y_pred - train_y)/train_sigma)**2) + 0.5*penalty
def squared_error_lasso(y_pred, train_y, train_sigma, weights, reg_factor):
  penalty = reg_factor*np.sum(np.abs(weights))
  return 0.5*np.sum(((y_pred - train_y)/train_sigma)**2) + 0.5*penalty

def generate_weight(rs, weight_lower, weight_upper):
  weight_start = []
  for idx in range(len(weight_lower)):
    r = rs.uniform()
    weight = weight_lower[idx] + r*(weight_upper[idx]-weight_lower[idx])
    weight_start.append(weight)
  return np.array(weight_start)

# linear regression
# train_x : x point of data
# train_y : true value for train_x
# train_sigma : error for each point
# iteration : total number of random start point in parameter space
# epochs : total number of epochs for linear regression
# rate : learning rate for w -> w - rate * gradients
# weight_lower : lower limit of parameters
# weight_upper : upper limit of parameters
# pred_func : prediction function
# deriv_func : derivative of cost function
# error_func : cost function
# reg_factor : regularization factor for Ridge and Lasso cases
# seed : starting seed number for random number (0:random start)
# nprint : number of print step (0: no print)
def minimize_regression(train_x, train_y, train_sigma, iteration, epochs, rate, weight_lower, weight_upper, pred_func, deriv_func, error_func, reg_factor, seed, nprint):
  # iteration for random start points
  min_weights = []
  min_losses = []
  final_epochs = []
  min_weight0 = None
  weight_start0 = None
  min_loss0 = np.inf
  final_epoch0 = None
  if seed > 0:
    randomSeed = seed
  else:
    randomSeed = int(time.time())
    if randomSeed%2 == 0:
      randomSeed = randomSeed + 1
  rs = RandomState(MT19937(SeedSequence(randomSeed)))
  for idx in range(iteration):
    weight_start = generate_weight(rs, weight_lower, weight_upper)
    min_weight, min_loss, final_epoch = minimize_regression_weight(train_x, train_y, train_sigma, epochs, rate, weight_start, pred_func, deriv_func, error_func, reg_factor, nprint)
    min_weights.append(min_weight)
    min_losses.append(min_loss)
    final_epochs.append(final_epoch)
    print("=====================================================================")
    print("Result for Iteration {0}, final_epoch = {1}, w_start = {2}".format(idx, final_epoch, weight_start))
    print("      Loss : {0}, w = {1}".format(min_loss, min_weight))
    print("=====================================================================")
    if min_loss0 > min_loss :
      min_loss0 = min_loss
      min_weight0 = min_weight
      weight_start0 = weight_start
      final_epoch0 = final_epoch
  return min_weight0, min_loss0, final_epoch0, weight_start0, min_weights, min_losses, final_epochs

def minimize_regression_weight(train_x, train_y, train_sigma, epochs, rate, weight_start, pred_func, deriv_func, error_func, reg_factor, nprint):
  # Number of loops for training through all your data to update the parameters
  training_epochs = epochs
  learning_rate = rate

  # declare weights
  weight = weight_start
  #print("Starting weights : {0}".format(weight))
  min_loss = np.inf
  loss0 = np.inf # initial previous loss
  epsilon = np.inf # initial epsilon for increasing check
  for epoch in range(training_epochs):

      # Compute loss within Gradient Tape context
      #print(weight)
      y_predicted = pred_func(train_x, weight)
      loss = error_func(y_predicted, train_y, train_sigma, weight, reg_factor)
      if loss < min_loss:
          min_loss = loss
          min_weight = weight

      if loss > loss0 + epsilon: # increase not decreasing, so stop
          print(f"Epoch count {epoch}: Loss value: {loss}",  end='\r', flush = True)
          print("Break: loss = {0}, loss0 = {1}, epsilon = {2}".format(loss, loss0, epsilon))
          break
      else:
        if loss0 < np.inf :
          epsilon = 0.5*(loss0 - loss)
          if epsilon < 0:
            epsilon = - epsilon
        else:
          epsilon = 0.5*loss
        loss0 = loss # update previous loss value
      # Get gradients
      gradients = deriv_func(train_x, weight, train_sigma, reg_factor)
      #print(gradients)

      # Adjust weights
      weight = weight - learning_rate*gradients

      # Print output
      if (nprint>0 and epoch%nprint == 0) or epoch == training_epochs-1:
          print(f"Epoch count {epoch}: Loss value: {loss}",  end='\r', flush = True)
  #print(min_weight)
  return min_weight, min_loss, epoch