## Python module for LLE calculation

import math
import numpy as np
import os


def calc_LLE(alpha,tau, z, x0):
    # Calculate a phase split between using the NRTL model
    # Set tolerances


    beta = 0.5
    n_comp = len(z)
    nitermax = 100;             # increasing nitermax to 2000 did not help
    TOL_mu = 1e-6
    TOL_beta = 1e-6
    TOL_gbeta = 1e-6
    
    x = x0
    y = np.zeros(n_comp)

    #mol balance for Phase 2
    for i in range(n_comp-1):
        y[i] = (z[i]-(1-beta)*x[i])/beta

    y[n_comp-1] = 1 - sum(y[0:-1])

    gamma_x = get_gamma(x, alpha, tau)
    gamma_y = get_gamma(y, alpha, tau)
    K = gamma_x /gamma_y
    delta_mu = np.absolute(gamma_y * y - gamma_x *x)

    # RR-Algorithm

    nit=0
    while nit < nitermax  and  np.amax(delta_mu) > TOL_mu:   
        nit = nit + 1
        [beta,x,y] = RRSOLVER(n_comp, z, K, nitermax, TOL_beta, TOL_gbeta)
        if beta < TOL_beta:
            x=z
            y=z
            break
        elif beta >= (1-TOL_beta):
            x=z
            y=z
            break
        
        gamma_x = get_gamma(x, alpha, tau)
        gamma_y = get_gamma(y, alpha, tau)

        K = gamma_x / gamma_y
        delta_mu = np.absolute(gamma_y * y - gamma_x * x)
  
## Results
    return x , y, beta 
        
def get_gamma(x, alpha, tau):
    #NRTL activity coefficient calculation
    # AlphaPlace & TauPlace -> Placeholder for certain Parameter
    # If the temperature is an existing input argument, the placeholders are 
    # Parameters to calculate alpha and tau, otherwise the placeholders are
    # already alpha and tau

    n_comp = len(x)
   
# G
    G=np.identity(n_comp)
    for i in range(n_comp):
        for j in range(n_comp):
            if i!=j:
                G[i,j] = math.exp(-alpha[i,j]*tau[i,j])
        
#B(i)
    B = np.zeros(n_comp)
    for i in range(n_comp):
        B[i]=0
        for j in range(n_comp):
            B[i] = B[i] + tau[j,i]*G[j,i]*x[j]
    
#A(i)
    A = np.zeros(n_comp)
    for i in range(n_comp):
        A[i]=0
        for l in range(n_comp):
            A[i] = A[i] + G[l,i]*x[l]
# gamma(i)
    summe_lngamma = np.zeros(n_comp)
    for i in range(n_comp):
        summe_lngamma[i]=0
        for j in range(n_comp):
            summe_lngamma[i] = summe_lngamma[i] + x[j]*G[i,j]/ A[j]*(tau[i,j]-B[j]/A[j])
    
    gamma = np.zeros(n_comp)
    lngamma = np.zeros(n_comp)
    for i in range(n_comp):
        lngamma[i] = B[i]/A[i]+summe_lngamma[i]
        gamma[i] = np.exp(lngamma[i])

    
    return gamma

def RRSOLVER(nc, z, K, ni, TOL_beta, TOL_gbeta):
    # Attention:    beta=n''/n_tot
    #               K(i) = x(i)''/x(i)' = gamma(i)'/gamma(i)''
    #               betafunction: sum(i, x(i)''-x(i)') = sum(i, z(i)*(K(i)-1) /
    #               (1-beta+beta*K(i)))            
    #               
    # nc: Number of components
    # z : feed mole fractions
    # K : k-factors
    # ni: number of iterations

    beta = 0.5
    beta_min = 1e-1
    beta_max = 1-beta_min
    #beta_max = 1/(1-min(K))
    #beta_min = -1/(max(K)-1)

    delta_beta = 1

    for zaehler in range(ni):
        g = 0
        g_alt = g
        for i in range(nc):
            g = g + z[i] *(K[i]-1) / (1-beta+beta * K[i])
           
        if g < 0:
            beta_max = beta
        else: 
            beta_min = beta
                
        g_strich = 0

        for i in range(nc):
            g_strich = g_strich - (np.sqrt(z[i])*(K[i]-1)/(beta*K[i]-beta+1))**2  #- z[i]*(K[i]-1)**2/(beta*K[i]-beta+1)**2
        
        beta_neu = beta - g/g_strich

        
        # normal RR %%%%%%%%%%%%%%%%%%%        
        if beta_neu >= beta_min and beta_neu <= beta_max and ni > 1:
            delta_beta = np.abs(beta-beta_neu)
            beta = beta_neu           
        else: 
            beta_neu = (beta_min + beta_max)/2
            beta = beta_neu
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        delta_g = abs(g_alt - g)
        
        if delta_beta <= TOL_beta and delta_g <= TOL_gbeta:
            break

    l = np.zeros(nc)
    v = np.zeros(nc)
    x = np.zeros(nc)
    y = np.zeros(nc)

    if ni > 1:
        for i in range(nc):
            l[i] = (1-beta)*z[i]/(1-beta+beta*K[i])  
            v[i] = (beta*K[i]*z[i])/(1-beta+beta*K[i])
            x[i] = l[i]/(1-beta)
            y[i] = v[i]/beta   

    return beta,x,y