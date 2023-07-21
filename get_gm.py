import numpy as np
import matplotlib.pyplot as plt
import py_nrtl
from num_dual import derive2

import torch
from torch import Tensor
import numpy as np


class Dual3:
    def __init__(self, re, v1, v2):
        self.re = re
        self.v1 = v1
        self.v2 = v2

    @classmethod
    def diff(cls, re):
        return cls(re, 1, 0)

    def __repr__(self):
        return f"{self.re} + {self.v1}v1 + {self.v2}v2"

    def __add__(self, other):
        if isinstance(other, Dual3):
            return Dual3(self.re + other.re, self.v1 + other.v1, self.v2 + other.v2)
        return Dual3(self.re + other, self.v1, self.v2)

    def __neg__(self):
        return Dual3(-self.re, -self.v1, -self.v2)

    def __sub__(self, other):
        if isinstance(other, Dual3):
            return Dual3(self.re - other.re, self.v1 - other.v1, self.v2 - other.v2)
        return Dual3(self.re - other, self.v1, self.v2)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if isinstance(other, Dual3):
            return Dual3(
                self.re * other.re,
                self.v1 * other.re + self.re * other.v1,
                self.v2 * other.re + 2 * self.v1 * other.v1 + self.re * other.v2,
            )
        return Dual3(self.re * other, self.v1 * other, self.v2 * other)

    def chain_rule(self, f0, f1, f2):
        return Dual3(f0, f1 * self.v1, f2 * self.v1**2 + f1 * self.v2)

    def recip(self):
        rec = 1 / self.re
        return self.chain_rule(rec, -(rec**2), 2 * rec**3)

    def __truediv__(self, other):
        if isinstance(other, Dual3):
            return self * other.recip()
        return Dual3(self.re / other, self.v1 / other, self.v2 / other)

    def __rtruediv__(self, other):
        return other * self.recip()

    def log(self):
        rec = 1 / self.re
        l = self.re.log() if isinstance(self.re, Tensor) else np.log(self.re)
        return self.chain_rule(l, rec, -(rec**2))

    def exp(self):
        e = self.re.exp() if isinstance(self.re, Tensor) else np.exp(self.re)
        return self.chain_rule(e, e, e)

    def sqrt(self):
        s = self.re.sqrt() if isinstance(self.re, Tensor) else np.sqrt(self.re)
        return self.chain_rule(s, 0.5 / s, -0.25 / (s * s * s))

    def dot(self, other):
        return Dual3(
            self.re.dot(other.re),
            self.v1.dot(other.re) + self.re.dot(other.v1),
            self.v2.dot(other.re) + 2 * self.v1.dot(other.v1) + self.re.dot(other.v2),
        )
        

Dual3.__radd__ = Dual3.__add__
Dual3.__rmul__ = Dual3.__mul__

def get_gamma(x, alpha, tau):

    n_comp = 2
    
    # G
    G=torch.tensor(np.identity(n_comp))
    for i in range(n_comp):
        for j in range(n_comp):
            if i!=j:
                G[i,j] = torch.exp(-alpha[i,j]*tau[i,j])

    #Binary dot product
    
    x = [x, 1-x]
    
    # Calculating B
    B1 = tau[0][0]*G[0][0]*x[0] + tau[0][1]*G[1][0]*x[0] + tau[1][0]*G[0][1]*x[0] + tau[1][1]*G[1][1]*x[0]
    B2 = tau[0][0]*G[0][0]*x[1] + tau[0][1]*G[1][0]*x[1] + tau[1][0]*G[0][1]*x[1] + tau[1][1]*G[1][1]*x[1]

    # Calculating A
    A1 = G[0][0]*x[0] + G[1][0]*x[1]
    A2 = G[0][1]*x[0] + G[1][1]*x[1]
    
    # gamma(i)
    summe_lngamma1 = x[0]*G[0][0]/A1*(tau[0][0]-B1/A1) + x[1]*G[0][1]/A2*(tau[0][1]-B2/A2)
    summe_lngamma2 = x[0]*G[1][0]/A1*(tau[1][0]-B1/A1) + x[1]*G[1][1]/A2*(tau[1][1]-B2/A2)
    
    gamma = torch.zeros(n_comp)
    lngamma = torch.zeros(n_comp)
    
    lngamma1 = B1/A1+summe_lngamma1
    lngamma2 = B2/A2+summe_lngamma2
    
    gamma1 = Dual3.exp(lngamma1)
    gamma2 = Dual3.exp(lngamma2)
   
    return [gamma1, gamma2]

def ideal_mix_g(x):
    x1 = x 
    x2 = 1 - x1
    ln_x1 = Dual3.log(x1) if x1 != 0 else 0
    ln_x2 = Dual3.log(x2) if x2 != 0 else 0
    return x1 * ln_x1 + x2 * ln_x2

def binary_mixture_g_profile(alpha, tau, x1):
    R = 8.314  # J/(mol*K), gas constant
    T = 298.15  # K, temperature
    
    x1 = Dual3.diff(x1)
        
    gamma = get_gamma(x1, alpha, tau)
    gamma1, gamma2 = gamma[0], gamma[1]

    ideal_mix = ideal_mix_g(x1)
    excess_mix = x1 * Dual3.log(gamma1) + (1 - x1) * Dual3.log(gamma2)

    gibbs_mix = R * T * (ideal_mix + excess_mix)
    
    gibbs_mix_value = gibbs_mix.re
    gibbs_mix_derive = gibbs_mix.v1
    gibbs_mix_derive2 = gibbs_mix.v2
    

    return gibbs_mix_value, gibbs_mix_derive, gibbs_mix_derive2

if __name__ == "__main__":

    #alpha = torch.tensor([[0. , 0.23449744],
    #                [0.23449744, 0. ]], requires_grad=True)
    #tau = torch.tensor([[0. , 1.1661877 ],
    #                [1.17877817, 0. ]], requires_grad=True)

    alpha = torch.tensor([[0. , 0.2341091], [0.2341091, 0. ]], requires_grad=True)
    tau = torch.tensor([[0. , 0.36579808], [0.38160984, 0. ]], requires_grad=True)

    x1 = torch.tensor(0.5, requires_grad=True)

    gibbs_mix, gibbs_mix_derive, gibbs_mix_derive2 = binary_mixture_g_profile(alpha, tau, x1)
    
    print(gibbs_mix)
    print(gibbs_mix_derive)
    print(gibbs_mix_derive2)
    
    # calculate the mse loss
    
    gibbs_mix_derive2.backward()

    # print the gradients of alpha and tau
    print("Gradient of alpha")
    print(alpha.grad)
    print("Gradient of tau")
    print(tau.grad)


