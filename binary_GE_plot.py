import numpy as np
import matplotlib.pyplot as plt
import py_nrtl
from num_dual import derive2


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
                G[i,j] = np.exp(-alpha[i,j]*tau[i,j])
        

    B = (tau*G).T.dot(x)
    A = G.T.dot(x)
    
# gamma(i)
    summe_lngamma = [0 for i in range(n_comp)]
    for i in range(n_comp):
        summe_lngamma[i]=0
        for j in range(n_comp):
            summe_lngamma[i] = summe_lngamma[i] + x[j]*G[i,j]/ A[j]*(tau[i,j]-B[j]/A[j])
    
    gamma = derive2(np.zeros(n_comp))
    lngamma = derive2(np.zeros(n_comp))
    for i in range(n_comp):
        lngamma[i] = B[i]/A[i]+summe_lngamma[i]
        gamma[i] = np.exp(lngamma[i])

    
    return gamma

def ideal_mix_g(x):
    x1 = x[0]
    x2 = x[1]
    ln_x1 = np.log(x1) if x1 != 0 else 0
    ln_x2 = np.log(x2) if x2 != 0 else 0
    return x1 * ln_x1 + x2 * ln_x2

def binary_mixture_g_profile(alpha, tau, n_points=5000):
    R = 8.314  # J/(mol*K), gas constant
    T = 298.15  # K, temperature

    x1_values = np.linspace(0, 1, n_points)
    x1_values = np.delete(x1_values, 0)
    x1_values = np.delete(x1_values, -1)
    x2_values = 1 - x1_values

    gibbs_mix_value = np.zeros(n_points-2)
    gibbs_mix_derive = np.empty(n_points-2)
    gibbs_mix_derive2 = np.empty(n_points-2)

    for i, x1 in enumerate(x1_values):
        
        x1 = derive2(x1)
        x2 = 1 - x1
        x = np.array([x1, x2])
        gamma = get_gamma(x, alpha, tau)
        gamma1, gamma2 = gamma[0], gamma[1]

        ideal_mix = ideal_mix_g(x)
        excess_mix = x1 * np.log(gamma1) + x2 * np.log(gamma2)

        gibbs_mix = R * T * (ideal_mix + excess_mix)
        
        gibbs_mix_value[i] = gibbs_mix.value
        gibbs_mix_derive[i] = gibbs_mix.first_derivative
        gibbs_mix_derive2[i] = gibbs_mix.second_derivative
        

    return x1_values, gibbs_mix_value, gibbs_mix_derive, gibbs_mix_derive2


alpha = np.array([[0. , 0.23449744],
                 [0.23449744, 0. ]])
tau = np.array([[0. , 1.1661877 ],
                [1.17877817, 0. ]])

#alpha = np.array([[0. , 0.2341091], [0.2341091, 0. ]])
#tau = np.array([[0. , 2.36579808], [2.38160984, 0. ]])

x1, gibbs_mix, gibbs_mix_derive, gibbs_mix_derive2 = binary_mixture_g_profile(alpha, tau)

plt.plot(x1, gibbs_mix, linewidth=2, color='blue')
zero_line = np.zeros(len(x1))
plt.plot(x1, zero_line, linewidth=1, color='black', linestyle='--')
plt.xlabel("Mole Fraction of x1", fontsize=14)
plt.ylabel("Gibbs Free Energy of Mixing (J/mol)", fontsize=14)
plt.title("Gibbs Free Energy of Mixing vs. Composition", fontsize=16)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0, 1)
plt.tight_layout()


plt.show()
plt.savefig("gibbs_free_energy_of_mixing.png", dpi=600)
plt.clf()

plt.plot(x1, gibbs_mix_derive, linewidth=2, color='blue')
zero_line = np.zeros(len(x1))
plt.plot(x1, zero_line, linewidth=1, color='black', linestyle='--')
plt.xlabel("Mole Fraction of x1", fontsize=14)
plt.ylabel("d Gibbs Free Energy of Mixing (J/mol)", fontsize=14)
plt.title("Gibbs Free Energy of Mixing vs. Composition", fontsize=16)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0, 1)
plt.tight_layout()

plt.show()
plt.savefig("gibbs_free_energy_of_mixing_d.png", dpi=600)
plt.clf()

plt.plot(x1, gibbs_mix_derive2, linewidth=2, color='blue')
zero_line = np.zeros(len(x1))
plt.plot(x1, zero_line, linewidth=1, color='black', linestyle='--')
plt.xlabel("Mole Fraction of x1", fontsize=14)
plt.ylabel("d Gibbs Free Energy of Mixing (J/mol)", fontsize=14)
plt.title("Gibbs Free Energy of Mixing vs. Composition", fontsize=16)
plt.grid(True)
plt.ylim(-1e5, 1e5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0, 1)
plt.tight_layout()

plt.show()
plt.savefig("gibbs_free_energy_of_mixing_d2.png", dpi=600)
