# -*-coding:Utf-8 -*

"""Some code to compute the MhaloPeak as a function of redshift as in Ishikawa et al 2017 or
Coupon et al 2017 draft, with the Behroozi et al 2010 and 2013 relations"""

import numpy as np
import matplotlib.pyplot as plt


"""Parameters"""

M1_0 = 11.514
M1_a = -1.793
M1_z = -0.251

e_0 = -1.777
e_a = -0.006
e_z = 0
e_a2 = -0.119

alpha_0 = -1.412
alpha_a = 0.731

delta_0 = 3.508
delta_a = 2.608
delta_z = -0.043

gamma_0 = 0.316
gamma_a = 1.319
gamma_z = 0.279

"""Functions"""

def af(z):
    return 1 / (1 + z)

def nu(a):
    return np.exp(-4 * a**2)

def log_M1(a, z):
    return M1_0 + (M1_a * (a - 1) + M1_z * z) * nu(a)

def log_e(a, z):
    return e_0 + (e_a * (a - 1) + e_z * z) * nu(a) + e_a2 * (a - 1)

def alpha(a):
    return alpha_0 + (alpha_a * (a - 1)) * nu(a)

def delta(a, z):
    return delta_0 + (delta_a * (a - 1) + delta_z * z ) * nu(a)

def gamma(a, z):
    return gamma_0 + (gamma_a * (a - 1) + gamma_z * z) * nu(a)

def f(x, a, z):
    return - np.log10(10**(alpha(a) * x) + 1) + delta(a, z) * (
        np.log10(1 + np.exp(x)))**gamma(a, z) / (1 + np.exp(10**(-x)))

def log_Ms(log_Mh, z):
    a = af(z)
    return log_e(a, z) + log_M1(a, z) + f(log_Mh - log_M1(a, z), a, z) - f(0, a, z)


"""Find maximum for each redshift"""

log_Mh = np.linspace(10, 14, 1000)

# plt.figure()
# for i in range(6):
#     plt.plot(log_Mh, log_Ms(log_Mh, i) - log_Mh)
# plt.show()

redshift = np.linspace(0, 10, 1000)
MhaloPeak = np.array(1000)

for i in range(1000):
    MhaloPeak[i] = np.argmax(log_Ms(log_Mh, redshift[i] - log_Mh))[0]
