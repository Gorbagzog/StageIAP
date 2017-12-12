# -*-coding:Utf-8 -*

"""Some code to compute the MhaloPeak as a function of redshift as in Martinez-Manso 2015,
 with the Yang 2012 relation"""


import numpy as np
import matplotlib.pyplot as plt
import time

"""Parameters"""

# I choose to take the model with WMAP7 parameters with the CSMF fit with the SMF1,
# because it seems to be the main result of the paper.

h = 0.7
log_Ms_0 = 10.19 - 2 * np.log10(h)
gamma1 = -0.62
log_M1_0 = 10.69 - np.log10(h)
gamma2 = -0.24
alpha0 = 0.29
gamma3 = 0.18
log_beta_0 = np.log10(8.15)
gamma4 = -0.07
gamma5 = 0.33

"""Functions"""

def log_Ms0(z):
    return log_Ms_0 + gamma1 * z

def log_M1(z):
    return log_M1_0 + gamma2 * z

def alpha(z):
    return alpha0 + gamma3 * z

def log_beta(z):
    return min(log_beta_0 + gamma4 * z + gamma5 * z**2, 2)

def log_Ms(log_Mh, z):
    return log_Ms0(z) + (alpha(z) + 10**log_beta(z)) * (
           log_Mh - log_M1(z)) - 10**log_beta(z) * np.log10(1 + 10**(log_Mh - log_M1(z)))

# def Ms(z):
#     return 10**log_Ms0 *

"""Compute data vectors"""

log_Mh = np.linspace(10, 14, 1000)

numpoints = 1000
zmax = 3.75
redshift = np.linspace(0, zmax, numpoints)
MhaloPeak = np.zeros(numpoints)

t = time.time()
for i in range(numpoints):
    MhaloPeak[i] = log_Mh[np.argmax(log_Ms(log_Mh, redshift[i]) - log_Mh)]


"""Plot"""
temp = [redshift, MhaloPeak]
np.savetxt('MhaloPeakYang.txt', temp)

plt.figure()
plt.plot(temp[0], temp[1])
plt.show()
