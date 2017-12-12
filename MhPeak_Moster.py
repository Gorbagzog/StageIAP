# -*-coding:Utf-8 -*

"""Some code to compute the MhaloPeak as a function of redshift as in Martinez-Manso 2015,
 with the Moster 2013 relation"""

import numpy as np
import matplotlib.pyplot as plt
import time

"""Parameters"""

M1_0 = 11.590
M1_1 = 1.195

beta_0 = 1.376
beta_1 = -0.826

gamma_0 = 0.608
gamma_1 = 0.329

"""Functions"""

def log_M1(z):
    return M1_0 + M1_1 * z/(1+z)

def beta(z):
    return beta_0 + beta_1 * z/(1+z)

def gamma(z):
    return gamma_0 + gamma_1 * z/(1+z)

def log_MhaloPeak(z):
    return log_M1(z) + 1/(beta(z) + gamma(z)) * np.log10(beta(z)/gamma(z))

"""Compute MhaloPeak"""

redshift = np.linspace(0, 4, 1000)
MhaloPeak = log_MhaloPeak(redshift)

temp = [redshift, MhaloPeak]

np.savetxt('MhaloPeakMoster.txt', temp)

"""Plot"""

# plt.figure()
# plt.plot(redshift, MhaloPeak)
# plt.show()