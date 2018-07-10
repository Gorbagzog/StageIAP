# -*-coding:Utf-8 -*

"""Some code to compute the MhaloPeak as a function of redshift as in Ishikawa et al 2017 or
Coupon et al 2017 draft, with the Behroozi et al 2010 and 2013 relations"""

import numpy as np
import matplotlib.pyplot as plt
import time

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

log_Mh = np.linspace(10, 14, 10000)

# plt.figure()
# for i in range(6):
#     plt.plot(log_Mh, log_Ms(log_Mh, i) - log_Mh)
# plt.show()

numpoints = 1000
zmax = 8
redshift = np.linspace(0, zmax, numpoints)
MhaloPeak = np.zeros(numpoints)

t = time.time()
for i in range(numpoints):
    MhaloPeak[i] = log_Mh[np.argmax(log_Ms(log_Mh, redshift[i]) - log_Mh)]


print(time.time() - t)

"""Plot"""

# temp = [redshift, MhaloPeak]

# plt.figure()
# plt.plot(temp[0], temp[1])
# plt.show()

"""Save file"""

# np.savetxt('MhaloPeakBehroozi.txt', temp)

# test = np.loadtxt('MhaloPeakBehroozi.txt')
# print(test[0])
# print(test[1])

"""PLot Ms vs Mh"""
redshiftcosmos = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
redshiftsbin = (redshiftcosmos[1:]+redshiftcosmos[:-1])/2
plt.figure()
numzbin=10
for i in range(numzbin):
    plt.plot(log_Ms(log_Mh, redshiftsbin[i]), log_Mh, label='z='+str(redshiftsbin[i]))
plt.xlabel('Log($M_{*}/M_{\odot}$)', size=20)
plt.ylabel('Log($M_{h}/M_{\odot}$)', size=20)
plt.legend()
plt.tight_layout()
plt.show()

np.save('SHMR_Behroozi_z0', np.array([log_Ms(log_Mh, redshiftsbin[0]), log_Mh]))

