# -*-coding:Utf-8 -*

"""Some code to compute the MhaloPeak as a function of redshift from the Behroozi+2018 paper"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import time

"""Parameters"""

paramfile = '../Data/umachine-edr/data/smhm/params/smhm_true_med_params.txt' # -> the one showed on the plots of B18
# paramfile = '../Data/umachine-edr/data/smhm/params/smhm_med_params.txt'

# Load params
param_file = open(paramfile, "r")
param_list = []
allparams = []
for line in param_file:
    param_list.append(float((line.split(" "))[1]))
    allparams.append(line.split(" "))

if (len(param_list) != 20):
    print("Parameter file not correct length.  (Expected 20 lines, got %d)." % len(param_list))
    quit()

names = "EFF_0 EFF_0_A EFF_0_A2 EFF_0_Z M_1 M_1_A M_1_A2 M_1_Z ALPHA ALPHA_A ALPHA_A2 ALPHA_Z BETA BETA_A BETA_Z DELTA GAMMA GAMMA_A GAMMA_Z CHI2".split(" ");
params = dict(zip(names, param_list))

"""Functions"""

def af(z):
    return 1 / (1+z)

def log_M1(a, z):
    return params['M_1'] + params['M_1_A'] * (a - 1.) - params['M_1_A2'] * np.log(a) + params['M_1_Z'] * z

def eps(a, z):
    return params['EFF_0'] + params['EFF_0_A'] * (a - 1.) - params['EFF_0_A2'] * np.log(a) + params['EFF_0_Z'] * z

def alpha(a, z):
    return params['ALPHA'] + params['ALPHA_A'] * (a - 1.) - params['ALPHA_A2'] * np.log(a) + params['ALPHA_Z'] * z

def beta(a, z):
    return params['BETA'] + params['BETA_A'] * (a - 1.) + params['BETA_Z'] * z

def delta():
    return params['DELTA']

def log_gamma(a, z):
    return params['GAMMA'] + params['GAMMA_A'] * (a - 1.) + params['GAMMA_Z'] * z

def log_Ms(log_Mh, z):
    a = af(z)
    x = log_Mh - log_M1(a, z)
    return log_M1(a, z) + eps(a, z) - np.log10(10**(-alpha(a, z) * x) + 10**(-beta(a, z) * x)) + 10**(log_gamma(a, z)) * np.exp(-0.5 * (x / delta())**2)

# def af(z):
#     return 1 / (1 + z)

# def nu(a):
#     return np.exp(-4 * a**2)

# def log_M1(a, z):
#     return M1_0 + (M1_a * (a - 1) + M1_z * z) * nu(a)

# def log_e(a, z):
#     return e_0 + (e_a * (a - 1) + e_z * z) * nu(a) + e_a2 * (a - 1)

# def alpha(a):
#     return alpha_0 + (alpha_a * (a - 1)) * nu(a)

# def delta(a, z):
#     return delta_0 + (delta_a * (a - 1) + delta_z * z ) * nu(a)

# def gamma(a, z):
#     return gamma_0 + (gamma_a * (a - 1) + gamma_z * z) * nu(a)

# def f(x, a, z):
#     return - np.log10(10**(alpha(a) * x) + 1) + delta(a, z) * (
#         np.log10(1 + np.exp(x)))**gamma(a, z) / (1 + np.exp(10**(-x)))

# def log_Ms(log_Mh, z):
#     a = af(z)
#     return log_e(a, z) + log_M1(a, z) + f(log_Mh - log_M1(a, z), a, z) - f(0, a, z)


"""Find maximum for each redshift"""

log_Mh = np.linspace(10, 14, 10000)

# plt.figure()
# for i in range(6):
#     plt.plot(log_Mh, log_Ms(log_Mh, i) - log_Mh)
# plt.show()

numpoints = 1000
zmax = 6
redshift = np.linspace(0, zmax, numpoints)
MhaloPeak = np.zeros(numpoints)

t = time.time()
for i in range(numpoints):
    MhaloPeak[i] = log_Mh[np.argmax(log_Ms(log_Mh, redshift[i]) - log_Mh)]


print(time.time() - t)

"""Plot"""

temp = [redshift, MhaloPeak]

plt.figure()
plt.plot(temp[0], temp[1])
plt.plot(redshift, 14.5-0.35*redshift, '--', label='B18 max Mhalo for the SMHM')

plt.ylabel('$\mathrm{log}_{10}(M_{\mathrm{h}}^{\mathrm{peak}})$', size=12)
plt.xlabel('z', size=12)
plt.title('B18, smhm_true_med_params')
plt.legend()
plt.tight_layout()
plt.show()


"""Log Mhwith the maximal cut defined in gen_smhm.py B18 script"""

plt.figure()
# for z in range(10):
for z in [0]:
    smhm_max = 14.5-0.35*z
    log_Mh =  np.linspace(10, smhm_max, 1000)
    plt.plot(log_Mh, log_Ms(log_Mh, z) - log_Mh, label='z='+str(z))
plt.legend()



"""Save file"""

np.savetxt('MhaloPeakB18.txt', temp)

# test = np.loadtxt('MhaloPeakBehroozi.txt')
# print(test[0])
# # print(test[1])

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

# np.save('SHMR_Behroozi_z0', np.array([log_Ms(log_Mh, redshiftsbin[0]), log_Mh]))








"""Compare with B18 script"""

# names = "EFF_0 EFF_0_A EFF_0_A2 EFF_0_Z M_1 M_1_A M_1_A2 M_1_Z ALPHA ALPHA_A ALPHA_A2 ALPHA_Z BETA BETA_A BETA_Z DELTA GAMMA GAMMA_A GAMMA_Z CHI2".split(" ");
# params = dict(zip(names, param_list))

# z = 5
# #Print SMHM relation
# a = 1.0/(1.0+z)
# a1 = a - 1.0
# lna = math.log(a)
# zparams = {}
# zparams['m_1'] = params['M_1'] + a1*params['M_1_A'] - lna*params['M_1_A2'] + z*params['M_1_Z']
# zparams['sm_0'] = zparams['m_1'] + params['EFF_0'] + a1*params['EFF_0_A'] - lna*params['EFF_0_A2'] + z*params['EFF_0_Z']
# zparams['alpha'] = params['ALPHA'] + a1*params['ALPHA_A'] - lna*params['ALPHA_A2'] + z*params['ALPHA_Z']
# zparams['beta'] = params['BETA'] + a1*params['BETA_A'] + z*params['BETA_Z']
# zparams['delta'] = params['DELTA']
# zparams['gamma'] = 10**(params['GAMMA'] + a1*params['GAMMA_A'] + z*params['GAMMA_Z'])

# smhm_max = 14.5-0.35*z
# print('#Log10(Mpeak/Msun) Log10(Median_SM/Msun) Log10(Median_SM/Mpeak)')
# print('#Mpeak: peak historical halo mass, using Bryan & Norman virial overdensity.')
# print('#Overall fit chi^2: %f' % params['CHI2'])
# if (params['CHI2']>200):
#     print('#Warning: chi^2 > 200 implies that not all features are well fit.  Comparison with the raw data (in data/smhm/median_raw/) is crucial.')

# hm = [x*0.05 for x in range(int(10.5*20),int(smhm_max*20+1),1)]
# sm = np.empty(np.size(hm))

# for i in range(np.size(hm)):
#     m = hm[i]
#     dm = m-zparams['m_1'];
#     dm2 = dm/zparams['delta'];
#     sm[i] = zparams['sm_0'] - math.log10(10**(-zparams['alpha']*dm) + 10**(-zparams['beta']*dm)) + zparams['gamma']*math.exp(-0.5*(dm2*dm2));
#     # print("%.2f %.6f %.6f" % (m,sm,sm-m))


# plt.plot(sm, hm)
# plt.show()