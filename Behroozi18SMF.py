#!/usr/bin/env python3
# -*-coding:Utf-8 -*

"""Script to compute the SMF from Behroozi et al. 2013 and Behroozi et al. 2018"""

import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.lss import mass_function

from MCMC_SHMR_main import *


"""Load HMF"""
# redshiftsbin = np.array([0.37, 0.668, 0.938, 1.286, 1.735, 2.220, 2.683, 3.271, 3.926, 4.803])
global redshiftsbin
redshiftsbin = np.array([0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
numzbin = redshiftsbin.size


hmf=[]
mdef='200m'
hmf_name = 'tinker08'
cosmo = cosmology.setCosmology('planck15')
redshift_haloes = redshiftsbin


"""Use the Colossus module for the HMF"""
print('Use '+hmf_name+' HMF in Planck15 cosmo from Colossus module')
if hmf_name == ('watson13' or 'bhattacharya11'):
    print(hmf_name)
    mdef='fof'
else:
    mdef = '200m'
print('Use '+mdef+' for the SO defintion.')
redshift_haloes = redshiftsbin

log_Mh = np.linspace(10, 15, num=1000)

M = 10**log_Mh * cosmo.h
for i in range(numzbin):
    hmf.append(
        np.transpose(
            np.array(
                [np.log10(M / cosmo.h),
                    np.log10(mass_function.massFunction(
                        M, redshift_haloes[i], mdef = mdef, model =hmf_name, q_out = 'dndlnM'
                    ) * np.log(10) * cosmo.h**3
                    ## Mass functions are in h^3 Mpc^-3, and need to multiply by ln(10) to have dndlog10m
                    )]
                )
            )
        )



"""Behroozi 2018"""
"""Parameters"""

paramfile = '../Data/umachine-edr/data/smhm/params/smhm_med_params.txt' # -> the one showed on the plots of B18
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
params18 = dict(zip(names, param_list))

"""Functions"""

def af(z):
    return 1 / (1+z)

def log_M1(a, z):
    return params18['M_1'] + params18['M_1_A'] * (a - 1.) - params18['M_1_A2'] * np.log(a) + params18['M_1_Z'] * z

def eps(a, z):
    return params18['EFF_0'] + params18['EFF_0_A'] * (a - 1.) - params18['EFF_0_A2'] * np.log(a) + params18['EFF_0_Z'] * z

def alpha(a, z):
    return params18['ALPHA'] + params18['ALPHA_A'] * (a - 1.) - params18['ALPHA_A2'] * np.log(a) + params18['ALPHA_Z'] * z

def beta(a, z):
    return params18['BETA'] + params18['BETA_A'] * (a - 1.) + params18['BETA_Z'] * z

def delta():
    return params18['DELTA']

def log_gamma(a, z):
    return params18['GAMMA'] + params18['GAMMA_A'] * (a - 1.) + params18['GAMMA_Z'] * z

def log_Ms18(log_Mh, z):
    a = af(z)
    print(log_M1(a, z))
    x = log_Mh - log_M1(a, z)
    return log_M1(a, z) + eps(a, z) - np.log10(10**(-alpha(a, z) * x) + 10**(-beta(a, z) * x)) + 10**(log_gamma(a, z)) * np.exp(-0.5 * (x / delta())**2)


def log_phi18(log_Mh, idx_z, z, params):
    epsilon = 0.001
    log_Ms1 = log_Ms18(log_Mh, z)
    log_Ms2 = log_Ms18(log_Mh + epsilon, z)
    # Select the index of the HMF corresponding to the halo masses
    index_Mh = np.argmin(
        np.abs(
            np.tile(hmf[idx_z][:, 0], (len(log_Mh), 1)) -
            np.transpose(np.tile(log_Mh, (len(hmf[idx_z][:, 0]), 1)))
        ), axis=1)
    # if np.any(hmf[idx_z][index_Mh, 1] < -100):
    #     # print('HMF not defined')
    #     return log_Ms1 * 0. + 1
    # else:
    log_phidirect = hmf[idx_z][index_Mh, 1] - np.log10((log_Ms2 - log_Ms1)/epsilon)
    return log_phidirect


def log_phi18_true(log_Mh, idx_z, z, params):
    epsilon = 0.0001 * log_Mh
    ksi=0.1
    logphi1 =log_phi18(log_Mh, idx_z, z, params)
    logphi2 = log_phi18(log_Mh + epsilon, idx_z, z, params)
    logphitrue = logphi1 + ksi**2 / 2 * np.log(10) * ((logphi2 - logphi1)/epsilon)**2
    return logphitrue


"""Compute log_Ms and phi corresponding to a given log_Mh and HMF"""

# log_Mh = np.linspace(10, 15, num=1000)

# for idx_z in range(redshiftsbin.size):
# # for idx_z in range(1):
#     z = redshiftsbin[idx_z]
#     log_phi = log_phi18(log_Mh, idx_z, z, params18)

#     # print(log_phi)
#     plt.plot(log_Ms18(log_Mh, z), log_phi, label='z='+str(z))
#     # plt.plot(log_Mh, log_Ms18(log_Mh, z))

# plt.ylim(-7, 0)
# plt.legend()
# plt.show()


z = np.linspace(0, 10)
a= af(z)

plt.figure()
plt.plot(z, log_M1(a, z) )
plt.show()

plt.figure()
plt.plot(z, eps(a, z) )
plt.show()

plt.figure()
plt.plot(z, alpha(a, z) )
plt.show()

plt.figure()
plt.plot(z, beta(a, z) )
plt.show()

plt.figure()
plt.plot(z, log_gamma(a, z) )
plt.show()
