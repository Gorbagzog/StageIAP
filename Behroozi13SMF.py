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

ksi_0 = 0.218
ksi_a = -0.023

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

def ksi(a):
    return ksi_0 + ksi_a*(a-1)

def f(x, a, z):
    return - np.log10(10**(alpha(a) * x) + 1) + delta(a, z) * (
        np.log10(1 + np.exp(x)))**gamma(a, z) / (1 + np.exp(10**(-x)))

def log_Ms13(log_Mh, z):
    a = af(z)
    return log_e(a, z) + log_M1(a, z) + f(log_Mh - log_M1(a, z), a, z) - f(0, a, z)


def log_phi13(log_Mh, idx_z, z):
    epsilon = 0.001
    log_Ms1 = log_Ms13(log_Mh, z)
    log_Ms2 = log_Ms13(log_Mh + epsilon, z)
    # Select the index of the HMF corresponding to the halo masses
    index_Mh = np.argmin(
        np.abs(
            np.tile(hmf[idx_z][:, 0], (len(log_Mh), 1)) -
            np.transpose(np.tile(log_Mh, (len(hmf[idx_z][:, 0]), 1)))
        ), axis=1)
    # if np.any(hmf[idx_z][index_Mh, 1] < -100):
    #     # print('HMF not defined')
    #     return log_Ms1 * 0. + 1
    log_phidirect = hmf[idx_z][index_Mh, 1] - np.log10((log_Ms2 - log_Ms1)/epsilon)
    return log_phidirect


def log_phi13_true(log_Mh, idx_z, z):
    epsilon = 0.0001 * log_Mh
    logphi1 =log_phi13(log_Mh, idx_z, z)
    logphi2 = log_phi13(log_Mh + epsilon, idx_z, z)
    a = af(z)
    logphitrue = logphi1 + ksi(a) **2 / 2 * np.log(10) * ((logphi2 - logphi1)/epsilon)**2
    return logphitrue


"""Compute log_Ms and phi corresponding to a given log_Mh and HMF"""

# log_Mh = np.linspace(7, 20, num=1000)

for idx_z in range(redshiftsbin.size):
# for idx_z in range(1):
    z = redshiftsbin[idx_z]
    log_phi = log_phi13_true(log_Mh, idx_z, z)

    print(log_phi)
    plt.plot(log_Ms13(log_Mh, z), log_phi)
    # plt.plot(log_Mh, log_Ms13(log_Mh, z))

plt.ylim(-7, -1)
plt.show()