#!/usr/bin/env python3
# -*-coding:Utf-8 -*

"""Script to compute the SMF from Behroozi et al. 2013 and Behroozi et al. 2018"""

import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from astropy.cosmology import LambdaCDM, Planck15

from MCMC_SHMR_main import *


"""Load SMF"""

"""Load B10 plot from plot_digitizer"""

tmp = np.loadtxt('../Data/B10_plotdigit.txt')
log_Ms_B10 = tmp[:, 0]
log_phi_B10 = tmp[:, 1]

"""Parameters"""

Ms0_0 = 10.62
Ms0_a = 0.55

M1_0 = 12.35
M1_a = 0.28

beta_0 = 0.44
beta_a = 0.18

delta_0 = 0.57
delta_a = 0.17

gamma_0 = 1.56
gamma_a = 2.51

ksi = 0.15



"""Functions"""

def af(z):
    return 1 / (1 + z)


def logM1(a):
    return M1_0 + M1_a*(a-1)

def logMs0(a):
    return Ms0_0 + Ms0_a * (a-1)

def beta(a):
    return beta_0 + beta_a*(a-1)

def delta(a):
    return delta_0 + delta_a*(a-1)
    
def gamma(a):
    return gamma_0 + gamma_a*(a-1)
    

"""Compute the SMF"""

paramfile = 'MCMC_param.ini'
params = load_params(paramfile)

smf = load_smf(params)
hmf = load_hmf(params)

z = 0.1
idx_z = 0

select = np.where(smf[idx_z][:, 1] > -40)[0]  # select points where the smf is defined
# We choose to limit the fit only for abundances higher than 10**-7
logMs = smf[idx_z][select[:], 0]

log_Mh = logMh(logMs, logM1(af(z)), logMs0(af(z)), beta(af(z)), delta(af(z)), gamma(af(z)))

# plt.plot(log_Mh, logMs)
# plt.show()

idx_z=0

logphidir  = log_phi_direct(logMs, hmf, idx_z, logM1(af(z)), logMs0(af(z)), beta(af(z)), delta(af(z)), gamma(af(z)))
# logphidir  = log_phi_direct(logMs, hmf, idx_z, M1_0, Ms0_0, beta_0, delta_0, gamma_0)
logphitrue = log_phi_true(logMs, hmf, idx_z, params, logM1(af(z)), logMs0(af(z)), beta(af(z)), delta(af(z)), gamma(af(z)), ksi)

plt.plot(logMs, logphidir, label='B10 best fit, no convolution')
plt.plot(logMs, logphitrue, label='B10 best fit, with convolution')
plt.plot(log_Ms_B10, log_phi_B10, '--', label='B10 (plot digitizer)')
plt.legend()
plt.xlabel('log($M_*$)')
plt.ylabel('log($\phi$)')
plt.show()