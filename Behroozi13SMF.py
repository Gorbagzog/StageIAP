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

redshiftsbin = np.array([0.37, 0.668, 0.938, 1.286, 1.735, 2.220, 2.683, 3.271, 3.926, 4.803])
redshifts = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
# redshiftsbin = np.array([0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
numzbin = redshiftsbin.size


smf = []
tmp = []
print('Use the COSMOS Schechter fit SMF')
for i in range(numzbin):
    tmp.append(np.loadtxt(
        '../Data/Davidzon/Davidzon+17_SMF_v3.0/mf_mass2b_fl5b_tot_VmaxFit2D'
        + str(i) + '.dat')
    )
    # Do not take points that are below -1000
    smf.append(
        tmp[i][np.where(
            np.logical_and(
                np.logical_and(
                    tmp[i][:, 1] > -1000,
                    tmp[i][:, 2] > -1500),
                tmp[i][:, 3] > -1000),
    ), :][0])
    # Take the error bar values as in Vmax data file, and not the boundaries.
    # /!\ Warning, in the Vmax file, smf[:][:,2] gives the higher bound and smf[:][:,3] the lower bound.
    # It is the inverse for the Schechter fit
    # I use the Vmax convention to keep the same structure.
    temp = smf[i][:, 1] - smf[i][:, 2]
    smf[i][:, 2] = smf[i][:, 3] - smf[i][:, 1]
    smf[i][:, 3] = temp

D17_Cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
print('Rescale the SMF to Planck15 cosmology')
for i in range(numzbin):
    # Correction of the comoving Volume :
    VmaxD17 = D17_Cosmo.comoving_volume(redshifts[i+1]) - D17_Cosmo.comoving_volume(redshifts[i])
    VmaxP15 = Planck15.comoving_volume(redshifts[i+1]) - Planck15.comoving_volume(redshifts[i])
    """In the case where we use the Vmax points and not the VmaxFit, the errors bars are relative and
    are not the absolute uncertainty as in the Vmax Fit, so we don't rescale the error bars"""
    smf[i][:, 1] = smf[i][:, 1] + np.log10(VmaxD17/VmaxP15)
    # Correction of the measured stellar mass
    # Equivalent to multiply by (Planck15.H0/D17_Cosmo.H0)**-2
    smf[i][:, 0] = smf[i][:, 0] - 2 * np.log10(Planck15.H0/D17_Cosmo.H0)
    # Correct for the dependance on the luminosity distance
    DL_D17 = D17_Cosmo.luminosity_distance(redshiftsbin[i])
    DL_Planck = Planck15.luminosity_distance(redshiftsbin[i])
    smf[i][:, 0] = smf[i][:, 0] + 2*np.log10(DL_Planck/DL_D17)


"""Load HMF"""

hmf=[]
mdef='200m'
hmf_name = 'despali16'
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

log_Mh = np.linspace(10, 17, num=100)

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

"""Load B13 plot from plot_digitizer"""

tmp = np.loadtxt('../Data/B13_plotdigit.txt')
log_Ms_B13 = np.log10(tmp[:, 0])
log_phi_B13 = np.log10(tmp[:, 1])

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


def gauss(y, ksi):
    return 1. / (ksi * np.sqrt(2 * np.pi)) * np.exp(- 1/2 * (y / ksi)**2)


def log_phi13_true(log_Mh, idx_z, z):
    a = af(z)
    print(ksi(a))
    log_phi_dir = log_phi13(log_Mh, idx_z, z)
    phitrue = log_phi_dir * 0.
    logMs = log_Ms13(log_Mh, z)
    # gauss_array = gauss(np.linspace(-10, 10), ksi(a))
    for i in range(phitrue.size):
        for j in range(phitrue.size -1):
            phitrue[i] = phitrue[i] + 10**log_phi_dir[j] * gauss(logMs[j] - logMs[i], ksi(a)) * (logMs[j+1] - logMs[j])
            # print(gauss(logMs[i] - logMs[j], ksi(a)))
    # print(logphitrue)
    # plt.plot(logMs, logphitrue)
    # plt.plot(logMs, log_phi_dir)
    # plt.show()
    return np.log10(phitrue)

# a = 11
# y = np.linspace(6, 14)
# plt.plot(y, gauss(a-y, 2))
# plt.show()

"""Compute log_Ms and phi corresponding to a given log_Mh and HMF"""

# log_Mh = np.linspace(7, 20, num=1000)

# for idx_z in range(redshiftsbin.size):
for idx_z in range(1):
    plt.figure()
    z = redshiftsbin[idx_z]
    log_phi_true = log_phi13_true(log_Mh, idx_z, z)
    log_phi_dir = log_phi13(log_Mh, idx_z, z)
    # print(log_phi)
    # print(smf[idx_z])

    plt.plot(log_Ms_B13, log_phi_B13, color='black', linestyle='--', label='B13 z=0.2 (plot digitizer)')

    plt.plot(log_Ms13(log_Mh, z), log_phi_dir, label='B13 best fit params, no convolution')
    plt.plot(log_Ms13(log_Mh, z), log_phi_true, label='B13 best fit params, with convolution', color='green')

    plt.plot(smf[idx_z][:, 0], smf[idx_z][:,1], label='cosmos Schechter Fit '+ str(redshifts[idx_z])+'<z<'+ str(redshifts[idx_z+1]), color="C{}".format(1),)
    plt.fill_between(smf[idx_z][:, 0], smf[idx_z][:, 1] - smf[idx_z][:, 3], smf[idx_z][:, 1] + smf[idx_z][:, 2], alpha=0.2, color="C{}".format(1),)
    # plt.plot(log_Mh, log_Ms13(log_Mh, z) - log_Mh)

    plt.legend()
    plt.xlabel('$M_*$', size=12)
    plt.ylabel('$\phi$', size=12)
    plt.ylim(-7, -1)
    plt.show()


# log_phi13_true(log_Mh, 0, redshiftsbin[0])