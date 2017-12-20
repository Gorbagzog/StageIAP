#!/usr/bin/env python3
# -*-coding:Utf-8 -*

"""Use MCMC to find the stellar mass halo mass relation.

Based on the Behroozi et al 2010 paper.
Use a parametrization of the SHMR, plus a given HMF to find the expected SMF and compare it
to the observed SMF with its uncertainties using a likelihod maximisation.

Started on december 18th by Louis Legrand at IAP and IAS.
"""

import numpy as np
# import matplotlib.pyplot as plt
# import emcee
from astropy.cosmology import LambdaCDM
import scipy.optimize as op
from scipy import signal

#################
### Load data ###
#################

"""Load HMF"""

# redshifts of the BolshoiPlanck files
redshift_haloes = np.arange(0, 10, step=0.1)
numredshift_haloes = np.size(redshift_haloes)

"""Definition of hmf_bolshoi columns :

hmf_bolshoi[redshift][:,0] = Log10(mass) [Msun]
hmf_bolshoi[redshift][:,1] = Log10(cen_mf), ie central haloes mass function
(density) [1/Mpc^3]
hmf_bolshoi[redshift][:,2] = Log10(all_macc_mf), ie all haloes mass function
(density) [1/Mpc^3]
"""
hmf_bolshoi = []
for i in range(numredshift_haloes):
    hmf_bolshoi.append(
        np.loadtxt('../Data/HMFBolshoiPlanck/mf_planck/mf_planck_z' +
                   '{:4.3f}'.format(redshift_haloes[i]) + '_mvir.dat'))

"""Load the SMF from Iary Davidzon+17"""

# redshifts of the Iari SMF
redshifts = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
numzbin = np.size(redshifts) - 1

smf_cosmos = []
for i in range(10):
    smf_cosmos.append(np.loadtxt(
        # Select the SMFs to use : tot, pas or act; D17 or SchechterFixedMs
        '/home/llegrand/StageIAP/Data/Davidzon/Davidzon+17_SMF_V3.0/mf_mass2b_fl5b_tot_VmaxFit2D'
        + str(i) + '.dat')
        # '../Data/Davidzon/schechter_fixedMs/mf_mass2b_fl5b_tot_VmaxFit2E'
        # + str(i) + '.dat')
    )

"""Adapt SMF to match the Bolshoi-Planck Cosmology"""
# Bolshoi-Planck cosmo : (flat LCMD)
# Om = 0.3089, Ol = 0.6911, Ob = 0.0486, h = 0.6774, s8 = 0.8159, ns = 0.9667
BP_Cosmo = LambdaCDM(H0=67.74, Om0=0.3089, Ode0=0.6911)

# Davidzon+17 SMF cosmo : (flat LCDM)
# Om = 0.3, Ol = 0.7, h=0.7
D17_Cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

for i in range(10):
    # Correction of the comoving Volume :
    VmaxD17 = D17_Cosmo.comoving_volume(redshifts[i+1]) - D17_Cosmo.comoving_volume(redshifts[i])
    VmaxBP = BP_Cosmo.comoving_volume(redshifts[i+1]) - BP_Cosmo.comoving_volume(redshifts[i])
    # VmaxD17 = get_Vmax_mod.main(redshifts[i], redshifts[i+1], cosmo=[70, 0.3, 0.7])
    # VmaxBP = get_Vmax_mod.main(redshifts[i], redshifts[i+1], cosmo=[67.74, 0.3089, 0.6911])
    # Add the log, equivalent to multiply by VmaxD17/VmaxBP
    smf_cosmos[i][:, 1] = smf_cosmos[i][:, 1] + np.log10(VmaxD17/VmaxBP)
    smf_cosmos[i][:, 2] = smf_cosmos[i][:, 2] + np.log10(VmaxD17/VmaxBP)
    smf_cosmos[i][:, 3] = smf_cosmos[i][:, 3] + np.log10(VmaxD17/VmaxBP)

    # Correction of the measured stellar mass
    # Equivalent to multiply by (BP_Cosmo.H0/D17_Cosmo.H0)**-2
    smf_cosmos[i][:, 0] = smf_cosmos[i][:, 0] - 2 * np.log10(BP_Cosmo.H0/D17_Cosmo.H0)


#######################################
### Define functions and parameters ###
#######################################


def logMh(logMs, M1, Ms0, beta, delta, gamma):
    # SM-HM relation
    return np.log10(M1) + beta*np.log10(logMs/Ms0) + ((logMs/Ms0)**delta)/(1 + (logMs/Ms0)**delta) - 0.5


def phi_direct(logMs1, logMs2, idx_z, M1, Ms0, beta, delta, gamma):
    # SMF obtained from the SM-HM relation and the HMF
    log_Mh1 = logMh(logMs1, M1, Ms0, beta, delta, gamma)
    log_Mh2 = logMh(logMs2, M1, Ms0, beta, delta, gamma)
    index_Mh = np.argmin(np.abs(hmf_bolshoi[idx_z][:, 0] - log_Mh1))
    phidirect = 10**hmf_bolshoi[idx_z][index_Mh, 2] * (log_Mh1 - log_Mh2)/(logMs1 - logMs2)
    return phidirect


Mmin = 7
Mmax = 16
numpoints = 1000
y = np.linspace(Mmin, Mmax, num=numpoints)

def lognorm(y, logMs, ksi):
    return 1/np.sqrt(2 * np.pi * ksi**2) * np.exp((y-logMs)/(2*ksi**2))


def phi_true(idx_z, logMs, M1, Ms0, beta, delta, gamma, ksi):
    # SMF with a log-normal scatter in stellar mass for a given halo mass
    # This is the same as convolving phi_true with a log-normal density probability function
    phitrue = 0
    for i in range(numpoints-1):
        phitrue += phi_direct(
            y[i], y[i+1], idx_z, M1, Ms0, beta, delta, gamma) * lognorm(y[i], logMs, ksi)
    # phitrue = np.sum(
    #     print(phi_direct(
    #          y[:-2][:], y[1:][:], idx_z, M1, Ms0, beta, delta, gamma) * lognorm(y[:-2], logMs, ksi)
    # )
    return phitrue


# def phi_true(idx_z, logMs, M1, Ms0, beta, delta, gamma, ksi):
#     y = np.linspace(Mmin, Mmax, num=numpoints)
#     phidirect = phi_direct(y[1:], y[:-1], idx_z, M1, Ms0, beta, delta, gamma)
#     lognorm = signal.gaussian(50, std=ksi)
#     return np.covolve()


# def phi_expect(z1, z2, logMs, M1, Ms0, beta, delta, gamma, ksi):
#     # Take into account that the observed SMF is for a range of redshift
#     numpoints = 10
#     redshifts = np.linspace(z1, z2, num=numpoints)
#     top = 0
#     bot = 0
#     for i in range(numpoints - 1):
#         dVc = BP_Cosmo.comoving_volume(redshifts[i+1]) - BP_Cosmo.comoving_volume(redshifts[i])
#         top += phi_true(redshifts[i], logMs, M1, Ms0, beta, delta, gamma, ksi) * dVc
#         bot += dVc
#     return top/bot


def chi2(idx_z, M1, Ms0, beta, delta, gamma, ksi):
    # return the chi**2 between the observed and the expected SMF
    # z1 = redshifts[idx_z]
    # z2 = redshifts[idx_z + 1]
    logMs = smf_cosmos[idx_z][smf_cosmos[idx_z][:, 1] > -1000, 0]  # select points where the smf is defined
    numpoints = len(logMs)

    chi2 = 0
    for i in range(numpoints):
        chi2 += (np.log10(
            phi_true(idx_z, logMs[i], M1, Ms0, beta, delta, gamma, ksi) /
            10**smf_cosmos[idx_z][i, 1]) / ((smf_cosmos[idx_z][i, 2] + smf_cosmos[idx_z][i, 3])/2))**2
    return chi2

def negloglike(theta, idx_z):
    # return the likelihood
    M1, Ms0, beta, delta, gamma, ksi = theta[:]
    return chi2(idx_z, M1, Ms0, beta, delta, gamma, ksi)/2


##########################################
### Find maximum likelihood estimation ###
##########################################

idx_z = 0
theta0 = np.array([12, 11, 0.5, 0.5, 2.5, 0.15])
# results = op.minimize(negloglike, theta0, args=(idx_z))
print(negloglike(theta0, idx_z))