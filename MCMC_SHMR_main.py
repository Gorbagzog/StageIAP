#!/usr/bin/env python3
# -*-coding:Utf-8 -*

"""Use MCMC to find the stellar mass halo mass relation.

Based on the Behroozi et al 2010 paper.
Use a parametrization of the SHMR, plus a given HMF to find the expected SMF and compare it
to the observed SMF with its uncertainties using a likelihod maximisation.

Started on december 18th by Louis Legrand at IAP and IAS.
"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
from astropy.cosmology import LambdaCDM
import scipy.optimize as op
from scipy import signal


def load_hmf():
    """Load HMF from Bolshoi Planck simulation"""
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
    global hmf_bolshoi
    hmf_bolshoi = []
    for i in range(numredshift_haloes):
        hmf_bolshoi.append(
            np.loadtxt('../Data/HMFBolshoiPlanck/mf_planck/mf_planck_z' +
                       '{:4.3f}'.format(redshift_haloes[i]) + '_mvir.dat'))

def load_smf():
    """Load the SMF from Iary Davidzon+17"""
    # redshifts of the Iari SMF
    redshifts = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
    numzbin = np.size(redshifts) - 1
    global smf_cosmos
    smf_cosmos = []
    for i in range(10):
        smf_cosmos.append(np.loadtxt(
            # Select the SMFs to use : tot, pas or act; D17 or SchechterFixedMs
            '../Data/Davidzon/Davidzon+17_SMF_V3.0/mf_mass2b_fl5b_tot_VmaxFit2D'
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


"""Function definitions for computation of the theroretical SFM phi_true"""


def logMh(logMs, M1, Ms0, beta, delta, gamma):
    # SM-HM relation
    return M1 + beta*(logMs - Ms0) + (10**(delta*(logMs-Ms0)))/(1 + (10**(-gamma*(logMs - Ms0)))) - 0.5


def log_phi_direct(logMs, idx_z, M1, Ms0, beta, delta, gamma):
    # SMF obtained from the SM-HM relation and the HMF
    epsilon = 0.01*logMs
    log_Mh1 = logMh(logMs, M1, Ms0, beta, delta, gamma)
    log_Mh2 = logMh(logMs + epsilon, M1, Ms0, beta, delta, gamma)
    # index_Mh = np.argmin(np.abs(hmf_bolshoi[idx_z][:, 0] - log_Mh1))
    index_Mh = np.argmin(np.abs(
        np.tile(hmf_bolshoi[idx_z][:, 0], (len(log_Mh1), 1)) - 
        np.transpose(np.tile(log_Mh1, (len(hmf_bolshoi[idx_z][:, 0]), 1)))
    ), axis=1)
    log_phidirect = hmf_bolshoi[idx_z][index_Mh, 2] + np.log10((log_Mh2 - log_Mh1)/epsilon)
    return log_phidirect


def log_phi_true(idx_z, logMs, M1, Ms0, beta, delta, gamma, ksi):
    # Use the approximation of the convolution defined in Behroozi et al 2010 equation (3)
    epsilon = 0.01 * logMs
    logphi1 = log_phi_direct(logMs, idx_z, M1, Ms0, beta, delta, gamma)
    logphi2 = log_phi_direct(logMs + epsilon, idx_z, M1, Ms0, beta, delta, gamma)
    logphitrue = logphi1 + ksi**2 /2 * np.log(10) * ((logphi2 - logphi1)/epsilon)**2
    return logphitrue


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
    select = np.where(smf_cosmos[idx_z][:, 1] > -1000)[0] # select points where the smf is defined
    logMs = smf_cosmos[idx_z][select, 0]  
    chi2 = np.sum(
        ((log_phi_true(idx_z, logMs, M1, Ms0, beta, delta, gamma, ksi) -
        smf_cosmos[idx_z][select, 1]) / ((smf_cosmos[idx_z][select, 2] + smf_cosmos[idx_z][select, 3])/2))**2
        )
    return chi2


def negloglike(theta, idx_z):
    # return the likelihood
    M1, Ms0, beta, delta, gamma, ksi = theta[:]
    if beta<0 or delta<0 or gamma<0:
        return -10**8
    if M1<6 or M1>15 or Ms0<6 or Ms0>15:
        return -10**8
    if ksi<0 or ksi > 4:
        return -10**8
    else:
        return chi2(idx_z, M1, Ms0, beta, delta, gamma, ksi)/2


"""Find maximum likelihood estimation"""

def main():
    load_hmf()
    load_smf()
    idx_z = 0
    theta0 = np.array([12, 11, 0.5, 0.5, 2.5, 0.15])
    bounds = ((10, 14), (8, 13), (0, 2), (0, 3), (0, 5), (0, 1))
    results = op.minimize(negloglike, theta0, bounds=bounds, args=(idx_z))
    print(negloglike(theta0, idx_z))

if __name__ == "__main__":
    main()

"""Plots and tests"""


logMs = np.linspace(6, 12, num=100)
plt.plot(logMs, logMh(logMs, 12, 10, 0.5, 0.5, 2.5))

plt.plot(logMh(logMs, 12, 10, 0.5, 0.5, 2.5), logMs - logMh(logMs, 12, 10, 0.5, 0.5, 2.5))

thetavar =  np.array([np.linspace(10,14, num=100), np.full(100, 11), np.full(100,0.5), np.full(100,0.5), np.full(100,2.5), np.full(100,0.15)])

neglog = np.zeros(100)
for i in range(100):
    neglog[i] = negloglike(thetavar[:,i], idx_z)

plt.plot(neglog)

"""Test emcee sampling"""

nwalker=250
ndim=6
std = np.array([1, 1, 0.1, 0.1, 0.1, 0.1])
p0 = emcee.utils.sample_ball(theta0, std, size=nwalker)
sampler= emcee.EnsembleSampler(nwalker, ndim, negloglike, args=[idx_z])

pos, prob, state = sampler.run_mcmc(p0, 100) ## burn phase

sampler.run_mcmc(pos, 1000) ## samble phase