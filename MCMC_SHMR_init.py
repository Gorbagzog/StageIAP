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
import sys
import emcee
from astropy.cosmology import LambdaCDM
import scipy.optimize as op
from scipy import signal
import corner
from getdist import plots, MCSamples
import time
import os
import datetime

def load_smf():
    """Load the SMF from Iary Davidzon+17"""
    # redshifts of the Iari SMF
    global redshifts
    global redshiftsbin
    redshifts = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
    redshiftsbin = (redshifts[1:]+redshifts[:-1])/2
    global numzbin
    numzbin = np.size(redshifts) - 1
    global smf_cosmos
    smf_cosmos = []
    for i in range(10):
        smf_cosmos.append(np.loadtxt(
            # Select the SMFs to use : tot, pas or act; D17 or SchechterFixedMs
            # '../Data/Davidzon/Davidzon+17_SMF_v3.0/mf_mass2b_fl5b_tot_VmaxFit2D'
            # + str(i) + '.dat')
            '../Data/Davidzon/Davidzon+17_SMF_v3.0/mf_mass2b_fl5b_tot_Vmax'
            + str(i) + '.dat') # Use the 1/Vmax points directly and not the schechter fit on them
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
        # smf_cosmos[i][:, 1] = smf_cosmos[i][:, 1] + np.log10(VmaxD17/VmaxBP)
        # smf_cosmos[i][:, 2] = smf_cosmos[i][:, 2] + np.log10(VmaxD17/VmaxBP)
        # smf_cosmos[i][:, 3] = smf_cosmos[i][:, 3] + np.log10(VmaxD17/VmaxBP)

        """In the case where we use the Vmax points and not the VmaxFit, the errors bars are relative and
        are not the absolute uncertainty as in the Vmax Fit, so we don't rescale the error bars"""
        smf_cosmos[i][:, 1] = smf_cosmos[i][:, 1] + np.log10(VmaxD17/VmaxBP)

        # Correction of the measured stellar mass
        # Equivalent to multiply by (BP_Cosmo.H0/D17_Cosmo.H0)**-2
        smf_cosmos[i][:, 0] = smf_cosmos[i][:, 0] - 2 * np.log10(BP_Cosmo.H0/D17_Cosmo.H0)


def load_hmf():
    # """Load HMF from Bolshoi Planck simulation"""
    # # redshifts of the BolshoiPlanck files
    redshift_haloes = np.arange(0, 10, step=0.1)
    numredshift_haloes = np.size(redshift_haloes)
    """Definition of hmf columns :
    hmf_bolshoi[redshift][:,0] = Log10(mass) [Msun]
    hmf_bolshoi[redshift][:,1] = Log10(cen_mf), ie central haloes mass function
    (density) [1/Mpc^3]
    hmf_bolshoi[redshift][:,2] = Log10(all_macc_mf), ie all haloes mass function
    (density) [1/Mpc^3]
    """
    global hmf
    hmf_bolshoi_tot = []
    for i in range(numredshift_haloes):
        hmf_bolshoi_tot.append(
            np.loadtxt('../Data/HMFBolshoiPlanck/mf_planck/mf_planck_z' +
                       '{:4.3f}'.format(redshift_haloes[i]) + '_mvir.dat'))

    """Select the redhifts slices that matches the slices of Iary"""
    global redshift_id_selec
    redshift_id_selec = np.empty(numzbin)
    for i in range(numzbin):
        redshift_id_selec[i] = np.argmin(
            np.abs(redshift_haloes - (redshifts[i] + redshifts[i + 1]) / 2))

    redshift_id_selec = redshift_id_selec.astype(int)
    print('Redshifts of Iari SMFs : ' + str((redshifts[:-1] + redshifts[1:]) / 2))
    print('Closest redshifts for Bolshoi HMFs : '
        + str(redshift_haloes[redshift_id_selec]))
    hmf = []
    for i in redshift_id_selec:
        hmf.append(hmf_bolshoi_tot[i])

    """Load Tinker+08 HMF computed with HFMCalc of Murray+13
    parameters : Delta = 200 times the mean density of the universe (same at all z)
    """
    redshift_haloes = np.array([0.35, 0.65, 0.95, 1.3, 1.75, 2.25, 2.75, 3.25, 4, 5])
    numredshift_haloes = len(redshift_haloes)
    global hmf
    hmf = []
    for i in range(numredshift_haloes):
        hmf.append(
            np.loadtxt('../Data/Tinker08HMF/HMFCalc_Dm200/mVector_PLANCK-SMT_z{:1.2f}.txt'.format(
                redshift_haloes[i]), usecols=(0, 7)))
        hmf[i][:, 0] = np.log10(hmf[i][:, 0] / 0.6774)
        hmf[i][:, 1] = np.log10(hmf[i][:, 1] * (0.6774)**3)


"""Function definitions for computation of the theroretical SFM phi_true"""


def logMh(logMs, M1, Ms0, beta, delta, gamma):
    # SM-HM relation
    return M1 + beta*(logMs - Ms0) + (10 ** (delta * (logMs - Ms0))) / (1 + (10 ** (-gamma * (logMs - Ms0)))) - 0.5
    # Ms = 10**logMs
    # logMh = M1 + beta * np.log10(Ms / 10**Ms0) + (Ms / 10**Ms0)**delta / (1 + (Ms / 10**Ms0)**(-gamma)) - 0.5
    # return logMh


def log_phi_direct(logMs, idx_z, M1, Ms0, beta, delta, gamma):
    # SMF obtained from the SM-HM relation and the HMF
    epsilon = 0.0001
    log_Mh1 = logMh(logMs, M1, Ms0, beta, delta, gamma)
    log_Mh2 = logMh(logMs + epsilon, M1, Ms0, beta, delta, gamma)
    # print(logMs)
    # print(log_Mh1, log_Mh2)
    # index_Mh = np.argmin(np.abs(hmf[idx_z][:, 0] - log_Mh1))
    # Select the index of the HMF corresponing to the halo masses
    index_Mh = np.argmin(
        np.abs(
            np.tile(hmf[idx_z][:, 0], (len(log_Mh1), 1)) -
            np.transpose(np.tile(log_Mh1, (len(hmf[idx_z][:, 0]), 1)))
        ), axis=1)
    # print(np.tile(hmf[idx_z][:, 0], (len(log_Mh1), 1)))
    # print(np.transpose(np.tile(log_Mh1, (len(hmf[idx_z][:, 0]), 1))))
    """ If only central HMF from Bolshoi or Tinker HMF :"""
    # log_phidirect = hmf[idx_z][index_Mh, 1] + np.log10((log_Mh2 - log_Mh1)/epsilon)
    """ If all haloes from Bolshoi"""
    log_phidirect = hmf[idx_z][index_Mh, 2] + np.log10((log_Mh2 - log_Mh1)/epsilon)
    # print(np.log10((log_Mh2 - log_Mh1)/epsilon))
    # Keep only points where the halo mass is defined in the HMF
    log_phidirect[log_Mh1 > hmf[idx_z][-1, 0]] = -1000
    log_phidirect[log_Mh1 < hmf[idx_z][0, 0]] = -1000
    # print(log_phidirect)
    # print(hmf[idx_z][index_Mh, 2])
    # print(log_phidirect)
    return log_phidirect


def log_phi_true(logMs, idx_z, M1, Ms0, beta, delta, gamma, ksi):
    # Use the approximation of the convolution defined in Behroozi et al 2010 equation (3)
    epsilon = 0.01 * logMs
    logphi1 = log_phi_direct(logMs, idx_z, M1, Ms0, beta, delta, gamma)
    logphi2 = log_phi_direct(logMs + epsilon, idx_z, M1, Ms0, beta, delta, gamma)
    logphitrue = logphi1 + ksi**2 / 2 * np.log(10) * ((logphi2 - logphi1)/epsilon)**2
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
    # select = np.where(np.logical_and(
    #     smf_cosmos[idx_z][:, 1] > -6,  # select points where the smf is defined
    #     smf_cosmos[idx_z][:, 3] < 900))[0]  # select points where the error bar is defined
    select = np.where(smf_cosmos[idx_z][:, 1] > -7)  # select points where the smf is defined
    # We choose to limit the fit only for abundances higher than 10**-7
    logMs = smf_cosmos[idx_z][select, 0]
    pred = 10**log_phi_true(logMs, idx_z, M1, Ms0, beta, delta, gamma, ksi)
    chi2 = np.sum(
            # When using the Vmax directly (give the error bars directly, in a linear scale)
            # Need to use a linear scale to compute the chi2 with the right uncertainty
            ((pred - 10**smf_cosmos[idx_z][:, 1]) / (
                10**(smf_cosmos[idx_z][:, 2] + smf_cosmos[idx_z][:, 1]) - 10**smf_cosmos[idx_z][:, 1]))**2
        )
    return chi2


def chi2_noksi(idx_z, M1, Ms0, beta, delta, gamma):
    # return the chi**2 between the observed and the expected SMF
    # select = np.where(np.logical_and(
    #     smf_cosmos[idx_z][:, 1] > -7,  # select points where the smf is defined
    #     smf_cosmos[idx_z][:, 3] < 900))[0]  # select points where the error bar is defined
    # We choose to limit the fit only for abundances higher than 10**-7 --> no more necessary as the error bars are defined for en linear scale
    # logMs = smf_cosmos[idx_z][select, 0]
    logMs = smf_cosmos[idx_z][:, 0]
    pred = 10**log_phi_direct(logMs, idx_z, M1, Ms0, beta, delta, gamma)
    chi2 = np.sum(
            # When using the VmaxFit2D (give the bands and not the sigma)
            # smf_cosmos[idx_z][select, 1]) / ((smf_cosmos[idx_z][select, 3] - smf_cosmos[idx_z][select, 2])/2))**2
            # When using the Vmax directly (give the error bars directly, in a linear scale)
            # Need to use a linear scale to compute the chi2 with the right uncertainty
        ((pred - 10**smf_cosmos[idx_z][:, 1]) / (
            10**(smf_cosmos[idx_z][:, 2] + smf_cosmos[idx_z][:, 1]) - 10**smf_cosmos[idx_z][:, 1]))**2
            # smf_cosmos[idx_z][select, 1]))**2
        )
    # print( (pred - smf_cosmos[idx_z][select, 1]) / ((smf_cosmos[idx_z][select, 3] + smf_cosmos[idx_z][select, 2])/2))
    return chi2


# def loglike(theta, idx_z):
#     # return the likelihood
#     # print(theta)
#     # bouds for the idx_z = 0
#     M1, Ms0, beta, delta, gamma, ksi = theta[:]
#     if beta < 0.1 or delta < 0.1 or gamma < 0:
#         return -np.inf
#     if beta > 1 or delta > 1 or gamma > 3:
#         return -np.inf
#     if M1 < 11 or M1 > 13 or Ms0 < 10 or Ms0 > 12:
#         return -np.inf
#     if ksi < 0 or ksi > 1:
#         return -np.inf
#     else:
#         return -chi2(idx_z, M1, Ms0, beta, delta, gamma, ksi)/2


def loglike(theta, idx_z):
    # return the likelihood
    # bounds for the idx_z = 1
    M1, Ms0, beta, delta, gamma, ksi = theta[:]
    if idx_z < 5 :
        if beta < 0 or delta < 0 or gamma < 0:
            return -np.inf
        if beta > 1 or delta >  1 or gamma > 5:
            return -np.inf
        if M1 < 11 or M1 > 13.2 or Ms0 < 10 or Ms0 > 12:
            return -np.inf
        if ksi < 0 or ksi > 1:
            return -np.inf
        else:
            return -chi2(idx_z, M1, Ms0, beta, delta, gamma, ksi)/2
    elif idx_z == 5 :
        if beta < 0 or delta < 0 or gamma < 0:
            return -np.inf
        if beta > 1 or delta >  1 or gamma > 5:
            return -np.inf
        if M1 < 11.5 or M1 > 14 or Ms0 < 11 or Ms0 > 12:
            return -np.inf
        if ksi < 0 or ksi > 1:
            return -np.inf
        else:
            return -chi2(idx_z, M1, Ms0, beta, delta, gamma, ksi)/2
    elif idx_z == 6 :
        if beta < 0 or delta < 0 or gamma < 0:
            return -np.inf
        if beta > 1 or delta >  1 or gamma > 5:
            return -np.inf
        if M1 < 12 or M1 > 15 or Ms0 < 10 or Ms0 > 13:
            return -np.inf
        if ksi < 0 or ksi > 1:
            return -np.inf
        else:
            return -chi2(idx_z, M1, Ms0, beta, delta, gamma, ksi)/2
    elif idx_z > 6 :
        if beta < 0 or delta < 0 or gamma < 0:
            return -np.inf
        if beta > 1 or delta > 1 or gamma > 5:
            return -np.inf
        if M1 < 12 or M1 > 15 or Ms0 < 10 or Ms0 > 15:
            return -np.inf
        if ksi < 0 or ksi > 1:
            return -np.inf
        else:
            return -chi2(idx_z, M1, Ms0, beta, delta, gamma, ksi)/2


def loglike_noksi(theta, idx_z):
    # return the likelihood for a fixed ksi
    # print(theta)
    M1, Ms0, beta, delta, gamma = theta[:]
    if idx_z < 5 :
        if beta < 0 or delta < 0 or gamma < 0:
            return -np.inf
        if beta > 1 or delta >  1 or gamma > 5:
            return -np.inf
        if M1 < 11 or M1 > 13.2 or Ms0 < 10 or Ms0 > 12:
            return -np.inf
        else:
            return -chi2_noksi(idx_z, M1, Ms0, beta, delta, gamma)/2
    elif idx_z == 5 :
        if beta < 0 or delta < 0 or gamma < 0:
            return -np.inf
        if beta > 1 or delta >  1 or gamma > 5:
            return -np.inf
        if M1 < 11.5 or M1 > 14 or Ms0 < 11 or Ms0 > 12:
            return -np.inf
        else:
            return -chi2_noksi(idx_z, M1, Ms0, beta, delta, gamma)/2
    elif idx_z == 6 :
        if beta < 0 or delta < 0 or gamma < 0:
            return -np.inf
        if beta > 1 or delta >  1 or gamma > 5:
            return -np.inf
        if M1 < 12 or M1 > 15 or Ms0 < 10 or Ms0 > 13:
            return -np.inf
        else:
            return -chi2_noksi(idx_z, M1, Ms0, beta, delta, gamma)/2
    elif idx_z > 6 :
        if beta < 0 or delta < 0 or gamma < 0:
            return -np.inf
        if beta > 1 or delta >  1 or gamma > 5:
            return -np.inf
        if M1 < 12 or M1 > 15 or Ms0 < 10 or Ms0 > 15:
            return -np.inf
        else:
            return -chi2_noksi(idx_z, M1, Ms0, beta, delta, gamma)/2


def negloglike(theta, idx_z):
    return -loglike(theta, idx_z)


def negloglike_noksi(theta, idx_z):
    return -loglike_noksi(theta, idx_z)


"""Find maximum likelihood estimation"""


def maxlikelihood(idx_z, theta0, bounds):
    load_smf()
    load_hmf()
    # idx_z = 0
    # theta0 = np.array([11, 10, 0.1, 0.1, 1])
    # bounds = ((11, 13), (10, 12), (0, 1), (0, 1), (0, 3), (0, 1))
    # results = op.minimize(negloglike_noksi, theta0, bounds=bounds, args=(idx_z), method='TNC')
    # results = op.basinhopping(negloglike, theta0, niter=1, T=1000, minimizer_kwargs={'args': idx_z})
    # results = op.minimize(negloglike_noksi, theta0, args=(idx_z), method='Nelder-Mead', options={'fatol':10**-6})
    results = op.minimize(negloglike, theta0, args=(idx_z), method='Nelder-Mead', options={'fatol':10**-6})
    print(results)


""" Run MCMC """

def runMCMC_allZ(starting_point, std, iterations, burn, nthreads=1):
    load_smf()
    load_hmf()
    nwalker = 20
    
    # Create save direcory
    now = datetime.datetime.now()
    directory = "../MCMC_"+str(now.year)+'-'+str(now.month)+'-'+str(now.day)+'T'+str(now.hour)+'-'+str(now.minute)
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(directory+'/Chain')
        os.makedirs(directory+'/Plots')
        os.makedirs(directory+'/Plots/MhaloPeak')
        os.makedirs(directory+'/Results')

    # run all MCMC for all zbins
    for idx_z in range(numzbin):
        runMCMC(directory, idx_z, starting_point, std, iterations, burn, nthreads)


def runMCMC(directory, idx_z, starting_point, std, iterations, burn, nthreads=1):
    # load_smf()
    # load_hmf()
    # nwalker = 20
    # starting_point =  np.array([12.5, 10.8, 0.5, 0.5, 0.5, 0.15])
    # std =np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.01])
    # starting_point =  np.array([12.5, 11, 0.5, 0.7, 0.5, 0.15])
    start_time = time.time()
    p0 = emcee.utils.sample_ball(starting_point, std, size=nwalker)
    ndim = len(starting_point)
    sampler = emcee.EnsembleSampler(nwalker, ndim, loglike, args=[idx_z], threads=nthreads)
    print("idx_z = " +str (idx_z))
    print("ndim = " + str(ndim))
    print("start = " + str(starting_point))
    print("std = " + str(std))
    print("iterations = " + str(iterations))
    start_time = time.time()
    sampler.run_mcmc(p0, iterations)
    elapsed_time = time.time() - start_time
    print('Time elapsed : ' + str(elapsed_time))

    # Save chains and loglike of chains
    chainfile = directory + "/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
    savenameln = directory + "/Chain/LnProb_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
    np.save(chainfile, sampler.chain)
    np.save(savenameln, sampler.lnprobability)

    # Plot all relevant figures
    plt.close('all')
    plotchain(directory, chainfile, idx_z, iterations, burn)
    plt.close('all')
    plotdist_noksi(directory, chainfile, idx_z, iterations, burn)
    plt.close('all')
    plotSMF(directory, idx_z, iterations, burn)
    plt.close('all')
    plotSMHM(directory, idx_z, iterations, burn)
    plt.close('all')
    plot_Mhpeak(directory, chainfile, idx_z, iterations, burn)
    plt.close('all')
    save_results(directory, chainfile, idx_z, iterations, burn)


# def runMCMC_noksi(idx_z, starting_point, std, iterations, burn, nthreads=1):
#     load_smf()
#     load_hmf()
#     nwalker = 20
#     # Put more nthreads for multiprocessing automatically.
#     # starting_point = np.array([12, 11, 0.5, 0.5, 2.5])
#     # std = np.array([1, 1, 0.1, 0.1, 0.1])

#     p0 = emcee.utils.sample_ball(starting_point, std, size=nwalker)
#     ndim = len(starting_point)
#     sampler = emcee.EnsembleSampler(nwalker, ndim, loglike_noksi, args=[idx_z], threads=nthreads)
#     print("idx_z = " +str (idx_z))
#     print("ndim = " + str(ndim))
#     print("start = " + str(starting_point))
#     print("std = " + str(std))
#     print("iterations = " + str(iterations))
#     start_time = time.time()
#     sampler.run_mcmc(p0, iterations)
#     ## Monitor the sampling progress
#     # nsteps = iterations/100
#     # width = 30
#     # for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
#     #     n = int((width+1) * float(i) / nsteps)
#     #     sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
#     # sys.stdout.write("\n")
#     elapsed_time = time.time() - start_time
#     print('Time elapsed : ' + str(elapsed_time))
#     savename = "../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
#     savenameln = "../MCMC/Chain/LnProb_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
#     np.save(savename, sampler.chain)
#     np.save(savenameln, sampler.lnprobability)
#     plt.close('all')
#     plotchain_noksi(savename, idx_z, iterations, burn)
#     plt.close('all')
#     # plotdist_noksi(savename, idx_z, iterations, burn)
#     # plt.close('all')
#     plotSMF_noksi(idx_z, iterations, burn)
#     plt.close('all')
#     plotSMHM_noksi(idx_z, iterations, burn)
#     plt.close('all')
#     plot_Mhpeak(savename, idx_z, iterations, burn)
#     plt.close('all')
#     save_results_noksi(savename, idx_z, iterations, burn)


def save_results_noksi(chainfile, idx_z, iterations, burn):
    chain = np.load(chainfile)
    # names = ['$M_{1}$', '$M_{s,0}$', '$\\beta$', '$\delta$', '$\gamma$', 'ksi']
    names = ['$M_{1}$', '$M_{s,0}$', '$\\beta$', '$\delta$', '$\gamma$']
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    samples = MCSamples(samples = samples, names = names)
    res = samples.getTable()
    #res.write("../MCMC/Results/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".txt")
    res.write("../MCMC/Results/Chain_Noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".txt")


def save_results(directory, chainfile, idx_z, iterations, burn):
    chain = np.load(chainfile)
    names = ['$M_{1}$', '$M_{s,0}$', '$\\beta$', '$\delta$', '$\gamma$', 'ksi']
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    samples = MCSamples(samples = samples, names = names)
    res = samples.getTable()
    res.write(directory+"/Results/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".txt")


def MhPeak(chainfile, idx_z, iterations, burn):
    chain = np.load(chainfile)
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    chainsize = np.shape(samples)[0]
    logMs = np.linspace(8, 12, num=300)
    Mhalopeak = np.zeros(chainsize)
    for i in range(chainsize):
        logmhalo = logMh(logMs, samples[i, 0], samples[i, 1], samples[i, 2], samples[i, 3], samples[i, 4])
        Mhalopeak_idx = np.argmax(logMs - logmhalo)
        Mhalopeak[i] = logmhalo[Mhalopeak_idx]
    return Mhalopeak


def allMhPeak(directory, iterations, burn):
    mhpeakall = np.zeros(numzbin)
    mhpeakallstd = np.zeros(numzbin)
    for idx_z in range(numzbin):
        chainfile = directory+"/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
        mhpeak = MhPeak(chainfile, idx_z, iterations, burn)
        mhpeakall[idx_z] = np.median(mhpeak)
        mhpeakallstd[idx_z] = np.std(mhpeak)
    return mhpeakall, mhpeakallstd


    # f = open("chain.dat", "w")
    # f.close()

    # for result in sampler.sample(p0, iterations=iterations):
    #     position = result[0]
    #     f = open("chain.dat", "a")
    #     for k in range(position.shape[0]):
    #         f.write("{0:4d} {1:s}\n".format(k, " ".join(position[k])))
    #     f.close()


# if __name__ == "__main__":
#     main()

"""Plots"""


def plotSMF_noksi(idx_z, iterations, burn):
    load_smf()
    load_hmf()
    chain = np.load("../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy")
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    # select = np.where(np.logical_and(
    #     smf_cosmos[idx_z][:, 1] > -7,  # select points where the smf is defined
    #     smf_cosmos[idx_z][:, 3] < 900))[0]  # select points where the error bar is defined
    select = np.where(smf_cosmos[idx_z][:, 1] > -40)  # select points where the smf is defined
    logMs = smf_cosmos[idx_z][select, 0]
    plt.errorbar(logMs, smf_cosmos[idx_z][select, 1],
        yerr=[smf_cosmos[idx_z][select, 3], smf_cosmos[idx_z][select, 2]], fmt='o')
    plt.ylim(-7.5, -1)
    for M1, Ms0, beta, delta, gamma in samples[np.random.randint(len(samples), size=100)]:
        logphi = log_phi_direct(logMs, idx_z, M1, Ms0, beta, delta, gamma)
        plt.plot(logMs[0], logphi[0], color="k", alpha=0.1)
    # plt.show()
    plt.savefig('../MCMC/Plots/SMF_noksi'+ str(idx_z) + "_niter=" + str(iterations) + '.pdf')


def plotSMF(directory, idx_z, iterations, burn):
    load_smf()
    load_hmf()
    chain = np.load(directory + "Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy")
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    select = np.where(smf_cosmos[idx_z][:, 1] > -1000)[0]
    logMs = smf_cosmos[idx_z][select, 0]
    plt.errorbar(logMs, smf_cosmos[idx_z][select, 1],
        yerr=[smf_cosmos[idx_z][select, 3], smf_cosmos[idx_z][select, 2]], fmt='o')
    plt.ylim(-7.5, -1)
    for M1, Ms0, beta, delta, gamma, ksi in samples[np.random.randint(len(samples), size=100)]:
        logphi = log_phi_true(logMs, idx_z, M1, Ms0, beta, delta, gamma, ksi)
        plt.plot(logMs, logphi, color="k", alpha=0.1)
    # plt.show()
    plt.savefig(directory+'/Plots/SMF_ksi'+ str(idx_z) + "_niter=" + str(iterations) + '.pdf')


def plotSMHM(directory, idx_z, iterations, burn):
    load_smf()
    load_hmf()
    chain = np.load(directory+"/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy")
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    logMs = np.linspace(9, 11.5, num=200)
    for M1, Ms0, beta, delta, gamma, ksi in samples[np.random.randint(len(samples), size=100)]:
        logmhalo = logMh(logMs, M1, Ms0, beta, delta, gamma)
        logphi = log_phi_true(logMs, idx_z, M1, Ms0, beta, delta, gamma, ksi)
        plt.plot(logmhalo, logMs-logmhalo, color="k", alpha=0.1)
    plt.savefig(directory+'/Plots/SMHM_ksi'+ str(idx_z) + "_niter=" + str(iterations) + '.pdf')


def plotSMHM_noksi(idx_z, iterations, burn):
    load_smf()
    load_hmf()
    chain = np.load("../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy")
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    logMs = np.linspace(9, 11.5, num=200)
    plt.figure()
    for M1, Ms0, beta, delta, gamma in samples[np.random.randint(len(samples), size=100)]:
        logmhalo = logMh(logMs, M1, Ms0, beta, delta, gamma)
        plt.plot(logmhalo, logMs-logmhalo, color="k", alpha=0.1)
    # plt.show()
    plt.savefig('../MCMC/Plots/SMHM_noksi'+ str(idx_z) + "_niter=" + str(iterations) + '.pdf')


def plotHMvsSM_noksi(idx_z, iterations, burn):
    load_smf()
    load_hmf()
    chain = np.load("../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy")
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    logMs = np.linspace(9, 11.5, num=200)
    for M1, Ms0, beta, delta, gamma in samples[np.random.randint(len(samples), size=100)]:
        logmhalo = logMh(logMs, M1, Ms0, beta, delta, gamma)
        plt.plot(logMs, logmhalo, color="k", alpha=0.1)
    # plt.show()
    plt.savefig('../MCMC/Plots/HMvsSM_noksi'+ str(idx_z) + "_niter=" + str(iterations) + '.pdf')


def plotchain(directory, chainfile, idx_z, iterations, burn):
    chain = np.load(chainfile)
    figname = directory + "/Plots/Ksi_z" + str(idx_z) + "_niter=" + str(iterations) + "_burn=" + str(burn)

    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    fig = corner.corner(
        samples, labels=['$M_{1}$', '$M_{*,0}$', '$\\beta$', '$\delta$', '$\gamma$', 'ksi'])
    fig.savefig(figname + ".pdf")
    plt.close('all')


def plotchain_noksi(chainfile, idx_z, iterations, burn):
    chain = np.load(chainfile)
    figname = "../MCMC/Plots/Noksi_z" + str(idx_z) + "_niter=" + str(iterations) + "_burn=" + str(burn)

    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    fig = corner.corner(
        samples, labels=['$M_{1}$', '$M_{*,0}$', '$\\beta$', '$\delta$', '$\gamma$'])
    fig.savefig(figname + ".pdf")
    plt.close('all')


    # for (p, loglike, state) in sampler.sample(p0, iterations=iterations):
    #     print(p)
    #     print(loglike)


def plotdist(chainfile, idx_z, iterations, burn):
    chain = np.load(chainfile)
    figname = "../MCMC/Plots/Ksi_z" + str(idx_z) + "_niter=" + str(iterations) + "_burn=" + str(burn)
    names = ['$M_{1}$', '$M_{s,0}$', '$\\beta$', '$\delta$', '$\gamma$', 'ksi']
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    samples = MCSamples(samples = samples, names = names)
    g = plots.getSubplotPlotter()
    g.triangle_plot(samples, filled=True)
    g.export(figname + '_gd.pdf' )
    plt.clf()


def plotdist_noksi(directory, chainfile, idx_z, iterations, burn):
    chain = np.load(chainfile)
    figname = directory + "/Plots/Noksi_z" + str(idx_z) + "_niter=" + str(iterations) + "_burn=" + str(burn)
    names = ['$M_{1}$', '$M_{s,0}$', '$\\beta$', '$\delta$', '$\gamma$']
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    samples = MCSamples(samples = samples, names = names)
    g = plots.getSubplotPlotter()
    g.triangle_plot(samples, filled=True)
    g.export(figname + '_gd.pdf' )
    plt.clf()


def plotLnprob(idx_z, iterations, nwalker=20):
    lnprob = np.load("../MCMC/Chain/LnProb_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy")
    for k in range(nwalker):
        plt.plot(lnprob[k, :])


def plot_Mhpeak(directory, chainfile, idx_z, iterations, burn):
    mhpeak = MhPeak(chainfile, idx_z, iterations, burn)
    # avg_mhpeak = np.mean(mhpeak)
    med_mhpeak = np.median(mhpeak)
    std_mhpeak = np.std(mhpeak)
    plt.figure()
    plt.hist(mhpeak, bins=100)
    plt.axvline(med_mhpeak, color='orange')
    plt.title(str(redshifts[idx_z]) +'<z<' + str(redshifts[idx_z+1]) + ', MhPeak = ' + str(med_mhpeak) + '+/-' + str(std_mhpeak))
    plt.savefig(directory+'/Plots/MhaloPeak/MhPeak_z' + str(idx_z) + '.pdf')


def plotSigmaHMvsSM(idx_z, iterations, burn):
    load_smf()
    load_hmf()
    chainfile = "../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
    chain = np.load(chainfile)
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    numpoints = 100
    logMs = np.linspace(9, 12, num=numpoints)
    logmhalo = np.zeros([samples.shape[0], numpoints])
    for idx_simu in range(samples.shape[0]):
        M1, Ms0, beta, delta, gamma = samples[idx_simu]
        logmhalo[idx_simu, :] = logMh(logMs, M1, Ms0, beta, delta, gamma)
    av_logMh = np.average(logmhalo, axis=0)
    conf_min_logMh = np.percentile(logmhalo, 16, axis=0)  # 16th percentile = median - 1sigma (68% confidence interval)
    conf_max_logMh = np.percentile(logmhalo, 84, axis=0)
    # for i in range(numpoints):
    #     av_logMh[i] = np.average(logmhalo[:, i])
    # plt.close('all')
    # plt.figure()
    # plt.fill_between(logMs, conf_min_logMh, conf_max_logMh, alpha=0.3)
    # plt.plot(logMs, av_logMh, label=str(redshifts[idx_z])+'<z<'+str(redshifts[idx_z+1]))
    # plt.xlabel('Log($M_{*}/M_{\odot}$)', size=20)
    # plt.ylabel('Log($M_{h}/M_{\odot}$)', size=20)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('../MCMC/Plots/SigmaHMvsSM' + str(idx_z) + "_niter=" +
    #     str(iterations) + "_burn=" + str(burn) + '.pdf')
    return av_logMh, conf_min_logMh, conf_max_logMh


def plotAllSigmaHMvsSM(iterations, burn):
    load_smf()
    load_hmf()
    plt.close('all')
    plt.figure()
    numpoints = 100
    logMs = np.linspace(9, 12, num=numpoints)
    for idx_z in range(numzbin):
        chainfile = "../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
        chain = np.load(chainfile)
        samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
        logmhalo = np.zeros([samples.shape[0], numpoints])
        for idx_simu in range(samples.shape[0]):
            M1, Ms0, beta, delta, gamma = samples[idx_simu]
            logmhalo[idx_simu, :] = logMh(logMs, M1, Ms0, beta, delta, gamma)
        av_logMh = np.average(logmhalo, axis=0)
        conf_min_logMh = np.percentile(logmhalo, 16, axis=0)  # 16th percentile = median - 1sigma (68% confidence interval)
        conf_max_logMh = np.percentile(logmhalo, 84, axis=0)
        # for i in range(numpoints):
        #     av_logMh[i] = np.average(logmhalo[:, i])
        plt.fill_between(logMs, conf_min_logMh, conf_max_logMh, alpha=0.3)
        plt.plot(logMs, av_logMh, label=str(redshifts[idx_z])+'<z<'+str(redshifts[idx_z+1]))
    plt.xlabel('Log($M_{*}/M_{\odot}$)', size=20)
    plt.ylabel('Log($M_{h}/M_{\odot}$)', size=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../MCMC/Plots/SigmaHMvsSM_Allz_niter=' +
        str(iterations) + "_burn=" + str(burn) + '.pdf')


def temp():
    # Plot Ms/Mh en ayant pris le Mh average. tester après en prenant la moyenne de Ms/Mh pour un Ms donné
    numpoints = 100
    logMs = np.linspace(9, 12, num=numpoints)
    for idx_z in range(10):
        plt.plot(tot[idx_z][0], logMs-tot[idx_z][0], label=str(redshifts[idx_z])+'<z<'+str(redshifts[idx_z+1]))
        plt.fill_between(tot[idx_z][0], logMs - tot[idx_z][1], logMs-tot[idx_z][2], alpha=0.3)
    plt.legend()
    plt.xlabel('Log($M_{h}/M_{\odot}$)', size=20)
    plt.ylabel('Log($M_{*}/M_{h}$)', size=20)


def plotSigmaSHMR(idx_z, iterations, burn):
    load_smf()
    load_hmf()
    chainfile = "../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
    chain = np.load(chainfile)
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    numpoints = 100
    logMs = np.linspace(9, 12, num=numpoints)
    logmhalo = np.zeros([samples.shape[0], numpoints])
    for idx_simu in range(samples.shape[0]):
        M1, Ms0, beta, delta, gamma = samples[idx_simu]
        logmhalo[idx_simu, :] = logMh(logMs, M1, Ms0, beta, delta, gamma)
    av_SHMR = np.average(logMs - logmhalo, axis=0)
    conf_min_SHMR = np.percentile(logMs - logmhalo, 16, axis=0)  # 16th percentile = median - 1sigma (68% confidence interval)
    conf_max_SHMR = np.percentile(logMs - logmhalo, 84, axis=0)
    # for i in range(numpoints):
    #     av_logMh[i] = np.average(logmhalo[:, i])
    plt.close('all')
    plt.figure()
    plt.fill_between(logMs, conf_min_SHMR, conf_max_SHMR, alpha=0.3)
    plt.plot(logMs, av_SHMR, label=str(redshifts[idx_z])+'<z<'+str(redshifts[idx_z+1]))
    plt.xlabel('Log($M_{*}/M_{\odot}$)', size=20)
    plt.ylabel('Log($M_{*}/M_{h}$)', size=20)
    plt.legend()
    plt.tight_layout()
    # plt.savefig('../MCMC/Plots/SigmaSHMRvsSM' + str(idx_z) + "_niter=" +
        # str(iterations) + "_burn=" + str(burn) + '.pdf')
    return av_logMh, conf_min_logMh, conf_max_logMh


def plotAllSigmaSHMRvsSM(iterations, burn):
    load_smf()
    load_hmf()
    numpoints = 100
    logMs = np.linspace(9, 12, num=numpoints)
    plt.close('all')
    plt.figure()
    for idx_z in range(numzbin):
        chainfile = "../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
        chain = np.load(chainfile)
        samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
        logmhalo = np.zeros([samples.shape[0], numpoints])
        for idx_simu in range(samples.shape[0]):
            M1, Ms0, beta, delta, gamma = samples[idx_simu]
            logmhalo[idx_simu, :] = logMh(logMs, M1, Ms0, beta, delta, gamma)
        av_logMh = np.average(logmhalo, axis=0)
        av_SHMR = np.average(logMs - logmhalo, axis=0)
        conf_min_SHMR = np.percentile(logMs - logmhalo, 16, axis=0)  # 16th percentile = median - 1sigma (68% confidence interval)
        conf_max_SHMR = np.percentile(logMs - logmhalo, 84, axis=0)
    # for i in range(numpoints):
    #     av_logMh[i] = np.average(logmhalo[:, i])
        plt.fill_between(av_logMh, conf_min_SHMR, conf_max_SHMR, alpha=0.3)
        plt.plot(av_logMh, av_SHMR, label=str(redshifts[idx_z])+'<z<'+str(redshifts[idx_z+1]))
    plt.xlabel('Log($M_{*}/M_{\odot}$)', size=20)
    plt.ylabel('Log($M_{*}/M_{h}$)', size=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../MCMC/Plots/SigmaSHMRvsSM_All_niter=' +
        str(iterations) + "_burn=" + str(burn) + '.pdf')


def plotFakeAllSigmaSHMRvsMH(iterations, burn):
    load_smf()
    load_hmf()
    plt.close('all')
    plt.figure()
    numpoints = 100
    logMs = np.linspace(9, 12, num=numpoints)
    for idx_z in range(numzbin):
        chainfile = "../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
        chain = np.load(chainfile)
        samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
        logmhalo = np.zeros([samples.shape[0], numpoints])
        for idx_simu in range(samples.shape[0]):
            M1, Ms0, beta, delta, gamma = samples[idx_simu]
            logmhalo[idx_simu, :] = logMh(logMs, M1, Ms0, beta, delta, gamma)
        av_logMh = np.average(logmhalo, axis=0)
        conf_min_logMh = np.percentile(logmhalo, 16, axis=0)  # 16th percentile = median - 1sigma (68% confidence interval)
        conf_max_logMh = np.percentile(logmhalo, 84, axis=0)
        # for i in range(numpoints):
        #     av_logMh[i] = np.average(logmhalo[:, i])
        plt.fill_between(av_logMh, logMs - conf_min_logMh, logMs - conf_max_logMh, alpha=0.3)
        plt.plot(av_logMh, logMs - av_logMh, label=str(redshifts[idx_z])+'<z<'+str(redshifts[idx_z+1]))
    # plt.plot([10.519, 10.693, 10.968, 11.231, 11.337, 11.691, 11.940, 12.219, 12.610],
    #     [-3.232, -3.072, -2.828, -2.629, -2.488, -2.306, -2.172, -2.057, -2.010], label='Harikane z=4')
    # plt.plot([10.975, 11.292, 12.041]+np.log10(67/70) , [-2.36, -2.206, -2.132], label='Hariakne z=6')
    plt.xlabel('Log($M_{h}/M_{\odot}$)', size=20)
    plt.ylabel('Log($M_{*}/M_{h}$)', size=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../MCMC/Plots/SigmaFAKE_SHRMvsHM_Allz_niter=' +
        str(iterations) + "_burn=" + str(burn) + '.pdf')


def plotAllSHMRvsSM(directory, iterations, burn):
    load_smf()
    load_hmf()
    plt.close('all')
    plt.figure()
    numpoints = 100
    logMs = np.linspace(9, 11.5, num=numpoints)
    logMhbins = np.linspace(11.5, 14, num=numpoints)
    avg_MSonMH = np.zeros([numzbin, numpoints-1])
    confminus_MSonMH = np.zeros([numzbin, numpoints-1])
    confplus_MSonMH = np.zeros([numzbin, numpoints-1])
    for idx_z in range(numzbin):
        chainfile = directory+"/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
        chain = np.load(chainfile)
        samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
        logmhalo = np.zeros([samples.shape[0], numpoints])
        for idx_simu in range(samples.shape[0]):
            M1, Ms0, beta, delta, gamma, ksi = samples[idx_simu]
            logmhalo[idx_simu, :] = logMh(logMs, M1, Ms0, beta, delta, gamma)
        for idx_bin in range(numpoints-1):
            idx_MhinBin = np.where(
                            np.logical_and(
                                logmhalo >= logMhbins[idx_bin],
                                logmhalo < logMhbins[idx_bin+1]
                            )
            )
            smhm_tmp = logMs[idx_MhinBin[1]] - logmhalo[idx_MhinBin]
            avg_MSonMH[idx_z, idx_bin] = np.average(smhm_tmp)
            confminus_MSonMH[idx_z, idx_bin] = np.percentile(smhm_tmp, 16, axis=0)
            confplus_MSonMH[idx_z, idx_bin] = np.percentile(smhm_tmp, 84, axis=0)
        plt.plot((logMhbins[1:] + logMhbins[:-1])/2, avg_MSonMH[idx_z],
            label=str(redshifts[idx_z])+'<z<'+str(redshifts[idx_z+1]))
        plt.fill_between((logMhbins[1:] + logMhbins[:-1])/2,
            confminus_MSonMH[idx_z], confplus_MSonMH[idx_z], alpha=0.3)
    plt.xlabel('Log($M_{h}/M_{\odot}$)', size=20)
    plt.ylabel('Log($M_{*}/M_{h}$)', size=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory+'/Plots/Sigma_SHRMvsHM_Allz_niter=' +
        str(iterations) + "_burn=" + str(burn) + '.pdf')

"""Plots and tests"""


# logMs = np.linspace(6, 12, num=100)
# plt.plot(logMs, logMh(logMs, 13, 10, 0.5, 0.5, 2.5))

# logmhtest =logMh(logMs, 13, 14, 0.5, 0.5, 2.5)
# plt.plot(logMh(logMs, 12, 10, 0.5, 0.5, 2.5), logMs - logMh(logMs, 12, 10, 0.5, 0.5, 2.5))

# Compare Observed and predicted SMF :
# load_smf()
# load_hmf()
# select = np.where(smf_cosmos[idx_z][:, 1] > -1000)[0]
# logMs = smf_cosmos[idx_z][select, 0]
# plt.errorbar(logMs, smf_cosmos[idx_z][select, 1],
#     yerr=[smf_cosmos[idx_z][select, 3], smf_cosmos[idx_z][select, 2]], fmt='o')
# plt.ylim(-6, 0)
# # logphi = log_phi_direct(logMs, idx_z, 12.2, 10.8, 0.3, 0, 0.3)
# """ Leauthaud fit parameters for idx_z=0, we note a small difference maybe coming form the HMF"""
# # logphi = log_phi_direct(logMs, idx_z, 12.52, 10.916, 0.457, 0.566, 1.53)
# # # logphi = log_phi_true(logMs, idx_z, 12.52, 10.916, 0.457, 0.566, 1.53, 0.206**2)
# # logphi = log_phi_direct(logMs, idx_z, 12.518, 10.917, 0.456, 0.582, 1.48)

# """ Leauthaud fit parametres for idx_z=1 """
# # logphi = log_phi_direct(logMs, idx_z, 12.725, 11.038, 0.466, 0.61, 1.95)

# logphi = log_phi_direct(logMs, idx_z, 12.725, 11.038, 0.466, 0.61, 0.7) # fits better with smaller gamma

# plt.plot(logMs, logphi)

# logphi = log_phi_true(logMs, idx_z, M1, Ms0, beta, delta, gamma, ksi)
# logmhalo = logMh(logMs, M1, Ms0, beta, delta, gamma)
# plt.plot(logMs, logmhalo)

"""Good fit by eye for the idx_z=1, no_ksi
starting_point = ([12.7, 11.1, 0.5, 0.3, 1.2])
"""

# chi2_noksi(0, 12.7, 8.9, 0.3, 0.6, 2.5)
# theta = np.array([12.7, 8.9, 0.3, 0.6, 2.5])
# theta = np.array([ 11.73672883,  10.63457168 ,  0.55492575 ,  0.45137568  , 2.58689832])

# plt.plot(hmf[0][:,0], hmf[0][:,2])


# thetavar = np.array([np.linspace(10, 14, num=100), np.full(100, 11), np.full(100,0.5),
# np.full(100,0.5), np.full(100,2.5), np.full(100,0.15)])

# neglog = np.zeros(100)
# idx_z = 0
# for i in range(100):
#     neglog[i] = negloglike(thetavar[:,i], idx_z)

# plt.plot(neglog)

# for i in range(ndim):
#     plt.figure()
#     for j in range(nwalker):
#         plt.plot(chain[j,  :, i], '.')
#     plt.show()



# """Test emcee sampling"""

# nwalker=250
# ndim=6
# std = np.array([1, 1, 0.1, 0.1, 0.1, 0.1])
# p0 = emcee.utils.sample_ball(theta0, std, size=nwalker)
# sampler= emcee.EnsembleSampler(nwalker, ndim, negloglike, args=[idx_z])

# pos, prob, state = sampler.run_mcmc(p0, 100) ## burn phase

# sampler.run_mcmc(pos, 1000) ## samble phase


"""Select the chains that converged"""

# chain = np.load("../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy")
# lnprob = np.load("../MCMC/Chain/LnProb_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy")
# for k in range(20):
#     plt.plot(lnprob[k, :])
# select = np.where(lnprob[:, -1]>-30)
# chain = chain[lnprob[:, -1]>-30, :, :]