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
import getconf
from shutil import copyfile


def load_smf(smf_name):
    """Load the SMF"""
    if smf_name == 'cosmos':
        print('Use the COSMOS SMF')
        """Load the SMF from Iary Davidzon+17"""
        # redshifts of the Iari SMF
        global redshifts
        global redshiftsbin
        redshifts = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
        redshiftsbin = (redshifts[1:]+redshifts[:-1])/2
        global numzbin
        numzbin = np.size(redshifts) - 1
        print('numzbin: '+str(numzbin))
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

        for i in range(numzbin):
            # Correction of the comoving Volume :
            VmaxD17 = D17_Cosmo.comoving_volume(redshifts[i+1]) - D17_Cosmo.comoving_volume(redshifts[i])
            VmaxBP = BP_Cosmo.comoving_volume(redshifts[i+1]) - BP_Cosmo.comoving_volume(redshifts[i])

            """In the case where we use the Vmax points and not the VmaxFit, the errors bars are relative and
            are not the absolute uncertainty as in the Vmax Fit, so we don't rescale the error bars"""
            smf_cosmos[i][:, 1] = smf_cosmos[i][:, 1] + np.log10(VmaxD17/VmaxBP)

            # Correction of the measured stellar mass
            # Equivalent to multiply by (BP_Cosmo.H0/D17_Cosmo.H0)**-2
            smf_cosmos[i][:, 0] = smf_cosmos[i][:, 0] - 2 * np.log10(BP_Cosmo.H0/D17_Cosmo.H0)


def load_hmf(hmf_name):
    """Load the HMF"""
    global hmf
    hmf = []
    if hmf_name in ['bolshoi_tot', 'bolshoi_cen']:
        """Load HMF from Bolshoi Planck simulation"""
        print('Use Bolshoi tot or central MF (default central)')
        # redshifts of the BolshoiPlanck files
        redshift_haloes = np.arange(0, 10, step=0.1)
        numredshift_haloes = np.size(redshift_haloes)
        """Definition of hmf columns :
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
        for i in redshift_id_selec:
            hmf.append(hmf_bolshoi[i])
        if hmf_name == 'bolshoi_tot':
            # In the bolshoi tot case, change to the correct bolshoi HMF
            print('Use Bolshoi total HMF')
            for i in range(numzbin):
                hmf[i][:, 1] = hmf[i][:, 2]

    if hmf_name == 'tinker200':
        """Load Tinker+08 HMF computed with HFMCalc of Murray+13
        parameters : Delta = 200 times the mean density of the universe (same at all z)
        """
        print('Use Tinker 200 HMF')
        redshift_haloes = np.array([0.35, 0.65, 0.95, 1.3, 1.75, 2.25, 2.75, 3.25, 4, 5])
        numredshift_haloes = len(redshift_haloes)
        for i in range(numredshift_haloes):
            hmf.append(
                np.loadtxt('../Data/Tinker08HMF/HMFCalc_Dm200/mVector_PLANCK-SMT_z{:1.2f}.txt'.format(
                    redshift_haloes[i]), usecols=(0, 7)))
            hmf[i][:, 0] = np.log10(hmf[i][:, 0] / 0.6774)
            hmf[i][:, 1] = np.log10(hmf[i][:, 1] * (0.6774)**3)



"""Function definitions for computation of the theroretical SFM phi_true"""


def logMh(logMs, M1, Ms0, beta, delta, gamma):
    """SM-HM relation"""
    return M1 + beta*(logMs - Ms0) + (10 ** (delta * (logMs - Ms0))) / (1 + (10 ** (-gamma * (logMs - Ms0)))) - 0.5


def log_phi_direct(logMs, idx_z, M1, Ms0, beta, delta, gamma):
    """"SMF obtained from the SM-HM relation and the HMF"""
    epsilon = 0.0001
    log_Mh1 = logMh(logMs, M1, Ms0, beta, delta, gamma)
    log_Mh2 = logMh(logMs + epsilon, M1, Ms0, beta, delta, gamma)
    # Select the index of the HMF corresponing to the halo masses
    index_Mh = np.argmin(
        np.abs(
            np.tile(hmf[idx_z][:, 0], (len(log_Mh1), 1)) -
            np.transpose(np.tile(log_Mh1, (len(hmf[idx_z][:, 0]), 1)))
        ), axis=1)
    log_phidirect = hmf[idx_z][index_Mh, 1] + np.log10((log_Mh2 - log_Mh1)/epsilon)

    log_phidirect[log_Mh1 > hmf[idx_z][-1, 0]] = -1000 # Do not use points with halo masses not defined in the HMF
    log_phidirect[log_Mh1 < hmf[idx_z][0, 0]] = -1000

    return log_phidirect


def log_phi_true(logMs, idx_z, M1, Ms0, beta, delta, gamma, ksi):
    """Use the approximation of the convolution defined in Behroozi et al 2010 equation (3)"""
    epsilon = 0.01 * logMs
    logphi1 = log_phi_direct(logMs, idx_z, M1, Ms0, beta, delta, gamma)
    logphi2 = log_phi_direct(logMs + epsilon, idx_z, M1, Ms0, beta, delta, gamma)
    logphitrue = logphi1 + ksi**2 / 2 * np.log(10) * ((logphi2 - logphi1)/epsilon)**2
    return logphitrue



def chi2(idx_z, M1, Ms0, beta, delta, gamma, ksi):
    """"return the chi**2 between the observed and the expected SMF"""
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


def loglike(theta, idx_z, minbound, maxbound):
    """return the loglikelihood"""
    if all(theta >= minbound[idx_z]) and all(theta <= maxbound[idx_z]):
        M1, Ms0, beta, delta, gamma, ksi = theta[:]
        return -chi2(idx_z, M1, Ms0, beta, delta, gamma, ksi)/2
    else:
        return -np.inf
   

def negloglike(theta, idx_z, minbound, maxbound):
    return -loglike(theta, idx_z, minbound, maxbound)


""" Run MCMC """

def runMCMC_allZ(paramfile):
    # Load parameters and config
   
    config = getconf.ConfigGetter('getconf', [paramfile])
    save_path = config.getstr('Path.save_path')
    smf_name = config.getstr('Mass_functions.SMF')
    hmf_name = config.getstr('Mass_functions.HMF')
    iterations = config.getint('MCMC_run_parameters.iterations')
    burn = config.getint('MCMC_run_parameters.burn')
    minboundfile = config.getstr('Values.minbound')
    maxboundfile = config.getstr('Values.maxbound')
    minbound = np.loadtxt(minboundfile, delimiter=',')
    maxbound = np.loadtxt(maxboundfile, delimiter=',')
    starting_point_file = config.getstr('Values.starting_point') 
    starting_point = np.loadtxt(starting_point_file, delimiter=',')
    # np.array(config.getlist('MCMC_run_parameters.starting_point')).astype('float')
    std = np.array(config.getlist('MCMC_run_parameters.std')).astype('float')
    nthreads = config.getint('MCMC_run_parameters.nthreads')
    nwalkers = config.getint('MCMC_run_parameters.nwalkers')
    # global numzbin
    load_smf(smf_name)
    load_hmf(hmf_name)
    
    # Create save direcory
    now = datetime.datetime.now()
    directory = save_path + "MCMC_"+str(now.year)+'-'+str(now.month)+'-'+str(now.day)+'T'+str(now.hour)+'-'+str(now.minute)
    print('Save direcory : ' + directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(directory+'/Chain')
        os.makedirs(directory+'/Plots')
        os.makedirs(directory+'/Plots/MhaloPeak')
        os.makedirs(directory+'/Results')
        print('Created new directory')
    # Copy parameter files in the save directory
    copyfile(paramfile, directory + '/' + paramfile)
    copyfile(minboundfile, directory + '/' + minboundfile)
    copyfile(maxboundfile, directory + '/' + maxboundfile)

    # run all MCMC for all zbins
    for idx_z in range(numzbin):
    # for idx_z in [6, 7, 8, 9]:
        print('Starting MCMC run for idx_z =' + str(idx_z) )
        print('Min bound: ' + str(minbound[idx_z]))
        print('Max bound: ' + str(maxbound[idx_z]))
        runMCMC(directory, minbound, maxbound, idx_z, starting_point, std, iterations, burn, nthreads, nwalkers)


def runMCMC(directory,  minbound, maxbound, idx_z, starting_point, std, iterations, burn, nthreads, nwalkers):
    # load_smf()
    # load_hmf()
    # nwalker = 20
    # starting_point =  np.array([12.5, 10.8, 0.5, 0.5, 0.5, 0.15])
    # std =np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.01])
    # starting_point =  np.array([12.5, 11, 0.5, 0.7, 0.5, 0.15])
    start_time = time.time()
    p0 = emcee.utils.sample_ball(starting_point[idx_z], std, size=nwalkers)
    ndim = len(starting_point[idx_z])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike, args=[idx_z, minbound, maxbound], threads=nthreads)
    print("idx_z = " +str (idx_z))
    print("ndim = " + str(ndim))
    print("start = " + str(starting_point[idx_z]))
    print("std = " + str(std))
    print("iterations = " + str(iterations))
    print("burn = " + str(burn))
    start_time = time.time()
    sampler.run_mcmc(p0, iterations)
    elapsed_time = time.time() - start_time
    print('Time elapsed : ' + str(elapsed_time))
    print('acceptance fraction :')
    print(sampler.acceptance_fraction)
    # Save chains and loglike of chains
    chainfile = directory + "/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
    savenameln = directory + "/Chain/LnProb_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
    np.save(chainfile, sampler.chain)
    np.save(savenameln, sampler.lnprobability)
    sampler.reset()
    # Plot all relevant figures
    plt.close('all')
    plotchain(directory, chainfile, idx_z, iterations, burn)
    plt.close('all')
    # plotdist(directory, chainfile, idx_z, iterations, burn)
    # plt.close('all')
    plotSMF(directory, idx_z, iterations, burn)
    plt.close('all')
    plotSMHM(directory, idx_z, iterations, burn)
    plt.close('all')
    plot_Mhpeak(directory, chainfile, idx_z, iterations, burn)
    plt.close('all')
    save_results(directory, chainfile, idx_z, iterations, burn)


def save_results(directory, chainfile, idx_z, iterations, burn):
    chain = np.load(chainfile)
    names = ['$M_{1}$', '$M_{s,0}$', '$\\beta$', '$\delta$', '$\gamma$', 'ksi']
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    samples = MCSamples(samples = samples, names = names)
    res = samples.getTable()
    res.write(directory+"/Results/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".txt")
    del chain

    
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


"""Plots"""


def plotSMF(directory, idx_z, iterations, burn):
    # load_smf()
    # load_hmf()
    chainfile = directory + "/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
    chain = np.load(chainfile)
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    # chain.close()
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
    # load_smf()
    # load_hmf()
    chainfile =  directory + "/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
    chain =  np.load(chainfile)
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    # chain.close()
    logMs = np.linspace(9, 11.5, num=200)
    for M1, Ms0, beta, delta, gamma, ksi in samples[np.random.randint(len(samples), size=100)]:
        logmhalo = logMh(logMs, M1, Ms0, beta, delta, gamma)
        logphi = log_phi_true(logMs, idx_z, M1, Ms0, beta, delta, gamma, ksi)
        plt.plot(logmhalo, logMs-logmhalo, color="k", alpha=0.1)
    plt.savefig(directory+'/Plots/SMHM_ksi'+ str(idx_z) + "_niter=" + str(iterations) + '.pdf')


# def plotHMvsSM_noksi(idx_z, iterations, burn):
#     load_smf()
#     load_hmf()
#     chain = np.load("../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy")
#     samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
#     logMs = np.linspace(9, 11.5, num=200)
#     for M1, Ms0, beta, delta, gamma in samples[np.random.randint(len(samples), size=100)]:
#         logmhalo = logMh(logMs, M1, Ms0, beta, delta, gamma)
#         plt.plot(logMs, logmhalo, color="k", alpha=0.1)
#     # plt.show()
#     plt.savefig('../MCMC/Plots/HMvsSM_noksi'+ str(idx_z) + "_niter=" + str(iterations) + '.pdf')


def plotchain(directory, chainfile, idx_z, iterations, burn):
    figname = directory + "/Plots/Ksi_z" + str(idx_z) + "_niter=" + str(iterations) + "_burn=" + str(burn)
    chain = np.load(chainfile)
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    # chain.close()
    fig = corner.corner(
        samples, labels=['$M_{1}$', '$M_{*,0}$', '$\\beta$', '$\delta$', '$\gamma$', 'ksi'])
    fig.savefig(figname + ".pdf")
    plt.close('all')


def plotdist(directory, chainfile, idx_z, iterations, burn):
    figname = directory + "/Plots/Ksi_z" + str(idx_z) + "_niter=" + str(iterations) + "_burn=" + str(burn)
    names = ['$M_{1}$', '$M_{s,0}$', '$\\beta$', '$\delta$', '$\gamma$', 'ksi']
    chain = np.load(chainfile)
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    samples = MCSamples(samples = samples, names = names)
    # chain.close()
    g = plots.getSubplotPlotter()
    g.triangle_plot(samples, filled=True)
    g.export(figname + '_gd.pdf' )
    plt.close('all')


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


def plotSigmaHMvsSM(directory, idx_z, iterations, burn):
    # load_smf()
    # load_hmf()
    chainfile = directory + "/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
    chain = np.load(chainfile)
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    # chain.close()
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
    # load_smf()
    # load_hmf()
    plt.close('all')
    plt.figure()
    numpoints = 100
    logMs = np.linspace(9, 12, num=numpoints)
    for idx_z in range(numzbin):
        chainfile = "../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
        chain = np.load(chainfile)
        samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
        # chain.close()
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
    # load_smf()
    # load_hmf()
    chainfile = "../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
    chain = np.load(chainfile)
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    # chain.close()
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
    # load_smf()
    # load_hmf()
    numpoints = 100
    logMs = np.linspace(9, 12, num=numpoints)
    plt.close('all')
    plt.figure()
    for idx_z in range(numzbin):
        chainfile = "../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
        chain = np.load(chainfile)
        samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
        # chain.close()
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
    # load_smf()
    # load_hmf()
    plt.close('all')
    plt.figure()
    numpoints = 100
    logMs = np.linspace(9, 12, num=numpoints)
    for idx_z in range(numzbin):
        chainfile = "../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
        chain = np.load(chainfile)
        samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
        # chain.close()
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
    load_smf('cosmos')
    load_hmf('bolshoi_tot')
    #plt.close('all')
    plt.figure()
    numpoints = 100
    Ms_min = np.maximum(np.log10(6.3 * 10**7 * (1 + (redshifts[1:] + redshifts[:-1]) / 2)**2.7), np.full(numzbin, 9))
    logMhbins = np.linspace(11.5, 14, num=numpoints)
    avg_MSonMH = np.zeros([numzbin, numpoints-1])
    confminus_MSonMH = np.zeros([numzbin, numpoints-1])
    confplus_MSonMH = np.zeros([numzbin, numpoints-1])
    for idx_z in range(1):
        logMs = np.linspace(Ms_min[idx_z], 11.8, num=numpoints)
        chainfile = directory+"/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
        chain = np.load(chainfile)
        samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
        samples = samples[np.random.randint(len(samples), size=100000)]
        # chain.close()
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
        plt.plot((logMhbins[1:] + logMhbins[:-1])/2, avg_MSonMH[idx_z], 'ro',
            label=str(redshifts[idx_z])+'<z<'+str(redshifts[idx_z+1]))
        plt.fill_between((logMhbins[1:] + logMhbins[:-1])/2,
            confminus_MSonMH[idx_z], confplus_MSonMH[idx_z], alpha=0.3)
    plt.xlabel('Log($M_{h}/M_{\odot}$)', size=20)
    plt.ylabel('Log($M_{*}/M_{h}$)', size=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory+'/Plots/Test=' +
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
