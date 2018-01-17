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
import corner
from getdist import plots, MCSamples


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
            # '../Data/Davidzon/Davidzon+17_SMF_V3.0/mf_mass2b_fl5b_tot_VmaxFit2D'
            # + str(i) + '.dat')
            '../Data/Davidzon/Davidzon+17_SMF_V3.0/mf_mass2b_fl5b_tot_Vmax'
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
        smf_cosmos[i][:, 1] = smf_cosmos[i][:, 1] + np.log10(VmaxD17/VmaxBP)
        smf_cosmos[i][:, 2] = smf_cosmos[i][:, 2] + np.log10(VmaxD17/VmaxBP)
        smf_cosmos[i][:, 3] = smf_cosmos[i][:, 3] + np.log10(VmaxD17/VmaxBP)

        # Correction of the measured stellar mass
        # Equivalent to multiply by (BP_Cosmo.H0/D17_Cosmo.H0)**-2
        smf_cosmos[i][:, 0] = smf_cosmos[i][:, 0] - 2 * np.log10(BP_Cosmo.H0/D17_Cosmo.H0)

"""Function definitions for computation of the theroretical SFM phi_true"""


def logMh(logMs, M1, Ms0, beta, delta, gamma):
    # SM-HM relation
    return M1 + beta*(logMs - Ms0) + (10 ** (delta * (logMs - Ms0))) / (1 + (10 ** (-gamma * (logMs - Ms0)))) - 0.5
    # Ms = 10**logMs
    # logMh = M1 + beta * np.log10(Ms / 10**Ms0) + (Ms / 10**Ms0)**delta / (1 + (Ms / 10**Ms0)**(-gamma)) - 0.5
    # return logMh


def log_phi_direct(logMs, idx_z, M1, Ms0, beta, delta, gamma):
    # SMF obtained from the SM-HM relation and the HMF
    epsilon = 0.01
    log_Mh1 = logMh(logMs, M1, Ms0, beta, delta, gamma)
    log_Mh2 = logMh(logMs + epsilon, M1, Ms0, beta, delta, gamma)
    # print(logMs)
    # print(log_Mh1, log_Mh2)
    # index_Mh = np.argmin(np.abs(hmf_bolshoi[idx_z][:, 0] - log_Mh1))
    index_Mh = np.argmin(np.abs(
        np.tile(hmf_bolshoi[idx_z][:, 0], (len(log_Mh1), 1)) -
        np.transpose(np.tile(log_Mh1, (len(hmf_bolshoi[idx_z][:, 0]), 1)))
    ), axis=1)  # Select the index of the HMF corresponing to the halo masses
    log_phidirect = hmf_bolshoi[idx_z][index_Mh, 2] + np.log10((log_Mh2 - log_Mh1)/epsilon)
    # print(np.log10((log_Mh2 - log_Mh1)/epsilon))
    # Keep only points wher the halo mass is defined in the HMF
    log_phidirect[log_Mh1 > hmf_bolshoi[idx_z][-1, 0]] = -1000
    log_phidirect[log_Mh1 < hmf_bolshoi[idx_z][0, 0]] = -1000
    # print(log_phidirect)
    # print(hmf_bolshoi[idx_z][index_Mh, 2])
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
    select = np.where(smf_cosmos[idx_z][:, 1] > -6)  # select points where the smf is defined
    # We choose to limit the fit only fro abundances higher than 10**-6
    logMs = smf_cosmos[idx_z][select, 0]
    pred = log_phi_true(logMs, idx_z, M1, Ms0, beta, delta, gamma, ksi)
    chi2 = np.sum(
        ((pred -
                #smf_cosmos[idx_z][select, 1]) / ((smf_cosmos[idx_z][select, 3] + smf_cosmos[idx_z][select, 2])/2))**2
                smf_cosmos[idx_z][select, 1]) / smf_cosmos[idx_z][select, 2])**2
        )
    return chi2


# def chi2_noksi(idx_z, M1, Ms0, beta, delta, gamma):
#     # return the chi**2 between the observed and the expected SMF
#     select = np.where(np.logical_and(
#         smf_cosmos[idx_z][:, 1] > -6,  # select points where the smf is defined
#         smf_cosmos[idx_z][:, 3] < 900))[0]  # select points where the error bar is defined
#     # We choose to limit the fit only fro abundances higher than 10**-6
#     logMs = smf_cosmos[idx_z][select, 0]
#     pred = log_phi_direct(logMs, idx_z, M1, Ms0, beta, delta, gamma)
#     chi2 = np.sum(
#         ((pred -
#             # When using the VmaxFit2D (give the bands and not the sigma)
#             # smf_cosmos[idx_z][select, 1]) / ((smf_cosmos[idx_z][select, 3] - smf_cosmos[idx_z][select, 2])/2))**2
#             # When using the Vmax directly (give the error bars directly)
#             smf_cosmos[idx_z][select, 1]) / ((smf_cosmos[idx_z][select, 3] + smf_cosmos[idx_z][select, 2])/2))**2
#             # smf_cosmos[idx_z][select, 1]))**2
#         )
#     return chi2


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
    # bouds for the idx_z = 1
    M1, Ms0, beta, delta, gamma, ksi = theta[:]
    if beta < 0.3 or delta < 0.1 or gamma < 0:
        return -np.inf
    if beta > 0.6 or delta > 0.7 or gamma > 3:
        return -np.inf
    if M1 < 12 or M1 > 13 or Ms0 < 10.5 or Ms0 > 12:
        return -np.inf
    if ksi < 0 or ksi > 1:
        return -np.inf
    else:
        return -chi2(idx_z, M1, Ms0, beta, delta, gamma, ksi)/2

# def loglike_noksi(theta, idx_z):
#     # return the likelihood for a fixed ksi
#     # print(theta)
#     M1, Ms0, beta, delta, gamma = theta[:]
#     if beta < 0 or delta < 0 or gamma < -2:
#         return -np.inf
#     if beta > 2 or delta >  2 or gamma > 3:
#         return -np.inf
#     if M1 < 10 or M1 > 14 or Ms0 < 10 or Ms0 > 14:
#         return -np.inf
#     else:
#         return -chi2_noksi(idx_z, M1, Ms0, beta, delta, gamma)/2


def negloglike(theta, idx_z):
    return -loglike(theta, idx_z)


# def negloglike_noksi(theta, idx_z):
#     return -loglike_noksi(theta, idx_z)

"""Find maximum likelihood estimation"""


def maxlikelihood(idx_z, theta0, bounds):
    load_hmf()
    load_smf()
    # idx_z = 0
    # theta0 = np.array([11, 10, 0.1, 0.1, 1])
    # bounds = ((11, 13), (10, 12), (0, 1), (0, 1), (0, 3), (0, 1))
    # results = op.minimize(negloglike_noksi, theta0, bounds=bounds, args=(idx_z), method='TNC')
    # results = op.basinhopping(negloglike, theta0, niter=1, T=1000, minimizer_kwargs={'args': idx_z})
    # results = op.minimize(negloglike_noksi, theta0, args=(idx_z), method='Nelder-Mead', options={'fatol':10**-6})
    results = op.minimize(negloglike, theta0, args=(idx_z), method='Nelder-Mead', options={'fatol':10**-6})
    print(results)



"""Plot chain file"""


def plotSMF(idx_z, iterations, burn):
    load_smf()
    load_hmf()
    chain = np.load("../MCMC/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy")
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    select = np.where(smf_cosmos[idx_z][:, 1] > -1000)[0]
    logMs = smf_cosmos[0][select, 0]
    plt.errorbar(logMs, smf_cosmos[0][select, 1], 
        yerr=[smf_cosmos[0][select, 3], smf_cosmos[0][select, 2]], fmt='o')
    plt.ylim(-6, 0)
    for M1, Ms0, beta, delta, gamma, ksi in samples[np.random.randint(len(samples), size=100)]:
        logphi = log_phi_true(logMs, idx_z, M1, Ms0, beta, delta, gamma, ksi)
        plt.plot(logMs, logphi, color="k", alpha=0.1)
    plt.show()


def plotchain(chainfile, idx_z, iterations, burn):
    chain = np.load(chainfile)
    figname = "../MCMC/Plots/Ksi_z" + str(idx_z) + "_niter=" + str(iterations) + "_burn=" + str(burn)

    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    fig = corner.corner(
        samples, labels=['$M_{1}$', '$M_{*,0}$', '$\\beta$', '$\delta$', '$\gamma$', 'ksi'])
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



""" Run MCMC """


def runMCMC(idx_z, starting_point, std, iterations, burn):
    load_hmf()
    load_smf()
    nwalker = 20
    nthreads = 1  # Put more for multiprocessing automatically.
    # starting_point =  np.array([12.5, 10.8, 0.5, 0.5, 0.5, 0.15])
    # std =np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.01])

    p0 = emcee.utils.sample_ball(starting_point, std, size=nwalker)
    ndim = len(starting_point)
    sampler = emcee.EnsembleSampler(nwalker, ndim, loglike, args=[idx_z], threads=nthreads)

    print("ndim = " + str(ndim))
    print("start = " + str(starting_point))
    print("std = " + str(std))
    print("iterations = " + str(iterations))

    sampler.run_mcmc(p0, iterations)
    chainfile = "../MCMC/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
    np.save(chainfile, sampler.chain)
    
    plotchain(chainfile, idx_z, iterations, burn)


def runMCMC_noksi(idx_z, starting_point, std, iterations, burn):
    load_hmf()
    load_smf()
    nwalker = 12
    nthreads = 1  # Put more for multiprocessing automatically.
    # starting_point = np.array([12, 11, 0.5, 0.5, 2.5])
    # std = np.array([1, 1, 0.1, 0.1, 0.1])

    p0 = emcee.utils.sample_ball(starting_point, std, size=nwalker)
    ndim = len(starting_point)
    sampler = emcee.EnsembleSampler(nwalker, ndim, loglike_noksi, args=[idx_z], threads=nthreads)

    print("ndim = " + str(ndim))
    print("start = " + str(starting_point))
    print("std = " + str(std))
    print("iterations = " + str(iterations))

    sampler.run_mcmc(p0, iterations)
    savename = "../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
    np.save(savename, sampler.chain)

    plotchain(savename, idx_z, iterations, burn)


def save_results(chainfile, idx_z, iterations, burn):
    chain = np.load(chainfile)
    names = ['$M_{1}$', '$M_{s,0}$', '$\\beta$', '$\delta$', '$\gamma$', 'ksi']
    samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    samples = MCSamples(samples = samples, names = names)
    res = samples.getTable()
    res.write("../MCMC/Results/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".txt")


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

"""Plots and tests"""


# logMs = np.linspace(6, 12, num=100)
# plt.plot(logMs, logMh(logMs, 13, 10, 0.5, 0.5, 2.5))

# logmhtest =logMh(logMs, 13, 14, 0.5, 0.5, 2.5)
# plt.plot(logMh(logMs, 12, 10, 0.5, 0.5, 2.5), logMs - logMh(logMs, 12, 10, 0.5, 0.5, 2.5))

# Compare Observed and predicted SMF :
# load_smf()
# load_hmf()
# select = np.where(smf_cosmos[idx_z][:, 1] > -1000)[0]
# logMs = smf_cosmos[0][select, 0]
# plt.errorbar(logMs, smf_cosmos[0][select, 1], 
#     yerr=[smf_cosmos[0][select, 3], smf_cosmos[0][select, 2]], fmt='o')
# plt.ylim(-6, 0)
# # logphi = log_phi_direct(logMs, 0, 12.5, 10.5, 0.5, 0.3, 2)
# logphi = log_phi_true(logMs, 0, M1, Ms0, beta, delta, gamma, ksi)

# plt.plot(logMs, logphi)

# chi2_noksi(0, 12.7, 8.9, 0.3, 0.6, 2.5)
# theta = np.array([12.7, 8.9, 0.3, 0.6, 2.5])
# theta = np.array([ 11.73672883,  10.63457168 ,  0.55492575 ,  0.45137568  , 2.58689832])

# plt.plot(hmf_bolshoi[0][:,0], hmf_bolshoi[0][:,2])


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