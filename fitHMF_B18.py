# -*-coding:Utf-8 -*

from IPython.display import display, Math
"""
Fit a HMF function to the data outputs of Bolshoi-Planck15 given by Behroozi et al. 18 in the Universe machine code.
This is in scope of using th epeak halo mass of the halo as the HMF inour SHAM code.

Started on January 18th 2019 by Louis Legrand at IAP and IAS.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from colossus import defaults
from colossus.lss import mass_function
from colossus.halo import mass_so
from colossus.lss import peaks
from colossus.cosmology import cosmology
import emcee
import corner
from multiprocessing import Pool
cosmology.setCosmology('planck15')

os.environ["OMP_NUM_THREADS"] = "1"


def load_um_hmf():
    """Load the hmf of universe machine code, which is using the peak halo mass and not the actual mass of the halo.
    The simulation is in the Planck 2015 cosmology.

    Returns:
    hmf : a dictionary with the redshift as index and giving the HMFs in numpy array.
        Peak halo masses in Msun
        Number densities in comoving Mpc^-3 dex^-1.
        Columns of the HMFS[a] array = Log10(HM) Number_Density Err+ Err- Satellite_fraction Err+ Err- HM_Left_Edge HM_Right_Edge
    redshifts : numpy array of the redshifts where the HMFs are evaluated
    """
    path = '../Data/umachine-edr/data/hmfs/'
    list_files = os.listdir(path)
    hmf = {}
    for file in list_files:
        a = float(file[5:-4])
        z = 1/a - 1
        hmf[z] = np.loadtxt(path + file).T
    redshifts = np.array(sorted(hmf.keys()))

    return hmf, redshifts


def plot_model(theta, hmf, z, mass_min, mdef):
    """Plot the hmf of bolshoi planck at a given redshift
    Aslo plot the Despali hmf modified"""
    cosmo = cosmology.getCurrent()
    plt.figure()
    plt.errorbar(hmf[z][0], hmf[z][1], yerr=[hmf[z][2], hmf[z][3]], label='Bolshoi planck at z =' +str(z))
    M = 10**hmf[z][0] /cosmo.h
    despalihmf = modelDespali16_fit(theta, M, z, mdef)
    plt.plot(hmf[z][0], despalihmf, label='Modified Despali HMF')
    plt.axvline(mass_min, linestyle='--', label='minimal mass fitted')
    myList = reversed(hmf[z][1])
    idx_massmax = -next((i for i, x in enumerate(myList) if x), None) - 1
    plt.axvline(hmf[z][0, idx_massmax-1], linestyle='--', label='max mass fitted')
    plt.yscale('log')
    plt.xlabel('Peak halo masses in Msun')
    plt.ylabel('Number densities in comoving Mpc^-3 dex^-1')
    plt.legend()
    plt.show()


def plot_all(hmf, redshifts):
    plt.figure()
    for z in redshifts[redshifts < 5][::10]:
        plt.errorbar(hmf[z][0], hmf[z][1], yerr=[
                     hmf[z][2], hmf[z][3]], label=str(z), fmt='o')
    plt.yscale('log')
    plt.xlabel('Peak halo masses in Msun')
    plt.ylabel('Number densities in comoving Mpc^-3 dex^-1')
    plt.legend()


# def despali(hmf, z):
#     """Get the Despali HMF on the HMF from the Bolshoi PLanck simulation.

#     Inputs:
#         hmf : HMFs dictionnary from load_um_hmf
#         z : redshift corresponding to one of the keys of hmf
#     """
#     mfunc = mass_function.massFunction(
#         hmf[z][0], z, mdef='200m', model='despali16')


def modelDespali16_fit(theta, M, z, mdef, deltac_args={'corrections': True}):
    """
    The mass function model of Despali et al 2016.
    This function is updated from the Colossus function to allow to fit paramters on the BP15 HMF.

    The parameters were fit for a number of different mass definitions, redshifts, and cosmologies.
    Here, we use the most general parameter set using the rescaling formula for redshift and mass
    definitions given in Equation 12.

    Furthermore, the user can choose between results based on
    conventional SO halo finding and an ellipsoidal halo finder.

    Parameters
    -----------------------------------------------------------------------------------------------
    theta: array_like
        Contains the 7 free parameters of Depsali HMF to be fitted
    M: array_like
    Mass in M_{\odot}/h; can be a number or a numpy array.
    z: float
    Redshift
    mdef: str
    The mass definition to which M corresponds. See :doc:`halo_mass` for details.

    Returns
    -----------------------------------------------------------------------------------------------
    dndlnM: array_like
    The halo mass function :math:`dndlnM(M)` in (Mpc/h)^-3.
    """
    cosmo = cosmology.getCurrent()
    sigma_args = defaults.SIGMA_ARGS
    R = peaks.lagrangianR(M)
    sigma = cosmo.sigma(R, z, **sigma_args)


    Delta = mass_so.densityThreshold(z, mdef)
    Delta_vir = mass_so.densityThreshold(z, 'vir')
    x = np.log10(Delta / Delta_vir)

    # A = -0.1362 * x + 0.3292
    # a = 0.4332 * x**2 + 0.2263 * x + 0.7665
    # p = -0.1151 * x**2 + 0.2554 * x + 0.2488
    # theta = [-0.1262, 0.3292, 0.4332, 0.2263, 0.7665, -0.1151, 0.2554, 0.2488]
    # A1, A0, a2, a1, a0, p2, p1, p0 = theta
    # A = A1 * x + A0
    # a = a2* x**2 + a1 * x + a0
    # p = p2 * x**2 + p1 * x + p0

    A, a, p = theta

    tmp = np.array([A, a, p])

    if any(tmp > 1) or any(tmp < 0):
        return - np.inf * M

    else:
        delta_c = peaks.collapseOverdensity(z=z, **deltac_args)

        nu_p = a * delta_c ** 2 / sigma ** 2
        # if np.abs(nu_p).any() >10:
        #     return -M*np.inf
        # else:
        f = 2.0 * A * np.sqrt(nu_p / 2.0 / np.pi) * np.exp(-0.5 * nu_p) * (1.0 + nu_p ** -p)

        q_out = 'dndlnM'
        mfunc = mass_function.convertMassFunction(f, M, z, 'f', q_out)
        return mfunc


def loglike(theta, hmf, selected_redshifts, mass_min):
    """Compute tthe chi2 between the BP HMF and he despali HMF
    Returns
        logike = log likelikelihood """
    # if any(theta > 1) or any(theta < 0):
    #     return - np.inf
    # else :
    cosmo = cosmology.getCurrent()
    z0 = selected_redshifts[0]
    idx_massmin = np.argmin(np.abs(hmf[z0][0] - mass_min))

    M = 10**hmf[z0][0, idx_massmin:] / cosmo.h  # Mass in Msun/h
    # mdef = '200c'
    mdef='vir'
    chi2 = 0
    for z in selected_redshifts:
    # for z in [selected_redshifts[0]]:
        # Get the first zero value at the end of the HMF
        myList = reversed(hmf[z][1])
        idx_massmax = -next((i for i, x in enumerate(myList) if x), None) - 1
        despalihmf = modelDespali16_fit(theta, M, z, mdef)
        # tmp_hmf = hmf[z][1]
        chi2 += np.sum((np.log10(despalihmf[:idx_massmax]) - np.log10(hmf[z][1, idx_massmin:idx_massmax]))** 2 / 1e-7)
    if np.isnan(chi2):
        return -np.inf
    else:
        return -0.5 * chi2


def fitemcee():
    cosmology.setCosmology('planck15')
    hmf, redshifts = load_um_hmf()
    selected_redshifts = redshifts[redshifts > 0]
    selected_redshifts = selected_redshifts[selected_redshifts < 5]

    # theta0 = [-0.1262, 0.3292, 0.4332, 0.2263, 0.7665, -0.1151, 0.2554, 0.2488]
    # theta0 = [0.08122554,  0.56682306,  0.63291531,  0.25406583,  0.67429588,
    #           -0.186546,  0.10609465,  0.15872272]
    theta0 = [0.3292, 0.7665, 0.2488]
    ndim = len(theta0)
    std = np.full(ndim, 0.01)
    nwalkers = 250
    iterations = 100
    p0 = emcee.utils.sample_ball(theta0, std, size=nwalkers)
    # ensure that everything is positive at the begining to avoid points stucked

    mass_min = 11  # minimal mass in log(Msun) for fitting the HMFs

    filename = './fitBP15HMF/samples.h5'
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    # with Pool() as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, loglike, args=[hmf, selected_redshifts, mass_min],
        pool=Pool(),
        backend=backend)
    # sampler.run_mcmc(None, iterations, progress=True)
    sampler.run_mcmc(p0, iterations, progress=True)

    mean_logprob = np.mean(sampler.get_log_prob(), axis=1)
    plt.plot(mean_logprob)
    plt.show()

    # with Pool() as pool:
    #     sampler = emcee.EnsembleSampler(
    #         nwalkers, ndim, loglike, args=[hmf, selected_redshifts, mass_min],
    #         pool=pool, backend=backend)

    #     sampler.run_mcmc(p0, iterations, progress=True)
        # We'll track how the average autocorrelation time estimate changes
        # index = 0
        # autocorr = np.empty(iterations)
        # # This will be useful to testing convergence∏
        # old_tau = np.inf
        # # Now we'll sample for up to iterations steps
        # for sample in sampler.sample(p0, iterations=iterations, progress=True):
            # Only check convergence every 100 steps

            # if sampler.iteration % 100:
            #     continue
            # # Compute the autocorrelation time so far
            # # Using tol=0 means that we'll always get an estimate even
            # # if it isn't trustworthy
            # tau = sampler.get_autocorr_time(tol=0)
            # autocorr[index] = np.mean(tau)
            # index += 1
            # # Check convergence
            # converged = np.all(tau * 50 < sampler.iteration)
            # converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            # if converged:
            #     print('Breaking MCMC because chain converged () at step ' + str(index))
            #     print(
            #         'More than 50 worst autocorr time iterations, and autocorr time varied by less than 1\%')
            #     break
            # old_tau = tau


def analyse():
    # burnin = 0
    # thin = 1
    # samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    # samples = sampler.get_chain()
    # filename = './fitBP15HMF/save190121_ndim3_samples.h5'
    filename = './fitBP15HMF/samples.h5'
    backend = emcee.backends.HDFBackend(filename)
    samples = backend.get_chain()
    logprob = backend.get_log_prob()
    mean_logprob = np.mean(backend.get_log_prob(), axis=1)
    plt.plot(mean_logprob)

    plt.figure()
    for i in range(250):
        plt.plot(logprob[:, i])

    ndim = samples.shape[-1]
    for i in range(ndim):
        plt.figure()
        for j in range(250):
            plt.plot(samples[:, j, i])
    plt.show()

    samp = backend.get_last_sample().coords

    labels = ['A', 'a', 'p']
    flat_samples = backend.get_chain(discard=150, thin=15, flat=True)
    fig = corner.corner(flat_samples, labels=['A', 'a', 'p'])

    ndim = flat_samples.shape[-1]
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(labels[i] + ' = ' + str(mcmc[1]))
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))

    best = np.array([0.35036116, 0.35306773, 0.35574115])

    hmf, redshifts = load_um_hmf()
    mass_min = 11
    selected_redshifts = redshifts[redshifts > 0]
    selected_redshifts = selected_redshifts[selected_redshifts < 5]
    for z in selected_redshifts[::10]:
        plot_model(best, hmf, z, mass_min, mdef='vir')


if __name__ == "__main__":
    fitemcee()
