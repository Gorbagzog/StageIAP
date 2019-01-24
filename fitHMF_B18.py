# -*-coding:Utf-8 -*

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
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import display, Math



cosmology.setCosmology('planck15')

os.environ["OMP_NUM_THREADS"] = "1"


def load_um_hmf():
    """Load the hmf of universe machine code, which is using the peak halo mass and not the actual mass of the halo.
    The simulation is in the Planck 2015 cosmology.

    I am not sure how the mass is computed from the simulation, but if they did like in Behroozi et al 2013 paper,
    the mass is defined with the overdensity criterion of Bryan and Norman 97, and the mass of satellites halo is
    computed as the peak historical mass of the halo progenitor (before merging).

    The halo mass function gives the total halo counts (central plus satellites).
    It gives also in one of the columns the satellite fraction.

    Returns:
    hmf : a dictionary with the redshift as index and giving the HMFs in numpy array.
        Peak halo masses in Msun (or Msun/h ??????)
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
    # M = 10**hmf[z][0] / cosmo.h
    M = 10**hmf[z][0] * cosmo.h
    # M = 10**hmf[z][0]
    despalihmf = modelDespali16_fit(theta, M, z)
    plt.plot(hmf[z][0], despalihmf, label='Modified Despali HMF')
    plt.axvline(mass_min, linestyle='--', label='minimal mass fitted')
    myList = reversed(hmf[z][1])
    idx_massmax = -next((i for i, x in enumerate(myList) if x), None) - 1
    plt.axvline(hmf[z][0, idx_massmax-1], linestyle='--', label='max mass fitted')

    # Default theta from Despali16 paper for mvir:
    def_theta = [0.333, 0.794, 0.247]
    def_despalihmf = modelDespali16_fit(def_theta, M, z)
    plt.plot(hmf[z][0], def_despalihmf, label='Default Despali HMF')
    plt.yscale('log')
    plt.xlabel('peak halo masses in Msun')
    plt.ylabel('number densities in comoving Mpc^-3 dex^-1')
    plt.legend()
    # plt.show()


def plot_all(theta, hmf, redshifts, mass_min):
    cosmo = cosmology.getCurrent()
    plt.figure()
    plt.yscale('log')
    plt.xlabel('$M_{h, \mathrm{max}}$ [$M_\odot / h$]', size=17)
    plt.ylabel('halo mass function [$Mpc^{-3} dex^{-1}$]', size=17)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.xlim(11, 15)
    plt.ylim(10**-6, 0.1)
    for z in redshifts:
        M = 10**hmf[z][0] * cosmo.h
        despalihmf = modelDespali16_fit(theta, M, z)
        plt.plot(hmf[z][0], despalihmf)
        idx_massmin = np.argmin(np.abs(hmf[z][0] - mass_min))
        myList = reversed(hmf[z][1])
        idx_massmax = -next((i for i, x in enumerate(myList) if x), None) - 1
        plt.scatter(hmf[z][0, idx_massmin:idx_massmax], hmf[z][1, idx_massmin:idx_massmax],
                    label='z = {0:.0f}'.format(z))

    plt.legend()
    plt.tight_layout()


def compareHmf(hmf, z):
    """Get the Despali HMF on the HMF from the Bolshoi PLanck simulation.

    Inputs:
        hmf : HMFs dictionnary from load_um_hmf
        z : redshift corresponding to one of the keys of hmf
    """
    M = 10**np.linspace(8, 16)
    mfunc = mass_function.massFunction(
        M, z, mdef = 'vir', model = 'despali16', q_out='dndlnM')

    best_fit = [0.682451203557801, 0.6805870483767835, 0.3530677251620634]
    mfuncfit = modelDespali16_fit(best_fit, M, z,
                       deltac_args = {'corrections': True})
    plt.loglog(M, mfunc)
    plt.loglog(M, mfuncfit)
    plt.show()



def modelDespali16_fit(theta, M, z, deltac_args={'corrections': True}):
    """
    The mass function model of Despali et al 2016.
    This function is updated from the Colossus function to allow to fit paramters on the BP15 HMF.

    The parameters were fit for a number of different mass definitions, redshifts, and cosmologies.
    Here, we use the most general parameter set using the rescaling formula for redshift and mass
    definitions given in Equation 12.

    Furthermore, the user can choose between results based on
    conventional SO halo finding and an ellipsoidal halo finder.

    I modified the code so it fits for only one definition of the halo mass, which is the one used in theouptuts of the BolshoiPlanck simulation

    Parameters
    -----------------------------------------------------------------------------------------------
    theta: array_like
        Contains the 7 free parameters of Depsali HMF to be fitted
    M: array_like
        Mass in M_{\odot}/h; can be a number or a numpy array.
    z: float
        Redshift

    Returns
    -----------------------------------------------------------------------------------------------
    dndlnM: array_like
    The halo mass function :math:`dndlnM(M)` in (Mpc/h)^-3.
    """

    A, a, p = theta

    cosmo = cosmology.getCurrent()
    sigma_args = defaults.SIGMA_ARGS
    R = peaks.lagrangianR(M)
    sigma = cosmo.sigma(R, z, **sigma_args)

    # delta_c = peaks.collapseOverdensity(z=z, **deltac_args)
    delta_c = peaks.collapseOverdensity(z=z, corrections=True)

    nu_p = a * delta_c ** 2 / sigma ** 2
    # if np.abs(nu_p).any() >10:
    #     return -M*np.inf
    # else:
    f = 2.0 * A * np.sqrt(nu_p / 2.0 / np.pi) * np.exp(-0.5 * nu_p) * (1.0 + nu_p ** -p)

    q_out = 'dndlnM'
    mfunc = mass_function.convertMassFunction(f, M, z, 'f', q_out)
    return mfunc


def loglike(theta, hmf, selected_redshifts, mass_min):
    """Compute the chi2 between the BP HMF and he despali HMF
    Returns
        logike = log likelikelihood """

    if any(theta > 1) or any(theta < 0):
        return - np.inf

    cosmo = cosmology.getCurrent()
    z0 = selected_redshifts[0]
    idx_massmin = np.argmin(np.abs(hmf[z0][0] - mass_min))

    # M = 10**hmf[z0][0, idx_massmin:] / cosmo.h  # Mass in Msun/h
    # M = 10**hmf[z0][0, idx_massmin:]  # Mass in Msun/h
    M = 10**hmf[z0][0, idx_massmin:] * cosmo.h  # Mass in Msun/h

    chi2 = 0
    for z in selected_redshifts:
    # for z in [selected_redshifts[0]]:
        # Get the first zero value at the end of the HMF
        myList = reversed(hmf[z][1])
        idx_massmax = -next((i for i, x in enumerate(myList) if x), None) - 1
        despalihmf = modelDespali16_fit(theta, M, z)
        # tmp_hmf = hmf[z][1]
        izero = np.where(hmf[z][1] == 0)
        hmf[z][1, izero ] = 10**(-42)
        # chi2 += np.sum((np.log10(despalihmf[:idx_massmax]) - np.log10(
        #     hmf[z][1, idx_massmin:idx_massmax])) ** 2 / 1e-7)
        chi2 += np.sum((despalihmf[:idx_massmax] - hmf[z][1, idx_massmin:idx_massmax]) ** 2 / 1e-7)
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
    iterations = 700
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
    plt.close('all')
    # burnin = 0
    # thin = 1
    # samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    # samples = sampler.get_chain()
    # filename = './fitBP15HMF/save190121_ndim3_samples'
    # filename = './fitBP15HMF/save190123_ndim3_Msunh_samples'
    filename = './fitBP15HMF/save190123_ndim3_corectbyMsunh_samples'  # Good one to use
    backend = emcee.backends.HDFBackend(filename + '.h5')
    samples = backend.get_chain()
    logprob = backend.get_log_prob()
    mean_logprob = np.mean(backend.get_log_prob(), axis=1)
    # plt.semilogy(-mean_logprob)
    labels = ['A', 'a', 'p']

    plt.figure()
    plt.title('Log-probability')
    plt.xlabel('Iterations')
    plt.ylabel('$\chi^2/2$')
    for i in range(250):
        plt.semilogy(-logprob[:, i])

    flat_samples = backend.get_chain(discard=150, thin=15, flat=True)

    ndim = flat_samples.shape[-1]
    best = np.zeros(ndim)
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        best[i] = mcmc[1]
        q = np.diff(mcmc)
        print(labels[i] + ' = ' + str(mcmc[1]))
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))

    for i in range(ndim):
        plt.figure()
        plt.title('Parameter fit:' + str(best[i]))
        plt.ylabel(labels[i])
        plt.xlabel('Iterations')
        for j in range(250):
            plt.plot(samples[:, j, i])
    # plt.show()

    # samp = backend.get_last_sample().coords

    fig = corner.corner(flat_samples, labels=labels)


    # Best fit when we suppose universe machine gives masses in Msun, but correcting by Msun/h (wrong)
    # best = np.array(
    #     [0.682451203557801, 0.6805870483767835, 0.3530677251620634])

    # Best fit when we suppose universe machine gives masses in Msun/h
    # best = [0.4752772505046221, 0.7532675208782209, 0.3520302943602699]

    # Best fit when we suppose universe machine gives masses in Msun, but correcting by Msun * h
    # best = [0.33090366262139637, 0.8311237426711129, 0.3512604117780071]

    hmf, redshifts = load_um_hmf()
    mass_min = 11
    selected_redshifts = redshifts[redshifts > 0]
    selected_redshifts = selected_redshifts[selected_redshifts < 5]

    z_plots = np.arange(6)
    idxz = z_plots * 0
    for i, z in enumerate(z_plots):
        idxz[i] = np.argmin(np.abs(selected_redshifts - z))

    for z in selected_redshifts[::10]:
        plot_model(best, hmf, z, mass_min, mdef='vir')

    plot_all(best, hmf, selected_redshifts[idxz], mass_min)

    multipage(filename + 'multipage.pdf')


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


if __name__ == "__main__":
    fitemcee()
