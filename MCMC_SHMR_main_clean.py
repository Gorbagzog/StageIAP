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
# import sys
import emcee
from astropy.cosmology import LambdaCDM, Planck15
from astropy import convolution
from astropy.convolution import Gaussian1DKernel
# import scipy.optimize as op
from scipy import signal
import corner
from getdist import plots, MCSamples
import os
import platform
import sys
import datetime
import getconf
from shutil import copyfile
import hmf as hmf_calc
from colossus.cosmology import cosmology
from colossus.lss import mass_function
import Plot_MhaloPeak
from multiprocessing import Pool
import multiprocessing.pool
from joblib import Parallel, delayed
import multiprocessing
from functools import partial
import Plot_results
from cycler import cycler


# os.environ["OMP_NUM_THREADS"] = "1"


def load_smf(params):
    """Load the SMF."""
    smf_name = params['smf_name']
    if smf_name == 'cosmos' or smf_name == 'cosmos_schechter':

        """Load the SMF from Iary Davidzon+17"""
        # redshifts of the Iari SMF
        params['redshifts'] = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
        print('Use the average redshift of the bin from the Iary SMF fit')
        params['redshiftsbin'] = np.array([0.37, 0.668, 0.938, 1.286, 1.735, 2.220, 2.683,
                                          3.271, 3.926, 4.803])
        params['numzbin'] = np.size(params['redshifts']) - 1
        print('numzbin: '+str(params['numzbin']))

        smf = []
        tmp = []
        if smf_name == 'cosmos':
            print('Use the COSMOS 1/Vmax SMF')
            for i in range(params['numzbin']):
                smf.append(np.loadtxt(
                    # Select the SMFs to use : tot, pas or act; D17 or SchechterFixedMs
                    '../Data/Davidzon/Davidzon+17_SMF_v3.0/mf_mass2b_fl5b_tot_Vmax'
                    + str(i) + '.dat')  # Use the 1/Vmax points directly and not the schechter fit on them
                    # '../Data/Davidzon/schechter_fixedMs/mf_mass2b_fl5b_tot_VmaxFit2E'
                    # + str(i) + '.dat')
                )
            print('/!\ /!\ Warning the step in stellar mass may not be good !!! needed for the convoltution in phi true')
            print('/!\ /!\ Warning the step in stellar mass may not be good !!! needed for the convoltution in phi true')
            return None
        elif smf_name == 'cosmos_schechter':
            print('Use the COSMOS Schechter fit SMF')
            for i in range(params['numzbin']):
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


            if params['do_sm_cut']:
                params['SM_cut_min'] = np.log10(6.3 * 10**7 * (1 + params['redshiftsbin'])**2.7)
                print('Use D17 relation for minimal stellar mass: Ms_min='+str(params['SM_cut_min']))
                """ The stellar mass cut is reported further, after the computation of the convolution to avoid border effects"""
            #     print('Do a cut for minimal and maximal stellar masses')
            #     print('Use a cut for high stellar mass at '+str(params['SM_cut_max'])+' $M_{\odot}$')
            #     params['SM_cut_min'] = np.log10(6.3 * 10**7 * (1 + params['redshiftsbin'])**2.7)
            #     print('Use D17 relation for minimal stellar mass: Ms_min='+str(params['SM_cut_min']))
            #     for i in range(params['numzbin']):
            #         smf[i] = smf[i][np.where(
            #             np.logical_and(
            #                 smf[i][:,0] > params['SM_cut_min'][i],
            #                 smf[i][:,0] < params['SM_cut_max'][i]
            #             )
            #         ), :][0]


        """Adapt SMF to match the Planck Cosmology"""
        # Davidzon+17 SMF cosmo : (flat LCDM)
        # Om = 0.3, Ol = 0.7, h=0.7
        D17_Cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
        print('Rescale the SMF to Planck15 cosmology')
        for i in range(params['numzbin']):
            # Correction of the comoving Volume :
            VmaxD17 = D17_Cosmo.comoving_volume(params['redshifts'][i+1]) - D17_Cosmo.comoving_volume(params['redshifts'][i])
            VmaxP15 = Planck15.comoving_volume(params['redshifts'][i+1]) - Planck15.comoving_volume(params['redshifts'][i])
            """In the case where we use the Vmax points and not the VmaxFit, the errors bars are relative and
            are not the absolute uncertainty as in the Vmax Fit, so we don't rescale the error bars"""
            smf[i][:, 1] = smf[i][:, 1] + np.log10(VmaxD17/VmaxP15)
            # Correction of the measured stellar mass
            # Equivalent to multiply by (Planck15.H0/D17_Cosmo.H0)**-2
            smf[i][:, 0] = smf[i][:, 0] - 2 * np.log10(Planck15.H0/D17_Cosmo.H0)
            # Correct for the dependance on the luminosity distance
            DL_D17 = D17_Cosmo.luminosity_distance(params['redshiftsbin'][i])
            DL_Planck = Planck15.luminosity_distance(params['redshiftsbin'][i])
            smf[i][:, 0] = smf[i][:, 0] + 2*np.log10(DL_Planck/DL_D17)

    if smf_name == 'candels':
        print('Use the Candels SMF')
        print('/!\ /!\ Warning the step in stellar mass may not be good !!! needed for the convoltution in phi true')
        return None
        """Load the SMF from Candels Grazian"""
        # Code is copied from IaryDavidzonSMF.py as of 12 june
        # redshifts of the Candels+15 data
        params['redshifts'] = np.array([3.5, 4.5, 5.5, 6.5, 7.5])
        params['redshiftsbin'] = (params['redshifts'][1:]+params['redshifts'][:-1])/2
        params['numzbin'] = np.size(params['redshifts'])-1
        print('numzbin: '+str(params['numzbin']))
        smf = []
        for i in range(params['numzbin']):
            smf.append(np.loadtxt(
                # Select the SMFs to use : JEWELS or v2
                # '../Data/Candels/grazian15_68CL_z' + str(i+4) + '_JEWELS.txt')
                '../Data/Candels/grazian15_68CL_v2_z' + str(i+4) + '.txt')
            )
        """Adapt SMF to match the Bolshoi-Planck Cosmology"""
        # Bolshoi-Planck cosmo : (flat LCMD)
        # Om = 0.3089, Ol = 0.6911, Ob = 0.0486, h = 0.6774, s8 = 0.8159, ns = 0.9667
        BP_Cosmo = LambdaCDM(H0=67.74, Om0=0.3089, Ode0=0.6911)
        # CANDELS+17 SMF cosmo : (flat LCDM) (same as Davidzon17_COSMO)
        # Om = 0.3, Ol = 0.7, h=0.7
        C17_Cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
        for i in range(params['numzbin']):
            # Correction of the comoving Volume :
            VmaxD17 = C17_Cosmo.comoving_volume(params['redshifts'][i+1]) - C17_Cosmo.comoving_volume(params['redshifts'][i])
            VmaxBP = BP_Cosmo.comoving_volume(params['redshifts'][i+1]) - BP_Cosmo.comoving_volume(params['redshifts'][i])
            # Add the log, equivalent to multiply by VmaxD17/VmaxBP
            smf[i][:, 1] = smf[i][:, 1] + np.log10(VmaxD17/VmaxBP)
            smf[i][:, 2] = smf[i][:, 2] + np.log10(VmaxD17/VmaxBP)
            smf[i][:, 3] = smf[i][:, 3] + np.log10(VmaxD17/VmaxBP)
            # Correction of the measured stellar mass
            # Equivalent to multiply by (BP_Cosmo.H0/C17_Cosmo.H0)**-2
            smf[i][:, 0] = smf[i][:, 0] - 2 * np.log10(BP_Cosmo.H0/C17_Cosmo.H0)
            """/!\ problem with error bars in candels SMF !!"""
            # plt.errorbar(smf[idx_z][:, 0], smf[idx_z][:, 1],
            #   yerr=[smf[idx_z][:,1]-smf[idx_z][:, 2], smf[idx_z][:, 3]- smf[idx_z][:,1]])
    return smf


def load_hmf(params):
    """Load the HMF."""
    hmf_name = params['hmf_name']
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
        # global redshift_id_selec
        redshift_id_selec = np.empty(params['numzbin'])
        params['redshift_id_selec'] = redshift_id_selec
        for i in range(params['numzbin']):
            redshift_id_selec[i] = np.argmin(
                np.abs(redshift_haloes - (params['redshifts'][i] + params['redshifts'][i + 1]) / 2))
        redshift_id_selec = redshift_id_selec.astype(int)
        print('Redshifts of Iari SMFs : ' + str((params['redshifts'][:-1] + params['redshifts'][1:]) / 2))
        print('Closest redshifts for Bolshoi HMFs : '
            + str(redshift_haloes[redshift_id_selec]))
        for i in redshift_id_selec:
            hmf.append(hmf_bolshoi[i])
        if hmf_name == 'bolshoi_tot':
            # In the bolshoi tot case, change to the correct bolshoi HMF
            print('Use Bolshoi total HMF')
            for i in range(params['numzbin']):
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

    if hmf_name == 'hmf_module_tinker':
        """Use the python module hmf from HFMCalc of Murray+13
        parameters : Planck+15 cosmology by default, with  'delta_h': 200.0, 'delta_wrt': 'mean'"""
        # print hmf parameters with h.param_values
        print('Use Tinker+08 HMF in PLanck cosmo from hmf module')
        h = hmf_calc.MassFunction()
        h.update(Mmin=8)
        h.update(Mmax=16)
        print(h.parameter_values)
        redshift_haloes = params['redshiftsbin']
        for i in range(params['numzbin']):
            h.update(z=params['redshiftsbin'][i])
            hmf.append(np.transpose(np.array([np.log10(h.m / h.cosmo_model.h),
                       np.log10(h.dndlog10m * (h.cosmo_model.h)**3)])))  # Replace the h implicit in the HMF

    if hmf_name == 'hmf_module_behroozi':
        """Use the python module for Behrrozi HMF"""
        print('Use Behroozi HMF in Planck cosmo from hmf module')
        h = hmf_calc.MassFunction()
        h.update(Mmin=8)
        h.update(Mmax=16)
        print(h.parameter_values)
        redshift_haloes = params['redshiftsbin']
        for i in range(params['numzbin']):
            h.update(z=params['redshiftsbin'][i])
            hmf.append(np.transpose(np.array([np.log10(h.m / h.cosmo_model.h),
                       np.log10(h.dndlog10m * (h.cosmo_model.h)**3)])))  # Replace the h implicit in the HMF

    if hmf_name == 'despali16' or hmf_name == 'tinker08' or hmf_name == 'watson13' or hmf_name == 'bocquet16' or hmf_name == 'bhattacharya11':
        """Use the Colossus module for the HMF"""
        print('Use '+hmf_name+' HMF in Planck15 cosmo from Colossus module')
        if hmf_name == 'watson13' or hmf_name =='bhattacharya11':
            print(hmf_name)
            mdef='fof'
        else:
            mdef = '200m'
        print('Use '+mdef+' for the SO defintion.')
        cosmo = cosmology.setCosmology('planck15')
        redshift_haloes = params['redshiftsbin']
        M = 10**np.arange(8.0, 20, 0.01) # Mass in Msun / h
        for i in range(params['numzbin']):
            hmf.append(
                np.transpose(
                    np.array(
                        [np.log10(M / cosmo.h),
                         np.log10(mass_function.massFunction(M, redshift_haloes[i], mdef = mdef, model =hmf_name, q_out = 'dndlnM') * np.log(10) * cosmo.h**3
                            ## Mass functions are in h^3 Mpc^-3, and need to multiply by ln(10) to have dndlog10m
                            )]
                        )
                    )
                )
    return hmf

"""Function definitions for computation of the theoretical SMF phi_true"""


def logMh(logMs, M1, Ms0, beta, delta, gamma):
    """SM-HM relation"""
    return M1 + beta*(logMs - Ms0) + (10 ** (delta * (logMs - Ms0))) / (1 + (10 ** (-gamma * (logMs - Ms0)))) - 0.5


def log_phi_direct(logMs, idx_z, M1, Ms0, beta, delta, gamma):
    # print(delta)
    """"SMF obtained from the SM-HM relation and the HMF"""
    epsilon = 0.0001
    log_Mh1 = logMh(logMs, M1, Ms0, beta, delta, gamma)
    log_Mh2 = logMh(logMs + epsilon, M1, Ms0, beta, delta, gamma)
    # if np.any(log_Mh2 > hmf[idx_z][-1, 0]) or np.any(log_Mh1 < hmf[idx_z][0, 0]):
    #     print('above hmf')
    #     return log_Mh1 * 0. + 10
    #     return log_Mh1 * 0. - np.inf
    # else :
    # if True:
    # Select the index of the HMF corresponding to the halo masses
    index_Mh = np.argmin(
        np.abs(
            np.tile(hmf[idx_z][:, 0], (len(log_Mh1), 1)) -
            np.transpose(np.tile(log_Mh1, (len(hmf[idx_z][:, 0]), 1)))
        ), axis=1)
    # if np.any(hmf[idx_z][index_Mh, 1] < -100):
    #     print('HMF not defined')
    #     return log_Mh1 * 0. + 1
    # else:
    log_phidirect = hmf[idx_z][index_Mh, 1] + np.log10((log_Mh2 - log_Mh1)/epsilon)
    # if np.any(log_Mh2 > hmf[idx_z][-1, 0]):
    #     print('above hmf')
    #     log_phidirect = log_phidirect[log_Mh2 > hmf[idx_z][-1, 0]] = -np.inf
    return log_phidirect

    # log_phidirect[log_Mh1 > hmf[idx_z][-1, 0]] = 10**6 # Do not use points with halo masses not defined in the HMF
    # log_phidirect[log_Mh1 < hmf[idx_z][0, 0]] = 10**6


def gauss(y, ksi):
    return 1. / (ksi * np.sqrt(2 * np.pi)) * np.exp(- 1/2 * (y / ksi)**2)


def phi_true(logMs, idx_z, M1, Ms0, beta, delta, gamma, ksi):
    """Use convolution defined in Behroozi et al 2010"""
    log_phidirect = log_phi_direct(logMs, idx_z, M1, Ms0, beta, delta, gamma)
    # if params['smf_name'] == 'cosmos_schechter':
    dx = np.mean(logMs[1:] - logMs[:-1])
    # else:
    #     print('Warning, the step between two mass bins in not defined in this case.')
    u = np.arange(0, 10*ksi, dx)
    x = np.concatenate((-np.flip(u, 0)[:-1], u)) # kepp the zero at the center of the array
    gaussian = 1. / (ksi * np.sqrt(2 * np.pi)) * np.exp(- 1/2 * (x / ksi)**2) * dx
    # return np.log10(signal.convolve(10**log_phi_dir, gaussian, mode='same'))

    """Make an extension of the array on the left side to avoid convolution border effects"""
    n_ext = x.shape[0] // 2
    log_phi_dir_extend = np.concatenate((np.full(n_ext, log_phidirect[0]), log_phidirect))
    """Make a zero padding on the right side of the array"""
    log_phi_dir_extend = np.concatenate((log_phi_dir_extend, np.full(n_ext*2, -np.inf)))
    phi_dir_extend = 10**log_phi_dir_extend
    phi_true_extend = signal.convolve(phi_dir_extend, gaussian, mode='same')
    phi_true = phi_true_extend[n_ext: -n_ext*2 or None]   # Put None in case n_ext is 0 (avoid empty list)
    return phi_true
    # return np.log10(signal.convolve(10**log_phi_dir_extend, gaussian, mode='same')[x.shape[0] // 2:])
    # if any(np.isnan(phi_true_extend)):
    #     print( M1, Ms0, beta, delta, gamma, ksi)
    #     print(phi_true_extend[x.shape[0] // 2:])
    #     return
    # else:
    # print(phi_true.shape)
    # return phi_true
    # gaussian = Gaussian1DKernel(stddev=ksi/dx)
    # return np.log10(convolution.convolve(10**log_phi_dir, gaussian, boundary='extend'))
    # phitrue = log_phidirect * 0.
    # for i in range(phitrue.size):
    #     for j in range(phitrue.size -1):
    #         phitrue[i] = phitrue[i] + 10**log_phidirect[j] * gauss(logMs[j] - logMs[i], ksi) * (logMs[j+1] - logMs[j])
    # return phitrue


def chi2(idx_z, M1, Ms0, beta, delta, gamma, ksi, subsampling_step, sub_start):
    """"return the chi**2 between the observed and the expected SMF."""
    select = np.where(smf[idx_z][:, 1] > -40)[0]  # select points where the smf is defined
    # We choose to limit the fit only for abundances higher than 10**-7
    logMs = smf[idx_z][select[:], 0]
    pred = phi_true(logMs, idx_z, M1, Ms0, beta, delta, gamma, ksi)
    # if params['smf_name'] == ('cosmos' or 'candels'):
    #     print('Check that there may be an issue with the SMF cut')
        # chi2 = np.sum(
        #         # When using the Vmax directly (give the error bars directly, in a linear scale)
        #         # Need to use a linear scale to compute the chi2 with the right uncertainty
        #         ((pred - 10**smf[idx_z][select, 1]) / (
        #             10**(smf[idx_z][select, 2] + smf[idx_z][select, 1]) - 10**smf[idx_z][select, 1]))**2
        #     )
    # elif params['smf_name'] == 'cosmos_schechter':
    if params['do_sm_cut']:
        sm_select = np.where(np.logical_and(
                    logMs > params['SM_cut_min'][idx_z],
                    logMs < params['SM_cut_max'][idx_z]
                )
            )
    else:
        sm_select = True

    chi2 = np.sum(
        ((np.log10(pred[sm_select]) - smf[idx_z][select, 1][sm_select]) / ((smf[idx_z][select, 2][sm_select] + smf[idx_z][select, 2][sm_select])/2) )[sub_start::subsampling_step]**2)
    return chi2


# def var_std(x_est , x_true, sigm, sigp):
#     """Compute a varying standard deviation to have a varying gaussian for the likelihood with non symmetric errors"""
#     sig0 = 2 * sigm * sigp / (sigm + sigp)
#     sig1 = (sigm -  sigp) / (sigp + sigm)
#     return sig0 + sig1 * (x_true - x_est)


def loglike(theta, idx_z, minbound, maxbound, subsampling_step, sub_start):
    """return the loglikelihood"""
    if all(theta > minbound[idx_z]) and all(theta < maxbound[idx_z]):
        M1, Ms0, beta, delta, gamma, ksi = theta[:]
        return -chi2(idx_z, M1, Ms0, beta, delta, gamma, ksi, subsampling_step, sub_start)/2
    else:
        return -np.inf


""" Run MCMC """


def get_platform():
    """Returns the save path and the number of processes for parallelization
    depending on the machin teh script is ran."""
    if platform.uname()[1] == 'imac-de-louis':
        numprocess = 4
        print('Run on ' + platform.uname()[1] + ' with ' + str(numprocess) + ' process.')
        return '../', numprocess
    elif platform.uname()[1] == 'glx-calcul3':
        numprocess = 4
        print('Run on ' + platform.uname()[1] + ' with ' + str(numprocess) + ' process.')
        return '/data/glx-calcul3/data1/llegrand/StageIAP/', numprocess
    elif platform.uname()[1] == 'glx-calcul1':
        numprocess = 3
        print('Run on ' + platform.uname()[1] + ' with ' + str(numprocess) + ' process.')
        return '/data/glx-calcul3/data1/llegrand/StageIAP/', numprocess
    elif platform.uname()[1] == 'MacBook-Pro-de-Louis.local' or platform.uname()[1] == 'MBP-de-Louis':
        numprocess = 1
        print('Run on ' + platform.uname()[1] + ' with ' + str(numprocess) + ' process.')
        return '../', numprocess
    elif platform.uname()[1] == 'mic-llegrand.ias.u-psud.fr':
        numprocess = 4
        print('Run on ' + platform.uname()[1] + ' with ' + str(numprocess) + ' process.')
        return '../', numprocess
    else:
        print('Unknown machine, please update the save path')
        sys.exit("Unknown machine, please update the save path")


def load_params(paramfile):
    """Load parameters and config"""
    config = getconf.ConfigGetter('getconf', [paramfile])
    params = {}
    params['save_path'], params['nthreads'] = get_platform()
    params['smf_name'] = config.getstr('Mass_functions.SMF')
    params['do_sm_cut'] = config.getbool('Mass_functions.do_sm_cut')
    params['SM_cut_max'] = np.array(config.getlist('Mass_functions.SM_cut')).astype('float')
    params['SMF_subsampling'] = config.getbool('Mass_functions.SMF_subsampling')
    params['subsampling_step'] = config.getint('Mass_functions.subsampling_step')
    params['subsampling_start'] = config.getint('Mass_functions.subsampling_start')
    params['hmf_name'] = config.getstr('Mass_functions.HMF')
    params['iterations'] = config.getint('MCMC_run_parameters.iterations')
    # params['burn'] = config.getint('MCMC_run_parameters.burn')
    params['minboundfile'] = config.getstr('Values.minbound')
    params['maxboundfile'] = config.getstr('Values.maxbound')
    params['minbound'] = np.loadtxt(params['minboundfile'], delimiter=',')
    params['maxbound'] = np.loadtxt(params['maxboundfile'], delimiter=',')
    params['starting_point_file'] = config.getstr('Values.starting_point')
    params['starting_point'] = np.loadtxt(params['starting_point_file'], delimiter=',')
    params['noksi'] = config.getbool('MCMC_run_parameters.noksi')
    # np.array(config.getlist('MCMC_run_parameters.starting_point')).astype('float')
    params['std'] = np.array(config.getlist('MCMC_run_parameters.std')).astype('float')
    # params['nthreads'] = config.getint('MCMC_run_parameters.nthreads')
    params['nwalkers'] = config.getint('MCMC_run_parameters.nwalkers')
    params['selected_redshifts'] = np.array(config.getlist('MCMC_run_parameters.redshifts')).astype('int')
    params['progress'] = config.getbool('MCMC_run_parameters.progress')
    return params


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def runMCMC_allZ(paramfile):
    """Main function to run all MCMC on all zbins based on the param file."""
    global params
    params = load_params(paramfile)
    # Create save direcory
    now = datetime.datetime.now()
    dateName = "MCMC_"+str(now.year)+'-'+str(now.month)+'-'+str(now.day)+'T'+str(now.hour)+'-'+str(now.minute)
    directory = params['save_path'] + dateName
    print('Save direcory : ' + directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(directory+'/Chain')
        os.makedirs(directory+'/Plots')
        # os.makedirs(directory+'/Plots/MhaloPeak')
        os.makedirs(directory+'/Results')
        print('Created new directory')
    orig_stdout = sys.stdout
    f = open(directory+'/out.txt', 'w')
    sys.stdout.flush()
    sys.stdout = f

    global smf
    global hmf
    smf = load_smf(params)
    hmf = load_hmf(params)

    # Copy parameter files in the save directory
    copyfile(params['starting_point_file'], directory + '/' + params['starting_point_file'])
    copyfile(paramfile, directory + '/' + paramfile)
    copyfile(params['minboundfile'], directory + '/' + params['minboundfile'])
    copyfile(params['minboundfile'], directory + '/' + params['minboundfile'])
    print(params['selected_redshifts'])
    # Make the file for autocorr, burn in length and thin length
    with open(directory + "/Chain/Autocorr.txt", "a") as myfile:
        myfile.write("# Mean autocorrelation, Burn in length,  Thin length \n")
    with open(directory + "/Results.txt", "a") as myfile:
        myfile.write(r'Print mean value, 68% lower and 68% upper limits' + '\n')
        myfile.write('idx_z, M1, Ms0, beta, delta, gamma, ksi \n')
    # run all MCMC for all zbins
    # for idx_z in range(params['numzbin']):
    # for idx_z in params['selected_redshifts']:
    #     print(idx_z)
    #     print('Starting MCMC run for idx_z =' + str(idx_z))
    #     print('Min bound: ' + str(params['minbound'][idx_z]))
    #     print('Max bound: ' + str(params['maxbound'][idx_z]))
    #     runMCMC(directory, idx_z, params)

    # Run all redshift at the same time
    #Pool().map(partial(runMCMC, directory=directory, params=params), params['selected_redshifts'])
    print("Creating xx (non-daemon) workers and jobs in main process.")
    pool = MyPool()
    pool.map(partial(runMCMC, directory=directory, params=params),
        params['selected_redshifts'])

    # The following is not really needed, since the (daemon) workers of the
    # child's pool are killed when the child is terminated, but it's good
    # practice to cleanup after ourselves anyway.
    pool.close()
    pool.join()
    # return result

    # Plot all SHMR on one graph
    plotSHMR_delta(directory, load=False, selected_redshifts=params['selected_redshifts'])
    # Plot the MhaloPeak graph
    plt.clf()
    plt.figure(figsize=(10, 5))
    Plot_MhaloPeak.plotLiterrature()
    Plot_MhaloPeak.plotFit(directory, params['smf_name'], params['hmf_name'])
    Plot_MhaloPeak.savePlot(directory)
    plt.clf()

    # Plot evolution of parameters with redshift
    Plot_results.plot_all(directory)

    sys.stdout = orig_stdout
    f.close()


def runMCMC(idx_z, directory, params):
    minbound, maxbound, starting_point, std, iterations, nwalkers = params['minbound'], params['maxbound'], params['starting_point'], params['std'], params['iterations'], params['nwalkers']
    p0 = emcee.utils.sample_ball(starting_point[idx_z], std, size=nwalkers)
    p0 = np.abs(p0)  # ensure that everything is positive at the begining to avoid points stucked
    ndim = len(starting_point[idx_z])
    """SMF subsampling"""
    if params['SMF_subsampling']:
        subsampling_step = params['subsampling_step']
        sub_start = params['subsampling_start']
    else:
        subsampling_step = None
        sub_start = None
    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = directory+'/Chain/samples_'+str(idx_z)+'.h5'
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    print('Using backend to save the chain to '+filename)
    with Pool(processes=2) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike,
                    args=[idx_z, minbound, maxbound, subsampling_step, sub_start], pool=pool,
                    backend=backend)
        print("idx_z = " + str (idx_z))
        print("ndim = " + str(ndim))
        print("start = " + str(starting_point[idx_z]))
        print("std = " + str(std))
        print("iterations = " + str(iterations))
        print(sampler.backend)

        # Old style run
        # start_time = time.time()
        # sampler.run_mcmc(p0, iterations)
        # elapsed_time = time.time() - start_time
        # print('Time elapsed: ' + str(elapsed_time))
        # print('Acceptance fraction:')
        # print(sampler.acceptance_fraction)


        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(iterations)
        # This will be useful to testing convergence∏
        old_tau = np.inf
        # Now we'll sample for up to iterations steps
        for sample in sampler.sample(p0, iterations=iterations, progress=True):
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue
            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1
            # Check convergence
            converged = np.all(tau * 50 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                print('Breaking MCMC because chain converged () at step ' + str(index))
                print('More than 50 worst autocorr time iterations, and autocorr time varied by less than 1\%')
                break
            old_tau = tau
        # print('Autocorrelation time:')
        # print(sampler.get_autocorr_time(tol=0, discard=burn))

        plt.close('all')
        plotAutocorr(directory, idx_z, autocorr, index)

        # Clean samples
        tau = sampler.get_autocorr_time(tol=0)
        burnin = int(2*np.max(tau))
        print("Burnin "+str(burnin))
        thin = int(0.5*np.min(tau))
        with open(directory + "/Chain/Autocorr.txt", "a") as myfile:
            myfile.write(str(np.mean(tau)) + "  " + str(burnin) + "  " + str(thin) + "\n")

        samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)

        # Plot all relevant figures
        plt.close('all')
        plotchain(directory, samples, idx_z, params)
        plt.close('all')
        plotdist(directory, samples, idx_z, params)
        plt.close('all')
        plotSMF(directory, samples, smf, hmf, idx_z, params,  subsampling_step, sub_start)
        plt.close('all')
        plotSMHM(directory, samples, smf, idx_z)
        plt.close('all')
        plotSHMR(directory, samples, smf, idx_z)
        plt.close('all')
        plot_Mhpeak(directory, samples, idx_z, params)
        plt.close('all')
        save_results(directory, samples, idx_z, params)


def save_results(directory, samples, idx_z, params):
    names = ['$M_{1}$', '$M_{s,0}$', '$\\beta$', '$\delta$', '$\gamma$', r'$\xi$']
    ranges = dict(zip(names, np.transpose(np.array([params['minbound'][idx_z], params['maxbound'][idx_z]]))))
    samples = MCSamples(samples=samples, names=names, ranges=ranges)
    res = samples.getTable()
    res.write(directory+"/Results/Chain_ksi_z" + str(idx_z) + ".txt")

    margeStats = samples.getMargeStats()
    results = np.empty(3 * len(names) +1 )
    with open(directory + "/Results.txt", "a") as myfile:
        for i in range(len(names)):
            results[0] = idx_z
            results[3*i + 1] = margeStats.names[i].mean
            results[3*i + 2] = margeStats.names[i].limits[0].lower
            results[3*i + 3] = margeStats.names[i].limits[0].upper
        # myfile.write(str(results) + "\n")
        np.savetxt(myfile, results.reshape(1, results.shape[0]))


def MhPeak(samples, idx_z, Ms_max):
    chainsize = np.shape(samples)[0]
    logMs = np.linspace(8, Ms_max, num=300)
    Mhalopeak = np.zeros(chainsize)
    for i in range(chainsize):
        logmhalo = logMh(logMs, samples[i, 0], samples[i, 1], samples[i, 2], samples[i, 3], samples[i, 4])
        Mhalopeak_idx = np.argmax(logMs - logmhalo)
        Mhalopeak[i] = logmhalo[Mhalopeak_idx]
    return Mhalopeak


def readAndAnalyseBin(directory, idx_z):
    """Read previously computed chains and make the analysis."""
    paramfile = directory + '/MCMC_param.ini'
    global params
    params = load_params(paramfile)
    global smf
    global hmf
    smf = load_smf(params)
    hmf = load_hmf(params)
    """SMF subsampling"""
    if params['SMF_subsampling']:
        subsampling_step = params['subsampling_step']
        sub_start = params['subsampling_start']
    else:
        subsampling_step = None
        sub_start = None
    print('Start loading and plotting '+str(idx_z))
    filename = directory+'/Chain/samples_'+str(idx_z)+'.h5'
    reader = emcee.backends.HDFBackend(filename, read_only=True)
    tau = reader.get_autocorr_time(tol=0)
    burnin = int(2*np.nanmax(tau))
    thin = int(0.5*np.nanmin(tau))
    samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
    # Plot all relevant figures
    # plt.close('all')
    # plotchain(directory, samples, idx_z, params)
    # plt.close('all')
    # plotdist(directory, samples, idx_z, params)
    # plt.close('all')
    # plotSMF(directory, samples, smf, hmf, idx_z, params, subsampling_step, sub_start)
    # plt.close('all')
    # plotSMHM(directory, samples, smf, idx_z)
    # plt.close('all')
    # plotSHMR(directory, samples, smf, idx_z)
    # plt.close('all')
    # plot_Mhpeak(directory, samples, idx_z, params)
    # plt.close('all')
    # with open(directory + "/Results.txt", "a") as myfile:
    #     myfile.write(r'Print mean value, 68% lower and 68% upper limits')
    #     myfile.write('M1, Ms0, beta, delta, gamma, ksi')
    save_results(directory, samples, idx_z, params)


"""Plots"""


def plotAutocorr(directory, idx_z, autocorr, index):
    n = 100*np.arange(1, index+1)
    y = autocorr[:index]
    plt.plot(n, n / 50.0, "--k")
    plt.plot(n, y)
    # plt.xlim(0, n.max())
    # plt.ylim(0, y.max() + 0.1*(y.max() - y.min()))
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$");
    plt.savefig(directory+'/Plots/TestConvergence_'+str(idx_z)+'.pdf')


def plotSMF(directory, samples, smf, hmf, idx_z, params, subsampling_step, sub_start):
    if params['do_sm_cut']:
        select = np.where(np.logical_and(
                np.logical_and(
                    smf[idx_z][:, 0] > params['SM_cut_min'][idx_z],
                    smf[idx_z][:, 0] < params['SM_cut_max'][idx_z]
                ),
                smf[idx_z][:, 1] > -40)
            )[0]
    else:
        select = np.where(smf[idx_z][:, 1] > -40)[0]
    cut_smf = np.array([smf[idx_z][select, 0][sub_start::subsampling_step], smf[idx_z][select, 1][sub_start::subsampling_step],
        smf[idx_z][select, 2][sub_start::subsampling_step], smf[idx_z][select, 3][sub_start::subsampling_step]])
    plt.errorbar(smf[idx_z][select, 0], smf[idx_z][select, 1],
        yerr=[smf[idx_z][select, 3], smf[idx_z][select, 2]], fmt='o', label='all points')
    plt.errorbar(cut_smf[0], cut_smf[1],
        yerr=[cut_smf[3], cut_smf[2]], fmt='o', label='fitted points')
    logMs = np.linspace(smf[idx_z][select[0], 0], smf[idx_z][select[-1], 0], num=100)
    for M1, Ms0, beta, delta, gamma, ksi in samples[np.random.randint(len(samples), size=100)]:
        logphi = np.log10(phi_true(logMs, idx_z, M1, Ms0, beta, delta, gamma, ksi))
        plt.plot(logMs, logphi, color="k", alpha=0.1)
    plt.xlabel('$\mathrm{log}_{10}(M_* / M_{\odot})$')
    plt.ylabel('$\mathrm{log}_{10}(\phi)$')
    plt.legend()
    plt.savefig(directory+'/Plots/SMF_'+ str(idx_z) + '.pdf')
    # plt.close()
    plt.figure()
    plt.errorbar(smf[idx_z][select, 0][sub_start::subsampling_step], smf[idx_z][select, 1][sub_start::subsampling_step],
        yerr=[smf[idx_z][select, 3][sub_start::subsampling_step], smf[idx_z][select, 2][sub_start::subsampling_step]], fmt='o')
    for M1, Ms0, beta, delta, gamma, ksi in samples[np.random.randint(len(samples), size=100)]:
        logphi = np.log10(phi_true(logMs, idx_z, M1, Ms0, beta, delta, gamma, ksi))
        plt.plot(logMs, logphi, color="k", alpha=0.1)
    plt.xlim(9, 12)
    plt.ylim(-6, -2)
    plt.xlabel('$\mathrm{log}_{10}(M_* / M_{\odot})$')
    plt.ylabel('$\mathrm{log}_{10}(\phi)$')
    plt.savefig(directory+'/Plots/SMF_zoom'+ str(idx_z) + '.pdf')


def plotSMHM(directory, samples, smf, idx_z):
    select = np.where(smf[idx_z][:, 1] > -40)[0]
    logMs = np.linspace(smf[idx_z][select[0], 0], smf[idx_z][select[-1], 0], num=50)
    for M1, Ms0, beta, delta, gamma, ksi in samples[np.random.randint(len(samples), size=100)]:
        logmhalo = logMh(logMs, M1, Ms0, beta, delta, gamma)
        plt.plot(logmhalo, logMs-logmhalo, color="k", alpha=0.1)
    plt.xlim(10, 16)
    plt.ylim(-4, -0.5)
    plt.xlabel('$M_{\odot} / \mathrm{log}_{10}(M_{\mathrm{h}})$')
    plt.ylabel('$\mathrm{log}_{10}(M_{\mathrm{h}} / M_{*})$')
    plt.savefig(directory+'/Plots/SMHM_ksi'+ str(idx_z) + "_niter=" + '.pdf')


def plotSHMR(directory, samples, smf, idx_z):
    select = np.where(smf[idx_z][:, 1] > -40)[0]
    logMs = np.linspace(smf[idx_z][select[0], 0], smf[idx_z][select[-1], 0], num=50)
    for M1, Ms0, beta, delta, gamma, ksi in samples[np.random.randint(len(samples), size=100)]:
        logmhalo = logMh(logMs, M1, Ms0, beta, delta, gamma)
        plt.plot(logMs, logmhalo, color="k", alpha=0.1)
    plt.xlabel('$\mathrm{log}_{10}(M_{*} / M_{\odot})$')
    plt.ylabel('$\mathrm{log}_{10}(M_{\mathrm{h}} / M_{\odot})$')
    plt.savefig(directory+'/Plots/SHMR'+ str(idx_z) + "_niter=" + '.pdf')


def plotchain(directory, samples, idx_z, params):
    figname = directory + "/Plots/Triangle_z" + str(idx_z)
    fig = corner.corner(
        samples, labels=['$M_{1}$', '$M_{*,0}$', '$\\beta$', '$\delta$', '$\gamma$', r'$\xi$'])
    fig.savefig(figname + ".pdf")
    plt.close('all')


def plotdist(directory, samples, idx_z, params):
    figname = directory + "/Plots/GetDist_z" + str(idx_z)
    names = ['$M_{1}$', '$M_{s,0}$', '$\\beta$', '$\delta$', '$\gamma$', r'$\xi$']
    samples = MCSamples(samples=samples, names=names,
        ranges = dict(zip(names, np.transpose(np.array([params['minbound'][idx_z],
        params['maxbound'][idx_z]])))))
    g = plots.getSubplotPlotter()
    g.triangle_plot(samples, filled=True)
    g.export(figname + '.pdf' )
    plt.close('all')


def plot_Mhpeak(directory, samples, idx_z, params):
    mhpeak = MhPeak(samples, idx_z, params['SM_cut_max'][idx_z])
    med_mhpeak = np.median(mhpeak)
    std_mhpeak = np.std(mhpeak)
    with open(directory + "/MhaloPeak.txt", "a") as myfile:
        myfile.write(str(idx_z) + "  " + str(med_mhpeak) + "  " + str(std_mhpeak) + "\n")
    plt.figure()
    plt.hist(mhpeak, bins=100)
    plt.axvline(med_mhpeak, color='orange')
    plt.title(str(params['redshifts'][idx_z]) +'<z<' + str(params['redshifts'][idx_z+1]) + ', MhPeak = ' + str(med_mhpeak) + '+/-' + str(std_mhpeak))
    plt.savefig(directory+'/Plots/MhPeak_z' + str(idx_z) + '.pdf')


def save_load_smhm(directory, Ms_min, Ms_max, numpoints, load, selected_redshifts):
    """Compute the median, average and scatter of Mh for a given Ms."""
    logMs = np.empty([params['numzbin'], numpoints])
    logMhbins = np.linspace(11.5, 14, num=numpoints)
    av_logMh = np.empty([params['numzbin'], numpoints])
    med_logMh = np.empty([params['numzbin'], numpoints])
    conf_min_logMh = np.empty([params['numzbin'], numpoints])
    conf_max_logMh = np.empty([params['numzbin'], numpoints])
    if load is False:
        print('Computing arrays')
        for idx_z in selected_redshifts:
            logMs[idx_z] = np.linspace(Ms_min[idx_z], Ms_max[idx_z], num=numpoints)
            filename = directory+'/Chain/samples_'+str(idx_z)+'.h5'
            reader = emcee.backends.HDFBackend(filename, read_only=True)
            tau = reader.get_autocorr_time(tol=0)
            burnin = int(2*np.max(tau))
            thin = int(0.5*np.min(tau))
            samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
            print('Chain loaded for idx_z = '+str(idx_z))
            nsimu = samples.shape[0]
            print('Number of samples:'+str(nsimu))
            logmhalo = np.zeros([nsimu, numpoints])
            for idx_simu in range(nsimu):
                # M1, Ms0, beta, delta, gamma, ksi = samples[idx_simu]
                logmhalo[idx_simu, :] = logMh(logMs[idx_z], *samples[idx_simu][:-1])
                if idx_simu % (nsimu/10) == 0:
                    print('    Computing SHMR in chains at '+str(idx_simu / nsimu * 100) + '%')
            print('    All logmhalo computed')
            av_logMh[idx_z] = np.average(logmhalo, axis=0)
            med_logMh[idx_z] = np.median(logmhalo, axis=0)
            conf_min_logMh[idx_z] = np.percentile(logmhalo, 16, axis=0)  # 16th percentile = median - 1sigma (68% confidence interval)
            conf_max_logMh[idx_z] = np.percentile(logmhalo, 84, axis=0)
        np.save(directory + '/logMs.npy', logMs)
        np.save(directory + '/av_logMh.npy', av_logMh)
        np.save(directory + '/med_logMh.npy', med_logMh)
        np.save(directory + '/conf_min_logMh.npy', conf_min_logMh)
        np.save(directory + '/conf_max_logMh.npy', conf_max_logMh)
        print('Arrays saved')
    else:
        print('Load arrays')
        logMs = np.load(directory + '/logMs.npy')
        av_logMh = np.load(directory + '/av_logMh.npy')
        med_logMh = np.load(directory + '/med_logMh.npy')
        conf_min_logMh = np.load(directory + '/conf_min_logMh.npy')
        conf_max_logMh = np.load(directory + '/conf_max_logMh.npy')
    return logMs, av_logMh, med_logMh, conf_min_logMh, conf_max_logMh


def plotSHMR_noerror(directory, load=True, selected_redshifts=np.arange(10)):
    paramfile = directory + '/MCMC_param.ini'
    global params
    params = load_params(paramfile)
    global smf
    global hmf
    smf = load_smf(params)
    hmf = load_hmf(params)
    Ms_min = np.maximum(np.log10(6.3 * 10**7 * (1 + params['redshiftsbin'])**2.7), np.full(params['numzbin'], 9))
    print(Ms_min)
    Ms_max = params['SM_cut_max']
    numpoints = 100
    logMs, av_logMh, med_logMh, conf_min_logMh, conf_max_logMh = save_load_smhm(
        directory, Ms_min, Ms_max, numpoints, load, selected_redshifts)

    """Plot the SHMR"""
    plt.figure()
    # plt.set_prop_cycle('viridis')
    plt.ylim(11, 15)
    plt.xlim(9, 12)
    for idx_z in selected_redshifts:
        # plt.fill_between(logMs[idx_z], conf_min_logMh[idx_z], conf_max_logMh[idx_z], color="C{}".format(idx_z), alpha=0.3)
        plt.plot(
            logMs[idx_z], med_logMh[idx_z],
            label=str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]),
            color="C{}".format(idx_z))
    plt.xlabel('$\mathrm{log}_{10}(M_{*}/M_{\odot})$', size=16, labelpad=5)
    plt.ylabel('$\mathrm{log}_{10}(M_{\mathrm{h}}/M_{\odot})$', size=16, labelpad=5)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(prop={'size': 12})
    plt.tight_layout()
    plt.savefig(directory + '/Plots/SHMR_Allz_noerror.pdf')

    """Plot the Ms/Mh ratio"""
    plt.figure()
    plt.set_cmap('viridis')
    for idx_z in selected_redshifts:
        """Plot the median"""
        x = med_logMh[idx_z]
        y = logMs[idx_z] - med_logMh[idx_z]
        xerr = [x - conf_min_logMh[idx_z], conf_max_logMh[idx_z] - x]
        yerr = [xerr[1], xerr[0]]
        plt.plot(
            x, y,
            label=str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]),
            color="C{}".format(idx_z))
    logspace = np.linspace(11, 16)
    # plt.plot(logspace, np.max(Ms_max) - logspace, c='black', linestyle='--', label='$M_{*}= Max cut in stellar mass$')
    plt.xlabel('$\mathrm{log}_{10}(M_{\mathrm{h}}/M_{\odot})$', size=16, labelpad=5)
    plt.ylabel('$\mathrm{log}_{10}(M_{*}/M_{\\mathrm{h}})$', size=16, labelpad=5)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlim(11.2, 14.5)
    plt.ylim(-2.2, -1.4)
    plt.legend(ncol=2, loc=3)
    plt.tight_layout()
    plt.savefig(directory + '/Plots/DeltaSMHM_Allz_noerror.pdf')


def plotSHMR_delta(directory, load=True, selected_redshifts=np.arange(10)):
    """Good version to use to plot the SHMR and the Ms(Mh)"""
    paramfile = directory + '/MCMC_param.ini'
    global params
    params = load_params(paramfile)
    global smf
    global hmf
    smf = load_smf(params)
    hmf = load_hmf(params)
    Ms_min = np.maximum(np.log10(6.3 * 10**7 * (1 + params['redshiftsbin'])**2.7), np.full(params['numzbin'], 9))
    print(Ms_min)
    Ms_max = params['SM_cut_max']
    numpoints = 100
    logMs, av_logMh, med_logMh, conf_min_logMh, conf_max_logMh = save_load_smhm(
        directory,logMs, Ms_min, Ms_max, numpoints, load, selected_redshifts)

    plt.figure()
    plt.ylim(11, 15)
    plt.xlim(9, 12)
    for idx_z in selected_redshifts:
        plt.fill_between(logMs[idx_z], conf_min_logMh[idx_z], conf_max_logMh[idx_z], color="C{}".format(idx_z), alpha=0.3)
        plt.plot(logMs[idx_z], med_logMh[idx_z], label=str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]), color="C{}".format(idx_z))
    """PLot the Behroozi SHMR"""
    # log_ms_boo, log_mh_boo = np.load('SHMR_Behroozi_z0.npy')
    # plt.plot(log_ms_boo, log_mh_boo, c='black', linestyle='--', label='Behroozi et al. 2013, z=0.35')
    plt.xlabel('$\mathrm{log}_{10}(M_{*}/M_{\odot})$', size=17)
    plt.ylabel('$\mathrm{log}_{10}(M_{\mathrm{h}}/M_{\odot})$', size=17)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.legend(prop={'size': 12})
    plt.tight_layout()
    #plt.show()
    plt.savefig(directory + '/Plots/SHMR_Allz.pdf')

    if np.size(selected_redshifts)==10:
        plt.figure()
        plt.ylim(11, 15)
        plt.xlim(9, 12)
        for idx_z in np.arange(6):
            plt.fill_between(logMs[idx_z], conf_min_logMh[idx_z], conf_max_logMh[idx_z], color="C{}".format(idx_z), alpha=0.3)
            plt.plot(logMs[idx_z], med_logMh[idx_z], label=str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]), color="C{}".format(idx_z))
        plt.xlabel('$\mathrm{log}_{10}(M_{*}/M_{\odot})$', size=17)
        plt.ylabel('$\mathrm{log}_{10}(M_{\mathrm{h}}/M_{\odot})$', size=17)
        plt.tick_params(axis='both', which='major', labelsize=13)
        plt.legend(prop={'size': 12})
        plt.tight_layout()
        #plt.show()
        plt.savefig(directory + '/Plots/SHMR_upto6.pdf')

        plt.figure()
        plt.ylim(11, 15)
        plt.xlim(9, 12)
        for idx_z in np.arange(4)+6:
            plt.fill_between(logMs[idx_z], conf_min_logMh[idx_z], conf_max_logMh[idx_z], color="C{}".format(idx_z), alpha=0.3)
            plt.plot(logMs[idx_z], med_logMh[idx_z], label=str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]), color="C{}".format(idx_z))
        plt.xlabel('$\mathrm{log}_{10}(M_{*}/M_{\odot})$', size=17)
        plt.ylabel('$\mathrm{log}_{10}(M_{\mathrm{h}}/M_{\odot})$', size=17)
        plt.tick_params(axis='both', which='major', labelsize=13)
        plt.legend(prop={'size': 12})
        plt.tight_layout()
        #plt.show()
        plt.savefig(directory + '/Plots/SHMR_789.pdf')

    plt.figure()
    for idx_z in selected_redshifts:
        """Plot the average"""
        # x = av_logMh[idx_z]
        # y = logMs[idx_z] - av_logMh[idx_z]
        # xerr = [x - conf_min_logMh[idx_z], conf_max_logMh[idx_z] - x]
        # # yerr = [y - conf_max_logMh[idx_z], conf_min_logMh[idx_z] - y]
        # yerr = [xerr[1], xerr[0]]
        # # plt.errorbar(x, y, yerr= yerr, xerr=xerr)
        # plt.fill_between(x, y - yerr[0], yerr[1] + y, alpha=0.3)
        # plt.plot(x, y, label='av '+str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]))
        """Plot the median"""
        x = med_logMh[idx_z]
        y = logMs[idx_z] - med_logMh[idx_z]
        xerr = [x - conf_min_logMh[idx_z], conf_max_logMh[idx_z] - x]
        # yerr = [y - conf_max_logMh[idx_z], conf_min_logMh[idx_z] - y]
        yerr = [xerr[1], xerr[0]]
        # plt.errorbar(x, y, yerr= yerr, xerr=xerr)
        plt.fill_between(x, y - yerr[0], yerr[1] + y, alpha=0.3, color="C{}".format(idx_z))
        plt.plot(x, y, label=str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]), color="C{}".format(idx_z))
    logspace = np.linspace(11, 16)
    plt.plot(logspace, np.max(Ms_max) - logspace, c='black', linestyle='--', label='$M_{*}= Max cut in stellar mass$')
    plt.xlabel('$\mathrm{log}_{10}(M_{\mathrm{h}}/M_{\odot})$', size=17)
    plt.ylabel('$\mathrm{log}_{10}(M_{*}/M_{\\mathrm{h}})$', size=17)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.xlim(11.2, 14.6)
    plt.ylim(-2.85, -0.9)
    plt.legend(ncol=2, loc=3)
    plt.tight_layout()
    # plt.show()
    plt.savefig(directory + '/Plots/DeltaSMHM_Allz.pdf')

    if np.size(selected_redshifts)==10:
        plt.figure()
        for idx_z in np.arange(6):
            """Plot the median"""
            x = med_logMh[idx_z]
            y = logMs[idx_z] - med_logMh[idx_z]
            xerr = [x - conf_min_logMh[idx_z], conf_max_logMh[idx_z] - x]
            # yerr = [y - conf_max_logMh[idx_z], conf_min_logMh[idx_z] - y]
            yerr = [xerr[1], xerr[0]]
            # plt.errorbar(x, y, yerr= yerr, xerr=xerr)
            plt.fill_between(x, y - yerr[0], yerr[1] + y, alpha=0.3, color="C{}".format(idx_z))
            plt.plot(x, y, label=str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]), color="C{}".format(idx_z))
        logspace = np.linspace(11, 16)
        # plt.plot(logspace, np.max(Ms_max) - logspace, c='black', linestyle='--', label='$M_{*}= Max cut in stellar mass$')
        plt.xlabel('$\mathrm{log}_{10}(M_{\mathrm{h}}/M_{\odot})$', size=17)
        plt.ylabel('$\mathrm{log}_{10}(M_{*}/M_{\\mathrm{h}})$', size=17)
        plt.tick_params(axis='both', which='major', labelsize=13)
        plt.xlim(11.2, 13.5)
        plt.ylim(-2.5, -1.4)
        plt.legend(ncol=2, loc=3)
        plt.tight_layout()
        # plt.show()
        plt.savefig(directory + '/Plots/DeltaSMHM_upto6.pdf')
        plt.figure()
        for idx_z in np.arange(4)+6:
            """Plot the median"""
            x = med_logMh[idx_z]
            y = logMs[idx_z] - med_logMh[idx_z]
            xerr = [x - conf_min_logMh[idx_z], conf_max_logMh[idx_z] - x]
            # yerr = [y - conf_max_logMh[idx_z], conf_min_logMh[idx_z] - y]
            yerr = [xerr[1], xerr[0]]
            # plt.errorbar(x, y, yerr= yerr, xerr=xerr)
            plt.fill_between(x, y - yerr[0], yerr[1] + y, alpha=0.3, color="C{}".format(idx_z))
            plt.plot(x, y, label=str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]), color="C{}".format(idx_z))
        logspace = np.linspace(11, 16)
        # plt.plot(logspace, np.max(Ms_max) - logspace, c='black', linestyle='--', label='$M_{*}= Max cut in stellar mass$')
        plt.xlabel('$\mathrm{log}_{10}(M_{\mathrm{h}}/M_{\odot})$', size=17)
        plt.ylabel('$\mathrm{log}_{10}(M_{*}/M_{\\mathrm{h}})$', size=17)
        plt.tick_params(axis='both', which='major', labelsize=13)
        plt.xlim(11.2, 14.6)
        plt.ylim(-2.85, -0.9)
        plt.legend(ncol=2, loc=3)
        plt.tight_layout()
        # plt.show()
        plt.savefig(directory + '/Plots/DeltaSMHM_789.pdf')


def testSMF(idx_z, M1, Ms0, beta, delta, gamma, ksi):
    """Good version to use to plot the SHMR and the Ms(Mh)"""
    paramfile = 'MCMC_param.ini'
    global params
    params = load_params(paramfile)
    global smf
    global hmf
    smf = load_smf(params)
    hmf = load_hmf(params)
    if params['do_sm_cut']:
        select = np.where(np.logical_and(
                np.logical_and(
                    smf[idx_z][:, 0] > params['SM_cut_min'][idx_z],
                    smf[idx_z][:, 0] < params['SM_cut_max'][idx_z]
                ),
                smf[idx_z][:, 1] > -40)
            )[0]
    else:
        select = np.where(smf[idx_z][:, 1] > -40)[0]
    logMs = np.linspace(smf[idx_z][select[0], 0], smf[idx_z][select[-1], 0], num=50)
    plt.errorbar(smf[idx_z][select, 0], smf[idx_z][select, 1],
        yerr=[smf[idx_z][select, 3], smf[idx_z][select, 2]], fmt='o')

    logphitrue = np.log10(phi_true(logMs, idx_z, M1, Ms0, beta, delta, gamma, ksi))
    plt.plot(logMs, logphitrue)
    plt.show()


def plotMsMh_fixedMh(directory):
    paramfile = directory + '/MCMC_param.ini'
    global params
    params = load_params(paramfile)
    global smf
    global hmf
    smf = load_smf(params)
    hmf = load_hmf(params)
    Ms_min = np.maximum(np.log10(6.3 * 10**7 * (1 + params['redshiftsbin'])**2.7), np.full(params['numzbin'], 9))
    print(Ms_min)
    numpoints = 100
    Ms_max = params['SM_cut_max']
    logMs = np.empty([params['numzbin'], numpoints])
    for idx_z in range(params['numzbin']):
        logMs[idx_z] = np.linspace(Ms_min[idx_z], Ms_max[idx_z], num=numpoints)
    av_logMh = np.load(directory + '/av_logMh.npy')
    conf_min_logMh = np.load(directory + '/conf_min_logMh.npy')
    conf_max_logMh = np.load(directory + '/conf_max_logMh.npy')

    fixed_mh = np.array([11.5, 12, 13])
    idx_fix = np.zeros([fixed_mh.shape[0], params['numzbin']]).astype('int')
    smhm_fix = np.zeros([fixed_mh.shape[0], params['numzbin']])
    conf_smhm_fix = np.zeros([fixed_mh.shape[0], params['numzbin'], 2])


    for fix in range(fixed_mh.shape[0]):
        for idx_z in range(params['numzbin']):
            idx_fix[fix, idx_z] = np.argmin(np.abs(av_logMh[idx_z, :] - fixed_mh[fix]))
            if idx_fix[fix, idx_z] == 0 or idx_fix[fix, idx_z] == av_logMh.shape[1]:
                smhm_fix[fix, idx_z] = None
                conf_smhm_fix[fix, idx_z, 0] = None
                conf_smhm_fix[fix, idx_z, 1] = None
            else:
                smhm_fix[fix, idx_z] = logMs[idx_z, idx_fix[fix, idx_z]] - av_logMh[idx_z, idx_fix[fix, idx_z]]
                # The error interval on the log of the SMHM ratio is the same as the error on the Halo mass
                tmp = np.array([av_logMh[idx_z, idx_fix[fix, idx_z]] - conf_min_logMh[idx_z, idx_fix[fix, idx_z]],
                    conf_max_logMh[idx_z, idx_fix[fix, idx_z]] - av_logMh[idx_z, idx_fix[fix, idx_z]]])
                conf_smhm_fix[fix, idx_z, 0] = tmp[0]
                conf_smhm_fix[fix, idx_z, 1] = tmp[1]

    max_z = 7
    plt.figure()
    for fix in range(fixed_mh.shape[0]):
        plt.errorbar(
            params['redshiftsbin'][:max_z], smhm_fix[fix, :max_z],
            yerr=np.transpose(conf_smhm_fix[fix, :max_z, :]), capsize=3,
            label='$M_{{\mathrm{{h}}}} = 10^{{{:.1f}}} M_{{\odot}}$'.format(fixed_mh[fix]))
    plt.xlabel('Redshift', size=17)
    plt.ylabel('$\mathrm{log}_{10}(M_{*}/M_{\mathrm{h}})$', size=17)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.legend()
    plt.tight_layout()
    plt.show()
