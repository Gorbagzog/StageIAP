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
# import scipy.optimize as op
from scipy import signal
import corner
from getdist import plots, MCSamples
import time
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

import os
os.environ["OMP_NUM_THREADS"] = "1"

def load_smf(params):
    """Load the SMF"""
    print(params)
    smf_name = params['smf_name']
    if smf_name == 'cosmos' or smf_name == 'cosmos_schechter':

        """Load the SMF from Iary Davidzon+17"""
        # redshifts of the Iari SMF
        params['redshifts'] = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
        #params['redshiftsbin'] = (redshifts[1:]+redshifts[:-1])/2
        print('Use the average redshift of the bin from the Iary SMF fit')
        params['redshiftsbin'] = np.array([0.37, 0.668, 0.938, 1.286, 1.735, 2.220, 2.683, 3.271, 3.926, 4.803])
        params['numzbin'] = np.size(params['redshifts']) - 1
        print('numzbin: '+str( params['numzbin']))

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

        """Test to do subsampling of the SMF"""
        # print(params['SMF_subsampling'])
        if params['SMF_subsampling']:
            subsampling_step = params['subsampling_step']
            print('Do a subsampling of the SMF with a step of '+str(subsampling_step))
            for i in range(params['numzbin']):
                smf[i] = np.array(np.transpose([smf[i][::subsampling_step, 0], smf[i][::subsampling_step, 1], smf[i][::subsampling_step, 2], smf[i][::subsampling_step, 3]]))
        else:
            print('No subsampling of the SMF')

    if smf_name == 'candels':
        print('Use the Candels SMF')
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
    """Load the HMF"""
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


    if hmf_name == ('despali16' or 'tinker08' or 'watson13' or 'bocquet16' or 'bhattacharya11'):
        """Use the Colossus module for the HMF"""
        print('Use '+hmf_name+' HMF in Planck15 cosmo from Colossus module')
        if hmf_name == ('watson13' or 'bhattacharya11'):
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
                         np.log10(mass_function.massFunction(
                                M, redshift_haloes[i], mdef = mdef, model =hmf_name, q_out = 'dndlnM'
                            ) * np.log(10) * cosmo.h**3
                            ## Mass functions are in h^3 Mpc^-3, and need to multiply by ln(10) to have dndlog10m
                            )]
                        )
                    )
                )
    return hmf

    # if hmf_name == 'colossus_tinker08':
    #     """Use the Colossus module for Tinker 2008 HMF"""
    #     print('Use Tinker+08 HMF in Planck15 cosmo from Colossus module')
    #     mdef = '200m'
    #     print('Use '+mdef+' for the SO defintion.')
    #     cosmo = cosmology.setCosmology('planck15')
    #     redshift_haloes = params['redshiftsbin']
    #     M = 10**np.arange(8.0, 17, 0.01) # Mass in Msun / h
    #     for i in range(params['numzbin']):
    #         hmf.append(
    #             np.transpose(
    #                 np.array(
    #                     [np.log10(M / cosmo.h),
    #                      np.log10(mass_function.massFunction(
    #                             M, redshift_haloes[i], mdef = mdef, model ='tinker08', q_out = 'dndlnM'
    #                         ) * np.log(10) * cosmo.h**3
    #                         ## Mass functions are in h^3 Mpc^-3, and need to multiply by ln(10) to have dndlog10m
    #                         )]
    #                     )
    #                 )
    #             )

    # if hmf_name == 'colossus_watson13':
    #     """Use the Colossus module for Tinker 2008 HMF"""
    #     print('Use Watson+13 HMF in Planck15 cosmo from Colossus module')
    #     mdef = 'fof'
    #     print('Use '+mdef+' for the SO defintion.')
    #     cosmo = cosmology.setCosmology('planck15')
    #     redshift_haloes = params['redshiftsbin']
    #     M = 10**np.arange(8.0, 17, 0.01) # Mass in Msun / h
    #     for i in range(params['numzbin']):
    #         hmf.append(
    #             np.transpose(
    #                 np.array(
    #                     [np.log10(M / cosmo.h),
    #                      np.log10(mass_function.massFunction(
    #                             M, redshift_haloes[i], mdef = mdef, model ='watson13', q_out = 'dndlnM'
    #                         ) * np.log(10) * cosmo.h**3
    #                         ## Mass functions are in h^3 Mpc^-3, and need to multiply by ln(10) to have dndlog10m
    #                         )]
    #                     )
    #                 )
    #             )


"""Function definitions for computation of the theoretical SMF phi_true"""


def logMh(logMs, M1, Ms0, beta, delta, gamma):
    """SM-HM relation"""
    return M1 + beta*(logMs - Ms0) + (10 ** (delta * (logMs - Ms0))) / (1 + (10 ** (-gamma * (logMs - Ms0)))) - 0.5


def log_phi_direct(logMs, hmf, idx_z, M1, Ms0, beta, delta, gamma):
    # print(delta)
    """"SMF obtained from the SM-HM relation and the HMF"""
    epsilon = 0.0001
    log_Mh1 = logMh(logMs, M1, Ms0, beta, delta, gamma)
    log_Mh2 = logMh(logMs + epsilon, M1, Ms0, beta, delta, gamma)
    # if np.any(log_Mh2 > hmf[idx_z][-1, 0]) or np.any(log_Mh1 < hmf[idx_z][0, 0]):
    #     # print('above hmf')
    #     return log_Mh1 * 0. + 10
    # else :
    if True:
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
        return log_phidirect

    # log_phidirect[log_Mh1 > hmf[idx_z][-1, 0]] = 10**6 # Do not use points with halo masses not defined in the HMF
    # log_phidirect[log_Mh1 < hmf[idx_z][0, 0]] = 10**6


# def gauss(y, ksi):
#     return 1. / (ksi * np.sqrt(2 * np.pi)) * np.exp(- 1/2 * (y / ksi)**2)

def log_phi_true(logMs, hmf, idx_z, params, M1, Ms0, beta, delta, gamma, ksi):
    """Use convolution defined in Behroozi et al 2010"""
    log_phi_dir = log_phi_direct(logMs, hmf, idx_z, M1, Ms0, beta, delta, gamma)
    phitrue = log_phi_dir * 0.

    # dx = logMs[1] - logMs[0]
    if params['smf_name'] == 'cosmos_schechter':
        dx = 0.1
    else:
        print('Warning, the step between two mass bins in not defined in this case.')
    x = np.arange(-10*ksi/dx, 10*ksi/dx, dx)
    gaussian = 1. / (ksi * np.sqrt(2 * np.pi)) * np.exp(- 1/2 * (x / ksi)**2) * dx
    # return np.log10(np.convolve(10**log_phi_dir, gaussian, mode='same'))
    return np.log10(signal.convolve(10**log_phi_dir, gaussian, mode='same'))

    # for i in range(phitrue.size):
    #     for j in range(phitrue.size -1):
    #         phitrue[i] = phitrue[i] + 10**log_phi_dir[j] * gauss(logMs[j] - logMs[i], ksi) * (logMs[j+1] - logMs[j])
    # print(phitrue)
    # return np.log10(phitrue)

    # return log_phi_dir

def chi2(smf, hmf, idx_z, params, M1, Ms0, beta, delta, gamma, ksi):
    """"return the chi**2 between the observed and the expected SMF"""
    select = np.where(smf[idx_z][:, 1] > -40)[0]  # select points where the smf is defined
    # We choose to limit the fit only for abundances higher than 10**-7
    logMs = smf[idx_z][select[:], 0]
    pred = 10**log_phi_true(logMs, hmf, idx_z, params, M1, Ms0, beta, delta, gamma, ksi)
    if params['smf_name'] == ('cosmos' or 'candels'):
        print('Check that there may be an issue with the SMF cut')
        # chi2 = np.sum(
        #         # When using the Vmax directly (give the error bars directly, in a linear scale)
        #         # Need to use a linear scale to compute the chi2 with the right uncertainty
        #         ((pred - 10**smf[idx_z][select, 1]) / (
        #             10**(smf[idx_z][select, 2] + smf[idx_z][select, 1]) - 10**smf[idx_z][select, 1]))**2
        #     )
    elif params['smf_name'] == 'cosmos_schechter':
        if params['do_sm_cut']:
            sm_select = np.where(np.logical_and(
                        logMs > params['SM_cut_min'][idx_z],
                        logMs < params['SM_cut_max'][idx_z]
                    )
                )
        else:
            sm_select = True
        """In the case of the Schechter fit, error bars are non symmetric."""
        chi2 = np.sum(
                ((pred[sm_select] - 10**smf[idx_z][select, 1][sm_select]) / (
                    10**(smf[idx_z][select, 2][sm_select] + smf[idx_z][select, 1][sm_select]) - 10**smf[idx_z][select, 1][sm_select]))**2
                    #10**smf[idx_z][select, 1] - 10**(smf[idx_z][select, 1] - smf[idx_z][select, 3])))**2
        )
    return chi2


# def chi2_minimize(p0, ksi, idx_z):
#     """"test to use this definition of chi to do a scipy.minimize"""
#     M1, Ms0, beta, delta, gamma = p0
#     select = np.where(smf[idx_z][:, 1] > -40)[0]  # select points where the smf is defined
#     # We choose to limit the fit only for abundances higher than 10**-7
#     logMs = smf[idx_z][select[:], 0]
#     pred = 10**log_phi_true(logMs, idx_z, M1, Ms0, beta, delta, gamma, ksi)
#     chi2 = np.sum(
#             ((pred - 10**smf[idx_z][select, 1]) / (
#                 10**smf[idx_z][select, 1] - 10**(smf[idx_z][select, 1] - smf[idx_z][select, 3])))**2
#     )
#     return chi2


# def var_std(x_est , x_true, sigm, sigp):
#     """Compute a varying standard deviation to have a varying gaussian for the likelihood with non symmetric errors"""
#     sig0 = 2 * sigm * sigp / (sigm + sigp)
#     sig1 = (sigm -  sigp) / (sigp + sigm)
#     return sig0 + sig1 * (x_true - x_est)


def loglike(theta, smf, hmf, idx_z, params, minbound, maxbound):
    """return the loglikelihood"""
    if all(theta >= minbound[idx_z]) and all(theta <= maxbound[idx_z]):
        M1, Ms0, beta, delta, gamma, ksi = theta[:]
        return -chi2(smf, hmf, idx_z, params, M1, Ms0, beta, delta, gamma, ksi)/2
    else:
        return -np.inf


# def negloglike(theta, idx_z, minbound, maxbound):
#     return -loglike(theta, idx_z, minbound, maxbound)


""" Run MCMC """

def get_platform():
    """Returns the save path and the number of threads (=cores for parallelization)
    depending on the machin teh script is ran."""
    if platform.uname()[1] == 'imac-de-louis':
        print('Run locally')
        return '../', 1
    elif platform.uname()[1] == 'glx-calcul3':
        print('Run on the glx-calcul3 machine')
        return '/data/glx-calcul3/data1/llegrand/StageIAP/', 20
    elif platform.uname()[1] == 'glx-calcul1':
        print('Run on the glx-calcul1 machine')
        return '/data/glx-calcul3/data1/llegrand/StageIAP/', 20
    elif platform.uname()[1] == 'MacBook-Pro-de-Louis.local':
        print('Run on local on my MBP')
        return '../', 1
    elif platform.uname()[1] == 'mic-llegrand.ias.u-psud.fr':
        print('Run on local on my MBP on IAS network')
        return '../', 1
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

def runMCMC_allZ(paramfile):
    """Main function to run all MCMC on all zbins based on the param file"""
    params = load_params(paramfile)
    smf = load_smf(params)
    hmf = load_hmf(params)

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
    # Copy parameter files in the save directory
    copyfile(params['starting_point_file'], directory + '/' + params['starting_point_file'])
    copyfile(paramfile, directory + '/' + paramfile)
    copyfile(params['minboundfile'], directory + '/' + params['minboundfile'])
    copyfile(params['minboundfile'], directory + '/' + params['minboundfile'])
    print(params['selected_redshifts'])
    # Make the file for autocorr, burn in length and thin length
    with open(directory + "/Chain/Autocorr.txt", "a") as myfile:
        myfile.write("# Mean autocorrelation, Burn in length,  Thin length \n")
    # run all MCMC for all zbins
    # for idx_z in range(params['numzbin']):
    for idx_z in params['selected_redshifts']:
        print(idx_z)
        print('Starting MCMC run for idx_z =' + str(idx_z) )
        print('Min bound: ' + str(params['minbound'][idx_z]))
        print('Max bound: ' + str(params['maxbound'][idx_z]))
        runMCMC(directory, smf, hmf, idx_z, params)
    # Plot all SHMR on one graph
    plotSHMR_delta(directory, params['iterations'], params, load=False, selected_redshifts = params['selected_redshifts'])
    # Plot the MhaloPeak graph
    plt.clf()
    plt.figure(figsize=(10, 5))
    Plot_MhaloPeak.plotLiterrature()
    Plot_MhaloPeak.plotFit(directory, params['smf_name'], params['hmf_name'])
    Plot_MhaloPeak.savePlot(directory)
    plt.clf()

def runMCMC(directory, smf, hmf, idx_z, params):
    # load_smf()
    # load_hmf()
    # nwalker = 20
    # starting_point =  np.array([12.5, 10.8, 0.5, 0.5, 0.5, 0.15])
    # std =np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.01])
    # starting_point =  np.array([12.5, 11, 0.5, 0.7, 0.5, 0.15])
    minbound, maxbound, starting_point, std, iterations, nthreads, nwalkers, noksi = params['minbound'], params['maxbound'], params['starting_point'], params['std'], params['iterations'], params['nthreads'], params['nwalkers'], params['noksi']
    start_time = time.time()
    p0 = emcee.utils.sample_ball(starting_point[idx_z], std, size=nwalkers)
    ndim = len(starting_point[idx_z])

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = directory+'/Chain/samples_'+str(idx_z)+'.h5'
    # filename = "tutorial.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    print('Using backend to save the chain to '+filename)
    # print('Nthreads: '+str(nthreads))
    with Pool(processes=15) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike,
            args=[smf, hmf, idx_z, params, minbound, maxbound], pool=pool,
            backend=backend)
        print("idx_z = " +str (idx_z))
        print("ndim = " + str(ndim))
        print("start = " + str(starting_point[idx_z]))
        print("std = " + str(std))
        print("iterations = " + str(iterations))
        # print("burn = " + str(burn))
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

        # # Save chains and loglike of chains ### Now already saved with the backend
        # chainfile = directory + "/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
        # savenameln = directory + "/Chain/LnProb_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
        # np.save(chainfile, sampler.chain)
        # np.save(savenameln, sampler.lnprobability)
        # # test convergence
        # # test_convergence(directory, idx_z, iterations, burn)

        # Clean samples
        tau = sampler.get_autocorr_time(tol=0)
        burnin = int(2*np.max(tau))
        print("Burnin "+str(burnin))
        thin = int(0.5*np.min(tau))
        with open(directory + "/Chain/Autocorr.txt", "a") as myfile:
            myfile.write(str(np.mean(tau))+ "  " + str(burnin) + "  " + str(thin) + "\n")

        samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)

        # Plot all relevant figures
        plt.close('all')
        plotchain(directory, samples, idx_z, params)
        plt.close('all')
        plotdist(directory, samples, idx_z, iterations, params)
        plt.close('all')
        plotSMF(directory, samples, smf, hmf, idx_z, params, iterations)
        plt.close('all')
        plotSMHM(directory, samples, smf, idx_z, iterations)
        plt.close('all')
        plotSHMR(directory, samples, smf, idx_z, iterations)
        plt.close('all')
        plot_Mhpeak(directory, samples, idx_z, iterations, params)
        plt.close('all')
        save_results(directory, samples, idx_z, iterations, params['noksi'], params)

        # # Reset before new MCMC
        # sampler.reset()


def plotAutocorr(directory, idx_z, autocorr, index):
    n = 100*np.arange(1, index+1)
    y = autocorr[:index]
    plt.plot(n, n / 50.0, "--k")
    plt.plot(n, y)
    plt.xlim(0, n.max())
    plt.ylim(0, y.max() + 0.1*(y.max() - y.min()))
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$");
    plt.savefig(directory+'/Plots/TestConvergence_'+str(idx_z)+'.pdf')


def save_results(directory, samples, idx_z, iterations, noksi, params):
    # chain = np.load(chainfile)
    # if noksi:
    #     print('This may not be supported form 13th of July 2018')
    #     chain =chain[:,:,:5]
    #     names = ['$M_{1}$', '$M_{s,0}$', '$\\beta$', '$\delta$', '$\gamma$']
    #     ranges = dict(zip(names, np.transpose(np.array([params['minbound'][idx_z], params['maxbound'][idx_z]]))))
    # else:
    names = ['$M_{1}$', '$M_{s,0}$', '$\\beta$', '$\delta$', '$\gamma$', r'$\xi$']
    ranges = dict(zip(names, np.transpose(np.array([params['minbound'][idx_z], params['maxbound'][idx_z]]))))
    # samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    samples = MCSamples(samples=samples, names=names, ranges=ranges)
    res = samples.getTable()
    res.write(directory+"/Results/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".txt")


# def getParam(directory, chainfile, idx_z, iterations, burn):
#     """Get paramInfo objects giving results of the MCMC"""
#     chain = np.load(chainfile)
#     names = ['$M_{1}$', '$M_{s,0}$', '$\\beta$', '$\delta$', '$\gamma$', r'$\xi$']
#     samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
#     samples = MCSamples(samples = samples, names = names)
#     marge = samples.getMargeStats()
#     marge.parWithName('$M_{1}$').mean
#     marge.parWithName('$M_{1}$').err
#     marge.parWithName('$M_{1}$').limits[0].lower # 1 sigma = 68% lower limit


def MhPeak(samples, idx_z, iterations):
    # chain = np.load(chainfile)
    # samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    chainsize = np.shape(samples)[0]
    logMs = np.linspace(8, 12, num=300)
    Mhalopeak = np.zeros(chainsize)
    for i in range(chainsize):
        logmhalo = logMh(logMs, samples[i, 0], samples[i, 1], samples[i, 2], samples[i, 3], samples[i, 4])
        Mhalopeak_idx = np.argmax(logMs - logmhalo)
        Mhalopeak[i] = logmhalo[Mhalopeak_idx]
    return Mhalopeak


# def allMhPeak(directory, iterations, burn):
#     # numzbin = 10
#     mhpeakall = np.zeros(params['numzbin'])
#     mhpeakallstd = np.zeros(params['numzbin'])
#     for idx_z in range(params['numzbin']):
#         chainfile = directory+"/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
#         mhpeak = MhPeak(chainfile, idx_z, iterations, burn)
#         mhpeakall[idx_z] = np.median(mhpeak)
#         mhpeakallstd[idx_z] = np.std(mhpeak)
#     return mhpeakall, mhpeakallstd


# def gelman_rubin(chain):
#     # chainfile = directory+"/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
#     # chain = np.load(chainfile)
#     ssq = np.var(chain, axis=1, ddof=1)
#     W = np.mean(ssq, axis=0)
#     θb = np.mean(chain, axis=1)
#     θbb = np.mean(θb, axis=0)
#     m = chain.shape[0]
#     n = chain.shape[1]
#     B = n / (m - 1) * np.sum((θbb - θb)**2, axis=0)
#     var_θ = (n - 1) / n * W + 1 / n * B
#     R = np.sqrt(var_θ / W)
#     return R


# def test_convergence(directory, idx_z, iterations, burn):
#     """Test the convergence of the chains"""
#     # ndim_arr = [6]
#     chainfile = directory+"/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
#     chain = np.load(chainfile)
#     print("ndim\tμ\t\tσ\tGelman-Rubin")
#     print("============================================")
#     ndim = 6
#     print("{0:3d}\t{1: 5.4f}\t\t{2:5.4f}\t\t{3:3.2f}".format(
#                 ndim,
#                 chain.reshape(-1, chain.shape[-1]).mean(axis=0)[0],
#                 chain.reshape(-1, chain.shape[-1]).std(axis=0)[0],
#                 gelman_rubin(chain)[0]))
#     # names = ['$M_{1}$', '$M_{s,0}$', '$\\beta$', '$\delta$', '$\gamma$', 'ksi']
#     # samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
#     # samples = MCSamples(samples = samples, names = names)
#     # R = gelman_rubin(chain)
#     # print(R)


# def delete_non_converged_chains(directory, iterations, burn, idx_z, selec_chain):
#     load_smf('cosmos')
#     load_hmf('hmf_module')
#     chainfile = directory + "/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
#     chain = np.load(chainfile)
#     chain = chain[selec_chain, :, :]
#     np.save(chainfile, chain)
#     # Plot all relevant figures
#     plt.close('all')
#     plotchain(directory, chainfile, idx_z, iterations, burn)
#     plt.close('all')
#     # plotdist(directory, chainfile, idx_z, iterations, burn)
#     # plt.close('all')
#     plotSMF(directory, idx_z, iterations, burn)
#     plt.close('all')
#     plotSMHM(directory, idx_z, iterations, burn)
#     plt.close('all')
#     plot_Mhpeak(directory, chainfile, idx_z, iterations, burn)
#     plt.close('all')
#     save_results(directory, chainfile, idx_z, iterations, burn)


"""Plots"""


def plotSMF(directory, samples, smf, hmf, idx_z, params, iterations):
    # load_smf()
    # load_hmf()
    # chainfile = directory + "/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
    # chain = np.load(chainfile)
    # samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    # chain.close()
    # select = np.where(smf[idx_z][:, 1] > -40)[0]
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
    #plt.ylim(-50, -1)
    for M1, Ms0, beta, delta, gamma, ksi in samples[np.random.randint(len(samples), size=100)]:
        logphi = log_phi_true(logMs, hmf, idx_z, params, M1, Ms0, beta, delta, gamma, ksi)
        plt.plot(logMs, logphi, color="k", alpha=0.1)
    # plt.show()
    plt.xlabel('$\mathrm{log}_{10}(M_* / M_{\odot})$')
    plt.ylabel('$\mathrm{log}_{10}(\phi)$')
    plt.savefig(directory+'/Plots/SMF_ksi'+ str(idx_z) + "_niter=" + str(iterations) + '.pdf')


def plotSMHM(directory, samples, smf, idx_z, iterations):
    # load_smf()
    # load_hmf()
    # chainfile =  directory + "/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
    # chain =  np.load(chainfile)
    # samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    # chain.close()
    #logMs = np.linspace(9, 13, num=200)
    select = np.where(smf[idx_z][:, 1] > -40)[0]
    logMs = np.linspace(smf[idx_z][select[0], 0], smf[idx_z][select[-1], 0], num=50)
    for M1, Ms0, beta, delta, gamma, ksi in samples[np.random.randint(len(samples), size=100)]:
        logmhalo = logMh(logMs, M1, Ms0, beta, delta, gamma)
        # logphi = log_phi_true(logMs, idx_z, M1, Ms0, beta, delta, gamma, ksi)
        plt.plot(logmhalo, logMs-logmhalo, color="k", alpha=0.1)
    plt.xlabel('$\mathrm{log}_{10}(M_{\mathrm{h}} / M_{\odot})$')
    plt.ylabel('$\mathrm{log}_{10}(M_{\mathrm{h}} / M_{*})$')
    plt.savefig(directory+'/Plots/SMHM_ksi'+ str(idx_z) + "_niter=" + str(iterations) + '.pdf')


def plotSHMR(directory, samples, smf, idx_z, iterations):
    # load_smf()
    # load_hmf()
    # chainfile =  directory + "/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
    # chain =  np.load(chainfile)
    # samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    # chain.close()
    #logMs = np.linspace(9, 13, num=200)
    select = np.where(smf[idx_z][:, 1] > -40)[0]
    logMs = np.linspace(smf[idx_z][select[0], 0], smf[idx_z][select[-1], 0], num=50)
    for M1, Ms0, beta, delta, gamma, ksi in samples[np.random.randint(len(samples), size=100)]:
        logmhalo = logMh(logMs, M1, Ms0, beta, delta, gamma)
        # logphi = log_phi_true(logMs, idx_z, M1, Ms0, beta, delta, gamma, ksi)
        plt.plot(logMs, logmhalo, color="k", alpha=0.1)
    plt.xlabel('$\mathrm{log}_{10}(M_{*} / M_{\odot})$')
    plt.ylabel('$\mathrm{log}_{10}(M_{\mathrm{h}} / M_{\odot})$')
    plt.savefig(directory+'/Plots/SHMR'+ str(idx_z) + "_niter=" + str(iterations) + '.pdf')

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


def plotchain(directory, samples, idx_z, params):
    iterations, noksi = params['iterations'],  params['noksi']
    figname = directory + "/Plots/Triangle_z" + str(idx_z) + "_niter=" + str(iterations)
    # chain = np.load(chainfile)
    # if noksi:
    #     chain = chain[:,:,:5]
    # samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    # chain.close()
    fig = corner.corner(
        samples, labels=['$M_{1}$', '$M_{*,0}$', '$\\beta$', '$\delta$', '$\gamma$', r'$\xi$'])
    fig.savefig(figname + ".pdf")
    plt.close('all')

# def plotchain_noksi(directory, chainfile, idx_z, iterations, burn):
#     figname = directory + "/Plots/Ksi_z" + str(idx_z) + "_niter=" + str(iterations) + "_burn=" + str(burn)
#     chain = np.load(chainfile)
#     chain = chain[:,:,:5]
#     samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
#     # chain.close()
#     fig = corner.corner(
#         samples, labels=['$M_{1}$', '$M_{*,0}$', '$\\beta$', '$\delta$', '$\gamma$'])
#     fig.savefig(figname + ".pdf")
#     plt.close('all')

def plotdist(directory, samples, idx_z, iterations, params):
    figname = directory + "/Plots/GetDist_z" + str(idx_z) + "_niter=" + str(iterations)
    names = ['$M_{1}$', '$M_{s,0}$', '$\\beta$', '$\delta$', '$\gamma$', r'$\xi$']
    # chain = np.load(chainfile)
    # samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
    samples = MCSamples(samples=samples, names=names,
        ranges = dict(zip(names, np.transpose(np.array([params['minbound'][idx_z],
        params['maxbound'][idx_z]])))))
    # chain.close()
    g = plots.getSubplotPlotter()
    g.triangle_plot(samples, filled=True)
    g.export(figname + '.pdf' )
    plt.close('all')


# def plotLnprob(idx_z, iterations, nwalker=20):
#     lnprob = np.load("../MCMC/Chain/LnProb_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy")
#     for k in range(nwalker):
#         plt.plot(lnprob[k, :])


def plot_Mhpeak(directory, samples, idx_z, iterations, params):
    mhpeak = MhPeak(samples, idx_z, iterations)
    # avg_mhpeak = np.mean(mhpeak)
    med_mhpeak = np.median(mhpeak)
    std_mhpeak = np.std(mhpeak)
    with open(directory + "/MhaloPeak.txt", "a") as myfile:
        myfile.write(str(idx_z) + "  " + str(med_mhpeak) + "  " + str(std_mhpeak) + "\n")
    plt.figure()
    plt.hist(mhpeak, bins=100)
    plt.axvline(med_mhpeak, color='orange')
    plt.title(str(params['redshifts'][idx_z]) +'<z<' + str(params['redshifts'][idx_z+1]) + ', MhPeak = ' + str(med_mhpeak) + '+/-' + str(std_mhpeak))
    plt.savefig(directory+'/Plots/MhPeak_z' + str(idx_z) + '.pdf')


# def plotSigmaHMvsSM(directory, idx_z, iterations, burn):
#     # load_smf()
#     # load_hmf()
#     chainfile = directory + "/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
#     chain = np.load(chainfile)
#     samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
#     # chain.close()
#     numpoints = 100
#     logMs = np.linspace(9, 12, num=numpoints)
#     logmhalo = np.zeros([samples.shape[0], numpoints])
#     for idx_simu in range(samples.shape[0]):
#         M1, Ms0, beta, delta, gamma = samples[idx_simu]
#         logmhalo[idx_simu, :] = logMh(logMs, M1, Ms0, beta, delta, gamma)
#     av_logMh = np.average(logmhalo, axis=0)
#     conf_min_logMh = np.percentile(logmhalo, 16, axis=0)  # 16th percentile = median - 1sigma (68% confidence interval)
#     conf_max_logMh = np.percentile(logmhalo, 84, axis=0)
#     # for i in range(numpoints):
#     #     av_logMh[i] = np.average(logmhalo[:, i])
#     # plt.close('all')
#     # plt.figure()
#     # plt.fill_between(logMs, conf_min_logMh, conf_max_logMh, alpha=0.3)
#     # plt.plot(logMs, av_logMh, label=str(redshifts[idx_z])+'<z<'+str(redshifts[idx_z+1]))
#     # plt.xlabel('Log($M_{*}/M_{\odot}$)', size=20)
#     # plt.ylabel('Log($M_{h}/M_{\odot}$)', size=20)
#     # plt.legend()
#     # plt.tight_layout()
#     # plt.savefig('../MCMC/Plots/SigmaHMvsSM' + str(idx_z) + "_niter=" +
#     #     str(iterations) + "_burn=" + str(burn) + '.pdf')
#     return av_logMh, conf_min_logMh, conf_max_logMh


# def plotAllSigmaHMvsSM(directory, iterations, burn):
#     # load_smf()
#     # load_hmf()
#     params['numzbin'] =10
#     plt.close('all')
#     plt.figure()
#     numpoints = 100
#     logMs = np.linspace(9, 12, num=numpoints)
#     for idx_z in range(params['numzbin']):
#         chainfile = directory + "/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
#         chain = np.load(chainfile)
#         samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
#         # chain.close()
#         logmhalo = np.zeros([samples.shape[0], numpoints])
#         for idx_simu in range(samples.shape[0]):
#             M1, Ms0, beta, delta, gamma, ksi = samples[idx_simu]
#             logmhalo[idx_simu, :] = logMh(logMs, M1, Ms0, beta, delta, gamma)
#         av_logMh = np.average(logmhalo, axis=0)
#         conf_min_logMh = np.percentile(logmhalo, 16, axis=0)  # 16th percentile = median - 1sigma (68% confidence interval)
#         conf_max_logMh = np.percentile(logmhalo, 84, axis=0)
#         # for i in range(numpoints):
#         #     av_logMh[i] = np.average(logmhalo[:, i])
#         plt.fill_between(logMs, conf_min_logMh, conf_max_logMh, alpha=0.3)
#         plt.plot(logMs, av_logMh, label=str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]))
#     plt.xlabel('Log($M_{*}/M_{\odot}$)', size=20)
#     plt.ylabel('Log($M_{h}/M_{\odot}$)', size=20)
#     plt.legend()
#     plt.tight_layout()
#     # plt.savefig('../MCMC/Plots/SigmaHMvsSM_Allz_niter=' +
#     #     str(iterations) + "_burn=" + str(burn) + '.pdf')


# def temp():
#     # Plot Ms/Mh en ayant pris le Mh average. tester après en prenant la moyenne de Ms/Mh pour un Ms donné
#     numpoints = 100
#     logMs = np.linspace(9, 12, num=numpoints)
#     for idx_z in range(10):
#         plt.plot(tot[idx_z][0], logMs-tot[idx_z][0], label=str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]))
#         plt.fill_between(tot[idx_z][0], logMs - tot[idx_z][1], logMs-tot[idx_z][2], alpha=0.3)
#     plt.legend()
#     plt.xlabel('Log($M_{h}/M_{\odot}$)', size=20)
#     plt.ylabel('Log($M_{*}/M_{h}$)', size=20)


# def plotSigmaSHMR(idx_z, iterations, burn):
#     # load_smf()
#     # load_hmf()
#     chainfile = "../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
#     chain = np.load(chainfile)
#     samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
#     # chain.close()
#     numpoints = 100
#     logMs = np.linspace(9, 12, num=numpoints)
#     logmhalo = np.zeros([samples.shape[0], numpoints])
#     for idx_simu in range(samples.shape[0]):
#         M1, Ms0, beta, delta, gamma = samples[idx_simu]
#         logmhalo[idx_simu, :] = logMh(logMs, M1, Ms0, beta, delta, gamma)
#     av_SHMR = np.average(logMs - logmhalo, axis=0)
#     conf_min_SHMR = np.percentile(logMs - logmhalo, 16, axis=0)  # 16th percentile = median - 1sigma (68% confidence interval)
#     conf_max_SHMR = np.percentile(logMs - logmhalo, 84, axis=0)
#     # for i in range(numpoints):
#     #     av_logMh[i] = np.average(logmhalo[:, i])
#     plt.close('all')
#     plt.figure()
#     plt.fill_between(logMs, conf_min_SHMR, conf_max_SHMR, alpha=0.3)
#     plt.plot(logMs, av_SHMR, label=str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]))
#     plt.xlabel('Log($M_{*}/M_{\odot}$)', size=20)
#     plt.ylabel('Log($M_{*}/M_{h}$)', size=20)
#     plt.legend()
#     plt.tight_layout()
#     # plt.savefig('../MCMC/Plots/SigmaSHMRvsSM' + str(idx_z) + "_niter=" +
#         # str(iterations) + "_burn=" + str(burn) + '.pdf')
#     return av_logMh, conf_min_logMh, conf_max_logMh


# def plotAllSigmaSHMRvsSM(iterations, burn):
#     # load_smf()
#     # load_hmf()
#     numpoints = 100
#     logMs = np.linspace(9, 12, num=numpoints)
#     plt.close('all')
#     plt.figure()
#     for idx_z in range(params['numzbin']):
#         chainfile = "../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
#         chain = np.load(chainfile)
#         samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
#         # chain.close()
#         logmhalo = np.zeros([samples.shape[0], numpoints])
#         for idx_simu in range(samples.shape[0]):
#             M1, Ms0, beta, delta, gamma = samples[idx_simu]
#             logmhalo[idx_simu, :] = logMh(logMs, M1, Ms0, beta, delta, gamma)
#         av_logMh = np.average(logmhalo, axis=0)
#         av_SHMR = np.average(logMs - logmhalo, axis=0)
#         conf_min_SHMR = np.percentile(logMs - logmhalo, 16, axis=0)  # 16th percentile = median - 1sigma (68% confidence interval)
#         conf_max_SHMR = np.percentile(logMs - logmhalo, 84, axis=0)
#     # for i in range(numpoints):
#     #     av_logMh[i] = np.average(logmhalo[:, i])
#         plt.fill_between(av_logMh, conf_min_SHMR, conf_max_SHMR, alpha=0.3)
#         plt.plot(av_logMh, av_SHMR, label=str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]))
#     plt.xlabel('Log($M_{*}/M_{\odot}$)', size=20)
#     plt.ylabel('Log($M_{*}/M_{h}$)', size=20)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('../MCMC/Plots/SigmaSHMRvsSM_All_niter=' +
#         str(iterations) + "_burn=" + str(burn) + '.pdf')


# def plotFakeAllSigmaSHMRvsMH(iterations, burn):
#     # load_smf()
#     # load_hmf()
#     plt.close('all')
#     plt.figure()
#     numpoints = 100
#     logMs = np.linspace(9, 12, num=numpoints)
#     for idx_z in range(params['numzbin']):
#         chainfile = "../MCMC/Chain/Chain_noksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
#         chain = np.load(chainfile)
#         samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
#         # chain.close()
#         logmhalo = np.zeros([samples.shape[0], numpoints])
#         for idx_simu in range(samples.shape[0]):
#             M1, Ms0, beta, delta, gamma = samples[idx_simu]
#             logmhalo[idx_simu, :] = logMh(logMs, M1, Ms0, beta, delta, gamma)
#         av_logMh = np.average(logmhalo, axis=0)
#         conf_min_logMh = np.percentile(logmhalo, 16, axis=0)  # 16th percentile = median - 1sigma (68% confidence interval)
#         conf_max_logMh = np.percentile(logmhalo, 84, axis=0)
#         # for i in range(numpoints):
#         #     av_logMh[i] = np.average(logmhalo[:, i])
#         plt.fill_between(av_logMh, logMs - conf_min_logMh, logMs - conf_max_logMh, alpha=0.3)
#         plt.plot(av_logMh, logMs - av_logMh, label=str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]))
#     # plt.plot([10.519, 10.693, 10.968, 11.231, 11.337, 11.691, 11.940, 12.219, 12.610],
#     #     [-3.232, -3.072, -2.828, -2.629, -2.488, -2.306, -2.172, -2.057, -2.010], label='Harikane z=4')
#     # plt.plot([10.975, 11.292, 12.041]+np.log10(67/70) , [-2.36, -2.206, -2.132], label='Hariakne z=6')
#     plt.xlabel('Log($M_{h}/M_{\odot}$)', size=20)
#     plt.ylabel('Log($M_{*}/M_{h}$)', size=20)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('../MCMC/Plots/SigmaFAKE_SHRMvsHM_Allz_niter=' +
#         str(iterations) + "_burn=" + str(burn) + '.pdf')


# def plotAllSHMRvsSM(directory, iterations, burn):
#     # load_smf('cosmos')
#     # load_hmf('bolshoi_tot')
#     # plt.close('all')
#     plt.figure()
#     numpoints = 100
#     # Use the interploation formula of Mlim(z) in Davidzon et al. 2017
#     Ms_min = np.maximum(np.log10(6.3 * 10**7 * (1 + params['redshiftsbin'])**2.7), np.full(params['numzbin'], 9))
#     print(Ms_min)
#     # Arbitrary maximum as read on the plots of the SMF of Davidzon+17
#     Ms_max = 11.8
#     nselect = 100000  # Number of samples o randomly select in the chains
#     logMhbins = np.linspace(11.5, 14, num=numpoints)
#     for idx_z in range(params['numzbin']):
#         # idx_z += 9
#         logMs = np.linspace(9, 11.8, num=numpoints)
#         avg_MSonMH = np.zeros(numpoints-1)
#         confminus_MSonMH = np.zeros(numpoints-1)
#         confplus_MSonMH = np.zeros(numpoints-1)
#         chainfile = directory+"/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
#         chain = np.load(chainfile)
#         samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
#         #print(len(samples))
#         samples = samples[np.random.randint(len(samples), size=nselect)]
#         del chain
#         print('Chain loaded for idx_z = '+str(idx_z))
#         nsimu = samples.shape[0]
#         print(nsimu)
#         logmhalo = np.zeros([nsimu, numpoints])
#         for idx_simu in range(nsimu):
#             # M1, Ms0, beta, delta, gamma, ksi = samples[idx_simu]
#             logmhalo[idx_simu, :] = logMh(logMs, *samples[idx_simu][:-1])
#             if idx_simu % (nsimu/10) == 0:
#                 print('    Computing SHMR in chains at '+str(idx_simu / nsimu * 100) + '%')
#         print('    All logmhalo computed')
#         print('Computing bins of halo mass..')
#         for idx_bin in range(numpoints-1):
#             idx_MhinBin = np.where(
#                             np.logical_and(
#                                 logmhalo >= logMhbins[idx_bin],
#                                 logmhalo < logMhbins[idx_bin+1]
#                             )
#             )  # Select points that have a halo mass inside the bin
#             smhm_tmp = logMs[idx_MhinBin[1]] - logmhalo[idx_MhinBin]
#             avg_MSonMH[idx_bin] = np.average(smhm_tmp)
#             confminus_MSonMH[idx_bin] = np.percentile(smhm_tmp, 16, axis=0)
#             confplus_MSonMH[idx_bin] = np.percentile(smhm_tmp, 84, axis=0)
#         print('Bins computed')
#         np.save(directory + '/Plots/avg_MSonMH' + str(idx_z) + '.npy', avg_MSonMH)
#         np.save(directory + '/Plots/confminus_MSonMH' + str(idx_z) + '.npy', confminus_MSonMH)
#         np.save(directory + '/Plots/confplus_MSonMH' + str(idx_z) + '.npy', confplus_MSonMH)
#         mh_plotmax = np.min(logmhalo[:, -1])
#         print(mh_plotmax)
#         idx_mhplotmax = np.argmin(np.abs(logMhbins -  mh_plotmax))
#         np.save(directory + '/Plots/idx_mhplotmax' + str(idx_z) + '.npy', idx_mhplotmax)
#         print(idx_mhplotmax)
#         plt.plot((logMhbins[1:idx_mhplotmax] + logMhbins[:idx_mhplotmax-1])/2, avg_MSonMH[:idx_mhplotmax-1],
#             label=str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]))
#         plt.fill_between((logMhbins[1:idx_mhplotmax] + logMhbins[:idx_mhplotmax-1])/2,
#             confminus_MSonMH[:idx_mhplotmax-1], confplus_MSonMH[:idx_mhplotmax-1], alpha=0.3)
#         plt.plot((logMhbins[1:idx_mhplotmax] + logMhbins[:idx_mhplotmax-1])/2,
#             9 - (logMhbins[1:idx_mhplotmax] + logMhbins[:idx_mhplotmax-1])/2,
#             'b--')
#         plt.plot((logMhbins[1:idx_mhplotmax] + logMhbins[:idx_mhplotmax-1])/2,
#             11.8 - (logMhbins[1:idx_mhplotmax] + logMhbins[:idx_mhplotmax-1])/2,
#             'b--')
#         print('Ploted redshift bin ' + str(idx_z))
#         del logmhalo
#         del idx_MhinBin
#         del samples
#         plt.xlabel('Log($M_{h}/M_{\odot}$)', size=20)
#         plt.ylabel('Log($M_{*}/M_{h}$)', size=20)
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(directory+'/Plots/Test' +
#         str(iterations) + "_burn=" + str(burn) + 'z' + str(idx_z) + '.pdf')


# def plotAllSHMRvsSM(directory):
#     """Load previously computed SHMR(HM) and plot them in one figure"""
#     params['numzbin'] = 10
#     numpoints = 100
#     avg_MSonMH = np.zeros(numpoints-1)
#     confminus_MSonMH = np.zeros(numpoints-1)
#     confplus_MSonMH = np.zeros(numpoints-1)
#     logMhbins = np.linspace(11.5, 14, num=numpoints)
#     plt.figure()
#     params['redshifts'] = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
#     for idx_z in range(params['numzbin']):
#         avg_MSonMH = np.load(directory + '/avg_MSonMH' + str(idx_z) + '.npy')
#         confminus_MSonMH = np.load(directory + '/confminus_MSonMH' + str(idx_z) + '.npy')
#         confplus_MSonMH = np.load(directory + '/confplus_MSonMH' + str(idx_z) + '.npy')
#         idx_mhplotmax = np.load(directory + '/idx_mhplotmax' + str(idx_z) + '.npy')
#         plt.plot((logMhbins[1:idx_mhplotmax] + logMhbins[:idx_mhplotmax-1])/2, avg_MSonMH[:idx_mhplotmax-1],
#             label=str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]))
#         plt.fill_between((logMhbins[1:idx_mhplotmax] + logMhbins[:idx_mhplotmax-1])/2,
#             confminus_MSonMH[:idx_mhplotmax-1], confplus_MSonMH[:idx_mhplotmax-1], alpha=0.3)
#     # plt.plot((logMhbins[1:idx_mhplotmax] + logMhbins[:idx_mhplotmax-1])/2,
#     #     9 - (logMhbins[1:idx_mhplotmax] + logMhbins[:idx_mhplotmax-1])/2,
#     #     color='black', linestyle=':')
#     # plt.plot((logMhbins[1:] + logMhbins[:-1])/2,
#     #     11.8 - (logMhbins[1:] + logMhbins[:-1])/2,
#     #     color='black', linestyle=':')
#     plt.xlabel('Log($M_{h}/M_{\odot}$)', size=20)
#     plt.ylabel('Log($M_{*}/M_{h}$)', size=20)
#     plt.legend()
#     plt.tight_layout()


def plotSHMR_delta(directory, iterations, params, load=True, selected_redshifts=np.arange(10)):
    """Good version to use to plot the SHMR and the Ms(Mh)"""
    smf = load_smf(params)
    hmf = load_hmf(params)
    Ms_min = np.maximum(np.log10(6.3 * 10**7 * (1 + params['redshiftsbin'])**2.7), np.full(params['numzbin'], 9))
    print(Ms_min)
    # Arbitrary maximum as read on the plots of the SMF of Davidzon+17
    Ms_max = 11.8
    numpoints = 100
    logMs = np.empty([params['numzbin'], numpoints])
    # nselect = 100000  # Number of samples o randomly select in the chains
    logMhbins = np.linspace(11.5, 14, num=numpoints)
    av_logMh = np.empty([params['numzbin'], numpoints])
    med_logMh = np.empty([params['numzbin'], numpoints])
    conf_min_logMh = np.empty([params['numzbin'], numpoints])
    conf_max_logMh = np.empty([params['numzbin'], numpoints])
    meantau, burnin, thin = np.transpose(np.loadtxt(directory + "/Chain/Autocorr.txt"))
    burnin = burnin.astype('int')
    thin = thin.astype('int')
    if load is False :
        print('Computing arrays')
        for idx_z in selected_redshifts:
        #for idx_z in [6,7,8,9]:
            logMs[idx_z] = np.linspace(Ms_min[idx_z], Ms_max, num=numpoints)
            # chainfile = directory+"/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy"
            # chain = np.load(chainfile)
            # samples = chain[:, burn:, :].reshape((-1, chain.shape[2]))
            #print(len(samples))
            # samples = samples[np.random.randint(len(samples), size=nselect)]
            filename = directory+'/Chain/samples_'+str(idx_z)+'.h5'
            sampler = emcee.backends.HDFBackend(filename)
            samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
            print('Chain loaded for idx_z = '+str(idx_z))
            nsimu = samples.shape[0]
            print(nsimu)
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
    else :
        print('Load arrays')
        logMs = np.load(directory + '/logMs.npy')
        av_logMh = np.load(directory + '/av_logMh.npy')
        med_logMh = np.load(directory + '/med_logMh.npy')
        conf_min_logMh = np.load(directory + '/conf_min_logMh.npy')
        conf_max_logMh = np.load(directory + '/conf_max_logMh.npy')


    plt.figure()
    M1, Ms0, beta, delta, gamma = 12.51, 10.82, 0.484, 0.47, 1.02
    for idx_z in selected_redshifts:
    # for idx_z in [6,7,8,9]:
        plt.fill_between(logMs[idx_z], conf_min_logMh[idx_z], conf_max_logMh[idx_z], color="C{}".format(idx_z), alpha=0.3)
        plt.plot(logMs[idx_z], av_logMh[idx_z], label=str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]), color="C{}".format(idx_z))
        # plt.plot(logMs[idx_z], med_logMh[idx_z], label='med '+str(params['redshifts'][idx_z])+'<z<'+str(params['redshifts'][idx_z+1]))
    # plt.plot(logMs[idx_z], logMh(logMs[idx_z], M1, Ms0, beta, delta, gamma), label='best fit z0')

    """PLot the Behroozi SHMR"""
    # log_ms_boo, log_mh_boo = np.load('SHMR_Behroozi_z0.npy')
    # plt.plot(log_ms_boo, log_mh_boo, c='black', linestyle='--', label='Behroozi et al. 2013, z=0.35')
    plt.xlabel('$\mathrm{log}_{10}(M_{*}/M_{\odot})$', size=17)
    plt.ylabel('$\mathrm{log}_{10}(M_{\mathrm{h}}/M_{\odot})$', size=17)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.legend(prop={'size': 12})
    plt.tight_layout()
    #plt.show()
    plt.savefig(directory + '/Plots/SHMR_Allz0_niter=' +
        str(iterations) + '.pdf')

    plt.figure()
    for idx_z in selected_redshifts:
    # for idx_z in [6,7,8,9]:
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
    plt.plot(logspace, 11.8 -logspace, c='black', linestyle='--', label='$M_{*}= 10^{11.8} M_{\odot}$')
    plt.xlabel('$\mathrm{log}_{10}(M_{\mathrm{h}}/M_{\odot})$', size=17)
    plt.ylabel('$\mathrm{log}_{10}(M_{*}/M_{\\mathrm{h}})$', size=17)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.xlim(11.2, 14.6)
    plt.ylim(-2.85, -0.9)
    plt.legend(ncol=2, loc=3)
    plt.tight_layout()
    # plt.show()
    plt.savefig(directory + '/Plots/DeltaSHMR_Allz0_niter=' +
        str(iterations) + '.pdf')

# def plotMsMh_fixedMh(directory):
#     load_smf('cosmos')
#     Ms_min = np.maximum(np.log10(6.3 * 10**7 * (1 + params['redshiftsbin'])**2.7), np.full(params['numzbin'], 9))
#     print(Ms_min)
#     # Arbitrary maximum as read on the plots of the SMF of Davidzon+17
#     Ms_max = 11.8
#     numpoints = 100
#     logMs = np.empty([params['numzbin'], numpoints])
#     for idx_z in range(params['numzbin']):
#         logMs[idx_z] = np.linspace(Ms_min[idx_z], Ms_max, num=numpoints)
#     av_logMh = np.load(directory + '/av_logMh.npy')
#     conf_min_logMh = np.load(directory + '/conf_min_logMh.npy')
#     conf_max_logMh = np.load(directory + '/conf_max_logMh.npy')
#     idx_12 = np.zeros(params['numzbin']).astype('int')
#     idx_13 = np.zeros(params['numzbin']).astype('int')
#     smhm_12 = np.zeros(params['numzbin'])
#     conf_smhm_12 = np.zeros([2, params['numzbin']])
#     smhm_13 = np.zeros(params['numzbin'])
#     conf_smhm_13 = np.zeros([2, params['numzbin']])
#     for idx_z in range(params['numzbin']-2):
#         idx_12[idx_z] = np.argmin(np.abs(av_logMh[idx_z, :] - 12))
#         idx_13[idx_z] = np.argmin(np.abs(av_logMh[idx_z, :] - 13))
#         smhm_12[idx_z] = logMs[idx_z, idx_12[idx_z]] - av_logMh[idx_z, idx_12[idx_z]]
#         smhm_13[idx_z] = logMs[idx_z, idx_13[idx_z]] - av_logMh[idx_z, idx_13[idx_z]]
#         # The error interval on the log of the SMHM ratio is the same as the error on the Halo mass
#         conf_smhm_12[:, idx_z] = [av_logMh[idx_z, idx_12[idx_z]] - conf_min_logMh[idx_z, idx_12[idx_z]],
#             conf_max_logMh[idx_z, idx_12[idx_z]] - av_logMh[idx_z, idx_12[idx_z]]]
#         conf_smhm_13[:, idx_z] = [av_logMh[idx_z, idx_13[idx_z]] - conf_min_logMh[idx_z, idx_13[idx_z]],
#             conf_max_logMh[idx_z, idx_13[idx_z]] - av_logMh[idx_z, idx_13[idx_z]]]
#     plt.figure()
#     plt.errorbar(params['redshiftsbin'][:-2], smhm_12[:-2], yerr=conf_smhm_12[:, :-2], capsize=3, label='$M_{\mathrm{h}} = 10^{12} M_{\odot}$')
#     plt.errorbar(params['redshiftsbin'][:-2], smhm_13[:-2], yerr=conf_smhm_13[:, :-2], capsize=3, label='$M_{\mathrm{h}} = 10^{13} M_{\odot}$')
#     plt.xlabel('Redshift', size=17)
#     plt.ylabel('$\mathrm{log}_{10}(M_{*}/M_{\mathrm{h}})$', size=17)
#     plt.tick_params(axis='both', which='major', labelsize=13)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# def show_nll():
#     chain = np.load("../MCMC_2018-4-25T18-31/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy")
#     lnprob = np.load("../MCMC_2018-4-25T18-31/Chain/LnProb_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy")
#     # plt.figure()
#     for k in range(20):
#         # plt.plot(lnprob[k, :])
#         plt.figure()
#         plt.plot(chain[k, ::1000, 3])
#         plt.title(str(k))
#     # for k in selec_chain:
#     #     plt.plot(chain[k, ::1000, 5])


"""Plots and tests"""


# logMs = np.linspace(6, 12, num=100)
# plt.plot(logMs, logMh(logMs, 13, 10, 0.5, 0.5, 2.5))

# logmhtest =logMh(logMs, 13, 14, 0.5, 0.5, 2.5)
# plt.plot(logMh(logMs, 12, 10, 0.5, 0.5, 2.5), logMs - logMh(logMs, 12, 10, 0.5, 0.5, 2.5))

# Compare Observed and predicted SMF :
# load_smf()
# load_hmf()
# select = np.where(smf[idx_z][:, 1] > -1000)[0]
# logMs = smf[idx_z][select, 0]
# plt.errorbar(logMs, smf[idx_z][select, 1],
#     yerr=[smf[idx_z][select, 3], smf[idx_z][select, 2]], fmt='o')
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

# chain = np.load("../MCMC_2018-4-25T18-31/Chain/Chain_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy")
# lnprob = np.load("../MCMC_2018-4-25T18-31/Chain/LnProb_ksi_z" + str(idx_z) + "_niter=" + str(iterations) + ".npy")
# for k in range(20):
#     plt.plot(lnprob[k, :])
# select = np.where(lnprob[:, -1]>-30)
# chain = chain[lnprob[:, -1]>-30, :, :]
