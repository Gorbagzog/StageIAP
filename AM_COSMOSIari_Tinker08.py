#!/usr/bin/env python3
# -*-coding:Utf-8 -*

"""Abundance Matchnig between COSMOS/Iary and Tinker HMF.

This script also load the SMF of the COSMOS field provided by Iari Davidzon.

Then with the HMF and SMF the script do an Abundance Matching and compute the
Ms/Mh vs Mh relation.

Started in June 2017 by Louis Legrand.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
# import get_Vmax
from astropy.cosmology import LambdaCDM
import get_Vmax_mod

"""Load files"""

redshift_haloes = np.array([0.35, 0.65, 0.95, 1.3, 1.75, 2.25, 2.75, 3.25, 4, 5])
numredshift_haloes = len(redshift_haloes)


"""Load Tinker+08 HMF computed with HFMCalc of Murray+13
parameters : Delta = 70, mean overdensity.
"""


hmf = []
for i in range(numredshift_haloes):
    hmf.append(
        np.loadtxt('../Data/Tinker08HMF/HMFCalc_Dm200/mVector_PLANCK-SMT_z{:1.2f}.txt'.format(
            redshift_haloes[i]), usecols=(0, 7)))
    hmf[i][:, 0] = np.log10(hmf[i][:, 0] / 0.6774)
    hmf[i][:, 1] = hmf[i][:, 1] * (0.6774)**3
    #hmf[i][:, 0] = np.log10(hmf[i][:, 0])

""" Plot"""
#for i in range(numredshift_haloes):
for i in [0]:
    plt.semilogy(hmf[i][:, 0], hmf[i][:, 1], label=redshift_haloes[i])
    plt.ylim(10**-6, 10**-1)
    plt.xlim(9.5, 15)
plt.legend()

""" Compute Halo cumulative density """

numpoints = np.size(hmf[0][:, 0])
Nbolshoi = []
for i in range(numredshift_haloes):
    Nbolshoi.append([])
    for j in range(np.size(hmf[i][:, 0])):
        Nbolshoi[i].append(
            np.trapz(hmf[i][j:, 1], hmf[i][j:, 0]))
for i in range(numredshift_haloes):
    Nbolshoi[i] = np.asarray(Nbolshoi[i])

"""Plots"""

# plt.figure()
# for i in range(numredshift_haloes):
#     plt.plot(hmf[i][:, 0], Nbolshoi[i][:])
# plt.ylim(10**-7, 1)
# plt.xlim(8, 16)
# plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]')
# plt.ylabel('N(>M) [$Mpc^{-3}$]')
# plt.yscale('log')
# plt.title('Abundances for Bolshoï Planck 0<z<9.9')
# plt.show()

"""Load the SMF from Iary Davidzon+17"""

# Code is copied from IaryDavidzonSMF.py as of 12 june
# redshifts of the Iari SMF
redshifts = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
numzbin = np.size(redshifts) - 1

smf = []
for i in range(10):
    smf.append(np.loadtxt(
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
    # VmaxD17 = D17_Cosmo.comoving_volume(redshifts[i+1]) - D17_Cosmo.comoving_volume(redshifts[i])
    # VmaxBP = BP_Cosmo.comoving_volume(redshifts[i+1]) - BP_Cosmo.comoving_volume(redshifts[i])
    # print(i)
    # print(VmaxBP)
    # print(VmaxD17)
    VmaxD17 = get_Vmax_mod.main(redshifts[i], redshifts[i+1], cosmo=[70, 0.3, 0.7])
    VmaxBP = get_Vmax_mod.main(redshifts[i], redshifts[i+1], cosmo=[67.74, 0.3089, 0.6911])
    # Add the log, equivalent to multiply by VmaxD17/VmaxBP
    smf[i][:, 1] = smf[i][:, 1] + np.log10(VmaxD17/VmaxBP)
    smf[i][:, 2] = smf[i][:, 2] + np.log10(VmaxD17/VmaxBP)
    smf[i][:, 3] = smf[i][:, 3] + np.log10(VmaxD17/VmaxBP)

    # Correction of the measured stellar mass
    # Equivalent to multiply by (BP_Cosmo.H0/D17_Cosmo.H0)**-2
    smf[i][:, 0] = smf[i][:, 0] - 2 * np.log10(BP_Cosmo.H0/D17_Cosmo.H0)

"""Plot SMF"""

# plt.figure()
# for i in range(10):
#     plt.fill_between(smf[i][:, 0], smf[i][:, 2], smf[i][:, 3], alpha=0.5,
#                      label=str(redshifts[i])+'<z<'+str(redshifts[i+1]))
#     plt.ylim(-6, -2)
#     plt.xlim(8, 14)
#     plt.title('Davidzon+17 Schechter fits')
#     plt.ylabel('Log($\phi$) [Log($Mpc^{-3}$)]')
#     plt.xlabel('Log($M_{*}$) [Log($M_{\odot}$)]')
#     plt.legend(loc=3)
# plt.show()


"""Compute Galaxy Cumulative Density
"""

# Compute integrals to have the cumulative density = int(phi(m)dm)

numpoints = np.size(smf[0][:, 0])
Nstar = np.empty([numzbin, numpoints])
Nstarminus = np.empty([numzbin, numpoints])
Nstarplus = np.empty([numzbin, numpoints])
for i in range(numzbin):
    for j in range(numpoints):
        Nstar[i, j] = np.trapz(10 ** smf[i][j:, 1], smf[i][j:, 0])
        Nstarminus[i, j] = np.trapz(10 ** smf[i][j:, 2], smf[i][j:, 0])
        Nstarplus[i, j] = np.trapz(10 ** smf[i][j:, 3], smf[i][j:, 0])

"""Do interpolation for abundance matching"""

MstarIary = []
MstarIaryPlus = []
MstarIaryMinus = []
Mhalo = []

for i in range(numzbin):
    """do the interpolation for each redshift bin, in order to have the functions
        StellarMass(abundane) and HaloMass(abundance)"""
    MstarIary.append(interp1d(
        np.log10(Nstar[i, Nstar[i, :] > 0]),
        smf[i][Nstar[i, :] > 0, 0],
        kind='cubic'))
    MstarIaryMinus.append(interp1d(
        np.log10(Nstarminus[i, Nstarminus[i, :] > 0]),
        smf[i][Nstarminus[i, :] > 0, 0],
        kind='cubic'))
    MstarIaryPlus.append(interp1d(
        np.log10(Nstarplus[i, Nstarplus[i, :] > 0]),
        smf[i][Nstarplus[i, :] > 0, 0],
        kind='cubic'))
    Mhalo.append(interp1d(np.log10(
        Nbolshoi[i][Nbolshoi[i] > 0]),
        hmf[i][Nbolshoi[i] > 0, 0],
        kind='cubic'))

"""Compute M*/Mh with uncertainties coming only from the uncertainties on the M*."""

n_fit = 3000
x = np.empty([numzbin, n_fit])  # x is the density variable to trace Ms(x) and Mh(x), in logscale
xm = np.empty([numzbin, n_fit])
ym = np.empty([numzbin, n_fit])
yminus = np.empty([numzbin, n_fit])
yplus = np.empty([numzbin, n_fit])

for i in range(numzbin):
    print('Compute Ms/Mh, z=' + str(redshifts[i]))

    x[i] = np.geomspace(
        max(
            min(np.log10(Nstar[i, Nstar[i, :] > 0])),
            np.log10(Nstarminus[i, Nstarminus[i, :] > 0][-1]),
            np.log10(Nstarplus[i, Nstarplus[i, :] > 0][-1]),
            np.log10(Nbolshoi[i][-2])
            ),
        min(
            np.log10(Nstar[i, 0]),
            np.log10(Nstarminus[i, 0]),
            np.log10(Nstarplus[i, 0]),
            np.log10(Nbolshoi[i][0])
            ),
        n_fit)
    # to ensure that geomspace respects the given boundaries :
    x[i][0] = max(
        min(np.log10(Nstar[i, Nstar[i, :] > 0])),
        np.log10(Nstarminus[i, Nstarminus[i, :] > 0][-1]),
        np.log10(Nstarplus[i, Nstarplus[i, :] > 0][-1]),
        np.log10(Nbolshoi[i][-2])
        )
    x[i][-1] = min(
        np.log10(Nstar[i, 0]),
        np.log10(Nstarminus[i, 0]),
        np.log10(Nstarplus[i, 0]),
        np.log10(Nbolshoi[i][0])
        )
    xm[i] = Mhalo[i](x[i])
    ym[i] = MstarIary[i](x[i]) - Mhalo[i](x[i])  # minus because we are divinding two log
    yminus[i] = MstarIaryMinus[i](x[i]) - Mhalo[i](x[i])
    yplus[i] = MstarIaryPlus[i](x[i]) - Mhalo[i](x[i])

    # ym[i] = np.log10(10**(MstarIary[i](x[i])) / 10**(Mhalo[i](x[i])))
    # yminus[i] = np.log10(10**(MstarIaryMinus[i](x[i])) / 10**(Mhalo[i](x[i])))
    # yplus[i] = np.log10(10**(MstarIaryPlus[i](x[i])) / 10**(Mhalo[i](x[i])))

"""Plot interpolations Ms(N) and Mh(N)"""

# plt.figure()
# for i in range(numzbin):
#     plt.plot(x[i], MstarIary[i](x[i]))
#     plt.fill_between(x[i], MstarIaryMinus[i](x[i]), MstarIaryPlus[i](x[i]))

"""Plot Ms/Mh vs Mh"""

plt.figure()
for i in range(numzbin):
    plt.plot(xm[i][:], ym[i][:], label=str(redshifts[i]) + '<z<' + str(redshifts[i + 1]))
    plt.fill_between(xm[i], yminus[i], yplus[i], alpha=0.5)
plt.legend()
plt.ylabel('$Log(M_{*}/M_{h})$', size=20)
plt.xlabel('Log($M_{h}$)  [Log($M_{\odot}$)]', size=20)
plt.tight_layout()
# plt.title('IariDavidzon Mass Function vs Bolshoï simulation')
plt.show()


# Use the interploation formula of Mlim(z) in Davidzon et al. 2017
Ms_min = np.log10(6.3 * 10**7 * (1 + (redshifts[1:] + redshifts[:-1]) / 2)**2.7)
# Arbitrary maximum as read on the plots of the SMF of Davidzon+17
Ms_max = 11.8

cmap = plt.get_cmap('gist_rainbow')
plt.figure()
for i in range(numzbin):
    index_min = np.argmin(np.abs(MstarIary[i](x[i]) - Ms_max))
    index_max = np.argmin(np.abs(MstarIary[i](x[i]) - Ms_min[i]))
    print(index_min)
    print(index_max)
    plt.fill_between(
        xm[i][index_min:index_max], yminus[i][index_min:index_max],
        yplus[i][index_min:index_max], alpha=0.2, color=cmap(i/numzbin),
        linewidth=0.0)
    plt.plot(
        xm[i][index_min:index_max], ym[i][index_min:index_max],
        label=str(redshifts[i]) + '<z<' + str(redshifts[i + 1]), color=cmap(i/numzbin))
plt.plot(
    np.linspace(12.5, 15), Ms_max - np.linspace(12.5, 15),
    linestyle='--', c='black', label='$M_{*}=10^{'+str(Ms_max)+'}$')
plt.legend()
plt.ylabel('$\mathrm{Log(M_{*}/M_{h})}$', size=20)
plt.xlabel('Log($\mathrm{M_{h}/M_{\odot}}$)', size=20)
plt.tight_layout()
plt.show()


# """Plot Mh/Ms vs Ms"""

# plt.figure()
# for i in range(numzbin):
#     plt.plot(MstarIary[i](x[i]), Mhalo[i](x[i]) - MstarIary[i](x[i]),
#              label=str(redshifts[i]) + '<z<' + str(redshifts[i + 1]))
#     plt.fill_between(MstarIary[i](x[i]), Mhalo[i](x[i]) - MstarIaryMinus[i](x[i]),
#                      Mhalo[i](x[i]) - MstarIaryPlus[i](x[i]), alpha=0.5)
# plt.legend()
# plt.ylabel('$Log(M_{h}/M_{*})$', size=20)
# plt.xlabel('Log($M_{*}$)  [Log($M_{\odot}$)]', size=20)
# plt.tight_layout()
# # plt.title('IariDavidzon Mass Function vs Bolshoï simulation')
# plt.show()

# """Plot Ms vs Mh"""

# plt.figure()
# for i in range(numzbin):
#     plt.plot(xm[i][:], MstarIary[i](x[i]), label=str(redshifts[i]) + '<z<' + str(redshifts[i + 1]))
#     plt.fill_between(xm[i], MstarIaryMinus[i](x[i]), MstarIaryPlus[i](x[i]), alpha=0.5)
# plt.legend()
# plt.ylabel('Log($M_{*}$)  [Log($M_{\odot}$)]', size=20)
# plt.xlabel('Log($M_{h}$)  [Log($M_{\odot}$)]', size=20)
# plt.tight_layout()
# # plt.title('IariDavidzon Mass Function vs Bolshoï simulation')
# plt.show()

# """ Fit a Behroozi+10 law on Mh(Ms)"""


# def boo_MhMs(Ms, M1, Ms0, beta, delta, gamma):
#     """Behroozi et al. 2010 Mh(Ms) function. All masses are in logscale."""
#     return(M1 + beta * (Ms - Ms0) +
#            10 ** (delta * (Ms - Ms0)) / (1 + 10 ** (-gamma * (Ms - Ms0))) - 0.5)


# boo_fit = np.empty([numzbin, 5])
# boo_cov = np.empty([numzbin, 5, 5])
# boo_sigma = np.empty([numzbin, 5])
# for i in range(numzbin):
#     print(i)
#     # For the low redshift cases, it seems that the SMHM for SM>10**12 causes
#     # problems for the fit, so we will cut at 10**12 for the first five bins.
#     if i < 5:
#         stop = np.argmin(np.abs(MstarIary[i](x[i]) - 12))
#         # stop=0
#     else:
#         stop = 0
#     boo_fit[i], boo_cov[i] = curve_fit(boo_MhMs, MstarIary[i](x[i][stop:]),
#                                        Mhalo[i](x[i][stop:]),
#                                        bounds=[[10, 8, 0, 0, 0], [15, 14, 5, 5, 5]])
#     boo_sigma[i] = np.sqrt(np.diag(boo_cov[i]))

# """Plot"""

# # Plot Mh(Ms)
# for i in range(numzbin):
#     plt.figure()
#     ax = plt.subplot(111)
#     plt.plot(
#         MstarIary[i](x[i]),
#         xm[i][:],
#         label='COSMOS and Bolshoï AM, ' + str(redshifts[i])+'<z<'+str(redshifts[i+1]))
#     if i < 5:
#         cut = 'cut at Log(Ms)<12'
#     else:
#         cut = 'no cut'
#     plt.plot(
#         MstarIary[i](x[i]),
#         boo_MhMs(MstarIary[i](x[i]), *boo_fit[i]),
#         label=str('Behroozi function fit, '+cut), c='r'
#         )
#     plt.legend()
#     plt.xlabel('Log($M_{*}$)  [Log($M_{\odot}$)]', size=20)
#     plt.ylabel('Log($M_{h}$)  [Log($M_{\odot}$)]', size=20)
#     plt.text(
#         0.6, 0.1,
#         '''
#         $log(M_{{1}})=${:.2f}
#         $log(M_{{*,0}})=${:.2f}
#         $\\beta=${:.2f}
#         $\delta=${:.2f}
#         $\gamma={:.2f}$'''.format(*boo_fit[i]),
#         transform=ax.transAxes
#         )
#     plt.show()
#     plt.tight_layout()
#     # plt.savefig('../Plots/COSMOSBolshoi_AM/Behroozi+10_fits/MsMh_Bfit_cut_z=' +
#     #             str(redshifts[i])+'-'+str(redshifts[i+1])+'.pdf')

# # Plot Ms(Mh)
# for i in range(numzbin):
#     plt.figure()
#     ax = plt.subplot(111)
#     plt.plot(
#         xm[i][:],
#         MstarIary[i](x[i]),
#         label='COSMOS and Bolshoï AM, ' + str(redshifts[i])+'<z<'+str(redshifts[i+1])
#         )
#     plt.fill_between(
#         xm[i],
#         MstarIaryMinus[i](x[i]),
#         MstarIaryPlus[i](x[i]),
#         alpha=0.5
#         )
#     if i < 5:
#         cut = 'cuted at Log(Ms)<12'
#     else:
#         cut = 'no cut'
#     plt.plot(
#         boo_MhMs(MstarIary[i](x[i]), *boo_fit[i]),
#         MstarIary[i](x[i]),
#         label=str('Behroozi function fit, '+cut), c='r')
#     plt.legend()
#     plt.ylabel('Log($M_{*}$)  [Log($M_{\odot}$)]', size=20)
#     plt.xlabel('Log($M_{h}$)  [Log($M_{\odot}$)]', size=20)
#     plt.text(
#         0.6, 0.1,
#         '''
#         $log(M_{{1}})=${:.2f}
#         $log(M_{{*,0}})=${:.2f}
#         $\\beta=${:.2f}
#         $\delta=${:.2f}
#         $\gamma={:.2f}$'''.format(*boo_fit[i]),
#         transform=ax.transAxes)
#     plt.show()
#     plt.tight_layout()
#     # plt.savefig('../Plots/COSMOSBolshoi_AM/Behroozi+10_fits/MsMh_Bfit_cut_z=' +
#     #             str(redshifts[i])+'-'+str(redshifts[i+1])+'.pdf')


# """Plot the evolution of the Behroozi+10 parameters as a function of redshift"""

# plt.figure()
# plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, boo_fit[:, 0], yerr=boo_sigma[:, 0],
#              fmt='o', capsize=5)
# plt.xlabel('Redshift', size=20)
# plt.ylabel('$M_{1}$ of Behroozi+10 fit', size=20)
# plt.tight_layout()
# plt.show()

# plt.figure()
# plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, boo_fit[:, 1], yerr=boo_sigma[:, 1],
#              fmt='o', capsize=5)
# plt.xlabel('Redshift', size=20)
# plt.ylabel('$M_{*,0}$ of Behroozi+10 fit', size=20)
# plt.tight_layout()
# plt.show()

# plt.figure()
# plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, boo_fit[:, 2], yerr=boo_sigma[:, 2],
#              fmt='o', capsize=5)
# plt.xlabel('Redshift', size=20)
# plt.ylabel('$\\beta$ of Behroozi+10 fit', size=20)
# plt.tight_layout()
# plt.show()

# plt.figure()
# plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, boo_fit[:, 3], yerr=boo_sigma[:, 3],
#              fmt='o', capsize=5)
# plt.xlabel('Redshift', size=20)
# plt.ylabel('$\\delta$ of Behroozi+10 fit', size=20)
# plt.tight_layout()
# plt.show()

# plt.figure()
# plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, boo_fit[:, 4], yerr=boo_sigma[:, 4],
#              fmt='o', capsize=5)
# plt.xlabel('Redshift', size=20)
# plt.ylabel('$\\gamma$ of Behroozi+10 fit', size=20)
# plt.tight_layout()
# plt.show()

"""Find Mpeak and its uncertainties"""

MhaloPeak = np.empty(numzbin)
MsOnMhaloPeak = np.empty(numzbin)
MhaloPeakMin = np.empty(numzbin)
MhaloPeakMax = np.empty(numzbin)
index_halopeak = np.empty(numzbin, dtype='int')
for i in range(numzbin):
    index_halopeak[i] = np.argmax(ym[i])
    MsOnMhaloPeak[i] = ym[i][index_halopeak[i]]
    MhaloPeak[i] = xm[i][index_halopeak[i]]
    index_min = np.argmin(
        np.abs(yplus[i][index_halopeak[i]:] - MsOnMhaloPeak[i])) + index_halopeak[i]
    index_max = np.argmin(np.abs(yplus[i][:index_halopeak[i]] - MsOnMhaloPeak[i]))
    MhaloPeakMin[i] = xm[i][index_min]
    MhaloPeakMax[i] = xm[i][index_max]

MstarPeak = np.empty(numzbin)
MhOnMsPeak = np.empty(numzbin)
MstarPeakMin = np.empty(numzbin)
MstarPeakMax = np.empty(numzbin)
index_starpeak = np.empty(numzbin, dtype='int')
for i in range(numzbin):
    index_starpeak[i] = np.argmin(np.abs(Mhalo[i](x[i]) - MstarIary[i](x[i])))
    MhOnMsPeak[i] = Mhalo[i](x[i][index_starpeak[i]]) - MstarIary[i](x[i][index_starpeak[i]])
    MstarPeak[i] = MstarIary[i](x[i][index_starpeak[i]])
    # Test taking the MstarIaryMinus and MstarIaryPlus as the uncertainties for MstarPeak
    MstarPeakMin[i] = MstarIaryMinus[i](x[i][index_starpeak[i]])
    MstarPeakMax[i] = MstarIaryPlus[i](x[i][index_starpeak[i]])

"""Use the plot Ms(Mh) to compute uncertainties on MhaloPeak.
For a given MhaloPeak, we take the value of Mstar on the Ms(Mh) plot and
take the Mh dispersion for this value."""

MhaloPeakSigma = np.empty([numzbin, 2])
for i in range(numzbin):
    tmp_mstar = MstarIary[i](x[i][index_halopeak[i]])
    ind_Mstarplus = np.argmin(np.abs(MstarIaryPlus[i](x[i]) - tmp_mstar))
    ind_Mstarminus = np.argmin(np.abs(MstarIaryMinus[i](x[i]) - tmp_mstar))
    MhaloPeakSigma[i, 0] = MhaloPeak[i] - Mhalo[i](x[i][ind_Mstarplus])
    MhaloPeakSigma[i, 1] = Mhalo[i](x[i][ind_Mstarminus]) - MhaloPeak[i]

# plt.figure()
# plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, MhaloPeak,
#              yerr=np.transpose(MhaloPeakSigma),
#              fmt='o', capsize=5, label='This work')

"""Definition of the evolution of Mpeak for Leauthaud et al, Behroozi et al et Moster et al"""

# TODO : check that there aure all in the same cosmo (h=0.7)

# Lautahud+17 use a different cosmology with H0=72
redshiftLeauthaud = np.array([(0.22 + 0.48) / 2, (0.48 + 0.74) / 2, (0.74 + 1) / 2])
MhaloPeakLeauthaud = np.log10(np.array([9.5 * 10**11, 1.45 * 10**12, 1.4 * 10**12]))
MhaloSigmaLeauthaud = np.log10(np.array(
    [1.05 * 10**12, 1.55 * 10**12, 1.5 * 10**12])) - MhaloPeakLeauthaud
MstarPeakLeauthaud = np.array([3.55 * 10**10, 4.9 * 10**10, 5.75 * 10**10])
MstarSigmaLeauthaud = np.array([0.17, 0.15, 0.13])*10**10

redshiftBehroozi = np.array([])

redshiftCoupon15 = np.array([0.75])
MhaloPeakCoupon15 = np.log10(np.array([1.92*10**12]))
MhaloSigmaCoupon15 = np.array([[np.log10((1.92 + 0.17)) - np.log10(1.92)],
                               [np.log10(1.92) - np.log10(1.92 - 0.14)]])

redshiftCoupon12 = np.array([0.3, 0.5, 0.7, 1])
MhaloPeakCoupon12 = np.array([11.65, 11.79, 11.88, 11.9]) - np.log10(0.7)
MhaloSigmaCoupon12 = np.vstack([[0.05, 0.05, 0.06, 0.07], [0.05, 0.05, 0.06, 0.07]])

redshiftMartinezManso2014 = 1.5
MhaloPeakMartinezManso2014 = 12.44
MhaloSigmaMartinezManso2014 = [[0.08], [0.08]]

# Graphic read of Coupon draft
# redshiftCoupon17 = [0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 1.8, 2.1, 2.8, 3.4, 4, 4.8]
# MhaloPeakCoupon17 = [12.2, 12.5, 12.3, 12.25, 12.35, 12.3, 12.4, 12.45, 12.45, 11.95, 12.2, 12.25]
# MhaloSigmaCoupon17 = [0, 0.05, 0.05, 0.05, 0, 0, 0.05, 0, 0.1, 0.1, 0.3, 0.25]

# Test graphic reading of McCracken+15 Mpeak
redshiftMcCracken15 = np.array([0.65, 0.95, 1.3, 1.75, 2.25])
MhaloPeakMcCracken15 = np.array([12.15, 12.1, 12.2, 12.35, 12.4])

# Load Coupon+17 draft Peak values
# We use PeakPosMCMCMean and PeakPosMCMCstd
# Values are given in Log10(Mh*h^-1 Msun)
redshiftCoupon17 = np.array([0.34, 0.52, 0.70, 0.90, 1.17, 1.50,
                            1.77, 2.15, 2.75, 3.37, 3.96, 4.83])
MhaloPeakCoupon17 = np.zeros([np.size(redshiftCoupon17)])
MhaloSigmaCoupon17 = np.zeros([np.size(redshiftCoupon17)])
for i in range(len(redshiftCoupon17)):
    MhaloPeakCoupon17[i], MhaloSigmaCoupon17[i] = np.loadtxt(
        '../Data/Coupon17/peak/peak_{:1.2f}.ascii'.format(redshiftCoupon17[i]),
        usecols=(2, 3))

"""Save MhPeak(z)"""

# np.savetxt(
#     "../Plots/MhPeak/Tinker08_Dm200.txt",
#     np.transpose(np.stack(((redshifts[1:] + redshifts[:-1]) / 2, MhaloPeak + np.log10(67.74/70),
#              MhaloPeakSigma[:, 0], MhaloPeakSigma[:, 1]))),
#     header='z   MhaloPeak   MhaloPeakSigma'
#     )

"""Plot"""

# Verify that the Mpeak is at the correct position for Ms/Mh vs Mh
# plt.figure()
# for i in range(numzbin):
#     plt.plot(xm[i][:], ym[i][:], label=str(redshifts[i]) + '<z<' + str(redshifts[i + 1]))
#     plt.fill_between(xm[i], yminus[i], yplus[i], alpha=0.5)
#     plt.scatter(MhaloPeak[i], MsOnMhaloPeak[i])
#     plt.plot((MhaloPeakMin[i], MhaloPeakMax[i]), (MsOnMhaloPeak[i], MsOnMhaloPeak[i]))
# # plt.legend()
# # plt.ylabel('$M_{*}/M_{h}$', size=20)
# # plt.xlabel('Log($M_{h}$)  [Log($M_{\odot}$)]', size=20)
# # plt.tight_layout()
# # plt.title('IariDavidzon Mass Function vs Bolshoï simulation')
# plt.show()

# Plot the MhaloPeak vs z evolution on a linear scale.
plt.figure()
# plt.yscale('log')
# plt.xscale('log')
plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, MhaloPeak + np.log10(67.74/70),
             yerr=np.transpose(MhaloPeakSigma),
             fmt='o', color='red', capsize=5, label='This work')
plt.errorbar(redshiftLeauthaud, MhaloPeakLeauthaud + np.log10(72/70),
             yerr=MhaloSigmaLeauthaud,
             fmt='o', capsize=5, label='Leauthaud et al. 2011')
plt.errorbar(redshiftCoupon12, MhaloPeakCoupon12, yerr=MhaloSigmaCoupon12,
             fmt='o', capsize=5, label='Coupon et al. 2012')
plt.errorbar(redshiftCoupon15, MhaloPeakCoupon15, yerr=MhaloSigmaCoupon15,
             fmt='o', capsize=5, label='Coupon et al. 2015')
# plt.errorbar(redshiftMartinezManso2014, MhaloPeakMartinezManso2014,
#              yerr=MhaloSigmaMartinezManso2014,
#              fmt='o', capsize=5, label='Martinez-Manso et al. 2014')
plt.errorbar(redshiftCoupon17, MhaloPeakCoupon17 - np.log10(0.7),
             yerr=MhaloSigmaCoupon17,
             fmt='o', color='blue', capsize=5, label='Coupon et al. 2017 Draft')

# plt.errorbar(redshiftMcCracken15, MhaloPeakMcCracken15,
#              fmt='o', capsize=5, label='"Revised" McCracken15')
plt.xlabel('Reshift', size=20)
plt.ylabel('Log($M_{halo}^{peak}$) [Log($M_{\odot}$)]', size=20)
plt.legend()
plt.tight_layout()
plt.show()

# # Verify that the Mpeak is at the correct position for Mh/Ms vs Ms
# plt.figure()
# for i in range(numzbin):
#     plt.plot(MstarIary[i](x[i]), Mhalo[i](x[i]) - MstarIary[i](x[i]),
#              label=str(redshifts[i]) + '<z<' + str(redshifts[i + 1]))
#     plt.fill_between(MstarIary[i](x[i]), Mhalo[i](x[i]) - MstarIaryMinus[i](x[i]),
#                      Mhalo[i](x[i]) - MstarIaryPlus[i](x[i]), alpha=0.5)
#     plt.scatter(MstarPeak[i], MhOnMsPeak[i])
#     plt.plot((MstarPeakMin[i], MstarPeakMax[i]), (MhOnMsPeak[i], MhOnMsPeak[i]))
# plt.legend()
# plt.ylabel('$M_{h}/M_{*}$', size=20)
# plt.xlabel('Log($M_{*}$)  [Log($M_{\odot}$)]', size=20)
# plt.tight_layout()
# # plt.title('IariDavidzon Mass Function vs Bolshoï simulation')
# plt.show()

# # Plot the MstarPeak vs z evolution
# plt.figure()
# plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, MstarPeak,
#              yerr=[MstarPeak-MstarPeakMin, MstarPeakMax-MstarPeak], fmt='o', capsize=5)
# plt.xlabel('Reshift', size=20)
# plt.ylabel('Log($M_{star}^{peak}$) [Log($M_{\odot}$)]', size=20)
# plt.tight_layout()
# plt.show()

# # PLot MstarPeak vs z in a linear scale
# plt.figure()
# plt.yscale('log')
# plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, 10**MstarPeak,
#              yerr=[10**MstarPeak-10**MstarPeakMin, 10**MstarPeakMax-10**MstarPeak],
#              fmt='o', capsize=5, label='This work')
# plt.errorbar(redshiftLeauthaud, MstarPeakLeauthaud, yerr=MstarSigmaLeauthaud,
#              fmt='o', capsize=5, label='Leauthaud et al. 2011')
# plt.xlabel('Reshift', size=20)
# plt.ylabel('Log($M_{*}^{peak}$) [Log($M_{\odot}$)]', size=20)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Plot the evolution of Ms/Mh(peak) vs z
# yerr = [[ym[i][index_halopeak[i]]-yminus[i][index_halopeak[i]],
#         yplus[i][index_halopeak[i]] - ym[i][index_halopeak[i]]] for i in range(10)]
# plt.figure()
# plt.errorbar(
#     (redshifts[1:] + redshifts[:-1]) / 2,
#     MsOnMhaloPeak,
#     yerr=np.transpose(yerr),
#     fmt='o')
# plt.xlabel('Reshift', size=20)
# plt.ylabel('Log($[M_{*}/M_{h}]^{peak}$)', size=20)
# plt.tight_layout()
# plt.show()

# """Fit a Yang law on Ms/Mh"""


# def mstar_over_mh_yang(x, A, m1, beta, gamma):
#     """Yang et al. 2004 function, see Moster et al. 2010. All masses are in logscale."""
#     return 2.0 * A / (10 ** ((x - m1) * (- beta)) + 10 ** ((x - m1) * gamma))


# n_fit = 1000
# yang_fit = np.empty([numzbin, 4])
# yang_cov = np.empty([numzbin, 4, 4])
# yang_sigma = np.empty([numzbin, 4])

# for i in range(numzbin):
#     print('Fiting a Yang law, z=' + str(i))
#     yang_fit[i], yang_cov[i] = curve_fit(mstar_over_mh_yang,
#                                          Mhalo[i](x[i]),
#                                          10**(MstarIary[i](x[i]) - Mhalo[i](x[i])),
#                                          # uncertainty on the data :
#                                          sigma=10**(yplus[i] - yminus[i]),
#                                          bounds=[[-np.inf, 8, 0, 0], [np.inf, 14, 5, 5]])
#     yang_sigma[i] = np.sqrt(np.diag(yang_cov[i]))

# """Plot"""

# # Plot the Yang fit on the Ms/Mh ratio
# for i in range(numzbin):
#     plt.figure()
#     plt.plot(Mhalo[i](x[i]), MstarIary[i](x[i]) - Mhalo[i](x[i]))
#     plt.plot(Mhalo[i](x[i]), np.log10(mstar_over_mh_yang(Mhalo[i](x[i]), *yang_fit[i])), label='')
#     plt.fill_between(xm[i], yminus[i], yplus[i], alpha=0.5)
# # plt.legend()
# plt.xlabel('xlabel', size=20)
# plt.ylabel('ylabel', size=20)
# plt.tight_layout()
# plt.show()

# # PLot the evolution of the parameters of the Yang Fit

# plt.figure()
# plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, yang_fit[:, 2], yerr=yang_sigma[:, 2],
#              fmt='o', capsize=5)
# plt.xlabel('Redshift', size=20)
# plt.ylabel('$\\beta$ of Yang fit', size=20)
# plt.tight_layout()
# plt.show()

# plt.figure()
# plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, yang_fit[:, 3], yerr=yang_sigma[:, 3],
#              fmt='o', capsize=5)
# plt.xlabel('Redshift', size=20)
# plt.ylabel('$\\gamma$ of Yang fit', size=20)
# plt.tight_layout()
# plt.show()

# plt.figure()
# plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, yang_fit[:, 1], yerr=yang_sigma[:, 1],
#              fmt='o', capsize=5)
# plt.xlabel('Redshift', size=20)
# plt.ylabel('$M_{1}$ of Yang fit', size=20)
# plt.tight_layout()
# plt.show()
