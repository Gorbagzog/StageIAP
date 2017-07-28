#!/usr/bin/env python3
# -*-coding:Utf-8 -*

"""Abundance Matchnig between COSMOS/Iary and BloshoïPlanck.

Script used to load and display the halo mass functions using Mvir
of the new Bolshoï simulation using Planck15 cosmology :
h=0.6774, s8=0.8159, Om=0.3089, Ob=0.0486, ns=0.9667.
HaloMassFunctions are provided by Peter Behroozi.

This script also load the SMF of the COSMOS field provided by Iari Davidzon.

Then with the HMF and SMF the script do an Abundance Matching and compute the
Ms/Mh vs Mh relation.

Started in June 2017 by Louis Legrand.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


"""Load files"""
# redshifts of the BolshoiPlanck files
redshift_haloes = np.arange(0, 10, step=0.1)
numredshift_haloes = np.size(redshift_haloes)
smf_bolshoi = []

for i in range(numredshift_haloes):
    smf_bolshoi.append(
        np.loadtxt('../Data/HMFBolshoiPlanck/mf_planck/mf_planck_z' +
                   '{:4.3f}'.format(redshift_haloes[i]) + '_mvir.dat'))

# smf_bolshoi[redshift][:,0] = Log10(mass) [Msun]
# smf_bolshoi[redshift][:,1] = Log10(cen_mf), ie central haloes mass function
# (density) [1/Mpc^3]
# smf_bolshoi[redshift][:,2] = Log10(all_macc_mf), ie all haloes mass function
# (density) [1/Mpc^3]


""" Plot"""
# for i in range(numredshift_haloes):
#     plt.plot(smf_bolshoi[i][:,0], smf_bolshoi[i][:,1])

""" Compute Halo cumulative density """

numpoints = np.size(smf_bolshoi[0][:, 0])
Nbolshoi = []

for i in range(numredshift_haloes):
    Nbolshoi.append([])
    for j in range(np.size(smf_bolshoi[i][:, 0])):
        Nbolshoi[i].append(
            np.trapz(10 ** smf_bolshoi[i][j:, 1], smf_bolshoi[i][j:, 0]))
for i in range(numredshift_haloes):
    Nbolshoi[i] = np.asarray(Nbolshoi[i])

"""Plots"""

# for i in range(numredshift_haloes):
#     plt.plot(smf_bolshoi[i][:,0], Nbolshoi[i][:])
# plt.ylim(10**-7, 1)
# plt.xlim(8,16)
# plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]')
# plt.ylabel('N(>M) [$Mpc^{-3}$]')
# plt.yscale('log')
# plt.title('Abundances for Bolshoï Planck 0<z<9.9')


"""Load the SMF from Iary Davidzon+17"""

# Code is copied from IaryDavidzonSMF.py as of 12 june
# redshifts of the Iari SMF
redshifts = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
numzbin = np.size(redshifts) - 1

smf = []
for i in range(10):
    smf.append(np.loadtxt(
        '../Data/Davidzon/Davidzon+17_SMF_V3.0/mf_mass2b_fl5b_tot_VmaxFit2D'
        + str(i) + '.dat'))

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

"""Select redshifts of haloes to match with Davidzon intervals"""

redshift_id_selec = np.empty(numzbin)
for i in range(numzbin):
    redshift_id_selec[i] = np.argmin(
        np.abs(redshift_haloes - (redshifts[i] + redshifts[i + 1]) / 2))

redshift_id_selec = redshift_id_selec.astype(int)
print('Redshifts of Iari SMFs : ' + str((redshifts[:-1] + redshifts[1:]) / 2))
print('Closest redshifts for Bolshoi HMFs : '
      + str(redshift_haloes[redshift_id_selec]))

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
        Nbolshoi[redshift_id_selec[i]][Nbolshoi[redshift_id_selec[i]] > 0]),
        smf_bolshoi[redshift_id_selec[i]][Nbolshoi[redshift_id_selec[i]] > 0, 0],
        kind='cubic'))

"""Compute M*/Mh with uncertainties coming only from the uncertainties on the m"""

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
            np.log10(Nbolshoi[redshift_id_selec[i]][-1])
            ),
        min(
            np.log10(Nstar[i, 0]),
            np.log10(Nstarminus[i, 0]),
            np.log10(Nstarplus[i, 0]),
            np.log10(Nbolshoi[redshift_id_selec[i]][0])
            ),
        n_fit)
    # to ensure that geomspace respects the given boundaries :
    x[i][0] = max(
        min(np.log10(Nstar[i, Nstar[i, :] > 0])),
        np.log10(Nstarminus[i, Nstarminus[i, :] > 0][-1]),
        np.log10(Nstarplus[i, Nstarplus[i, :] > 0][-1]),
        np.log10(Nbolshoi[redshift_id_selec[i]][-1])
        )
    x[i][-1] = min(
        np.log10(Nstar[i, 0]),
        np.log10(Nstarminus[i, 0]),
        np.log10(Nstarplus[i, 0]),
        np.log10(Nbolshoi[redshift_id_selec[i]][0])
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

"""Plot Ms/Mh vs Mh"""

plt.figure()
for i in range(numzbin - 2):
    plt.plot(xm[i][:], ym[i][:], label=str(redshifts[i]) + '<z<' + str(redshifts[i + 1]))
    plt.fill_between(xm[i], yminus[i], yplus[i], alpha=0.5)
plt.legend()
plt.ylabel('$Log(M_{*}/M_{h})$', size=20)
plt.xlabel('Log($M_{h}$)  [Log($M_{\odot}$)]', size=20)
plt.tight_layout()
# plt.title('IariDavidzon Mass Function vs Bolshoï simulation')
plt.show()

"""Plot Mh/Ms vs Ms"""

plt.figure()
for i in range(numzbin):
    plt.plot(MstarIary[i](x[i]), Mhalo[i](x[i]) - MstarIary[i](x[i]),
             label=str(redshifts[i]) + '<z<' + str(redshifts[i + 1]))
    plt.fill_between(MstarIary[i](x[i]), Mhalo[i](x[i]) - MstarIaryMinus[i](x[i]),
                     Mhalo[i](x[i]) - MstarIaryPlus[i](x[i]), alpha=0.5)
plt.legend()
plt.ylabel('$Log(M_{h}/M_{*})$', size=20)
plt.xlabel('Log($M_{*}$)  [Log($M_{\odot}$)]', size=20)
plt.tight_layout()
# plt.title('IariDavidzon Mass Function vs Bolshoï simulation')
plt.show()

"""Plot Ms vs Mh"""

plt.figure()
for i in range(numzbin):
    plt.plot(xm[i][:], MstarIary[i](x[i]), label=str(redshifts[i]) + '<z<' + str(redshifts[i + 1]))
    plt.fill_between(xm[i], MstarIaryMinus[i](x[i]), MstarIaryPlus[i](x[i]), alpha=0.5)
plt.legend()
plt.ylabel('Log($M_{*}$)  [Log($M_{\odot}$)]', size=20)
plt.xlabel('Log($M_{h}$)  [Log($M_{\odot}$)]', size=20)
plt.tight_layout()
# plt.title('IariDavidzon Mass Function vs Bolshoï simulation')
plt.show()

""" Fit a Behroozi+10 law on Mh(Ms)"""


def boo_MhMs(Ms, M1, Ms0, beta, delta, gamma):
    """Behroozi et al. 2010 Mh(Ms) function. All masses are in logscale."""
    return(M1 + beta * (Ms - Ms0) +
           10 ** (delta * (Ms - Ms0)) / (1 + 10 ** (-gamma * (Ms - Ms0))) - 0.5)


boo_fit = np.empty([numzbin, 5])
boo_cov = np.empty([numzbin, 5, 5])
boo_sigma = np.empty([numzbin, 5])
for i in range(numzbin):
    print(i)
    # For the low redshift cases, it seems that the SMHM for SM>10**12 causes
    # problems for the fit, so we will cut at 10**12 for the first five bins.
    if i < 5:
        stop = np.argmin(np.abs(MstarIary[i](x[i]) - 12))
        # stop=0
    else:
        stop = 0
    boo_fit[i], boo_cov[i] = curve_fit(boo_MhMs, MstarIary[i](x[i][stop:]),
                                       Mhalo[i](x[i][stop:]),
                                       bounds=[[10, 8, 0, 0, 0], [15, 14, 5, 5, 5]])
    boo_sigma[i] = np.sqrt(np.diag(boo_cov[i]))

"""Plot"""

# Plot Mh(Ms)
for i in range(numzbin):
    plt.figure()
    ax = plt.subplot(111)
    plt.plot(
        MstarIary[i](x[i]),
        xm[i][:],
        label='COSMOS and Bolshoï AM, ' + str(redshifts[i])+'<z<'+str(redshifts[i+1]))
    if i < 5:
        cut = 'cut at Log(Ms)<12'
    else:
        cut = 'no cut'
    plt.plot(
        MstarIary[i](x[i]),
        boo_MhMs(MstarIary[i](x[i]), *boo_fit[i]),
        label=str('Behroozi function fit, '+cut), c='r'
        )
    plt.legend()
    plt.xlabel('Log($M_{*}$)  [Log($M_{\odot}$)]', size=20)
    plt.ylabel('Log($M_{h}$)  [Log($M_{\odot}$)]', size=20)
    plt.text(
        0.6, 0.1,
        '''
        $log(M_{{1}})=${:.2f}
        $log(M_{{*,0}})=${:.2f}
        $\\beta=${:.2f}
        $\delta=${:.2f}
        $\gamma={:.2f}$'''.format(*boo_fit[i]),
        transform=ax.transAxes
        )
    plt.show()
    plt.tight_layout()
    # plt.savefig('../Plots/COSMOSBolshoi_AM/Behroozi+10_fits/MsMh_Bfit_cut_z=' +
    #             str(redshifts[i])+'-'+str(redshifts[i+1])+'.pdf')

# Plot Ms(Mh)
for i in range(numzbin):
    plt.figure()
    ax = plt.subplot(111)
    plt.plot(
        xm[i][:],
        MstarIary[i](x[i]),
        label='COSMOS and Bolshoï AM, ' + str(redshifts[i])+'<z<'+str(redshifts[i+1])
        )
    plt.fill_between(
        xm[i],
        MstarIaryMinus[i](x[i]),
        MstarIaryPlus[i](x[i]),
        alpha=0.5
        )
    if i < 5:
        cut = 'cuted at Log(Ms)<12'
    else:
        cut = 'no cut'
    plt.plot(
        boo_MhMs(MstarIary[i](x[i]), *boo_fit[i]),
        MstarIary[i](x[i]),
        label=str('Behroozi function fit, '+cut), c='r')
    plt.legend()
    plt.ylabel('Log($M_{*}$)  [Log($M_{\odot}$)]', size=20)
    plt.xlabel('Log($M_{h}$)  [Log($M_{\odot}$)]', size=20)
    plt.text(
        0.6, 0.1,
        '''
        $log(M_{{1}})=${:.2f}
        $log(M_{{*,0}})=${:.2f}
        $\\beta=${:.2f}
        $\delta=${:.2f}
        $\gamma={:.2f}$'''.format(*boo_fit[i]),
        transform=ax.transAxes)
    plt.show()
    plt.tight_layout()
    # plt.savefig('../Plots/COSMOSBolshoi_AM/Behroozi+10_fits/MsMh_Bfit_cut_z=' +
    #             str(redshifts[i])+'-'+str(redshifts[i+1])+'.pdf')


"""Plot the evolution of the Behroozi+10 parameters as a function of redshift"""

plt.figure()
plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, boo_fit[:, 0], yerr=boo_sigma[:, 0],
             fmt='o', capsize=5)
plt.xlabel('Redshift', size=20)
plt.ylabel('$M_{1}$ of Behroozi+10 fit', size=20)
plt.tight_layout()
plt.show()

plt.figure()
plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, boo_fit[:, 1], yerr=boo_sigma[:, 1],
             fmt='o', capsize=5)
plt.xlabel('Redshift', size=20)
plt.ylabel('$M_{*,0}$ of Behroozi+10 fit', size=20)
plt.tight_layout()
plt.show()

plt.figure()
plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, boo_fit[:, 2], yerr=boo_sigma[:, 2],
             fmt='o', capsize=5)
plt.xlabel('Redshift', size=20)
plt.ylabel('$\\beta$ of Behroozi+10 fit', size=20)
plt.tight_layout()
plt.show()

plt.figure()
plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, boo_fit[:, 3], yerr=boo_sigma[:, 3],
             fmt='o', capsize=5)
plt.xlabel('Redshift', size=20)
plt.ylabel('$\\delta$ of Behroozi+10 fit', size=20)
plt.tight_layout()
plt.show()

plt.figure()
plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, boo_fit[:, 4], yerr=boo_sigma[:, 4],
             fmt='o', capsize=5)
plt.xlabel('Redshift', size=20)
plt.ylabel('$\\gamma$ of Behroozi+10 fit', size=20)
plt.tight_layout()
plt.show()

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
    index_starpeak[i] = np.argmin(Mhalo[i](x[i]) - MstarIary[i](x[i]))
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
    MhaloPeakSigma[i, 0] = np.log10(10**MhaloPeak[i] - 10**Mhalo[i](x[i][ind_Mstarplus]))
    MhaloPeakSigma[i, 1] = np.log10(10**Mhalo[i](x[i][ind_Mstarminus]) - 10**MhaloPeak[i])

plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, 10**MhaloPeak,
             yerr=np.transpose(10**MhaloPeakSigma),
             fmt='o', capsize=5, label='This work')

"""Definition of the evolution of Mpeak for Leauthaud et al, Behroozi et al et Moster et al"""

redshiftLeauthaud = np.array([(0.22 + 0.48) / 2, (0.48 + 0.74) / 2, (0.74 + 1) / 2])
MhaloPeakLeauthaud = np.array([9.5 * 10**11, 1.45 * 10**12, 1.4 * 10**12])
MhaloSigmaLeauthaud = np.array([0.5 * 10**11, 10**11, 10**11])


"""Plot"""

# Verify that the Mpeak is at the correct position for Ms/Mh vs Mh
plt.figure()
for i in range(numzbin):
    plt.plot(xm[i][:], ym[i][:], label=str(redshifts[i]) + '<z<' + str(redshifts[i + 1]))
    plt.fill_between(xm[i], yminus[i], yplus[i], alpha=0.5)
    plt.scatter(MhaloPeak[i], MsOnMhaloPeak[i])
    plt.plot((MhaloPeakMin[i], MhaloPeakMax[i]), (MsOnMhaloPeak[i], MsOnMhaloPeak[i]))
# plt.legend()
# plt.ylabel('$M_{*}/M_{h}$', size=20)
# plt.xlabel('Log($M_{h}$)  [Log($M_{\odot}$)]', size=20)
# plt.tight_layout()
# plt.title('IariDavidzon Mass Function vs Bolshoï simulation')
plt.show()

# Plot the MhaloPeak vs z evolution, old computation of z
plt.figure()
plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, MhaloPeak,
             yerr=[MhaloPeak-MhaloPeakMin, MhaloPeakMax-MhaloPeak],
             fmt='o', capsize=5, label='This work')
plt.errorbar(redshiftLeauthaud, MhaloPeakLeauthaud, yerr=MhaloSigmaLeauthaud,
             fmt='o', capsize=5, label='Leauthaud et al. 2011')
plt.xlabel('Reshift', size=20)
plt.ylabel('Log($M_{halo}^{peak}$) [Log($M_{\odot}$)]', size=20)
plt.tight_layout()
plt.show()

# Plot the MhaloPeak vs z evolution on a linear scale.
plt.figure()
plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, 10**MhaloPeak,
             yerr=[10**MhaloPeak - 10**MhaloPeakMin, 10**MhaloPeakMax - 10**MhaloPeak],
             fmt='o', capsize=5, label='This work')
plt.errorbar(redshiftLeauthaud, MhaloPeakLeauthaud, yerr=MhaloSigmaLeauthaud,
             fmt='o', capsize=5, label='Leauthaud et al. 2011')
plt.xlabel('Reshift', size=20)
plt.ylabel('Log($M_{halo}^{peak}$) [Log($M_{\odot}$)]', size=20)
plt.tight_layout()
plt.show()

# Verify that the Mpeak is at the correct position for Mh/Ms vs Ms
plt.figure()
for i in range(numzbin):
    plt.plot(MstarIary[i](x[i]), Mhalo[i](x[i]) - MstarIary[i](x[i]),
             label=str(redshifts[i]) + '<z<' + str(redshifts[i + 1]))
    plt.fill_between(MstarIary[i](x[i]), Mhalo[i](x[i]) - MstarIaryMinus[i](x[i]),
                     Mhalo[i](x[i]) - MstarIaryPlus[i](x[i]), alpha=0.5)
    plt.scatter(MstarPeak[i], MhOnMsPeak[i])
    plt.plot((MstarPeakMin[i], MstarPeakMax[i]), (MhOnMsPeak[i], MhOnMsPeak[i]))
plt.legend()
plt.ylabel('$M_{h}/M_{*}$', size=20)
plt.xlabel('Log($M_{*}$)  [Log($M_{\odot}$)]', size=20)
plt.tight_layout()
# plt.title('IariDavidzon Mass Function vs Bolshoï simulation')
plt.show()

# Plot the MstarPeak vs z evolution
plt.figure()
plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, MstarPeak,
             yerr=[MstarPeak-MstarPeakMin, MstarPeakMax-MstarPeak], fmt='o', capsize=5)
plt.xlabel('Reshift', size=20)
plt.ylabel('Log($M_{star}^{peak}$) [Log($M_{\odot}$)]', size=20)
plt.tight_layout()
plt.show()

# Plot the evolution of Ms/Mh(peak) vs z
yerr = [[ym[i][index_halopeak[i]]-yminus[i][index_halopeak[i]],
        yplus[i][index_halopeak[i]] - ym[i][index_halopeak[i]]] for i in range(10)]
plt.figure()
plt.errorbar(
    (redshifts[1:] + redshifts[:-1]) / 2,
    MsOnMhaloPeak,
    yerr=np.transpose(yerr),
    fmt='o')
plt.xlabel('Reshift', size=20)
plt.ylabel('Log($[M_{*}/M_{h}]^{peak}$)', size=20)
plt.tight_layout()
plt.show()

"""Fit a Yang law on Ms/Mh"""


def mstar_over_mh_yang(x, A, m1, beta, gamma):
    """Yang et al. 2004 function, see Moster et al. 2010. All masses are in logscale."""
    return 2.0 * A / (10 ** ((x - m1) * (- beta)) + 10 ** ((x - m1) * gamma))


n_fit = 1000
yang_fit = np.empty([numzbin, 4])
yang_cov = np.empty([numzbin, 4, 4])
yang_sigma = np.empty([numzbin, 4])

for i in range(numzbin):
    print('Fiting a Yang law, z=' + str(i))
    yang_fit[i], yang_cov[i] = curve_fit(mstar_over_mh_yang,
                                         Mhalo[i](x[i]),
                                         10**(MstarIary[i](x[i]) - Mhalo[i](x[i])),
                                         # uncertainty on the data :
                                         sigma=10**(yplus[i] - yminus[i]),
                                         bounds=[[-np.inf, 8, 0, 0], [np.inf, 14, 5, 5]])
    yang_sigma[i] = np.sqrt(np.diag(yang_cov[i]))

"""Plot"""

# Plot the Yang fit on the Ms/Mh ratio
for i in range(numzbin):
    plt.figure()
    plt.plot(Mhalo[i](x[i]), MstarIary[i](x[i]) - Mhalo[i](x[i]))
    plt.plot(Mhalo[i](x[i]), np.log10(mstar_over_mh_yang(Mhalo[i](x[i]), *yang_fit[i])), label='')
    plt.fill_between(xm[i], yminus[i], yplus[i], alpha=0.5)
# plt.legend()
plt.xlabel('xlabel', size=20)
plt.ylabel('ylabel', size=20)
plt.tight_layout()
plt.show()

# PLot the evolution of the parameters of the Yang Fit

plt.figure()
plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, yang_fit[:, 2], yerr=yang_sigma[:, 2],
             fmt='o', capsize=5)
plt.xlabel('Redshift', size=20)
plt.ylabel('$\\beta$ of Yang fit', size=20)
plt.tight_layout()
plt.show()

plt.figure()
plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, yang_fit[:, 3], yerr=yang_sigma[:, 3],
             fmt='o', capsize=5)
plt.xlabel('Redshift', size=20)
plt.ylabel('$\\gamma$ of Yang fit', size=20)
plt.tight_layout()
plt.show()

plt.figure()
plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, yang_fit[:, 1], yerr=yang_sigma[:, 1],
             fmt='o', capsize=5)
plt.xlabel('Redshift', size=20)
plt.ylabel('$M_{1}$ of Yang fit', size=20)
plt.tight_layout()
plt.show()
