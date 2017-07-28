#!/usr/bin/env python3
# -*-coding:Utf-8 -*

import pyfits
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad

"""Script used to load SMF from Davizon+17 paper with uncertainties and marke
abundance matching with it
"""

redshifts = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
numzbin = np.size(redshifts)-1


"""Load the SMF"""

smf = []
for i in range(10):
    # smf.append(np.loadtxt('../Data/Davidzon/Davidzon+17_SMF_V3.0/mf_mass2b_fl5b_tot_VmaxFit2D'+str(i)+'.dat'))
    smf.append(np.loadtxt('../Data/Davidzon/schechter_fixedMs/mf_mass2b_fl5b_tot_VmaxFit2E'
        + str(i) + '.dat'))


"""Plot"""
plt.figure()
for i in range(10):
    plt.fill_between(smf[i][:,0], smf[i][:,2], smf[i][:,3], alpha=0.5,
    label=str(redshifts[i])+'<z<'+str(redshifts[i+1]))
    plt.ylim(-6,-2)
    plt.xlim(9,12)
    plt.title('Davidzon+17 Schechter fits')
    plt.ylabel('Log($\phi$) [Log($Mpc^{-3}$)]')
    plt.xlabel('Log($M_{*}$) [Log($M_{\odot}$)]')
    plt.legend(loc=3)
plt.show()


"""Compute Galaxy Cumulative Density
"""

# Compute integrals to have the cumulative density = int(phi(m)dm)

numpoints = np.size(smf[0][:,0])
Nstar  = np.empty([numzbin, numpoints])
Nstarminus = np.empty([numzbin, numpoints])
Nstarplus = np.empty([numzbin, numpoints])
for i in range(numzbin):
    for j in range(numpoints):
        Nstar[i,j]= np.trapz(10**smf[i][j:,1], smf[i][j:,0])
        Nstarminus[i,j] = np.trapz(10**smf[i][j:,2], smf[i][j:,0])
        Nstarplus[i,j] = np.trapz(10**smf[i][j:,3], smf[i][j:,0])



"""Plot"""

plt.figure()
for i in range(10):
    plt.fill_between(smf[i][:,0], Nstarminus[i,:], Nstarplus[i,:], alpha=0.5,
        label=str(redshifts[i])+'<z<'+str(redshifts[i+1]))
    ## To compare with my own density computations on JeanCoupon catalog
    ## plt.scatter(np.linspace(mmingal, mmaxgal, num=n),Ngal[i], marker='+', color='red')
plt.yscale('log')
plt.ylim(10**-6, 0.1)
plt.xlim(8, 12)
plt.ylabel('N(>$M_{*}$), [$Mpc^{-3}$]')
plt.xlabel('Log(Stellar Mass), [Log($M_{\odot}$)]')
plt.legend(loc=3)
plt.show()


"""Load Density of DM halos from Bolshoï simulation"""

Nhalo = np.load('Nhalo.npy')
MvirNhalo = np.load('MvirNhalo.npy')


"""Interpolate
"""

MstarIary = []
MstarIaryPlus = []
MstarIaryMinus = []
Mhalo = []

for i in range(numzbin):
    """do the interpolation for each redshift bin, in order to have the functions
    StellarMass(abundane) and HaloMass(abundance)"""
    MstarIary.append(interp1d(Nstar[i,:], 10**smf[i][:,0]))
    MstarIaryMinus.append(interp1d(Nstarminus[i,:], 10**smf[i][:,0]))
    MstarIaryPlus.append(interp1d(Nstarplus[i,:], 10**smf[i][:,0]))
    Mhalo.append(interp1d(Nhalo[i][:], MvirNhalo))



"""Compute M*/Mh with uncertainties"""

n_fit=1000
x = np.empty([numzbin, n_fit])
xm =np.empty([numzbin, n_fit])
ym =np.empty([numzbin, n_fit])
yminus =np.empty([numzbin, n_fit])
yplus =np.empty([numzbin, n_fit])

for i in range(numzbin):
    print(i)
    x[i] = np.geomspace(max(min(Nstar[i, Nstar[i,:]>0]),
     Nstarminus[i,-1], Nstarplus[i,-1], Nhalo[i, -1]),
     min(Nstar[i, 0], Nstarminus[i,0], Nstarplus[i,0], Nhalo[i, 0]), 1000)
    x[i][0] = max(min(Nstar[i, Nstar[i,:]>0]), Nstarminus[i,-1], Nstarplus[i,-1], Nhalo[i, -1])
    x[i][-1] = min(Nstar[i, 0], Nstarminus[i,0], Nstarplus[i,0], Nhalo[i, 0])
    xm[i] = Mhalo[i](x[i])
    ym[i] = MstarIary[i](x[i])/Mhalo[i](x[i])
    yminus[i] = MstarIaryMinus[i](x[i])/Mhalo[i](x[i])
    yplus[i] = MstarIaryPlus[i](x[i])/Mhalo[i](x[i])

"""Plot"""

index_min = np.empty(numzbin).astype('int')
for i in range(numzbin):
    index_min[i] = np.argmin(ym[i, :950])


for i in range(numzbin):
    plt.figure()
    #index_min = np.argmin(Nhalo[i]>0)
    plt.plot(xm[i][index_min[i]:], ym[i][index_min[i]:], label=str(redshifts[i])+'<z<'+str(redshifts[i+1]))
    plt.fill_between(xm[i], yminus[i], yplus[i], alpha=0.5)

    plt.scatter(Mpeak[i], ym[i][Mpeak_idx[i]])
    plt.plot((Mpeakmin[i], Mpeakmax[i]), (ym[i][Mpeak_idx[i]], ym[i][Mpeak_idx[i]] ))

    plt.legend()
    plt.xlim(2.8*10**9,10**15)
    plt.ylim(0.9*10**-3, 0.11)
    plt.ylabel('$M_{*}/M_{h}$', size=20)
    plt.xlabel('$M_{h}$  [$M_{\odot}]$', size=20)
    plt.xscale('log');plt.yscale('log')
    plt.tight_layout()
# plt.title('IariDavidzon Mass Function vs Bolshoï simulation')
plt.show()


"""Compute Mpeak and uncertainties"""
# Mpeak correspond to the Mh with the highest M*/Mh
# Need to restrain the interval tto look for the local minimum
# -> it is not very clean to use 640 empiriicaly, should find a better method

Mpeak = np.empty(numzbin)
Mpeakmin = np.empty(numzbin)
Mpeakmax = np.empty(numzbin)
Mpeak_idx = np.empty(numzbin)
for i in range(numzbin):
    Mpeak_idx[i] = np.argmax(ym[i][650:])+650
    Mpeak_idx = Mpeak_idx.astype('int')
    Mpeak[i] = xm[i][Mpeak_idx[i]]
    Mpeakmax[i] = xm[i][np.argmin(np.abs((yplus[i][640:Mpeak_idx[i]]-ym[i][Mpeak_idx[i]])))+640]
    Mpeakmin[i] = xm[i][np.argmin(np.abs((yplus[i][Mpeak_idx[i]:]-ym[i][Mpeak_idx[i]])))+Mpeak_idx[i]]

# for i in range(numzbin):
#     plt.scatter(Mpeak[i], ym[i][Mpeak_idx[i]])
#     plt.plot((Mpeakmin[i], Mpeakmax[i]), (ym[i][Mpeak_idx[i]], ym[i][Mpeak_idx[i]] ))
#
# for i in range(numzbin):
#     plt.plot(ym[i], label=str(redshifts[i])+'<z<'+str(redshifts[i+1]))
#     plt.scatter(Mpeak_idx[i], ym[i][Mpeak_idx[i]])
#     plt.scatter(np.argmin(np.abs((yplus[i][640:Mpeak_idx[i]]-ym[i][Mpeak_idx[i]])))+640, ym[i][np.argmin(np.abs((yplus[i][600:Mpeak_idx[i]]-ym[i][Mpeak_idx[i]])))+600])
#     plt.yscale('log')
#
# plt.scatter(Mpeak[i], ym[i][Mpeak_idx[i]])
# plt.plot((Mpeakmin[i], Mpeakmax[i]), (ym[i][Mpeak_idx[i]], ym[i][Mpeak_idx[i]] ))

plt.errorbar((redshifts[1:]+redshifts[:-1])/2, Mpeak[:],
    yerr=[Mpeak[:]-Mpeakmin[:], Mpeakmax[:]-Mpeak[:]],
    xerr=((redshifts[1:]+redshifts[:-1])/2-redshifts[:-1],
    redshifts[1:] -(redshifts[1:]+redshifts[:-1])/2),
    fmt='o', capsize=2)
plt.yscale('log')
plt.xlabel('Redshift', fontsize=15)
plt.ylabel('$M_{peak}$, [$M_{\odot}$]', fontsize=15)
