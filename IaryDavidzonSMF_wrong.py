#!/usr/bin/env python3
# -*-coding:Utf-8 -*

import pyfits
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


"""Script used to load SMF from Davizon+17 paper with uncertainties and marke
abundance matching with it
"""

redshifts = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
numzbin = np.size(redshifts)-1


"""Load the SMF"""

smf = []
for i in range(10):
    smf.append(np.loadtxt('../Data/Davidzon/Davidzon+17_SMF_V3.0/mf_mass2b_fl5b_tot_VmaxFit2D'+str(i)+'.dat'))



"""Plot"""
# for i in range(10):
#     plt.fill_between(smf[i][:,0], smf[i][:,2], smf[i][:,3], alpha=0.5,
#     label=str(redshifts[i])+'<z<'+str(redshifts[i+1]))
# plt.ylim(-6,-2)
# plt.xlim(9,12)
# plt.title('Davidzon+17 Schechter fits')
# plt.ylabel('Log($\phi$) [Log($Mpc^{-3}$)]')
# plt.xlabel('Log($M_{*}$) [Log($M_{\odot}$)]')
# plt.legend(loc=3)
# plt.show()


"""Compute Galaxy Cumulative Density
"""

numpoints = np.size(smf[0][:,0])
Nstar  = np.empty([numzbin, numpoints])
Nstarminus = np.empty([numzbin, numpoints])
Nstarplus = np.empty([numzbin, numpoints])

for i in range(numzbin):
    for j in range(numpoints):
        Nstar[i,j]= np.sum(10**smf[i][j:,1])
        Nstarminus[i,j] = np.sum(10**smf[i][j:,2])
        Nstarplus[i,j] = np.sum(10**smf[i][j:,3])

"""Plot"""
# plt.figure()
# for i in range(10):
#     plt.fill_between(10**smf[i][:,0], Nstarminus[i,:], Nstarplus[i,:], alpha=0.5)
# plt.xscale('log');plt.yscale('log')
# plt.ylim(10**-6, 1)
# plt.xlim(10**8, 10**12)
# plt.ylabel('N(>M)')
# plt.xlabel('Mass')
# plt.show()

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

for i in range(numzbin-1):
    index_min = np.argmin(Nhalo[i]>0)
    plt.plot(xm[i][index_min:], ym[i][index_min:], label=str(redshifts[i])+'<z<'+str(redshifts[i+1]))
for i in range(numzbin-1):
    plt.fill_between(xm[i], yminus[i], yplus[i], alpha=0.5)
plt.legend()
plt.xlim(2.8*10**9,10**15)
plt.ylabel('$M_{*}/M_{h}$', size=20)
plt.xlabel('$M_{h}$  [$M_{\odot}]$', size=20)
plt.xscale('log');plt.yscale('log')
plt.title('IariDavidzon Mass Function vs Bolshoï simulation')
plt.show()
