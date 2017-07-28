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
for i in range(10):
    plt.figure()
    plt.fill_between(smf[i][:,0], smf[i][:,2], smf[i][:,3], alpha=0.5,
    label=str(redshifts[i])+'<z<'+str(redshifts[i+1]))
    plt.scatter(np.linspace(mmingal, mmaxgal, num=1000), np.log10(Ngal[i]), marker='+',
        color='red')
    plt.ylim(-6,-2)
    plt.xlim(9,12)
    plt.title('Davidzon+17 Schechter fits')
    plt.ylabel('Log($\phi$) [Log($Mpc^{-3}$)]')
    plt.xlabel('Log($M_{*}$) [Log($M_{\odot}$)]')
    plt.legend(loc=3)
plt.show()


"""Compute Galaxy Cumulative Density
"""
## The cumulative density is defined by the integral of phi(m)dm between m=M*
## an m=+inf.
## So we need to interpolate to have the mass function for every mass.


# Make interpolations to have the function phi(m)
phiIary = []
phiIaryminus = []
phiIaryplus = []

for i in range(numzbin):
    phiIary.append(interp1d(10**smf[i][:,0], 10**smf[i][:,1]))
    phiIaryminus.append(interp1d(10**smf[i][:,0], 10**smf[i][:,2]))
    phiIaryplus.append(interp1d(10**smf[i][:,0], 10**smf[i][:,3]))

NgalIary = []
NgalIaryminus = []
NgalIaryplus = []

# Compute integrals to have the cumulative density = int(phi(m)dm)


"""Plot"""

for i in range(10):
    plt.figure()
    plt.xscale('log');plt.yscale('log')
    plt.fill_between(10**smf[i][:,0], NgalIaryminus[i](10**smf[i][:,0]),
        NgalIaryplus[i](10**smf[i][:,0]), alpha=0.5)
    plt.scatter(10**np.linspace(mmingal, mmaxgal, num=1000) ,Ngal[i], marker='+',
        color='red')
    plt.plot(10**smf[i][:,0], NgalIary[i](10**smf[i][:,0]))
    plt.plot(xlog, Number_density[i])

    plt.ylim(10**-7, 0.1)
    plt.xlim(10**8, 10**12)
    plt.ylabel('N(>M)')
    plt.xlabel('Mass')
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
    MstarIary.append(interp1d(NgalIary[i](10**smf[i][:,0]), 10**smf[i][:,0]))
    MstarIaryMinus.append(interp1d(NgalIaryminus[i](10**smf[i][:,0]), 10**smf[i][:,0]))
    MstarIaryPlus.append(interp1d(NgalIaryplus[i](10**smf[i][:,0]), 10**smf[i][:,0]))
    Mhalo.append(interp1d(Nhalo[i][:], MvirNhalo))



"""Compute M*/Mh with uncertainties"""

n_fit=1000
x = np.zeros([numzbin, n_fit])
xm =np.zeros([numzbin, n_fit])
ym =np.zeros([numzbin, n_fit])
yminus =np.zeros([numzbin, n_fit])
yplus =np.zeros([numzbin, n_fit])

for i in range(numzbin):
    print(i)
    xtmp = MstarIary[i].x[:] # equals to np.flip(NgalIary[i](10**smf[i][:,0]),0)
    x[i] = np.geomspace( max(xtmp[np.where(xtmp>0)][0], MstarIaryMinus[i].x[0],
    MstarIaryPlus[i].x[0], Nhalo[i][-1]), min(xtmp[-1], MstarIaryMinus[i].x[-1],
    MstarIaryPlus[i].x[-1], Nhalo[i][0]), 1000)
    x[i][0] = max(xtmp[np.where(xtmp>0)][0], MstarIaryMinus[i].x[0],
        MstarIaryPlus[i].x[0], Nhalo[i][-1])
    x[i][-1] = min(xtmp[-1], MstarIaryMinus[i].x[-1],
        MstarIaryPlus[i].x[-1], Nhalo[i][0])
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
