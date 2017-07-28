#!/usr/bin/env python3
# -*-coding:Utf-8 -*

'''Some code to compute the abuncance matching'''


import pyfits
import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.cosmology import Planck15 as cosmo

#exec(open('TestCosmos2015.py').read())


def comov_volume(omega_sample, zmin, zmax):
    "Compute the comoving volume between two redshifts in a solid angle."
    V = omega_sample/41253.0*(cosmo.comoving_volume(zmax)-cosmo.comoving_volume(
        zmin))
    return V


""" For Laigle Catalog"""

"""Redshift selection"""

"""
zmin = 0.3
zmax = 0.7
zbin = galdata[galdata['PHOTOZ']>zmin]
zbin = zbin[zbin['PHOTOZ']<zmax]
omega_sample = 1.2
V = comov_volume(omega_sample, zmin, zmax)

#Compute abundances

n = 100 #number of mass bins for our graph
mmingal = zbin['MASS_MED'].min()
mmaxgal = zbin['MASS_MED'].max()
step = (mmaxgal-mmingal)/n #resolution
h = cosmo.h
N = np.empty(n)

for i in range(n):
    "Compute the number of galaxies more massive than m for each mass bin"
    N[i] = np.sum(zbin['MASS_MED']>(mmingal+step*i)) / (V*h*h*h)


"""
"""For Jean Coupon Catalog """

zmin = 0.2
zmax=0.5
zbin = tdata[tdata['photo_z']>zmin]
zbin = zbin[zbin['photo_z']<=zmax]

n = 500 #number of mass bins for our graph
mmingal = zbin['mstar_cosmo'].min()
mmaxgal = zbin['mstar_cosmo'].max()
step = (mmaxgal-mmingal)/n #resolution

omega_sample = 1.2
V = comov_volume(omega_sample, zmin, zmax)

# # Compute Density for a linear scale
Ngal = np.empty(n)
for i in range(n):
    "Compute the number of galaxies more massive than m for each mass bin"
    Ngal[i] = np.sum(zbin['mstar_cosmo']>(mmingal+step*i)) / V.value


#Compute density for a log scale
Ngallog = np.empty(n)
massbinlog = np.logspace(mmingal, mmaxgal, num=n)
for i in range(n):
    "Compute the number of galaxies more massive than m for each mass bin"
    Ngallog[i] = np.sum(zbin['mstar_cosmo']>np.log10(massbinlog[i])) / V.value



#Plots
# plt.plot(np.linspace(mmingal, mmaxgal, num=n),Ngal)
# plt.ylabel('N(>m), $Mpc^{-3}$')
# plt.xlabel('Stellar Mass, $log( M_{*} / M_{\odot})$')
# plt.title('Abundance for Jean Coupon Catalog')
# plt.show()

# plt.loglog(massbinlog, Ngallog)
# plt.ylabel('log( N(>m) $Mpc^{-3})$')
# plt.xlabel('$log( M_{*} / M_{\odot})$')
# plt.title('Abundance for Jean Coupon Catalog')
# plt.show()
