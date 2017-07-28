
# coding: utf-8


get_ipython().magic('matplotlib osx')
import pyfits
import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.cosmology import Planck15 as cosmo


def comov_volume(omega_sample, zmin, zmax):
    "Compute the comoving volume between two redshifts in a solid angle."
    V = omega_sample/41253*(cosmo.comoving_volume(zmax)-cosmo.comoving_volume(
        zmin))
    return V



"""Load Jean's Catalog"""
hdulist = pyfits.open('../Data/COSMOS2015_clustering_v2.0_clean.fits')
tdata = hdulist[1].data
hdulist.close()

tdata = tdata[tdata['photo_z']<99]
tdata = tdata[tdata['clean']>0]
tdata = tdata[tdata['mstar_cosmo']>7.2]


#Redshift selection
zmin = 0.3
zmax=0.7
zbin = tdata[tdata['photo_z']>zmin]
zbin = zbin[zbin['photo_z']<zmax]

n = 100 #number of mass bins for our graph
mmin = zbin['mstar_cosmo'].min()
mmax = zbin['mstar_cosmo'].max()
step = (mmax-mmin)/n #resolution

omega_sample = 1.2
V = comov_volume(omega_sample, zmin, zmax)

zmoy = np.average(zbin['photo_z']) #We take the average z to compute h
h = cosmo.H(zmoy)/(100*cosmo.H0.unit)
V_corr = V*h*h*h
V_corr = V_corr.value

N = np.empty(n)
for i in range(n):
    "Compute the number of galaxies more massive than m for each mass bin"
    N[i] = np.sum(zbin['mstar_cosmo']>(mmin+step*i))
N = N / (V*h*h*h)


#plot
fig, ax = plt.subplots()
ax.plot(np.linspace(mmin, mmax, num=n)*h,N)
ax.set_title('Abundance for Jean\'s catalog')
ax.set_ylabel('N(>M*), $h^{3}.Mpc^{-3}$', size=14)
ax.set_xlabel('Mass, $log(M_{\odot}/h)$', size=14)
plt.show()
# a = 1/(1+zmoy)
# print('Le redshift moyen est '+str(zmoy)+', le facteur d\'échelle est donc de '+str(a))
