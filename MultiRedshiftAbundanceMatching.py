#!/usr/bin/env python3
# -*-coding:Utf-8 -*

import pyfits
import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.cosmology import Planck15 as cosmo
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


""" Script used to compute Stellar mass/Halo mass relation for several
redshfit bins.
"""

n = 1000
h = 0.7
# # Select z boundaries
redshifts = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
numzbin = np.size(redshifts)-1
#Find corresponding scale factor for the Behroozi simulation
# zmoy = np.empty(np.size(redshifts)-1)
# for i in range(np.size(redshifts)-1):
#     zbin = tdata[tdata['photo_z']>redshifts[i]]
#     zbin = zbin[zbin['photo_z']<redshifts[i+1]]
#     zmoy[i] = np.average(zbin['photo_z'])
# a = 1/(1+zmoy)
# print(a)

# result for a :
# redshifts = np.array([0, 0.1, 0.3, 0.6, 1, 2, 3, 4, 5, 6, 7])
# [ 0.93096792  0.82347761  0.69119113  0.55298196  0.4059967   0.28837653
#  0.22816628  0.18574409  0.15691317  0.13445679]

# redshifts = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
# a = [ 0.73505654  0.60270537  0.51882271  0.43708564  0.3644327   0.30974748
#   0.26983487  0.23574484  0.20329374  0.17188638]

a_halos = [0.73635, 0.60435, 0.51635, 0.43835, 0.36635, 0.30635, 0.27035,
    0.23435, 0.20235, 0.17435]

""" Compute abundance for halos from Behroozi et al.
"""

V_behroozi = 250*250*250 / h**3 # Mpc^3



mminhalo = 8.5
mmaxhalo= 15
step = (mmaxhalo-mminhalo)/n
massbinlog = np.logspace(mminhalo, mmaxhalo, num=n)


data_halos = []
Nhalo = np.load('Nhalo.npy')
# Nhalo = np.empty([np.size(redshifts)-1,n])
# for i in range(np.size(redshifts)-1):
#     print(redshifts[i])
#     data_halos.append(np.load('../Data/halocat_'+str(a_halos[i])+'.npz'))
#     logmvir = np.log10(data_halos[i]['mvir'] / h) #Replace h value and take log10
#     print(logmvir.min())
#     print(logmvir.max())
#     for j in range(n):
#         "Compute the number of galaxies more massive than m for each mass bin"
#         Nhalo[i][j] = np.sum(logmvir>(mminhalo+step*j)) / V_behroozi



""" Compute abundance for galaxies from Jean Coupon et al.
"""

# # For Jean Coupon catalog
photoz= 'photo_z'
mstar = 'mstar_cosmo'

# For Laigle catalog
# photoz = 'PHOTOZ'
# mstar = 'MASS_MED'



omega_sample = 1.2

def comov_volume(omega_sample, zmin, zmax):
    "Compute the comoving volume between two redshifts in a solid angle."
    V = omega_sample/41253.0*(cosmo.comoving_volume(zmax)-cosmo.comoving_volume(
        zmin))
    return V

zbin = []
Ngal = np.empty([np.size(redshifts)-1,n])
mmingal = 7
mmaxgal = 12.5
step = (mmaxgal-mmingal)/n #resolution

for i in range(np.size(redshifts)-1):
    tmp = tdata[tdata[photoz]>redshifts[i]]
    tmp = tmp[tmp[photoz]<=redshifts[i+1]]
    zbin.append(tmp)
    print(zbin[i][mstar].min())

    V = comov_volume(omega_sample, redshifts[i], redshifts[i+1])

    for j in range(n):
        "Compute the number of galaxies more massive than m for each mass bin"
        Ngal[i, j] = np.sum(zbin[i][mstar]>(mmingal+step*j)) / V.value



""" Interpolate
"""

Mstar = []
Mhalo = []

for i in range(np.size(redshifts)-1):
    """do the interpolation for each redshift bin, in order to have the functions
    StellarMass(abundane) and HaloMass(abundance)"""
    Mstar.append(interp1d(Ngal[i][:], np.linspace(mmingal, mmaxgal, num=n)))
    Mhalo.append(interp1d(Nhalo[i][:], np.linspace(mminhalo, mmaxhalo, num=n)))
    # print(Ngal[i][0], Ngal[i][-1])
    # print(Nhalo[i][0], Nhalo[i][-1])


""" Plots
"""

## Plot galaxy abundances
for i in range(np.size(redshifts)-1):
    plt.semilogy(np.linspace(mmingal, mmaxgal, num=n),Ngal[i][:], label=str(redshifts[i])+'<z<'+str(redshifts[i+1]))
plt.ylabel('N (>M) [$Mpc^{-3}$]')
plt.xlabel('log(M) [$M_{\odot}$]')
plt.title('J.Coupon Abundances')
plt.legend()

MvirNhalo = np.load('MvirNhalo.npy')
## PLot halo abundances
for i in range(np.size(redshifts)-1):
    plt.semilogy(MvirNhalo ,Nhalo[i][:], label=str(redshifts[i])+'<z<'+str(redshifts[i+1]))
# plt.title('Halo Abundances from Behroozi')
plt.xscale('log')
plt.xlabel('Halo mass [$M_{\odot}$]')
plt.legend()
plt.ylabel('N (>$M_{halo}$) [$Mpc^{-3}$]')
#
# # Plot M*/Mh vs Mh
# for i in range(numzbin):
#     x = np.logspace(min(Ngal[i][0], Nhalo[i][0]), max(Ngal[i][-1], Nhalo[i][-1]), 10000)
#     plt.loglog(10**Mhalo[i](np.log10(x)), (10**Mstar[i](np.log10(x)))/(10**Mhalo[i](np.log10(x))), label=str(redshifts[i])+'<z<'+str(redshifts[i+1]))
# plt.legend()
# plt.ylabel('$M_{*}/M_{h}$')
# plt.xlabel('$M_{h}$  [$M_{\odot}]$')
# plt.title('J.Coupon vs Behroozi')
#
#
# for i in range(numzbin):
#     x = np.linspace(min(Ngal[i][0], Nhalo[i][0]), max(Ngal[i][-1], Nhalo[i][-1]), 10000)
#     plt.loglog(10**Mhalo[i](x), (10**Mstar[i](x))/(10**Mhalo[i](x)), label=str(redshifts[i])+'<z<'+str(redshifts[i+1]))
# plt.legend()
# plt.ylabel('$M_{*}/M_{h}$')
# plt.xlabel('$M_{h}$  [$M_{\odot}]$')
#
# for i in range(numzbin):
# x = (np.linspace(np.log10(min(Ngal[i][0], Nhalo[i][0])), np.log10(max(Ngal[i][-1], Nhalo[i][-1]))), 10000)
# plt.loglog(10**Mhalo[i](x), (10**Mstar[i](x))/(10**Mhalo[i](x)), label=str(redshifts[i])+'<z<'+str(redshifts[i+1]))
# plt.legend()
# plt.ylabel('$M_{*}/M_{h}$')
# plt.xlabel('$M_{h}$  [$M_{\odot}]$')

# ## Testing interpolation
# for i in range(numzbin):
#     x = np.linspace(Ngal[i][0], Ngal[i][-1], 10000)
#     plt.plot(Mstar[i](x), x)
#     plt.plot(np.linspace(mmingal, mmaxgal, num=n),Ngal[i][:])
# plt.show()
# for i in range(numzbin):
#     x = np.linspace( Nhalo[i][0], Nhalo[i][-1], 10000)
#     plt.plot(Mhalo[i](x), x)
#     plt.plot(np.linspace(mmingal, mmaxgal, num=n),Nhalo[i][:])
# plt.show()


"""Fit on M*/Mh
"""

def mstar_over_mh_yang(x, A, m1, beta, gamma):
    """Yang et al. 2004 function, see Moster et al. 2010"""
    return 2.0*A*((x/m1)**(-beta)+(x/m1)**gamma)**(-1)

n_fit = 10000
x = np.empty([numzbin, n_fit])
xm =np.empty([numzbin, n_fit])
ym =np.empty([numzbin, n_fit])
yang_fit = np.empty([numzbin, 4])
yang_cov = np.empty([numzbin, 4, 4])


for i in range(numzbin):
    x[i] =  np.linspace(min(Ngal[i][0], Nhalo[i][0]),
        max(Ngal[i][-1], Nhalo[i][-1]), n_fit )
    xm[i] = 10**Mhalo[i](x[i])
    ym[i] = (10**Mstar[i](x[i])/10**Mhalo[i](x[i]))
    yang_fit[i], yang_cov[i] = curve_fit(mstar_over_mh_yang, xm[i], ym[i], p0=[0.1, 10**12, 0.1, 0.1],
    bounds = [[-inf, 10**9, 0, 0], [inf, 10**14, 5, 5]])

"""Plots
"""
## Plot the fit of the Yang function
for i in range(numzbin):
    p = plt.loglog(xm[i], mstar_over_mh_yang(xm[i], *yang_fit[i]),
       label=str(redshifts[i])+'<z<'+str(redshifts[i+1]))
    plt.loglog(xm[i], ym[i], color= p[0].get_color())

plt.title('Fit on M*/Mh')
plt.ylabel('$M_{*}/M_{h}$')
plt.xlabel('$M_{h}$  [$M_{\odot}$]')
plt.legend()
plt.ylim(0.01, 0.05)
plt.xlim(10**11, 10**14)
plt.show()

# ## Plot the Mpeak of the M*/Mh relation vs redshift
# Mpeak = np.empty(numzbin)
# for i in range(numzbin-3):
#     Mpeak[i] = xm[i][np.argmax(mstar_over_mh_yang(xm[i], *yang_fit[i]))]
#     plt.scatter((redshifts[i+1]+redshifts[i])/2, Mpeak[i])
# plt.yscale('log')
# plt.xlabel('Redshift bin')
# plt.ylabel('Mpeak [$M_{\odot}$]')
# plt.title('Mhalo Peak for M*/Mh with Yang fit')
