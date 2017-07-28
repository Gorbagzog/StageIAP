#!/usr/bin/env python3
# -*-coding:Utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import IariStellarHaloRelation as Iari

# import importlib
# importlib.reload(Iari)

""" Script used to compute Stellar mass/Halo mass relation for several
redshfit bins, using the Behroozi simulation and the Davidzon et al.
mass function.
"""

n_int = 1000
h = 0.7
# # Select z boundaries
redshifts = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
numzbin = np.size(redshifts)-1

a_halos = [0.73635, 0.60435, 0.51635, 0.43835, 0.36635, 0.30635, 0.27035,
    0.23435, 0.20235, 0.17435]

""" Load abundances for halos from Behroozi et al.
"""

V_behroozi = 250*250*250 / h**3 # Mpc^3
mminhalo = 8.5
mmaxhalo= 15
mmingal = 7
mmaxgal = 12.5

# Create a dictionnary to pass parameters in functions
param = dict()
param['V_behroozi']=V_behroozi
param['mminhalo']=mminhalo
param['mmaxhalo']=mmaxhalo
param['mmingal']=mmingal
param['mmaxgal']=mmaxgal
param['numzbin']=numzbin
param['n_int']=n_int

massbinlog = np.logspace(mminhalo, mmaxhalo, num=n_int)

Nhalo = np.load('Nhalo.npy')


"""Compute abundances with Iari Davidzon Shchechter functions :
"""

#Compute Number_density from IariStellarHaloRelation Script

Number_density, xlog = Iari.number_density(param)

"""Interpolate
"""
Mstar = []
Mhalo = []

for i in range(np.size(redshifts)-1):
    """do the interpolation for each redshift bin, in order to have the functions
    StellarMass(abundane) and HaloMass(abundance)"""

    """ L'interpolation ne fonctionne pas !!!!!!"""
    Mstar.append(interp1d(Number_density[i][:], xlog))
    Mhalo.append(interp1d(Nhalo[i][:], np.linspace(mminhalo, mmaxhalo, num=n_int)))


"""Fit on M*/Mh
"""
def mstar_over_mh_yang(x, A, m1, beta, gamma):
    """Yang et al. 2004 function, see Moster et al. 2010"""
    return 2.0*A*((x/m1)**(-beta)+(x/m1)**gamma)**(-1)

n_fit = 1000
x = np.empty([numzbin, n_fit])
xm =np.empty([numzbin, n_fit])
ym =np.empty([numzbin, n_fit])
yang_fit = np.empty([numzbin, 4])

for i in range(numzbin):
    print(i)
    # x[i] = np.linspace(max(Number_density[i][-1], Nhalo[i][-1]),
    #    min(Number_density[i][0], Nhalo[i][0]), n_fit )
    # xlog[i] = np.logspace(np.log10(max(Mstar[i].x[0], Mhalo[i].x[0])),
    #     np.log10(min(Mstar[i].x[-1], Mhalo[i].x[-1])), n_fit )
    x[i] = np.geomspace(max(Number_density[i, -2], Nhalo[i, -1]),
        min(Number_density[i, 0], Nhalo[i, 0]), n_fit)
    x[i][0] = max(Number_density[i, -1], Nhalo[i, -1])
    x[i][-1] = min(Number_density[i, 0], Nhalo[i, 0])
    xm[i] = 10**Mhalo[i](x[i])
    ym[i] = (Mstar[i](x[i])/10**Mhalo[i](x[i]))
#     yang_fit[i], _ = curve_fit(mstar_over_mh_yang, xm[i], ym[i], p0=[0.1, 10**12, 0.1, 0.1],
#     bounds = [[-inf, 10**9, 0, 0], [inf, 10**14, 5, 5]])

""" Plots
"""

# for i in range(numzbin):
#     plt.loglog(xm[i], ym[i], label=str(redshifts[i])+'<z<'+str(redshifts[i+1]))
# plt.legend()
# plt.ylabel('$M_{*}/M_{h}$')
# plt.xlabel('$M_{h}$  [$M_{\odot}]$')
# plt.show()

for i in range(numzbin):
    plt.scatter(xm[i], ym[i], label=str(redshifts[i])+'<z<'+str(redshifts[i+1]),
        marker='.')
plt.xscale('log')
plt.xscale('log')
plt.legend(loc=2)
plt.ylabel('$M_{*}/M_{h}$')
plt.xlabel('$M_{h}$  [$M_{\odot}]$')
plt.title('I.Davidzon mass functions vs Bolshoï simulation')
plt.show()
