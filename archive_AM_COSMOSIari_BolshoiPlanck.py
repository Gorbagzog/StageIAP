#!/usr/bin/env python3
# -*-coding:Utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import interpolate


"""Script used to load and display the halo mass functions using Mvir
of the new Bolshoï simulation using Planck15 cosmology :
h=0.6774, s8=0.8159, Om=0.3089, Ob=0.0486, ns=0.9667

HaloMassFunctions are provided by Peter Behroozi.

This script also load the SMF of the COSMOS field provided by Iari Davidzon.

Then with the HMF and SMF teh script do an Abundance Matching and the
Ms/Mh vs Mh relation.

"""

""" Load files """
redshift_haloes = np.arange(0, 10, step=0.1) #redshifts of the BolshoiPlanck files
numredshift_haloes = np.size(redshift_haloes)
smf_bolshoi = []
 
for i in range(numredshift_haloes):
    smf_bolshoi.append(np.loadtxt('../Data/HMFBolshoiPlanck/mf_planck/mf_planck_z'+
    '{:4.3f}'.format(redshift_haloes[i])+'_mvir.dat'))

## smf_bolshoi[redshift][:,0] = Log10(mass) [Msun]
## smf_bolshoi[redshift][:,1] = Log10(cen_mf), ie central haloes mass function (density) [1/Mpc^3]
## smf_bolshoi[redshift][:,2] = Log10(all_macc_mf), ie all haloes mass function (density) [1/Mpc^3]


""" Plot"""
# for i in range(numredshift_haloes):
#     plt.plot(smf_bolshoi[i][:,0], smf_bolshoi[i][:,1])

""" Compute Halo cumulative density """

numpoints = np.size(smf_bolshoi[0][:,0])
Nbolshoi = []

for i in range(numredshift_haloes):
    Nbolshoi.append([])
    for j in range(np.size(smf_bolshoi[i][:,0])):
        Nbolshoi[i].append( np.trapz(10**smf_bolshoi[i][j:,1], smf_bolshoi[i][j:,0]))
for i in range(numredshift_haloes):
    Nbolshoi[i] = np.asarray(Nbolshoi[i])
        
"""Plots"""

for i in range(numredshift_haloes):
    plt.plot(smf_bolshoi[i][:,0], Nbolshoi[i][:])
plt.ylim(10**-7, 1)
plt.xlim(8,16)
plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]')
plt.ylabel('N(>M) [$Mpc^{-3}$]')
plt.yscale('log')
plt.title('Abundances for Bolshoï Planck 0<z<9.9')


"""Load the SMF from Iary Davidzon+17"""

## Code is copied from IaryDavidzonSMF.py as of 12 june
redshifts = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5]) # redshifts of the Iari SMF
numzbin = np.size(redshifts)-1

smf = []
for i in range(10):
    smf.append(np.loadtxt('../Data/Davidzon/Davidzon+17_SMF_V3.0/mf_mass2b_fl5b_tot_VmaxFit2D'+str(i)+'.dat'))


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



"""Select redshifts of haloes to match with Davidzon intervals"""

redshift_id_selec = np.empty(numzbin)
for i in range(numzbin):
    redshift_id_selec[i]= np.argmin(np.abs(redshift_haloes - 
        (redshifts[i]+redshifts[i+1])/2))


redshift_id_selec = redshift_id_selec.astype(int)
print('Redshifts of Iari SMFs : '+str((redshifts[:-1]+redshifts[1:])/2))
print('Closest redshifts for Bolshoi HMFs : '+str(redshift_haloes[redshift_id_selec]))



"""Do interpolation for abundance matching"""



MstarIary = []
MstarIaryPlus = []
MstarIaryMinus = []
Mhalo = []

for i in range(numzbin):
    """do the interpolation for each redshift bin, in order to have the functions
    StellarMass(abundane) and HaloMass(abundance)"""
    # MstarIary.append(interp1d(Nstar[i,:], 10**smf[i][:,0]))
    # MstarIaryMinus.append(interp1d(Nstarminus[i,:], 10**smf[i][:,0]))
    # MstarIaryPlus.append(interp1d(Nstarplus[i,:], 10**smf[i][:,0]))
    # Mhalo.append(interp1d(Nbolshoi[redshift_id_selec[i]][:], 
    #     10**smf_bolshoi[redshift_id_selec[i]][:,0]))
    
    MstarIary.append(interp1d(
        np.log10(Nstar[i, Nstar[i,:]>0]), smf[i][Nstar[i,:]>0,0], kind='cubic'))
    MstarIaryMinus.append(interp1d(
        np.log10(Nstarminus[i, Nstarminus[i,:]>0]), smf[i][Nstarminus[i,:]>0,0], kind='quadratic'))
    MstarIaryPlus.append(interp1d(
        np.log10(Nstarplus[i, Nstarplus[i,:]>0]), smf[i][Nstarplus[i,:]>0,0], kind='quadratic'))
    Mhalo.append(interp1d(np.log10(Nbolshoi[redshift_id_selec[i]][Nbolshoi[redshift_id_selec[i]]>0]), 
        smf_bolshoi[redshift_id_selec[i]][Nbolshoi[redshift_id_selec[i]]>0, 0], kind='quadratic'))

# MstarIary = []
# MstarIaryPlus = []
# MstarIaryMinus = []
# Mhalo = []
# for i in range(numzbin):
#     """do the interpolation for each redshift bin, in order to have the functions
#     StellarMass(abundane) and HaloMass(abundance)"""
#     MstarIary.append(interpolate.InterpolatedUnivariateSpline(Nstar[i,:], smf[i][:,0]))
#     MstarIaryMinus.append(interpolate.InterpolatedUnivariateSpline(Nstarminus[i,:], smf[i][:,0]))
#     MstarIaryPlus.append(interpolate.InterpolatedUnivariateSpline(Nstarplus[i,:], smf[i][:,0]))
#     Mhalo.append(interpolate.InterpolatedUnivariateSpline(Nbolshoi[redshift_id_selec[i]][:], 
#         smf_bolshoi[redshift_id_selec[i]][:,0]))
# 
# 
# MstarIary = []
# MstarIaryPlus = []
# MstarIaryMinus = []
# Mhalo = []
# for i in range(numzbin):
#     """do the interpolation for each redshift bin, in order to have the functions
#     StellarMass(abundane) and HaloMass(abundance)"""
#     MstarIary.append(interpolate.InterpolatedUnivariateSpline(Nstar[i,:], 10**smf[i][:,0]))
#     MstarIaryMinus.append(interpolate.InterpolatedUnivariateSpline(Nstarminus[i,:], 10**smf[i][:,0]))
#     MstarIaryPlus.append(interpolate.InterpolatedUnivariateSpline(Nstarplus[i,:], 10**smf[i][:,0]))
#     Mhalo.append(interpolate.InterpolatedUnivariateSpline(Nbolshoi[redshift_id_selec[i]][:], 
#         10**smf_bolshoi[redshift_id_selec[i]][:,0]))



"""Compute M*/Mh with uncertainties"""

n_fit=3000
x = np.empty([numzbin, n_fit]) # x is the density variable to trace Ms(x) and Mh(x), in logscale
xm =np.empty([numzbin, n_fit])
ym =np.empty([numzbin, n_fit])
yminus =np.empty([numzbin, n_fit])
yplus =np.empty([numzbin, n_fit])

for i in range(numzbin):
    print('Compute Ms/Mh, z='+str(redshifts[i]))
#     x[i] = np.geomspace(max(
#     min(Nstar[i, Nstar[i,:]>0]),
#         Nstarminus[i,-1], Nstarplus[i,-1], Nbolshoi[redshift_id_selec[i]][ -1]),
#     min(Nstar[i, 0], Nstarminus[i,0], Nstarplus[i,0],
#         Nbolshoi[redshift_id_selec[i]][ 0]), n_fit)
#     # to ensure that geomspace respects the given boundaries :
#     x[i][0] = max(min(Nstar[i, Nstar[i,:]>0]), Nstarminus[i,-1], 
#         Nstarplus[i,-1], Nbolshoi[redshift_id_selec[i]][ -1])
#     x[i][-1] = min(Nstar[i, 0], Nstarminus[i,0], Nstarplus[i,0], 
#         Nbolshoi[redshift_id_selec[i]][0])
#     
#     
    x[i] = np.geomspace(max(
    min(np.log10(Nstar[i, Nstar[i,:]>0])),
        np.log10(Nstarminus[i,Nstarminus[i,:]>0][-1]), np.log10(Nstarplus[i,Nstarplus[i,:]>0][-1]), 
        np.log10(Nbolshoi[redshift_id_selec[i]][ -1])),
    min(np.log10(Nstar[i, 0]), np.log10(Nstarminus[i,0]), np.log10(Nstarplus[i,0]),
        np.log10(Nbolshoi[redshift_id_selec[i]][ 0])), n_fit)
    # to ensure that geomspace respects the given boundaries :
    x[i][0] = max(min(np.log10(Nstar[i, Nstar[i,:]>0])), np.log10(Nstarminus[i,Nstarminus[i,:]>0][-1]), 
        np.log10(Nstarplus[i,Nstarplus[i,:]>0][-1]), np.log10(Nbolshoi[redshift_id_selec[i]][ -1]))
    x[i][-1] = min(np.log10(Nstar[i, 0]), np.log10(Nstarminus[i,0]), np.log10(Nstarplus[i,0]), 
        np.log10(Nbolshoi[redshift_id_selec[i]][0]))
    xm[i] = Mhalo[i](x[i])
    ym[i] = MstarIary[i](x[i])/Mhalo[i](x[i])
    yminus[i] = MstarIaryMinus[i](x[i])/Mhalo[i](x[i])
    yplus[i] = MstarIaryPlus[i](x[i])/Mhalo[i](x[i])


"""Plot Ms/Mh vs Mh"""

# index_min = np.empty(numzbin).astype('int')
# for i in range(numzbin):
#     index_min[i] = np.argmin(ym[i, :950])
plt.figure()
for i in range(numzbin-2):
    
    #index_min = np.argmin(Nhalo[i]>0)
    plt.plot(xm[i][:], ym[i][:], label=str(redshifts[i])+'<z<'+str(redshifts[i+1]))
    plt.fill_between(xm[i], yminus[i], yplus[i], alpha=0.5)

    # plt.scatter(Mpeak[i], ym[i][Mpeak_idx[i]])
    # plt.plot((Mpeakmin[i], Mpeakmax[i]), (ym[i][Mpeak_idx[i]], ym[i][Mpeak_idx[i]] ))

plt.legend()
plt.ylabel('$M_{*}/M_{h}$', size=20)
plt.xlabel('Log($M_{h}$)  [Log($M_{\odot})]$', size=20)
plt.tight_layout()
# plt.title('IariDavidzon Mass Function vs Bolshoï simulation')
plt.show()

# 
# """Plot interpolations Ms(N) and Mh(N)"""
# 
# plt.figure()
# for i in range(numzbin):
#     plt. plot(x[i], MstarIary[i](x[i]))