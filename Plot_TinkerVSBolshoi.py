#!/usr/bin/env python3
# -*-coding:Utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
# import get_Vmax
from astropy.cosmology import LambdaCDM
import get_Vmax_mod



"""Load Bolshoi HMF"""
# redshifts of the BolshoiPlanck files
redshift_haloes_bolshoi = np.arange(0, 10, step=0.1)
numredshift_haloes_bolshoi = np.size(redshift_haloes_bolshoi)

"""Definition of hmf_bolshoi columns :

hmf_bolshoi[redshift][:,0] = Log10(mass) [Msun]
hmf_bolshoi[redshift][:,1] = Log10(cen_mf), ie central haloes mass function
(density) [1/Mpc^3]
hmf_bolshoi[redshift][:,2] = Log10(all_macc_mf), ie all haloes mass function
(density) [1/Mpc^3]
"""
hmf_bolshoi = []
for i in range(numredshift_haloes_bolshoi):
    hmf_bolshoi.append(
        np.loadtxt('../Data/HMFBolshoiPlanck/mf_planck/mf_planck_z' +
                   '{:4.3f}'.format(redshift_haloes_bolshoi[i]) + '_mvir.dat'))


""" Select corresponding id of redhsift for COSMOS"""

redshifts = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
numzbin = np.size(redshifts) - 1

redshift_id_selec = np.empty(numzbin)
for i in range(numzbin):
    redshift_id_selec[i] = np.argmin(
        np.abs(redshift_haloes_bolshoi - (redshifts[i] + redshifts[i + 1]) / 2))

redshift_id_selec = redshift_id_selec.astype(int)
print('Redshifts of Iari SMFs : ' + str((redshifts[:-1] + redshifts[1:]) / 2))
print('Closest redshifts for Bolshoi HMFs : '
      + str(redshift_haloes_bolshoi[redshift_id_selec]))



"""Load files Tinker"""

redshift_haloes_tinker = np.array([0.35, 0.65, 0.95, 1.3, 1.75, 2.25, 2.75, 3.25, 4, 5])
numredshift_haloes_tinker = len(redshift_haloes_tinker)


"""Load Tinker+08 HMF computed with HFMCalc of Murray+13
parameters : Delta = 70, mean overdensity.
"""
hmf_tinker = []
for i in range(numredshift_haloes_tinker):
    hmf_tinker.append(
        np.loadtxt('../Data/Tinker08HMF/HMFCalc_Dm200/mVector_PLANCK-SMT_z{:1.2f}.txt'.format(
            redshift_haloes_tinker[i]), usecols=(0, 7)))
    hmf_tinker[i][:, 0] = np.log10(hmf_tinker[i][:, 0] / 0.6774)
    hmf_tinker[i][:, 1] = hmf_tinker[i][:, 1] * (0.6774)**3
    #hmf[i][:, 0] = np.log10(hmf[i][:, 0])




""" Plot Tinker"""
#for i in range(numredshift_haloes):
plt.figure()
for i in [0]:
    plt.semilogy(hmf_tinker[i][:, 0], hmf_tinker[i][:, 1], label=redshift_haloes_tinker[i])
    plt.ylim(10**-6, 10**-1)
    plt.xlim(9.5, 15)
plt.legend()


"""Plot Bolshoi"""
# for i in range(numredshift_haloes):
#     plt.plot(hmf_bolshoi[i][:,0], hmf_bolshoi[i][:,1])

# Plot central HMF and central+satellite HMF
plt.figure()
for i in [0, 20, 40, 60]:
#for i in [redshift_id_selec[0]]:
    p = plt.plot(hmf_bolshoi[i][:, 0], 10**hmf_bolshoi[i][:, 2], label='z='+str(redshift_haloes_bolshoi[i]))
    plt.plot(hmf_bolshoi[i][:, 0], 10**hmf_bolshoi[i][:, 1], linestyle='--', color=p[0].get_color())
plt.ylim(10**-6, 10**-0.5)
plt.xlim(9.5, 15)
plt.yscale('log')
plt.legend()
plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]')
plt.ylabel('dN/dlog($M_{vir}$)   [$Mpc^{-3}$]')
# plt.title('HMF for Bolshoï Planck')


""" Plot both for each redhsift"""

for j in range(numzbin):
    plt.figure()
    i = redshift_id_selec[j]
    print(i)
    p = plt.plot(hmf_bolshoi[i][:, 0], 10**hmf_bolshoi[i][:, 2], label='z='+str(redshift_haloes_bolshoi[i]))
    plt.plot(hmf_bolshoi[i][:, 0], 10**hmf_bolshoi[i][:, 1], linestyle='--', color=p[0].get_color())
    
    plt.semilogy(hmf_tinker[j][:, 0], hmf_tinker[j][:, 1], label=redshift_haloes_tinker[j])

    plt.ylim(10**-6, 10**-1)
    plt.xlim(9.5, 15)
    plt.legend()