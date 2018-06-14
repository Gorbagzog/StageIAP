#!/usr/bin/env python3
# -*-coding:Utf-8 -*

"""Script to plot MhaloPeak found with different scripts"""

import numpy as np
import matplotlib.pyplot as plt

"""Load different MhaloPeak values"""

# MhaloTinker = np.loadtxt("../Plots/MhPeak/Tinker08_Dm200.txt")
# MhaloCosmos = np.loadtxt("../Plots/MhPeak/COSMOS.txt")
# MhaloCandels = np.loadtxt("../Plots/MhPeak/Candels.txt")

# MhaloCosmosTinker = np.loadtxt("../MCMC_Tinker_save_3-5/MhPeak_CosmosTinker.txt")
# MhaloCosmosTinker = np.loadtxt("../MCMC_Tinker_2202/MhPeak_CosmosTinker.txt")
MhaloCosmosTinker = np.loadtxt("../MCMC_2018-4-25T18-31/MhaloPeak.txt")
MhaloCosmosSchTinker = np.loadtxt("../MCMC_save_Schechter_6-8_1828_X_6-11_1454/MhaloPeak.txt")
# MhaloCosmosTinker[:,0] = MhaloCosmosTinker[:,0].astype('int')

MhaloCosmosTinker[:,1] += np.log10(67.74/70)
MhaloCosmosSchTinker[:,1] += np.log10(67.74/70)
redshifts = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
redshiftsbin = (redshifts[1:]+redshifts[:-1])/2

# MhaloCosmosMCMC = np.loadtxt("../MCMC_select/MhPeak_CosmosBolshoiTot.txt")

"""Definition of the evolution of Mpeak for Leauthaud et al, Behroozi et al et Moster et al"""

# Warning : All masses are in the same cosmo H0 = 70

# Use a small redshift offset for Leauthaud and Cowley
offset = 0.05

# Harikane et al+17 -> only the minimal position of the peak, not the complete shape of SHMR
redshiftHarikane = np.array([3.8, 4.9, 5.9])
MhaloPeakHarikane = np.log10(np.array([2 * 10**12, 2 * 10**12, 8 * 10**11])) + np.log10(67.74/70)
MhaloSigmaHarikane = np.array([0.1, 0.1, 0.3])


# Leauthaud+17 use a different cosmology with H0=72
redshiftLeauthaud = np.array([(0.22 + 0.48) / 2, (0.48 + 0.74) / 2, (0.74 + 1) / 2]) - offset # shift redshift to see the points
MhaloPeakLeauthaud = np.log10(np.array([9.5 * 10**11, 1.45 * 10**12, 1.4 * 10**12])) + np.log10(72/70)
MhaloSigmaLeauthaud = np.log10(np.array(
    [1.05 * 10**12, 1.55 * 10**12, 1.5 * 10**12])) - MhaloPeakLeauthaud
# MstarPeakLeauthaud = np.array([3.55 * 10**10, 4.9 * 10**10, 5.75 * 10**10])
# MstarSigmaLeauthaud = np.array([0.17, 0.15, 0.13])*10**10

redshiftCoupon15 = np.array([0.75])
MhaloPeakCoupon15 = np.log10(np.array([1.92*10**12])) + np.log10(72/70)
MhaloSigmaCoupon15 = np.array([[np.log10((1.92 + 0.17)) - np.log10(1.92)],
                               [np.log10(1.92) - np.log10(1.92 - 0.14)]])

redshiftCoupon12 = np.array([0.3, 0.5, 0.7, 1])
MhaloPeakCoupon12 = np.array([11.65, 11.79, 11.88, 11.9]) - np.log10(0.7)
MhaloSigmaCoupon12 = np.vstack([[0.05, 0.05, 0.06, 0.07], [0.05, 0.05, 0.06, 0.07]])

redshiftMartinezManso2014 = 1.5
MhaloPeakMartinezManso2014 = 12.44
MhaloSigmaMartinezManso2014 = [[0.08], [0.08]]

# Test graphic reading of McCracken+15 Mpeak
# redshiftMcCracken15 = np.array([0.65, 0.95, 1.3, 1.75, 2.25])
# MhaloPeakMcCracken15 = np.array([12.15, 12.1, 12.2, 12.35, 12.4])

# Load Coupon+17 draft Peak values
# We use PeakPosMCMCMean and PeakPosMCMCstd
# Values are given in Log10(Mh*h^-1 Msun)
# redshiftCoupon17 = np.array([0.34, 0.52, 0.70, 0.90, 1.17, 1.50,
#                             1.77, 2.15, 2.75, 3.37, 3.96, 4.83])
# MhaloPeakCoupon17 = np.zeros([np.size(redshiftCoupon17)])
# MhaloSigmaCoupon17 = np.zeros([np.size(redshiftCoupon17)])
# for i in range(len(redshiftCoupon17)):
#     MhaloPeakCoupon17[i], MhaloSigmaCoupon17[i] = np.loadtxt(
#         '../Data/Coupon17/peak/peak_{:1.2f}.ascii'.format(redshiftCoupon17[i]),
#         usecols=(2, 3))

# Graphic read of MhaloPeakvalues in Yang+2012 for SMF1 :
redshiftYang12 = np.array([0.1, 0.5, 1.15, 1.80, 2.75, 3.75])
MhaloPeakYang12 = np.array([11.75, 11.7, 12, 12.5, 12.7, 12.8]) - np.log10(0.7)
MhaloSigmaYang12 = np.array([[0.1, 0.1, 0.1, 0.1, 0.15, 0.2], [0.1, 0.1, 0.15, 0.2, 0.3, 0.5]])

# Mhalo Peak from Ishikawa 17
redshiftIshikawa17 = np.array([3, 4, 5])
MhaloPeakIshikawa17 = np.array([12.10, 11.99, 11.77]) - np.log10(0.7)
MhaloSigmaIshikawa17 = np.array([0.053, 0.057, 0.097])

# MhaloPeak from Cowley+2017
redshiftCowley17 = np.array([1.75, 2.5]) - offset # shift redshift to see the points
MhaloPeakCowley17 = np.array([12.33, 12.5])
MhaloSigmaCowley17 = np.array([[0.06, 0.08], [0.07, 0.10]])

# Load the MhaloPeak(z) from Behroozi et al 2013
tmp = np.loadtxt('MhaloPeakBehroozi.txt')
redshiftBehroozi13 = tmp[0]
MhaloPeakBehroozi13 = tmp[1]

# Load the MhaloPeak(z) from Moster et al 2013
tmp = np.loadtxt('MhaloPeakMoster.txt')
redshiftMoster13 = tmp[0]
MhaloPeakMoster13 = tmp[1]

# Load the MhaloPeak(z) from Yang et al 2012
tmp = np.loadtxt('MhaloPeakYang.txt')
redshiftYang12curve = tmp[0]
MhaloPeakYang12curve = tmp[1]


"""Plot"""

plt.figure(figsize=(10, 5))
# ax = plt.subplot(111)
# plt.figure()
# plt.errorbar(redshiftCoupon17, MhaloPeakCoupon17 - np.log10(0.7),
#              yerr=MhaloSigmaCoupon17,
#              fmt='o', color='blue', capsize=5, label='Coupon et al. 2017 Draft')

plt.errorbar(redshiftLeauthaud, MhaloPeakLeauthaud,
             yerr=MhaloSigmaLeauthaud, markersize=5, elinewidth=1,
             fmt='o', c='green', markerfacecolor='white', capsize=1, label='Leauthaud et al. 2011')
plt.errorbar(redshiftCoupon12, MhaloPeakCoupon12, yerr=MhaloSigmaCoupon12, elinewidth=1,
             fmt='v', c='grey', markerfacecolor='white', capsize=2, label='Coupon et al. 2012',
             markersize=5)
plt.errorbar(redshiftCoupon15, MhaloPeakCoupon15, yerr=MhaloSigmaCoupon15, elinewidth=1,
             fmt='s', c='turquoise', markerfacecolor='white', capsize=2, label='Coupon et al. 2015',
             markersize=5)
plt.errorbar(redshiftMartinezManso2014, MhaloPeakMartinezManso2014, elinewidth=1,
             yerr=MhaloSigmaMartinezManso2014, markersize=5,
             fmt='D', c='purple', markerfacecolor='white', capsize=2, label='Martinez-Manso et al. 2014')
# plt.errorbar(redshiftYang12, MhaloPeakYang12, yerr= MhaloSigmaYang12, markersize=5, elinewidth=1,
            #  fmt='^', c='lightblue', markerfacecolor='white', capsize=2, label='Yang et al. 12')
plt.errorbar(redshiftIshikawa17, MhaloPeakIshikawa17, yerr=MhaloSigmaIshikawa17, markersize=5,
             fmt='v', c='violet', markerfacecolor='white', capsize=2, label='Ishikawa et al. 2017',
             elinewidth=1)
plt.errorbar(redshiftCowley17, MhaloPeakCowley17, yerr=MhaloSigmaCowley17, markersize=5,
             fmt='*', c='orange', markerfacecolor='white', capsize=2, label='Cowley et al. 2018',
             elinewidth=1,)
plt.errorbar(redshiftHarikane, MhaloPeakHarikane,
             yerr=0.1, elinewidth=1,
             c='brown', label='Harikane et al. 2018, low lim',
             fmt='o', linestyle='none', capsize=3, lolims=True,
             markersize=3)
plt.plot(redshiftBehroozi13, MhaloPeakBehroozi13, color='limegreen', linestyle='--',
         label='Behroozi et al. 2013')
plt.plot(redshiftMoster13, MhaloPeakMoster13, color='royalblue', linestyle='--',
         label='Moster et al. 2013')
plt.plot(redshiftYang12curve, MhaloPeakYang12curve, color='lightblue', linestyle='--',
         label='Yang et al. 2012')
# plt.errorbar(redshiftMcCracken15, MhaloPeakMcCracken15,
#              fmt='d', markerfacecolor='none', capsize=5, label='"Revised" McCracken15')
# plt.errorbar(MhaloCosmos[:-2, 0], MhaloCosmos[:-2, 1], yerr=[MhaloCosmos[:-2, 2],
#              MhaloCosmos[:-2, 3]], fmt='o', color='red', capsize=3, label='Case 1',
#              markersize=7)
# plt.errorbar(MhaloTinker[:-2, 0], MhaloTinker[:-2, 1], yerr=[MhaloTinker[:-2, 2],
#              MhaloTinker[:-2, 3]], fmt='^', color='blue', capsize=3, label='Case 2',
#              markersize=7)
# plt.errorbar(MhaloCandels[:, 0], MhaloCandels[:, 1], yerr=[MhaloCandels[:, 2],
#              MhaloCandels[:, 3]], fmt='d', color='darkgreen', capsize=3,
#              label='Case 3', markersize=7)
plt.errorbar(redshiftsbin[MhaloCosmosTinker[:,0].astype('int')[:-3]], MhaloCosmosTinker[:-3, 1], yerr=MhaloCosmosTinker[:-3, 2],
             fmt='o', color='red', capsize=3, label='This work, Vmax',
             markersize=8)
plt.errorbar(redshiftsbin[MhaloCosmosTinker[:,0].astype('int')[7:]], MhaloCosmosTinker[7:, 1], yerr= 0.1, # yerr=MhaloCosmosTinker[7:, 2],
             fmt='o', linestyle='none', color='red', capsize=3, markeredgewidth=2,  lolims=True,
             markersize=6)
plt.errorbar(redshiftsbin[MhaloCosmosSchTinker[:,0].astype('int')[:-3]], MhaloCosmosSchTinker[:-3, 1], yerr=MhaloCosmosSchTinker[:-3, 2],
             fmt='o', color='green', capsize=3, label='This work, Schechter fit',
             markersize=8)
plt.errorbar(redshiftsbin[MhaloCosmosSchTinker[:,0].astype('int')[7:]], MhaloCosmosSchTinker[7:, 1], yerr= 0.1, # yerr=MhaloCosmosTinker[7:, 2],
             fmt='o', linestyle='none', color='green', capsize=3, markeredgewidth=2,  lolims=True,
             markersize=6)
# plt.errorbar(MhaloCosmosMCMC[:-3, 0], MhaloCosmosMCMC[:-3, 1], yerr=MhaloCosmosMCMC[:-3, 2],
#              fmt='o', color='red', capsize=3, label='AM, COSMOS + Bolshoi', # label='AM, COSMOS + Bolshoi Tot'
#              markersize=7)
# plt.errorbar(MhaloCosmosMCMC[7:, 0], MhaloCosmosMCMC[7:, 1], yerr=MhaloCosmosMCMC[7:, 2],
#              fmt='-', linestyle='none', color='red', capsize=3, lolims=True,
#              markersize=7, alpha=0.5)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.xticks(np.arange(0, 5.5, 0.5))
plt.xlabel('Redshift', size=17)
plt.ylabel('$\mathrm{log}_{10}(M_{\mathrm{h}}^{\mathrm{peak}}/M_{\odot})$', size=17)
# plt.ylim(11.7, 13)
plt.ylim(11.7, 13.8)
plt.xlim(0, 5.5)
# plt.xlim(0.2, 8)
# plt.xscale('log')
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])

# Put a legend below current axis
# ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", fontsize=12)
# ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.legend(loc=2, ncol=2, fontsize=12, edgecolor='white', framealpha=0)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.tight_layout(rect=[0,0,0.65,1])
plt.tight_layout()
plt.show()