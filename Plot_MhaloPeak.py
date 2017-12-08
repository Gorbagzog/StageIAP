#!/usr/bin/env python3
# -*-coding:Utf-8 -*

"""Script to plot MhaloPeak found with different scripts"""

import numpy as np
import matplotlib.pyplot as plt

"""Load different MhaloPeak values"""

MhaloTinker = np.loadtxt("../Plots/MhPeak/Tinker08_Dm200.txt")
MhaloCosmos = np.loadtxt("../Plots/MhPeak/COSMOS.txt")
MhaloCandels = np.loadtxt("../Plots/MhPeak/Candels.txt")


"""Definition of the evolution of Mpeak for Leauthaud et al, Behroozi et al et Moster et al"""

# TODO : check that there aure all in the same cosmo (h=0.7)

# Leauthaud+17 use a different cosmology with H0=72
redshiftLeauthaud = np.array([(0.22 + 0.48) / 2, (0.48 + 0.74) / 2, (0.74 + 1) / 2])
MhaloPeakLeauthaud = np.log10(np.array([9.5 * 10**11, 1.45 * 10**12, 1.4 * 10**12]))
MhaloSigmaLeauthaud = np.log10(np.array(
    [1.05 * 10**12, 1.55 * 10**12, 1.5 * 10**12])) - MhaloPeakLeauthaud
MstarPeakLeauthaud = np.array([3.55 * 10**10, 4.9 * 10**10, 5.75 * 10**10])
MstarSigmaLeauthaud = np.array([0.17, 0.15, 0.13])*10**10

redshiftBehroozi = np.array([])

redshiftCoupon15 = np.array([0.75])
MhaloPeakCoupon15 = np.log10(np.array([1.92*10**12]))
MhaloSigmaCoupon15 = np.array([[np.log10((1.92 + 0.17)) - np.log10(1.92)],
                               [np.log10(1.92) - np.log10(1.92 - 0.14)]])

redshiftCoupon12 = np.array([0.3, 0.5, 0.7, 1])
MhaloPeakCoupon12 = np.array([11.65, 11.79, 11.88, 11.9]) - np.log10(0.7)
MhaloSigmaCoupon12 = np.vstack([[0.05, 0.05, 0.06, 0.07], [0.05, 0.05, 0.06, 0.07]])

redshiftMartinezManso2014 = 1.5
MhaloPeakMartinezManso2014 = 12.44
MhaloSigmaMartinezManso2014 = [[0.08], [0.08]]

# Test graphic reading of McCracken+15 Mpeak
redshiftMcCracken15 = np.array([0.65, 0.95, 1.3, 1.75, 2.25])
MhaloPeakMcCracken15 = np.array([12.15, 12.1, 12.2, 12.35, 12.4])

# Load Coupon+17 draft Peak values
# We use PeakPosMCMCMean and PeakPosMCMCstd
# Values are given in Log10(Mh*h^-1 Msun)
redshiftCoupon17 = np.array([0.34, 0.52, 0.70, 0.90, 1.17, 1.50,
                            1.77, 2.15, 2.75, 3.37, 3.96, 4.83])
MhaloPeakCoupon17 = np.zeros([np.size(redshiftCoupon17)])
MhaloSigmaCoupon17 = np.zeros([np.size(redshiftCoupon17)])
for i in range(len(redshiftCoupon17)):
    MhaloPeakCoupon17[i], MhaloSigmaCoupon17[i] = np.loadtxt(
        '../Data/Coupon17/peak/peak_{:1.2f}.ascii'.format(redshiftCoupon17[i]),
        usecols=(2, 3))

# Graphic read of MhaloPeakvalues in Yang+2012 for SMF1 :
redshiftYang12 = np.array([0.1, 0.5, 1.15, 1.80, 2.75, 3.75])
MhaloPeakYang12 = np.array([11.75, 11.7, 12, 12.5, 12.7, 12.8]) - np.log10(0.7)
MhaloSigmaYang12 = np.array([[0.2, 0.2, 0.2, 0.2, 0.3, 0.4], [0.2, 0.2, 0.3, 0.4, 0.6, 1]])


"""Plot"""

plt.figure(figsize=(8, 4))
ax = plt.subplot(111)

plt.errorbar(MhaloCosmos[:-2, 0], MhaloCosmos[:-2, 1], yerr=[MhaloCosmos[:-2, 2], MhaloCosmos[:-2, 3]],
             fmt='o', color='red', capsize=5, label='SMF:D17, HMF:Bolshoï')
plt.errorbar(MhaloTinker[:-2, 0], MhaloTinker[:-2, 1], yerr=[MhaloTinker[:-2, 2], MhaloTinker[:-2, 3]],
             fmt='o', color='peru', capsize=5, label='SMF:D17, HMF:T08')
plt.errorbar(MhaloCandels[:, 0], MhaloCandels[:, 1], yerr=[MhaloCandels[:, 2], MhaloCandels[:, 3]],
             fmt='o', color='green', capsize=5, label='SMF:Candels, HMF:Bolshoï')
# plt.errorbar(redshiftCoupon17, MhaloPeakCoupon17 - np.log10(0.7),
#              yerr=MhaloSigmaCoupon17,
#              fmt='o', color='blue', capsize=5, label='Coupon et al. 2017 Draft')
plt.errorbar(redshiftLeauthaud, MhaloPeakLeauthaud + np.log10(72/70),
             yerr=MhaloSigmaLeauthaud,
             fmt='d', c='lightcoral', markerfacecolor='white', capsize=5, label='Leauthaud et al. 2011')
plt.errorbar(redshiftCoupon12, MhaloPeakCoupon12, yerr=MhaloSigmaCoupon12,
             fmt='d', c='dodgerblue', markerfacecolor='white', capsize=5, label='Coupon et al. 2012')
plt.errorbar(redshiftCoupon15, MhaloPeakCoupon15, yerr=MhaloSigmaCoupon15,
             fmt='d', c='turquoise', markerfacecolor='white', capsize=5, label='Coupon et al. 2015')
plt.errorbar(redshiftMartinezManso2014, MhaloPeakMartinezManso2014,
             yerr=MhaloSigmaMartinezManso2014,
             fmt='d', c='purple', markerfacecolor='white', capsize=5, label='Martinez-Manso et al. 2014')
# plt.errorbar(redshiftYang12, MhaloPeakYang12, yerr= MhaloSigmaYang12,
#              fmt='d', c='lightgreen', markerfacecolor='white', capsize=5, label='Yang et al. 2012')

# plt.errorbar(redshiftMcCracken15, MhaloPeakMcCracken15,
#              fmt='d', markerfacecolor='white', capsize=5, label='"Revised" McCracken15')
plt.xlabel('Redshift', size=20)
plt.ylabel('Log($\mathrm{M_{halo}^{peak}}/\mathrm{M_{\odot}}$)', size=20)
plt.ylim(11.7, 13)
plt.xlim(0,8)
# plt.xlim(0.2, 8)
# plt.xscale('log')
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])

# Put a legend below current axis
# ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", fontsize=12)
ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout(rect=[0,0,0.65,1])
plt.show()