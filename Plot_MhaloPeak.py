#!/usr/bin/env python3
# -*-coding:Utf-8 -*

"""Script to plot MhaloPeak found with different scripts"""

import numpy as np
import matplotlib.pyplot as plt

"""Load different MhaloPeak values"""

MhaloTinker = np.loadtxt("../Plots/MhPeak/Tinker08_Dm200.txt")
MhaloCosmos = np.loadtxt("../Plots/MhPeak/COSMOS.txt")
MhaloCandels = np.loadtxt("../Plots/MhPeak/Candels.txt")

redshiftCoupon17 = np.array([0.34, 0.52, 0.70, 0.90, 1.17, 1.50,
                            1.77, 2.15, 2.75, 3.37, 3.96, 4.83])
MhaloPeakCoupon17 = np.zeros([np.size(redshiftCoupon17)])
MhaloSigmaCoupon17 = np.zeros([np.size(redshiftCoupon17)])
for i in range(len(redshiftCoupon17)):
    MhaloPeakCoupon17[i], MhaloSigmaCoupon17[i] = np.loadtxt(
        '../Data/Coupon17/peak/peak_{:1.2f}.ascii'.format(redshiftCoupon17[i]),
        usecols=(2, 3))


"""Plot"""

plt.figure()
plt.errorbar(MhaloCosmos[:, 0], MhaloCosmos[:, 1], yerr=[MhaloCosmos[:, 2], MhaloCosmos[:, 3]],
             fmt='o', color='red', capsize=5, label='SMF:D+17, HMF:BolsoïPlanck')
plt.errorbar(MhaloTinker[:, 0], MhaloTinker[:, 1], yerr=[MhaloTinker[:, 2], MhaloTinker[:, 3]],
             fmt='o', color='orange', capsize=5, label='SMF:D+17, HMF:Tinker08')
plt.errorbar(MhaloCandels[:, 0], MhaloCandels[:, 1], yerr=[MhaloCandels[:, 2], MhaloCandels[:, 3]],
             fmt='o', color='green', capsize=5, label='SMF:Candels, HMF:BolsoïPlanck')
plt.errorbar(redshiftCoupon17, MhaloPeakCoupon17 - np.log10(0.7),
             yerr=MhaloSigmaCoupon17,
             fmt='o', color='blue', capsize=5, label='Coupon et al. 2017 Draft')

plt.xlabel('Reshift', size=20)
plt.ylabel('Log($M_{halo}^{peak}$) [Log($M_{\odot}$)]', size=20)
plt.legend()
plt.tight_layout()
plt.show()