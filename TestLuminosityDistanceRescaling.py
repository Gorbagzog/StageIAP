#!/usr/bin/env python3
# -*-coding:Utf-8 -*

"""Simple script to plot the histogram of the rescaling of stellar mass in all redshift bins, and 
check if rescaling by a co,nstant value equal to the mean redshift is OK"""


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import LambdaCDM, Planck15



hdul = fits.open('../Data/Davidzon/cosmos2015_L16_v2.0_zmin-zmax.fits')
data = hdul[1].data

redshifts = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5.5])
# Average redshift of the bin from the Iary SMF fit
redshiftsbin = np.array([0.37, 0.668, 0.938, 1.286, 1.735, 2.220, 2.683, 3.271, 3.926, 4.803])

numzbin = np.size(redshifts) - 1

D17_Cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

selec = []
corrTrue = []
corrDirect = np.empty(numzbin)
for i in range(numzbin):
    selec.append(data[data['ZPHOT']>redshifts[i]])
    selec[i] = selec[i][selec[i]['ZPHOT']<redshifts[i+1]]

    dl_planck = Planck15.luminosity_distance(selec[i]['ZPHOT'])
    dl_d17 = D17_Cosmo.luminosity_distance(selec[i]['ZPHOT'])
    corrTrue.append(2 * np.log10(dl_planck / dl_d17))

    plt.figure()
    plt.hist(corrTrue[i], bins=10)
    av_dl_planck = Planck15.luminosity_distance(redshiftsbin[i])
    av_dl_d17= D17_Cosmo.luminosity_distance(redshiftsbin[i])
    corrDirect[i] = 2 * np.log10(av_dl_planck / av_dl_d17)
    plt.axvline(corrDirect[i], c='orange', label='Correction with average z of the bin')
    plt.xlabel('$\mathrm{log}_{10}(M_*)$ correction')
    plt.title(str(redshifts[i])+'<z<'+str(redshifts[i+1]))
    plt.savefig('../TestDLRescaling/zbin'+str(i)+'.pdf')

# plt.show()


# corrMax = np.empty(numzbin)
# corrMin = np.empty(numzbin)
# corrAve = np.empty(numzbin)
# corrMed = np.empty(numzbin)

# for i in range(numzbin):
#     corrMax[i] = np.max(corrTrue[i])
#     corrMin[i] = np.min(corrTrue[i])
#     corrAve[i] = np.average(corrTrue[i])
#     corrMed[i] = np.median(corrTrue[i])