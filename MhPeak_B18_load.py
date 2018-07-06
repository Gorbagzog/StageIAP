# -*-coding:Utf-8 -*

"""Load the SHMR from the Behroozi2018 data release"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import re

"""Load the raw median data, as opposed to the fitted data on this raw SMHM"""
rawDataDir = '../Data/umachine-edr/data/smhm/median_raw/'

"""List and sort files by decreasing a (increasing z)"""
fileList = sorted(glob.glob(rawDataDir + 'smhm_a*.dat'))[::-1]
numzbin = size(fileList)


pattern = re.compile('[0-1]\.\d*')

a = np.empty(numzbin)
z = np.empty(numzbin)

smhm = []
for i in range(numzbin):
    a[i] = float(pattern.findall(fileList[i])[0])
    z[i] = np.abs(np.round(1/a[i] - 1, decimals=1))
    smhm.append(np.loadtxt(fileList[i]))

"""Plot SMHM for Obs case"""
plt.figure()
# for i in range(numzbin)[::3]:
for i in [0]:
    select = np.where(smhm[i][:, 1] < 0)[0]
    plt.errorbar(
        smhm[i][select, 0], smhm[i][select, 1],
        yerr=[smhm[i][select,2], smhm[i][select,3]], #label='z='+str(z[i]),
        )

plt.xlabel('$\mathrm{log}_{10}(M_{\mathrm{h}}/M_\odot)$', size=12)
plt.ylabel('$\mathrm{log}_{10}(M_*/M_{\mathrm{h}})$', size=12)
plt.title('Median_raw_obs')
plt.legend()

"""Plot SMHM for True case"""
plt.figure()
# for i in range(numzbin)[::3]:
for i in [0]:
    select = np.where(smhm[i][:, 1] < 0)[0]
    plt.errorbar(
        smhm[i][select, 0], smhm[i][select, 22],
        yerr=[smhm[i][select,23], smhm[i][select,24]], label='z='+str(z[i]),
        )

plt.xlabel('$\mathrm{log}_{10}(M_{\mathrm{h}}/M_\odot)$', size=12)
plt.ylabel('$\mathrm{log}_{10}(M_*/M_{\mathrm{h}})$', size=12)
plt.title('Median_raw_true')
plt.legend()


"""Compare True and Obs data"""
for i in range(numzbin)[::3]:
    plt.figure()
    select = np.where(smhm[i][:, 1] < 0)[0]
    plt.errorbar(
        smhm[i][select, 0], smhm[i][select, 22],
        yerr=[smhm[i][select,23], smhm[i][select,24]], label='True z='+str(z[i]),
        )
    plt.errorbar(
        smhm[i][select, 0], smhm[i][select, 1],
        yerr=[smhm[i][select,2], smhm[i][select,3]], label='Obs z='+str(z[i]),
        )
    plt.legend()


"""Find the halo peak mass of the SMHM, for the median raw true case"""

MhPeak = np.empty(numzbin)
ind_max = np.empty(numzbin).astype('int')
nopeak = np.empty(numzbin).astype('bool')

for i in range(numzbin):
    ind_max[i] = int(np.argmax(smhm[i][smhm[i][:, 1] < 0, 22]))
    MhPeak[i] = smhm[i][ind_max[i], 0]
    if smhm[i][ind_max[i] + 1, 22] == 0:
        nopeak[i] = True
    else:
        nopeak[i] = False

plt.plot(z[~nopeak], MhPeak[~nopeak], 'o', label='peak found')
plt.errorbar(z[nopeak], MhPeak[nopeak], yerr=0.1, lolims=True,
             fmt='-', linestyle='none', capsize=3, label='peak not found')
plt.xlabel('$z$', size=12)
plt.ylabel('$M_\mathrm{h}^{\mathrm{peak}}$', size=12)
plt.legend()
