"""Scrit to plot Horizon AGN and COSMOS on the same plot.

First need to load the data form AM_COSMOSIari_BolsoiPlanck.py and HorizonAGNE.py
"""

import numpy as np
import pyfits
import matplotlib.pyplot as plt

"""Plot Ms vs Mh in COSMOS"""

plt.figure()
for i in range(numzbin):
    plt.plot(xm[i][:], MstarIary[i](x[i]), label=str(redshifts[i]) + '<z<' + str(redshifts[i + 1]))
    plt.fill_between(xm[i], MstarIaryMinus[i](x[i]), MstarIaryPlus[i](x[i]), alpha=0.5)
plt.legend()
plt.ylabel('Log($M_{*}$)  [Log($M_{\odot}$)]', size=20)
plt.xlabel('Log($M_{h}$)  [Log($M_{\odot}$)]', size=20)
plt.tight_layout()
# plt.title('IariDavidzon Mass Function vs Bolshoï simulation')
plt.show()

"""Plot Ms vs Mh in HAGN"""

select = [[0, 1, 2], [3, 4], [5, 6], [7, 8, 9]]
for i in range(4):
    plt.figure()
    indices = np.where(np.logical_and(hal_centgal[i]>0, halodata[i]['level']==1 ))
    # verification that all galaxies selected are central
    # print(galdata[i]['level'][hal_centgal[i][indices]-1].min())
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11),
        bins=100, cmin=1, norm=mpl.colors.LogNorm(), alpha=0.95)
    plt.colorbar()
    for j in select[i]:
        plt.plot(xm[j][:], MstarIary[j](x[j]), label=str(redshifts[j]) + '<z<' + str(redshifts[j + 1]))
        plt.fill_between(xm[j], MstarIaryMinus[j](x[j]), MstarIaryPlus[j](x[j]), alpha=0.7)
    plt.legend()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Log($M_{*}$) [Log($M_{\odot}$)]', size=12)
    plt.title('HorizonAGN, hal_centralgal_new, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))

