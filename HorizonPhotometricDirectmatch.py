#!/usr/bin/env python3
# -*-coding:Utf-8 -*

"""H-AGN LightCone photometric Catalog.

Load catalog and make a match with the true lightcone catalog.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfits
from scipy.spatial import cKDTree
from timeit import default_timer as timer

start = timer()
"""Load Horizon-AGN Lightcone Photometric catalog."""
col_names = ['Id', 'Ra', 'Dec', 'zphot', 'zphot_err', 'Mass', 'Mass_err', 'mag_u', 'magerr_u',
             'mag_B', 'magerr_B', 'mag_V', 'magerr_V', 'mag_r', 'magerr_r', 'mag_i', 'magerr_i',
             'mag_z', 'magerr_z', 'mag_Y', 'magerr_Y', 'mag_J', 'magerr_J', 'mag_H', 'magerr_H',
             'mag_K', 'magerr_K', 'SFR']

Hphoto = pd.read_table(
    '../Data/HorizonAGNLightconePhotometric/Salp_0.0-3.0_dust_v15c.in_Init_Small',
    sep=' ', skipinitialspace=True, header=None, names=col_names)

"""Load Horizon-AGN Lightcone original galaxies catalogs."""
zbins_Cone = np.array([0, 1, 2, 3, 6])
numzbin = np.size(zbins_Cone)-1
Htrue = [None]*(numzbin-1)
# The Photometric catalogs stops at z=3, so no need to take the last section of the lightcone.
for i in range(numzbin-1):
    with pyfits.open('../Data/HorizonAGNLaigleCatalogs/Galaxies_' +
                     str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+'.fits') as data:
        Htrue[i] = pd.DataFrame(data[1].data)
        Htrue[i] = Htrue[i].apply(lambda x: x.values.byteswap().newbyteorder())
        Htrue[i].loc[:, 'zbin'] = i

Htrue = pd.concat((Htrue[i] for i in range(numzbin-1))).reset_index()
end = timer()
print(end - start)
# Htrue = Htrue.sort_values('Ra')


"""Load Horizon-AGN Lighcone Halos catalogs"""

Hhalo = [None]*(numzbin-1)
# The Photometric catalogs stops at z=3, so no need to take the last section of the lightcone.
for i in range(numzbin-1):
    with pyfits.open('../Data/HorizonAGNLaigleCatalogs/Haloes_' +
                     str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+'.fits') as data:
        Hhalo[i] = pd.DataFrame(data[1].data)
        Hhalo[i] = Hhalo[i].apply(lambda x: x.values.byteswap().newbyteorder())
        Hhalo[i].loc[:, 'zbin'] = i

Hhalo = pd.concat((Hhalo[i] for i in range(numzbin-1)))


"""Load catalog matching halos to their central galaxies"""
# Contains the IDs (starts at 1) of the central galaxy of each halo
hal_centgal = []
for i in range(np.size(zbins_Cone)-1):
    hal_centgal.append(
        np.loadtxt('../Data/HorizonAGNLaigleCatalogs/Cat_' +
                   str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+'_Hal_CentralGal_new.txt',
                   dtype='i4'))


"""Round Htrue Ra and Dec to be the same as Hphoto"""

Htrue[['Ra', 'Dec']] = np.round(Htrue[['Ra', 'Dec']], decimals=5)
Hphoto[['Ra', 'Dec']] = np.round(Hphoto[['Ra', 'Dec']], decimals=5)

Hmerge = pd.merge(Htrue, Hphoto, on=['Ra', 'Dec'], how='inner')

"""Plot Match for redshift"""
# plt.hist2d(test.zphot, test.z, bins=100, cmin=1, range=[[0, 3], [0,3]])
# plt.ylabel('True Z')
# plt.xlabel('Photo Z')

"""Plot the Halo Mass vs Central Gal Mass"""

# TODO selectionner les ID des centrales gal en fonction de l'index des halos

# for idx_halo in range(len(hal_centgal[0])):
#     test.set_value(hal_centgal[0][idx_halo]-1, 'idx_halo', idx_halo)

haloToCentgal = pd.concat(
    [pd.DataFrame(hal_centgal[0]),
     pd.DataFrame(hal_centgal[1]),
     pd.DataFrame(hal_centgal[2])])

haloToCentgal = haloToCentgal.reset_index()
haloToCentgal.rename(columns={'index': 'Halo_idx', 0: 'ID_cent_gal'}, inplace=True)

Hmerge.rename(columns={'ID': 'ID_cent_gal'}, inplace=True)
test = pd.merge(Hmerge, haloToCentgal, how='inner', on='ID_cent_gal')

test['Halo_idx'] = test['Halo_idx']+1
test.rename(columns={'Halo_idx': 'ID_halo'}, inplace=True)
Hhalo.rename(columns={'ID': 'ID_halo'}, inplace=True)
testor = pd.merge(test, Hhalo, how='inner', on='ID_halo')


plt.hist2d(np.log10(testor['Mass']*10**11), np.log10(testor['Mass_x']*10**11), bins=100, cmin=1)