#!/usr/bin/env python3
# -*-coding:Utf-8 -*

"""H-AGN LightCone photometric Catalog.

Load catalog and make a match with the true lightcone catalog.
"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import pyfits
from scipy.spatial import cKDTree

"""Load Horizon-AGN Lightcone Photometric catalog."""
col_names = ['Id', 'Ra', 'Dec', 'zphot', 'zphot_err', 'Mass', 'Mass_err', 'mag_u', 'magerr_u',
             'mag_B', 'magerr_B', 'mag_V', 'magerr_V', 'mag_r', 'magerr_r', 'mag_i', 'magerr_i',
             'mag_z', 'magerr_z', 'mag_Y', 'magerr_Y', 'mag_J', 'magerr_J', 'mag_H', 'magerr_H',
             'mag_K', 'magerr_K', 'SFR']

Hphoto = pd.read_table(
    '../Data/HorizonAGNLightconePhotometric/Salp_0.0-3.0_dust_v15c.in_Init_Small',
    sep=' ', skipinitialspace=True, header=None, names=col_names)
# Hphoto = Hphoto.sort_values('Ra')

"""Load Horizon-AGN Lightcone true catalogs."""
zbins_Cone = np.array([0, 1, 2, 3, 6])
numzbin = np.size(zbins_Cone)-1
Htrue = [None]*numzbin

for i in range(numzbin):
    with pyfits.open('../Data/HorizonAGNLaigleCatalogs/Galaxies_' +
                     str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+'.fits') as data:
        Htrue[i] = pd.DataFrame(data[1].data)
        Htrue[i] = Htrue[i].apply(lambda x: x.values.byteswap().newbyteorder())
        Htrue[i].loc[:, 'zbin'] = i

Htrue = pd.concat(Htrue[i] for i in range(numzbin))
# Htrue = Htrue.sort_values('Ra')

# Simple merge between Htrue and Hphoto, no results
# print(pd.merge(Htrue, Hphoto, on=['Ra', 'Dec'], how='inner').shape)

# def find_nearest(ra, dec):
#     """Return the ID and the distance to the closest galaxy."""
#     index = (np.sqrt((Htrue.Ra - ra)**2 + (Htrue.Dec - dec)**2)).argmin()
#     # print(index)
#     distance = np.sqrt((Htrue.iloc[index].Ra - ra)**2 + (Htrue.iloc[index].Dec - dec)**2)
#     return Htrue.iloc[index].ID, distance


# Hphoto['Nearest_ID'], Hphoto['distance'] = Hphoto[['Ra', 'Dec']].apply(
#     lambda x: find_nearest(*x), axis=1)

"""Algorithm to find nearest value using a KDTree"""

kdtree = cKDTree(np.transpose([Htrue.Ra, Htrue.Dec]))

tmp = Hphoto[['Ra', 'Dec']].apply(lambda x: kdtree.query(x), axis=1)

Hphoto['Distance'] = tmp.apply(lambda x: x[0])  # Need to do this to break the tuple
Hphoto['Nearest_ID'] = tmp.apply(lambda x: x[1]+1)


# Find if there are duplactes, ie two galaxies having the same closets neigbhour
Hphoto.loc[Hphoto['Nearest_ID'].duplicated()]