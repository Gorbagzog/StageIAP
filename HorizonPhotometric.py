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
# Hphoto = Hphoto.sort_values('Ra')

# Select only positive values of Hphoto.zphot
# Hphoto = Hphoto.loc[Hphoto.zphot >= 0]

"""Load Horizon-AGN Lightcone original galaxies catalogs."""
zbins_Cone = np.array([0, 1, 2, 3, 6])
numzbin = np.size(zbins_Cone)-1
Htrue = [None]*(numzbin-1)
# The Photometric catalogs stops at z=3, so no need to take the last section of the lightcone.
#for i in range(1:numzbin-1):
for i in [2]:
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
#for i in [2]:
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


""" Simple merge between Htrue and Hphoto, no results"""

# print(pd.merge(np.round(Htrue, decimals=5), Hphoto, on=['Ra', 'Dec'], how='inner').shape)

# def find_nearest(ra, dec):
#     """Return the ID and the distance to the closest galaxy."""
#     index = (np.sqrt((Htrue.Ra - ra)**2 + (Htrue.Dec - dec)**2)).argmin()
#     # print(index)
#     distance = np.sqrt((Htrue.iloc[index].Ra - ra)**2 + (Htrue.iloc[index].Dec - dec)**2)
#     return Htrue.iloc[index].ID, distance


# Hphoto['Nearest_ID'], Hphoto['distance'] = Hphoto[['Ra', 'Dec']].apply(
#     lambda x: find_nearest(*x), axis=1)

"""Algorithm to find nearest value using a KDTree.

We make a match between nearest galaxies in projection on the sky.
Maybe we should also take into account the third dimension, to have
a better match. But it will give more importance to the error in redshift
in the observed catalog."""

start = timer()
kdtree = cKDTree(np.transpose([Htrue.Ra, Htrue.Dec]))

tmp = Hphoto[['Ra', 'Dec']].apply(lambda x: kdtree.query(x), axis=1)

Hphoto['Distance'] = tmp.apply(lambda x: x[0])  # Need to do this to break the tuple
# Gives the index (starts at 0) of the nearest galaxy in th Htrue merged catalog (all redshifts)
Hphoto['Nearest_index'] = tmp.apply(lambda x: x[1])
end = timer()
print(end - start)

# Find if there are duplactes, ie two galaxies having the same closets neigbhour
# Hphoto.loc[Hphoto['Nearest_index'].duplicated()]

"""Plot Ztrue vs Zobserved"""

# plt.hist2d(
#     Htrue.iloc[Hphoto['Nearest_index']].z,
#     Hphoto.zphot,
#     bins=100, cmin=1)
# plt.xlabel('True Z')
# plt.ylabel('Observed Z')

"""Plot Mtrue vs Mobserved"""

# plt.figure()
# plt.hist2d(
#     np.log10(
#         Htrue.iloc[Hphoto.loc[Hphoto.Mass > 0]['Nearest_index']].Mass*10**11),
#     Hphoto.loc[Hphoto.Mass > 0].Mass,
#     bins=100, cmin=1)
# plt.xlabel('True Mass')
# plt.ylabel('Observed Mass')

"""Plot SFR_true vs SFR_observed"""

# plt.figure()
# plt.hist2d(
#     Htrue.iloc[Hphoto['Nearest_index']].SFRcorr,
#     Hphoto.SFR,
#     bins=100, cmin=1)
# plt.xlabel('True SFR')
# plt.ylabel('Observed SFR')


"""Link observed galaxies catalog to their DM halos"""

# Add a column in True catalog with the index of the observed galaxy.
for idx_phot in range(Hphoto.shape[0]):
    Htrue.set_value(
        Hphoto.get_value(idx_phot, 'Nearest_index'), 'Photo_gal_idx', idx_phot)


# Liste les galaxies centrales et observées pour chaque halo

df_tmp = [None]*(numzbin-1)
for i in range(numzbin-1):
#for i in [2]:
    # for each redshift bin, gives the idx of the observed central galaxy of the halo
    # df_tmp[i] = Htrue[Htrue.zbin==i]['Photo_gal_idx'][hal_centgal[i] - 1].reset_index()
    # df_tmp[i].rename(columns={'index': 'Central_gal_idx'}, inplace=True)
    df_tmp[i] = Htrue[Htrue.zbin==i].reset_index()['Photo_gal_idx'][hal_centgal[i] - 1].reset_index()
    df_tmp[i].rename(columns={'index': 'Central_gal_idx'}, inplace=True)

# Concat all bins and add the columns to the Hhalo dataframe
df_tmp2 = pd.concat([df_tmp[0], df_tmp[1], df_tmp[2]], axis=0)
# df_tmp2.rename(columns={'index': 'Halo_idx'}, inplace=True)
Hhalo = pd.concat([Hhalo, df_tmp2], axis=1)

"""Plot Halo Mass vs Observed Central Galaxies Mass"""

for i in range(numzbin-1):
#for i in [2]:
    plt.figure()
    plt.hist2d(
        np.log10(Hhalo[Hhalo['zbin'] == i]['Mass'].loc[(Hhalo[Hhalo['zbin'] == i]['Photo_gal_idx'].notnull())] * 10**11),
        Hphoto['Mass'].iloc[Hhalo[Hhalo['zbin'] == i]['Photo_gal_idx'].dropna()],
        bins=100,
        cmin=1,
        range=[[10, 14.5], [7, 12]])
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
    plt.ylabel('Observed Log($M_{*}$) [Log($M_{\odot}$)]', size=15)
    plt.title('Observed central gal in halos, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.savefig('../Plots/HorizonAGN/ObservedCat/ObsMass_HaloMass' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')

for i in range(numzbin-1):
    plt.figure()
    plt.hist2d(
        np.log10(Hhalo['Mass'].loc[
            (Hhalo['zbin'] == i) & (Hhalo['Central_gal_idx'] >= 0)] * 10**11),
        np.log10(Htrue[Htrue.zbin == i]['Mass'].iloc[
            Hhalo['Central_gal_idx'].loc[(Hhalo['zbin'] == i) & (Hhalo['Central_gal_idx'] >= 0)]]*10**11),
        bins=100,
        cmin=1,
        range=[[10, 14.5], [7, 12]])


Hphoto.loc[Hphoto.index.isin(Hhalo[Hhalo['zbin']==i]['Photo_gal_idx'])]

test = Hhalo[['Mass', 'Photo_gal_idx']].loc[(Hhalo['zbin'] == i) & (Hhalo['Photo_gal_idx'].notnull())]

for i in range(numzbin-1):
    plt.figure()
    test = Hhalo[['Mass', 'Photo_gal_idx']].loc[(Hhalo['zbin'] == i) & (Hhalo['Photo_gal_idx'].notnull())]
    plt.hist2d(
        np.log10(test['Mass'] * 10**11),
        Hphoto['Mass'].iloc[test.Photo_gal_idx],
        bins=100,
        cmin=1,
        range=[[10, 14.5], [7, 12]])