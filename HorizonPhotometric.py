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
Htrue.rename(columns={'index': 'True_gal_idx'}, inplace=True)
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
    # for each redshift bin, gives the idx of the observed central galaxy of the halo
    df_tmp[i] = Htrue[Htrue.zbin == i].reset_index()[
        'Photo_gal_idx'][hal_centgal[i] - 1].reset_index()
    df_tmp[i].rename(columns={'index': 'Central_gal_idx'}, inplace=True)

# Concat all bins and add the columns to the Hhalo dataframe
df_tmp2 = pd.concat([df_tmp[0], df_tmp[1], df_tmp[2]], axis=0)
# df_tmp2.rename(columns={'index': 'Halo_idx'}, inplace=True)
Hhalo = pd.concat([Hhalo, df_tmp2], axis=1)

"""Plot Halo Mass vs Observed Central Galaxies Mass"""

# for i in range(numzbin-1):
#     plt.figure()
#     tmp = Hhalo[['Mass', 'Photo_gal_idx']].loc[
#         (Hhalo['zbin'] == i) & (Hhalo['Photo_gal_idx'].notnull())
#         ]
#     plt.hist2d(
#         np.log10(tmp.Mass * 10**11),
#         Hphoto['Mass'].iloc[tmp.Photo_gal_idx],
#         bins=100,
#         cmin=1,
#         range=[[10, 14.5], [7, 12]])
#     plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
#     plt.ylabel('Observed Log($M_{*}$) [Log($M_{\odot}$)]', size=15)
#     plt.title('Observed central gal in halos, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
#     plt.savefig('../Plots/HAGN_Matching/ObsMass_HaloMass' +
#                 str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')


""" Plot M*_true(Mh)"""

for i in range(numzbin-1):
    plt.figure()
    tmp = Hhalo[['Mass', 'Central_gal_idx']].loc[
        (Hhalo['zbin'] == i) & (Hhalo['Central_gal_idx'] >= 0)
        ]
    plt.hist2d(
        np.log10(tmp.Mass * 10**11),
        np.log10(Htrue[Htrue.zbin == i]['Mass'].iloc[tmp.Central_gal_idx]*10**11),
        bins=100,
        cmin=1,
        range=[[10, 14.5], [7, 12]])
    plt.colorbar()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
    plt.ylabel('Observed Log($M_{*}$) [Log($M_{\odot}$)]', size=15)

"""Select Median of Halo Mass for each stellar mass bin"""

stellarmassbins = np.linspace(8.1, 12, num=100)
avHMperSM = np.zeros([numzbin, np.size(stellarmassbins)-1])
medHMperSM = np.zeros([numzbin, np.size(stellarmassbins)-1])

for i in range(numzbin):
    for j in range(np.size(stellarmassbins)-1):
        m1 = stellarmassbins[j]
        m2 = stellarmassbins[j+1]
        HtrueMassSelec = Htrue[Htrue.zbin == i].loc[
            Htrue[Htrue.zbin == i]['True_gal_idx'].isin(Hhalo[Hhalo.zbin == i]['Central_gal_idx']),
            'Mass']
        Hhalo.loc[(HtrueMassSelec >= 10**(m1-11)) & (HtrueMassSelec < 10**(m2-11)), '']
# NOT WORKING

# for i in range(4):
#     for j in range(np.size(stellarmassbins)-1):
#         m1 = stellarmassbins[j]
#         m2 = stellarmassbins[j+1]
#         ## select indices of central galaxies with a mass
#         ## between m1 and m2 :
#         indices = np.where(np.logical_and(
#         select_gal_hal_mass[i][0]>m1,
#         select_gal_hal_mass[i][0]<m2))
#         #indices_cent = np.intersect1d(indices, central_gal[i])
#         sel_avHMperSM[i,j] = np.average(select_gal_hal_mass[i][1][indices])
#         sel_medHMperSM[i,j] = np.median(select_gal_hal_mass[i][1][indices])
