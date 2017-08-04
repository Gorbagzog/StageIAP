#!/usr/bin/env python3
# -*-coding:Utf-8 -*

"""H-AGN LightCone photometric Catalog.

Load catalog and make a match with the true lightcone catalog.
"""


import numpy as np
import matplotlib.pyplot as plt
import pyfits
# from scipy.spatial import cKDTree
# from timeit import default_timer as timer
import numpy.lib.recfunctions as rfn


"""Load true galdata from the H-AGN Lightcone"""

zbins_Cone = np.array([0, 1, 2, 3, 6])
numzbin = np.size(zbins_Cone)-1

galdata = []
for i in range(np.size(zbins_Cone)-1):
    hdulist = pyfits.open('../Data/HorizonAGNLaigleCatalogs/Galaxies_' +
                          str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+'.fits')
    galdata.append(hdulist[1].data)
    hdulist.close()
# cols = hdulist[1].columns
# cols.info()

"""It looks like the good catalogs to use are the Haloes and not the Halos"""

halodata = []
for i in range(np.size(zbins_Cone)-1):
    hdulist2 = pyfits.open('../Data/HorizonAGNLaigleCatalogs/Haloes_' +
                           str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+'.fits')
    halodata.append(hdulist2[1].data)
    hdulist2.close()


"""Load Horizon-AGN Lightcone Photometric catalog."""

col_names = ['Id', 'Ra', 'Dec', 'zphot', 'zphot_err', 'Mass', 'Mass_err', 'mag_u', 'magerr_u',
             'mag_B', 'magerr_B', 'mag_V', 'magerr_V', 'mag_r', 'magerr_r', 'mag_i', 'magerr_i',
             'mag_z', 'magerr_z', 'mag_Y', 'magerr_Y', 'mag_J', 'magerr_J', 'mag_H', 'magerr_H',
             'mag_K', 'magerr_K', 'SFR']

galphot = np.genfromtxt(
    '../Data/HorizonAGNLightconePhotometric/Salp_0.0-3.0_dust_v15c.in_Init_Small',
    names=col_names)

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

# galdata_allz = np.concatenate((galdata[0], galdata[1], galdata[2]))

# start = timer()

# kdtree = cKDTree(np.transpose([galdata_allz['Ra'], galdata_allz['Dec']]))
# obstotrue = np.apply_along_axis(kdtree.query, 0, [galphot['Ra'], galphot['Dec']])
# obstotrue[1][:] = obstotrue[1][:].astype('int')
# # add index of true gal corresponding to each observed gal
# galphot = rfn.append_fields(galphot, ['Distance', 'True_gal_idx'],  obstotrue, usemask=False)

# # add index of observed gal to each true gal
# truetoobs = np.empty(galdata_allz.shape)
# truetoobs[:] = np.nan
# for idx_obs in range(len(obstotrue[0])):
#     truetoobs[obstotrue[1][idx_obs].astype('int')] = idx_obs

# galdata_allz = rfn.append_fields(galdata_allz, 'Obs_gal_idx',  truetoobs, usemask=False)


# end = timer()
# print('Positional match took :' + str(end - start))

"""Use the match Catalaog of Clotilde"""

galdata_allz = np.concatenate((galdata[0], galdata[1], galdata[2]))

obstotrue = np.loadtxt('../Data/HorizonAGNLightconePhotometric/Match.dat')

# I prefer to work with index (starts at 0) than with ID (starts at 1)
obstotrue = obstotrue[:, 1] - 1

# add index of observed gal to each true gal
truetoobs = np.empty(galdata_allz.shape)
truetoobs[:] = -1
for idx_obs in range(len(obstotrue)):
    truetoobs[obstotrue[idx_obs].astype('int')] = idx_obs

galdata_allz = rfn.append_fields(galdata_allz, 'Obs_gal_idx',  truetoobs, usemask=False)


"""Compute median, average and percentiles"""

# For true catalog
stellarmassbins = np.linspace(9, 12, num=100)
avHMperSM = np.zeros([numzbin, np.size(stellarmassbins)-1])
medHMperSM = np.zeros([numzbin, np.size(stellarmassbins)-1])
for i in range(numzbin):
    for j in range(np.size(stellarmassbins)-1):
        m1 = stellarmassbins[j]
        m2 = stellarmassbins[j+1]
        # select indices of central galaxies with a mass
        # between m1 and m2 :
        indices = np.where(
            np.logical_and(
                np.logical_and(
                    np.log10(galdata[i]['Mass'][hal_centgal[i]-1]*10**11) > m1,
                    np.log10(galdata[i]['Mass'][hal_centgal[i]-1]*10**11) <= m2
                    ),
                hal_centgal[i] > 0
            )
        )
        avHMperSM[i, j] = np.average(np.log10(halodata[i]['Mass'][indices] * 10**11))
        medHMperSM[i, j] = np.median(np.log10(halodata[i]['Mass'][indices] * 10**11))

# For photometric catalog
stellarmassbins = np.linspace(9, 12, num=100)
avHMperSMPhot = np.zeros([numzbin, np.size(stellarmassbins)-1])
medHMperSMPhot = np.zeros([numzbin, np.size(stellarmassbins)-1])
for i in range(numzbin-1):
    for j in range(np.size(stellarmassbins)-1):
        m1 = stellarmassbins[j]
        m2 = stellarmassbins[j+1]
        # select indices of central galaxies with a mass
        # between m1 and m2 :
        indices = np.where(
            np.logical_and(
                np.logical_and(
                    hal_centgal[i] > 0,
                    galdata_allz['Obs_gal_idx'][
                        hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
                    ] > 0
                ),
                np.logical_and(
                    galphot['Mass'][
                        galdata_allz['Obs_gal_idx'][
                            hal_centgal[i][:] - 1 +
                            sum(len(galdata[j]) for j in range(i))
                        ].astype('int')
                    ] > m1,
                    galphot['Mass'][
                        galdata_allz['Obs_gal_idx'][
                            hal_centgal[i][:] - 1 +
                            sum(len(galdata[j]) for j in range(i))
                        ].astype('int')
                    ] <= m2
                ),
            )
        )
        avHMperSMPhot[i, j] = np.average(np.log10(halodata[i]['Mass'][indices] * 10**11))
        medHMperSMPhot[i, j] = np.median(np.log10(halodata[i]['Mass'][indices] * 10**11))


# stellarmassbins = np.linspace(8.1, 12, num=100)
# first_per = np.zeros([numzbin, np.size(stellarmassbins)-1])
# last_per = np.zeros([numzbin, np.size(stellarmassbins)-1])

# for i in range(numzbin):
#     for j in range(np.size(stellarmassbins)-1):
#         m1 = stellarmassbins[j]
#         m2 = stellarmassbins[j+1]
#         # select indices of central galaxies with a mass
#         # between m1 and m2 :
#         indices = np.where(
#             np.logical_and(
#                 np.logical_and(
#                     np.log10(galdata[i]['Mass'][hal_centgal[i]-1]*10**11) > m1,
#                     np.log10(galdata[i]['Mass'][hal_centgal[i]-1]*10**11) <= m2
#                     ),
#                 hal_centgal[i] > 0
#             )
#         )
#         if indices[0].size : #check if the array is not empty
#             first_per[i,j] = np.percentile(np.log10(
#             halodata[i]['Mass'][gal_mainhaloes[i][centGalInCentHalo[i][indices]]-1]*10**11), 10)
#             last_per[i,j] = np.percentile(np.log10(
#             halodata[i]['Mass'][gal_mainhaloes[i][centGalInCentHalo[i][indices]]-1]*10**11), 90)
#         else:
#             first_per[i,j] = numpy.nan
#             last_per[i,j] = numpy.nan

"""Compute average and median Ms for a given Mh"""

massbins = np.linspace(10, 15, num=100)
avSMperHM = np.zeros([numzbin, np.size(massbins)-1])
medSMperHM = np.zeros([numzbin, np.size(massbins)-1])

for i in range(4):
    for j in range(np.size(massbins)-1):
        m1 = massbins[j]
        m2 = massbins[j+1]
        # select indices of galaxies contained in the haloes with a mass
        # between m1 and m2 :
        indices = np.where(np.logical_and(
            np.log10(halodata[i]['Mass']*10**11) > m1,
            np.log10(halodata[i]['Mass']*10**11) <= m2))[0]
        # indices_cent = np.intersect1d(indices, halodata[i]['level'] == 1)
        if len(indices)>0:
            avSMperHM[i, j] = np.average(
                np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11))
            medSMperHM[i, j] = np.median(
                np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11))
        else:
            avSMperHM[i, j] = np.nan
            medSMperHM[i, j] = np.nan

"""Plot Ms(Mh) for true galaxies and level 1 halos"""

for i in range(numzbin):
    plt.figure()
    indices = np.where(np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1))
    # verification that all galaxies selected are central
    # print(galdata[i]['level'][hal_centgal[i][indices]-1].min())
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11),
        bins=100, cmin=1)
    plt.colorbar()
    plt.scatter((massbins[:-1]+massbins[1:])/2, avSMperHM[i][:], color='red',
                label='Average SM for a given HM')
    plt.scatter((massbins[:-1]+massbins[1:])/2, medSMperHM[i][:],
                color='green', label='Median SM for a given HM')
    plt.scatter(avHMperSM[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
                color='black', label='Average HM for a given SM')
    plt.scatter(medHMperSM[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
                color='pink', label='Median HM for a given SM')
    plt.legend()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Log($M_{*}$) [Log($M_{\odot}$)]', size=12)
    plt.title('HorizonAGN, Central galz='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.savefig('../Plots/HAGN_Matching/ClotMatch/TrueMass_HaloMass' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')


"""Plot Ms_observed(Mh) and level 1 halos"""

for i in range(numzbin-1):
    plt.figure()
    indices = np.where(
        np.logical_and(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
            galdata_allz['Obs_gal_idx'][
                    hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
            ] > 0
        )
    )
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        galphot['Mass'][
                galdata_allz['Obs_gal_idx'][
                    hal_centgal[i][indices] - 1 + sum(len(galdata[j]) for j in range(i))
                ].astype('int')
        ],
        bins=100, cmin=1, range=[[10, 14], [8, 12]])
    plt.colorbar()
    plt.scatter(avHMperSMPhot[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
                color='black', label='Average HM for a given SM')
    plt.scatter(medHMperSMPhot[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
                color='pink', label='Median HM for a given SM')
    plt.legend()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Log($M_{*}$) Photometric [Log($M_{\odot}$)]', size=12)
    plt.title('HorizonAGN, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.savefig('../Plots/HAGN_Matching/ClotMatch/PhotoMass_HaloMass' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')

"""Plot Ms/Mh vs Mh for true and photometric catalogs"""

plt.figure()
cmap = ['blue', 'green', 'red']
marker = ['v', '>', '^']
for i in range(numzbin-1):
    plt.scatter(
        medHMperSM[i],
        (stellarmassbins[:-1]+stellarmassbins[1:]) / 2 - medHMperSM[i],
        label='True catalog, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]),
        edgecolors=cmap[i], facecolors='none'
    )
    plt.scatter(
        medHMperSMPhot[i],
        (stellarmassbins[:-1]+stellarmassbins[1:]) / 2 - medHMperSM[i],
        label='Phot catalog, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]),
        edgecolors=cmap[i], facecolors=cmap[i]
    )
    plt.legend()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
    plt.ylabel('Log($M_{s}/M_{h}$)', size=15)
    plt.title('H-AGN, Central gal and level 1 halos')


"""Plot sSFR vs Mh for true catalogs"""

for i in range(numzbin):
    plt.figure()
    indices = np.where(np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1))
    # verification that all galaxies selected are central
    # print(galdata[i]['level'][hal_centgal[i][indices]-1].min())
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        np.log10(galdata[i]['SFRCorr'][hal_centgal[i][indices]-1] /
                 (galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11)),
        bins=100, cmin=1, range=[[10, 14], [-12, -8]])
    plt.colorbar()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Log(sSFR) [Log($yr^{-1}$)]', size=12)
    plt.title('HorizonAGN, Central galz='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.savefig('../Plots/HAGN_Matching/ClotMatch/TrueSpecificSFR_HaloMass' +
                str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')

# TODO : compute median sSFR for true and photo galaxies

"""Plot SFR vs Mh for photo catalogs"""

# TODO plot only for Ms > 10**9

for i in range(numzbin-1):
    plt.figure()
    indices = np.where(
        np.logical_and(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
            galdata_allz['Obs_gal_idx'][
                    hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
            ] > 0
        )
    )
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        np.log10(galphot['SFR'][
            galdata_allz['Obs_gal_idx'][
                hal_centgal[i][indices] - 1 + sum(len(galdata[j]) for j in range(i))
            ].astype('int')
        ]),
        bins=100, cmin=1)
    plt.colorbar()
    plt.legend()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Log(SFR) Photometric [Log($M_{\odot}/yr$)]', size=12)
    plt.title('HorizonAGN, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.savefig('../Plots/HAGN_Matching/ClotMatch/PhotoSFR_HaloMass' +
                str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')

"""PLot sSFR vs Mh for photo cat"""

for i in range(numzbin-1):
    plt.figure()
    indices = np.where(
        np.logical_and(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
            galdata_allz['Obs_gal_idx'][
                    hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
            ] > 0
        )
    )
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        np.log10(galphot['SFR'][
            galdata_allz['Obs_gal_idx'][
                hal_centgal[i][indices] - 1 + sum(len(galdata[j]) for j in range(i))
            ].astype('int')
        ] / 10**(galphot['Mass'][
            galdata_allz['Obs_gal_idx'][
                hal_centgal[i][indices] - 1 + sum(len(galdata[j]) for j in range(i))
            ].astype('int')])
        ),
        bins=100, cmin=1, range=[[10, 14], [-13.5, -6.5]])
    plt.colorbar()
    plt.legend()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Log(sSFR) Photometric [Log($yr^{-1}$)]', size=12)
    plt.title('HorizonAGN, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.savefig('../Plots/HAGN_Matching/ClotMatch/PhotoSpecificSFR_HaloMass' +
                str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')


"""Gas Met vs Mh for photo catalog"""

# Load gas mass and gas met
gas_mass, gas_met = np.loadtxt('../Data/HorizonAGNLightconePhotometric/GasInfo.dat', unpack=True)

# Add a column with gas mass and metalicity in galphot catalog

galphot = rfn.append_fields(galphot, 'Gas_mass',  gas_mass, usemask=False)
galphot = rfn.append_fields(galphot, 'Gas_met',  gas_met, usemask=False)


def boost(z):
    """Boost the metalicity of gas and stars because of the low resolution of H-AGN."""
    return 4.08430 - 0.213574 * z - 0.111197 * z**2


# Compute boosted Metalicity for photometric catalog
gas_met_boost = np.empty(gas_met.shape)
for idx_phot in range(len(gas_met_boost)):
    gas_met_boost[idx_phot] = gas_met[idx_phot] * boost(
        galdata_allz['z'][obstotrue[idx_phot].astype('int')])

# Add a column on gal_phot
galphot = rfn.append_fields(galphot, 'Gas_met_boost',  gas_met_boost, usemask=False)

"""Plot Gas metalicity vs Mh"""

for i in range(numzbin-1):
    plt.figure()
    indices = np.where(
        np.logical_and(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
            galdata_allz['Obs_gal_idx'][
                    hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
            ] > 0
        )
    )
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        galphot['Gas_met_boost'][
            galdata_allz['Obs_gal_idx'][
                hal_centgal[i][indices] - 1 + sum(len(galdata[j]) for j in range(i))
            ].astype('int')
        ],
        bins=100, cmin=1)
    plt.colorbar()
    plt.legend()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Gas Metalicity Photometric', size=12)
    plt.title('HorizonAGN, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))



"""Plot Stellar Met vs Mh for true and photo catalogs"""
