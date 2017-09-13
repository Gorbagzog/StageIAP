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
# import matplotlib.mlab as mlab
import matplotlib as mpl
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

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
hal_centgal = [[] for x in range(numzbin)]
for i in range(numzbin-1):
    hal_centgal[i] = np.loadtxt('../Data/HorizonAGNLaigleCatalogs/Cat_' +
            str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+'_Hal_CentralGal_newb.txt',
            dtype='i4')
        # np.loadtxt('../Data/HorizonAGNLaigleCatalogs/Cat_' +
        #            str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+'_Hal_CentralGal_new.txt',
        #            dtype='i4'))
        # New match with Yohan algorithm to find galaxies by decreasing spheres.


"""Load halos environment.
Header is #dens dfil dnod1 dnod2.

"dens" est une estimation de la densité locale (basée sur la tesselation de delaunay)
lissée à 3Mpc, "dfiil" est la distance au filament le plus proche, "dnod1" est la distance
au noeud le plus proche, et "dnod2" la distance au noeud le plus proche en suivant le
filament. Les distances sont en Mpc.

Si tu veux pour commencer, tu pourrais separer les halos en fonction de leur distance
au filament et au noeud, e.g:
Noeuds: dnod1 < 5Mpc
Filament: dfil < 2 Mpc
Walls/voids: le reste des galaxies """

haloes_env = []
for i in range(3):
    haloes_env.append(
        np.loadtxt('../Data/HorizonAGNLaigleCatalogs/Haloes_' +
                   str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+'_env.txt'))


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

"""Use the match Catalog of Clotilde"""

galdata_allz = np.concatenate((galdata[0], galdata[1], galdata[2]))

# Load the 2 columns matching catalog, first column is the ID of the galaxy in the Photo catalog,
# the second is the ID in the original catalog, concatenated in one big catalog
# Galaxies_0-1.fits, Galaxies_1-2.fits, Galaxies_2-3.fits.
obstotrue = np.loadtxt('../Data/HorizonAGNLightconePhotometric/Match.dat')

# I prefer to work with index (starts at 0) than with ID (starts at 1), and the first column is
# useless because it is just the position in the array.
# galdata_allz[obstotrue[i]] = original galaxy corresponding to galphot[i]
obstotrue = obstotrue[:, 1] - 1

# add index of observed gal to each true gal
truetoobs = np.empty(galdata_allz.shape)
truetoobs[:] = -1
for idx_obs in range(len(obstotrue)):
    truetoobs[obstotrue[idx_obs].astype('int')] = idx_obs

galdata_allz = rfn.append_fields(galdata_allz, 'Obs_gal_idx',  truetoobs, usemask=False)


"""Plot MsObserved vs MsTrue"""

plt.figure()
x = np.arange(5, 13)
plt.hist2d(
    np.log10(galdata_allz['Mass'][galdata_allz['Obs_gal_idx'] > 0]*10**11),
    galphot['Mass'][galdata_allz[galdata_allz['Obs_gal_idx'] > 0]['Obs_gal_idx'].astype('int')],
    cmin=1, bins=100, range=[[9, 12], [9, 12]], norm=mpl.colors.LogNorm(), cmap='jet'
)
plt.colorbar()
# plt.plot(x, x, lab='y=x')
plt.xlabel('Original Mass', size=12)
plt.ylabel('Photometric mass', size=12)
plt.title('H-AGN, stellar photometric mass dispersion')

"""Compute median, average and percentiles for masses."""

# For true catalog
stellarmassbins = np.linspace(9, 12, num=100)
avHMperSM = np.full([numzbin, np.size(stellarmassbins)-1], np.nan)
medHMperSM = np.full([numzbin, np.size(stellarmassbins)-1], np.nan)
stdHMperSM = np.full([numzbin, np.size(stellarmassbins)-1], np.nan)
for i in range(numzbin - 1):
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
        if np.size(indices) > 2:
            avHMperSM[i, j] = np.average(np.log10(halodata[i]['Mass'][indices] * 10**11))
            medHMperSM[i, j] = np.median(np.log10(halodata[i]['Mass'][indices] * 10**11))
            stdHMperSM[i, j] = np.std(np.log10(halodata[i]['Mass'][indices] * 10**11))

# For photometric catalog
stellarmassbins = np.linspace(9, 12, num=100)
avHMperSMPhot = np.full([numzbin, np.size(stellarmassbins)-1], np.nan)
medHMperSMPhot = np.full([numzbin, np.size(stellarmassbins)-1], np.nan)
stdHMperSMPhot = np.full([numzbin, np.size(stellarmassbins)-1], np.nan)
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
        if np.size(indices) > 2:
            # print(np.size(indices))
            avHMperSMPhot[i, j] = np.average(np.log10(halodata[i]['Mass'][indices] * 10**11))
            medHMperSMPhot[i, j] = np.median(np.log10(halodata[i]['Mass'][indices] * 10**11))
            stdHMperSMPhot[i, j] = np.std(np.log10(halodata[i]['Mass'][indices] * 10**11))

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

for i in range(numzbin-1):
    for j in range(np.size(massbins)-1):
        m1 = massbins[j]
        m2 = massbins[j+1]
        # select indices of galaxies contained in the haloes with a mass
        # between m1 and m2 :
        indices = np.where(np.logical_and(
            np.log10(halodata[i]['Mass']*10**11) > m1,
            np.log10(halodata[i]['Mass']*10**11) <= m2))[0]
        # indices_cent = np.intersect1d(indices, halodata[i]['level'] == 1)
        if len(indices) > 0:
            avSMperHM[i, j] = np.average(
                np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11))
            medSMperHM[i, j] = np.median(
                np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11))
        else:
            avSMperHM[i, j] = np.nan
            medSMperHM[i, j] = np.nan


"""Fit the Behroozi 2010 relation on Mh(Ms)"""


def boo_MhMs(Ms, M1, Ms0, beta, delta, gamma):
    """Behroozi et al. 2010 Mh(Ms) relation
    All masses are in logscale"""
    return M1+beta*(Ms-Ms0)+10**(delta*(Ms-Ms0))/(1+10**(-gamma*(Ms-Ms0)))-0.5


boo_fit_true = np.empty([numzbin-1, 5])
boo_cov_true = np.empty([numzbin-1, 5, 5])
for i in range(numzbin-1):
    print(i)
    indices = np.where(
        np.logical_and(
            np.log10(galdata[i]['Mass'][hal_centgal[i]-1]*10**11) > 9,
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1)
            )
    )
    boo_fit_true[i], boo_cov_true[i] = curve_fit(
        boo_MhMs,
        np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11),
        np.log10(halodata[i]['Mass'][indices]*10**11),
        bounds=[[10, 8, 0, 0, 0], [13, 11, 5, 5, 5]],
        method='trf')
print(boo_fit_true)

boo_fit_phot = np.empty([numzbin-1, 5])
boo_cov_phot = np.empty([numzbin-1, 5, 5])
for i in range(numzbin-1):
    print(i)
    indices = np.where(
        np.logical_and(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
            np.logical_and(
                galdata_allz['Obs_gal_idx'][
                    hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
                ] > 0,
                galphot['Mass'][
                    galdata_allz['Obs_gal_idx'][
                        hal_centgal[i] - 1 + sum(len(galdata[j]) for j in range(i))
                    ].astype('int')
                ] > 9)
        )
    )
    boo_fit_phot[i], boo_cov_phot[i] = curve_fit(
        boo_MhMs,
        galphot['Mass'][
                galdata_allz['Obs_gal_idx'][
                    hal_centgal[i][indices] - 1 + sum(len(galdata[j]) for j in range(i))
                ].astype('int')
        ],
        np.log10(halodata[i]['Mass'][indices]*10**11),
        bounds=[[10, 8, 0, 0, 0], [13, 11, 5, 5, 5]],
        method='trf')
print(boo_fit_phot)


"""Plot Ms(Mh) for true galaxies and level 1 halos"""

boofitsSMbins = np.linspace(9, 12, num=100)

for i in range(numzbin-1):
    plt.figure()
    indices = np.where(np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1))
    # verification that all galaxies selected are central
    # print(galdata[i]['level'][hal_centgal[i][indices]-1].min())
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11),
        bins=100, cmin=1)
    plt.colorbar()
    # plt.scatter((massbins[:-1]+massbins[1:])/2, avSMperHM[i][:], color='red',
    #             label='Average SM for a given HM')
    # plt.scatter((massbins[:-1]+massbins[1:])/2, medSMperHM[i][:],
    #             color='green', label='Median SM for a given HM')
    plt.scatter(avHMperSM[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
                color='black', label='Average HM for a given SM')
    plt.scatter(medHMperSM[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
                color='pink', label='Median HM for a given SM')
    # Plot Behroozi fit
    # plt.plot(boo_MhMs(boofitsSMbins, *boo_fit_true[i]), boofitsSMbins,
    #          label=str('True Behroozi function fit'), c='r')
    plt.legend()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Log($M_{*}$) [Log($M_{\odot}$)]', size=12)
    plt.title('HorizonAGN, Central galz='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.savefig('../Plots/HAGN_Matching/ClotMatchBis/TrueMass_HaloMass_Boofit' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')


"""Plot Ms(Mh) on the same figure"""

fig, ax = plt.subplots(2, 2)
# fig.suptitle('Horizon AGN CONE, WARNING !! colorbars not homogeneous')
for i in range(3):
    indices = np.where(np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1))
    ax1 = ax[i//2, i % 2]
    if i == 0:
        counts, xedges, yedges, im = ax1.hist2d(
            np.log10(halodata[i]['Mass'][indices]*10**11),
            np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11),
            bins=100, cmin=1)
    else:
        _, _, _, im = ax1.hist2d(
            np.log10(halodata[i]['Mass'][indices]*10**11),
            np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11),
            bins=(xedges, yedges), cmin=1)
    # Put the colorbar inside of the axes.

    axins1 = inset_axes(
        ax1,
        width="5%",  # width = 10% of parent_bbox width
        height="35%",  # height : 50%
        loc=8,
        bbox_to_anchor=[0.5, 0, 0.5, 1],
        bbox_transform=ax1.transAxes
        # bbox_transform=ax1.transAxes,
        # bbox_to_anchor=(1, 1)
        )
    cbar = fig.colorbar(im, cax=axins1)
    ax1.set_xlabel('Log($M_{h}/M_{\odot}$)', size=12)
    ax1.set_ylabel('Log($M_{*}/M_{\odot}$)', size=12)
    cbar.ax.tick_params(labelsize=9)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()

    ax1.plot(boo_MhMs(boofitsSMbins, *boo_fit_true[i]), boofitsSMbins,
             label=str('Behroozi function fit'), c='r')

    plt.text(0.1, 0.8, str(zbins_Cone[i])+'<z<'+str(zbins_Cone[i+1]),
             size=12, transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white'))
fig.tight_layout()
# plt.subplots_adjust(top=0.95)
plt.show()


"""Plot Ms_observed(Mh) and level 1 halos"""

boofitsSMbins = np.linspace(9, 12, num=100)
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
        bins=100, cmin=1, range=[[10, 14], [9, 12]])
    plt.colorbar()
    # plt.errorbar(avHMperSMPhot[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
    #              xerr=stdHMperSMPhot[i],
    #              color='red', label='Average HM for a given SM')
    # plt.scatter(medHMperSMPhot[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
    #             color='pink', label='Median HM for a given SM')
    # Plot Behroozi fit
    plt.plot(boo_MhMs(boofitsSMbins, *boo_fit_phot[i]), boofitsSMbins,
             label=str('Behroozi function fit'), c='r')
    plt.legend()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Log($M_{*}$) Photometric [Log($M_{\odot}$)]', size=12)
    plt.title('HorizonAGN photo, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.savefig('../Plots/HAGN_Matching/ClotMatchBis/PhotoMass_HaloMass_Boofit' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')


"""Plot Ms/Mh vs Mh for true and photometric catalogs"""

plt.figure()
# cmap = ['blue', 'green', 'red']
# marker = ['v', '>', '^']
for i in range(numzbin-1):
    # plt.scatter(
    #     medHMperSM[i],
    #     (stellarmassbins[:-1]+stellarmassbins[1:]) / 2 - medHMperSM[i],
    #     label='True catalog, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]),
    #     edgecolors=cmap[i], facecolors='none'
    # )
    plt.plot(boo_MhMs(boofitsSMbins, *boo_fit_true[i]),
             boofitsSMbins - boo_MhMs(boofitsSMbins, *boo_fit_true[i]),
             label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.scatter(
    #     medHMperSMPhot[i],
    #     (stellarmassbins[:-1]+stellarmassbins[1:]) / 2 - medHMperSMPhot[i],
    #     label='Phot catalog, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]),
    #     edgecolors=cmap[i], facecolors=cmap[i]
    # )
    plt.legend()
    plt.xlabel('Log($M_{h} / M_{\odot}$)', size=15)
    plt.ylabel('Log($M_{s}/M_{h}$)', size=15)


"""Plot Ms/Mh histogram for true catalog"""


for i in range(numzbin-1):
    plt.figure()
    indices = np.where(np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1))
    # verification that all galaxies selected are central
    # print(galdata[i]['level'][hal_centgal[i][indices]-1].min())
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]/halodata[i]['Mass'][indices]),
        bins=100, cmin=1)
    plt.colorbar()
    plt.scatter(
        medHMperSM[i],
        (stellarmassbins[:-1]+stellarmassbins[1:]) / 2 - medHMperSM[i],
        label='Phot catalog, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]),
        facecolors='none', color='red'
    )
    # Plot Behroozi fit
    plt.plot(boo_MhMs(boofitsSMbins, *boo_fit_true[i]),
             boofitsSMbins - boo_MhMs(boofitsSMbins, *boo_fit_true[i]),
             label=str('Behroozi function fit'), c='r')
    plt.legend()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Log($M_{*}$) [Log($M_{\odot}$)]', size=12)
    plt.title('HorizonAGN, Central galz='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))


"""Plot Ms/Mh for photometric catalog and with median found with Ms(Mh)"""

for i in range(numzbin-1):
    plt.figure()
    indices = np.where(
        np.logical_and(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
            np.logical_and(
                galdata_allz['Obs_gal_idx'][
                    hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
                ] > 0,
                galphot['Mass'][
                    galdata_allz['Obs_gal_idx'][
                        hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
                    ].astype('int')
                ] > 9
            )
        )
    )
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        galphot['Mass'][
                galdata_allz['Obs_gal_idx'][
                    hal_centgal[i][indices] - 1 + sum(len(galdata[j]) for j in range(i))
                ].astype('int')
        ] - np.log10(halodata[i]['Mass'][indices]*10**11),
        bins=100, cmin=1, range=[[10, 14], [-2, 1]]
    )
    # plt.plot(
    #     medHMperSMPhot[i],
    #     (stellarmassbins[:-1]+stellarmassbins[1:]) / 2 - medHMperSMPhot[i],
    #     label='Phot catalog, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]),
    #     color='red'
    # )
    # plt.plot(
    #     boo_MhMs(boofitsSMbins, *boo_fit_phot[i]),
    #     boofitsSMbins - boo_MhMs(boofitsSMbins, *boo_fit_phot[i]),
    #     label=str('phot Behroozi function fit'), c='black')
    plt.legend()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
    plt.ylabel('Log($M_{s}/M_{h}$)', size=15)
    plt.title('H-AGN, Central gal and level 1 halos')


""""With gas mass"""

# plt.hist2d(
#         np.log10(halodata[i]['Mass'][indices]*10**11),
#         np.log10(10**galphot['Mass'][
#                 galdata_allz['Obs_gal_idx'][
#                     hal_centgal[i][indices] - 1 + sum(len(galdata[j]) for j in range(i))
#                 ].astype('int')
#         ] + gas_mass[galdata_allz['Obs_gal_idx'][
#                 hal_centgal[i][indices] - 1 + sum(len(galdata[j]) for j in range(i))
#                 ].astype('int')]) - np.log10(halodata[i]['Mass'][indices]*10**11),
#         bins=100, cmin=1, range=[[10, 14], [-2, 1]]
#     )

"""Fit the Yang relation on the M*/Mh relation"""


def mstar_over_mh_yang(x, A, m1, beta, gamma):
    """Yang et al. 2012 function, see Moster et al. 2010."""
    return 2.0 * A * ((x / m1)**(-beta) + (x / m1)**gamma)**(-1)


# yang_fit_true = np.empty([numzbin, 4])
# yang_cov_true = np.empty([numzbin, 4, 4])
# for i in range(numzbin-1):
#     yang_fit_true[i], yang_cov_true[i] = curve_fit(
#         mstar_over_mh_yang,
#         10**medHMperSM[i][~np.isnan(medHMperSM[i])],
#         10**(((stellarmassbins[:-1]+stellarmassbins[1:]) / 2)[~np.isnan(medHMperSM[i])] -
#              medHMperSM[i][~np.isnan(medHMperSM[i])]),
#         sigma=stdHMperSM[i][~np.isnan(medHMperSM[i])],
#         p0=[0.01, 10**12, 0.1, 0.1],
#         bounds=[[0, 10**9, 0, 0], [0.5, 10**14, 5, 5]], method='trf')

# yang_fit_phot = np.empty([numzbin-1, 4])
# yang_cov_phot = np.empty([numzbin-1, 4, 4])
# for i in range(numzbin-1):
#     yang_fit_phot[i], yang_cov_phot[i] = curve_fit(
#         mstar_over_mh_yang,
#         10**medHMperSMPhot[i][~np.isnan(medHMperSMPhot[i])],
#         10**(((stellarmassbins[:-1]+stellarmassbins[1:]) / 2)[~np.isnan(medHMperSMPhot[i])] -
#              medHMperSMPhot[i][~np.isnan(medHMperSMPhot[i])]),
#         sigma=stdHMperSMPhot[i][~np.isnan(medHMperSMPhot[i])],
#         p0=[0.01, 10**12, 0.5, 0.1],
#         bounds=[[0, 10**10, 0, 0], [0.5, 10**13, 5, 5]], method='trf')
# print(yang_fit_phot)

yang_fit_true = np.empty([numzbin-1, 4])
yang_cov_true = np.empty([numzbin-1, 4, 4])
for i in range(numzbin-1):
    print(i)
    indices = np.where(
        np.logical_and(
            np.logical_and(
                np.log10(galdata[i]['Mass'][hal_centgal[i]-1]*10**11) > 9,
                np.log10(halodata[i]['Mass']*10**11) > 10.8),
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1)
            )
    )
    yang_fit_true[i], yang_cov_true[i] = curve_fit(
        mstar_over_mh_yang,
        halodata[i]['Mass'][indices]*10**11,
        galdata[i]['Mass'][hal_centgal[i][indices]-1] / halodata[i]['Mass'][indices],
        p0=[0.01, 10**12, 0.1, 0.1],
        bounds=[[0, 10**9, 0, 0], [0.5, 10**14, 5, 5]], method='trf')
print(yang_fit_true)

yang_fit_phot = np.empty([numzbin-1, 4])
yang_cov_phot = np.empty([numzbin-1, 4, 4])
for i in range(numzbin-1):
    yang_fit_phot[i], yang_cov_phot[i] = curve_fit(
        mstar_over_mh_yang,
        10**medHMperSMPhot[i][~np.isnan(medHMperSMPhot[i])],
        10**(((stellarmassbins[:-1]+stellarmassbins[1:]) / 2)[~np.isnan(medHMperSMPhot[i])] -
             medHMperSMPhot[i][~np.isnan(medHMperSMPhot[i])]),
        sigma=stdHMperSMPhot[i][~np.isnan(medHMperSMPhot[i])],
        p0=[0.01, 10**12, 0.5, 0.1],
        bounds=[[0, 10**10, 0, 0], [0.5, 10**13, 5, 5]], method='trf')
print(yang_fit_phot)


"""Plot Yang fit"""


x = np.logspace(10, 14, num=1000)
for i in range(numzbin-1):
    plt.figure()
    indices = np.where(
        np.logical_and(
            np.log10(galdata[i]['Mass'][hal_centgal[i]-1]*10**11) > 9,
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1)
            )
    )
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1] / halodata[i]['Mass'][indices]),
        bins=100, cmin=1, range=[[10.3, 13], [-2.5, -0.5]]
    )
    p = plt.plot(
        np.log10(x), np.log10(mstar_over_mh_yang(x, *yang_fit_true[i])),
        label=str('Moster et al. fit'), c='b')
    plt.plot(boo_MhMs(boofitsSMbins, *boo_fit_true[i]),
             boofitsSMbins - boo_MhMs(boofitsSMbins, *boo_fit_true[i]),
             label=str('Behroozi et al. fit'), c='r')
    plt.xlabel('Log($M_{h} / M_{\odot}$)', size=15)
    plt.ylabel('Log($M_{s}/M_{h}$)', size=15)
    plt.legend()
    plt.text(0.1, 0.1, str(zbins_Cone[i])+'<z<'+str(zbins_Cone[i+1]),
             size=12, transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white'))
    plt.tight_layout()
    plt.savefig('../Plots/HAGN_Matching/ClotMatchBis/True_MsonMH_fits' +
                str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')


x = np.logspace(10, 14, num=1000)
plt.figure()
for i in range(numzbin-1):
    # p = plt.plot(np.log10(x), np.log10(mstar_over_mh_yang(x, *yang_fit_true[i])))
    # plt.plot(np.log10(x), np.log10(mstar_over_mh_yang(x, *yang_fit_phot[i])),
    #          color=p[0].get_color())
    # plt.scatter(
    #     medHMperSM[i],
    #     (stellarmassbins[:-1]+stellarmassbins[1:]) / 2 - medHMperSM[i],
    #     facecolors='none', edgecolors=p[0].get_color(),
    #     label='True catalog, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])
    # )
    # plt.scatter(
    #     medHMperSM[i],
    #     (stellarmassbins[:-1]+stellarmassbins[1:]) / 2 - medHMperSMPhot[i],
    #     facecolors='none', edgecolors=p[0].get_color(),
    #     label='Photo catalog, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])
    # )
    # Plot Behroozi fit
    p = plt.plot(
            boo_MhMs(boofitsSMbins, *boo_fit_true[i]),
            boofitsSMbins - boo_MhMs(boofitsSMbins, *boo_fit_true[i]),
            label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.plot(boo_MhMs(boofitsSMbins, *boo_fit_phot[i]),
             boofitsSMbins - boo_MhMs(boofitsSMbins, *boo_fit_phot[i]),
             linestyle ='--', color=p[0].get_color())
plt.legend()
plt.xlabel('Log($M_{h} / M_{\odot}$)', size=15)
plt.ylabel('Log($M_{s}/M_{h}$)', size=15)
plt.show()

"""Find MhPeak with the Yanf fit"""

MhaloPeak_true = np.zeros(numzbin-1)
for i in range(numzbin-1):
    MhaloPeak_true[i] = np.log10(x[np.argmax(mstar_over_mh_yang(x, *yang_fit_true[i]))])

MhaloPeak_phot = np.zeros(numzbin-1)
for i in range(numzbin-1):
    MhaloPeak_phot[i] = np.log10(x[np.argmax(mstar_over_mh_yang(x, *yang_fit_phot[i]))])


"""Find MhPeak with Behroozi fit"""

MhaloPeak_true_boo = np.zeros(numzbin-1)
for i in range(numzbin-1):
    idx_max = np.argmax(boofitsSMbins - boo_MhMs(boofitsSMbins, *boo_fit_true[i]))
    MhaloPeak_true_boo[i] = boo_MhMs(boofitsSMbins[idx_max], *boo_fit_true[i])

# MhaloPeak_phot_boo = np.zeros(numzbin-1)
# for i in range(numzbin-1):
#     MhaloPeak_phot_boo[i] = np.log10(x[np.argmax(mstar_over_mh_yang(x, *yang_fit_phot[i]))])


"""Plot MhaloPeak versus z"""

# Lauthaud+17 use a different cosmology with H0=72
redshiftLeauthaud = np.array([(0.22 + 0.48) / 2, (0.48 + 0.74) / 2, (0.74 + 1) / 2])
MhaloPeakLeauthaud = np.log10(np.array([9.5 * 10**11, 1.45 * 10**12, 1.4 * 10**12]))
MhaloSigmaLeauthaud = np.log10(np.array(
    [1.05 * 10**12, 1.55 * 10**12, 1.5 * 10**12])) - MhaloPeakLeauthaud
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


plt.figure()
# plt.plot((zbins_Cone[1:-1]+zbins_Cone[:-2])/2, MhaloPeak_true, 'd',
#          label='Original Catalog')
# plt.plot((zbins_Cone[1:-1]+zbins_Cone[:-2])/2, MhaloPeak_phot, 'd',
#          label='Photometric Catalog')

# Coming From AM__COSMOSIari_BolshoiPlanc.py
plt.errorbar((redshifts[1:] + redshifts[:-1]) / 2, MhaloPeak + np.log10(67.74/70),
             yerr=np.transpose(MhaloPeakSigma),
             fmt='o', color='red', capsize=5, label='Cosmos AM')
plt.errorbar(
    (zbins_Cone[1:-1]+zbins_Cone[:-2])/2, MhaloPeak_true_boo,
    yerr=0.1, fmt='o', capsize=5, c='g',
    label='Horizon-AGN Lightcone')
plt.errorbar(redshiftCoupon17, MhaloPeakCoupon17 - np.log10(0.7),
             yerr=MhaloSigmaCoupon17, c='b',
             fmt='o', capsize=5, label='Coupon et al. 2017 Draft')
plt.errorbar(redshiftLeauthaud, MhaloPeakLeauthaud + np.log10(72/70),
             yerr=MhaloSigmaLeauthaud, c='black',
             fmt='o', capsize=5, label='Leauthaud et al. 2011')
plt.ylabel('Log($M_{halo}^{peak}/ M_{\odot}$)', size=15)
plt.xlabel('Redshift',  size=15)
plt.legend(loc=2)
# plt.title('Horizon-AGN, MhaloPeak')
plt.tight_layout()

"""Plot sSFR vs Mh for true catalogs"""

for i in range(numzbin-1):
    plt.figure()
    indices = np.where(np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1))
    # verification that all galaxies selected are central
    # print(galdata[i]['level'][hal_centgal[i][indices]-1].min())
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        # np.log10(galdata[i]['SFRCorr'][hal_centgal[i][indices]-1] /
        #          (galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11)),
        np.log10(galdata[i]['SFRCorr'][hal_centgal[i][indices]-1]),
        bins=100, cmin=1, range=[[10, 14], [-2, 2]])
    plt.colorbar()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Log(SFR) [Log($yr^{-1}$)]', size=12)
    plt.title('HorizonAGN, Central galz='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.savefig('../Plots/HAGN_Matching/ClotMatchBis/TrueSpecificSFR_HaloMass' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')

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
        np.log10(galphot['Ra'][
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
    # plt.savefig('../Plots/HAGN_Matching/ClotMatch/PhotoSFR_HaloMass' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')

"""PLot sSFR vs Mh for photo cat"""

for i in range(numzbin-1):
    plt.figure()
    indices = np.where(
        np.logical_and(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
            np.logical_and(
                galdata_allz['Obs_gal_idx'][
                    hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
                ] > 0,
                galphot['Mass'][
                    galdata_allz['Obs_gal_idx'][
                        hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
                    ].astype('int')] > 9
            )
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
    plt.savefig('../Plots/HAGN_Matching/ClotMatchBis/PhotoSpecificSFR_HaloMass' +
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

plt.close('all')

"""Compute Median Metalicity per halo mass and 68% interval."""

massbins = np.linspace(10, 15, num=100)
medMetperHMPhot = np.zeros([numzbin, np.size(massbins)-1])
avMetperHMPhot = np.zeros([numzbin, np.size(massbins)-1])
stdMetperHMPhot = np.zeros([numzbin, np.size(massbins)-1])
# supMetperHM = np.zeros([numzbin, np.size(massbins)-1])
# infMetperHM = np.zeros([numzbin, np.size(massbins)-1])

for i in range(numzbin-1):
    indices_selec = np.where(
        np.logical_and(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
            np.logical_and(
                np.logical_and(
                    galdata_allz['Obs_gal_idx'][
                        hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
                    ] > 0,
                    galphot['Gas_met_boost'][
                        galdata_allz['Obs_gal_idx'][
                            hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
                        ].astype('int')
                    ]
                ),
                galphot['Mass'][
                        galdata_allz['Obs_gal_idx'][
                            hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
                        ].astype('int')
                    ] > 9
            )
        )
    )
    gal_gasmet = galphot['Gas_met_boost'][
            galdata_allz['Obs_gal_idx'][
                hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
            ].astype('int')]
    for j in range(np.size(massbins)-1):
        m1 = massbins[j]
        m2 = massbins[j+1]
        indices = np.where(np.logical_and(
            np.log10(halodata[i]['Mass']*10**11) > m1,
            np.log10(halodata[i]['Mass']*10**11) <= m2))[0]
        indices = np.intersect1d(indices_selec, indices)
        if len(indices) > 4:
            avMetperHMPhot[i, j] = np.average(gal_gasmet[indices])
            medMetperHMPhot[i, j] = np.median(gal_gasmet[indices])
            stdMetperHMPhot[i, j] = np.std(gal_gasmet[indices])
        else:
            avMetperHMPhot[i, j] = np.nan
            medMetperHMPhot[i, j] = np.nan
            stdMetperHMPhot[i, j] = np.nan


"""Plot Gas metalicity vs Mh for photo galaxies"""

# TODO: problem with certain galaxies having a gas metalicity of 0
for i in range(numzbin-1):
    plt.figure()
    indices = np.where(
        np.logical_and(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
            np.logical_and(
                np.logical_and(
                    galdata_allz['Obs_gal_idx'][
                        hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
                    ] > 0,
                    galphot['Gas_met_boost'][
                        galdata_allz['Obs_gal_idx'][
                            hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
                        ].astype('int')
                    ]
                ),
                galphot['Mass'][
                        galdata_allz['Obs_gal_idx'][
                            hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
                        ].astype('int')
                    ] > 9
            )
        )
    )
    plt.hist2d(
        #np.log10(halodata[i]['Mass'][indices]*10**11),
        np.log10(haloes_env[i][indices, 0][0]),
        galphot['Gas_met_boost'][
            galdata_allz['Obs_gal_idx'][
                hal_centgal[i][indices] - 1 + sum(len(galdata[j]) for j in range(i))
            ].astype('int')
        ],
        bins=100, cmin=1)
    plt.colorbar()
    # plt.plot((massbins[:-1]+massbins[1:])/2, avMetperHMPhot[i],
    #          color='red', label='Average Metalicity for a given HM, $\pm 1\sigma$')
    # plt.plot((massbins[:-1]+massbins[1:])/2, avMetperHMPhot[i] + stdMetperHMPhot[i],
    #          color='red', linestyle='--')
    # plt.plot((massbins[:-1]+massbins[1:])/2, avMetperHMPhot[i] - stdMetperHMPhot[i],
    #          color='red', linestyle='--')

    # plt.errorbar((massbins[:-1]+massbins[1:])/2, avMetperHMPhot[i][:],
    #              color='red', yerr=stdMetperHMPhot[i],
    #              label='Average Metalicity for a given HM')
    # plt.scatter((massbins[:-1]+massbins[1:])/2, medMetperHMPhot[i][:],
    #             color='green', label='Median Metalicity for a given HM')
    plt.legend()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Gas Metalicity', size=12)
    plt.title('Photometric HorizonAGN, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.savefig('../Plots/HAGN_Matching/ClotMatchBis/GasMet/gasmet_' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')

"""Evolution of photometric Gas metalicity with redshift"""

plt.figure()
for i in range(numzbin-1):
    plt.plot((massbins[:-1]+massbins[1:])/2, avMetperHMPhot[i],
             label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.fill_between(
        (massbins[:-1]+massbins[1:])/2, avMetperHMPhot[i] + stdMetperHMPhot[i],
        avMetperHMPhot[i] - stdMetperHMPhot[i], alpha=0.3,
        linestyle='--')
    plt.legend()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Gas Metalicity', size=12)
    plt.title('Photometric HorizonAGN Gas metalicity')
    plt.tight_layout()

"""Boost stellar metalicity in True catalog"""

stellar_met_boost = np.empty(galdata_allz['met'].shape)
for idx_true in range(len(stellar_met_boost)):
    stellar_met_boost[idx_true] = galdata_allz['met'][idx_true] * boost(
        galdata_allz['z'][idx_true])
galdata_allz = rfn.append_fields(galdata_allz, 'Stellar_met_boost',
                                 stellar_met_boost, usemask=False)

"""Compute average of stellar metalicity and standard deviation"""

massbins = np.linspace(10, 15, num=100)
medMetperHMtrue = np.zeros([numzbin, np.size(massbins)-1])
avMetperHMtrue = np.zeros([numzbin, np.size(massbins)-1])
stdMetperHMtrue = np.zeros([numzbin, np.size(massbins)-1])
# supMetperHM = np.zeros([numzbin, np.size(massbins)-1])
# infMetperHM = np.zeros([numzbin, np.size(massbins)-1])

for i in range(numzbin-1):
    indices_selec = np.where(np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1))
    gal_stemet = galdata_allz['Stellar_met_boost'][
        hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))]
    for j in range(np.size(massbins)-1):
        m1 = massbins[j]
        m2 = massbins[j+1]
        indices = np.where(np.logical_and(
            np.log10(halodata[i]['Mass']*10**11) > m1,
            np.log10(halodata[i]['Mass']*10**11) <= m2))[0]
        indices = np.intersect1d(indices_selec, indices)
        if len(indices) > 0:
            avMetperHMtrue[i, j] = np.average(gal_stemet[indices])
            medMetperHMtrue[i, j] = np.median(gal_stemet[indices])
            stdMetperHMtrue[i, j] = np.std(gal_stemet[indices])
        else:
            avMetperHMtrue[i, j] = np.nan
            medMetperHMtrue[i, j] = np.nan
            stdMetperHMtrue[i, j] = np.nan


"""Plot Stellar Met vs Mh for photo catalogs"""

for i in range(numzbin-1):
    plt.figure()
    indices = np.where(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
    )
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        galdata_allz['Stellar_met_boost'][
            hal_centgal[i][indices] - 1 + sum(len(galdata[j]) for j in range(i))],
        bins=100, cmin=1
    )
    plt.colorbar()
    plt.plot((massbins[:-1]+massbins[1:])/2, avMetperHMtrue[i],
             color='red', label='Average Metalicity for a given HM, $\pm 1\sigma$')
    plt.plot((massbins[:-1]+massbins[1:])/2, avMetperHMtrue[i] + stdMetperHMtrue[i],
             color='red', linestyle='--')
    plt.plot((massbins[:-1]+massbins[1:])/2, avMetperHMtrue[i] - stdMetperHMtrue[i],
             color='red', linestyle='--')
    plt.legend()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Stellar Metalicity', size=12)
    plt.title('Original HorizonAGN, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.savefig('../Plots/HAGN_Matching/ClotMatch/StellarMet/stellarmet_' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')


"""Evolution of stellar metalicity with redshift"""

plt.figure()
for i in range(numzbin-1):
    plt.plot((massbins[:-1]+massbins[1:])/2, avMetperHMtrue[i],
             label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.fill_between(
        (massbins[:-1]+massbins[1:])/2, avMetperHMtrue[i] + stdMetperHMtrue[i],
        avMetperHMtrue[i] - stdMetperHMtrue[i], alpha=0.3,
        linestyle='--')
    plt.legend()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Stellar Metalicity', size=12)
    plt.title('Original HorizonAGN Stellar metalicity')
    plt.tight_layout()


"""Compare Photometric Gas Metalicity and Original Stellar Metalicity"""

for i in range(numzbin-1):
    plt.figure()
    plt.plot((massbins[:-1]+massbins[1:])/2, avMetperHMPhot[i],
             color='green', label='Photometric Gas Metalicity $\pm 1\sigma$')
    plt.fill_between(
        (massbins[:-1]+massbins[1:])/2, avMetperHMPhot[i] + stdMetperHMPhot[i],
        avMetperHMPhot[i] - stdMetperHMPhot[i], alpha=0.3,
        color='green', linestyle='--')
    plt.plot((massbins[:-1]+massbins[1:])/2, avMetperHMtrue[i],
             color='red', label='True Stellar Metalicity $\pm 1\sigma$')
    plt.fill_between(
        (massbins[:-1]+massbins[1:])/2, avMetperHMtrue[i] + stdMetperHMtrue[i],
        avMetperHMtrue[i] - stdMetperHMtrue[i], alpha=0.3,
        color='red', linestyle='--')
    plt.legend()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Metalicity', size=12)
    plt.title('HorizonAGN, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.savefig('../Plots/HAGN_Matching/ClotMatch/Gas+StellarMet/gas+stellarmet_' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')


plt.close('all')

"""Compute average stellar met for a given halo local density"""

densbins = np.linspace(-2.5, 1, num=100)
medMetperHDtrue = np.zeros([numzbin, np.size(densbins)-1])
avMetperHDtrue = np.zeros([numzbin, np.size(densbins)-1])
stdMetperHDtrue = np.zeros([numzbin, np.size(densbins)-1])
# supMetperHM = np.zeros([numzbin, np.size(massbins)-1])
# infMetperHM = np.zeros([numzbin, np.size(massbins)-1])

for i in range(numzbin-1):
    indices_selec = np.where(np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1))
    gal_stemet = galdata_allz['Stellar_met_boost'][
        hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))]
    for j in range(np.size(densbins)-1):
        d1 = densbins[j]
        d2 = densbins[j+1]
        indices = np.where(np.logical_and(
            np.log10(haloes_env[i][:, 0]) > d1,
            np.log10(haloes_env[i][:, 0]) <= d2))[0]
        indices = np.intersect1d(indices_selec, indices)
        if len(indices) > 0:
            avMetperHDtrue[i, j] = np.average(gal_stemet[indices])
            medMetperHDtrue[i, j] = np.median(gal_stemet[indices])
            stdMetperHDtrue[i, j] = np.std(gal_stemet[indices])
        else:
            avMetperHDtrue[i, j] = np.nan
            medMetperHDtrue[i, j] = np.nan
            stdMetperHDtrue[i, j] = np.nan


"""Evolution of stellar metalicity with environment density"""

for i in range(numzbin-1):
    plt.figure()
    indices = np.where(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
    )
    plt.hist2d(
        np.log10(haloes_env[i][indices, 0][0]),
        galdata_allz['Stellar_met_boost'][
            hal_centgal[i][indices] - 1 + sum(len(galdata[j]) for j in range(i))],
        bins=100, cmin=1
    )
    plt.colorbar()
    plt.plot((densbins[:-1]+densbins[1:])/2, avMetperHDtrue[i],
             color='red', label='Average Original Stellar Metalicity $\pm 1\sigma$')
    plt.fill_between(
        (densbins[:-1]+densbins[1:])/2, avMetperHDtrue[i] + stdMetperHDtrue[i],
        avMetperHDtrue[i] - stdMetperHDtrue[i], alpha=0.3,
        color='red', linestyle='--')
    plt.legend()
    plt.xlabel('Halo local density smoothed at 3Mpc (log)', size=12)
    plt.ylabel('Stellar Metalicity', size=12)
    plt.title('Original HorizonAGN, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.tight_layout()
    # plt.savefig('../Plots/HAGN_Matching/ClotMatch/StellarMet/Stellarmet_Density_' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')

"""Density of haloes versus halo mass"""

for i in range(numzbin-1):
    plt.figure()
    # Comment this if you want to plot all haloes and not only central haloes with central galaxies
    indices = np.where(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
    )
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        np.log10(haloes_env[i][indices, 0][0]),
        bins=100, cmin=1
    )
    plt.colorbar()
    plt.legend()
    plt.xlabel('Halo Mass', size=12)
    plt.ylabel('Halo local density', size=12)
    plt.title('Original HorizonAGN, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.tight_layout()


"""Original Ms/Mh versus density"""

# compute average and std deviation

densbins = np.linspace(-2.5, 1, num=100)
medMSMHperHDtrue = np.zeros([numzbin, np.size(densbins)-1])
avMSMHperHDtrue = np.zeros([numzbin, np.size(densbins)-1])
stdMSMHperHDtrue = np.zeros([numzbin, np.size(densbins)-1])
# supMetperHM = np.zeros([numzbin, np.size(massbins)-1])
# infMetperHM = np.zeros([numzbin, np.size(massbins)-1])

for i in range(numzbin-1):
    indices_selec = np.where(np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1))
    for j in range(np.size(densbins)-1):
        d1 = densbins[j]
        d2 = densbins[j+1]
        indices = np.where(np.logical_and(
            np.log10(haloes_env[i][:, 0]) > d1,
            np.log10(haloes_env[i][:, 0]) <= d2))[0]
        indices = np.intersect1d(indices_selec, indices)
        if len(indices) > 0:
            avMSMHperHDtrue[i, j] = np.average(
                np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1] /
                         halodata[i]['Mass'][indices]))
            medMSMHperHDtrue[i, j] = np.median(
                np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1] /
                         halodata[i]['Mass'][indices]))
            stdMSMHperHDtrue[i, j] = np.std(
                np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1] /
                         halodata[i]['Mass'][indices]))
        else:
            avMSMHperHDtrue[i, j] = np.nan
            medMSMHperHDtrue[i, j] = np.nan
            stdMSMHperHDtrue[i, j] = np.nan

"""Plot Original Ms/Mh versus density"""
for i in range(numzbin-1):
    plt.figure()
    indices = np.where(np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1))
    # indices = np.where(hal_centgal[i] > 0)
    plt.hist2d(
        np.log10(haloes_env[i][indices, 0][0]),
        np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1] /
                 halodata[i]['Mass'][indices]),
        bins=100, cmin=1)
    plt.colorbar()
    plt.plot((densbins[:-1]+densbins[1:])/2, avMSMHperHDtrue[i],
             color='red', label='Average $\pm 1\sigma$')
    plt.fill_between(
        (densbins[:-1]+densbins[1:])/2, avMSMHperHDtrue[i] + stdMSMHperHDtrue[i],
        avMSMHperHDtrue[i] - stdMSMHperHDtrue[i], alpha=0.3,
        color='red', linestyle='--')
    plt.legend()
    plt.xlabel('Log(Halo density)', size=12)
    plt.ylabel('Log($M_{*}/M_{h}$)', size=12)
    plt.title('Original HorizonAGN, Central gal, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.savefig('../Plots/HAGN_Matching/ClotMatch/Density/dens_msmh' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')


"""Plot Hexbins of everything for original lightcone catalog"""

# Trace a line for node distance vs halo mass
# x = np.linspace(10, 14)
# y = 0.375*x - 4.75


for i in range(numzbin-1):
    plt.figure()
    indices = np.where(np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1))
    # indices = np.where(np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] > 1))
    # indices = np.where(hal_centgal[i] > 0)
    plt.hexbin(
        # np.log10(halodata[i]['mass'][indices]*10**11),
        # np.log10(galdata[i]['mass'][hal_centgal[i][indices]-1]*10**11),
        np.log10(halodata[i]['Mass'][indices]*10**11),
        # np.log10(halodata[i]['mvir'][indices]*10**11),
        np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11),
        # np.log10(haloes_env[i][indices, 1][0]),
        C=np.log10(galdata[i]['SFRcorr'][hal_centgal[i][indices]-1] /
                   (galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11)),
        # C=np.log10(galdata[i]['spin'][hal_centgal[i][indices]-1]),
        # C=np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]/halodata[i]['Mass'][indices]),
        # C=np.log10(haloes_env[i][indices, 1][0]),
        # C=np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]),
        gridsize=60, mincnt=50, cmap='jet', extent=[10, 14, 8, 12]
    )
    cb = plt.colorbar()
    cb.set_label('Log(sSFR)', size=12)
    plt.xlabel('Log(Halo Mass)', size=12)
    plt.ylabel('Log(Stellar Mass)', size=12)
    plt.title('Original HorizonAGN, Central haloes, z=' +
              str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.savefig('../Plots/HorizonAGN/Hexbins/NodesFilaments/HM_Fil_MsMh_' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')

"""Plot Hexbins on one fig"""

boofitsSMbins = np.linspace(8, 12, num=100)
fig = plt.figure(figsize=(12, 4))
gs = GridSpec(1, 3, width_ratios=[1, 1, 1])
for i in range(3):
    ax1 = plt.subplot(gs[i])
    indices = np.where(np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1))
    # indices = np.where(np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] > 1))
    # indices = np.where(hal_centgal[i] > 0)
    im = ax1.hexbin(
        np.log10(halodata[i]['mass'][indices]*10**11),
        # np.log10(galdata[i]['mass'][hal_centgal[i][indices]-1]*10**11),
        # np.log10(halodata[i]['Mass'][indices]*10**11),
        # np.log10(halodata[i]['mvir'][indices]*10**11),
        # np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11),
        np.log10(haloes_env[i][indices, 0][0]),
        # C=np.log10(galdata[i]['SFRcorr'][hal_centgal[i][indices]-1] /
        #            (galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11)),
        # C=np.log10(galdata[i]['SFRcorr'][hal_centgal[i][indices]-1]),
        # C=np.log10(galdata[i]['spin'][hal_centgal[i][indices]-1]),
        C=np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]/halodata[i]['Mass'][indices]),
        # C=np.log10(haloes_env[i][indices, 1][0]),
        # C=np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]),
        # gridsize=60, mincnt=50, cmap='jet', extent=[10, 14, 8, 12]
        gridsize=60, mincnt=50, cmap='viridis', extent=[10, 12.5, -2.5, 0.5]
    )
    # ax1.plot(boo_MhMs(boofitsSMbins, *boo_fit_true[i]),
    #          boofitsSMbins,
    #          label=str('Behroozi function fit'), c='black')
    # ax1.axvline(x=MhaloPeak_true_boo[i], linestyle='--')
    axins1 = inset_axes(
        ax1,
        width="10%",  # width = 10% of parent_bbox width
        height="35%",  # height : 50%
        loc=8,
        bbox_to_anchor=[0.5, 0, 0.5, 1],
        bbox_transform=ax1.transAxes
        )
    cbar = fig.colorbar(im, cax=axins1)
    # cbar.set_label('Log(sSFR)', size=10)
    # ax1.set_xlabel('Log($M_{h}/M_{\odot}$)', size=12)
    # ax1.set_ylabel('Log($M_{*}/M_{\odot}$)', size=12)
    cbar.set_label('Log(Ms)', size=10)
    # cbar.set_label('Log(SFR)', size=10)
    ax1.set_xlabel('Log($M_{h}/M_{\odot}$)', size=12)
    # ax1.set_ylabel('Log($d_{node}/Mpc$)', size=12)
    ax1.set_ylabel('Log($n(halo)/Mpc^{-3}$)', size=12)
    cbar.ax.tick_params(labelsize=9)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    plt.text(0.7, 0.9, str(zbins_Cone[i])+'<z<'+str(zbins_Cone[i+1]),
             size=12, transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white'))
fig.tight_layout()

"""Plot sSFR hexbins for photo catalog"""

fig = plt.figure(figsize=(12, 4))
gs = GridSpec(1, 3, width_ratios=[1, 1, 1])
for i in range(3):
    ax1 = plt.subplot(gs[i])
    indices = np.where(
        np.logical_and(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
            np.logical_and(
                galdata_allz['Obs_gal_idx'][
                    hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
                ] > 0,
                galphot['Mass'][
                    galdata_allz['Obs_gal_idx'][
                        hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
                    ].astype('int')] > 9
            )
        )
    )
    # indices = np.where(np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] > 1))
    # indices = np.where(hal_centgal[i] > 0)
    im = ax1.hexbin(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        galphot['Mass'][
                galdata_allz['Obs_gal_idx'][
                    hal_centgal[i][indices] - 1 + sum(len(galdata[j]) for j in range(i))
                ].astype('int')
        ],
        C=np.log10(galphot['SFR'][
            galdata_allz['Obs_gal_idx'][
                hal_centgal[i][indices] - 1 + sum(len(galdata[j]) for j in range(i))
            ].astype('int')
        ] / 10**(galphot['Mass'][
            galdata_allz['Obs_gal_idx'][
                hal_centgal[i][indices] - 1 + sum(len(galdata[j]) for j in range(i))
            ].astype('int')])
        ),
        gridsize=60, mincnt=50, cmap='jet', extent=[10, 14, 8, 12],
    )
    axins1 = inset_axes(
        ax1,
        width="10%",  # width = 10% of parent_bbox width
        height="35%",  # height : 50%
        loc=8,
        bbox_to_anchor=[0.5, 0, 0.5, 1],
        bbox_transform=ax1.transAxes
        )
    cbar = fig.colorbar(im, cax=axins1)
    # cbar.set_label('Log(sSFR)', size=10)
    ax1.set_xlabel('Log($M_{h}/M_{\odot}$)', size=12)
    ax1.set_ylabel('Log($M_{*}/M_{\odot}$)', size=12)
    cbar.ax.tick_params(labelsize=9)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    plt.text(0.1, 0.9, str(zbins_Cone[i])+'<z<'+str(zbins_Cone[i+1]),
             size=12, transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white'))
fig.tight_layout()


"""Plot sSFR versus Halo mass"""

for i in range(numzbin-1):
    plt.figure()
    indices = np.where(
        np.logical_and(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
            np.log10(halodata[i]['Mass']*10**11) > 0
        )
    )
    plt.hist2d(
        np.log10(galdata[i]['SFRcorr'][hal_centgal[i][indices]-1] /
                 (galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11)),
        np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1] / halodata[i]['Mass'][indices]),
        range=[[-12, -8], [-4, 0]], bins=100, cmin=1
    )
    plt.xlabel('sSFR', size=20)
    plt.ylabel('HaloMass', size=20)


"""Plot sSFR vs SM/HM"""

for i in range(numzbin):
    plt.figure()
    indices = np.where(np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1))
    plt.hist2d(
        np.log10(galdata[i]['SFRcorr'][hal_centgal[i][indices]-1]),
        galdata[i]['Mass'][hal_centgal[i][indices]-1]/halodata[i]['Mass'][indices],
        range=[[-2, 2], [-4, 0]], bins=100, cmin=20
    )
    plt.colorbar()
    plt.xlabel('Log(SFR)', size=20)
    plt.ylabel('Log(SM/HM)', size=20)
    plt.title('Original HorizonAGN, Central gal, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.tight_layout()

"""Select galaxies with distance to node < 10**-0.5"""

d = 10**-0.5
for i in range(numzbin-1):
    plt.figure()
    # plot histogram for halos with distance to node > 10**-0.5 Mpc
    indices = np.where(
        np.logical_and(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
            haloes_env[i][:, 2] > d
        )
    )
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11),
        bins=100, cmin=1)
    plt.colorbar()
    # add a scatter for haloes > 10**-0.5 Mpc
    indices = np.where(
        np.logical_and(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
            haloes_env[i][:, 2] < d
        )
    )
    print('N haloes close to nodes : ' + str(len(indices[0])))
    plt.scatter(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11),
        c='red', label=('Haloes with d(Node)<10**-0.5 Mpc'))
    plt.legend()
    plt.xlabel('Log(Halo Mass)', size=12)
    plt.ylabel('Log(Stellar Mass)', size=12)
    plt.title('Original HorizonAGN, Central gal, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.savefig('../Plots/HorizonAGN/Hexbins/NodesFilaments/Ms_Mh_distanceSeparation' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')


"""Plot Hexbins for the photometric catalog"""

# selection of relevant galaxies (central with level 1 halo and matched)
indices_allz = []
galphotselec = []
for i in range(numzbin-1):
    indices_allz.append(np.where(
        np.logical_and(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
            galdata_allz['Obs_gal_idx'][
                    hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))] > 0
        )
    ))
    galphotselec.append(galphot[
                galdata_allz['Obs_gal_idx'][
                    hal_centgal[i][:] - 1 + sum(len(galdata[j]) for j in range(i))
                ].astype('int')
        ])

for i in range(numzbin-1):
    plt.figure()
    indices = np.intersect1d(indices_allz[i], np.where(galphotselec[i]['Mass'] > 9))
    plt.hexbin(
        galphotselec[i]['Mass'][indices],
        # galphotselec[i]['mag_u'][indices],
        galphotselec[i]['mag_J'][indices],
        C=galphotselec[i]['Mass'][indices] - np.log10(halodata[i]['Mass'][indices]*10**11),
        # np.log10(haloes_env[i][indices, 2][0]),
        # galphotselec[i]['Mass'][indices],
        # C=np.log10(galphotselec[i]['SFR'][indices]/(galphotselec[i]['Mass'][indices]*10**11)),
        # C=np.log10(galphotselec[i]['SFR'][indices]),
        # C=np.log10(haloes_env[i][indices, 2][0]),
        # galphotselec[i]['mag_K'][indices],
        # C=galphotselec[i]['mag_J'][indices]-galphotselec[i]['mag_u'][indices],
        gridsize=60, mincnt=20, cmap='jet', extent=[9, 12, 20, 30]
    )
    cb = plt.colorbar()
    cb.set_label('Log(Ms/Mh)', size=12)
    plt.xlabel('Stellar mass', size=12)
    plt.ylabel('Mag J', size=12)
    plt.title('Photometric HorizonAGN, Central gal, z=' +
              str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.savefig('../Plots/HAGN_Matching/ClotMatch/Hexbins/Colors/J_U_MsMH_' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')


"""Plot gas mass vs Halo mass"""

for i in range(numzbin-1):
    plt.figure()
    indices = indices_allz[i]
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        np.log10(galphotselec[i]['Gas_mass'][indices] / (halodata[i]['Mass'][indices]*10**11)),
        bins=50, cmin=20, range=[[10, 12], [-1.5, -0.5]]
    )
    plt.xlabel('Log(Halo mass)', size=12)
    plt.ylabel('Log(Gas mass)', size=12)
    plt.title('Photometric HorizonAGN, Central gal, z=' +
              str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')

""" Compute average gas mass per halo mass"""


def averageperHM(data, data_name, indices_selec, numzbin, massbins):
    """Retun average, median and standard eviation of the data per halo mass.

    Routine to compute useful info on the data.
    Warning : it is full of particular cases, as for instance for gas mass I take only
    positiove masses, and I compute them on a logscale.
    """
    medperHM = np.zeros([numzbin, np.size(massbins)-1])
    avperHM = np.zeros([numzbin, np.size(massbins)-1])
    stdperHM = np.zeros([numzbin, np.size(massbins)-1])

    for i in range(numzbin):
        for j in range(np.size(massbins)-1):
            m1 = massbins[j]
            m2 = massbins[j+1]
            indices = np.where(np.logical_and(
                np.logical_and(
                    np.log10(halodata[i]['Mass']*10**11) > m1,
                    np.log10(halodata[i]['Mass']*10**11) <= m2),
                data[i][data_name] > 0
                ))[0]
            indices = np.intersect1d(indices_selec[i], indices)
            if len(indices) > 0:
                avperHM[i, j] = np.average(np.log10(data[i][data_name][indices]))
                medperHM[i, j] = np.median(np.log10(data[i][data_name][indices]))
                stdperHM[i, j] = np.std(np.log10(data[i][data_name][indices]))
            else:
                avperHM[i, j] = np.nan
                medperHM[i, j] = np.nan
                stdperHM[i, j] = np.nan

    return avperHM, medperHM, stdperHM


massbins = np.linspace(10, 13, num=20)
avGMperHM, medGMperHM, stdGMperHM = averageperHM(galphotselec, 'Gas_mass',
                                                 indices_allz, 3, massbins)


"""Plot Gas_mass versus Halo_mass"""

for i in range(numzbin-1):
    plt.figure()
    indices = indices_allz[i]
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        np.log10(galphotselec[i]['Gas_mass'][indices]) / np.log10(
            halodata[i]['Mass'][indices]*10**11),
        bins=100, cmin=1, range=[[10, 13], [0.6, 1.1]]
    )
    plt.colorbar()
    # plt.errorbar(
    #     (massbins[:-1]+massbins[1:])/2, avGMperHM[i],
    #     yerr=stdGMperHM[i], color='red'
    # )
    plt.xlabel('Log(Halo virial mass)', size=12)
    plt.ylabel('Log(Gas virial mass)/Log(Halo Mass)', size=12)
    plt.title('Photometric HorizonAGN, Central gal, z=' +
              str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')
    # plt.savefig('../Plots/HAGN_Matching/ClotMatch/Hexbins/GasMass/logGMonlogHVM_' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')


for i in range(numzbin-1):
    plt.figure()
    indices = np.intersect1d(indices_allz[i], np.where(galphotselec[i]['Mass'] > 0))
    plt.hexbin(
        np.log10(halodata[i]['mvir'][indices]*10**11),
        np.log10(galphotselec[i]['Gas_mass'][indices]) / np.log10(
            halodata[i]['mvir'][indices]*10**11),
        # C=galphotselec[i]['Mass'][indices] - np.log10(
        #     halodata[i]['Mass'][indices]*10**11) ,
        gridsize=60, mincnt=10, cmap='jet', extent=[10, 13, 0.6, 1.1]
    )
    cb = plt.colorbar()
    cb.set_label('Log(Ms/Mh)', size=12)
    plt.xlabel('Log(Halo mass)', size=12)
    plt.ylabel('Log(Gas mass)/Log(Halo Mass)', size=12)
    plt.title('Photometric HorizonAGN, Central gal, z=' +
              str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')
    # plt.savefig('../Plots/HAGN_Matching/ClotMatch/Hexbins/GasMass/logGMonlogHM_' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')


cut = 0.85
for i in range(numzbin-1):
    plt.figure()
    # plot histogram for halos with distance to node > 10**-0.5 Mpc
    indices = indices_allz[i]
    indices = np.intersect1d(indices, np.where(galphotselec[i]['Mass'] > 9))
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        galphotselec[i]['Mass'][indices],
        bins=100, cmin=1)
    plt.colorbar()
    # add a scatter for haloes > 10**-0.5 Mpc
    indices = np.intersect1d(indices,
            np.where(np.log10(galphotselec[i]['Gas_mass'][:]) / np.log10(
                halodata[i]['mvir'][:]*10**11) < cut)
        )
    print('N haloes inferior at cut : ' + str(len(indices)))
    plt.scatter(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        galphotselec[i]['Mass'][indices],
        c='red', label=('Haloes with d(Node)<10**-0.5 Mpc'))
    plt.legend()
    plt.xlabel('Log(Halo Mass)', size=12)
    plt.ylabel('Log(Stellar Mass)', size=12)
    plt.title('Original HorizonAGN, Central gal, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.savefig('../Plots/HorizonAGN/Hexbins/NodesFilaments/Ms_Mh_distanceSeparation' +


"""Plot colors"""

for i in range(numzbin-1):
    plt.figure()
    indices = np.intersect1d(indices_allz[i], np.where(galphotselec[i]['Mass'] > 9))
    plt.hist2d(
        # galphotselec[i]['Mass'][indices],
        np.log10(halodata[i]['Mass'][indices]*10**11),
        galphotselec[i]['mag_u'][indices],
        cmin=1, bins=50
    )


"""Test de faire des corner plot"""

from getdist import plots, MCSamples

i = 0
indices = indices_allz[i]
indices = np.intersect1d(indices_allz[i], np.where(galphotselec[i]['Mass'] > 9))
indices = np.intersect1d(indices, np.where(galphotselec[i]['Gas_mass'] > 0) )
# names = ['Ms', 'Mh', 'Ms/Mh', 'J-U', 'U-R']
# data = [
#     galphotselec[i]['Mass'][indices],
#     np.log10(halodata[i]['Mass'][indices]*10**11),
#     galphotselec[i]['Mass'][indices] - np.log10(halodata[i]['Mass'][indices]*10**11),
#     galphotselec[i]['mag_J'][indices] - galphotselec[i]['mag_u'][indices],
#     galphotselec[i]['mag_u'][indices] - galphotselec[i]['mag_r'][indices],
#     ]
names = ['Ms', 'Mh', 'Mg', 'log(Mg)/log(Mh)']
data = [
    galphotselec[i]['Mass'][indices],
    np.log10(halodata[i]['Mass'][indices]*10**11),
    np.log10(galphotselec[i]['Gas_mass'][indices]),
    np.log10(galphotselec[i]['Gas_mass'][indices])/np.log10(halodata[i]['Mass'][indices]*10**11),
    ]
samples = MCSamples(samples=data, names=names)
# Si l'on souhaite changer les zones de confiance des graphs,
# par défaut ce sont les zones de confiance à 65% et 95%
samples.contours = np.array([0.68, 0.95, 0.99])
samples.updateBaseStatistics()


g = plots.getSubplotPlotter()
g.settings.num_plot_contours = 3
g.triangle_plot(samples, filled=True, contours=0.2)
#g.export('statistiques')
#plt.close('all')



"""Try to do Principal component analysis on the data"""

i=2
indices = np.intersect1d(indices_allz[i], np.where(galphotselec[i]['Mass'] > 9))
indices = np.intersect1d(indices, np.where(galphotselec[i]['Gas_mass'] > 0) )
data = np.transpose(np.array([
    galphotselec[i]['Mass'][indices],
    np.log10(halodata[i]['Mass'][indices]*10**11),
    np.log10(galphotselec[i]['Gas_mass'][indices]),
    ]))

# result = mlab.PCA(data)

# from mpl_toolkits.mplot3d import Axes3D

# x = []
# y = []
# z = []
# for item in result.Y:
#  x.append(item[0])
#  y.append(item[1])
#  z.append(item[2])

# plt.close('all') # close all latent plotting windows
# fig1 = plt.figure() # Make a plotting figure
# ax = Axes3D(fig1) # use the plotting figure to create a Axis3D object.
# pltData = [x,y,z]
# ax.scatter(pltData[0], pltData[1], pltData[2], 'bo') # make a scatter plot of blue dots from the
# data

# # make simple, bare axis lines through space:
# xAxisLine = ((min(pltData[0]), max(pltData[0])), (0, 0), (0,0)) # 2 points make the x-axis line
#  at the data extrema along x-axis
# ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r') # make a red line for the x-axis.
# yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1])), (0,0)) # 2 points make the y-axis line
# at the data extrema along y-axis
# ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r') # make a red line for the y-axis.
# zAxisLine = ((0, 0), (0,0), (min(pltData[2]), max(pltData[2]))) # 2 points make the z-axis line
# at the data extrema along z-axis
# ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r') # make a red line for the z-axis.

# # label the axes
# ax.set_xlabel("x-axis label")
# ax.set_ylabel("y-axis label")
# ax.set_zlabel("y-axis label")
# ax.set_title("The title of the plot")
# plt.show() # show the plot

from sklearn.decomposition import PCA

sk_pca = PCA(n_components=2)
sklearn_result = sk_pca.fit_transform(data)

plt.plot(sklearn_result[:, 0], sklearn_result[:, 1], '.')



""""PLot Ms/Mh separating halos with environment"""

# Fit Boo on it :

boo_fit_hd = np.empty([numzbin-1, 5])
boo_cov_hd = np.empty([numzbin-1, 5, 5])
for i in range(numzbin-1):
    print(i)
    indices = np.where(
        np.logical_and(
            np.logical_and(
                np.log10(galdata[i]['Mass'][hal_centgal[i]-1]*10**11) > 9,
                np.log10(haloes_env[i][:, 0]) > -0.5
            ),
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1)
            )
    )
    boo_fit_hd[i], boo_cov_hd[i] = curve_fit(
        boo_MhMs,
        np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11),
        np.log10(halodata[i]['Mass'][indices]*10**11),
        bounds=[[10, 8, 0, 0, 0], [13, 11, 5, 5, 5]],
        method='trf')
print(boo_fit_hd)


boofitsSMbins = np.linspace(9, 12, num=100)

for i in range(numzbin-1):
    plt.figure()
    indices = np.where(
        np.logical_and(
            np.logical_and(hal_centgal[i] > 0, halodata[i]['level'] == 1),
            np.log10(haloes_env[i][:, 0]) > 0
        )
    )
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11),
        bins=100, cmin=1)
    plt.colorbar()
    # Plot Behroozi fit
    # plt.plot(boo_MhMs(boofitsSMbins, *boo_fit_true[i]), boofitsSMbins,
    #          label=str('True Behroozi function fit'), c='r')
    plt.legend()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Log($M_{*}$) [Log($M_{\odot}$)]', size=12)
    plt.title('HorizonAGN, Central galz='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.savefig('../Plots/HAGN_Matching/ClotMatchBis/TrueMass_HaloMass_Boofit' +
    #             str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) + '.pdf')


plt.figure()
for i in range(3):
    p = plt.plot(
        boo_MhMs(boofitsSMbins, *boo_fit_true[i]),
        boofitsSMbins-boo_MhMs(boofitsSMbins, *boo_fit_true[i]),
        label=str('All '+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])))
    plt.plot(
        boo_MhMs(boofitsSMbins, *boo_fit_hd[i]),
        boofitsSMbins-boo_MhMs(boofitsSMbins, *boo_fit_hd[i]),
        label=str('HD '+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])),
        color=p[0].get_color(), linestyle='--')
    plt.plot(
        boo_MhMs(boofitsSMbins, *boo_fit_ld[i]),
        boofitsSMbins-boo_MhMs(boofitsSMbins, *boo_fit_ld[i]),
        label=str('LD '+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])),
        color=p[0].get_color(), linestyle=':')
plt.legend()
plt.xlabel('Log($M_{h}/M_{\odot}$)', size=12)
plt.ylabel('Log($M_{*}/M_{h}$)', size=12)

"""Load data to compute number of galaxies per halo"""


# Main halos
gal_mainhaloes = []
mainHaloMass = []
for i in range(np.size(zbins_Cone)-2):
    gal_mainhaloes.append(
        np.loadtxt('../Data/HorizonAGNLaigleCatalogs/Cat_' +
                   str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+'_Gal_MainHaloes_newb.txt',
                   dtype='i4'))
    mainHaloMass.append(halodata[i]['Mass'][gal_mainhaloes[i][:].astype(int)-1])


# Sub halos
gal_subhaloes = []
subHaloMass = []
for i in range(np.size(zbins_Cone)-2):
    gal_subhaloes.append(
        np.loadtxt('../Data/HorizonAGNLaigleCatalogs/Cat_' +
                   str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+'_Gal_SubHaloes_newb.txt',
                   dtype='i4'))
    subHaloMass.append(halodata[i]['Mass'][gal_subhaloes[i][:].astype(int)-1])


"""Number of galaxies per halo"""

# Number of halo with minimal mass
# minimum = min(10**11*halodata['Mass'])
# indices = [i for i, v in enumerate(10**11*halodata['Mass']) if v == minimum]
# np.size(indices)


nbgalaxiesperhalos = []
for i in range(numzbin-1):
    # index j of nbgalaxiesperhalos gives the number of galaxies in the halo of
    # ID = j+1
    nbgalaxiesperhalos.append(np.zeros(np.size(halodata[i]['Mass'])))
    for j in gal_subhaloes[i].astype(int):
        nbgalaxiesperhalos[i][j-1] += 1

# I dont know if I have to use mainhalos or subhalos, but it seems that main
# halos give better results for the number of galaxy per halos

nbgalaxiesperhalos_main = []
for i in range(numzbin-1):
    # index j of nbgalaxiesperhalos gives the number of galaxies in the halo of
    # ID = j+1
    nbgalaxiesperhalos_main.append(np.zeros(np.size(halodata[i]['Mass'])))
    for j in gal_mainhaloes[i].astype(int):
        nbgalaxiesperhalos_main[i][j-1] += 1


nb_centralgalaxies_per_mainhalo = []
for i in range(numzbin-1):
    print(i)
    nb_centralgalaxies_per_mainhalo.append(np.zeros(np.size(halodata[i]['Mass'])))
    indices = np.where(np.logical_and(
            galdata[i]['level'] == 1,
            gal_mainhaloes[i] > 0))
    for j in gal_mainhaloes[i][indices]:
        nb_centralgalaxies_per_mainhalo[i][j-1] += 1

nb_satgalaxies_per_mainhalo = []
for i in range(numzbin-1):
    print(i)
    nb_satgalaxies_per_mainhalo.append(np.zeros(np.size(halodata[i]['Mass'])))
    indices = np.where(np.logical_and(
            galdata[i]['level'] > 1,
            gal_mainhaloes[i] > 0))
    for j in gal_mainhaloes[i][indices]:
        nb_satgalaxies_per_mainhalo[i][j-1] += 1

"""Plot"""
# for i in range(4):
#     plt.hist(nbgalaxiesperhalos[i], bins=range(nbgalaxiesperhalos[i].max().astype(int)))
#     plt.yscale('log')
#     plt.show()

"""Number galaxy per halo versus Halo Mass"""

# Compute Average mass of halos for a given number of galaxies in the halo

# averageHaloMassPerNgal = []
# for i in range(numzbin-1):
#     averageHaloMassPerNgal.append(np.empty(nbgalaxiesperhalos_main[i].astype(int).max()+1))
#     for j in range(nbgalaxiesperhalos_main[i].astype(int).max()+1):
#         averageHaloMassPerNgal[i][j] = np.mean(
#             halodata[i]['Mass'][nbgalaxiesperhalos_main[i] == j])

# Compute average number of galaxies in halos given a halo mass interval

massbins = np.linspace(10, 15, num=100)
averageNgalperHaloMass = np.zeros([numzbin-1, np.size(massbins)-1])
av_centralgalaxies_per_mainhalo = np.zeros([numzbin-1, np.size(massbins)-1])
av_satgalaxies_per_mainhalo = np.zeros([numzbin-1, np.size(massbins)-1])

for i in range(numzbin-1):
    for j in range(np.size(massbins)-1):
        m1 = massbins[j]
        m2 = massbins[j+1]
        averageNgalperHaloMass[i][j] = np.average(
            nbgalaxiesperhalos_main[i][
                np.logical_and(
                    np.log10(halodata[i]['Mass']*10**11) > m1,
                    np.log10(halodata[i]['Mass']*10**11) < m2)
            ])
        av_centralgalaxies_per_mainhalo[i][j] = np.average(
            nb_centralgalaxies_per_mainhalo[i][
                np.logical_and(
                    np.log10(halodata[i]['Mass']*10**11) > m1,
                    np.log10(halodata[i]['Mass']*10**11) < m2)
            ])
        av_satgalaxies_per_mainhalo[i][j] = np.average(
            nb_satgalaxies_per_mainhalo[i][
                np.logical_and(
                    np.log10(halodata[i]['Mass']*10**11) > m1,
                    np.log10(halodata[i]['Mass']*10**11) < m2)
            ])


massbins = np.linspace(10, 15, num=100)
averageNgalperSubHaloMass = np.zeros([numzbin, np.size(massbins)-1])

for i in range(numzbin-1):
    for j in range(np.size(massbins)-1):
        m1 = massbins[j]
        m2 = massbins[j+1]
        averageNgalperSubHaloMass[i][j] = np.average(
            nbgalaxiesperhalos[i][
                np.logical_and(
                    np.log10(halodata[i]['Mass']*10**11) > m1,
                    np.log10(halodata[i]['Mass']*10**11) < m2)
            ])

"""Plot"""

# plt.hist2d(np.log10(halodata[0]['Mass'][nbgalaxiesperhalos_main[0]>0]*10**11),
#     nbgalaxiesperhalos_main[0][nbgalaxiesperhalos_main[0]>0], bins=100, cmin=1)


# for i in range(4):
#     fig = plt.figure()
#     plt.yscale('log')
#     plt.scatter(np.log10(halodata[i]['Mass'][nbgalaxiesperhalos_main[i]>0]*10**11),
#      nbgalaxiesperhalos_main[i][nbgalaxiesperhalos_main[i]>0],
#         marker='.')
#     # plt.scatter(np.log10(averageHaloMassPerNgal[i][1:]*10**11),
#     # np.arange(1, nbgalaxiesperhalos_main[i].astype(int).max()+1), label='Average Mass')
#     plt.title('HorizonAGN, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) +
#         ', match gal-Mainhalo')
#     plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]')
#     plt.ylabel('Number of galaxies in the halo')
#     plt.legend()
# plt.show()

for i in range(numzbin-1):
    plt.scatter(
        (massbins[:-1]+massbins[1:])/2, av_satgalaxies_per_mainhalo[i][:],
        label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
plt.yscale('log')
plt.ylabel('Average number of galaxies per halo', size=15)
plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)

