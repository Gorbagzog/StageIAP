#!/usr/bin/env python3
# -*-coding:Utf-8 -*

import numpy as np
import pyfits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time

""" Code for loading the Clotilde Laigle Catalog on Horizon AGN and performing
computations with it.
"""

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

# plt.hist(np.log10(galdata[0]['Mass']*10**11), bins=100)


"""It looks like the good catalogs to use are the Haloes and not the Halos"""

halodata = []
for i in range(np.size(zbins_Cone)-1):
    hdulist2 = pyfits.open('../Data/HorizonAGNLaigleCatalogs/Haloes_' +
                           str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+'.fits')
    halodata.append(hdulist2[1].data)
    hdulist2.close()

"""Load the index of halos corresponding to each galaxy"""

# Indexes of the halos in the files gal_mainhaloes and gal_subhaloes
# start from 1, whereas the python arrays start at 0, so you have to translate
# all IDs given by gal_*halos by 1.
# gal_mainhaloes[redshift][id_gal-1] gives  the id of the main haloes containing
# the galaxy id_gal

# Main halos
gal_mainhaloes = []
mainHaloMass = []
for i in range(np.size(zbins_Cone)-1):
    gal_mainhaloes.append(
        np.loadtxt('../Data/HorizonAGNLaigleCatalogs/Cat_' +
                   str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+'_Gal_MainHaloes_new.txt',
                   dtype='i4'))
    mainHaloMass.append(halodata[i]['Mass'][gal_mainhaloes[i][:].astype(int)-1])


# Sub halos
gal_subhaloes = []
subHaloMass = []
for i in range(np.size(zbins_Cone)-1):
    gal_subhaloes.append(
        np.loadtxt('../Data/HorizonAGNLaigleCatalogs/Cat_' +
                   str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+'_Gal_SubHaloes_new.txt',
                   dtype='i4'))
    subHaloMass.append(halodata[i]['Mass'][gal_subhaloes[i][:].astype(int)-1])

# Haloes central gal
# contains the ID of the central galaxy in each halo
hal_centgal = []
for i in range(np.size(zbins_Cone)-1):
    hal_centgal.append(
        np.loadtxt('../Data/HorizonAGNLaigleCatalogs/Cat_' +
                   str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+'_Hal_CentralGal_new.txt',
                   dtype='i4'))


"""Load haloes environment data"""

haloes_env = []
for i in range(3):
    haloes_env.append(
        np.loadtxt('../Data/HorizonAGNLaigleCatalogs/Haloes_' +
                   str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+'_env.txt'))


""" Stellar Mass versus redshift """

# Concatenate redshift and mass arrays to plot a homogeneous histogram (ie with
# bins of the same size)

# galZall = []
# galMassall = []
# for i in range(4):
#     galZall.append(galdata[i]['z'])
#     galMassall.append(np.log10(galdata[i]['Mass']*10**11))
# galZall = np.concatenate(galZall[:],0)
# galMassall = np.concatenate(galMassall[:],0)
#
# plt.hist2d(galZall, galMassall, bins=1000, cmin=1)
# plt.xlim(0,6)
# plt.ylim(8, 12)
# plt.title('HorizonAGN-CONE, $M_{*}$ vs z')
# plt.xlabel('Redshift')
# plt.ylabel('Log($M_{*}/M_{\odot}$)')
# plt.colorbar()
#
# plt.text(3, 11.5, 'Z bins = [0,1], [1,2], [2,3], [3,6]')

"""Fit the Yang relation on the M*/Mh relation"""


def mstar_over_mh_yang(x, A, m1, beta, gamma):
    """Yang et al. 2004 function, see Moster et al. 2010."""
    return 2.0*A*((x/m1)**(-beta)+(x/m1)**gamma)**(-1)


# xm =subHaloMass*10**11
# ym = galdata[:]['Mass']/subHaloMass[:]
yang_fit = np.empty([numzbin, 4])
yang_cov = np.empty([numzbin, 4, 4])


for i in range(numzbin):
    yang_fit[i], yang_cov[i] = curve_fit(
        mstar_over_mh_yang,
        subHaloMass[i]*10**11, galdata[i]['Mass']/subHaloMass[i],
        p0=[0.1, 10**12, 0.1, 0.1],
        bounds=[[-np.inf, 10**9, 0, 0], [np.inf, 10**14, 5, 5]], method='trf')

"""Plot"""

# x = np.logspace(10, 14, num=1000)
# for i in range(numzbin):
#     plt.plot(x, mstar_over_mh_yang(x, *yang_fit[i]))
#

"""Plot M*/Mh vs Mh in Horizon AGN"""

# fig, ax = plt.subplots(2,2)
# #fig.suptitle('Horizon AGN CONE, WARNING !! colorbars not homogeneous')
# for i in range(4):
#     ax1 = ax[i//2, i%2]
#     if i==0:
#         counts,xedges,yedges, im =ax1.hist2d(np.log10(subHaloMass[i]*10**11),
#             np.log10(galdata[i]['Mass']/subHaloMass[i]), bins=100, cmin=1)
#     else :
#         _,_,_,im = ax1.hist2d(np.log10(subHaloMass[i]*10**11),
#         np.log10(galdata[i]['Mass']/subHaloMass[i]), bins=(xedges, yedges), cmin=1)
#     axins1 = inset_axes(ax1,
#                     width="5%",  # width = 10% of parent_bbox width
#                     height="35%",  # height : 50%
#                     loc=3)
#     ax1.set_xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
#     ax1.set_ylabel('Log($M_{*}/M_{h}$)', size=12)
#     cbar = fig.colorbar(im, cax=axins1)
#     cbar.ax.tick_params(labelsize=9)
#     tick_locator = ticker.MaxNLocator(nbins=5)
#     cbar.locator = tick_locator
#     cbar.update_ticks()
#
#     x = np.logspace(10, 14, num=1000)
#     ax1.plot(np.log10(x), np.log10(mstar_over_mh_yang(x, *yang_fit[i])), c='red')
#
#     plt.text(0.7, 0.8, str(zbins_Cone[i])+'<z<'+str(zbins_Cone[i+1]),
#         size= 12, transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white'))
# #fig.tight_layout()
# #plt.subplots_adjust(top=0.95)
# plt.show()


"""Mstar vs Mh plot """

# plt.hist2d(np.log10(subHaloMass[i]*10**11), np.log10(galdata[i]['Mass']*10**11), cmin=1,
#            bins=100)
# plt.ylabel('Stellar Mass')
# plt.xlabel('Halo Mass')

"""Age/Metallicity/SFR of galaxies vs Mh plot """

# for i in range(4):
#     plt.figure()
#     plt.hist2d(np.log10(subHaloMass[i]*10**11), np.log10(galdata[i]['agegal2']), cmin=1,
#                bins=100)
#     plt.ylabel('Age of Galaxies in the halo')
#     plt.xlabel('Halo Mass')
#     plt.title('z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))

# for i in range(4):
#     plt.figure()
#     hist = hist2d(np.log10(subHaloMass[i]*10**11), np.log10(galdata[i]['met']), cmin=1, bins=100)
#     moy = np.empty(np.size(hist[1])-1)
#     for j in range(size(hist[1])-1):
#         indices = ~np.isnan(hist[0][j])
#         if np.size(hist[0][j][indices])==0:
#             moy[j] = np.nan
#         else:
#             moy[j] = np.average(((hist[2][1:]+hist[2][:-1])/2)[indices],
#                                 weights=hist[0][j][indices])
#
#     plt.plot(((hist[1][1:]+hist[1][:-1])/2), moy, c='red',
#              label='Average metalicity of galaxies in the halo')
#     plt.ylabel('Log(Metalicity)')
#     plt.xlabel('Log($M_{h}/M_{\odot}$)')
#     plt.legend(loc=1)
#     plt.title('z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
#     plt.savefig('../Plots/HorizonAGN/AgeOfGal/MetalicityVsHaloMass_'+
#         str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+ '_log.pdf')

# Compute average directly for halo mass bins

massbins = np.linspace(10, 15, num=600)
averageAgeperHaloMass = np.zeros([numzbin, np.size(massbins)-1])
averageMetperHaloMass = np.zeros([numzbin, np.size(massbins)-1])
averageSFRperHaloMass = np.zeros([numzbin, np.size(massbins)-1])
averageSSFRperHaloMass = np.zeros([numzbin, np.size(massbins)-1])
# averageCentralSFRperHaloMass = np.zeros([numzbin, np.size(massbins)-1])
for i in range(numzbin):
    for j in range(np.size(massbins)-1):
        m1 = massbins[j]
        m2 = massbins[j+1]
        # select indices of galaxies contained in the haloes with a mass
        # between m1 and m2 :
        indices = np.where(np.logical_and(np.log10(
            halodata[i]['Mass'][gal_mainhaloes[i][:]-1]*10**11) > m1,
            np.log10(halodata[i]['Mass'][gal_mainhaloes[i][:]-1]*10**11) <= m2))
        averageAgeperHaloMass[i][j] = np.average(galdata[i]['agegal2'][indices])
        averageMetperHaloMass[i][j] = np.average(galdata[i]['met'][indices])
        averageSFRperHaloMass[i][j] = np.average(galdata[i]['SFR2'][indices])
        # averageCentralSFRperHaloMass[i][j] = np.average(
        #     galdata[i]['SFR2'][indices][galdata[i]['level'][indices]==1])
        averageSSFRperHaloMass[i][j] = np.average(
            galdata[i]['SFR2'][indices]/(galdata[i]['Mass'][indices]*10**11))

"""Plot"""


for i in range(4):
    plt.scatter((massbins[:-1]+massbins[1:])/2, averageAgeperHaloMass[i][:]*10**-9,
                label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]), marker='.')
plt.yscale('log')
plt.ylabel('Average Age of galaxies [Gyr]', size=15)
plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
plt.legend()

plt.figure()
for i in range(4):
    plt.scatter((massbins[:-1]+massbins[1:])/2, averageMetperHaloMass[i][:],
                label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]), marker='.')
plt.ylabel('Average Metallicity of galaxies', size=15)
plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
plt.legend()

plt.figure()
plt.yscale('log')
for i in range(4):
    plt.scatter((massbins[:-1]+massbins[1:])/2, averageSFRperHaloMass[i][:],
                label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]), marker='.')
    # plt.scatter((massbins[:-1]+massbins[1:])/2, averageCentralSFRperHaloMass[i][:],
    # marker='+' )
plt.ylabel('Average SFR of galaxies [$M_{\odot}$/yr]', size=15)
plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
plt.legend()


# scale= np.zeros(4)
# for i in range(4):
#     scale[i]= 1/averageSSFRperHaloMass[i][~np.isnan(averageSSFRperHaloMass[i])][0]
# scale= [10, 4, 2, 1]
scale = [1, 1, 1, 1]

plt.figure()
# plt.yscale('log')
for i in range(4):
    plt.scatter(
        (massbins[:-1]+massbins[1:])/2, averageSSFRperHaloMass[i][:]*scale[i],
        label='scale factor = '+str(scale[i])+'; z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]),
        marker='.')
plt.ylabel('Average sSFR of galaxies [$yr^{-1}$]', size=15)
plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
plt.ylim(10**-12, 10**-8)
plt.legend()

# plt.figure()
# for i in range(4):
#     plt.scatter((massbins[:-1]+massbins[1:])/2, averageCentralSFRperHaloMass[i][:],
#         label = 'z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]), marker='.' )
# plt.ylabel('Average SFR of Central galaxies [$M_{\odot}$/yr]', size=15)
# plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
# plt.legend()

"""Age/metallicity/SFR of galaxies VS Stellar massbins"""

for i in range(numzbin):
    plt.figure()
    plt.hist2d(np.log10(galdata[i]['Mass']*10**11), galdata[i]['met'],
               bins=100, cmin=1)
    plt.xlabel('$Log(M_{*}/M_{\odot})$')
    plt.ylabel('Log(Metallicity)')
    plt.colorbar()
    plt.title('z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.savefig('../Plots/HorizonAGN/PropertiesMS/MetMs_z='+str(zbins_Cone[i])+
    # '-'+str(zbins_Cone[i+1])+'.pdf')


# Compute average properties of galaxies for a given bin in stellar mass

stellarmassbins = np.linspace(8.2, 12.3, num=100)
avMetPerSM = np.zeros([numzbin, np.size(stellarmassbins)-1])
avSFRPerSM = np.zeros([numzbin, np.size(stellarmassbins)-1])
avsSFRPerSM = np.zeros([numzbin, np.size(stellarmassbins)-1])
avAgePerSM = np.zeros([numzbin, np.size(stellarmassbins)-1])

for i in range(numzbin):
    for j in range(np.size(stellarmassbins)-1):
        m1 = stellarmassbins[j]
        m2 = stellarmassbins[j+1]
        # select indices of central galaxies with a mass
        # between m1 and m2 :
        indices = np.where(np.logical_and(
            np.log10(galdata[i]['Mass'][:]*10**11) > m1,
            np.log10(galdata[i]['Mass'][:]*10**11) <= m2))[0]
        # avMetPerSM[i,j] = np.average(galdata[i]['met'][indices])
        # avSFRPerSM[i,j] = np.average(galdata[i]['SFR2'][indices])
        # avAgePerSM[i,j] = np.average(galdata[i]['agegal2'][indices])
        avsSFRPerSM[i, j] = np.average(galdata[i]['SFR2'][indices] /
                                       (galdata[i]['Mass'][indices]*10**11))

"""PLot"""

plt.figure()
plt.yscale('log')
for i in range(numzbin):
    plt.scatter(
        (stellarmassbins[:-1]+stellarmassbins[1:])/2, avMetPerSM[i][:],
        label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.title('Average Metallicity of galaxies')
    plt.xlabel('$Log(M_{*}/M_{\odot})$')
    plt.ylabel('Metallicity')
    plt.legend()

plt.figure()
plt.yscale('log')
for i in range(numzbin):
    plt.scatter(
        (stellarmassbins[:-1]+stellarmassbins[1:])/2,  avSFRPerSM[i][:],
        label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.title('Average SFR of galaxies')
    plt.xlabel('$Log(M_{*}/M_{\odot})$')
    plt.ylabel('SFR [$M_{\odot}/yr$]')
    plt.legend()

plt.figure()
plt.yscale('log')
for i in range(numzbin):
    plt.scatter(
        (stellarmassbins[:-1]+stellarmassbins[1:])/2,  avsSFRPerSM[i][:],
        label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.title('Average sSFR of galaxies')
    plt.xlabel('$Log(M_{*}/M_{\odot})$')
    plt.ylabel('sSFR [$yr^{-1}$]')
    plt.legend()


plt.figure()
for i in range(numzbin):
    plt.scatter(
        (stellarmassbins[:-1]+stellarmassbins[1:])/2,  avAgePerSM[i][:]*10**-9,
        label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.title('Average Age of galaxies')
    plt.xlabel('$Log(M_{*}/M_{\odot})$')
    plt.ylabel('Age [Gyr]')
    plt.legend()


"""Seperate central galaxies and others"""

# Take the galaxies that are in a sub_halo of level 1
# central_gal return the indice of the galaxies in a halo of level 1
central_gal = []
sat_gal = []
for i in range(4):
    central_gal.append(np.where(halodata[i]['level'][gal_subhaloes[i]-1] == 1)[0])
    sat_gal.append(np.where(halodata[i]['level'][gal_subhaloes[i]-1] > 1)[0])


# Compute average things vs halo mass

massbins = np.linspace(10, 15, num=100)
avCentAgeperHaloMass = np.zeros([numzbin, np.size(massbins)-1])
avCentMetperHaloMass = np.zeros([numzbin, np.size(massbins)-1])
avCentSFRperHaloMass = np.zeros([numzbin, np.size(massbins)-1])
avCentSSFRperHaloMass = np.zeros([numzbin, np.size(massbins)-1])
avSatAgeperHaloMass = np.zeros([numzbin, np.size(massbins)-1])
avSatMetperHaloMass = np.zeros([numzbin, np.size(massbins)-1])
avSatSFRperHaloMass = np.zeros([numzbin, np.size(massbins)-1])
avSatSSFRperHaloMass = np.zeros([numzbin, np.size(massbins)-1])

for i in range(4):
    for j in range(np.size(massbins)-1):
        m1 = massbins[j]
        m2 = massbins[j+1]
        # select indices of galaxies contained in the haloes with a mass
        # between m1 and m2 :
        indices = np.where(np.logical_and(np.log10(
            halodata[i]['Mass'][gal_mainhaloes[i][:]-1]*10**11) > m1,
            np.log10(halodata[i]['Mass'][gal_mainhaloes[i][:]-1]*10**11) <= m2))
        # take indices of galaxies that are also central galaxies.
        indices_cent = np.intersect1d(indices, central_gal[i])
        indices_sat = np.intersect1d(indices, sat_gal[i])
        # avCentAgeperHaloMass[i][j] = np.average(galdata[i]['agegal2'][indices_cent])
        # avCentMetperHaloMass[i][j] = np.average(galdata[i]['met'][indices_cent])
        # avCentSFRperHaloMass[i][j] = np.average(galdata[i]['SFR2'][indices_cent])
        # avCentSSFRperHaloMass[i][j] = np.average(
        #     galdata[i]['SFR2'][indices_cent]/(galdata[i]['Mass'][indices_cent]*10**11))
        avSatAgeperHaloMass[i][j] = np.average(galdata[i]['agegal2'][indices_sat])
        avSatMetperHaloMass[i][j] = np.average(galdata[i]['met'][indices_sat])
        avSatSFRperHaloMass[i][j] = np.average(galdata[i]['SFR2'][indices_sat])
        avSatSSFRperHaloMass[i][j] = np.average(
            galdata[i]['SFR2'][indices_sat]/(galdata[i]['Mass'][indices_sat]*10**11))


"""Plot"""

plt.figure()
for i in range(4):
    plt.scatter((massbins[:-1]+massbins[1:])/2, avCentAgeperHaloMass[i][:]*10**-9,
                label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+', central', marker='d')
    plt.scatter((massbins[:-1]+massbins[1:])/2, avSatAgeperHaloMass[i][:]*10**-9,
                label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+', satellite', marker='x')
plt.yscale('log')
plt.ylabel('Average Age of galaxies [Gyr]', size=15)
plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
plt.legend()

plt.figure()
for i in range(4):
    plt.scatter((massbins[:-1]+massbins[1:])/2, avCentMetperHaloMass[i][:],
                label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+', central', marker='d')
    plt.scatter((massbins[:-1]+massbins[1:])/2, avSatMetperHaloMass[i][:],
                label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+', satellite', marker='x')
plt.ylabel('Average Metallicity of galaxies', size=15)
plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
plt.legend()

plt.figure()
plt.yscale('log')
for i in range(4):
    plt.scatter((massbins[:-1]+massbins[1:])/2, avCentSFRperHaloMass[i][:],
                label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+', central', marker='d')
    plt.scatter((massbins[:-1]+massbins[1:])/2, avSatSFRperHaloMass[i][:],
                label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+', satellite', marker='x')
plt.ylabel('Average SFR of galaxies [$M_{\odot}$/yr]', size=15)
plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
plt.legend()

plt.figure()
plt.yscale('log')
for i in range(4):
    p = plt.scatter((massbins[:-1]+massbins[1:])/2, avCentSSFRperHaloMass[i][:],
                    label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+', central', marker='d')
    plt.scatter((massbins[:-1]+massbins[1:])/2, avSatSSFRperHaloMass[i][:],
                label='z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+', satellite', marker='x',
                color=p[0].get_color())
plt.ylabel('Average sSFR of galaxies [$yr^{-1}$]', size=15)
plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
plt.ylim(10**-12, 10**-8)
plt.legend()


"""Ms versus Mh only for central galaxies"""

# Take the galaxies that are in a sub_halo of level 1
# central_gal return the indice of the galaxies in a halo of level 1
central_gal = []
sat_gal = []
for i in range(4):
    central_gal.append(np.where(halodata[i]['level'][gal_subhaloes[i]-1]==1)[0])
    sat_gal.append(np.where(halodata[i]['level'][gal_subhaloes[i]-1]>1)[0])


# Compute galaxies that are of level 1 and in a halo of level 1
centGalInCentHalo = []
for i in range(numzbin):
    centGalInCentHalo.append(np.intersect1d(central_gal[i], np.where(galdata[i]['level'] == 1)[0]))


# Compute average and median stellar mass for a given halo mass

massbins = np.linspace(10, 15, num=100)
avSMperHM = np.zeros([numzbin, np.size(massbins)-1])
medSMperHM = np.zeros([numzbin, np.size(massbins)-1])

for i in range(4):
    for j in range(np.size(massbins)-1):
        m1 = massbins[j]
        m2 = massbins[j+1]
        # select indices of galaxies contained in the haloes with a mass
        # between m1 and m2 :
        indices = np.where(np.logical_and(np.log10(
            halodata[i]['Mass'][gal_mainhaloes[i][:]-1]*10**11) > m1,
            np.log10(halodata[i]['Mass'][gal_mainhaloes[i][:]-1]*10**11) <= m2))
        indices_cent = np.intersect1d(indices, centGalInCentHalo[i])
        avSMperHM[i, j] = np.average(np.log10(galdata[i]['Mass'][indices_cent]*10**11))
        medSMperHM[i, j] = np.median(np.log10(galdata[i]['Mass'][indices_cent]*10**11))

# Compute average and median halo mass for a given stellar mass

stellarmassbins = np.linspace(8.1, 12, num=100)
avHMperSM = np.zeros([numzbin, np.size(stellarmassbins)-1])
medHMperSM = np.zeros([numzbin, np.size(stellarmassbins)-1])
for i in range(4):
    for j in range(np.size(stellarmassbins)-1):
        m1 = stellarmassbins[j]
        m2 = stellarmassbins[j+1]
        ## select indices of central galaxies with a mass
        ## between m1 and m2 :
        indices = np.where(np.logical_and(np.log10(
            galdata[i]['Mass'][centGalInCentHalo[i][:]]*10**11)>m1,
            np.log10(galdata[i]['Mass'][centGalInCentHalo[i][:]]*10**11)<=m2))
        #indices_cent = np.intersect1d(indices, centGalInCentHalo[i])
        avHMperSM[i,j] = np.average(np.log10(
        halodata[i]['Mass'][gal_mainhaloes[i][centGalInCentHalo[i][indices]]-1]*10**11))
        medHMperSM[i,j] = np.median(np.log10(
        halodata[i]['Mass'][gal_mainhaloes[i][centGalInCentHalo[i][indices]]-1]*10**11))

## Compute 2d histogram of MS vs Mh, and take the max of the distribution of Mh
## for a given Ms.

hist = []
bins=100
MSMHpeak = np.empty([numzbin,bins, 2])
for i in range(numzbin):
    hist.append(np.histogram2d(
    np.log10(halodata[i]['Mass'][gal_mainhaloes[i][centGalInCentHalo[i]]-1]*10**11),
    np.log10(galdata[i]['Mass'][centGalInCentHalo[i]]*10**11), bins=bins))
    for j in range(bins):
        indice = np.argmax(hist[i][0][j])
        MSMHpeak[i][j] = [hist[i][1][j], hist[i][2][indice]]


"""Compute the percentile of the halo mass distribution given a stellar mass,
and select only 90% of the data, in order to eliminate outliiers.
"""


stellarmassbins = np.linspace(8.1, 12, num=100)
first_per = np.zeros([numzbin, np.size(stellarmassbins)-1])
last_per = np.zeros([numzbin, np.size(stellarmassbins)-1])

for i in range(numzbin):
    for j in range(np.size(stellarmassbins)-1):
        m1 = stellarmassbins[j]
        m2 = stellarmassbins[j+1]
        ## select indices of central galaxies with a mass
        ## between m1 and m2 :
        indices = np.where(np.logical_and(
            np.log10(galdata[i]['Mass'][centGalInCentHalo[i][:]]*10**11)>m1,
            np.log10(galdata[i]['Mass'][centGalInCentHalo[i][:]]*10**11)<=m2))
        if indices[0].size : #check if the array is not empty
            first_per[i,j] = np.percentile(np.log10(
            halodata[i]['Mass'][gal_mainhaloes[i][centGalInCentHalo[i][indices]]-1]*10**11), 10)
            last_per[i,j] = np.percentile(np.log10(
            halodata[i]['Mass'][gal_mainhaloes[i][centGalInCentHalo[i][indices]]-1]*10**11), 90)
        else:
            first_per[i,j] = numpy.nan
            last_per[i,j] = numpy.nan

# Create the array containing the galaxy mass and the halo mass corresponding.
gal_hal_mass=[]
for i in range(numzbin):
    gal_hal_mass.append(np.stack(
        (np.log10(galdata[i]['Mass'][centGalInCentHalo[i]]*10**11),
        np.log10(halodata[i]['Mass'][gal_mainhaloes[i][centGalInCentHalo[i]]-1]*10**11)),
        axis = 0))
    gal_hal_mass[i][:,gal_hal_mass[i][0,:].argsort()] #sort by galaxy mass


# Select galaxies and haoes inside the percentiles

select_gal_hal_mass = []
for i in range(numzbin):
    indices = []
    for j in range(np.size(stellarmassbins)-1):
        m1 = stellarmassbins[j]
        m2 = stellarmassbins[j+1]
        indices.append(np.where(np.logical_and(
            np.logical_and(gal_hal_mass[i][0,:]>m1, gal_hal_mass[i][0,:]<m2),
            np.logical_and(gal_hal_mass[i][1,:]>first_per[i,j],
                gal_hal_mass[i][1,:]<last_per[i,j])
        ))[0])
    indices = np.hstack(indices)
    select_gal_hal_mass.append([gal_hal_mass[i][0,indices], gal_hal_mass[i][1,indices]])

## Compute average and median of Ms/Mh

MsOnMh =[]
for i in range(numzbin):
    MsOnMh.append(select_gal_hal_mass[i][0]/select_gal_hal_mass[i][1])
massbins = np.linspace(10, 15, num=100)
avMsMhperHM = np.zeros([numzbin, np.size(massbins)-1])
medMsMhperHM = np.zeros([numzbin, np.size(massbins)-1])
for i in range(4):
    for j in range(np.size(massbins)-1):
        m1 = massbins[j]
        m2 = massbins[j+1]
        ## select indices of couples contained in the haloes with a mass
        ## between m1 and m2 :
        indices = np.where(np.logical_and(
            select_gal_hal_mass[i][1]>m1,
            select_gal_hal_mass[i][1]<=m2))
        #indices = np.hstack(indices[1])
        avMsMhperHM[i,j] = np.average(MsOnMh[i][indices])
        medMsMhperHM[i,j] = np.median(MsOnMh[i][indices])

## Fit Yang relation on Ms/Mh

def mstar_over_mh_yang(x, A, m1, beta, gamma):
    """Yang et al. 2004 function, see Moster et al. 2010"""
    return 2.0*A*1/(10**((x-m1)*(-beta))+10**((x-m1)*gamma))

# n_fit = 1000
# x = np.empty([numzbin, n_fit])
# xm =np.empty([numzbin, n_fit])
# ym =np.empty([numzbin, n_fit])
yang_fit = np.empty([numzbin, 4])
yang_cov = np.empty([numzbin, 4, 4])

for i in range(numzbin):
    print(i)
    yang_fit[i], yang_cov[i] = curve_fit(mstar_over_mh_yang, select_gal_hal_mass[i][1],
    MsOnMh[i],
    p0=[0.1, 11, 0.1, 0.1],
    bounds = [[-inf, 9, 0, 0], [inf, 14, 5, 5]],
    maxfev=1000,
    method='trf'
    )
    # x[i] = np.linspace(max(Number_density[i][-1], Nhalo[i][-1]),
    #    min(Number_density[i][0], Nhalo[i][0]), n_fit )
    # xlog[i] = np.logspace(np.log10(max(Mstar[i].x[0], Mhalo[i].x[0])),
    #     np.log10(min(Mstar[i].x[-1], Mhalo[i].x[-1])), n_fit )
    # x[i] = np.geomspace(max(Number_density[i, -2], Nhalo[i, -1]),
    #     min(Number_density[i, 0], Nhalo[i, 0]), n_fit)
    # x[i][0] = max(Number_density[i, -1], Nhalo[i, -1])
    # x[i][-1] = min(Number_density[i, 0], Nhalo[i, 0])
    # xm[i] = 10**Mhalo[i](x[i])
    # ym[i] = (Mstar[i](x[i])/10**Mhalo[i](x[i]))

### Fit a polynom
deg=4
poly_fit = np.empty([numzbin, deg+1])
for i in range(numzbin):
    poly_fit[i] = np.polyfit(select_gal_hal_mass[i][1], MsOnMh[i], deg=deg)


### Compute average Sm for a given HM using only selected couples
### inside the 10/90 percentiles.


massbins = np.linspace(10, 15, num=100)
sel_avSMperHM = np.zeros([numzbin, np.size(massbins)-1])
sel_medSMperHM = np.zeros([numzbin, np.size(massbins)-1])

for i in range(4):
    for j in range(np.size(massbins)-1):
        m1 = massbins[j]
        m2 = massbins[j+1]
        ## select indices of galaxies contained in the haloes with a mass
        ## between m1 and m2 :
        indices = np.where(np.logical_and(
        select_gal_hal_mass[i][1]>m1,
        select_gal_hal_mass[i][1]<m2))
        sel_avSMperHM[i,j] = np.average(select_gal_hal_mass[i][0][indices])
        sel_medSMperHM[i,j] = np.median(select_gal_hal_mass[i][0][indices])

## Compute average and median halo mass for a given stellar mass

stellarmassbins = np.linspace(8.1, 12, num=100)
sel_avHMperSM = np.zeros([numzbin, np.size(stellarmassbins)-1])
sel_medHMperSM = np.zeros([numzbin, np.size(stellarmassbins)-1])
for i in range(4):
    for j in range(np.size(stellarmassbins)-1):
        m1 = stellarmassbins[j]
        m2 = stellarmassbins[j+1]
        ## select indices of central galaxies with a mass
        ## between m1 and m2 :
        indices = np.where(np.logical_and(
        select_gal_hal_mass[i][0]>m1,
        select_gal_hal_mass[i][0]<m2))
        #indices_cent = np.intersect1d(indices, central_gal[i])
        sel_avHMperSM[i,j] = np.average(select_gal_hal_mass[i][1][indices])
        sel_medHMperSM[i,j] = np.median(select_gal_hal_mass[i][1][indices])


## Compute the 1 sigma around the median HM for a given SM
stellarmassbins = np.linspace(8.1, 12, num=100)
low1sig = np.zeros([numzbin, np.size(stellarmassbins)-1])
top1sig = np.zeros([numzbin, np.size(stellarmassbins)-1])

for i in range(numzbin):
    for j in range(np.size(stellarmassbins)-1):
        m1 = stellarmassbins[j]
        m2 = stellarmassbins[j+1]
        indices = np.where(np.logical_and(
        gal_hal_mass[i][0]>m1,
        gal_hal_mass[i][0]<m2
        ))
        if indices[0].size :
            low1sig[i,j] = np.percentile(gal_hal_mass[i][1][indices], 16)
            top1sig[i,j] = np.percentile(gal_hal_mass[i][1][indices], 84)
        else:
            low1sig[i,j] = np.nan
            top1sig[i,j] = np.nan

"""Plots"""

for i in range(4):
    plt.figure()
    plt.hist2d(
    np.log10(halodata[i]['Mass'][gal_subhaloes[i][centGalInCentHalo[i]]-1]*10**11),
    np.log10(galdata[i]['Mass'][centGalInCentHalo[i]]*10**11),
    bins=100, cmin=1)
    plt.colorbar()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
    plt.ylabel('Log($M_{*}$) [Log($M_{\odot}$)]', size=15)
    plt.title('Central galaxies, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.scatter((massbins[:-1]+massbins[1:])/2, avSMperHM[i][:], color='red',
    label='Average SM for a given HM')
    plt.scatter( (massbins[:-1]+massbins[1:])/2, medSMperHM[i][:],
    color='green', label='Median SM for a given HM')
    plt.scatter( avHMperSM[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
    color='black', label='Average HM for a given SM')
    plt.scatter( medHMperSM[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
    color='pink', label='Median HM for a given SM')
    plt.scatter(MSMHpeak[i][:][:,0], MSMHpeak[i][:][:,1], color='blue',
    label='Max of distrib')
    plt.legend()
    plt.savefig('../Plots/HorizonAGN/CentralGalaxies/MsMhtot='+str(zbins_Cone[i])+
    '-'+str(zbins_Cone[i+1])+'.pdf')

for i in range(4):
    plt.figure()
    plt.hist2d(
    np.log10(halodata[i]['Mass'][gal_mainhaloes[i][centGalInCentHalo[i]]-1]*10**11),
    np.log10(galdata[i]['Mass'][centGalInCentHalo[i]]/
    halodata[i]['Mass'][gal_mainhaloes[i][centGalInCentHalo[i]]-1]),
    bins=100, cmin=1)
    plt.colorbar()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
    plt.ylabel('$M_{*}/M_{h}$', size=15)
    plt.title('Central galaxies, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.savefig('../Plots/HorizonAGN/CentralGalaxies/MsonMh_z='+str(zbins_Cone[i])+
    # '-'+str(zbins_Cone[i+1]))

## Plot the percentiles
for i in range(4):
    plt.figure()
    plt.hist2d(
    np.log10(halodata[i]['Mass'][gal_mainhaloes[i][centGalInCentHalo[i]]-1]*10**11),
    np.log10(galdata[i]['Mass'][centGalInCentHalo[i]]*10**11),
    bins=100, cmin=1)
    plt.colorbar()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
    plt.ylabel('Log($M_{*}$) [Log($M_{\odot}$)]', size=15)
    plt.title('Central galaxies, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.scatter( first_per[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
    color='blue', label='10th percentile for a given SM')
    plt.scatter( last_per[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
    color='red', label='90th percentile for a given SM')
    plt.legend()

## Plot only haloes inside the percentile

for i in range(numzbin):
    plt.figure()
    plt.hist2d(select_gal_hal_mass[i][1], select_gal_hal_mass[i][0], bins=100, cmin=1)
    plt.colorbar()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
    plt.ylabel('Log($M_{*}$) [Log($M_{\odot}$)]', size=15)
    plt.title('Select only HM around the median for a given SM, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    # plt.plot( medHMperSM[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
    # color='black', label='Median HM for a given SM')
    # plt.plot( first_per[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
    # color='blue', label='10th percentile for a given SM')
    # plt.plot( last_per[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
    # color='red', label='90th percentile for a given SM')
    plt.scatter((massbins[:-1]+massbins[1:])/2, sel_avSMperHM[i][:], color='red',
    label='Average SM for a given HM')
    plt.scatter( (massbins[:-1]+massbins[1:])/2, sel_medSMperHM[i][:],
    color='green', label='Median SM for a given HM')
    plt.scatter( sel_avHMperSM[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
    color='black', label='Average HM for a given SM')
    plt.scatter( sel_medHMperSM[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
    color='pink', label='Median HM for a given SM')
    plt.legend()
    plt.savefig('../Plots/HorizonAGN/CentralGalaxies/MedCut_avmed_z='+str(zbins_Cone[i])+
    '-'+str(zbins_Cone[i+1]))

## Plot Ms/Mh for selected couples
massbins = np.linspace(10, 15, num=100)
for i in range(numzbin):
    plt.figure()
    plt.hist2d(select_gal_hal_mass[i][1], select_gal_hal_mass[i][0]/select_gal_hal_mass[i][1], bins=100, cmin=1)
    plt.colorbar()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('$M_{*}/M_{h}$',  size=12)
    plt.title('Selection of 10th and 90th percentiles')
    # plt.plot(massbins, mstar_over_mh_yang(massbins, *yang_fit[i]),
    #    label='Fit of the Yang Function', c='r')
    # plt.plot(massbins, polyval(poly_fit[i], massbins), c='r')
    # plt.plot( (massbins[:-1]+massbins[1:])/2, avMsMhperHM[i][:], color='red',
    # label='Average Ms/Mh for a given Mh')
    # plt.plot( (massbins[:-1]+massbins[1:])/2, medMsMhperHM[i][:], color='blue',
    # label='Median Ms/Mh for a given Mh')
    plt.savefig('../Plots/HorizonAGN/CentralGalaxies/MsMh_Select_z='+str(zbins_Cone[i])+
    '-'+str(zbins_Cone[i+1]))


## Plot the fit of the Yang function
massbins = np.linspace(10, 15, num=100)
plt.figure()
for i in range(numzbin):
    plt.plot(massbins, mstar_over_mh_yang(massbins, *yang_fit[i]),
       label=str(zbins_Cone[i])+'<z<'+str(zbins_Cone[i+1]))

##Plot 1sigma around HM median for a given SM
for i in range(numzbin):
    plt.figure()
    plt.hist2d(gal_hal_mass[i][1], gal_hal_mass[i][0], bins=100, cmin=1)
    plt.colorbar()
    plt.xlim(10.6, 14.3)
    plt.ylim(7.5, 12.5)
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
    plt.ylabel('Log($M_{*}$) [Log($M_{\odot}$)]', size=15)
    plt.title('Select only HM around the median for a given SM, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.plot( medHMperSM[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
    'r', label='Median HM for a given SM')
    plt.plot( low1sig[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
    'r--', label='1$\sigma$ around median ')
    plt.plot( top1sig[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
    'r--')
    plt.legend()
    plt.savefig('../Plots/HorizonAGN/CentralGalaxies/MedianSigma_rescale_z='+str(zbins_Cone[i])+
    '-'+str(zbins_Cone[i+1]))


"""Fit the Behroozi 2010 relation on Mh(Ms)"""

def boo_MhMs(Ms, M1, Ms0, beta, delta, gamma):
    """Behroozi et al. 2010 Mh(Ms) relation
    All masses are in logscale"""
    return M1+beta*(Ms-Ms0)+10**(delta*(Ms-Ms0))/(1+10**(-gamma*(Ms-Ms0)))-0.5

boo_fit = np.empty([numzbin, 5])
boo_cov = np.empty([numzbin, 5, 5])
for i in range(numzbin):
    print(i)
    boo_fit[i], boo_cov[i] = curve_fit(boo_MhMs, select_gal_hal_mass[i][0],
    select_gal_hal_mass[i][1],
    bounds = [[10, 8, 0, 0, 0], [13, 11, 5, 5, 5]])

# boo_fit = np.empty([numzbin, 5])
# boo_cov = np.empty([numzbin, 5, 5])
# for i in range(numzbin):
#     print(i)
#     boo_fit[i], boo_cov[i] = curve_fit(boo_MhMs, gal_hal_mass[i][0],
#     gal_hal_mass[i][1],
#     bounds = [[10, 8, 0, 0, 0], [13, 11, 5, 5, 5]])


"""Plots"""

## Plot the fit of the Behroozi function
stellarmassbins = np.linspace(8.1, 12, num=100)

# for i in range(numzbin):
#     plt.figure()
#     plt.hist2d(select_gal_hal_mass[i][0], select_gal_hal_mass[i][1], bins=100,
#     cmin=1)
#     plt.plot(stellarmassbins, boo_MhMs(stellarmassbins, *boo_fit[i]),
#        label=str('Behroozi function fit'), c='r')

for i in range(numzbin):
    plt.figure()
    plt.hist2d(select_gal_hal_mass[i][1], select_gal_hal_mass[i][0], bins=100,
    cmin=1)
    # plt.scatter( avHMperSM[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
    # color='black', label='Average HM for a given SM')
    # plt.scatter( medHMperSM[i][:], (stellarmassbins[:-1]+stellarmassbins[1:])/2,
    # color='pink', label='Median HM for a given SM')
    plt.plot(boo_MhMs(stellarmassbins, *boo_fit[i]), stellarmassbins,
    label=str('Behroozi function fit'), c='r')
    plt.title('Fit on Ms vs Mh HorizonAGN')
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Log($M_{*}$) [Log($M_{\odot}$)]', size=12)
    plt.savefig('../Plots/HorizonAGN/CentralGalaxies/MsMh_Bfit_z='+str(zbins_Cone[i])+
    '-'+str(zbins_Cone[i+1]))


for i in range(numzbin):
    plt.figure()
    plt.hist2d(select_gal_hal_mass[i][1],
    select_gal_hal_mass[i][0]/select_gal_hal_mass[i][1], bins=100,
    cmin=1)
    plt.plot(boo_MhMs(stellarmassbins, *boo_fit[i]),
    stellarmassbins/boo_MhMs(stellarmassbins, *boo_fit[i]),
    label=str('Behroozi function fit'), c='r')
    plt.title('Ms/Mh with Behroozi fit, HorizonAGN')
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('$M_{*}/M_{h}$', size=12)
    plt.savefig('../Plots/HorizonAGN/CentralGalaxies/MsOnMh_Bfit_z='+str(zbins_Cone[i])+
    '-'+str(zbins_Cone[i+1]))


""" Compute M*/Mh relation and dispersion using the hal_centralgal_new catalog
(using directly ids of galaxies at the center of every halo)
"""


"""Plots"""

for i in range(4):
    plt.figure()
    indices = np.where(np.logical_and(hal_centgal[i]>0, halodata[i]['level']==1 ))
    # verification that all galaxies selected are central
    # print(galdata[i]['level'][hal_centgal[i][indices]-1].min())
    plt.hist2d(
        np.log10(halodata[i]['Mass'][indices]*10**11),
        np.log10(galdata[i]['Mass'][hal_centgal[i][indices]-1]*10**11),
        bins=100, cmin=1)
    plt.colorbar()
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=12)
    plt.ylabel('Log($M_{*}$) [Log($M_{\odot}$)]', size=12)
    plt.title('HorizonAGN, hal_centralgal_new, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))


"""Number of galaxies per halo"""

# Number of halo with minimal mass
# minimum = min(10**11*halodata['Mass'])
# indices = [i for i, v in enumerate(10**11*halodata['Mass']) if v == minimum]
# np.size(indices)


nbgalaxiesperhalos = []
for i in range(4):
    # index j of nbgalaxiesperhalos gives the number of galaxies in the halo of
    # ID = j+1
    nbgalaxiesperhalos.append(np.zeros(np.size(halodata[i]['Mass'])))
    for j in gal_subhaloes[i].astype(int):
        nbgalaxiesperhalos[i][j-1]+=1

## I dont know if I have to use mainhalos or subhalos, but it seems that main
## halos give better results for the number of galaxy per halos

nbgalaxiesperhalos_main = []
for i in range(4):
    # index j of nbgalaxiesperhalos gives the number of galaxies in the halo of
    # ID = j+1
    nbgalaxiesperhalos_main.append(np.zeros(np.size(halodata[i]['Mass'])))
    for j in gal_mainhaloes[i].astype(int):
        nbgalaxiesperhalos_main[i][j-1]+=1

"""Plot"""
# for i in range(4):
#     plt.hist(nbgalaxiesperhalos[i], bins=range(nbgalaxiesperhalos[i].max().astype(int)))
#     plt.yscale('log')
#     plt.show()

"""Number galaxy per halo versus Halo Mass"""

### Compute Average mass of halos for a given number of galaxies in the halo

averageHaloMassPerNgal = []
for i in range(4):
    averageHaloMassPerNgal.append(np.empty(nbgalaxiesperhalos_main[i].astype(int).max()+1))
    for j in range(nbgalaxiesperhalos_main[i].astype(int).max()+1):
        averageHaloMassPerNgal[i][j] = np.mean(halodata[i]['Mass'][nbgalaxiesperhalos_main[i]==j])

### Compute average number of galaxies in halos given a halo mass interval

massbins = np.linspace(10, 15, num=100)
averageNgalperHaloMass = np.zeros([numzbin, np.size(massbins)-1])

for i in range(numzbin):
    for j in range(np.size(massbins)-1):
        m1 = massbins[j]
        m2 = massbins[j+1]
        averageNgalperHaloMass[i][j] = np.average(nbgalaxiesperhalos_main[i][np.logical_and(
        np.log10(halodata[i]['Mass']*10**11)>m1,
        np.log10(halodata[i]['Mass']*10**11)<m2)])

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

for i in range(4):
    plt.scatter((massbins[:-1]+massbins[1:])/2, averageNgalperHaloMass[i][:],
        label = 'z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) )
plt.yscale('log')
plt.ylabel('Average number of galaxies per halo', size=15)
plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)


"""Number of galaxies per halo vs halo mass, for central and satellite gaalxies"""

## Normalement il ne devrait y avoir qu'une seule galaxie centrale par halo.

# Compute number of central and satellites gaalxies per halo

nbCentgalaxiesperhalos_main = []
nbSatgalaxiesperhalos_main = []
for i in range(4):
    # index j of nbgalaxiesperhalos gives the number of galaxies in the halo of
    # ID = j+1
    nbCentgalaxiesperhalos_main.append(np.zeros(np.size(halodata[i]['Mass'])))
    nbSatgalaxiesperhalos_main.append(np.zeros(np.size(halodata[i]['Mass'])))
    for j in np.intersect1d(gal_mainhaloes[i].astype(int), central_gal[i]):
        nbCentgalaxiesperhalos_main[i][j-1]+=1
    for j in np.intersect1d(gal_mainhaloes[i].astype(int), sat_gal[i]):
        nbSatgalaxiesperhalos_main[i][j-1]+=1


massbins = np.linspace(10, 15, num=100)
avCentNgalperHaloMass = np.zeros([numzbin, np.size(massbins)-1])
avSatNgalperHaloMass = np.zeros([numzbin, np.size(massbins)-1])

for i in range(numzbin):
    for j in range(np.size(massbins)-1):
        m1 = massbins[j]
        m2 = massbins[j+1]
        avCentNgalperHaloMass[i][j] = np.average(nbCentgalaxiesperhalos_main[i][np.logical_and(
        np.log10(halodata[i]['Mass']*10**11)>m1,
        np.log10(halodata[i]['Mass']*10**11)<m2)])
        avSatNgalperHaloMass[i][j] = np.average(nbSatgalaxiesperhalos_main[i][np.logical_and(
        np.log10(halodata[i]['Mass']*10**11)>m1,
        np.log10(halodata[i]['Mass']*10**11)<m2)])

"""Plot"""

cmap = ['blue', 'red', 'green', 'orange']
for i in range(4):
    plt.scatter((massbins[:-1]+massbins[1:])/2, avCentNgalperHaloMass[i][:],
        label = 'z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+', central', marker='d', c=cmap[i])
    plt.scatter((massbins[:-1]+massbins[1:])/2, avSatNgalperHaloMass[i][:],
        label = 'z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1])+', satellite', marker='+', c=cmap[i] )
plt.yscale('log')
plt.legend()
plt.ylabel('Average number of galaxies per halo', size=15)
plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)


"""Mass of galaxies vs haloes mass for haloes in nodes, filaments and void"""

nodes_gal =[]
fil_gal=[]
void_gal=[]

## Select indices of galaxies that are in haloes in nodes, filaments and voids

for i in range(3):
    nodes_gal.append(np.where(haloes_env[i][gal_mainhaloes[i][:]-1, 2]<5)[0])
    fil_gal.append(np.where(np.logical_and(haloes_env[i][gal_mainhaloes[i][:]-1, 2]>5,
    haloes_env[i][gal_mainhaloes[i][:]-1, 1]<2))[0])
    void_gal.append(np.where(np.logical_and(haloes_env[i][gal_mainhaloes[i][:]-1, 2]>5,
     haloes_env[i][gal_mainhaloes[i][:]-1, 1]>2))[0])


"""Plots"""

for i in range(3):
    plt.figure()
    plt.hist2d(np.log10(halodata[i]['Mass'][gal_mainhaloes[i][nodes_gal[i]]-1]*10**11),
    np.log10(galdata[i]['Mass'][nodes_gal[i]]*10**11), bins=100, cmin=1)
    plt.title('z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]')
    plt.ylabel('Log($M_{*}$) [Log($M_{\odot}$)]')
    plt.colorbar()
    plt.savefig('../Plots/HorizonAGN/Env/dnod<5Mpc/MsMh_z='+str(zbins_Cone[i])+
    '-'+str(zbins_Cone[i+1]))
plt.show()

for i in range(3):
    plt.figure()
    plt.hist2d(np.log10(halodata[i]['Mass'][gal_mainhaloes[i][fil_gal[i]]-1]*10**11),
    np.log10(galdata[i]['Mass'][fil_gal[i]]*10**11), bins=100, cmin=1)
    plt.title('z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]')
    plt.ylabel('Log($M_{*}$) [Log($M_{\odot}$)]')
    plt.colorbar()
    plt.savefig('../Plots/HorizonAGN/Env/dfil<2/MsMh_z='+str(zbins_Cone[i])+
    '-'+str(zbins_Cone[i+1]))
plt.show()

for i in range(3):
    plt.figure()
    plt.hist2d(np.log10(halodata[i]['Mass'][gal_mainhaloes[i][void_gal[i]]-1]*10**11),
    np.log10(galdata[i]['Mass'][void_gal[i]]*10**11), bins=100, cmin=1)
    plt.title('z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]')
    plt.ylabel('Log($M_{*}$) [Log($M_{\odot}$)]')
    plt.colorbar()
    plt.savefig('../Plots/HorizonAGN/Env/void/MsMh_z='+str(zbins_Cone[i])+
    '-'+str(zbins_Cone[i+1]))
plt.show()


"""Number of galaxies per halo vs Halo mass, for nodes, filaments and voids"""

### Make a selection for the mass of gaalxies, to be sure to limit the systematic
### effect because galaxies and haloes in voids are less massive than galaxies
### in nodes and filaments.


massbins = np.linspace(10, 15, num=100)
averageNgalNodes = np.zeros([numzbin, np.size(massbins)-1])
averageNgalFilaments = np.zeros([numzbin, np.size(massbins)-1])
averageNgalVoid = np.zeros([numzbin, np.size(massbins)-1])


for i in range(3):
    for j in range(np.size(massbins)-1):
        m1 = massbins[j]
        m2 = massbins[j+1]
        averageNgalNodes[i][j] = np.average(nbgalaxiesperhalos_main[i][
        np.logical_and(np.logical_and(
        np.log10(halodata[i]['Mass']*10**11)>m1,
        np.log10(halodata[i]['Mass']*10**11)<m2),
        haloes_env[i][:, 2]<5)])
        averageNgalFilaments[i][j] = np.average(nbgalaxiesperhalos_main[i][
        np.logical_and(np.logical_and(
        np.log10(halodata[i]['Mass']*10**11)>m1,
        np.log10(halodata[i]['Mass']*10**11)<m2),
        haloes_env[i][:, 1]<2)])
        averageNgalVoid[i][j] = np.average(nbgalaxiesperhalos_main[i][
        np.logical_and(np.logical_and(
        np.log10(halodata[i]['Mass']*10**11)>m1,
        np.log10(halodata[i]['Mass']*10**11)<m2),
        np.logical_and(haloes_env[i][:, 2]>5, haloes_env[i][:, 1]>2))])

"""Plot"""

for i in range(3):
    plt.figure()
    plt.scatter((massbins[:-1]+massbins[1:])/2, averageNgalNodes[i][:],
    label='Nodes', marker='d', color='red')
    plt.scatter((massbins[:-1]+massbins[1:])/2, averageNgalFilaments[i][:],
    label='Filaments', marker='+', color='orange')
    plt.scatter((massbins[:-1]+massbins[1:])/2, averageNgalVoid[i][:],
    label='Void', marker='x', color='blue')
    plt.yscale('log')
    plt.ylabel('Average number of galaxies per halo', size=15)
    plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]', size=15)
    plt.title('z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))
    plt.legend()




"""Number of Galaxies per halo versus Average Age/Metallicity/SFR of Galaxies"""

# averageSFRperNumGal = []
# averageMetperNumGal = []
# averageAgeperNumGal = []
#
# for i in range(4):
#     size = (nbgalaxiesperhalos_main[i]).astype('int').max()
#     averageSFRperNumGal.append(np.zeros(size))
#     averageMetperNumGal.append(np.zeros(size))
#     averageAgeperNumGal.append(np.zeros(size))
#     print(i)
#     for j in range(size):
#          # do not take into account empty haloes
#         #print(j/size)
#         id_haloes = np.where(nbgalaxiesperhalos_main[i]==j)
#         #print(id_haloes)
#         s = set(id_haloes[0])
#         id_gal =  np.where([(a in s) for a in gal_mainhaloes[i]])
#
#         averageSFRperNumGal[i][j] = np.average(galdata[i]['SFR'][id_gal[0]])
#         averageMetperNumGal[i][j] = np.average(galdata[i]['met'][id_gal[0]])
#         averageAgeperNumGal[i][j] = np.average(galdata[i]['agegal2'][id_gal[0]])


"""Plot"""

# plt.figure()
# for i in range(4):
#     plt.scatter(np.arange(np.size(averageMetperNumGal[i])), averageMetperNumGal[i])
# plt.xlabel('Number of galaxies in the halo')
# plt.ylabel('Average Metallicity')
#
# plt.figure()
# for i in range(4):
#     plt.scatter(np.arange(np.size(averageAgeperNumGal[i])), averageAgeperNumGal[i]*10**-9)
# plt.xlabel('Number of galaxies in the halo')
# plt.ylabel('Average Age of galaxies in the halo [Gyr]')
#
# plt.figure()
# for i in range(4):
#     plt.scatter(np.arange(np.size(averageSFRperNumGal[i])), averageSFRperNumGal[i])
# plt.xlabel('Number of galaxies in the halo')
# plt.ylabel('Average SFR of gal in the halo [$M_{\odot}$/yr]')


"""Average number of galaxies in halos as a function average SFR in the halo"""

# IDEA: Could show the average number of galxies as a function average SFR in the halo


"""Put a mass threshold on the galaxies count into main haloes"""

log_mt_array = np.array([1., 8., 9., 10., 11., 12.]) ## Mass thresholds 10**log_mt Msun

start = time.time()
nbGalPerMainHalo_Tresh = [[] for x in range(np.size(log_mt_array))]
for i in range(4):
    for k in range(np.size(log_mt_array)):
        log_mt = log_mt_array[k]
        nbGalPerMainHalo_Tresh[k].append(np.zeros(np.size(halodata[i]['Mass'])))
        #for k in range(np.size(gal_mainhaloes[i].astype(int))):
        array_thresh = galdata[i]['Mass'][:] > 10**(log_mt-11)
        for j in gal_mainhaloes[i][array_thresh].astype(int):
            nbGalPerMainHalo_Tresh[k][i][j-1]+=1
    print(time.time()-start)

## Compute average nb of galaxies per mass bins

massbins = np.linspace(10, 15, num=100)
averageNgalperHaloMass_Tresh = np.zeros([np.size(log_mt_array), numzbin, np.size(massbins)-1])

for i in range(numzbin):
    for k in range(np.size(log_mt_array)):
        start = time.time()
        for j in range(np.size(massbins)-1):
            m1 = massbins[j]
            m2 = massbins[j+1]
            averageNgalperHaloMass_Tresh[k][i][j] = np.average(nbGalPerMainHalo_Tresh[k][i][np.logical_and(
            np.log10(halodata[i]['Mass']*10**11)>m1,
            np.log10(halodata[i]['Mass']*10**11)<m2)])
        print(time.time()-start)


""" Plot """

for i in range(4):
    fig = plt.figure()
    for k in range(np.size(log_mt_array)):
        plt.scatter((massbins[:-1]+massbins[1:])/2, averageNgalperHaloMass_Tresh[k][i][:],
            label = 'Log($M_{star}^{thresh}$) : '+str(log_mt_array[k]) )
    plt.ylabel('Average number of galaxies per halo', size=15)
    plt.xlabel('Log(Halo Mass), [Log($M_{\odot}$)]', size=15)
    plt.yscale('log')
    plt.legend()
    plt.title('Resdshift bin : '+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]))



"""Haloes Environment and properties"""


## Plot Average things in haloes in nodes, filmaments and voids:


massbins = np.linspace(10, 15, num=100)
averageAgeperHaloMass = np.zeros([numzbin, np.size(massbins)-1])
averageMetperHaloMass = np.zeros([numzbin, np.size(massbins)-1])
averageSFRperHaloMass = np.zeros([numzbin, np.size(massbins)-1])
averageCentralSFRperHaloMass = np.zeros([numzbin, np.size(massbins)-1])

for i in range(3):
    for j in range(np.size(massbins)-1):
        print(j)
        m1 = massbins[j]
        m2 = massbins[j+1]
        ## select indices of galaxies contained in the haloes in nodes and
        ## with a mass between m1 and m2 :
        # indices = np.where(np.logical_and( np.logical_and( np.log10(
        #     halodata[i]['Mass'][gal_mainhaloes[i][:]-1]*10**11)>m1,
        #     np.log10(halodata[i]['Mass'][gal_mainhaloes[i][:]-1]*10**11)<=m2),
        #     haloes_env[i][gal_mainhaloes[i][:]-1, 1]<2))
        indices = np.where(np.logical_and( np.logical_and( np.log10(
            halodata[i]['Mass'][gal_mainhaloes[i][:]-1]*10**11)>m1,
            np.log10(halodata[i]['Mass'][gal_mainhaloes[i][:]-1]*10**11)<=m2),
            np.logical_and(haloes_env[i][gal_mainhaloes[i][:]-1, 1]>2,
            haloes_env[i][gal_mainhaloes[i][:]-1, 2]<5)))

        averageAgeperHaloMass[i][j] = np.average(galdata[i]['agegal2'][indices])
        averageMetperHaloMass[i][j] = np.average(galdata[i]['met'][indices])
        averageSFRperHaloMass[i][j] = np.average(galdata[i]['SFR2'][indices])


"""Fit a poxer law to the Ngalperhalo(Mhalo) relation"""

# def powlaw(x, x0, n, C):
#     return x0*(x**n)+C
#
# popt = np.empty([4, 3])
# numgal = []
# for i in range(4):
#     numgal.append(np.arange(nbgalaxiesperhalos_main[i].astype(int).max()+1))
#     popt[i], _ = curve_fit(powlaw,
#     np.log10(averageHaloMassPerNgal[i][averageHaloMassPerNgal[i]>0]*10**11),
#     numgal[i][averageHaloMassPerNgal[i]>0], maxfev=10**6, xtol=10**-10)
#

"""Plot"""

# halomass = np.array( [np.log10(np.logspace(10.5, 14.6)),
#     np.log10(np.logspace(10.5, 14.3)),  np.log10(np.logspace(10.5, 13.6)),
#     np.log10(np.logspace(10.5, 13))])
# for i in range(4):
#     fig = plt.figure()
#     plt.scatter(np.log10(averageHaloMassPerNgal[i]*10**11), numgal[i])

# for i in range(4):
#     fig = plt.figure()
#     plt.scatter(np.log10(halodata[i]['Mass']*10**11), nbgalaxiesperhalos_main[i],
#         marker='.')
#     plt.scatter(np.log10(averageHaloMassPerNgal[i]*10**11),
#     range(nbgalaxiesperhalos_main[i].astype(int).max()+1), label='Average Mass')
#     plt.title('HorizonAGN, z='+str(zbins_Cone[i])+'-'+str(zbins_Cone[i+1]) +
#         ', match gal-Mainhalo')
#     plt.plot(halomass[i], powlaw(halomass[i], *popt[i]), label='Power law fit',
#     color='red')
#     plt.xlabel('Log($M_{h}$) [Log($M_{\odot}$)]')
#     plt.ylabel('Number of galaxies in the halo')
#     plt.legend()
# plt.show()


"""MCMC to fit Yang function"""



from pymc3 import Model, Normal, Uniform, HalfNormal, finc_MAP

yang_model = Model()

with yang_model:

    # Priors for unknown model parameters
    norm = Uniform('norm', lower=0, upper=10)
    logmstar = Normal('logmstar', mu=12, sd=2)
    beta = Uniform('beta', lower=0, upper=5)
    gamma = Uniform('gamma', lower=0, upper=5)
    sigma = HalfNormal('sigma', sd=2)

    # Expected value of outcome
    mu = np.log10(2*norm*( (xm/10**logmstar)**(-beta) + (xm/10**logmstar)**(gamma) )**(-1))

    # Likelihood (sampling distribution) of observations
    Y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=np.log10(ym))


start = {'norm_interval_':0.1, 'logmstar':12, 'beta_interval_':0.1,
    'gamma_interval_':0.1, 'sigma_log_':0.5 }
map_estimate = find_MAP(start=start, model=yang_model,
    fmin=optimize.fmin_powell)
print(map_estimate)

beta_est = 5*exp(map_estimate['beta_interval_']) /(1+exp(map_estimate['beta_interval_']))
gamma_est = 5*exp(map_estimate['gamma_interval_']) /(1+exp(map_estimate['gamma_interval_']))
norm_est = 10*exp(map_estimate['norm_interval_']) /(1+exp(map_estimate['norm_interval_']))



x = np.logspace(10, 14, num=1000)
plt.plot(np.log10(x), np.log10(mstar_over_mh_yang(x, norm_est,
    10**map_estimate['logmstar'], beta_est, gamma_est)))



""" Compute average of Ms/Mh on the histogram for a hgiven halo mass"""


hist = hist2d(np.log10(subHaloMass[0]*10**11),
    np.log10(galdata[0]['Mass']/subHaloMass[0]), bins=100, cmin=100)
moy = np.empty(np.size(hist[1])-1)
for i in range(size(hist[1])-1):
    indices = ~np.isnan(hist[0][i])
    if np.size(hist[0][i][indices])==0:
        moy[i] = np.nan
    else:
        moy[i] = np.average(((hist[2][1:]+hist[2][:-1])/2)[indices], weights=hist[0][i][indices])

"""Plot"""

plt.plot(((hist[1][1:]+hist[1][:-1])/2), moy, c='red', label='average')
plt.legend()
plt.title('Horizon zbin 0')
plt.xaxis('Halo Mass')
plt.xlabel('Halo Mass')
plt.ylabel('Mstar / Mhalo')
