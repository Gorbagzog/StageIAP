#!/usr/bin/env python3
# -*-coding:Utf-8 -*

import pyfits
import numpy as np
import matplotlib.pyplot as plt

# Open catalog

# cat = input('Catalog to load ? Type C for COSMOS2015_Laigle' \
#                 'D17 for IariD17 and L16 for Iari L16 :  ')
#
# if cat == 'C' :
#     catalog = '../Data/COSMOS2015_Laigle+_v1.1.fits'
#     photoz = 'PHOTOZ'
# elif cat == 'D17' :
#     catalog = '../Data/cosmos2015_D17_v2.0_zmin-zmax.fits'
#     photoz = 'ZPHOT'
# elif cat == 'L16' :
#     catalog = '../Data/cosmos2015_L16_v2.0_zmin-zmax.fits'
#     photoz = 'ZPHOT'
# else :
#     print('Wrong input')

print('This script will download the COSMOS2015 Laigle catalog.')
hdulist = pyfits.open('../Data/COSMOS2015_Laigle+_v1.1.fits')
tdata = hdulist[1].data
hdulist.close()

# Select galaxies with relevant data
tdata = tdata[tdata['FLAG_HJMCC']==0]
tdata = tdata[tdata['FLAG_COSMOS']==1]
tdata = tdata[tdata['FLAG_PETER']==0]
tdata = tdata[tdata['PHOTOZ']>0]
tdata = tdata[tdata['PHOTOZ']<7]
#tdata = tdata[tdata['SFR_MED']>-99]
tdata = tdata[tdata['MASS_MED']>7]


print('Catalog successfully loaded.')



#Plot SFR vs PhotoZ
# plt.hist2d(tdata['PHOTOZ'], tdata['SFR_MED'], bins=100, cmin=1)
# plt.xlabel('PhotoZ')
# plt.ylabel('SFR_MED (log)')
# plt.title('COSMOS2015')
# plt.colorbar()
#
# #Plot SFR vs Mass
# plt.hist2d(tdata['MASS_MED'], tdata['SFR_MED'], bins=100, cmin=1)
# plt.ylabel('SFR_MED (log)')
# plt.xlabel('Stellar mass (MASS_MED) (log)')
# plt.title('COSMOS2015')
# plt.colorbar()
