#!/usr/bin/env python3
# -*-coding:Utf-8 -*

import pyfits
import numpy as np
import matplotlib.pyplot as plt

hdulist = pyfits.open('../Data/COSMOS2015_clustering_v2.0_clean.fits')
tdata = hdulist[1].data
hdulist.close()

tdata = tdata[tdata['photo_z']<99]

tdata = tdata[tdata['clean']>0]

tdata = tdata[tdata['mstar_cosmo']>0]

# tdata = tdata[tdata['mstar_cosmo']>7.2]

mmingal = 7
mmaxgal = 12.5

plt.hist2d(tdata['photo_z'], tdata['mstar_cosmo'], bins =100, cmin =1)
plt.xlabel('Z_phot')
plt.ylabel('$Log(M/M_{\odot})$')
plt.title('Stellar mass vs PhotoZ, Jean\'s catalog')
plt.colorbar()
