#!/usr/bin/env python3
# -*-coding:Utf-8 -*

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""H-AGN LightCone photometric Catalog.

Load catalog and make a match with the true lightcone catalog.
"""

header = ['Id', 'Ra', 'Dec', 'zphot', 'zphot_err', 'Mass', 'Mass_err', 'mag_u', 'magerr_u',
          'mag_B', 'magerr_B', 'mag_V', 'magerr_V', 'mag_r', 'magerr_r', 'mag_i', 'magerr_i',
          'mag_z', 'magerr_z', 'mag_Y', 'magerr_Y', 'mag_J', 'magerr_J', 'mag_H', 'magerr_H',
          'mag_K', 'magerr_K', 'SFR']


HPhoto_tmp = np.loadtxt(
    '../Data/HorizonAGNLightconePhotometric/Salp_0.0-3.0_dust_v15c.in_Init_Small')

HPhoto = pd.DataFrame(HPhoto_tmp, columns=header)
