#!/usr/bin/env python3
# -*-coding:Utf-8 -*

import sys
sys.path.append("..")
sys.stdout.flush()

import matplotlib
matplotlib.use('Agg')
from MCMC_SHMR_main_clean import *


paramfile = 'MCMC_param.ini'
# minboundfile = 'MCMC_minbound_Cosmos.txt'
# maxboundfile = 'MCMC_maxbound_Cosmos.txt'

runMCMC_allZ(paramfile)


