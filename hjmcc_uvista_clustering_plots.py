#!/opt/local/bin/python
import argparse
import sys
import os 
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
from matplotlib import ticker
import matplotlib.cm as cm 
import re 
import itertools
import lmfit
import multiprocessing, subprocess
from lmfit import minimize, Parameters
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d 
from  scipy.integrate import quad
import scipy.constants as const
from matplotlib import rc
from matplotlib.ticker import FixedLocator
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.interpolate import interp1d 
from itertools import islice
from collections import deque
from bisect import insort
from bisect import bisect_left
from lmfit import model
import running_median
from astropy.table import Table, Column
rc('text', usetex=True)

class MFpar:
    def __init__(self,mlim_min,Mstar,phi1,alpha1,phi2,alpha2):
        self.mlim_min=mlim_min
        self.Mstar=Mstar
        self.phi1=phi1
        self.phi2=phi2
        self.alpha1=alpha1
        self.alpha2=alpha2


class MeasurementBin:
    def __init__(self,wt):
        pmc_mean = os.path.dirname(wt.name)+'/'+os.path.basename(wt.name).replace('wpmc_','FitVista') + '_mean'
        try:
            self.th,self.wth,self.wth_err = np.loadtxt(wt,usecols=(0,1,2),unpack=True)
            Mlim,z1,z2 = os.path.basename(wt.name).split('wpmc_')[1].split('_')
            self.Mlim=float(Mlim)
            self.z=(float(z1)+float(z2))/2.0
            self.z1=float(z1)
            self.z2=float(z2)
            self.params = lmfit.Parameters()
            self.params_shmr = lmfit.Parameters()
            # default values
            self.params.add('A',   value= 1e-2)
            self.params.add('gamma', value= 1.8, vary=args.varygamma)
        except IOError:
            print 'Cannot open halo model file'
            sys.exit(1)
#        try:
#            samplefile = wt.name.replace('all/wpmc_',
#                                         '/Users/hjmcc/cosmos/uvista/melody/PMC_utilsAll/All_')+'.sample.gz'
#            print 'reading ', samplefile
#            self.pmcsamples=np.loadtxt(samplefile,usecols=(2,3,4,5,6),unpack=False)
#        except:
#            print 'Cannot open samplefile',samplefile
        try:
            data_file      = os.path.dirname(wt.name)+'/'+ \
                             os.path.basename(wt.name).replace('wpmc_','Vista')
            z_in, M_min = np.loadtxt(data_file,usecols=(2,3),unpack=True, comments='#')
            self.z_med = np.median(z_in)
            self.M_med = np.median(M_min)
            self.M_mean = np.mean(M_min)
            self.Ngal = np.size(M_min)
        except:
            print 'Cannot open data file:', data_file

            # also load halo model stuff.
        try:
            dndz_file     = os.path.dirname(wt.name)+'/'+ \
                        os.path.basename(wt.name).replace('wpmc_','nz_Vista')
            (self.dz,self.dn) = np.loadtxt(dndz_file,usecols=(0,1),unpack=True, comments='#')
        except:
                print "Cannot open dndz_file", dndz_file
        try:
            haloplot_file = os.path.dirname(wt.name)+'/'+ \
                            os.path.basename(wt.name).replace('wpmc_','haloplotVista')+'.out'
            (self.th2,self.wth_hp,self.M,self.nm,self.nc,self.nsat)= \
                np.loadtxt(haloplot_file,usecols=(0,1,6,7,8,9),unpack=True, comments='#')
            with open(haloplot_file) as inputfile:
                for line in inputfile:
                    if line.startswith('#'):
                        if 'M_min' in line:
                            self.Mmin=float(re.split(r'#|\s+', line)[3])
                        if 'M_1' in line:
                            self.M1=float(re.split(r'#|\s+', line)[3])
                        if 'M_0' in line:
                            self.M0=float(re.split(r'#|\s+', line)[3])
                        if 'alpha' in line:
                            self.alpha=float(re.split(r'#|\s+', line)[3])
                        if 'weighted galaxy density n_g' in line:
                            self.n_g=float(re.split(r'#|\s+', line)[6])
                        if 'average galaxy bias' in line:
                            self.bg=float(re.split(r'#|\s+', line)[6])
        except IOError:
            print "Cannot open halo file", haloplot_file
        try:
            inputfile=open(pmc_mean,'r')
            for line in inputfile:
                if line.startswith('#'):
                    continue
                quantity = (re.split(r'\s+', line)[2])
                setattr(self,quantity,float(re.split(r'\s+', line)[3]))
                setattr(self,(quantity+'_perr'),float(re.split(r'\s+', line)[4]))
                setattr(self,(quantity+'_nerr'),float(re.split(r'\s+', line)[5]))
        except IOError: 
            print 'Cannot open file', pmc_mean

def WriteLatexTable(mes):
    slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
    z1 = np.array([ 0.5, 0.8, 1.1 , 1.5,  2.0])
    z2 = np.array([ 0.8,1.1,  1.5 , 2.0,  2.5])
    Mlim= ms(mes,'0.65','Mlim')
    Mlims = Mlim[np.argsort(Mlim)]
    tl = Table(masked=True)
    tl['Mlim']=Column(Mlims)
    for zl in slices:
        arr_Mlim = np.zeros(np.size(Mlims))
        arr_Nmed = np.zeros(np.size(Mlims))
        arr_Ngal = np.zeros(np.size(Mlims))
        Mlim= ms(mes,zl,'Mlim')
        ag=np.argsort(Mlim)
        Ngal = ms(mes,zl,'Ngal')[ag]
        Mmed = ms(mes,zl,'M_med')[ag]
        fmtz = r'$%5.2f < z < %5.2f$' % (z1[0],z2[0])
        indexes=np.searchsorted(Mlims,Mlim[ag])
        for n,i in enumerate(indexes):
            arr_Mlim[i]=Mlim[ag][n]
            arr_Ngal[i]=Ngal[n]
            arr_Nmed[i]=Mmed[n]
        tl['N_gal'+str(zl)] = MaskedColumn(arr_Ngal,mask=arr_Ngal==0,format='%5d')
        tl['M_med'+str(zl)] = MaskedColumn(arr_Nmed,mask=arr_Ngal==0,format='%5.2f')
    format_table=ascii.write(tl,Writer=ascii.Latex,latexdict=ascii.latex.latexdicts['AA'])
    return(format_table)

def residual(params,th,wth,wth_err):
    A=params['A'].value
    gamma = params['gamma'].value
    model = A*(th**(1.0-gamma)-args.intconst)
    return (model - wth)/wth_err

def wtheta_model(params,th):
    A=params['A'].value
    gamma = params['gamma'].value
    model = A*(th**(1.0-gamma)-args.intconst)
    return model

# central luminosity divided by halo mass
def mstar_over_mh(params_shmr,mmin):
    mt=params_shmr['Mt'].value
    am=params_shmr['am'].value
    A=params_shmr['A'].value
    mstar=np.array([1.0e10])
    ans = A*(mstar/mt)*(mmin/mt)**(am-1.0)*np.exp(-1.0*mt/mmin+1.0)
    return ans

def mstar_over_mh_yang(params_shmr_yang,mmin):
    m1=params_shmr_yang['M1'].value
    gamma=params_shmr_yang['gamma'].value
    beta=params_shmr_yang['beta'].value
    A=params_shmr_yang['A'].value
    ans = 2.0 * A * ( (mmin/m1)**-beta+(mmin/m1)**gamma)**-1.0 
    return ans

def resid_mstar_over_mh(params_shmr,M_med,Mmin,err):
    resid = (mstar_over_mh(params_shmr,Mmin) - M_med/Mmin)
    return resid


def resid_mstar_over_mh_yang(params_shmr_yang,M_med,Mmin,err):
    resid = (mstar_over_mh_yang(params_shmr_yang,Mmin) - M_med/Mmin)/err
#   print resid
    return resid

# given halo mass, predict median stellar mass
def mstar_mcen(params_mcen,mmin):
    mt=params_mcen['Mt'].value
    am=params_mcen['am'].value
    A=params_mcen['A'].value
    M_med=A*(mmin/mt)**am*np.exp(-1.0*mt/mmin+1)*1.0e10
    return M_med 

def resid_mstar_mcen(params_mcen,M_med,Mmin,err):
    resid = (mstar_mcen(params_mcen,Mmin) - M_med)/err
    return resid

# for all the measurements, compute the r0 values 

def gz(z):
    return cosmo.H0.value/(const.c/1e3)* \
                         np.sqrt((cosmo.Om0*(1.0+z)**3+(1.0-cosmo.Om0)))

def limber_func(z,mes,slice):
    gamma = mes[slice].params['gamma'].value
    if z>mes[slice].z1 and  z<mes[slice].z2 :
        cd = cosmo.comoving_distance(z).value
        ans = cd**(1.0-gamma)*gz(z)*dndz(z,mes,slice)*dndz(z,mes,slice)
    else:
        ans = 0 
    return ans 

def dndz(z,mes,slice):
    dndz_func = interp1d(mes[slice].dz,mes[slice].dn,'linear')
    return(dndz_func(z))

def fit_r0(mes):
    for slice in mes:
        A = mes[slice].params['A'].value
        Aerr = mes[slice].params['A'].stderr
        gamma  = mes[slice].params['gamma'].value
        aw     =  A/(const.pi/180)**(1.0-gamma)
        aw_err  =  Aerr/(const.pi/180)**(1.0-gamma)
        try: 
            integral1,err = quad(limber_func, mes[slice].z1, mes[slice].z2,args=(mes,slice,)) 
            integral2,err = quad(dndz,mes[slice].z1,mes[slice].z2, args=(mes,slice))
            gf = np.sqrt(const.pi)*math.gamma((gamma-1.0)/2.0)/math.gamma(gamma/2.0)
            mes[slice].r0 = (aw*integral2*integral2/(gf*integral1))**(1.0/gamma)
            mes[slice].r0err = mes[slice].r0/(gamma*aw)*aw_err
            if (args.verbose):
                print ('%5s %5.2f %5.2f %5.2f %5.2f') % \
                (mes[slice].Mlim,mes[slice].z1, mes[slice].z2, mes[slice].r0, mes[slice].r0err)
        except:
            print 'Couldn\'t compute r0, sorry about that'
    return()


# fit all the shmr measurements
# including for completeness the abundance matching measurements. 
def fit_mmin_mstar(mes,results_am):
    fitted_params_mcen = {}
    fitted_params_shmr = {}
    fitted_params_shmr_am = {}
    slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
    for zl in slices:
        params_mcen = lmfit.Parameters()
        params_mcen.add('A',value=0.7)
        params_mcen.add('am',value=0.3)
        params_mcen.add('Mt',value=1e11)

        params_shmr = lmfit.Parameters()
        params_shmr.add('A',value=0.7)
        params_shmr.add('am',0.3,max=0.999)
        params_shmr.add('Mt',value=1e11,min=1e10)

        params_shmr_yang = lmfit.Parameters()
        params_shmr_yang.add('A',value=0.1)
        params_shmr_yang.add('beta',0.1,min=0.0,max=5.0)
        params_shmr_yang.add('gamma',0.1,min=0.0,max=5.0)
        params_shmr_yang.add('M1',value=10**11.5,min=1e11,max=1e13,vary=True)

        params_shmr_yang_am = lmfit.Parameters()
        params_shmr_yang_am.add('A',value=0.1)
        params_shmr_yang_am.add('beta',0.1,min=0.0,max=5.0)
        params_shmr_yang_am.add('gamma',0.1,min=0.0,max=5.0)
        params_shmr_yang_am.add('M1',value=10**11.5,min=10**11,max=10**13,vary=True)

        (log10Mmin,log10Mmin_perr,log10Mmin_nerr) = ms(mes,zl,'log10Mmin'), \
                                ms(mes,zl,'log10Mmin_perr'), ms(mes,zl,'log10Mmin_nerr')
        ag=np.argsort(log10Mmin)
        M_med= 10**(ms(mes,zl,'M_med')[ag])

        Mmin=10**log10Mmin[ag]                                              
        M_med= 10**(ms(mes,zl,'M_med')[ag])
        w = 1./(10**(log10Mmin+log10Mmin_perr+log10Mmin_nerr)/10**log10Mmin)[ag]
        am_err=0.5
        result = lmfit.minimize(resid_mstar_over_mh_yang,params_shmr_yang, \
            args=(M_med,Mmin,w),method='leastsq')
        fit_am = lmfit.minimize(resid_mstar_over_mh_yang,params_shmr_yang_am, \
            args=(10**results_am[zl][0],10**results_am[zl][1],am_err),method='leastsq')
#       print ('%5s %5.2e %5.2f %5.2f %5.2f') % \
#               (zl,params_shmr['Mt'].value,params_shmr['am'],\
#                   np.log10(params_shmr['Mt'].value/(1.0-params_shmr['am'].value)),\
#                params_shmr['A'].value)
        print ('%5s %5.2e %5.2f %5.2f %5.2f %5.2f ') % \
                (zl,params_shmr_yang['A'], np.log10(params_shmr_yang['M1'].value), np.log10(params_shmr_yang['M1'].stderr),
                    params_shmr_yang['gamma'].value, params_shmr_yang['beta'].value)
        print ('%5s %5.2e %5.2f %5.2f %5.2f %5.2f AM') % \
                (zl,params_shmr_yang_am['A'], np.log10(params_shmr_yang_am['M1'].value), np.log10(params_shmr_yang_am['M1'].stderr),
                    params_shmr_yang_am['gamma'].value, params_shmr_yang_am['beta'].value)
        fitted_params_mcen[zl]=params_mcen
        fitted_params_shmr[zl]=params_shmr_yang
        fitted_params_shmr_am[zl]=params_shmr_yang_am
        print(lmfit.fit_report(params_shmr_yang_am))
    return(fitted_params_mcen,fitted_params_shmr,fitted_params_shmr_am)


# fit all the shmr measurements
# including for completeness the abundance matching measurements. 

def mstar_over_mh_yang_mod(mmin,m1,gamma,beta,A):
    ans = 2.0 * A * ( (mmin/m1)**-beta+(mmin/m1)**gamma)**-1.0 
    return ans



def get_error_fit_mmin_mstar(mes):
    fitted_params_shmr = {}
#    slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
    slices = np.array([ 0.65])
    yangmod = Model(mstar_over_mh_yang_mod)
    params = yangmod.make_params()
    nsample=10
    for zl in slices:
        M1_arr = np.zeros(nsample)
        (log10Mmin,log10Mmin_perr,log10Mmin_nerr) = ms(mes,zl,'log10Mmin'), \
                                ms(mes,zl,'log10Mmin_perr'), ms(mes,zl,'log10Mmin_nerr')
        ag=np.argsort(log10Mmin)
        w = 1./(10**(log10Mmin+log10Mmin_perr+log10Mmin_nerr)/10**log10Mmin)[ag]
        for ns in np.arange(nsample):
            yangmod.set_param_hint('A',value=0.1)
            yangmod.set_param_hint('gamma',value=0.1,min=0.,max=5.0)
            yangmod.set_param_hint('beta',value=0.1,min=0,max=5.0)
            yangmod.set_param_hint('M1',value=10**11.5,min=1e11,max=1e13)
            params = yangmod.make_params()
            Mmin_ns = 10**ms(mes,zl,'pmcsamples')[:,ns,0]
            Mmin=Mmin_ns[ag]
            Mmin=10*log10Mmin[ag]
            M_med= 10**(ms(mes,zl,'M_med')[ag])
            result = yangmod.fit(M_med/Mmin,mmin=Mmin,m1=10**11.5,gamma=0.1,beta=0.1,A=0.1,weights=w)
            if (result.success):
                M1_arr[ns]=result.best_values['m1']
                print result.best_values['m1']
        print zl,np.log10(np.median(M1_arr)),np.std(np.log10(M1_arr)),0.434*np.std(M1_arr)/np.median(M1_arr)
    return(fitted_params_shmr,M1_arr)



# fit all the wtheta measurements 
def fit_wtheta(mes,b1,b2,varygamma):
    for slice in mes:
        th = mes[slice].th 
        wth = mes[slice].wth
        wth_err = mes[slice].wth_err
        params = mes[slice].params
        params['gamma'].vary=varygamma

        try:
            result = lmfit.minimize(residual, params, args=(th[b1:b2], wth[b1:b2], \
            wth_err[b1:b2]),method='leastsq')
            if (args.verbose):
                print ('%5s %5s %5s %5.2e +/- %5.2e %5.2f %5.2f') % \
                (mes[slice].Mlim,mes[slice].z1, mes[slice].z2, \
                 params['A'].value, params['A'].stderr, params['gamma'].value, params['gamma'].stderr)
        except:
            print 'fit didnt work. sorry.'

def runhp(argin):
    param_line=argin['param_line']
    hp_exe = argin['hp_exe']
    prefix_dir = argin['prefix_dir']
    command = hp_exe+' '+param_line
    print command
    r=subprocess.call(command.split(),cwd=prefix_dir)


# this computes the value of the clustering strength
# from the best-fitting halo model. For this to work, you need to compute
# xi(r) using haloplot if this isn't already done. 
def compute_r0_Mstar_hm(mes,prefix_dir):
    attrs = ['log10Mmin', 'log10M1', 'log10M0', 'sigma_log_M', 'alpha_halo']
    argin = {}
    list_args=[]
    num_proc = 4
    for sl in mes:
        argin = {}
        argin['hp_exe']='/Users/hjmcc/src/COSMOSTAT/trunk/bin/haloplot'
        hplot_xifile = sl.replace('wpmc_','haloplot_Vista')+'_xir.out'
        hp_parfile = sl.replace('wpmc_','halomodel.parVista')
        if not os.path.isfile(prefix_dir+'/'+hplot_xifile):
            print hplot_xifile, 'file doesn\'t exist'
            param_line = ''.join([(str(getattr(mes[sl],a)))+' ' for a in attrs])
            param_line += hp_parfile + ' -o ' +hplot_xifile + ' -t xi'
            argin ['param_line']=param_line
            argin ['prefix_dir']=prefix_dir
            list_args.append(argin)
#       r0,xi= np.loadtxt(hplot_xifile,usecols=(0,1),unpack=True,comments='#')
#       xifunc = interp1d(sort(x1),np.argsort(r0),'linear')
#       print xifunc(1.0)
    if list_args:
        pool = multiprocessing.Pool()
        result=pool.map(runhp,list_args)
        pool.close()
        pool.join()
    for sl in mes:
        hplot_xifile = prefix_dir+'/'+sl.replace('wpmc_','haloplot_Vista')+'_xir.out'
        r0,xi= np.loadtxt(hplot_xifile,usecols=(0,1),unpack=True,comments='#')
        xifunc = interp1d(np.sort(xi),r0[np.argsort(xi)],'linear')
        mes[sl].r0_hm=xifunc(1.0)
    return(mes)


# compute the error in R0 based on a number of samples. 
def compute_r0_Mstar_error(mes,prefix_dir):
    list_args=[]
    num_proc = 4
    nsample=100
    for sl in mes:
        hp_parfile = sl.replace('wpmc_','halomodel.parVista')
        samplefile = '/Users/hjmcc/cosmos/uvista/melody/PMC_utilsAll/'+sl.replace('wpmc_','All_')+'.sample.gz'
        xiparams=np.loadtxt(samplefile,usecols=(2,3,4,5,6),unpack=False)
        for ns in np.arange(nsample):
            hplot_xifile_ns = sl.replace('wpmc_','haloplot_Vista')+'_xir_'+str(ns)+'.out'
            if not os.path.isfile(prefix_dir+'/'+hplot_xifile_ns):
                argin = {}
                argin['hp_exe']='/Users/hjmcc/src/COSMOSTAT/trunk/bin/haloplot'
                print hplot_xifile_ns, 'file doesn\'t exist'
                param_line = ''.join([str(x)+' ' for x in xiparams[ns]])
                param_line += hp_parfile + ' -o ' +hplot_xifile_ns + ' -t xi'
                argin ['param_line']=param_line
                argin ['prefix_dir']=prefix_dir
                list_args.append(argin)
    if list_args:
        pool = multiprocessing.Pool()
        result=pool.map(runhp,list_args)
        pool.close()
        pool.join()
    for sl in  mes:
        r0_arr = np.zeros(nsample)
        for ns in np.arange(nsample):
            hplot_xifile_ns = prefix_dir+'/'+sl.replace('wpmc_','haloplot_Vista')+'_xir_'+str(ns)+'.out'
            r0,xi= np.loadtxt(hplot_xifile_ns,usecols=(0,1),unpack=True,comments='#')
            xifunc = interp1d(np.sort(xi),r0[np.argsort(xi)],'linear')
            r0_arr[ns]=xifunc(1.0)
        mes[sl].r0_hm_err=r0_arr.std()
        print sl,np.median(r0_arr),r0_arr.std()
    return(mes)




# for a given redshift slice, return the values of object, where object
# is something in the 'measurement' dictionary object. 
def ms(mes,zr,obj):
    return np.array([mes[i].__dict__[obj] for i in mes if str(mes[i].z)==str(zr)])

def plot_r0(mes,figname):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_xlim(8.8,11.8)
    ax.set_ylim(3.8,8.8)
    plt.xlabel(r'$\mathrm {median} [\mathrm{log}~M(M_\odot)]$')
    plt.ylabel(r'$r_0(\mathrm{Mpc})$')
#   slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
    slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
#   slices = np.array([0.95])
    for z in slices:
        M_med= ms(mes,z,'M_med')
        z1= ms(mes,z,'z1')
        z2= ms(mes,z,'z2')
#        r0_hm= ms(mes,z,'r0')
        r0_hm= ms(mes,z,'r0_hm')
        r0_hm_err= ms(mes,z,'r0_hm_err')
#        r0err= ms(mes,z,'r0err')
        pars = ms(mes,z,'params')
        gamma=pars[z]['gamma'].value
        fmt = r'$%s < z < %s$' % (z1[0],z2[0])
#       plt.plot(M_med,r0**(1.0/gamma), 
#           marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=12, 
#           mew=1.5, linestyle='', alpha=1.0, label=fmt)
#       plt.plot(M_med,r0, 
#           marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=12, 
#           mew=1.5, linestyle='', alpha=1.0, label=fmt)
        plt.plot(M_med,r0_hm, 
            marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=12, 
            mew=1.5, linestyle='', alpha=1.0, label=fmt)

        ax.errorbar(M_med,r0_hm,yerr=r0_hm_err,xerr=0.1, fmt='',
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0) 

    plt.legend(loc='upper left',numpoints=1,fancybox=False)
#       ax.errorbar(M_med,r0,yerr=r0err,linestyle='',color='black')

    zm,msv, r0v,r0verr = np.loadtxt('/Users/hjmcc/Dropbox/uvista/clustering-dr1/table3-marullietal.txt', \
                          usecols=(0,1,2,3), unpack='True', comments='#')
    # don't forget about that coversion to h
    r0v_noh = r0v 
    msv_noh=msv
    plt.plot(msv_noh[0:4],r0v_noh[0:4],
             markersize=8, mew=1,linestyle='',label='Marulli et al., z=0.65', 
             color='r', marker = (4,1,0))
    ax.errorbar(msv_noh[0:4],r0v_noh[0:4],yerr=r0verr[0:4],linestyle='',color='red')

    plt.plot(msv_noh[5:8],r0v_noh[5:8],
             markersize=8, mew=1.,linestyle='',label='Marulli et al., z=1.0', 
             color='r', marker = 's')
    ax.errorbar(msv_noh[5:8],r0v_noh[5:8],yerr=r0verr[5:8],linestyle='',color='red')
    plt.legend(loc='upper left',numpoints=1,fancybox=False)
    plt.savefig(figures+figname)
    plt.close('all')


def plot_Mmin_Mstar_withmodel(figname,mes,fit_results,z):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_xlim(8.8,11.75)
    ax.set_ylim(10.75,14.75)
    plt.xlabel(r'$\mathrm {median} [\mathrm{log}~M(M_\odot)]$')
    plt.ylabel(r'$\mathrm{log}~M_h (M_\odot)$')
#   slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
#   slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
    slices = np.array([0.65])
    for zr in slices:
        (log10Mmin,log10Mmin_perr,log10Mmin_nerr) = ms(mes,z,'log10Mmin'), \
                                                ms(mes,z,'log10Mmin_perr'), ms(mes,z,'log10Mmin_nerr')
        (log10M1,log10M1_perr,log10M1_nerr) = ms(mes,z,'log10M1'), \
                                                ms(mes,z,'log10M1_perr'), ms(mes,z,'log10M1_nerr')
        print 'Doing redshift', z 
        M_med= ms(mes,z,'M_med')
        z1= ms(mes,z,'z1')
        z2= ms(mes,z,'z2')
        Mmin= ms(mes,z,'Mmin')
        M1= ms(mes,z,'M1')
        pars = ms(mes,z,'params')
        fmt = r'$%s < z < %s$' % (z1[0],z2[0])
        plt.plot(M_med,Mmin, 
            marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=15, 
            mew=1.5, linestyle='', alpha=0.8, label=fmt)
        plt.plot(M_med,M1, 
            marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=15, 
            mew=1.5, linestyle='', alpha=0.8)
        ax.errorbar(M_med,log10Mmin,yerr=[log10Mmin_perr,log10Mmin_nerr],xerr=0.1, fmt='',
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0)     
        ax.errorbar(M_med,log10M1,yerr=[log10M1_perr,log10M1_nerr],xerr=0.1, fmt='',
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0)
        Mh_model = 10**np.arange(10,15,0.1)
        Mstar_model = mstar_mcen(fit_results[0][z],Mh_model)
        plt.plot(np.log10(Mstar_model),np.log10(Mh_model), 
            marker='', color=syms_zed[str(z)][1],markersize=15, 
            mew=1.5, linestyle='--', alpha=1.0, label='')
    
#   ax.annotate(r'$M_{\mathrm{1}}$', xy=(11.35, 14.25), xycoords="data",
#                  va="center", ha="center",bbox='')

#   ax.annotate(r'$M_{\mathrm{min}}$', xy=(11.35, 13.0), xycoords="data",
#                  va="center", ha="center",bbox='')
    
    plt.legend(loc='upper left',numpoints=1,fancybox=False)
    plt.savefig(figures+figname)
    plt.close('all')

def plot_Mstar_over_Mh_withmodel(figname,mes,fit_results,fit_results_bolshoi,z):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_ylim(-2.1,-1.25)
    ax.set_xlim(10.75,13.25)
    plt.xlabel(r'$\mathrm {median} [\mathrm{log}~M(M_\odot)]$')
    plt.ylabel(r'$\mathrm{log}~M_h (M_\odot)$')
#   slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
    slices = np.array([ 0.65,  0.95,  1.3 , 2.25])
#   slices = np.array([0.65])
    for z in slices:
        (log10Mmin,log10Mmin_perr,log10Mmin_nerr) = ms(mes,z,'log10Mmin'), \
                                                ms(mes,z,'log10Mmin_perr'), ms(mes,z,'log10Mmin_nerr')
#       (log10Mmin,log10Mmin_perr,log10Mmin_nerr) = ms(mes,z,'log10Mhalo_av'), 
#                               ms(mes,z,'log10Mhalo_av_perr'), ms(mes,z,'log10Mhalo_av_nerr')
        print 'Doing redshift', z 
        M_med= ms(mes,z,'M_med')
        z1= ms(mes,z,'z1')
        z2= ms(mes,z,'z2')
        pars = ms(mes,z,'params')
        fmt = r'$%s < z < %s$' % (z1[0],z2[0])
        plt.plot(log10Mmin,np.log10(10**M_med/10**log10Mmin), 
            marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=15, 
            mew=1.5, linestyle='', alpha=0.8, label=fmt)
        ax.errorbar(log10Mmin,np.log10(10**M_med/10**log10Mmin),xerr=[log10Mmin_perr,log10Mmin_nerr],yerr=0.1, fmt='',
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0)     
#        Mstar_model = 10**np.arange(10,15,0.1)
#        Mstar_over_mh_model = mstar_over_mh_yang(fit_results[1][z],Mstar_model)
        plt.plot(results_am[z][1],(results_am[z][0]-results_am[z][1]),marker='',\
                 mew=1.5, linestyle='-', alpha=1.0, label='')
#       plt.plot(np.log10(Mstar_model),np.log10(Mstar_over_mh_model), 
#           marker='', color=syms_zed[str(z)][1],markersize=15, 
#           mew=1.5, linestyle='--', alpha=1.0, label='')
    plt.legend(loc='upper left',numpoints=1,fancybox=False)
    plt.savefig(figures+figname)
    plt.close('all')

def plot_Mstar_over_Mh_withmodel_onepanel(figname,mes,fit_results,fit_results_bolshoi,limits_mstar):
    fig=plt.figure(figsize=(8,8))
    fig, axs = plt.subplots(3, 2, sharex=True, sharey=True,figsize=(8,8))
    fig.delaxes(axs[2,1])
    plot_order=(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)
    slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
    majorLocator   = FixedLocator(np.linspace(-1.3,-2.1,5))
    minorLocator   = FixedLocator(np.linspace(-1.3,-2.1,10))
    for i,z in enumerate(slices):
        axs[plot_order[i]].set_ylim(-2.1,-1.25)
        axs[plot_order[i]].set_xlim(10.75,13.25)
        axs[plot_order[i]].tick_params(axis='both', which='major', labelsize=15, width=1.0,size=10)
        axs[plot_order[i]].tick_params(axis='both', which='minor', labelsize=15, width=0.5,size=5)
        axs[plot_order[i]].yaxis.set_major_locator(majorLocator)
        for axis in ['top','bottom','left','right']:
            axs[plot_order[i]].spines[axis].set_linewidth(1.5)
        (log10Mmin,log10Mmin_perr,log10Mmin_nerr) = ms(mes,z,'log10Mmin'), \
                                                ms(mes,z,'log10Mmin_perr'), ms(mes,z,'log10Mmin_nerr')
        M_med= ms(mes,z,'M_med')
        z1= ms(mes,z,'z1')
        z2= ms(mes,z,'z2')
        pars = ms(mes,z,'params')
        fmt = r'$%s < z < %s$' % (z1[0],z2[0])
        hod_symb,=axs[plot_order[i]].plot(log10Mmin,np.log10(10**M_med/10**log10Mmin),
                                marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=10,mew=1.5, linestyle='', 
                                alpha=0.8, label=fmt)
        axs[plot_order[i]].errorbar(log10Mmin,np.log10(10**M_med/10**log10Mmin),
                                xerr=[log10Mmin_perr,log10Mmin_nerr],yerr=0.1, fmt='',linestyle='None', color='black', alpha=1,
                                elinewidth=0.8,capthick=1.25, capsize=0.0)       
        Mstar_model = 10**np.arange(10,15,0.05)
        Mstar_over_mh_model = mstar_over_mh_yang(fit_results[1][z],Mstar_model)
        Mstar_over_mh_model_AM = mstar_over_mh_yang(fit_results[2][z],Mstar_model)
        Mstar_over_mh_model_AM_bolshoi = mstar_over_mh_yang(fit_results_bolshoi[2][z],Mstar_model)
        am_thisfit,=axs[plot_order[i]].plot(np.log10(Mstar_model),np.log10(Mstar_over_mh_model_AM),
                                   marker='', lw=1.8,color='black',mew=1., linestyle='--', alpha=1.0, label='AM fit (this work)')
        am_bolshoi,=axs[plot_order[i]].plot(np.log10(Mstar_model),np.log10(Mstar_over_mh_model_AM_bolshoi),
                                   marker='', lw=1.8,color='black',mew=1., linestyle=':', alpha=1.0, label='AM fit (Bolshoi)')

        mhfunc=interp1d(results_am[z][0],results_am[z][1])
        limits_mh = mhfunc(limits_mstar[i])
        axs[plot_order[i]].annotate("", (limits_mh, -2.1), (limits_mh,-1.9),arrowprops=dict(edgecolor='black',arrowstyle="->"))
        if z not in ([1.3, 2.25]):
            hod_fit,=axs[plot_order[i]].plot(np.log10(Mstar_model),np.log10(Mstar_over_mh_model),
                                             marker='', lw=2.,color='black',mew=1., linestyle='-', alpha=1.0, label='HOD fit')
#            axs[plot_order[i]].legend(handles=[hod_symb,hod_fit],loc='upper left',numpoints=1,fancybox=False,fontsize=8)
            fl=axs[plot_order[i]].legend(handles=[hod_symb,hod_fit],
                                         loc='upper left',numpoints=1,fancybox=False,fontsize=9, markerscale=0.8)
        else:
            fl=axs[plot_order[i]].legend(handles=[hod_symb],
                                         loc='upper left',numpoints=1,fancybox=False,fontsize=9, markerscale=0.8)
        ax=plt.gca().add_artist(fl)
#            axs[plot_order[i]].legend(handles=[hod_symb],loc='upper left',numpoints=1,fancybox=False,fontsize=10)

        axs[plot_order[i]].legend(handles=[am_thisfit,am_bolshoi],
                                  loc='lower right',numpoints=1,fancybox=False,fontsize=8)
#         plt.tight_layout()
    plt.setp(axs[plot_order[3]].get_xticklabels(), visible=True)
    axs[plot_order[4]].set_xlabel(r'$\mathrm{log}~M_h (M_\odot)$',size=14)
    axs[plot_order[4]].set_ylabel(r'$\mathrm{log}[M^*/M_h]  $',size=14)
    axs[plot_order[0]].set_ylabel(r'$\mathrm{log}[M^*/M_h] $',size=14)
    axs[plot_order[2]].set_ylabel(r'$\mathrm{log}[M^*/M_h]  $',size=14)
#   axs[plot_order[4]].xaxis.set_major_locator(majorLocator)
    axs[plot_order[3]].set_xlabel(r'$\mathrm{log}~M_h (M_\odot)$',size=14)
    plt.setp(axs[plot_order[3]].get_xticklabels(), visible=True)
#    plt.tight_layout()
    plt.savefig(figures+figname)
    plt.close('all')


def plot_Mmin_Mstar(figname,mes,mes_passive):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_xlim(8.8,11.75)
    ax.set_ylim(10.75,14.75)
    plt.xlabel(r'$\mathrm {median} [\mathrm{log}~M(M_\odot)]$')
    plt.ylabel(r'$\mathrm{log}~M_h (M_\odot)$')
    slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
    for z in slices:
        (log10Mmin,log10Mmin_perr,log10Mmin_nerr) = ms(mes,z,'log10Mmin'), \
                                                ms(mes,z,'log10Mmin_perr'), ms(mes,z,'log10Mmin_nerr')
        (log10M1,log10M1_perr,log10M1_nerr) = ms(mes,z,'log10M1'), \
                                                ms(mes,z,'log10M1_perr'), ms(mes,z,'log10M1_nerr')
        (log10Mmin_p,log10Mmin_perr_p,log10Mmin_nerr_p) = ms(mes_passive,z,'log10Mmin'), \
                                                ms(mes_passive,z,'log10Mmin_perr'), ms(mes_passive,z,'log10Mmin_nerr')

        (log10M1_p,log10M1_perr_p,log10M1_nerr_p) = ms(mes_passive,z,'log10M1'), \
                                                ms(mes_passive,z,'log10M1_perr'), ms(mes_passive,z,'log10M1_nerr')
                                            
        print 'Doing redshift', z 
        M_med= ms(mes,z,'M_med')
        M_medp= ms(mes_passive,z,'M_med')
        z1= ms(mes,z,'z1')
        z2= ms(mes,z,'z2')
        Mmin= ms(mes,z,'Mmin')
        M1= ms(mes,z,'M1')
        pars = ms(mes,z,'params')
        fmt = r'$%s < z < %s$' % (z1[0],z2[0])
        plt.plot(M_med,Mmin, 
            marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=15, 
            mew=1.5, linestyle='', alpha=0.8, label=fmt)
        plt.plot(M_med,M1, 
            marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=15, 
            mew=1.5, linestyle='', alpha=0.8)
        plt.plot(M_medp,log10Mmin_p, 
                marker=syms_zed[str(z)][0], color='white',markersize=15, 
                mew=1.5, mec='red', linestyle='', alpha=0.8, label='')
        ax.errorbar(M_medp,log10Mmin_p,yerr=[log10Mmin_perr_p,log10Mmin_nerr_p],xerr=0.1, fmt='',\
                linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0) 
        plt.plot(M_medp,log10M1_p, 
                marker=syms_zed[str(z)][0], color='white',markersize=15, 
                mew=1.5, mec='red', linestyle='', alpha=0.8, label='')
        ax.errorbar(M_medp,log10M1_p,yerr=[log10M1_perr_p,log10M1_nerr_p],xerr=0.1, fmt='',\
                linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0) 

        ax.errorbar(M_med,log10Mmin,yerr=[log10Mmin_perr,log10Mmin_nerr],xerr=0.1, fmt='',
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0)     
        ax.errorbar(M_med,log10M1,yerr=[log10M1_perr,log10M1_nerr],xerr=0.1, fmt='',
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0)     
    
    ax.annotate(r'$M_{\mathrm{1}}$', xy=(11.35, 14.25), xycoords="data",
                  va="center", ha="center",bbox='')

    ax.annotate(r'$M_{\mathrm{min}}$', xy=(11.35, 13.0), xycoords="data",
                  va="center", ha="center",bbox='')
#                  bbox=dict(boxstyle="square", fc="w"))
    plt.legend(loc='upper left',numpoints=1,fancybox=False)
    plt.savefig(figures+figname)
    plt.close('all')

def plot_bgal(figname,mes,mes_passive):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_ylim(0.75,4.25)
    ax.set_xlim(8.8,11.75)
    plt.xlabel(r'$\mathrm {median} [\mathrm{log}~M(M_\odot)]$')
    plt.ylabel(r'$\mathrm{b}_\mathrm{gal}$')
#   slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
    slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
#   slices = np.array([2.25])
    for z in slices:
        M_med= ms(mes,z,'M_med')
        z1= ms(mes,z,'z1')
        z2= ms(mes,z,'z2')
        pars = ms(mes,z,'params')
        fmt = r'$%s < z < %s$' % (z1[0],z2[0])
        (bgal_av,bgal_av_perr,bgal_av_nerr) = ms(mes,z,'bgal_av'), \
                                                ms(mes,z,'bgal_av_perr'), ms(mes,z,'bgal_av_nerr')
        (bgal_avp,bgal_avp_perr,bgal_avp_nerr) = ms(mes_passive,z,'bgal_av'), \
                                                ms(mes_passive,z,'bgal_av_perr'), ms(mes_passive,z,'bgal_av_nerr')
        M_med= ms(mes,z,'M_med')
        plt.plot(M_med,bgal_av, 
            marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=15, 
            mew=1.5, linestyle='', alpha=0.8, label=fmt)
        ax.errorbar(M_med,bgal_av,yerr=[bgal_av_perr,bgal_av_nerr],xerr=0.1, fmt='',
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0) 

        M_medp= ms(mes_passive,z,'M_med')
        plt.plot(M_medp,bgal_avp, 
            marker=syms_zed[str(z)][0], color='white',markersize=15, 
            mew=1.5, mec='red', linestyle='', alpha=0.8, label='')
        ax.errorbar(M_medp,bgal_avp,yerr=[bgal_avp_perr,bgal_avp_nerr],xerr=0.1, fmt='',
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0) 

    plt.legend(loc='upper left',numpoints=1,fancybox=False)
    plt.savefig(figures+figname)
    plt.close('all')

def plot_bgal_twopanel(figname,mes,mes_passive):
    fig=plt.figure(figsize=(8,8))
    fig, ((ax1, ax2)) = plt.subplots(1, 2, sharex=False, sharey=True)
    ax1.set_ylim(0.75,4.25)
    ax1.set_xlim(8.8,11.75)
#   ax1.xlabel(r'$\mathrm {median} [\mathrm{log}~M(M_\odot)]$')
#   ax1.ylabel(r'$\mathrm{b}_\mathrm{gal}$')
    slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
    for z in slices:
        M_med= ms(mes,z,'M_med')
        z1= ms(mes,z,'z1')
        z2= ms(mes,z,'z2')
        pars = ms(mes,z,'params')
        fmt = r'$%s < z < %s$' % (z1[0],z2[0])
        (bgal_av,bgal_av_perr,bgal_av_nerr) = ms(mes,z,'bgal_av'), \
                                                ms(mes,z,'bgal_av_perr'), ms(mes,z,'bgal_av_nerr')
        M_med= ms(mes,z,'M_med')
        ax1.plot(M_med,bgal_av, 
            marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=15, 
            mew=1.5, linestyle='', alpha=0.8, label=fmt)
        ax1.errorbar(M_med,bgal_av,yerr=[bgal_av_perr,bgal_av_nerr],xerr=0.1, fmt='',
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0) 
        ax1.legend(loc='upper left',numpoints=1,fancybox=False)
    for z in slices:
        pars = ms(mes,z,'params')
        fmt = r'$%s < z < %s$' % (ms(mes,z,'z1')[0],ms(mes,z,'z2')[0])

        (bgal_avp,bgal_avp_perr,bgal_avp_nerr) = ms(mes_passive,z,'bgal_av'), \
                                                ms(mes_passive,z,'bgal_av_perr'), ms(mes_passive,z,'bgal_av_nerr')
        M_medp= ms(mes_passive,z,'M_med')
        ax2.plot(M_medp,bgal_avp, 
            marker=syms_zed[str(z)][0], color='white',markersize=15, 
            mew=1.5, mec='red', linestyle='', alpha=0.8, label='')
        ax2.errorbar(M_medp,bgal_avp,yerr=[bgal_avp_perr,bgal_avp_nerr],xerr=0.1, fmt='',
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0) 
        ax2.legend(loc='upper left',numpoints=1,fancybox=False)
    plt.savefig(figures+figname)
    plt.close('all')


def plot_redfrac_ngal(figname,mes,mes_passive):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_ylim(0.01,1)
    ax.set_xlim(10.25,11.25)
    ax.set_yscale('log')
    fig.subplots_adjust(left=0.15)
    y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)
    slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
    for z in slices:
        z1= ms(mes,z,'z1')
        z2= ms(mes,z,'z2')
        fmt = r'$%s < z < %s$' % (z1[0],z2[0])
        nsat_red = ms(mes_passive,z,'fr_sat') * ms(mes_passive,z,'Ngal') 
        ngal_red = ms(mes_passive,z,'Ngal') 
        M_med= ms(mes_passive,z,'Mlim')
        Mlim=ms(mes,z,'Mlim')
        Mlim_p=ms(mes_passive,z,'Mlim')
        Mmedp=ms(mes_passive,z,'M_med')
        Ngal=ms(mes,z,'Ngal')
        Ngal_arr = {}
        f_red = {}
        f_red_err = {}

        for k,index in enumerate(np.argsort(Mlim)):
            Ngal_arr[Mlim[index]]= Ngal[index]
        f_red['f_red_sat'] = np.zeros(np.size(Mlim_p))
        f_red['f_red'] = np.zeros(np.size(Mlim_p))
        f_red['Mlim'] = np.zeros(np.size(Mlim_p))
        f_red_err['f_red_sat'] = np.zeros(np.size(Mlim_p))
        f_red_err['f_red'] = np.zeros(np.size(Mlim_p))
        f_red_err['Mlim'] = np.zeros(np.size(Mlim_p))
        for k,lim in enumerate(Mlim_p):
            f_red['f_red_sat'][k]=nsat_red[k]/float(Ngal_arr[lim])
            f_red['f_red'][k] = ngal_red[k]/float(Ngal_arr[lim])
            f_red['Mlim'][k] = Mmedp[k]
            f_red_err['f_red'][k] = np.sqrt(1.0/(ngal_red[k])+1.0/(float(Ngal_arr[lim])))
            f_red_err['f_red_sat'][k] = np.sqrt(1.0/(nsat_red[k])+1.0/(float(Ngal_arr[lim])))

        plt.plot(f_red['Mlim'],f_red['f_red_sat'], 
            marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=15, 
            mew=1.5, linestyle='', alpha=0.8, label=fmt)

        plt.plot(f_red['Mlim'],f_red['f_red'], 
                marker=syms_zed[str(z)][0], color='white',markersize=15, 
                mew=1.5, mec='red', linestyle='', alpha=0.8, label='')

        ax.errorbar(f_red['Mlim'],f_red['f_red_sat'],yerr=f_red_err['f_red_sat'][k],xerr=0.1, fmt='',
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0) 

        ax.errorbar(f_red['Mlim'],f_red['f_red'],yerr=f_red_err['f_red'][k],xerr=0.1, fmt='',
            linestyle='None', color='red', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0) 

    plt.xlabel(r'$\mathrm {median} [\mathrm{log}~M(M_\odot)]$')
    plt.ylabel(r'passive fraction')
    plt.legend(loc='lower right',numpoints=1,fancybox=False)
    plt.savefig(figures+figname)
    plt.close('all')

def plot_Ngal_frsat(figname,mes,mes_passive):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_ylim(0.01,0.3)
    ax.set_xlim(-1.25,-3.75)
    plt.xlabel(r'$\mathrm {median} [\mathrm{log}~M(M_\odot)]$')
#   plt.ylabel(r'$\mathrm{log}~M_h (M_\odot)$')
#   slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
    slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
#   slices = np.array([0.65])
    for z in slices:
        M_med= ms(mes,z,'M_med')
        z1= ms(mes,z,'z1')
        z2= ms(mes,z,'z2')
        lN_g= np.log10(ms(mes,z,'n_g'))
        frsat = ms(mes,z,'fr_sat')
        fmt = r'$%s < z < %s$' % (z1[0],z2[0])
        plt.plot(lN_g,frsat, 
            marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=15, 
            mew=1.5, linestyle='', alpha=0.8, label=fmt)

        lN_g_p= np.log10(ms(mes_passive,z,'n_g'))
        fr_sat_p= ms(mes_passive,z,'fr_sat')
        plt.plot(lN_g_p,fr_sat_p, 
            marker=syms_zed[str(z)][0], color='white',markersize=15, 
            mew=1.5, mec='red', linestyle='', alpha=0.8, label='')
        
    plt.legend(loc='upper left',numpoints=1,fancybox=False)
    plt.savefig(figures+figname)
    plt.close()


def plot_lMin_frsat(figname,mes,mes_passive):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_ylim(0.01,0.30)
    ax.set_xlim(10.75,13.75)
    plt.xlabel(r'$\mathrm{log}~M_h (M_\odot)$')
    plt.ylabel(r'Sattellite fraction')
#   plt.ylabel(r'$\mathrm{log}~M_h (M_\odot)$')
#   slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
    slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
#   slices = np.array([0.65])
    for z in slices:
        (log10Mmin,log10Mmin_perr,log10Mmin_nerr) = ms(mes,z,'log10Mmin'), \
                                                ms(mes,z,'log10Mmin_perr'), ms(mes,z,'log10Mmin_nerr')
        M_med= ms(mes,z,'M_med')
        z1= ms(mes,z,'z1')
        z2= ms(mes,z,'z2')
        frsat = ms(mes,z,'fr_sat')
        (frsat,frsat_perr,frsat_nerr) = ms(mes,z,'fr_sat'), \
                                                ms(mes,z,'fr_sat_perr'), ms(mes,z,'fr_sat_nerr')
        fmt = r'$%s < z < %s$' % (z1[0],z2[0])
        plt.plot(log10Mmin,frsat, 
            marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=15, 
            mew=1.5, linestyle='', alpha=0.8, label=fmt)
        f=(open,('frsat_all_%s.txt') % z)
        np.savetxt(f,np.transpose((log10Mmin,frsat,frsat_perr,frsat_nerr)),fmt='%5.2f %5.2f %5.2f %5.2f')
        f.close()
        ax.errorbar(log10Mmin,frsat,yerr=[frsat_perr,frsat_nerr],xerr=[log10Mmin_perr,log10Mmin_nerr], fmt='',
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0) 

        M_medp= ms(mes_passive,z,'M_med')
        (frsat_p,frsat_p_perr,frsat_p_nerr) = ms(mes_passive,z,'fr_sat'), \
                                              ms(mes_passive,z,'fr_sat_perr'), ms(mes_passive,z,'fr_sat_nerr')
#       plt.plot(M_medp,frsat_p, 
#               marker=syms_zed[str(z)][0], color='white',markersize=15, 
#               mew=1.5, mec='red', linestyle='', alpha=0.8, label='')
#       ax.errorbar(M_medp,frsat_p,yerr=[frsat_p_perr,frsat_p_nerr],xerr=0.1, fmt='',
#           linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0) 
    plt.legend(loc='upper left',numpoints=1,fancybox=False)
    plt.savefig(figures+figname)
    plt.close()



def plot_lMstar_frsat(figname,mes,mes_passive):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_ylim(0.01,0.425)
    ax.set_xlim(8.8,11.75)
    plt.xlabel(r'$\mathrm {median} [\mathrm{log}~M(M_\odot)]$')
    plt.ylabel(r'Sattellite fraction')
#   plt.ylabel(r'$\mathrm{log}~M_h (M_\odot)$')
#   slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
    slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
#   slices = np.array([0.65])
    for z in slices:
        M_med= ms(mes,z,'M_med')
        z1= ms(mes,z,'z1')
        z2= ms(mes,z,'z2')
        frsat = ms(mes,z,'fr_sat')
        (frsat,frsat_perr,frsat_nerr) = ms(mes,z,'fr_sat'), \
                                                ms(mes,z,'fr_sat_perr'), ms(mes,z,'fr_sat_nerr')

        fmt = r'$%s < z < %s$' % (z1[0],z2[0])
        plt.plot(M_med,frsat, 
            marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=15, 
            mew=1.5, linestyle='', alpha=0.8, label=fmt)

        ax.errorbar(M_med,frsat,yerr=[frsat_perr,frsat_nerr],xerr=0.1, fmt='',
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0) 

        f=(open,('frsat_all_%s.txt') % z)
        np.savetxt(f,np.transpose((M_med,frsat,frsat_perr,frsat_nerr)),fmt='%5.2f %5.2f %5.2f %5.2f')
        f.close()
 

        M_medp= ms(mes_passive,z,'M_med')
        (frsat_p,frsat_p_perr,frsat_p_nerr) = ms(mes_passive,z,'fr_sat'), \
                                              ms(mes_passive,z,'fr_sat_perr'), ms(mes_passive,z,'fr_sat_nerr')
        plt.plot(M_medp,frsat_p, 
                marker=syms_zed[str(z)][0], color='white',markersize=15, 
                mew=1.5, mec='red', linestyle='', alpha=0.8, label='')
        ax.errorbar(M_medp,frsat_p,yerr=[frsat_p_perr,frsat_p_nerr],xerr=0.1, fmt='',
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0)
#        print ('%5.2f %5.2f %5.2f %5.2f') % 
    plt.legend(loc='upper left',numpoints=1,fancybox=False)
    plt.savefig(figures+figname)
    plt.close()

def plot_Ngal(figname,mes,mes2):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_xlim(-1.5,-4.)
    ax.set_ylim(10.75,13.25)
#   ax.set_ylim(11.75,14.75)
    plt.xlabel(r'$\mathrm{log}~N(\mathrm{Mpc}^{-3})$')
    plt.ylabel(r'$\mathrm{log}~M_\mathrm{min}(M_\odot)$')
    slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
#   slices = np.array([ 0.65])
    lngal_jc = np.arange(-1.5,-3.0,-0.01)
    lmmin_jc = np.log10(((10**lngal_jc)**-0.84)*1e10)
    z=0.65
    lN_g= np.log10(ms(mes,z,'n_g'))
    (log10Mmin,log10Mmin_perr,log10Mmin_nerr) = ms(mes,z,'log10Mmin'), \
                                                ms(mes,z,'log10Mmin_perr'), ms(mes,z,'log10Mmin_nerr')
    params=Parameters()
    params.add('sl',value=-0.8)
    params.add('C',value=10.0)
    errMmin=(log10Mmin_perr+log10Mmin_nerr)/2.0
    fitfunc = lambda params,lngalf: params['sl'].value*lngalf+params['C'].value
    resid = lambda params, lngalf,lmminf,erry: (fitfunc(params,lngalf) - lmminf)/erry
    result = lmfit.minimize(resid, params, args=(np.sort(lN_g)[5:], log10Mmin[np.argsort(lN_g)][5:],  \
                             errMmin[np.argsort(lN_g)][5:]),method='leastsq')
    lmmin_hjmcc = lngal_jc*params['sl'].value + params['C'].value
    for z in slices:
        z1= ms(mes,z,'z1')
        z2= ms(mes,z,'z2')
        M1= ms(mes,z,'M1')
        lN_g= np.log10(ms(mes,z,'n_g'))
        fmt = r'$%s < z < %s$' % (z1[0],z2[0])
        (log10Mmin,log10Mmin_perr,log10Mmin_nerr) = ms(mes,z,'log10Mmin'), \
                                                    ms(mes,z,'log10Mmin_perr'), ms(mes,z,'log10Mmin_nerr')
        plt.plot(lN_g,log10Mmin, 
            marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=15, 
            mew=1.5, linestyle='', alpha=0.8, label=fmt)
        ax.errorbar(lN_g,log10Mmin,yerr=[log10Mmin_perr,log10Mmin_nerr],xerr=0.1, fmt='',
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0)     

        lN_g2= np.log10(ms(mes2,z,'n_g'))

        (log10Mminp,log10Mminp_perr,log10Mminp_nerr) = ms(mes2,z,'log10Mmin'), \
                                                ms(mes2,z,'log10Mmin_perr'), ms(mes2,z,'log10Mmin_nerr')

        ax.errorbar(lN_g2,log10Mminp,yerr=[log10Mminp_perr,log10Mminp_nerr],xerr=0.1, fmt='',
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0)     


        plt.plot(lN_g2,log10Mminp, 
            marker=syms_zed[str(z)][0], color='white',markersize=15, 
            mew=1.5, mec='red', linestyle='', alpha=0.8, label='')
#   ax.errorbar(11.25,11.5,yerr=0.2,xerr=0.2,linestyle='',color='red', \
#       label='typical error')
    plt.plot(lngal_jc,lmmin_hjmcc, linestyle='-',color='black',lw=1.5)
    plt.legend(loc='upper left',numpoints=1,fancybox=False)
    plt.savefig(figures+figname)
    plt.close()

def plot_Ngal_r0(figname,mes,mes2):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_xlim(-1.25,-3.75)
    ax.set_ylim(3,10)
    plt.xlabel(r'$N(h^{3})(\mathrm{Mpc}^{-3})$')
#   plt.ylabel(r'$\mathrm{log}~M_\mathrm{min}(M_\odot)$')
    plt.ylabel(r'$r_0(h^{-1} \mathrm{Mpc})$')
    slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
#   slices = np.array([ 0.65,  0.95,  1.3 ,  1.75])
    for z in slices:
        z1= ms(mes,z,'z1')
        z2= ms(mes,z,'z2')
        Mmin= ms(mes,z,'Mmin')
        M1= ms(mes,z,'M1')
        lN_g= np.log10(ms(mes,z,'n_g'))
        r0_hm= ms(mes,z,'r0_hm')
        fmt = r'$%s < z < %s$' % (z1[0],z2[0])
        plt.plot(lN_g,r0_hm, 
            marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=15, 
            mew=1.5, linestyle='', alpha=0.8, label=fmt)
#       if z != 2.25:
#           Mmin2= ms(mes2,z,'Mmin')            
#           lN_g2= np.log10(ms(mes2,z,'n_g'))
#           M12= ms(mes2,z,'M1')
#           plt.plot(lN_g2,Mmin2, 
#                    marker=syms_zed[str(z)][0], color='white',markersize=15, 
#                    mew=1.5, mec='red', linestyle='', alpha=0.8, label='')
#   ax.errorbar(11.25,11.5,yerr=0.2,xerr=0.2,linestyle='',color='red', \
#       label='typical error')
    plt.legend(loc='upper left',numpoints=1,fancybox=False)
    plt.savefig(figures+figname)
    plt.close('all')

def plot_Mmin_r0(figname,mes,mes2):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
#   ax.set_xlim(-1.25,-3.75)
    ax.set_xlim(10.75,13.75)
    ax.set_ylim(3,10)
#   plt.xlabel(r'$N(h^{3})(\mathrm{Mpc}^{-3})$')
    plt.xlabel(r'$\mathrm{log}~M_\mathrm{min}(M_\odot)$')
    plt.ylabel(r'$r_0(h^{-1} \mathrm{Mpc})$')
    slices = np.array([ 0.65])
#   slices = np.array([ 0.65,  0.95,  1.3 ,  1.75])
    for z in slices:
        z1= ms(mes,z,'z1')
        z2= ms(mes,z,'z2')
        Mmin= ms(mes,z,'Mmin')
        M1= ms(mes,z,'M1')
        lN_g= np.log10(ms(mes,z,'n_g'))
        r0_hm= ms(mes,z,'r0_hm')
        fmt = r'$%s < z < %s$' % (z1[0],z2[0])
#       plt.plot(lN_g,Mmin, 
#           marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=15, 
#           mew=1.5, linestyle='', alpha=0.8, label=fmt)
        plt.plot(Mmin,r0_hm, 
            marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=15, 
            mew=1.5, linestyle='', alpha=0.8, label=fmt)
#       if z != 2.25:
#           Mmin2= ms(mes2,z,'Mmin')            
#           lN_g2= np.log10(ms(mes2,z,'n_g'))
#           M12= ms(mes2,z,'M1')
#           plt.plot(lN_g2,Mmin2, 
#                    marker=syms_zed[str(z)][0], color='white',markersize=15, 
#                    mew=1.5, mec='red', linestyle='', alpha=0.8, label='')
#   ax.errorbar(11.25,11.5,yerr=0.2,xerr=0.2,linestyle='',color='red', \
#       label='typical error')
    plt.legend(loc='upper left',numpoints=1,fancybox=False)
    plt.savefig(figures+figname)
    plt.close()


def plot_M1overMminvsMstar(figname,mes,mes2):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_ylim(0.5,47.5)
    ax.set_xlim(8.8,11.75)
    ax.axhline(y=23,lw=1,ls='--',color='black')
    plt.ylabel(r'$M_1/M_{\mathrm{min}}$')
    plt.xlabel(r'$\mathrm {median} [\mathrm{log}~M(M_\odot)]$')

    slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
#   slices = np.array([ 0.65,  0.95,  1.3 ,  1.75])
    for z in slices:
        z1= ms(mes,z,'z1')
        z2= ms(mes,z,'z2')
        Mmin= ms(mes,z,'Mmin')
        log10M1= ms(mes,z,'log10M1')
        log10Mmin= ms(mes,z,'log10Mmin')
        log10M12= ms(mes2,z,'log10M1')
        log10Mmin2= ms(mes2,z,'log10Mmin')


        (log10M1_perr,log10M1_nerr) = ms(mes,z,'log10M1_perr'), ms(mes,z,'log10M1_nerr')
        (log10Mmin_perr,log10Mmin_nerr) = ms(mes,z,'log10Mmin_perr'), ms(mes,z,'log10Mmin_nerr')
        (log10M12_perr,log10M12_nerr) = ms(mes2,z,'log10M1_perr'), ms(mes2,z,'log10M1_nerr')
        (log10Mmin2_perr,log10Mmin2_nerr) = ms(mes2,z,'log10Mmin_perr'), ms(mes2,z,'log10Mmin_nerr')
        err_min=(10**log10Mmin-10**(log10Mmin-log10Mmin_nerr)+(10**(log10Mmin+log10Mmin_perr)-10**log10Mmin))/2.0/10**log10Mmin
        err_m1=(10**log10M1-10**(log10M1-log10M1_nerr)+(10**(log10M1+log10M1_perr)-10**log10M1))/2.0/10**log10M1
        err_min2=(10**log10Mmin2-10**(log10Mmin2-log10Mmin2_nerr)+(10**(log10Mmin2+log10Mmin2_perr)-10**log10Mmin2))/2.0/10**log10Mmin2
        err_m12=(10**log10M12-10**(log10M12-log10M12_nerr)+(10**(log10M12+log10M12_perr)-10**log10M12))/2.0/10**log10M12


        dy = np.sqrt(err_min**2+err_m1**2)*10**log10M1/10**log10Mmin
        dy2= np.sqrt(err_min2**2+err_m12**2)*10**log10M12/10**log10Mmin2


        print dy
        M_med= ms(mes,z,'M_med')
        M1= ms(mes,z,'M1')
        Mmin2= ms(mes2,z,'Mmin')
        M12= ms(mes2,z,'M1')
        M_med2 = ms(mes2,z,'M_med')
        fmt = r'$%s < z < %s$' % (z1[0],z2[0])
        plt.plot(M_med,10**M1/10**Mmin, \
            marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=15, 
            mew=1.5, linestyle='', alpha=0.8, label=fmt)
        ax.errorbar(M_med,10**M1/10**Mmin,yerr=dy,elinewidth=0.8,lw=2,capsize=0.0,
                    capthick=1.25,linestyle='',color='black',label='')
        if z != 2.25:
            Mmin2= ms(mes2,z,'Mmin')            
            M12= ms(mes2,z,'M1')
            plt.plot(M_med2,(10**M12/10**Mmin2), \
                     marker=syms_zed[str(z)][0], color='white',markersize=15, 
                     mec='red', mew=2, linestyle='', alpha=0.5, label='')
            ax.errorbar(M_med2,10**M12/10**Mmin2,yerr=dy2,elinewidth=0.5,lw=2,capsize=0.0,
                    capthick=1.25,linestyle='',color='red',label='',alpha=0.5)

    M1_b, M1_b_err, Mmin_b, Mmin_berr,  = np.loadtxt('/Users/hjmcc/Dropbox/uvista/clustering-dr1/beutler-tab3.txt', \
                          usecols=(1,2,5,6), unpack='True', comments='#')
    plt.legend(loc='upper left',numpoints=1,fancybox=False)
    plt.savefig(figures+figname)
    plt.close()

# compute fraction of passive / quiescent galaxies for a given redshift slice 
# to do that we need to associate the parameters from olivier's paper to each slice. 
# we need to loop over all slices at each threshold and calculate the fraction of quiescent
# galaxies. 


def compute_fraction(mes,mes_passive):

    for slice in mes:
        ntot = compute_M(mes[slice].Mlim,mes[slice].lMstar,mes[slice].phi1,mes[slice].alpha1,
                  mes[slice].alpha2,mes[slice].phi2)[0]

def compute_M(lmlim,lMstar,phi1,alpha1,phi2,alpha2):
    lmmin = 7.86
    Mstar=10**lMstar
    mlim =10**lmlim
    mmin = 10**lmmin
    phiMdm = lambda M : np.exp(-1.0*M/Mstar)*(phi1*(M/Mstar)**alpha1+phi2*(M/Mstar)**alpha2)/Mstar
    Mtot = quad(phiMdm,mmin,mlim)
    return (Mtot)


def compute_abundances_M(lmlim,lMstar,phi1,alpha1,phi2,alpha2):
    lmlim_max = 13.0
    Mstar=10**lMstar
    mlim =10**lmlim
    mmin = 10**lmmin
    mlim_max = 10**lmlim_max
    phiMdm = lambda M : np.exp(-1.0*M/Mstar)*(phi1*(M/Mstar)**alpha1+phi2*(M/Mstar)**alpha2)/Mstar
    Mtot = quad(phiMdm,mlim,mlim_max)
    return (Mtot)
  

def plot_wtheta(figname,mes,mes_ref,z,zr,haloplot=True):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(0.0008,0.2)
    ax.set_ylim(0.005,15)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3g'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2g'))
#    y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
#    ax.yaxis.set_major_formatter(y_formatter)
#    ax.xaxis.set_major_formatter(y_formatter)
    plt.xlabel(r'$\theta (^{\circ})$')
    plt.ylabel(r'$w$')
# add comoving distance to the horizontal axis. 
    Mlim=      np.array([mes[i].Mlim for i in mes if str(mes[i].z)==str(z)])
    Mlim_ref=  [mes_ref[i].Mlim for i in mes_ref if str(mes_ref[i].z)==str(zr)]

    wth =      np.array([mes[i].wth for i in mes if str(mes[i].z)==str(z)])
    wth_err =  np.array([mes[i].wth_err for i in mes if str(mes[i].z)==str(z)])
    th =       np.array([mes[i].th for i in mes if str(mes[i].z)==str(z)])
    pars   =   np.array([mes[i].params for i in mes if str(mes[i].z)==str(z)])
    z1     =   np.array([mes[i].z1 for i in mes if str(mes[i].z)==str(z)])
    z2     =   np.array([mes[i].z2 for i in mes if str(mes[i].z)==str(z)])
    pars_ref   =   np.array([mes_ref[i].params for i in mes_ref if str(mes_ref[i].z)==str(zr)])
    M= ms(mes,z,'M')
    nm= ms(mes,z,'nm')
    nc= ms(mes,z,'nc')
    nsat= ms(mes,z,'nsat')
    wth_hp= ms(mes,z,'wth_hp')
    th_hp= ms(mes,z,'th2')
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    rcomoving = lambda ang: cosmo.kpc_comoving_per_arcmin(z).value*60.0/1.0e3*ang
    ax2_wtheta = ax.twiny()
    ax2_wtheta.set_xscale('log')
    fmt = ticker.ScalarFormatter(useOffset=False)
    fmt.set_scientific('False')
    ax2_wtheta.xaxis.set_major_formatter(fmt)
    ax2_wtheta.xaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1f'))
    ax2_wtheta.set_xlim(rcomoving(0.0008),rcomoving(0.2))
    ax2_wtheta.set_xlabel(r"$r~(\mathrm{Mpc})$")

#   Mlim_keep=np.array([  9.0 ,   9.4,   9.8,  10.2,  10.6, 11.0 ])
    Mlim_keep=np.array([  9.0 ,   9.4,  10.2,  10.6,  11.0  ])
    if (haloplot):
        ax2 = fig.add_axes([.225, .18, .225, .225])
        ax2.tick_params(axis='both', length=7, width=1, which='major', labelsize=10)
        ax2.tick_params(axis='both', length=5, width=0.5, which='minor', labelsize=10)
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.minorticks_off()
        ax2.set_xlim(1e10,1e16)
        ax2.set_ylim(0.1,100)
        ax2.set_xlabel(r'$~M(M_\odot)$',size=12)
        ax2.set_ylabel(r'$<N>$',size=12)
        ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        [i.set_linewidth(1) for i in ax2.spines.itervalues()]
    for k,index in enumerate(np.argsort(Mlim)):
        if Mlim[index] not in Mlim_keep:
            continue
        lim=Mlim[index]
        fmtstring=(r'$\mathrm{log} M_*> %s$')%lim
        ax.plot(th[index],wth[index], 
            marker=syms[str(lim)][0], color=syms[str(lim)][1],markersize=12, 
            mew=1.5, linestyle='', alpha=1.0, label=fmtstring)
        wth_err2 = np.array(wth_err[index])
        wth_err2[wth_err[index]>=wth[index]] = wth[index][wth_err[index]>=wth[index]]*-0.99999
        ax.errorbar(th[index],wth[index],yerr=[wth_err2,wth_err[index]],
            uplims=wth_err[index]>=wth[index], 
            fmt='',linestyle='None', color='black', elinewidth=1.25,capthick=1.25, capsize=3.0)

        ax.legend(loc='upper right',numpoints=1,fancybox=False)
        # draw this curve only if we are doing the halo plot. 
        if (not haloplot):
            ir = Mlim_ref.index(Mlim[index])
            w_model=wtheta_model(pars_ref[ir],th[index])
            ax.plot(th[index],w_model,'--', color='red')

        if (haloplot):
            ax2.plot(M[index],nm[index],label='',color=syms[str(lim)][1], lw=2.5)
            ax2.plot(M[index],nsat[index],'--',label='',color=syms[str(lim)][1], lw=1)
            ax2.plot(M[index],nc[index],'-.',label='',color=syms[str(lim)][1], lw=1)
            ax.plot(th_hp[index],wth_hp[index],'-',alpha=0.8,color=syms[str(lim)][1],lw=3,label='')

    fmt = r'$%5.2f < z < %5.2f$' % (z1[0],z2[0])
    plt.figtext(0.55,0.175,fmt)
    plt.savefig(figures+figname)
    plt.close() 


def plot_Mstar_z(figname,inputfile,mes,vmax_val,histmax_val,limits):
    slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_xlim(0.0,3.0)
    ax.set_ylim(8.2,11.7)
    plt.xlabel(r'$z_\mathrm{phot}$')
    plt.ylabel(r'$\mathrm{log}~M(M_\odot)$')
    z,Mstar=np.loadtxt(inputfile,usecols=(2,3),unpack=True)
    H,xedges,yedges=np.histogram2d(z,Mstar,bins=100,range=[(0, 3.0), (8.2, 11.7)])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    sc=plt.imshow(H.T, extent=extent, interpolation='nearest',cmap=cm.Greys,vmin=0,vmax=vmax_val)
#   plt.colorbar(sc,shrink=0.9,pad=0.05)
    for zl in slices:
        z1= ms(mes,zl,'z1')
        z2= ms(mes,zl,'z2')
        print z1,z2
        Mlim=np.sort(ms(mes,zl,'Mlim'))

        plt.plot([z1[0],z1[0]],[np.min(Mlim),np.max(Mlim)],lw=3,color='green')
        plt.plot([z2[0],z2[0]],[np.min(Mlim),np.max(Mlim)],lw=3,color='green')
        for m in Mlim[:-1]:
            plt.plot([z1[0],z2[0]],[m,m],lw=3,color='green')
    plt.plot(slices,limits,lw=3,color='red')
    ax2 = fig.add_axes([.58, .18, .225, .225])
    ax2.tick_params(axis='both', length=7, width=1, which='major', labelsize=10)
    ax2.tick_params(axis='both', length=5, width=0.5, which='minor', labelsize=10)
    ax2.minorticks_off()
    ax2.set_xlim(0.0,3.0)
    ax2.set_ylim(0,histmax_val)
    ax2.set_xlabel(r'$z_\mathrm{phot}$',size=12)
    ax2.set_ylabel(r'$N$',size=12)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    [i.set_linewidth(1) for i in ax2.spines.itervalues()]
    ax2.hist(z,range=[0,3.0],bins=15,histtype='step',lw=2,align='mid',color='black')
    plt.savefig(figures+figname)
    plt.close() 

def running_median_slow(n, iterable):
   'Slow running-median with O(n) updates where n is the window size'
   it = iter(iterable)
   queue = deque(islice(it, n))
   sortedlist = sorted(queue)
   midpoint = len(queue) // 2
   yield sortedlist[midpoint]
   for newelem in it:
      oldelem = queue.popleft()
      sortedlist.remove(oldelem)
      queue.append(newelem)
      insort(sortedlist, newelem)
      yield sortedlist[midpoint]

# from these two, do the abundance matching: return 
# for a given mass limit the corresponding 
# halo mass. 

def abundance_match(mass_sel_lin,abundance_obs,vmax_sel_simu,abundance_sim,
                    data,z1,z2,do_plots):
   # first get a function which can return for a given abundance, what is 
   # the value of vmax log
   func_sim = interp1d(np.log10(np.sort(abundance_sim)), \
                       np.log10(vmax_sel_simu[np.argsort(abundance_sim)]))
   sel=(abundance_obs<np.max(abundance_sim)) & (abundance_obs > np.min(abundance_sim))
# we can only do this in overlapping ranges    
   log_vmax_obs = func_sim(np.log10(abundance_obs[sel]))
#   log_vmax_simu = data['lvmax']
#   lm200 = data['lm200'] 
   log_vmax_simu = data['lvmax']
   lm200 = data['lm200'] 

   lm200_sorted = lm200[np.argsort(lm200)]
   ymed=np.array(list(running_median_slow(101, lm200_sorted)))
#   ymed=np.array(list(running_median.RunningMedian(51, lm200_sorted)))
#   xmed=np.sort(data['lvmax'])[50:-50]
   xmed=np.sort(data['lvmax'])[50:-50]
   max_simu = np.max(xmed)
   # the simulation does not have enough massive haloes at high redshift
   func_vmax_lm200 = interp1d(xmed,ymed)
   lm200_fromobs = func_vmax_lm200(log_vmax_obs[log_vmax_obs<max_simu])
   mstar = mass_sel_lin[log_vmax_obs<max_simu]
   mh = func_vmax_lm200(log_vmax_obs[log_vmax_obs<max_simu]) 
   if (do_plots):
    figname='abundances_{}.pdf'.format(str(z1).replace('.','_'))
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_xlim(8,15)
    ax.set_ylim(0.5,3.0)
    figname='vmax_l200_{}.png'.format(str(z1).replace('.','_'))
    ax.plot(lm200,log_vmax_simu,'.',markersize=1.5,markevery=4)
    ax.plot(ymed,xmed,lw=3,c='r')
    ax.set_xlabel(r'$\mathrm{log} M_{200}$')
    ax.set_ylabel(r'$\mathrm{log} V_{\mathrm{max}}$')
    plt.savefig(figname)
    plt.close('all')
   return (mstar,mh)


def abundance_match_m200(mass_sel_lin,abundance_obs,m200_sel_simu,abundance_sim):
   # first get a function which can return for a given abundance, what is 
   # the value of m200 log
   func_sim = interp1d(np.log10(np.sort(abundance_sim)), \
                       np.log10(m200_sel_simu[np.argsort(abundance_sim)]))
   # m200 values corresponding to observations
   sel=(abundance_obs<np.max(abundance_sim)) & (abundance_obs > np.min(abundance_sim))
   log_m200_obs = func_sim(np.log10(abundance_obs[sel]))
   mstar = mass_sel_lin[sel]
   mh = log_m200_obs 
   return (mstar,mh)

def do_abundance_match(filename_simu,z1,z2,z,do_bolshoi,do_plots):
    (mass_sel_lin,abundance_obs)=abundance_data_MF(mf_data[z])
    if (do_bolshoi):
        data_bolshoi=np.load(filename_simu)
        (m200_sel_simu,abundance_sim)=abundance_simu_bolshoi(data_bolshoi)
        (mstar,mh) = abundance_match_m200(mass_sel_lin,abundance_obs,
                                          m200_sel_simu,abundance_sim)
    else:
        data = {}
        data['lm200'], data['lmvir'],data['lvmax'] =  np.loadtxt(filename_simu,unpack=True)
        (vmax_sel_simu,abundance_sim)=abundance_simu_seb(data)
        (mstar,mh) = abundance_match(mass_sel_lin,abundance_obs,vmax_sel_simu,abundance_sim, 
                                 data,z1,z2,do_plots)

#    (m200_sel_simu,abundance_sim)=abundance_simu_seb_m200(data)
#    (mass_sel_lin,abundance_obs)=abundance_data(z1,z2)
#    if (do_plots):
#        figname='abundances_{:2.1f}_{:2.1f}.pdf'.format(z1,z2)
#        plot_abundances_m200(figname,m200_sel_simu,abundance_sim,mass_sel_lin,
#                        abundance_obs,z1,z2)
#        figname='mstar_mh_{:2.1f}_{:2.1f}.pdf'.format(z1,z2)
#        plot_mstar_mh(figname,mstar,mh,z1,z2)
    return (mstar,mh)

def abundance_simu_bolshoi(data):
 # apply h-factor 
   m200= data['m200'] / 0.677
   m200_sel_simu = np.logspace(8.4,np.max(np.log10(data['m200'])-0.1))
 # apply h-factor for box 
   abundance_sim = np.array([np.size(m200[m200>x]) 
                            for x in m200_sel_simu]) / (250.0/0.677)**3   
   return(m200_sel_simu,abundance_sim)
# returns abundance of haloes above vmax 

def abundance_simu_seb(data):
   vmax=10**data['lvmax']
   vmax_sel_simu = np.logspace(np.min(data['lvmax']),np.max(data['lvmax'])-0.1)
   abundance_sim = np.array([np.size(vmax[vmax>x]) 
                            for x in vmax_sel_simu]) / (80.0/0.677)**3   
   return(vmax_sel_simu,abundance_sim)

def abundance_simu_seb_m200(data):
   m200=10**data['lm200']
   m200_sel_simu = np.logspace(np.min(data['lm200']),np.max(data['lm200'])-0.1)
   abundance_sim = np.array([np.size(m200[m200>x]) 
                            for x in m200_sel_simu]) / (80.0/0.677)**3   
   return(m200_sel_simu,abundance_sim)

# returns abundance of galaxies above mass selection limit. 
def abundance_data(z1,z2):
   pz = np.load('/Users/hjmcc/am/photozed.npz')
   cosmo = FlatLambdaCDM(H0=67.7, Om0=0.3)
   vol = (cosmo.comoving_volume(z2)-cosmo.comoving_volume(z1)) * (1.0/(4.0*np.pi*(180.0/np.pi)**2))*1.3
   sel = (pz['photoz']>z1) & (pz['photoz']<z2) & (pz['type'] == 0)
   mass_sel = pz['lmass'][sel]
   mass_sel_lin = np.linspace(np.min(mass_sel),np.max(mass_sel)-0.1)
   abundance_obs = np.array([np.size(mass_sel[mass_sel>x]) for x in mass_sel_lin])/vol.value
   return(mass_sel_lin,abundance_obs)


def abundance_data_MF(mfpar):
# calculate the abundance in the data by integrating 
# the mass function     
    Mstar=10**mfpar.Mstar
    phi1=1e-3*mfpar.phi1
    phi2=1e-3*mfpar.phi2
    phiMdm = lambda M : np.exp(-1.0*M/Mstar)*(phi1*(M/Mstar)**mfpar.alpha1+phi2*(M/Mstar)**mfpar.alpha2)/Mstar
    mass_sel_lin = np.linspace(mfpar.mlim_min,13.0)
    abundance_obs = np.array([quad(phiMdm,10**ml,10**13)[0] for ml in mass_sel_lin])
    return(mass_sel_lin,abundance_obs)

def plot_abundances_m200(figname,m200_sel_simu,abundance_sim,mass_sel_lin,abundance_obs,z1,z2):
    fig=plt.figure(figsize=(8,8))
    fig, ((ax1, ax2)) = plt.subplots(1, 2, sharex=False, sharey=True)
    ax1.tick_params(axis='both', which='major', labelsize=15, width=2.0,size=10)
    ax1.tick_params(axis='both', which='minor', labelsize=15, width=1,size=5)
    ax2.tick_params(axis='both', which='major', labelsize=15, width=2.0,size=10)
    ax2.tick_params(axis='both', which='minor', labelsize=15, width=1,size=5)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-6,100)
    ax1.set_xlim(9,15)
    ax1.set_ylabel(r'$N(>M_{200}) \mathrm{Mpc}^{-3}$') 
    ax1.set_xlabel(r'$\mathrm{log} M_{200}$') 
    ax1.plot(np.log10(m200_sel_simu),abundance_sim,lw=2)
    ax2.yaxis.set_label_position('right')
    ax2.set_ylabel(r'$N(>M^*) \mathrm{Mpc}^{-3}$')
    ax2.set_xlabel(r'log M*')
    ax2.set_xlim(7,13)
    ax2.set_ylim(1e-5,10)
    ax2.set_yscale('log')
    ax2.plot(mass_sel_lin,abundance_obs,lw=2)
    fmt = r'$%5.2f < z < %5.2f$' % (z1,z2)
    ax1.annotate(fmt, xy=(0.3, 0.2), xycoords="figure fraction",
                 size=15,va="center", ha="center",bbox='')
    ax2.annotate(fmt, xy=(0.7, 0.2), xycoords="figure fraction",
                 size=15,va="center", ha="center",bbox='')

    fig.savefig(figname)
    plt.close('all')


def plot_mstar_mh(figname,mstar,mh,z1,z2):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_xlim(9,14)
    ax.set_ylim(-3,0)
    ax.set_ylabel(r'$M^*/M_h$') 
    ax.set_xlabel(r'$\mathrm{log} M_h~(M_\odot)$') 
    ax.plot(mh,mstar-mh,lw=2)
    fmt = r'$%5.2f < z < %5.2f$' % (z1,z2)
    ax.annotate(fmt, xy=(0.3, 0.2), xycoords="figure fraction",
                 size=15,va="center", ha="center",bbox='')
    fig.savefig(figname)
    plt.close('all')


def plot_abundances_vmax(figname,vmax_log,abundance_sim,mass_sel_lin,abundance_obs,z1,z2):
    fig=plt.figure(figsize=(8,8))
    fig, ((ax1, ax2)) = plt.subplots(1, 2, sharex=False, sharey=True)
    ax1.tick_params(axis='both', which='major', labelsize=15, width=2.0,size=10)
    ax1.tick_params(axis='both', which='minor', labelsize=15, width=1,size=5)
    ax2.tick_params(axis='both', which='major', labelsize=15, width=2.0,size=10)
    ax2.tick_params(axis='both', which='minor', labelsize=15, width=1,size=5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-6,100)
    ax1.set_ylabel(r'$N(>V_{\mathrm {max}}) \mathrm{Mpc}^{-3}$') 
    ax1.set_xlabel(r'Vmax') 
    ax1.plot(vmax_log,abundance_sim,lw=2)
    ax2.set_xscale('linear')
    ax2.set_xlabel(r'log M*')
    ax2.set_ylim(1e-5,100)
    ax2.set_yscale('log')
    ax2.plot(mass_sel_lin,abundance_obs,lw=2)
    fmt = r'$%5.2f < z < %5.2f$' % (z1,z2)
    ax1.annotate(fmt, xy=(0.3, 0.2), xycoords="figure fraction",
                 size=15,va="center", ha="center",bbox='')
    ax2.annotate(fmt, xy=(0.7, 0.2), xycoords="figure fraction",
                 size=15,va="center", ha="center",bbox='')

    fig.savefig(figname)
    plt.close('all')



def plot_Mpeak_z(figname,result_fit):
    # plot the location of the peak as a function of redshift. 
    slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(1,1,1)
    ax.set_xlim(0.0,2.75)
#   ax.set_yscale('log')
    ax.set_ylim(11.35,12.85)
    plt.xlabel(r'$z_\mathrm{phot}$')
    plt.ylabel(r'$\mathrm{log}~M_\mathrm{peak} (M_\odot)$')
    zj, lMt_j, lMt_j_err,alpha,alpha_err=np.loadtxt('/Users/hjmcc/Dropbox/uvista/clustering-dr1/table_jean.txt',\
        usecols=(0,5,6,7,8),unpack='True',comments='#')
    lMt_j = lMt_j-np.log10(0.7)
    lmp =  lMt_j-np.log10(1.0-alpha)
    lalpha_err = np.log(10)*alpha_err/alpha 
    Mt_j_err = lMt_j_err * 10**lMt_j/np.log(10)
#    err_lmp = np.sqrt(lMt_j_err**2+lalpha_err**2)
    err_lmp = np.sqrt(lMt_j_err**2)
    plt.plot(1.5,12.44,
             markersize=12, mew=1,linestyle='',label='Martinez-Manso et al. 2014', 
             color='black', marker = (3,0,0))
    ax.errorbar(1.5,12.44,yerr=0.06,fmt='',linestyle='None', color='black', \
        alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0)
    zmjh, lM_mjh, lM_mjh_err,=np.loadtxt('/Users/hjmcc/Dropbox/uvista/clustering-dr1/mjh_table.txt',\
        usecols=(0,1,2),unpack='True',comments='#')
    plt.plot(zmjh,lM_mjh,
             markersize=12, mew=1,linestyle='',label='Hudson et al. 2014', 
             color='black', marker = (4,2,45))
    ax.errorbar(zmjh,lM_mjh,yerr=lM_mjh_err,fmt='',linestyle='None', color='black', \
        alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0)

    plt.plot(zj+0.1,lmp,
             markersize=12, mew=1,linestyle='',label='Coupon et. al. 2012', 
             color='black', marker = (4,1,0))

    ax.errorbar(zj+0.1,lmp,yerr=err_lmp,fmt='',linestyle='None', color='black', \
        alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0)

    zj, lMt_j, lMt_j_err1,lMt_j_err2 = np.loadtxt('/Users/hjmcc/Dropbox/uvista/clustering-dr1/table-alexie.txt',\
        usecols=(0,1,2,3),unpack='True',comments='#')

    plt.plot(zj,lMt_j,
             markersize=12, mew=1,linestyle='',label='Leauthaud et al. 2011', 
             color='black', marker = 'o')

    ax.errorbar(zj,lMt_j, yerr=lMt_j-lMt_j_err1,fmt='',linestyle='None', color='black', \
        alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0)

    for z in slices:
        peak_am =  np.log10(result_fit[2][z]['M1'].value)
        plt.plot(z+0.02,peak_am,\
            marker=syms_zed[str(z)][0], color='white',markersize=15, 
            mew=1.5, linestyle='', alpha=1.0, label='',mec='red')

        logerr_am = result_fit[2][z]['M1'].stderr/result[2][z]['M1'].value*0.434
        logerr_am = 0.15
        ax.errorbar(z+0.02,peak_am,yerr=logerr_am,fmt='',
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0)
        if z not in ([1.3, 2.25]):
            peak =  np.log10(result_fit[1][z]['M1'].value)
            plt.plot(z,peak,\
                     marker=syms_zed[str(z)][0], color=syms_zed[str(z)][1],markersize=15, 
                     mew=1.5, linestyle='', alpha=1.0, label='')
            logerr_value = result_fit[1][z]['M1'].stderr/result_fit[1][z]['M1'].value*0.434
            ax.errorbar(z,peak,yerr=logerr_value,fmt='',xerr=0.1,
            linestyle='None', color='black', alpha=1,elinewidth=0.8,capthick=1.25, capsize=0.0)
# now plot a few results from the literature 
    ax.legend(loc='upper right',numpoints=1,fancybox=False, prop={'size':15})
    plt.savefig(figures+figname)
    plt.close() 

#figures = '/Users/hjmcc/Dropbox/tex-melody/SHMR_Ultravistapaper/figs-hjmcc/'
#figures = '/Users/hjmcc/tex/uvista/melody/figs-hjmcc/'
#figures = '/Users/hjmcc/cosmos/uvista/melody/'
parser = argparse.ArgumentParser()
parser.add_argument('-w','--wtheta_files',
            dest='wtheta_files',help='input wtheta files', default='wtheta_list_all.txt')
parser.add_argument('-wp','--wtheta_passive_files',
            dest='wtheta_passive_files',help='input wtheta passivefiles', default='wtheta_passive.txt')
parser.add_argument("-g", "--varygamma", dest='varygamma',help="vary gamma",action="store_true")
parser.add_argument('-c','--intconst',
                    dest='intconst',help='integral constraint', type=float, default=1.43)
parser.add_argument("-v", "--verbose", dest = 'verbose', help="output fit",
                    action="store_true")



cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
args = parser.parse_args()
mes = {}
mes_passive = {}


for l in open(args.wtheta_files):
    try:
        head = re.findall(r"[\w']+",l)[0]
        Mlim,z1,z2 = os.path.basename(l.strip()).split('wpmc_')[1].split('_')
        try:
            wt = open(l.strip(),'r')
            mes[os.path.basename(l.strip())]=MeasurementBin(wt)

        except IOError:
            print "Cannot open", l.strip()

    except IOError:
        print "cannot open", args.wtheta_files
        sys.exit(1)

        
for l in open(args.wtheta_passive_files):
    try:
        head = re.findall(r"[\w']+",l)[0]
        Mlim,z1,z2 = os.path.basename(l.strip()).split('wpmc_')[1].split('_')
        try:
            wt = open(l.strip(),'r')
            mes_passive[os.path.basename(l.strip())]=MeasurementBin(wt)

        except IOError:
            print "Cannot open", l.strip()

    except IOError:
        print "cannot open", args.wtheta_passive_files
        sys.exit(1)

fit_wtheta(mes,9,14,args.varygamma)
fit_wtheta(mes_passive,9,14,args.varygamma)
# these are the values used for the paper plots, i.e., large scale fits. 


#fit_wtheta(mes,2,14,args.varygamma)
#fit_wtheta(mes_passive,2,14,args.varygamma)


#fit_r0(mes)
#fit_r0(mes_passive)

marker = itertools.cycle(('o', 'v', '>', 's', 'p', 'd', '*'))
syms = {}
syms_zed = {}
#cm=cm.get_cmap('YlGnBu')
cm_in=cm.get_cmap('Paired')
#cm_in=cm.get_cmap('Accent')
for index,i in enumerate(np.arange(9.0,11.2,0.2)):
    syms[str(i)]=marker.next(),cm_in(1.*index/20)

syms_zed = {}
slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
for index,i in enumerate(slices):
    syms_zed[str(i)]=marker.next(),cm_in(1.*index/10)


figures = '/Users/hjmcc/junk/test-figs/'
#figures = '/Users/hjmcc/tex/uvista/melody/figs-hjmcc/'
#figures = '/Users/hjmcc/Dropbox/tex-melody/SHMR_Ultravistapaper/figs-hjmcc/'

#figures='/Users/hjmcc/Dropbox/tex-melody/SHMR_Ultravistapaper/figs-hjmcc/'
slices=np.unique([mes[i].z for i in mes])
slices = np.array([0.65,0.95,1.3,1.75,2.25])
#slices = np.array([ 0.65,0.95])
# pool=multiprocessing.Pool(6)
# for z in slices:
#     print 'Doing redshift', z 
#     Mlim= ms(mes,z,'Mlim')
#     M_med= ms(mes,z,'M_med')
#     z_med= ms(mes,z,'z_med')
#     n_g= ms(mes,z,'n_g')
#     r0= ms(mes,z,'r0')
#     r0err= ms(mes,z,'r0err')
#     r0 =np.array([mes[i].r0 for i in mes if str(mes[i].z)==str(z)])
#     print Mlim,M_med,z_med,z,args.varygamma
# #    plot_wtheta('wtheta_full_{}.pdf'.format(str(z).replace('.','_')),mes,mes,z,0.65,haloplot=False)
# #    plot_wtheta('wtheta_passive_{}.pdf'.format(str(z).replace('.','_')),mes_passive,mes,z,0.65,haloplot=False)
# #    plot_wtheta('wtheta_hm_{}.pdf'.format(str(z).replace('.','_')),mes,mes,z,0.65)
#     pool.apply_async(plot_wtheta,args=('wtheta_hm_passive_{}.pdf'.format(str(z).replace('.','_')),mes_passive,mes,z,0.65,))
#     pool.apply_async(plot_wtheta,args=('wtheta_hm_full_{}.pdf'.format(str(z).replace('.','_')),mes,mes,z,0.65,))
#     pool.apply_async(plot_wtheta,args=('wtheta_passive_{}.pdf'.format(str(z).replace('.','_')),mes_passive,mes,z,0.65,False))
#     pool.apply_async(plot_wtheta,args=('wtheta_full_{}.pdf'.format(str(z).replace('.','_')),mes,mes,z,0.65,False))
# pool.close()
# pool.join()

#figures = '/Users/hjmcc/Dropbox/tex-melody/SHMR_Ultravistapaper/figs-hjmcc/'<
figures = '/Users/hjmcc/junk/figs-hjmcc/'

#plot_M1_Mmin('M1_min.pdf',mes,mes_passive)
mes=compute_r0_Mstar_hm(mes,'./all')
#mes_passive=compute_r0_Mstar_hm(mes_passive,'./passive')

#plot_Ngal('logNgal_Mmin.pdf',mes,mes_passive)
#plot_Mmin_Mstar('Mmin_Mstar.pdf',mes,mes_passive)
plot_lMstar_frsat('frsat-Mstar.pdf',mes,mes_passive)
#plot_redfrac_ngal('redfrac.pdf',mes,mes_passive)
#plot_r0(mes,'Mstar_r0hm.pdf')
#ns_sf = compute_M(10.6,1.1,0.22,1.13,-1.39,12.0)[0]
#ns_qu = compute_M(10.9,1.32,-0.67,0.03,-1.5,12.0)[0]
limits_full = np.array([8.70,9.13,9.42,9.67,10.04])
limits_passive = np.array([8.96,9.37,9.60,9.87,10.11])
plot_Mstar_z('Mstarz_passive.pdf','/Users/hjmcc/Dropbox/uvista/clustering-dr1/goodVistapassive.gz',\
  mes_passive,20,10000,limits_passive)
plot_Mstar_z('Mstarz_full.pdf','/Users/hjmcc/Dropbox/uvista/clustering-dr1/goodVistacutKlt24.gz',
  mes,100,30000,limits_full)


#result=fit_mmin_mstar(mes);
#for z in slices:
#  plot_Mstar_over_Mh_withmodel('testmh'+str(z)+'.png',mes,result,z)

##do the abundance matching measuremnts. 
#results_am={}
#results_am_bolshoi={}
#data_dir = '/Users/hjmcc/am/'
#slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
#slices_a = np.array(['0.60430', '0.51630','0.43230','0.36630','0.30630'])

#mf_data={}
#with open('/Users/hjmcc/Dropbox/uvista/clustering-dr1/MF-full-ilbert2013.txt') as infl:
#    for line in infl:
#        a=[float(i) for i in line.strip().split()]
#        mf_data[a[0]] = MFpar(a[1],a[2],a[3],a[4],a[5],a[6])

#slices = np.array([ 0.65,  0.95,  1.3 ,  1.75,  2.25])
#z1 = np.array([ 0.5, 0.8, 1.1 , 1.5,  2.0])
#z2 = np.array([ 0.8,1.1,  1.5 , 2.0,  2.5])
#for i,z in enumerate(slices):
#   filename_seb = data_dir + "data_"+str(z)+'.asc.gz'
#   filename_bolshoi = data_dir + "hlist_"+(slices_a[i])+'.list.npz'
#   z_pr = 1.0/np.float(slices_a[i]) - 1.0 
#   print "doing redshift",z 
#   results_am[z]=do_abundance_match(filename_seb,float(z1[i]),float(z2[i]),z,
#                                    do_bolshoi=False,do_plots=False)
#   results_am_bolshoi[z]=do_abundance_match(filename_bolshoi,float(z1[i]),float(z2[i]),z,
#                                            do_bolshoi=True,do_plots=False)

#result=fit_mmin_mstar(mes,results_am);
#result_bolshoi=fit_mmin_mstar(mes,results_am_bolshoi);    
#plot_Mstar_over_Mh_withmodel_onepanel('Mstar_over_Mh_withmodel_onepanel-both.pdf',
#                                     mes,result,result_bolshoi,limits_full)




#plot_Mpeak_z('Mpeak_versus_z.pdf',result)
#mes=compute_r0_Mstar_error(mes,'./all')
#mes=compute_r0_Mstar_hm(mes,'./all')
#mes=compute_r0_Mstar_error(mes,'./all')
#plot_r0(mes,'Mstar_r0hm.pdf')
