#!/usr/bin/env python3
# -*-coding:Utf-8 -*

from scipy.integrate import quad
import matplotlib.gridspec as gridspec
import numpy as np

"""Iari Davidzon et al 2017 Shchechter fit parameters"""

#for 0.2<z<0.5
# ms = 10**(10.78)
# alpha1 = -1.38
# phi1s= 1.187e-3
# alpha2 = -0.43
# phi2s= 1.92e-3


# Parameters for the Schechter function for each redshift bin :
# log(Mstar), alpha1, phistar1, alpha2, phistar2


dav_param = np.array([
                [10.78, -1.38, 1.187e-3, -0.43, 1.92e-3],
                [10.77, -1.36, 1.070e-3, 0.03, 1.68e-3],
                [10.56, -1.31, 1.428e-3, 0.51, 2.19e-3],
                [10.62, -1.28, 1.069e-3, 0.29, 1.21e-3],
                [10.51, -1.28, 0.969e-3, 0.82, 0.64e-3],
                [10.60, -1.57, 0.295e-3, 0.07, 0.45e-3],
                [10.59, -1.67, 0.228e-3, -0.08, 0.21e-3],
                [10.83, -1.76, 0.090e-3, 0, 0],
                [11.10, -1.98, 0.016e-3, 0, 0],
                [11.30, -2.11, 0.003e-3, 0, 0]])

dav_param_plus = np.array([
                [0.13, 0.08, 0.633e-3, 0.62, 0.73e-3],
                [0.09, 0.05, 0.287e-3, 0.43, 0.33e-3],
                [0.05, 0.05, 0.306e-3, 0.35, 0.40e-3],
                [0.08, 0.05, 0.222e-3, 0.40, 0.23e-3],
                [0.08, 0.06, 0.202e-3, 0.48, 0.18e-3],
                [0.15, 0.12, 0.173e-3, 0.70, 0.12e-3],
                [0.36, 0.26, 0.300e-3, 1.73, 0.14e-3],
                [0.15, 0.13, 0.064e-3, 0, 0],
                [0.21, 0.14, 0.020e-3, 0, 0],
                [1.22, 0.34, 0.002e-3, 0, 0]])

dav_param_minus = np.array([
                [0.14, 0.25, 0.969e-3, 0.60, 0.78e-3],
                [0.08, 0.06, 0.315e-3, 0.43, 0.33e-3],
                [0.05, 0.06, 0.308e-3, 0.34, 0.41e-3],
                [0.07, 0.05, 0.240e-3, 0.42, 0.22e-3],
                [0.07, 0.06, 0.208e-3, 0.52, 0.17e-3],
                [0.12, 0.21, 0.177e-3, 0.74, 0.12e-3],
                [0.36, 0.26, 0.300e-3, 1.73, 0.38e-3],
                [0.15, 0.11, 0.039e-3, 0, 0],
                [0.21, 0.13, 0.009e-3, 0, 0],
                [1.22, 0.22, 0.002e-3, 0, 0]])


def phi(M, dav_par):
    """Schechter function"""
    # return (phis1*(M/ms)**alpha1+phis2*(M/ms)**alpha2)*np.exp(-M/ms)*1/ms
    return (dav_par[2] * (M/10**dav_par[0])**dav_par[1] +
        dav_par[4] * (M/10**dav_par[0])**dav_par[3]) * np.exp(-M/10**dav_par[0])*1/10**dav_par[0]


def phi_one(M, dav_par):
    """Schechter function for z>2.5 (one power law)"""
    return dav_par[2] * (M/10**dav_par[0])**dav_par[1] * np.exp(-M/10**dav_par[0])*1/10**dav_par[0]


def delta_phi(M, delta_M, dav_par, delta_par):
    """Compute differential uncertainties for phi for all parameters"""
    logMs = dav_par[0]
    Ms = 10**logMs
    alpha1 = dav_par[1]
    alpha2 = dav_par[3]
    phi1s = dav_par[2]
    phi2s = dav_par[4]
    delta_logMs = delta_par[0]
    delta_alpha1 = delta_par[1]
    delta_alpha2 = delta_par[3]
    delta_phi1s = delta_par[2]
    delta_phi2s = delta_par[4]

    phi1 = delta_phi1s*abs((M/Ms)**alpha1 * np.exp(-M/Ms) *1/Ms) # dphi/dphi1s
    phi2 = delta_phi2s*abs((M/Ms)**alpha2 * np.exp(-M/Ms) *1/Ms) # dphi/dphi2s
    phi3 = delta_alpha1* abs(phi1s * np.log(M/Ms) * (M/Ms)**alpha1 *
        np.exp(-M/Ms) *1/Ms) #dphi/dalpha1
    phi4 = delta_alpha2* abs(phi2s * np.log(M/Ms) * (M/Ms)**alpha2 *
        np.exp(-M/Ms) *1/Ms) #dphi/dalpha2

    phi5 = delta_M * abs(
        (phi1s*alpha1/Ms*(M/Ms)**(alpha1-1) +
        phi2s*alpha2/Ms*(M/Ms)**(alpha2-1)) * np.exp(-M/Ms)*1/Ms -
        (phi1s*(M/Ms)**alpha1 +phi2s*(M/Ms)**alpha2) * np.exp(-M/Ms)*1/Ms**2 )

    phi6 = delta_logMs*Ms*np.log(10)*np.exp(-M/Ms)* abs(
        (-phi1s*alpha1*M/Ms**2 * (M/Ms)**(alpha1-1) -
        phi2s*alpha2*M/Ms**2 * (M/Ms)**(alpha2-1)) +
        (phi1s*(M/Ms)**alpha1 + phi2s*(M/Ms)**alpha2)*(M/Ms**3 - 1/Ms**2))

    return phi1+phi2+phi3+phi4+phi5+phi6
    #return sqrt(phi1**2 + phi2**2 + phi3**2 + phi4**2 + phi5**2 + phi6**2)

def wrap_delta_phi(M, args):
    return delta_phi(M, args[0], args[1:6], args[6:11])

def wrap_delta_phi_one(M, args):
    return delta_phi_one(M, ...)

"""Compute integral for number density > M """


def number_density(param):
    param['n_int'] = 1000
    x = np.linspace(10**param['mmingal'], 10**param['mmaxgal'], param['n_int'])
    xlog = np.logspace(param['mmingal'], param['mmaxgal'], param['n_int'])

    Number_density = np.empty([param['numzbin'], param['n_int']])
    Number_density_plus = np.empty([param['numzbin'], param['n_int']])
    Number_density_minus = np.empty([param['numzbin'], param['n_int']])
    delta_Number_density = np.zeros([param['numzbin'], param['n_int']])

    for i in range(param['numzbin']-3):
        print('Compute double powlaw : '+ str(i))
        for j in range(param['n_int']):

            Number_density[i, j] = quad(phi, xlog[j], xlog.max(), args=(dav_param[i]))[0]
            # Number_density_plus[i,j] = quad(phi, x[j], x.max(), args=(dav_param[i]+dav_param_plus[i]))[0]
            # Number_density_minus[i,j] =  quad(phi, x[j], x.max(), args=(dav_param[i]-dav_param_minus[i]))[0]
            # args = np.concatenate(([0],dav_param[i][:], dav_param_plus[i][:] + dav_param_minus[i][:]) )
            # delta_Number_density[i,j] = quad(wrap_delta_phi, xlog[j], xlog.max(), args=args)[0]

    for i in param['numzbin']-3 + np.array([0,1,2]):
        print('Compute simple powlaw: '+ str(i))
        for j in range(param['n_int']):
            Number_density[i,j] = quad(phi_one, xlog[j], xlog.max(),
                args=(dav_param[i]), points=[xlog[j], (xlog[j]+2*xlog.max())/3])[0]
            # Number_density_plus[i,j] = quad(phi_one, x[j], x.max(), args=(dav_param[i]+dav_param_plus[i]))[0]
            # Number_density_minus[i,j] =  quad(phi_one, x[j], x.max(), args=(dav_param[i]-dav_param_minus[i]))[0]
    
    return Number_density, xlog

"""Plot"""

### Only for 0.2<z<0.5, after running AbundanceGalaxies.py
# plt.loglog(massbinlog, Ngallog, label='J. Coupon Catalog, 0.2<z<0.5')
# plt.ylabel('log( N(>m) $Mpc^{-3})$')
# plt.xlabel('$log( M_{*} / M_{\odot})$')
# plt.title('Fit of I.Davidzon on J.Coupon, 0.2<z<0.5')
# plt.loglog(x,I, label='I.Davidzon Schechter fit, 0.2<z<0.5')
# plt.legend()
# plt.ylim(10**-7, 10**-1)
# plt.show()

#Plot all the Iari davidzon fits
# 
# fig, ax = plt.subplots(5, 2, sharex='col', sharey='row', figsize=(6, 8))
# fig.subplots_adjust(hspace=0, wspace=0)
# 
# for i in range(numzbin):
# 
#     #ax1 = ax[int(i/2), i%2]
#     fig, ax1 = plt.subplots(1)
#     ax1.set_xscale('log')
#     ax1.set_yscale('log')
#     # ax1.fill_between(x, Number_density_plus[i], Number_density_minus[i], color='grey')
# 
#     # ax1.scatter(10**np.linspace(mmingal, mmaxgal, num=n),Ngal[i], marker='+',
#     #      color='red')
# 
#     ax1.plot(xlog, Number_density[i])
# 
#     # ax1.errorbar(xlog[::100], Number_density[i][::100],
#     #     yerr=delta_Number_density[i][::100], fmt='.')
# 
# 
#     # ax1.errorbar(xlog[::100], Number_density[i][::100],
#     #     yerr=delta_phi(xlog[::100], 0, dav_param[i], dav_param_plus[i] + dav_param_minus[i]), fmt='.')
# 
#     ax1.set_xlim(10**8, 10**12)
#     ax1.set_ylim(10**-8, 1)
#     ax1.text(0.1,0.1, str(redshifts[i])+'<z<'+str(redshifts[i+1]), fontsize=10,
#         transform=ax1.transAxes)
#     if i%2==0 :
#         ax1.set_ylabel('N(>m), $Mpc^{-3}$')
#     if i==8 or i==9:
#         ax1.set_xlabel('Stellar Mass, $log( M_{*} / M_{\odot})$')
# 
# fig.tight_layout()
# fig.suptitle('Fit of I. Davidzon Schechter parameters on Jean Coupon catalog')
# plt.subplots_adjust(top=0.95)
# plt.show()
