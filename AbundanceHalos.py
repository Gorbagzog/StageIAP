import pyfits
import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.cosmology import Planck15 as cosmo

#(id, mvir,rvir, vmax, mvir_all) =  np.loadtxt('../Data/hlist_0.66435.list',usecols=(1,10,11,16,36),comments='#',unpack=True)
#np.savez('halocat_0.66435.npz',id=id, mvir=mvir,rvir=rvir, vmax=vmax, mvir_all=mvir_all)


#data =  np.load('../Data/halocat_0.66435.npz') #for 0.3<z<0.7
data =  np.load('../Data/halocat_0.73635.npz') #for 0.2<z<0.5


n = 500
h = 0.7
V = 250*250*250 / h**3 # Mpc^3

logmvir = np.log10(data['mvir'] / h) #Replace h value and take log10
mminhalo = logmvir.min()
mmaxhalo= logmvir.max()

step = (mmaxhalo-mminhalo)/n
Nhalo = np.empty(n)
for i in range(n):
    "Compute the number of galaxies more massive than m for each mass bin"
    Nhalo[i] = np.sum(logmvir>(mminhalo+step*i)) / V

Nhalolog = np.empty(n)
massbinlog = np.logspace(mminhalo, mmaxhalo, num=n)
for i in range(n):
    "Compute the number of galaxies more massive than m for each mass bin"
    Nhalolog[i] = np.sum(logmvir>np.log10(massbinlog[i])) / V


# plt.plot(np.linspace(mminhalo, mmaxhalo, num=n),Nhalo)
# plt.title('Abundance for Berhoozi 0.66 halos')
# plt.xlabel('$log(M/M_{\odot})$')
# plt.ylabel('N(>M) $Mpc^{-3}$')
# plt.show()
#
plt.loglog(massbinlog, Nhalolog)
plt.ylabel('log( N(>m) $Mpc^{-3})$')
plt.xlabel('$log( M_{*} / M_{\odot})$')
plt.title('Abundance for Berhoozi 0.73 halos')
plt.show()
