import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import pyNeuroChem as pync
import pyanitools as pyt

import hdnntools as hdt

#traj = '/home/jujuman/Research/MD_TEST/traj.dat'
mdcr = '/home/jujuman/Research/MD_TEST/mdcrd.xyz'

wkdir    = '/home/jujuman/Gits/ANI-Networks/networks/ANI-c08f-ntwk/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile  = wkdir + 'sae_6-31gd.dat'
nnfdir   = wkdir + 'networks/'

ncr = pync.conformers(cnstfile, saefile, nnfdir, 0, False)

data = np.loadtxt(traj)

X, S, Na = hdt.readxyz2(mdcr)

print(X.shape)
print(data.shape)

#s = int(3380000/8)
#e = int(3420000/8)

s = int(3396000/8)
e = int(3400000/8)

X = X[s:e]
print(X.shape)

ncr.setConformers(confs=X.copy(), types=list(S))

conv_au_ev = 27.21138505
E1 = conv_au_ev*ncr.energy() / len(S)


#t = 0.25 * d# ata[:,0] / 1000
t = data[s:e,0]
T = data[s:e,1]
Ep = data[s:e,2]
Ek = data[s:e,3]
Et = data[s:e,4]
print(t,'ps')

plt.plot(t, T)

plt.xlabel('t (ps)')
plt.ylabel('Temp')
#plt.legend(bbox_to_anchor=(0.01, 0.9), loc=2, borderaxespad=0.)
plt.show()
#plt.savefig('foo.png', bbox_inches='tight')

plt.plot(t[0::1], Ep[0::1]-Ep[0], color='blue', label='Epot')
plt.plot(t[0::1], Ek[0::1]-Ek[0], color='green', label='Ekin')
plt.plot(t[0::1], Et[0::1]-Et[0], color='red', label='Etot')
plt.scatter(t[0::1], E1-E1[0], color='orange', label='Eani')

plt.xlabel('t (ps)')
plt.ylabel('E (eV)')
plt.legend(bbox_to_anchor=(0.01, 0.9), loc=2, borderaxespad=0.)
plt.show()
#plt.savefig('foo.png', bbox_inches='tight')