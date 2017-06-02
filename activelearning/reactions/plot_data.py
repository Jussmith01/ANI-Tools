# Import pyNeuroChem
import pyNeuroChem as pync
import numpy as np
import hdnntools as hdt
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

wkdir = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_rxns/benz_dbm_rxns_full/irc_dbm_benz_1/'
file = wkdir+'irc.dat'
Rc = np.load(wkdir+'reaction_coordinate.npz')

#xyz,typ,Na = hdt.readxyz2(file1)
xyz,typ,Ea = hdt.readncdat(file,type=np.float32)

# Set required files for pyNeuroChem
rcdir  = '/home/jujuman/Research/ANI-DATASET/RXN1_TNET/training/rxn1to6/ani_benz_rxn_ntwk/'
cnstfile = '../../rHCNO-4.6A_16-3.1A_a4-8.params'
saefile  = '../../sae_6-31gd.dat'

ncr = pync.conformers(rcdir + cnstfile, rcdir + saefile, rcdir + 'networks/', 1)

# Set the conformers in NeuroChem
ncr.setConformers(confs=xyz, types=list(typ))

# Compute Energies of Conformations
E1 = ncr.energy()

# Shift ANI E to reactant
E1 = E1[0:][::-1]
Ea = Ea[0:][::-1]
#x1 = np.linalg.norm(xyz[:,9,:] - xyz[0,9,:],axis=1)
#x2 = np.linalg.norm(xyz2[:,9,:] - xyz[0,9,:],axis=1)
#print(x2)

print(hdt.calculaterootmeansqrerror(hdt.hatokcal * E1,hdt.hatokcal * Ea))

plt.plot(Rc['x'][:,1], hdt.hatokcal * (E1-E1[0]),color='blue',linewidth=3)
plt.plot(Rc['x'][:,1], hdt.hatokcal * (Ea-Ea[0]),'r--',color='black',linewidth=3)

plt.show()
