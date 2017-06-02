# Neuro Chem
from ase_interface import ANI
import pyNeuroChem as pync

import numpy as np
import nmstools as nm
import hdnntools as hdt
import random as rn


from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import LBFGS
from ase.vibrations import Vibrations

xyz = np.array([[-0.0000059, 0.0000036, -0.0000068],
                [-0.8073286, 0.6497303, -0.3458662],
                [-0.4167022, -0.9545966, 0.3296326],
                [0.5200714,  0.4776208,  0.8336411],
                [0.7039948, -0.1727763, -0.8173668]],dtype=np.float32)

spc = ['C',
       'H',
       'H',
       'H',
       'H',]

# Set required files for pyNeuroChem
anipath  = '/home/jujuman/Dropbox/ChemSciencePaper.AER/ANI-c08e-ntwk'
cnstfile = anipath + '/rHCNO-4.6A_16-3.1A_a4-8.params'
saefile  = anipath + '/sae_6-31gd.dat'
nnfdir   = anipath + '/networks/'

# Construct pyNeuroChem class
nc = pync.molecule(cnstfile, saefile, nnfdir, 0)

mol = Atoms(spc, xyz, calculator = ANI(False))
mol.calc.setnc(nc)
LBFGS(mol).run(fmax=0.00001)
vib = Vibrations(mol)
vib.run()
vib.summary()

print(xyz)
xyz = mol.get_positions().copy()
print(xyz)

nm_cr = vib.modes[6:]

Nf = 3*len(spc)-6
nmo = nm_cr.reshape(Nf,len(spc),3)

fcc = np.array([1.314580,
                1.3147106,
                1.3149728,
                1.5161799,
                1.5164505,
                5.6583018,
                6.7181139,
                6.7187967,
                6.7193842])

gen = nm.nmsgenerator(xyz,nmo,fcc,spc,2000.0)

N = 2000
gen_crd = np.zeros((N, len(spc),3),dtype=np.float32)
for i in range(N):
    gen_crd[i] = gen.get_random_structure()

hdt.writexyzfile('pynmstesting.xyz',gen_crd,spc)