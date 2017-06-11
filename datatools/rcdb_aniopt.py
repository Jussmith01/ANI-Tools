import hdnntools as hdn

import os
import re

# Neuro Chem
from ase_interface import ANI
import pyNeuroChem as pync

from ase import Atoms
from ase.optimize import BFGS, LBFGS

def get_smiles(file):
    f = open(file,'r').read()
    r = re.compile('#Smiles:(.+?)\n')
    s = r.search(f)
    return s.group(1).strip()

# Set required files for pyNeuroChem
anipath  = '/home/jujuman/Dropbox/ChemSciencePaper.AER/networks/ANI-c08f-ntwk/'
cnstfile = anipath + '/rHCNO-4.6A_16-3.1A_a4-8.params'
saefile  = anipath + '/sae_6-31gd.dat'
nnfdir   = anipath + '/networks/'

idir = '/home/jujuman/Scratch/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_07/inputs/'
sdir = '/home/jujuman/Scratch/Research/GDB-11-wB97X-6-31gd/dnntsgdb11_07/inputs_new/'

# Construct pyNeuroChem class
nc = pync.molecule(cnstfile, saefile, nnfdir, 0)

files = [f for f in os.listdir(idir) if f.split(".")[1] == "ipt"]
for i,f in enumerate(files):
    data = hdn.read_rcdb_coordsandnm(idir + f)
    X = data['coordinates']
    S = data['species']

    mol = Atoms(positions=X, symbols=S)

    mol.set_calculator(ANI(False))
    mol.calc.setnc(nc)

    dyn = LBFGS(mol,logfile='optimization.log')
    dyn.run(fmax=0.00001,steps=1000)

    X = mol.get_positions()

    Nc = int(f.split(".")[0].split("-")[1])
    Fp = f.split("-")[0]
    smiles = get_smiles(idir+f)

    print(i,'of',len(files),':',f, dyn.get_number_of_steps(),Nc)
    hdn.write_rcdb_input(X,S,Nc,sdir,Fp,50,'wb97x/6-31g*','800.0', fill=4, comment='Smiles: ' + smiles)
