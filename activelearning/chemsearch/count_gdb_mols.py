# Import pyNeuroChem
from __future__ import print_function

# Neuro Chem
from ase_interface import ANI
import pyNeuroChem as pync

import hdnntools as gt
import numpy as np
import matplotlib.pyplot as plt
import time as tm
from scipy import stats as st
import time

import hdnntools as hdt

from rdkit import Chem
from rdkit.Chem import AllChem

# ASE
import  ase
from ase.io import read, write
from ase.optimize import BFGS, LBFGS
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
from ase.units import Bohr
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms

def formatsmilesfile(file):
    ifile = open(file, 'r')
    contents = ifile.read()
    ifile.close()

    p = re.compile('([^\s]*).*\n')
    smiles = p.findall(contents)

    ofile = open(file, 'w')
    for mol in smiles:
        ofile.write(mol + '\n')
    ofile.close()

#def make_atoms

#--------------Parameters------------------
smfile = '/home/jujuman/Research/RawGDB11Database/gdb11_size08.smi' # Smiles file

At = ['C', 'O', 'N'] # Hydrogens added after check

Nnc = 5

#-------------------------------------------
#nnfdir   = wkdir + 'cv_c08e_ntw_' + str(0) + '/networks/'

molecules = Chem.SmilesMolSupplier(smfile, nameColumn=0)

total_mol = 0
for k,m in enumerate(molecules):
    if m is None: continue

    typecount = 0

    typecheck = False
    for a in m.GetAtoms():
        sym = str(a.GetSymbol())
        count = 0

        for i in At:
            if i is sym:
                count = 1

        if count is 0:
            typecheck = True

    if typecheck is False:
        total_mol = total_mol + 1


print('total_mol: ',total_mol)
