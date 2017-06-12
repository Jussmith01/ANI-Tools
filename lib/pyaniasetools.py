import sys
import time

# Numpy
import numpy as np

# Neuro Chem
from ase_interface import ANI
import pyNeuroChem as pync

from rdkit import Chem
from rdkit.Chem import AllChem

import  ase
from ase import Atoms
from ase.optimize import LBFGS
from ase.calculators.calculator import Calculator, all_changes

class anicomputetool(object):
    def __init__(self,cnstfile,saefile,nnfdir,gpuid=0, sinet=False):
        # Construct pyNeuroChem class
        self.nc = pync.molecule(cnstfile, saefile, nnfdir, gpuid, sinet)

    def __convert_rdkitmol_to_aseatoms__(self,mrdk,confId=-1):
        X, S = self.__convert_rdkitmol_to_nparr__(mrdk,confId)
        mol = Atoms(symbols=S, positions=X)
        return mol

    def __convert_rdkitmol_to_nparr__(self,mrdk,confId=-1):
        xyz = np.zeros((mrdk.GetNumAtoms(), 3), dtype=np.float32)
        spc = []

        Na = mrdk.GetNumAtoms()
        for i in range(0, Na):
            pos = mrdk.GetConformer(confId).GetAtomPosition(i)
            sym = mrdk.GetAtomWithIdx(i).GetSymbol()

            spc.append(sym)
            xyz[i, 0] = pos.x
            xyz[i, 1] = pos.y
            xyz[i, 2] = pos.z

        return xyz,spc

    def optimize_rdkit_molecule(self, mrdk, cid, fmax=0.0001, steps=500, logger='opt.out'):
        mol = self.__convert_rdkitmol_to_aseatoms__(mrdk,cid)
        mol.set_calculator(ANI(False))
        mol.calc.setnc(self.nc)
        dyn = LBFGS(mol,logfile=logger)
        dyn.run(fmax=fmax,steps=steps)
        stps = dyn.get_number_of_steps()

        xyz = mol.get_positions()
        for i,x in enumerate(xyz):
            mrdk.GetConformer(cid).SetAtomPosition(i,x)
        #print(stps)

    def energy_rdkit_conformers(self,mol,cids):
        E = []
        for cid in cids:
            X, S = self.__convert_rdkitmol_to_nparr__(mol,confId=cid)
            self.nc.setMolecule(coords=X, types=list(S))
            e = self.nc.energy().copy()
            E.append(e)
        return np.concatenate(E)

    def __in_list_within_eps__(self,val,ilist,eps):
        for i in ilist:
            if abs(i-val) < eps:
                return True
        return False

    def detect_unique_rdkitconfs(self, mol, cids):
        E = []
        for cid in cids:
            X, S = self.__convert_rdkitmol_to_nparr__(mol, confId=cid)
            self.nc.setMolecule(coords=X, types=list(S))
            e = self.nc.energy().copy()
            if self.__in_list_within_eps__(e,E,1.0E-6):
                mol.RemoveConformer(cid)
            else:
                E.append(e)
        return np.concatenate(E)