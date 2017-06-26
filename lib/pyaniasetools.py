import sys
import time

# Numpy
import numpy as np

# Neuro Chem
from ase_interface import ANI
import pyNeuroChem as pync

import hdnntools as hdt

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# SimDivPicker
from rdkit.SimDivFilters import rdSimDivPickers

# Scipy
import scipy.spatial as scispc

import  ase
from ase import Atoms
from ase.optimize import LBFGS
from ase.calculators.calculator import Calculator, all_changes

## Converts an rdkit mol class conformer to a 2D numpy array
def __convert_rdkitmol_to_nparr__(mrdk, confId=-1):
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

    return xyz, spc

## Converts all confromers of an rdkit mol to a 3D numpy array
def __convert_rdkitconfs_to_nparr__(mrdk):
    X = []
    for c in mrdk.GetConformers():
        xyz,S = __convert_rdkitmol_to_nparr__(mrdk,c.GetId())
        X.append(xyz)
    X = np.stack(X)
    return X,S

## Converts a single rdkit mol conformer to an ASE atoms
def __convert_rdkitmol_to_aseatoms__(mrdk, confId=-1):
    X, S = __convert_rdkitmol_to_nparr__(mrdk, confId)
    mol = Atoms(symbols=S, positions=X)
    return mol

##-------------------------------------
## Class for ANI cross validaiton tools
##--------------------------------------
class anicrossvalidationconformer(object):
    def __init__(self,cnstfile,saefile,nnfprefix,Nnet,gpuid=0, sinet=False):
        # Number of networks
        self.Nn = Nnet

        # Construct pyNeuroChem class
        self.ncl = [pync.conformers(cnstfile, saefile, nnfprefix+str(i)+'/networks/', gpuid, sinet) for i in range(self.Nn)]

    def compute_stddev_conformations(self,X,S):
        energies = np.zeros((self.Nn,X.shape[0]),dtype=np.float64)
        for i,nc in enumerate(self.ncl):
            nc.setConformers(confs=X,types=list(S))
            energies[i] = nc.energy().copy()
        sigma = hdt.hatokcal*np.std(energies,axis=0) / float(len(S))
        return sigma

    def compute_energy_delta_conformations(self,X,Ea,S):
        deltas = np.zeros((self.Nn,X.shape[0]),dtype=np.float64)
        for i,nc in enumerate(self.ncl):
            nc.setConformers(confs=X,types=list(S))
            deltas[i] = np.abs(nc.energy().copy()-Ea)
        deltas = hdt.hatokcal*deltas
        return deltas

    def compute_stddev_rdkitconfs(self,mrdk):
        X,S = __convert_rdkitconfs_to_nparr__(mrdk)
        return self.compute_stddev_conformations(X,S)

##--------------------------------
## Class for ANI compute tools
##--------------------------------
class anicomputetool(object):
    def __init__(self,cnstfile,saefile,nnfdir,gpuid=0, sinet=False):
        # Construct pyNeuroChem class
        self.nc = pync.molecule(cnstfile, saefile, nnfdir, gpuid, sinet)

    def optimize_rdkit_molecule(self, mrdk, cid, fmax=0.0001, steps=500, logger='opt.out'):
        mol = __convert_rdkitmol_to_aseatoms__(mrdk,cid)
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

    def detect_unique_rdkitconfs(self, mol, cids, eps=1.0E-6):
        E = []
        for cid in cids:
            X, S = __convert_rdkitmol_to_nparr__(mol, confId=cid)
            self.nc.setMolecule(coords=X, types=list(S))
            e = self.nc.energy().copy()
            if self.__in_list_within_eps__(e,E,eps):
                mol.RemoveConformer(cid)
            else:
                E.append(e)
        return np.concatenate(E)

class diverseconformers():
    def __init__(self,cnstfile,saefile,nnfdir,aevsize,gpuid=0,sinet=False):
        self.avs = aevsize

        # Construct pyNeuroChem class
        self.nc = pync.conformers(cnstfile, saefile, nnfdir, gpuid, sinet)

    def get_divconfs_ids(self, X, S, Ngen, Nkep, atmlist=[]):
        if len(atmlist) > 0:
            al = atmlist
        else:
            al = [i for i, s in enumerate(S) if s != 'H']

        self.nc.setConformers(confs=X, types=list(S))
        Ecmp = self.nc.energy()  # this generates AEVs

        aevs = np.empty([Ngen, len(al) * self.avs])
        for m in range(Ngen):
            for j, a in enumerate(al):
                aevs[m, j * self.avs:(j + 1) * self.avs] = self.nc.atomicenvironments(a, m).copy()

        dm = scispc.distance.pdist(aevs, 'sqeuclidean')
        picker = rdSimDivPickers.MaxMinPicker()
        seed_list = [i for i in range(Ngen)]
        np.random.shuffle(seed_list)
        ids = list(picker.Pick(dm, Ngen, Nkep, firstPicks=list(seed_list[0:5])))
        ids.sort()
        return ids
