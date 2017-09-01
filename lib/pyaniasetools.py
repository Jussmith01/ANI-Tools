import sys
import time
from random import randint

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
from ase.md.langevin import Langevin
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

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

## Get per conformer data from CV set base on crit
def getcvconformerdata(Ncv ,datacv, dataa, Scv, c):
    T1 = []
    T2 = []

    Nt = 0
    Nd = 0
    for id,sig in enumerate(Scv):
        bidx = np.where(sig <= c)[0]
        Nt += sig.size
        Nd += bidx.size
        T1.append(datacv[id][:, bidx].reshape(Ncv,-1))
        T2.append(dataa[id][bidx].reshape(-1))
    return np.hstack(T1),np.concatenate(T2),Nd,Nt

## Get per conformer data from energy set base on crit
def getenergyconformerdata(Ncv ,datacv, dataa, datae, e):
    T1 = []
    T2 = []

    Nt = 0
    Nd = 0
    for id,de in enumerate(datae):
        de = de - de.min()
        bidx = np.where(de <= e)[0]
        Nt += de.size
        Nd += bidx.size
        T1.append(datacv[id][:, bidx].reshape(Ncv,-1))
        T2.append(dataa[id][bidx].reshape(-1))
    return np.hstack(T1),np.concatenate(T2), Nd, Nt

##-------------------------------------
## Class for ANI cross validaiton tools
##--------------------------------------
class anicrossvalidationconformer(object):
    ''' Constructor '''
    def __init__(self,cnstfile,saefile,nnfprefix,Nnet,gpuid=0, sinet=False):
        # Number of networks
        self.Nn = Nnet

        # Construct pyNeuroChem class
        self.ncl = [pync.conformers(cnstfile, saefile, nnfprefix+str(i)+'/networks/', gpuid, sinet) for i in range(self.Nn)]

    ''' Compute the std. dev. from cross validation networks on a set of comformers '''
    def compute_stddev_conformations(self,X,S):
        energies = np.zeros((self.Nn,X.shape[0]),dtype=np.float64)
        for i,nc in enumerate(self.ncl):
            nc.setConformers(confs=X,types=list(S))
            energies[i] = nc.energy().copy()
        sigma = hdt.hatokcal*np.std(energies,axis=0) / float(len(S))
        return sigma

    ''' Compute the dE from cross validation networks on a set of comformers '''
    def compute_energy_delta_conformations(self,X,Ea,S):
        deltas = np.zeros((self.Nn,X.shape[0]),dtype=np.float64)
        energies = np.zeros((self.Nn,X.shape[0]),dtype=np.float64)
        for i,nc in enumerate(self.ncl):
            nc.setConformers(confs=X,types=list(S))
            Ec = nc.energy().copy()
            deltas[i] = Ec-Ea
            energies[i] = Ec - Ec.min()
        deltas = deltas
        return deltas,energies

    ''' Compute the energy of a set of conformers for the CV networks '''
    def compute_energy_conformations(self,X,S):
        energies = np.zeros((self.Nn, X.shape[0]), dtype=np.float64)
        forces   = np.zeros((self.Nn, X.shape[0], X.shape[1], X.shape[2]), dtype=np.float32)
        for i,nc in enumerate(self.ncl):
            nc.setConformers(confs=X,types=list(S))
            energies[i] = nc.energy().copy()
            forces[i] = nc.force().copy()
        return hdt.hatokcal*energies, hdt.hatokcal*forces

    ''' Compute the std. dev. of rdkit conformers '''
    def compute_stddev_rdkitconfs(self,mrdk):
        X,S = __convert_rdkitconfs_to_nparr__(mrdk)
        return self.compute_stddev_conformations(X,S)

##-------------------------------------
## Class for ANI cross validaiton tools
##--------------------------------------
class anicrossvalidationmolecule(object):
    def __init__(self, cnstfile, saefile, nnfprefix, Nnet, gpuid=0, sinet=False):
        # Number of networks
        self.Nn = Nnet

        # Construct pyNeuroChem class
        self.ncl = [pync.molecule(cnstfile, saefile, nnfprefix + str(i) + '/networks/', gpuid, sinet) for i in
                    range(self.Nn)]

    def set_molecule(self,X,S):
        for nc in self.ncl:
            nc.setMolecule(coords=X, types=list(S))

    def compute_stddev_molecule(self, X):
        energies = np.zeros((self.Nn), dtype=np.float64)
        for i, nc in enumerate(self.ncl):
            nc.setCoordinates(coords=X)
            energies[i] = nc.energy()[0]
        sigma = hdt.hatokcal * np.std(energies, axis=0) / float(X.shape[0])
        return sigma, hdt.hatokcal *energies

##--------------------------------
## Class for ANI compute tools
##--------------------------------
class anicomputetool(object):
    def __init__(self,cnstfile,saefile,nnfdir,gpuid=0, sinet=False):
        # Construct pyNeuroChem class
        self.nc = pync.molecule(cnstfile, saefile, nnfdir, gpuid, sinet)

    #def __init__(self, nc):
    #    # Construct pyNeuroChem class
    #    self.nc = nc

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
                #print('remove')
            else:
                E.append(e)
        return np.concatenate(E)

##--------------------------------
##    Active Learning ANI MD
##--------------------------------
class moldynactivelearning(object):
    """Initialze the CV set and ani compute tools
    """
    def __init__(self,cnstfile,saefile,nnfprefix,Nnet,gpuid=0, sinet=False):
        self.Nbad = 0

        # Number of networks
        self.Nn = Nnet

        # Construct pyNeuroChem class
        self.ncl = [pync.molecule(cnstfile, saefile, nnfprefix+str(i)+'/networks/', gpuid, sinet) for i in range(self.Nn)]

        # Initalize PBC
        self.pbc = False
        self.pbl = 0.0

        # ANI compute tools
        #self.ani = anicomputetool(self.ncl[0])

    """Sets the ase atoms classes for each random conformer
    """
    def setrdkitmol(self, mrdk, cids, steps=250, fmax=0.0001):
        #self.ani.detect_unique_rdkitconfs(mrdk, cids)
        confs = mrdk.GetConformers()

        self.mols = []
        for c in confs:
            cid = c.GetId()
            mol = __convert_rdkitmol_to_aseatoms__(mrdk,cid)
            self.mols.append(mol)

    """Sets the ase atoms classes for each random conformer
    """
    def setmol(self, X, S, steps=250, fmax=0.0001):
        #self.ani.detect_unique_rdkitconfs(mrdk, cids)
        mol = Atoms(symbols=S, positions=X)
        self.mols = [mol]

    """Sets the box size for use in PBC dynamics
    """
    def set_pbc_box(self, length):
        self.pbl = length
        self.pbc = True

    """Runs the supplied trajectory with a random starting point
    """
    def __run_rand_dyn__(self, mid, T, dt, Nc, Ns, dS):
        # Setup calculator
        mol = self.mols[0].copy()

        # Setup PBC if active
        if self.pbc:
            mol.set_cell(([[self.pbl, 0, 0],
                           [0, self.pbl, 0],
                           [0, 0, self.pbl]]))

            mol.set_pbc((self.pbc, self.pbc, self.pbc))

        # Setup calculator
        mol.set_calculator(ANI(False))
        mol.calc.setnc(self.ncl[0])

        # Set chemical symbols
        spc = mol.get_chemical_symbols()

        # Set the momenta corresponding to T=300K
        MaxwellBoltzmannDistribution(mol, (T/4.0) * units.kB)

        # Set the thermostat
        dyn = Langevin(mol, dt * units.fs, T * units.kB, 0.02)
        #print('Running...')
        for i in range(Nc):
            dyn.run(Ns)  # Do Ns steps of MD

            xyz = np.array(mol.get_positions(wrap=True), dtype=np.float32).reshape(len(spc), 3)
            energies = np.zeros((self.Nn), dtype=np.float64)
            N = 0
            for comp in self.ncl:
                comp.setMolecule(coords=xyz, types=list(spc))
                energies[N] = comp.energy()[0]
                N = N + 1

            energies = hdt.hatokcal * energies
            sigma = np.std(energies) / float(len(spc))
            #print('s:', sigma)
            if sigma > dS:
                self.Nbad += 1
                self.X.append(mol.get_positions())
                return True,dyn.get_number_of_steps()
        return False,dyn.get_number_of_steps()

    """Generate a set of conformations from MD
    """
    def generate_conformations(self, Nr, T, dt, Nc, Ns, dS):
        Ng = 0
        self.Nbad = 0
        self.X = []
        fsteps = []
        for r in range(Nr):
            found, steps = self.__run_rand_dyn__(mid=r, T=T, dt=dt, Nc=Nc, Ns=Ns, dS=dS)
            fsteps.append(steps)
            if found:
                Ng += 1

        fsteps = np.array(fsteps).mean()

        if len(self.X) > 0:
            X = np.stack(self.X)

        self._infostr_ = 'New confs: ' + str(Ng)+' of ' + str(Nr) + ' - ' + \
                          str(Nc * Ns * dt) + 'fs trajectories. Failed within '+"{:.2f}".format(fsteps*dt)+'fs on average.'

        return np.array(self.X)
##--------------------------------
##    Diverse conformers
##--------------------------------
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
