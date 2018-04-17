import sys
import time
import random
from random import randint
import os

# Numpy
import numpy as np

# Neuro Chem
from ase_interface import ANI
from ase_interface import ANIENS
from ase_interface import ensemblemolecule

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
from ase.optimize.fire import FIRE as QuasiNewton
from ase.optimize import BFGS, LBFGS, FIRE
from ase.constraints import FixInternals
from ase.io import read, write

import math
import copy
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

# -------------------------------------
#  Class for ANI cross validaiton tools
# --------------------------------------
class anicrossvalidationconformer(object):
    ''' Constructor '''
    def __init__(self,cnstfile,saefile,nnfprefix,Nnet,gpuid=0, sinet=False):
        # Number of networks
        self.Nn = Nnet

        # Construct pyNeuroChem class
        self.ncl = [pync.conformers(cnstfile, saefile, nnfprefix+str(i)+'/networks/', gpuid, sinet) for i in range(self.Nn)]
        #self.ncl = [pync.conformers(cnstfile, saefile, nnfprefix+str(1)+'/networks/', gpuid, sinet) for i in range(self.Nn)]

    ''' Compute the std. dev. from cross validation networks on a set of comformers '''
    def compute_stddev_conformations(self,X,S):
        energies = np.zeros((self.Nn,X.shape[0]),dtype=np.float64)
        for i,nc in enumerate(self.ncl):
            nc.setConformers(confs=X,types=list(S))
            energies[i] = nc.energy().copy()
        sigma1 = hdt.hatokcal * np.std(energies,axis=0) / np.sqrt(float(len(S)))
        #sigma2 = hdt.hatokcal * np.std(energies, axis=0) / float(len(S))
        return sigma1

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

    ''' Compute the energy and mean force of a set of conformers for the CV networks '''
    def compute_energyandforce_conformations(self,X,S,ensemble=True):
        energy = np.zeros((self.Nn, X.shape[0]), dtype=np.float64)
        forces = np.zeros((self.Nn, X.shape[0], X.shape[1], X.shape[2]), dtype=np.float32)
        for i,nc in enumerate(self.ncl):
            nc.setConformers(confs=X,types=list(S))
            energy[i] = nc.energy().copy()
            forces[i] = nc.force().copy()

        sigmap = hdt.hatokcal * np.std(energy,axis=0) / np.sqrt(X.shape[1])
        if ensemble:
            return hdt.hatokcal*np.mean(energy,axis=0), hdt.hatokcal*np.mean(forces,axis=0), sigmap#, charges
        else:
            return hdt.hatokcal*energy, hdt.hatokcal*forces, sigmap

    ''' Compute the energy and mean force of a set of conformers for the CV networks '''
    def compute_energy_conformations(self,X,S):
        Na = X.shape[0] * len(S)

        X_split = np.array_split(X, math.ceil(Na/20000))

        energies = np.zeros((self.Nn, X.shape[0]), dtype=np.float64)
        forces   = np.zeros((self.Nn, X.shape[0], X.shape[1], X.shape[2]), dtype=np.float32)
        shift = 0
        for j,x in enumerate(X_split):
            for i, nc in enumerate(self.ncl):
                nc.setConformers(confs=x,types=list(S))
                E = nc.energy().copy()
                print(E.shape,energies.shape)
                energies[i,j+shift:j+shift+E.shape[0]] = E
            shift += x.shape[0]

        return hdt.hatokcal*np.mean(energies,axis=0)#, charges

    ''' Compute the energy and mean force of a set of conformers for the CV networks '''
    def compute_separate(self,X,S,i):
        Na = X.shape[0] * len(S)

        X_split = np.array_split(X, math.ceil(Na/20000))
        nc = self.ncl[i]
        energies=[]
        forces=[]
        for x in X_split:
            nc.setConformers(confs=x,types=list(S))
            energies.append(nc.energy().copy())
            forces.append(nc.force().copy())

        energies = np.concatenate(energies)
        forces = np.vstack(forces)
        return hdt.hatokcal*energies, hdt.hatokcal*forces#, charges

    ''' Compute the weighted average energy and forces of a set of conformers for the CV networks '''
    def compute_wa_energy_conformations(self,X,S):
        energies = np.zeros((self.Nn, X.shape[0]), dtype=np.float64)
        #charges  = np.zeros((self.Nn, X.shape[0], X.shape[1]), dtype=np.float32)
        forces   = np.zeros((self.Nn, X.shape[0], X.shape[1], X.shape[2]), dtype=np.float32)
        for i,nc in enumerate(self.ncl):
            nc.setConformers(confs=X,types=list(S))
            energies[i] = nc.energy().copy()
            #charges[i] = nc.charge().copy()
            forces[i] = nc.force().copy()

        # Compute distribution weighted average
        mu = np.mean(energies, axis=0)
        sg = np.std(energies, axis=0)
        gamma = sg / np.sqrt(np.abs(mu - energies + 1.0e-40))
        gamma = gamma / np.sum(gamma, axis=0)
        energies = np.sum(energies * gamma, axis=0)

        return hdt.hatokcal*energies, hdt.hatokcal*np.mean(forces, axis=0)#, charges

    ''' Compute the energy of a set of conformers for the CV networks '''
    def get_charges_conformations(self,X,S):
        charges  = np.zeros((self.Nn, X.shape[0], X.shape[1]), dtype=np.float32)
        for i,nc in enumerate(self.ncl):
            #nc.setConformers(confs=X,types=list(S))
            charges[i] = nc.charge().copy()
        return charges

    ''' Compute the std. dev. of rdkit conformers '''
    def compute_stddev_rdkitconfs(self,mrdk):
        X,S = __convert_rdkitconfs_to_nparr__(mrdk)
        return self.compute_stddev_conformations(X,S)

    ''' Compute the energies of rdkit conformers '''
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

    def compute_energies_and_forces_molecule(self, x, S):
        Na = x.shape[0]
        energy = np.zeros((self.Nn), dtype=np.float64)
        forces = np.zeros((self.Nn, Na, 3), dtype=np.float32)
        for i,nc in enumerate(self.ncl):
            nc.setMolecule(coords=x, types=list(S))
            energy[i]       = nc.energy()[0]
            forces[i, :, :] = nc.force()

        sigmap = hdt.hatokcal * np.std(energy) / np.sqrt(Na)
        energy = hdt.hatokcal * energy.mean()
        forces = hdt.hatokcal * np.mean(forces, axis=0)

        return energy, forces, sigmap

    def compute_stats_multi_molecule(self, X, S):
        Nd = X.shape[0]
        Na = X.shape[1]
        for nc in self.ncl:
            nc.setMolecule(coords=X[0], types=list(S))

        energies = np.zeros((self.Nn, Nd), dtype=np.float64)
        forces = np.zeros((self.Nn, Nd, Na, 3), dtype=np.float32)

        for i, x in enumerate(X):
            for j, nc in enumerate(self.ncl):
                nc.setCoordinates(coords=x)
                energies[j,i] = nc.energy()[0]
                forces[j,i,:,:] = nc.force()

        sigma = hdt.hatokcal * np.std(energies, axis=0) / np.sqrt(float(X.shape[0]))
        return hdt.hatokcal * energies, hdt.hatokcal * forces, sigma

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
        #dyn = LBFGS(mol)
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
            e,s = self.nc.energy().copy()
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

        if len(E) > 0:
            return np.concatenate(E)
        else:
            return np.array([])

class anienscomputetool(object):
    def __init__(self,cnstfile,saefile,nnfdir,Nn,gpuid=0, sinet=False):
        # Construct pyNeuroChem class
        #self.nc = [pync.molecule(cnstfile, saefile, nnfdir+str(i)+'/', gpuid, sinet) for i in range(Nn)]

        self.ens = ensemblemolecule(cnstfile,saefile,nnfdir,Nn,gpuid,sinet)

    #def __init__(self, nc):
    #    # Construct pyNeuroChem class
    #    self.nc = nc

    def optimize_rdkit_molecule(self, mrdk, cid, fmax=0.1, steps=10000, logger='opt.out'):
        mol = __convert_rdkitmol_to_aseatoms__(mrdk,cid)
        mol.set_pbc((False, False, False))
        mol.set_calculator(ANIENS(self.ens))
        dyn = LBFGS(mol,logfile=logger)
        #dyn = LBFGS(mol)
        #dyn = QuasiNewton(mol,logfile=logger)
        dyn.run(fmax=fmax,steps=steps)
        stps = dyn.get_number_of_steps()

        opt = True
        if steps == stps:
            opt = False

        xyz = mol.get_positions()
        for i,x in enumerate(xyz):
            mrdk.GetConformer(cid).SetAtomPosition(i,x)

        return opt

    def optimize_molecule(self, X, S, fmax=0.1, steps=10000, logger='opt.out'):
        print(S)
        mol = Atoms(symbols=S, positions=X)
        mol.set_pbc((False, False, False))
        mol.set_calculator(ANIENS(self.ens))
        #dyn = LBFGS(mol,logfile=logger)
        dyn = LBFGS(mol)
        #dyn = QuasiNewton(mol,logfile=logger)
        dyn.run(fmax=fmax,steps=steps)
        stps = dyn.get_number_of_steps()

        opt = True
        if steps == stps:
            opt = False

        return np.array(mol.get_positions(),dtype=np.float32), opt

    def energy_rdkit_conformers(self,mol,cids):
        E = []
        V = []
        for cid in cids:
            X, S = __convert_rdkitmol_to_nparr__(mol,confId=cid)
            self.ens.set_molecule(X=X, S=list(S))
            e,v = self.ens.compute_mean_energies()
            E.append(e)
            V.append(v)
        return np.array(E),np.array(V)

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

        if len(E) > 0:
            return np.concatenate(E)
        else:
            return np.array([])

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

        # Build ANI ensemble
        self.aens = ensemblemolecule(cnstfile, saefile, nnfprefix, Nnet, gpuid)

        # Construct pyNeuroChem class
        #self.ncl = [pync.molecule(cnstfile, saefile, nnfprefix+str(i)+'/networks/', gpuid, sinet) for i in range(self.Nn)]

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
    def __run_rand_dyn__(self, mid, T1, T2, dt, Nc, Ns, dS):
        # Setup calculator
        mol = self.mols[0].copy()

        # Setup PBC if active
        if self.pbc:
            mol.set_cell(([[self.pbl, 0, 0],
                           [0, self.pbl, 0],
                           [0, 0, self.pbl]]))

            mol.set_pbc((self.pbc, self.pbc, self.pbc))

        # Setup calculator
        mol.set_calculator(ANIENS(self.aens))

        #mol.set_calculator(ANI(False))
        #mol.calc.setnc(self.ncl[0])

        # Set chemical symbols
        spc = mol.get_chemical_symbols()

        # Set the velocities corresponding to a boltzmann dist @ T/4.0
        MaxwellBoltzmannDistribution(mol, T1 * units.kB)

        # Set the thermostat
        dyn = Langevin(mol, dt * units.fs, T1 * units.kB, 0.02)
        dT = (T2 - T1)/Nc
        #print('Running...')
        for i in range(Nc):
            # Set steps temperature
            dyn.set_temperature((T1 + dT*i) * units.kB)

            # Do Ns steps of dynamics
            dyn.run(Ns)

            # Return sigma
            sigma = hdt.evtokcal * mol.calc.stddev

            ekin = mol.get_kinetic_energy() / len(mol)

            # Check for dynamics failure
            if sigma > dS:
                self.Nbad += 1
                self.X.append(mol.get_positions())
                #print('Step:', dyn.get_number_of_steps(), 'Sig:', sigma, 'Temp:',
                #      str(ekin / (1.5 * units.kB)) + '(' + str(T1 + dT * i) + ')')

                return True,dyn.get_number_of_steps()
        return False,dyn.get_number_of_steps()

    """Generate a set of conformations from MD
    """
    def generate_conformations(self, Nr, T1, T2, dt, Nc, Ns, dS):
        Ng = 0
        self.Nbad = 0
        self.X = []
        fsteps = []
        for r in range(Nr):
            found, steps = self.__run_rand_dyn__(mid=r, T1=T1, T2=T2, dt=dt, Nc=Nc, Ns=Ns, dS=dS)
            fsteps.append(steps)
            if found:
                Ng += 1

        fsteps = np.array(fsteps).mean()

        if len(self.X) > 0:
            X = np.stack(self.X)

        self.failtime = fsteps * dt

        self._infostr_ = 'New confs: ' + str(Ng)+' of ' + str(Nr) + ' - ' + \
                          str(Nc * Ns * dt) + 'fs trajectories. Failed within '+"{:.2f}".format(self.failtime)+'fs on average.'

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

class ani_tortion_scanner():
    def __init__(self,ens):
        self.ens = ens

    def opt(self, rdkmol, atid, logger='optlog.out'):
        Na = rdkmol.GetNumAtoms()
        X, S = __convert_rdkitmol_to_nparr__(rdkmol)
        mol=Atoms(symbols=S, positions=X)
        mol.set_calculator(ANIENS(self.ens))                #Set the ANI Ensemble as the calculator
        phi_restraint=mol.get_dihedral(atid)
        phi_fix = [phi_restraint, atid]
        c = FixInternals(dihedrals=[phi_fix], epsilon=1.e-9)
        mol.set_constraint(c)
        dyn = BFGS(mol, logfile=logger)                               #Choose optimization algorith
        dyn.run(fmax=0.000514221, steps=1000, )         #optimize molecule to Gaussian's Opt=Tight fmax criteria, input in eV/A (I think)
        e=mol.get_potential_energy()*hdt.evtokcal
        phi_value=mol.get_dihedral(atid)*180./np.pi
        X = mol.get_positions()
        print('Phi value (degrees), energy (kcal/mol)= ', "{0:.2f}".format(phi_value), "{0:.2f}".format(e))
        return phi_value, e, X

    def rot(self, mol, atid, phi):
        
        a0=int(atid[0])
        a1=int(atid[1])
        a2=int(atid[2])
        a3=int(atid[3])
        
        c=mol.GetConformer()
        Chem.rdMolTransforms.SetDihedralDeg(c, a0, a1, a2, a3, phi)
        phi_value, e, X = self.opt(mol, atid)
        #print(X)
        return phi_value, e, X

    def scan_tortion(self, mol, atid, inc, stps):
        mol_copy = copy.deepcopy(mol)

        a0=int(atid[0])
        a1=int(atid[1])
        a2=int(atid[2])
        a3=int(atid[3])

        c=mol_copy.GetConformer()
        init = Chem.rdMolTransforms.GetDihedralDeg(c, a0, a1, a2, a3)
        print(init)

        ang = []
        enr = []
        for i in range(stps):
            phi, e, X = self.rot(mol, atid, init - i*inc)
            ang.append(phi)
            enr.append(e)
        return np.array(ang), np.array(enr)

class MD_Sampler:
    def __init__(self, files, cnstfile, saefile, nnfprefix, gpuid=0, sinet=False):
        self.files=files                      #List of files containing the molecules to run md on
        self.coor_train=[]                    #Where the coordinates of the molecules with high standard deviation will be saved
        self.Na_train=[]                      #Where the number of atoms of the molecules with high standard deviation will be saved
        self.S_train=[]                       #Where the atomic species of the molecules with high standard deviation will be saved
        self.hstd=[]                          #Where the standard deviation of the molecules with high standard deviation will be saved

        #The path to the network
        self.net = ensemblemolecule(cnstfile, saefile, nnfprefix, Nnet, gpuid)            #Load the network

    def run_md(self, f, T, steps, n_steps, record=False):
        mol=read(f)
        mol.set_calculator(ANIENS(self.net,sdmx=20000000.0))
        f=os.path.basename(f)
        dyn = Langevin(mol, 0.1 * units.fs, T * units.kB, 0.01)
        MaxwellBoltzmannDistribution(mol, T * units.kB)
        dyn.set_temperature(T * units.kB)
#        steps=10000    #10000=1picosecond                             #Max number of steps to run
#        n_steps = 1                                                #Number of steps to run for before checking the standard deviation
        hsdt_Na=[]
        evkcal=hdt.evtokcal

        if record==True:                                           #Records the coordinates at every step of the dynamics
            fname = f + '_record_' + str(T) + 'K' + '.xyz'     #name of file to store coodinated in
            def printenergy(name=fname, a=mol):
                """Function to print the potential, kinetic and total energy."""
                fil= open(name,'a')
                Na=a.get_number_of_atoms()
                c = a.get_positions(wrap=True)
                fil.write('%s \n comment \n' %Na)
                for j, i in zip(a, c):
                    fil.write(str(j.symbol) + ' ' + str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + '\n')
                fil.close()
            dyn.attach(printenergy, interval=1)

        e=mol.get_potential_energy()                             #Calculate the energy of the molecule. Must be done to get the standard deviation
        s=mol.calc.stddev
        stddev =  s*evkcal
        tot_steps = 0
        while (tot_steps <= steps):
            if stddev > 0.34:                                #Check the standard deviation
                self.hstd.append(stddev)
                c = mol.get_positions(wrap=True)
                s = mol.get_chemical_symbols()
                Na=mol.get_number_of_atoms()
                self.Na_train.append(Na)
                self.coor_train.append(c)
                self.S_train.append(s)
                break
            else:                                           #if the standard deviation is low, run dynamics, then check it again
                tot_steps = tot_steps + n_steps
                dyn.run(n_steps)
                s=mol.calc.stddev
                stddev =  evkcal*s
                e=mol.get_potential_energy()
        return c, s


    def run_md_list(self):
        T = random.randrange(0, 20, 1)                              # random T (random velocities) from 0K to 20K
        for f in self.files:
            self.run_md(f, T)

    def write_training_xyz(self, fname):                                #Calling this function wirtes all the structures with high standard deviations to and xyz file
        alt=open('%s' %fname, 'w')
        for i in range(len(self.Na_train)):
            cm='STD= ' + str(self.hstd[i])
            alt.write('%s \n%s\n' %(str(self.Na_train[i]), cm))
            for j in range(len(self.S_train[i])):
                alt.write('%s %f %f %f \n' %(self.S_train[i][j], self.coor_train[i][j][0], self.coor_train[i][j][1], self.coor_train[i][j][2]))
        alt.close()

    def get_N(self, N):                                                  #Runs dynamics with random T and random file N times. This may result in running the same molecule multiple times
        for i in range(N):
            T = random.randrange(0, 20, 1)                       # random T (random velocities) from 0K to 20K
            f=random.choice(self.files)
            self.run_md(f, T)

