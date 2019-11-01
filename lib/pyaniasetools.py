import sys
import time
import random
from random import randint
import os
import re
import subprocess
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
from rdkit.Chem import Descriptors
from rdkit import Geometry

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
from ase.vibrations import Vibrations

import itertools

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
    def __init__(self,cnstfile,saefile,nnfprefix,Nnet,gpuid=[0], sinet=False):
        # Number of networks
        self.Nn = Nnet

        gpua = [gpuid[int(np.floor(i/(Nnet/len(gpuid))))] for i in range(self.Nn)]

        # Construct pyNeuroChem class
        self.ncl = [pync.conformers(cnstfile, saefile, nnfprefix+str(i)+'/networks/', gpua[i], sinet) for i in range(self.Nn)]
        #self.ncl = [pync.conformers(cnstfile, saefile, nnfprefix+str(1)+'/networks/', gpuid, sinet) for i in range(self.Nn)]

    ''' Compute the std. dev. from cross validation networks on a set of comformers '''
    def compute_stddev_conformations(self,X,S):
        energies = np.zeros((self.Nn,X.shape[0]),dtype=np.float64)
        for i,nc in enumerate(self.ncl):
            nc.setConformers(confs=np.array(X,dtype=np.float64),types=list(S))
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

#    ''' Compute the energy and mean force of a set of conformers for the CV networks '''
#    def compute_energyandforce_conformations(self,X,S,ensemble=True):
#        energy = np.zeros((self.Nn, X.shape[0]), dtype=np.float64)
#        forces = np.zeros((self.Nn, X.shape[0], X.shape[1], X.shape[2]), dtype=np.float32)
#        for i,nc in enumerate(self.ncl):
#            nc.setConformers(confs=np.array(X,dtype=np.float32),types=list(S))
#            energy[i] = nc.energy().copy()
#            forces[i] = nc.force().copy()
#
#        sigmap = hdt.hatokcal * np.std(energy,axis=0) / np.sqrt(X.shape[1])
#        if ensemble:
#            return hdt.hatokcal*np.mean(energy,axis=0), hdt.hatokcal*np.mean(forces,axis=0), sigmap#, charges
#        else:
#            return hdt.hatokcal*energy, hdt.hatokcal*forces, sigmap

    ''' Compute the energy and mean force of a set of conformers for the CV networks '''
    def compute_energy_conformations(self, X, S):
        Na = X.shape[0] * len(S)

        X_split = np.array_split(X, math.ceil(Na/10000))

        energies = np.zeros((self.Nn, X.shape[0]), dtype=np.float64)
        forces   = np.zeros((self.Nn, X.shape[0], X.shape[1], X.shape[2]), dtype=np.float32)
        shift = 0
        for j,x in enumerate(X_split):
            for i, nc in enumerate(self.ncl):
                nc.setConformers(confs=np.array(x,dtype=np.float64),types=list(S))
                E = nc.energy().copy()
                #print(E,x)
                energies[i,shift:shift+E.shape[0]] = E
            shift += x.shape[0]

        sigma = hdt.hatokcal * np.std(energies,axis=0) / np.sqrt(float(len(S)))
        return hdt.hatokcal*np.mean(energies,axis=0),sigma#, charges

    ''' Compute the energy and mean force of a set of conformers for the CV networks '''
    def compute_energyandforce_conformations(self,X,S,ensemble=True):
        Na = X.shape[0] * len(S)

        X_split = np.array_split(X, math.ceil(Na/10000))

        energies = np.zeros((self.Nn, X.shape[0]), dtype=np.float64)
        forces   = np.zeros((self.Nn, X.shape[0], X.shape[1], X.shape[2]), dtype=np.float32)
        shift = 0
        for j,x in enumerate(X_split):
            for i, nc in enumerate(self.ncl):
                nc.setConformers(confs=np.array(x,dtype=np.float64),types=list(S))
                E = nc.energy().copy()
                F = nc.force().copy()
                #print(E.shape,x.shape,energies.shape,shift)
                energies[i,shift:shift+E.shape[0]] = E
                forces[i,shift:shift+E.shape[0]] = F
            shift += x.shape[0]

        sigma = hdt.hatokcal * np.std(energies,axis=0) / np.sqrt(float(len(S)))
        if ensemble:
            return hdt.hatokcal*np.mean(energies,axis=0), hdt.hatokcal*np.mean(forces,axis=0), sigma#, charges
        else:
            return hdt.hatokcal*energies, hdt.hatokcal*forces, sigma

    ''' Compute the energy and mean force of a set of conformers for the CV networks '''
    def compute_energy_conformations_net(self,X,S,netid):
        Na = X.shape[0] * len(S)

        X_split = np.array_split(X, math.ceil(Na/10000))

        energies = np.zeros((X.shape[0]), dtype=np.float64)
        forces   = np.zeros((X.shape[0], X.shape[1], X.shape[2]), dtype=np.float32)
        shift = 0
        for j,x in enumerate(X_split):
            self.ncl[netid].setConformers(confs=np.array(x,dtype=np.float64),types=list(S))
            E = self.ncl[netid].energy().copy()
            F = self.ncl[netid].force().copy()
            energies[shift:shift+E.shape[0]] = E
            forces  [shift:shift+E.shape[0]] = F
            shift += x.shape[0]

        return hdt.hatokcal*energies,hdt.hatokcal*forces#, charges

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
            charges[i] = nc.get_charges().copy()
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
            nc.setMolecule(coords=np.array(X[0],dtype=np.float64), types=list(S))

        energies = np.zeros((self.Nn, Nd), dtype=np.float64)
        forces = np.zeros((self.Nn, Nd, Na, 3), dtype=np.float32)

        for i, x in enumerate(X):
            for j, nc in enumerate(self.ncl):
                nc.setCoordinates(coords=np.array(x,dtype=np.float64))
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
        self.charge_prep=False
        self.pbc=False
    #def __init__(self, nc):
    #    # Construct pyNeuroChem class
    #    self.nc = nc

    def set_pbc(self, cell, pbc):
        self.cell = cell
        self.celi = (np.linalg.inv(cell)).astype(np.float32)
        self.ens.set_pbc(pbc[0],pbc[1],pbc[2])
        self.pbc=True

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
        mol = Atoms(symbols=S, positions=X)
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

    def energy_molecule(self,X,S):
        self.ens.set_molecule(X=X, S=list(S))
        if self.pbc:
            self.ens.set_cell((self.cell).astype(np.float32), self.celi)

        e,v = self.ens.compute_mean_energies()
        return hdt.hatokcal*e,v

    def charge_molecule(self):
        c,v = self.ens.compute_mean_charges()
        return c,v

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

class ani_torsion_scanner():
    def __init__(self,ens,fmax=0.000514221,printer=False):
        self.ens = ens
        self.fmax=fmax
        self.printer=printer

    def opt(self, rdkmol, dhls, logger='optlog.out'):
        Na = rdkmol.GetNumAtoms()
        X, S = __convert_rdkitmol_to_nparr__(rdkmol)

        atm=Atoms(symbols=S, positions=X)
        atm.set_calculator(ANIENS(self.ens))                #Set the ANI Ensemble as the calculator
        self.ens.set_molecule(X,S)


        phi_fix = []
        for d in dhls:
            phi_restraint=atm.get_dihedral(d)
            phi_fix.append([phi_restraint, d])

        c = FixInternals(dihedrals=phi_fix, epsilon=1.e-9)
        atm.set_constraint(c)

        dyn = LBFGS(atm, logfile=logger)                               #Choose optimization algorithm
        #dyn = LBFGS(atm)                               #Choose optimization algorithm

        try:
            dyn.run(fmax=self.fmax, steps=5000)         #optimize molecule to Gaussian's Opt=Tight fmax criteria, input in eV/A (I think)
        except ValueError:
            print("Opt failed: continuing")
        e=atm.get_potential_energy()*hdt.evtokcal
        s=atm.calc.stddev*hdt.evtokcal
        
        phi_value = []
        for d in dhls:
            phi_value.append(atm.get_dihedral(d)*180./np.pi)
        #X = mol.get_positions()
        if self.printer: 
            print('Angle (degrees), energy (kcal/mol), sigma= ', ["{0:.1f}".format(an) for an in  phi_value], "{0:.2f}".format(e), "{0:.2f}".format(s))
        return phi_value, e, s, atm

    def rot(self, mol, dhls, phi):

        c = mol.GetConformer(-1)

        for d,p in zip(dhls,phi):
            a0=int(d[0])
            a1=int(d[1])
            a2=int(d[2])
            a3=int(d[3])

            #print(mol.get_dihedral(d) * 180./np.pi)
            Chem.rdMolTransforms.SetDihedralDeg(c, a0, a1, a2, a3, p)
            #print(Chem.rdMolTransforms.GetDihedralDeg(c, a0, a1, a2, a3,),p)
            #print(Chem.rdMolTransforms.GetDihedralDeg(c, a0, a1, a2, a3,),p,c)

        phi_value, e, s, atm = self.opt(mol, dhls)
        return phi_value, e, s, atm

    def scan_torsion(self, mol, atid, inc, stps, GetCharge=False):
        mol_copy = copy.deepcopy(mol)

        c=mol_copy.GetConformer()
        init = []
        keys = []
        dhls = []
        for key in atid:
            keys.append(key)
            dhls.append(np.array(atid[key]))
            init.append(Chem.rdMolTransforms.GetDihedralDeg(c, int(atid[key][0]),int(atid[key][1]),int(atid[key][2]),int(atid[key][3])))

        ind = itertools.product(np.arange(stps),repeat=len(atid))
        shape = [stps for _ in range(len(atid))]

        ang = np.empty(shape+[len(atid)],dtype=np.float64)
        sig = np.empty(shape,dtype=np.float64)
        enr = np.empty(shape,dtype=np.float64)
        crd = np.empty(shape+[mol_copy.GetNumAtoms(), 3], dtype=np.float32)

        if GetCharge:
            chg = np.empty(shape+[mol_copy.GetNumAtoms()], dtype=np.float32)

        for index,i in enumerate(ind):
            aset = [-180 + j*inc for j,ai in zip(i,init)]
            phi, e, s, atm = self.rot(mol_copy, dhls, aset)
            x = atm.get_positions()

            sidx = tuple([j for j in i])
            if GetCharge:
                q = self.ens.compute_mean_charges()
                print(q[0])
                print(atm.get_chemical_symbols())
                chg[sidx] = q[0]
                print('Dipole:',np.linalg.norm(np.sum(1.889725988579*(q[0]*x.T).T,axis=0))/0.393456)

            conf = mol_copy.GetConformer(-1)
            for aid in range(conf.GetNumAtoms()):
                pos = Geometry.rdGeometry.Point3D(x[aid][0],x[aid][1],x[aid][2])
                conf.SetAtomPosition(aid, pos)

            ang[sidx] = aset
            sig[sidx] = s
            enr[sidx] = e
            crd[sidx] = x
        
        self.keys=keys
        self.X = crd
        self.Q = chg
        return ang, enr, sig

    def get_modes(self,atm,freqname="vib."):
        for f in [f for f in os.listdir(".") if freqname in f and  '.pckl' in f]:
            os.remove(f)
        new_target = open(os.devnull, "w")
        old_target, sys.stdout = sys.stdout, new_target
        atm.set_calculator(ANIENS(self.ens))
        vib = Vibrations(atm, nfree=2, name=freqname)
        vib.run()
        freq = vib.get_frequencies()
        modes = np.stack(vib.get_mode(i) for i in range(freq.size))
        vib.clean()
        sys.stdout = old_target
        return modes

    def torsional_sampler(self, mol, Ngen, atid, inc, stps, sigma=0.15,rng=0.3, freqname="vib."):
        mol_copy = copy.deepcopy(mol)

        a0=int(atid[0])
        a1=int(atid[1])
        a2=int(atid[2])
        a3=int(atid[3])

        c=mol_copy.GetConformer()
        init = Chem.rdMolTransforms.GetDihedralDeg(c, a0, a1, a2, a3)

        ang = []
        sig = []
        enr = []
        X = []
        for i in range(stps):
            n_dir = -1.0 if np.random.uniform(-1.0,1.0,1) < 0.0 else 1.0 
            phi, e, s, atm = self.rot(mol_copy, [atid], [init + n_dir*i*inc])

            x = atm.get_positions()
            conf = mol_copy.GetConformer(-1)
            for aid in range(conf.GetNumAtoms()):
                pos = Geometry.rdGeometry.Point3D(x[aid][0],x[aid][1],x[aid][2])
                conf.SetAtomPosition(aid, pos)

            if s > sigma:
                ang.append(phi)
                sig.append(s)
                enr.append(e)
                modes = self.get_modes(atm, freqname=freqname)
                for i in range(Ngen):
                    r = x+np.sum(np.random.uniform(-rng,rng,modes.shape[0])[:,np.newaxis,np.newaxis]*modes,axis=0)
                    X.append(r)
        X_null, S = __convert_rdkitmol_to_nparr__(mol)
        if len(enr) > 0:
            return np.stack(X), S, np.array(ang), np.array(enr), np.array(sig) 
        else:
            return np.empty((0,len(S),2),dtype=np.float32), S, np.empty((0),dtype=np.float64), np.empty((0),dtype=np.float64), np.empty((0),dtype=np.float64)

class aniTorsionSampler:
    def __init__(self, netdict, storedir, smilefile, Nmol, Nsamp, sigma, rng, atsym=['H', 'C', 'N', 'O'], seed=np.random.randint(0,100000,1), gpuid=0):
        self.storedir = storedir
        self.Nmol = Nmol
        self.Nsamp = Nsamp
        self.sigma = sigma
        self.rng = rng
        self.atsym = atsym

        smiles = [sm for sm in np.loadtxt(smilefile, dtype=np.str,comments=None)]
        np.random.seed(seed)
        np.random.shuffle(smiles)
        self.smiles = smiles[0:Nmol*20]

        if len(netdict) > 0:
            self.ens = ensemblemolecule(netdict['cnstfile'], netdict['saefile'], netdict['nnfprefix'], netdict['num_nets'], gpuid)

    def get_mol_set(self, smiles, atsym=['H', 'C', 'N', 'O'], MaxNa=20):
        mols = []
        for i, sm in enumerate(smiles):
            mol = Chem.MolFromSmiles(sm)
            if mol:
                mol = Chem.AddHs(mol)
                check = AllChem.EmbedMolecule(mol)
                fc = 0
                for a in mol.GetAtoms():
                    fc += a.GetFormalCharge()
                    
                if check == 0:
                    X, S = __convert_rdkitmol_to_nparr__(mol)
                    if set(S).issubset(atsym) and len(S) < MaxNa and fc == 0:
                        dec = Descriptors.NumRotatableBonds(mol)
                        if dec > 0:
                            mols.append(mol)
        return mols

    def get_index_set(self, smiles, Nmol, MaxNa, considerH=False):
        mols = self.get_mol_set(smiles, atsym=self.atsym, MaxNa=MaxNa)[0:Nmol]
        dihedrals = []
        for i, mol in enumerate(mols):
            bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds() if
                      b.GetBondType() == Chem.rdchem.BondType.SINGLE and not b.IsInRing() and len(
                      mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetNeighbors()) > 1 and len(
                      mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetNeighbors()) > 1 and 
                      mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetHybridization()!=Chem.rdchem.HybridizationType.SP and 
                      mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetHybridization()!=Chem.rdchem.HybridizationType.SP]

            if len(bonds) > 0 and not considerH:
                new_bonds = []
                for bidx,b in enumerate(bonds):
                    h_fail = False

                    nsym = []
                    for n in mol.GetAtomWithIdx(b[0]).GetNeighbors():
                        if n.GetIdx() != b[1]:
                    	    nsym.append(n.GetSymbol())

                    if set(nsym) == set('H'):
                        h_fail = True

                    nsym = []
                    for n in mol.GetAtomWithIdx(b[1]).GetNeighbors():
                        if n.GetIdx() != b[0]:
                            nsym.append(n.GetSymbol())
                    if set(nsym) == set('H'):
                        h_fail = True
                   
                    if not h_fail:
                        new_bonds.append(b)
                bonds = new_bonds

            if len(bonds) > 0:
                AllChem.MMFFOptimizeMolecule(mol)
                idx = np.random.randint(0, len(bonds))
                bonds = bonds[idx]

                for n in mol.GetAtomWithIdx(bonds[0]).GetNeighbors():
                    if n.GetIdx() != bonds[1]:
                        bond1 = n.GetIdx()

                for n in mol.GetAtomWithIdx(bonds[1]).GetNeighbors():
                    if n.GetIdx() != bonds[0]:
                        bond4 = n.GetIdx()

                dihedrals.append((mol, np.array([bond1, bonds[0], bonds[1], bond4])))
        return dihedrals

    def generate_dhl_samples(self,MaxNa=25,fmax=0.005,fpref='dhl_scan-', freqname="vib."):
        ts = ani_torsion_scanner(self.ens, fmax)
        dhls = self.get_index_set(self.smiles, self.Nmol, MaxNa=MaxNa)
        print('Runnning ',len(dhls),'torsions.',freqname)
        for i, dhl in enumerate(dhls):
            mol = dhl[0]
            X, S, p, e, s = ts.tortional_sampler(mol, self.Nsamp, dhl[1], 10.0, 36, sigma=self.sigma, rng=self.rng, freqname=freqname)
            if e.size > 0:
                comment = Chem.MolToSmiles(dhl[0]) + '[' + ' '.join([str(j) for j in dhl[1]]) + ']'
                comment = [comment for i in range(e.size * self.Nsamp)]

                hdt.writexyzfilewc(self.storedir + fpref + str(i).zfill(3) + '.xyz', xyz=X, typ=S, cmt=comment)

class MD_Sampler:
    def __init__(self, files, cnstfile, saefile, nnfprefix, Nnet, gpuid=0, sinet=False):
        self.files=files                      #List of files containing the molecules to run md on
        self.coor_train=[]                    #Where the coordinates of the molecules with high standard deviation will be saved
        self.Na_train=[]                      #Where the number of atoms of the molecules with high standard deviation will be saved
        self.S_train=[]                       #Where the atomic species of the molecules with high standard deviation will be saved
        self.hstd=[]                          #Where the standard deviation of the molecules with high standard deviation will be saved

        #The path to the network
        self.net = ensemblemolecule(cnstfile, saefile, nnfprefix, Nnet, gpuid)            #Load the network


    def get_mode(self, fi, Na, mn=0):                           #Gets the normal modes and frequencies from the gaussian log file
        fil= open(fi,'r')
        mod=[]

        string = fil.read()
        regex="\s\d\d?\d?\s\s\s\s\s?([-]?\d*\.\d\d\s\s\s?[-]?\d\d?\.\d\d\s\s\s?[-]?\d\d?\.\d\d)"

        regex2="\s\d\d?\d?\s\s\s\s\s?[-]?\d*\.\d\d\s\s\s?[-]?\d\d?\.\d\d\s\s\s?[-]?\d\d?\.\d\d\s\s\s\s?\s?([-]?\d*\.\d\d\s\s\s?[-]?\d\d?\.\d\d\s\s\s?[-]?\d\d?\.\d\d)"

        regex3="\s\d\d?\d?\s\s\s\s\s?[-]?\d*\.\d\d\s\s\s?[-]?\d\d?\.\d\d\s\s\s?[-]?\d\d?\.\d\d\s\s\s\s?\s?[-]?\d*\.\d\d\s\s\s?[-]?\d\d?\.\d\d\s\s\s?[-]?\d\d?\.\d\d\s\s\s\s?\s?([-]?\d*\.\d\d\s\s\s?[-]?\d\d?\.\d\d\s\s\s?[-]?\d\d?\.\d\d)"

        matches2 = re.findall(regex,string)
        matches3 = re.findall(regex2,string)
        matches4 = re.findall(regex3,string)
        for i in range(Na):
            mod.append(matches2[0+Na*i:Na*(i+1)])
            mod.append(matches3[0+Na*i:Na*(i+1)])
            mod.append(matches4[0+Na*i:Na*(i+1)])
        mod=mod[:-6]
        for i in range(len(mod)):
            for j in range(Na):
                mod[i][j]=mod[i][j].split( )
            mod[i]=np.array(mod[i], dtype=np.float32)
            mod[i]=np.reshape(mod[i], (Na, 3))
        return mod[mn]


    def run_md(self, f, Tmax, steps, n_steps, nmfile=None, displacement=0, min_steps=0, sig=0.34, t=0.1, nm=0, record=False):
        X, S, Na, cm = hdt.readxyz2(f)
        
        if nmfile != None:
            mode=self.get_mode(nmfile, Na, mn=nm)
            X=X+mode*np.random.uniform(-displacement,displacement)
        X=X[0]
        mol=Atoms(symbols=S, positions=X)
        mol.set_calculator(ANIENS(self.net,sdmx=20000000.0))
        f=os.path.basename(f)
        T_eff = float(random.randrange(5, Tmax, 1)) # random T (random velocities) from 0K to TK
        minstep_eff = float(random.randrange(1, min_steps, 1)) # random T (random velocities) from 0K to TK
        dyn = Langevin(mol, t * units.fs, T_eff * units.kB, 0.01)
        MaxwellBoltzmannDistribution(mol, T_eff * units.kB)
#        steps=10000    #10000=1picosecond                             #Max number of steps to run
#        n_steps = 1                                                #Number of steps to run for before checking the standard deviation
        hsdt_Na=[]
        evkcal=hdt.evtokcal

        if record==True:                                           #Records the coordinates at every step of the dynamics
            fname = f + '_record_' + str(T_eff) + 'K' + '.xyz'     #name of file to store coodinated in
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
        stddev =  0
        tot_steps = 0
        failed = False
        while (tot_steps <= steps):
            if stddev > sig and tot_steps > minstep_eff:                                #Check the standard deviation
                self.hstd.append(stddev)
                c = mol.get_positions()
                s = mol.get_chemical_symbols()
                Na=mol.get_number_of_atoms()
                self.Na_train.append(Na)
                self.coor_train.append(c)
                self.S_train.append(s)
                failed=True
                break
            else:                                           #if the standard deviation is low, run dynamics, then check it again
                tot_steps = tot_steps + n_steps
                dyn.run(n_steps)
                stddev =  evkcal*mol.calc.stddev
                c = mol.get_positions()
                s = mol.get_chemical_symbols()
                e=mol.get_potential_energy()
                #print("{0:.2f}".format(tot_steps*t),':',"{0:.2f}".format(stddev),':',"{0:.2f}".format(evkcal*e))
        return c, s, tot_steps*t, stddev, failed, T_eff


    def run_md_list(self):
        #T = random.randrange(0, 20, 1)                              # random T (random velocities) from 0K to 20K
        T=20.0
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

# --------------------------------------------------------------------
##### ----------Class Subprocess py27 for pDynamo-------- #####
# --------------------------------------------------------------------

class subproc_pDyn():
    def __init__(self, cnstfile, saefile, nnfprefix, Nnet, gpuid=0, sinet=False):
        self.coor_train=[]                    #Where the coordinates of the molecules with high standard deviation will be saved
        self.Na_train=[]                      #Where the number of atoms of the molecules with high standard deviation will be saved
        self.S_train=[]                       #Where the atomic species of the molecules with high standard deviation will be saved
        self.hstd=[]

        #The path to the network

        self.net = ensemblemolecule(cnstfile, saefile, nnfprefix, Nnet, gpuid)            #Load the network
        print ("Network LOADED")
        self.count = 0
        
        # Define environment variables for pDynamo in py27
        self.my_env = os.environ.copy()
        # -------- pDynamo ---------------------------------------------------------------------------
        self.my_env["PDYNAMO_ROOT"] = "/home/kavi/pDynamo-1.9.0"
        self.my_env["PDYNAMO_PARAMETERS"] = "/home/kavi/pDynamo-1.9.0/parameters"
        self.my_env["PDYNAMO_SCRATCH"] = "/data/kavi/pDynamo"
        self.my_env["PDYNAMO_STYLE"] = "/home/kavi/pDynamo-1.9.0/parameters/cssStyleSheets/defaultStyle.css"
        self.my_env["PYTHONPATH"] = "/home/kavi/pDynamo-1.9.0/pBabel-1.9.0:" + self.my_env["PYTHONPATH"]
        self.my_env["PYTHONPATH"] = "/home/kavi/pDynamo-1.9.0/pCore-1.9.0:" + self.my_env["PYTHONPATH"]
        self.my_env["PYTHONPATH"] = "/home/kavi/pDynamo-1.9.0/pMolecule-1.9.0:" + self.my_env["PYTHONPATH"]
        self.my_env["PYTHONPATH"] = "/home/kavi/pDynamo-1.9.0/pMoleculeScripts-1.9.0:" + self.my_env["PYTHONPATH"]
        self.my_env["PYTHONPATH"] = "/home/kavi/pDynamo-1.9.0/pGraph-0.1:" + self.my_env["PYTHONPATH"]
        self.my_env["PDYNAMO_PMOLECULESCRIPTS"] = "/home/kavi/pDynamo-1.9.0/pMoleculeScripts-1.9.0"
        self.my_env["PYTHONPATH"] = "/home/kavi/pDyn_ANI:" + self.my_env["PYTHONPATH"]
        
        # -------- Boost -------------------------------------------------------------------------------
        self.my_env["BOOST_ROOT"] = "/home/kavi/boost_1_63_0"
        self.my_env["CPLUS_INCLUDE_PATH"] = "/home/kavi/boost_1_63_0/boost:" + self.my_env["CPLUS_INCLUDE_PATH"]
        self.my_env["LD_LIBRARY_PATH"] = "/home/kavi/boost_1_63_0/lib:" + self.my_env["LD_LIBRARY_PATH"]
        
        # -------- NeuroChem ---------------------------------------------------------------------------
        self.my_env["NC_ROOT"] = "/home/kavi/NeuroChem/build"
        self.my_env["PATH"] = "/home/kavi/NeuroChem/build/bin:" + self.my_env["PATH"]
        self.my_env["LD_LIBRARY_PATH"] = "/home/kavi/NeuroChem/build/lib:" + self.my_env["LD_LIBRARY_PATH"]
        self.my_env["PYTHONPATH"] = "/home/kavi/NeuroChem/build/lib:" + self.my_env["PYTHONPATH"]
        
    def write_pDynOPT(self, num_rxn, pDyn_dir, wkdir, cnstfilecv, saefilecv, Nnt):
    	fname = pDyn_dir + 'OPT.py'
    	sf = open(fname, 'w')
    	sf.write('import numpy as np'+ '\n')
    	sf.write('from Definitions import *'+ '\n')
    	sf.write('from QCModelANI import QCModelANI'+ '\n')
    	sf.write(''+ '\n')
    	sf.write('# Define the QC model using ORCA'+ '\n')
    	sf.write('wkdir = ' + '"'+ '%s' %wkdir + '"'+ '\n')
    	sf.write('cnstfilecv  = ' + '"'+ '%s' %cnstfilecv  + '"'+ '\n')
    	sf.write('saefilecv = ' + '"'+ '%s' %saefilecv + '"'+ '\n')
    	sf.write('Nnet = ' + '%s' %Nnt + '\n')
    	sf.write(''+ '\n')
    	sf.write('qcModel = QCModelANI (wkdir, cnstfilecv, saefilecv, Nnet)' + '\n')
    	sf.write(''+ '\n')
    	sf.write('# Define the molecule'+ '\n')
    	sf.write('for i in range(%i):' %num_rxn+ '\n')
    	sf.write('    molecule = XYZFile_ToSystem ( os.path.join ( xyzPath, "DA%02i.xyz" % (i+1) ) )'+ '\n')
    	sf.write('    fixedAtoms = Selection.FromIterable([0,1,2,3,4,5])'+ '\n')
    	sf.write('    molecule.DefineFixedAtoms(fixedAtoms)'+ '\n')
    	sf.write('    # Define the energy model.'+ '\n')
    	sf.write('    molecule.DefineQCModel ( qcModel )'+ '\n')
    	sf.write('    molecule.Summary ( )'+ '\n')
    	sf.write('    # Save a copy of the starting coordinates.'+ '\n')
    	sf.write('    coordinates3 = Clone ( molecule.coordinates3 )'+ '\n')
    	sf.write(''+ '\n')
    	sf.write('    # Optimization'+ '\n')
    	sf.write('    FIREMinimize_SystemGeometry ( molecule              ,    '+ '\n')
    	sf.write('                                  logFrequency         =  100   ,'+ '\n')
    	sf.write('                                  maximumIterations    = 10000  ,'+ '\n')
    	sf.write('                                  rmsGradientTolerance =  0.01 )'+ '\n')    
    	sf.write('    # Save the coordinates.'+ '\n')
    	sf.write('    molecule.label = "FIRE optimized."'+ '\n')
    	sf.write('    if os.path.isfile(os.path.join (F_OPT, "OPT_TS%02i.xyz" % (i+1) )):'+ '\n')
    	sf.write('        os.remove(os.path.join (F_OPT, "OPT_TS%02i.xyz" % (i+1) ))'+ '\n')
    	sf.write('    XYZFile_FromSystem ( os.path.join (F_OPT, "OPT_TS%02i.xyz" % (i+1) ), molecule )'+ '\n')
    	print ("OPT.py file written!")

    def write_pDynTS(self, num_rxn, pDyn_dir, wkdir, cnstfilecv, saefilecv, Nnt):
    	fname = pDyn_dir + 'TS.py'
    	sf = open(fname, 'w')
    	sf.write('import numpy as np'+ '\n')
    	sf.write('from Definitions import *'+ '\n')
    	sf.write('from QCModelANI import QCModelANI'+ '\n')
    	sf.write(''+ '\n')
    	sf.write('# Define the QC model using ORCA'+ '\n')
    	sf.write('wkdir = ' + '"'+ '%s' %wkdir + '"'+ '\n')
    	sf.write('cnstfilecv  = ' + '"'+ '%s' %cnstfilecv  + '"'+ '\n')
    	sf.write('saefilecv = ' + '"'+ '%s' %saefilecv + '"'+ '\n')
    	sf.write('Nnet = ' + '%s' %Nnt + '\n')
    	sf.write('img_freq = np.arange(100, dtype=float)'+ '\n')
    	sf.write('qcModel = QCModelANI (wkdir, cnstfilecv, saefilecv, Nnet)' + '\n')
    	sf.write(''+ '\n')
    	sf.write('# Define the molecule'+ '\n')
    	sf.write('for i in range(%i):' %num_rxn+ '\n')
    	sf.write('    molecule = XYZFile_ToSystem ( os.path.join (F_OPT, "OPT_TS%02i.xyz" % (i+1) ) )'+ '\n')
    	sf.write('    molecule.DefineQCModel ( qcModel )'+ '\n')
    	sf.write('    molecule.Summary ( )'+ '\n')
    	sf.write(''+ '\n')
    	sf.write('    # Optimization'+ '\n')
    	sf.write('    BakerSaddleOptimize_SystemGeometry ( molecule,'+ '\n')
    	sf.write('                                  logFrequency         =  1   ,'+ '\n')
    	sf.write('                                  maximumIterations    = 1000  ,'+ '\n')
    	sf.write('                                  rmsGradientTolerance =  1.0e-3 )'+ '\n')    
    	sf.write('    # Save the coordinates.'+ '\n')
    	sf.write('    if os.path.isfile(os.path.join (ANI_TS, "ANI_TS-%02i.xyz" % (i+1) )):'+ '\n')
    	sf.write('        os.remove(os.path.join (ANI_TS, "ANI_TS-%02i.xyz" % (i+1) ))'+ '\n')
    	sf.write('    molecule.label = "DA- TS ANI optimized."'+ '\n')
    	sf.write('    XYZFile_FromSystem ( os.path.join (ANI_TS, "ANI_TS-%02i.xyz" % (i+1) ), molecule )'+ '\n')
    	sf.write(''+ '\n')
    	sf.write('    # Calculate the normal modes.'+ '\n')
    	sf.write('    neg_freq_OPT = NormalModes_SystemGeometry ( molecule, modify = "project" )'+ '\n')
    	sf.write('    freq_OPT = getattr( neg_freq_OPT, "frequencies", None)'+ '\n')
    	sf.write('    img_freq[i] = freq_OPT[0]'+ '\n')
    	sf.write(''+ '\n')
    	sf.write('for i in range(%i):' %num_rxn+ '\n')
    	sf.write('    print "{:5.0f}".format(i+1) , "{:30.5f}".format(img_freq[i])'+ '\n')
    	print ("TS.py file written!")

    def write_pDynIRC(self, num_rxn, pDyn_dir, wkdir, cnstfilecv, saefilecv, Nnt):
    	fname = pDyn_dir + 'IRC.py'
    	sf = open(fname, 'w')
    	sf.write('import numpy as np'+ '\n')
    	sf.write('from Definitions import *'+ '\n')
    	sf.write('from QCModelANI import QCModelANI'+ '\n')
    	sf.write('from pBabel import XYZTrajectoryFileWriter'+ '\n')
    	sf.write(''+ '\n')
    	sf.write('# Define the QC model using ORCA'+ '\n')
    	sf.write('wkdir = ' + '"'+ '%s' %wkdir + '"'+ '\n')
    	sf.write('cnstfilecv  = ' + '"'+ '%s' %cnstfilecv  + '"'+ '\n')
    	sf.write('saefilecv = ' + '"'+ '%s' %saefilecv + '"'+ '\n')
    	sf.write('Nnet = ' + '%s' %Nnt + '\n')
    	sf.write('qcModel = QCModelANI (wkdir, cnstfilecv, saefilecv, Nnet)' + '\n')
    	sf.write(''+ '\n')
    	sf.write('# Define the molecule'+ '\n')
    	sf.write('for i in range(%i):' %num_rxn+ '\n')
    	sf.write('    molecule = XYZFile_ToSystem ( os.path.join ( xyzPath, "DA%02i.xyz" % (i+1) ) )'+ '\n')
    	sf.write('    molecule.coordinates3 = XYZFile_ToCoordinates3 ( os.path.join (ANI_TS, "ANI_TS-%02i.xyz" % (i+1) ) )'+ '\n')
    	sf.write('    molecule.DefineQCModel ( qcModel )'+ '\n')
    	sf.write('    molecule.Summary ( )'+ '\n')
    	sf.write(''+ '\n')
    	sf.write('    # Calculate an energy.'+ '\n')
    	sf.write('    molecule.Energy ( )'+ '\n')
    	sf.write(''+ '\n')
    	sf.write('    # . Create an output trajectory.'+ '\n')
    	sf.write('    if os.path.isfile(os.path.join ( ANI_IRC, "ANI_IRC-%02i.xyz" % (i+1) )):'+ '\n')
    	sf.write('        os.remove(os.path.join ( ANI_IRC, "ANI_IRC-%02i.xyz" % (i+1) ))'+ '\n')
    	sf.write('    trajectory = XYZTrajectoryFileWriter ( os.path.join ( ANI_IRC, "ANI_IRC-%02i.xyz" % (i+1) ), molecule )'+ '\n')
    	sf.write(''+ '\n')
    	sf.write('    # Optimization'+ '\n')
    	sf.write('    SteepestDescentPath_SystemGeometry ( molecule,                       '+ '\n')
    	sf.write('                                  functionStep      = 2.0,        '+ '\n')
    	sf.write('                                  logFrequency      = 10,        '+ '\n')
    	sf.write('                                  maximumIterations    = 1000  ,'+ '\n')
    	sf.write('                                  pathStep          = 0.025,      '+ '\n')
    	sf.write('                                  saveFrequency     = 10,        '+ '\n') 
    	sf.write('                                  trajectory        = trajectory, '+ '\n')
    	sf.write('                                  useMassWeighting  = True        )'+ '\n') 
    	print ("IRC.py file written!")
       
    def subprocess_cmd(self, python3_command, shl, logfile):
        with open(logfile,"wb") as out, open("stderr.txt","wb") as err:
            process = subprocess.Popen(python3_command.split(), env=self.my_env, shell=shl, stdout=out, stderr=err, universal_newlines=True)
            process.wait()
            chk = process.poll()
            return chk

    def getIRCpoints_toXYZ(self, n_points, inpf, filename, f_path):
        re1 = re.compile('\d+?\n.*?\n(?:[A-Z][a-z]?.+?(?:\n|$))+')
        gfile = open(inpf,'r').read()
        blocks = re1.findall(gfile)  
        for i in range(n_points):
            f = open(f_path + filename[:-4] + '-%03i.xyz' %(i+1), 'w') 
            f.write(blocks[i])
            f.close()
        print ("Individual IRC points obtained!")


    def check_stddev(self, f, sig):
        mol=read(f)                                          #Read the molecule for the file
        mol.set_calculator(ANIENS(self.net,sdmx=20000000.0))
        evkcal=hdt.evtokcal

        e=mol.get_potential_energy()                        #Calculate the energy of the molecule. Must be done to get the standard deviation
        s=mol.calc.stddev
        stddev =  s*evkcal

        if stddev > sig:        
            self.hstd.append(stddev)                            #Check the standard deviation
            c = mol.get_positions(wrap=False)
            s = mol.get_chemical_symbols()
            Na=mol.get_number_of_atoms()
            self.Na_train.append(Na)
            self.coor_train.append(c)
            self.S_train.append(s)
            return stddev
    
        else:                                           #if the standard deviation is low
            return stddev
        
    def get_nm(self, f):
        nmc_list = []
        mol=read(f)                                          #Read the molecule for the file
        mol.set_calculator(ANIENS(self.net,sdmx=20000000.0))
        vib = Vibrations(mol, nfree=2)
        vib.run()
        ANI_freq = vib.get_frequencies()
        for i in range(len(ANI_freq)):
            nm = vib.get_mode(i)
            nmc_list.append(nm)
        vib.clean()
        return nmc_list
        print ("Modes are generated!")

    def write_nm_xyz(self, fname):                                #Calling this function wirtes all the structures with high standard deviations to and xyz file
        alt=open('%s' %fname, 'w')
        for i in range(len(self.Na_train)):
            cm = 'comment'
            alt.write('%s\n%s\n' %(str(self.Na_train[i]), cm))
            for j in range(len(self.S_train[i])):
                alt.write('%s %f %f %f \n' %(self.S_train[i][j], self.coor_train[i][j][0], self.coor_train[i][j][1], self.coor_train[i][j][2]))
        alt.close()
        print ("XYZ file with high stdev struc. generated!")




