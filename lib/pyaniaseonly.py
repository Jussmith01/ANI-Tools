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
    def compute_energy_conformations(self,X,S):
        Na = X.shape[0] * len(S)

        X_split = np.array_split(X, math.ceil(Na/10000))

        energies = np.zeros((self.Nn, X.shape[0]), dtype=np.float64)
        forces   = np.zeros((self.Nn, X.shape[0], X.shape[1], X.shape[2]), dtype=np.float32)
        shift = 0
        for j,x in enumerate(X_split):
            for i, nc in enumerate(self.ncl):
                nc.setConformers(confs=x,types=list(S))
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
                nc.setConformers(confs=x,types=list(S))
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
            self.ncl[netid].setConformers(confs=x,types=list(S))
            E = self.ncl[netid].energy().copy()
            energies[shift:shift+E.shape[0]] = E
            shift += x.shape[0]

        return hdt.hatokcal*energies#, charges

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