import pyNeuroChem as pyc
import hdnntools as hdt
import numpy as np
import time

import math

from ase_interface import ANIENS
from ase_interface import ensemblemolecule

# ------------------------------------------------------------------------
#  Class for ANI cross validaiton computer for multiple conformers a time
# ------------------------------------------------------------------------
class anicrossvalidationconformer(object):
    ''' Constructor '''
    def __init__(self,cnstfile,saefile,nnfprefix,Nnet,gpuid=[0], sinet=False):
        # Number of networks
        self.Nn = Nnet

        gpua = [gpuid[int(np.floor(i/(Nnet/len(gpuid))))] for i in range(self.Nn)]

        # Construct pyNeuroChem class
        self.ncl = [pyc.conformers(cnstfile, saefile, nnfprefix+str(i)+'/networks/', gpua[i], sinet) for i in range(self.Nn)]
        #self.ncl = [pync.conformers(cnstfile, saefile, nnfprefix+str(1)+'/networks/', gpuid, sinet) for i in range(self.Nn)]

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

##--------------------------------------------------------------------------------
##  Class for ANI cross validaiton computer for a single molecule at a time
##--------------------------------------------------------------------------------
class anicrossvalidationmolecule(object):
    def __init__(self, cnstfile, saefile, nnfprefix, Nnet, gpuid=0, sinet=False):
        # Number of networks
        self.Nn = Nnet

        # Construct pyNeuroChem class
        self.ncl = [pyc.molecule(cnstfile, saefile, nnfprefix + str(i) + '/networks/', gpuid, sinet) for i in
                    range(self.Nn)]

    def set_molecule(self,X,S):
        for nc in self.ncl:
            nc.setMolecule(coords=X, types=list(S))

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

# ------------------------------------------------------------------------
#   pyNeuroChem -- single molecule batched optimizer (one at a time)
# ------------------------------------------------------------------------
class moleculeOptimizer(anicrossvalidationmolecule):
    def __init__(self, cns, sae, nnf, Nn, gpuid=0):
        anicrossvalidationmolecule.__init__(self,cns, sae, nnf, Nn, gpuid)

    # Gradient descent optimizer
    def optimizeGradientDescent (self, X, S, alpha=0.0004, convergence=0.027, maxsteps=10000, printer=True, printstep=50):

        Xf = np.zeros(X.shape,dtype=np.float32)
        for i,x in enumerate(X):
            print('--Optimizing conformation:',i,'--')
            xn = np.array(x, np.float32)
            for j in range(maxsteps):
                e, f, p = self.compute_energies_and_forces_molecule(xn, S)
                xn = xn + alpha*f

                if printer and j%printstep==0:
                    print('  -',j,"{0:.3f}".format(e),
                                  "{0:.4f}".format(np.abs(f).sum()),
                                  "{0:.4f}".format(np.max(np.abs(f))),
                                  "{0:.4f}".format(p))

                if np.max(np.abs(f)) < convergence:
                    break

            print('Complete')
            print('  -', j, "{0:.3f}".format(e),
                  "{0:.4f}".format(np.abs(f).sum()),
                  "{0:.4f}".format(np.max(np.abs(f))),
                  "{0:.4f}".format(p))

            Xf[i] = xn

        return Xf

    # Conjugate gradient optimizer
    def optimizeConjugateGradient (self, X, S, alpha=0.0004, convergence=0.027, maxsteps=10000, printer=True, printstep=50):

        Xf = np.zeros(X.shape,dtype=np.float32)

        for i,x in enumerate(X):
            print('--Optimizing conformation:',i,'--')
            xn = np.array(x, np.float32)
            hn = np.zeros(xn.shape, dtype=np.float32)
            fn = np.zeros(xn.shape, dtype=np.float32)
            for j in range(maxsteps):
                e, f, p = self.compute_energies_and_forces_molecule(xn, S)

                if j != 0:
                    gamma = np.power(np.linalg.norm(f), 2) / np.power(np.linalg.norm(fn), 2)
                else:
                    gamma = 0

                fn = f

                hn = f + gamma * hn
                xn = xn + alpha * hn

                if printer and j%printstep==0:
                    print('  -',j,"{0:.3f}".format(e),
                                  "{0:.4f}".format(np.abs(f).sum()),
                                  "{0:.4f}".format(np.max(np.abs(f))),
                                  "{0:.4f}".format(p))

                if np.max(np.abs(f)) < convergence:
                    break

            print('Complete')
            print('  -', j, "{0:.3f}".format(e),
                  "{0:.4f}".format(np.abs(f).sum()),
                  "{0:.4f}".format(np.max(np.abs(f))),
                  "{0:.4f}".format(p))

            Xf[i] = xn

        return Xf

    # In the works function for optimizing one at a time with lBFGS
    def optimizelBFGS (self, X, S, alpha=0.0004, convergence=0.027, maxsteps=10000, printer=True, printstep=50):

        Xf = np.zeros(X.shape,dtype=np.float32)
        for i,x in enumerate(X):
            print('--Optimizing conformation:',i,'--')
            xn = np.array(x, np.float32)
            for j in range(maxsteps):
                e, f, p = self.compute_energies_and_forces_molecule(xn, S)
                xn = xn + alpha*f

                if printer and j%printstep==0:
                    print('  -',j,"{0:.3f}".format(e),
                                  "{0:.4f}".format(np.abs(f).sum()),
                                  "{0:.4f}".format(np.max(np.abs(f))),
                                  "{0:.4f}".format(p))

                if np.max(np.abs(f)) < convergence:
                    break

            print('Complete')
            print('  -', j, "{0:.3f}".format(e),
                  "{0:.4f}".format(np.abs(f).sum()),
                  "{0:.4f}".format(np.max(np.abs(f))),
                  "{0:.4f}".format(p))

            Xf[i] = xn

        return Xf

# ------------------------------------------------------------------------
#       pyNeuroChem -- batched conformer optimizer (all at once)
# ------------------------------------------------------------------------
class conformerOptimizer(anicrossvalidationconformer):
    def __init__(self, cns, sae, nnf, Nn, gpuid=0):
        anicrossvalidationconformer.__init__(self,cns, sae, nnf, Nn, gpuid)
        self.__energies = None

    # Optimizing "all-at-once" with gradient descent
    def optimizeGradientDescent (self, X, S, alpha=0.0004, convergence=0.027, maxsteps=10000, printer=True, printstep=50):

        print('--Optimizing all conformations--')

        Nm = X.shape[0]
        mask = np.full(Nm, True, dtype=bool)
        Xn = np.array(X, np.float32)
        Fn = np.zeros(Xn.shape, dtype=np.float32)

        for j in range(maxsteps):

            Nm = np.bincount(mask)[1]
            idx = np.argsort(np.invert(mask))

            mask = mask[idx]
            Xn = Xn[idx]
            Fn = Fn[idx]

            E, F, p = self.compute_energyandforce_conformations(Xn[0:Nm], S)

            Fn[0:Nm] = F
            Xn[0:Nm] = Xn[0:Nm] + alpha * Fn[0:Nm]

            conv = np.where(np.max(np.abs(F.reshape(Nm, -1)), axis=1) < convergence)
            mask[conv] = False

            if printer and j%printstep==0:
                print('  -',str(j).zfill(4),')',np.bincount(mask)[0],'of',X.shape[0],
                            "{0:.4f}".format(np.sum(np.abs(Fn.reshape(X.shape[0],-1)),axis=1).mean()),
                            "{0:.4f}".format(np.max(np.abs(Fn.reshape(X.shape[0],-1)),axis=1).mean()),
                            "{0:.4f}".format(p.mean()))

            if np.bincount(mask)[0] == X.shape[0]:
                break

        print('Complete')
        print('  -', str(j).zfill(4), ')', np.bincount(mask)[0], 'of', X.shape[0],
              "{0:.4f}".format(np.sum(np.abs(Fn.reshape(X.shape[0], -1)), axis=1).mean()),
              "{0:.4f}".format(np.max(np.abs(Fn.reshape(X.shape[0], -1)), axis=1).mean()),
              "{0:.4f}".format(p.mean()))

        return Xn

    # Optimizing "all-at-once" with conjugate gradient
    def optimizeConjugateGradient (self, X, S, alpha=0.0001, convergence=0.027, maxsteps=10000, printer=True, printstep=50):

        print('--Optimizing all conformations--')

        Nm = X.shape[0]
        mask = np.full(Nm, True, dtype=bool)
        Xn = np.array(X, np.float32)
        Hn = np.zeros(Xn.shape, dtype=np.float32)
        Fn = np.zeros(Xn.shape, dtype=np.float32)

        for j in range(maxsteps):
            Nm = np.bincount(mask)[1]
            idx = np.argsort(np.invert(mask))

            mask = mask[idx]
            Xn = Xn[idx]
            Hn = Hn[idx]
            Fn = Fn[idx]

            E, F, p = self.compute_energyandforce_conformations(Xn[0:Nm], S)
            F = F

            # Do conjugate grad for the lower force structures
            if j != 0:
                gamma = np.power(np.linalg.norm(F.reshape(Nm, -1), axis=1), 2) / np.power(
                    np.linalg.norm(Fn[0:Nm].reshape(Nm, -1), axis=1), 2)

            else:
                gamma = np.zeros(F.shape[0])

            Fn[0:Nm] = F

            #print(F.shape,Hn.shape,gamma.shape)

            Hn[0:Nm] = -F + gamma[:,np.newaxis,np.newaxis] * Hn[0:Nm]

            #Hn[0:Nm] = (F.reshape(Nm, -1).T + gamma * Hn[0:Nm].reshape(Nm, -1).T).T.reshape(Nm, -1, 3)
            Xn[0:Nm] = Xn[0:Nm] - alpha * Hn[0:Nm]


            # Check convergence
            conv = np.where(np.max(np.abs(F.reshape(Nm, -1)), axis=1) < convergence)
            mask[conv] = False

            if printer and j%printstep==0:
                print('  -',str(j).zfill(4),')',np.bincount(mask)[0],'of',X.shape[0],
                            "{0:.4f}".format(np.sum(np.abs(Fn.reshape(X.shape[0],-1)),axis=1).mean()),
                            "{0:.4f}".format(np.max(np.abs(Fn.reshape(X.shape[0],-1)),axis=1).mean()),
                            "{0:.4f}".format(p.mean()))

            if np.bincount(mask)[0] == X.shape[0]:
                break

        print('Complete')
        print('  -', str(j).zfill(4), ')', np.bincount(mask)[0], 'of', X.shape[0],
              "{0:.4f}".format(np.sum(np.abs(Fn.reshape(X.shape[0], -1)), axis=1).mean()),
              "{0:.4f}".format(np.max(np.abs(Fn.reshape(X.shape[0], -1)), axis=1).mean()),
              "{0:.4f}".format(p.mean()))

        return Xn