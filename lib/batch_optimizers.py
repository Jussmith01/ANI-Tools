import pyNeuroChem as pyc
import hdnntools as hdt
import numpy as np
import time

from ase_interface import ANIENS
from ase_interface import ensemblemolecule

import pyaniasetools as pya

# pyNeuroChem molecule batched optimizer
class moleculeOptimizer(pya.anicrossvalidationmolecule):
    def __init__(self, cns, sae, nnf, Nn, gpuid=0):
        pya.anicrossvalidationmolecule.__init__(self,cns, sae, nnf, Nn, gpuid)

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

                #print('Gamma:', gamma, np.power(np.linalg.norm(f), 2), np.power(np.linalg.norm(fn), 2))

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

# pyNeuroChem conformer batched optimizer
class conformerOptimizer(pya.anicrossvalidationconformer):
    def __init__(self, cns, sae, nnf, Nn, gpuid=0):
        pya.anicrossvalidationconformer.__init__(self,cns, sae, nnf, Nn, gpuid)
        self.__energies = None

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