from ase_interface import ANI
from ase_interface import ANIENS
from ase_interface import ensemblemolecule

import pyNeuroChem as pync

import numpy as np
import hdnntools as hdn
import nmstools as nmt

import  ase
from ase import Atoms
from ase.optimize import LBFGS
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import random

# Snagged from here: http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

'''Molecule fragmentation class'''
class molfrag():
    def __init__(self,xyz,spc):
        self.xyz = xyz
        self.spc = spc

    def fragment(self, i, cut1, cut2):
        Xc = self.xyz[i]
        new_spc = [self.spc[i]]
        new_xyz = [self.xyz[i]-Xc]

        for j,x in enumerate(self.xyz):
            if i != j:
                dij = np.linalg.norm(self.xyz[i]-x)
                if dij <= cut1:
                    new_spc.append(self.spc[j])
                    new_xyz.append(x-Xc)
                elif dij > cut1 and dij < cut2 and self.spc[j] is 'H':
                    #print('Add shell:', j)
                    new_spc.append(self.spc[j])
                    new_xyz.append(x-Xc)

        del_elm = []
        for j,x1 in enumerate(new_xyz):
            if new_spc[j] == 'H':
                lone = True
                for k,x2 in enumerate(new_xyz):
                    if j != k and np.linalg.norm(x1-x2) < 1.1:
                        lone = False
                if not lone:
                    del_elm.append(j)
            else:
                del_elm.append(j)

        new_xyz = np.vstack(new_xyz)[del_elm]
        new_spc = np.array(new_spc)[del_elm]

        return new_xyz,new_spc

    def order_by_type(self,frags):
        for i,f in enumerate(frags):
            sort_idx = np.argsort(f[0])
            srt_spc = f[0][sort_idx]
            srt_xyz = f[1][sort_idx]
            frags[i] = (srt_spc,srt_xyz)


    def get_all_frags(self,type, cut1, cut2):
        fragments = []
        for i,s in enumerate(self.spc):
            if s is type:
                xyz, spc = self.fragment(i,cut1,cut2)
                fragments.append((spc,xyz))

        self.order_by_type(fragments)
        return fragments

'''Dimer generator'''
class dimergenerator():
    def __init__(self, cnstfile, saefile, nnfprefix, Nnet, molecule_list, gpuid=0, sinet=False):
        # Molecules list
        self.mols = molecule_list

        # Number of networks
        self.Nn = Nnet

        # Build ANI ensemble
        self.aens = ensemblemolecule(cnstfile, saefile, nnfprefix, Nnet, gpuid)

        # Construct pyNeuroChem class
        #self.ncl = [pync.molecule(cnstfile, saefile, nnfprefix+str(i)+'/networks/', gpuid, sinet) for i in range(self.Nn)]

    def __generategarbagebox__(self,Nm, L, T):
        self.X = np.empty((0, 3), dtype=np.float32)
        self.S = []
        self.C = np.zeros((Nm, 3), dtype=np.float32)

        rint = np.random.randint(len(self.mols), size=Nm)
        self.Na = np.zeros(Nm, dtype=np.int32)
        self.ctd = []

        pos = 0
        plc = 0
        attempts = 0
        for idx, j in enumerate(rint):
            if attempts >= 10000:
                break
            #x = self.mols[j]['coordinates']
            #s = self.mols[j]['species']

            s   = self.mols[j]['species']
            xyz = self.mols[j]['coordinates']
            nmc = self.mols[j]['nmdisplacements']
            frc = self.mols[j]['forceconstant']

            nms = nmt.nmsgenerator(xyz,nmc,frc,s,T,minfc=5.0E-2)
            x = nms.get_Nrandom_structures(1)[0]

            # Apply a random rotation
            M = rand_rotation_matrix()
            x = np.dot(x,M.T)

            maxd = hdn.generatedmatsd3(x).flatten().max() / 2.0
            #print('after:', maxd)

            fail = True
            while fail and attempts < 10000:
                ctr = np.random.uniform(2.0*maxd + 1.0, L - 2.0*maxd - 1.0, (3))
                fail = False
                for cid,c in enumerate(self.ctd):
                    if np.linalg.norm(c[0] - ctr) < maxd + c[1] + 10.0:
                        # search for atoms within r angstroms
                        minv = 1000.0
                        for xi in self.X[c[2]:c[2]+self.Na[cid],:]:
                            for xj in x + ctr:
                                dij = np.linalg.norm(xi-xj)
                                if dij < minv:
                                    minv = dij
                        if minv < 1.4:
                            fail = True
                            attempts += 1

                if not fail:
                    plc += 1
                    attempts = 0
                    #print('Added:',plc)
                    self.ctd.append((ctr, maxd, pos))
                    self.X = np.vstack([self.X, x + ctr])
                    self.Na[idx] = len(s)
                    pos += len(s)
                    self.S.extend(s)

        self.Na = self.Na[0:plc]

    def init_dynamics(self, Nm, V, L, dt, T):
        self.L = L

        # Generate the box of junk
        self.__generategarbagebox__(Nm, L, T)

        # Make mol
        self.mol = Atoms(symbols=self.S, positions=self.X)

        # Set box and PBC
        self.mol.set_cell(([[L, 0, 0],
                            [0, L, 0],
                            [0, 0, L]]))

        self.mol.set_pbc((True, True, True))

        # Set ANI calculator
        # Set ANI calculator
        self.mol.set_calculator(ANIENS(self.aens))
        #self.mol.set_calculator(ANI(False))
        #self.mol.calc.setnc(self.ncl[0])


        # Give molecules random velocity
        acc_idx = 0
        vel = np.empty_like(self.X)
        for n in self.Na:
            rv = np.random.uniform(-V, V, size=(3))
            for k in range(n):
                vel[acc_idx + k, :] = rv
            acc_idx += n
        #print(vel)

        self.mol.set_velocities(vel)

        # Declare Dyn
        self.dyn = Langevin(self.mol, dt * units.fs, T * units.kB, 0.1)

    def run_dynamics(self, Ni):
        # Open MD output
        #mdcrd = open(xyzfile, 'w')

        # Open MD output
        #traj = open(trajfile, 'w')

        # Define the printer
        #def printenergy(a=self.mol, d=self.dyn):  # store a reference to atoms in the definition.
        #    """Function to print the potential, kinetic and total energy."""
        #    epot = a.get_potential_energy() / len(a)
        #    ekin = a.get_kinetic_energy() / len(a)

        #    print('Step: %d Size: %d Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
        #          'Etot = %.3feV' % (d.get_number_of_steps(), len(a), epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

        # Attach the printer
        #self.dyn.attach(printenergy, interval=4)

        self.dyn.run(Ni) # Do Ni steps

        # Open MD output
        #mdcrd.close()

        # Open MD output
        #traj.close()

    def __fragmentbox__(self, file, sig_cut):
        self.X = self.mol.get_positions()

        self.frag_list = []

        self.Nd = 0
        self.Nt = 0

        max_sig = 0.0

        for i in range(len(self.Na)):
            si = self.ctd[i][2]
            di = self.ctd[i][1]
            Nai = self.Na[i]
            Xci = np.sum(self.X[si:si+Nai,:], axis=0)/ Nai
            Xi = self.X[si:si + Nai, :]

            if np.all(Xci > di) and np.all(Xci < self.L-di):

                for j in range(i+1, len(self.Na)):
                    sj = self.ctd[j][2]
                    dj = self.ctd[j][1]
                    Naj = self.Na[j]
                    Xcj = np.sum(self.X[sj:sj+Naj,:], axis=0)/ Naj
                    Xj = self.X[sj:sj+Naj,:]

                    if np.all(Xcj > dj) and np.all(Xcj < self.L - dj):
                        dc = np.linalg.norm(Xci - Xcj)
                        if dc < di + dj + 6.0:
                            min = 10.0
                            for ii in range(Nai):
                                Xiii = Xi[ii]
                                for jj in range(Naj):
                                    Xjjj = Xj[jj]
                                    v = np.linalg.norm(Xiii-Xjjj)
                                    if v < min:
                                        min = v

                            if min < 4.8 and min > 1.1:
                                Xf = np.vstack([Xi, Xj])
                                Sf = self.S[si:si+Nai]
                                Sf.extend(self.S[sj:sj+Naj])

                                Xcf = np.sum(Xf, axis=0) / (Nai+Naj)
                                Xf = Xf - Xcf

                                E = np.empty(len(self.aens.ncl), dtype=np.float64)
                                for id,nc in enumerate(self.aens.ncl):
                                    nc.setMolecule(coords=np.array(Xf,dtype=np.float32), types=Sf)
                                    E[id] = nc.energy()[0]

                                sig = np.std(hdn.hatokcal*E)/np.sqrt(Nai+Naj)

                                self.Nt += 1
                                if sig > sig_cut:
                                    if sig > max_sig:
                                        max_sig = sig
                                    self.Nd += 1
                                    hdn.writexyzfile(file+str(i).zfill(4)+'-'+str(j).zfill(4)+'.xyz', Xf.reshape(1,Xf.shape[0],3), Sf)
                                    self.frag_list.append(dict({'coords': Xf,'spec': Sf}))
                                    #print(dc, Sf, sig)
        return max_sig

'''Cluster generator'''
class clustergenerator():
    def __init__(self, cnstfile, saefile, nnfprefix, Nnet, solv_list, solu_list, gpuid=0, sinet=False):
        # Molecules list
        self.solv_list = solv_list
        self.solu_list = solu_list

        # Number of networks
        self.Nn = Nnet

        # Build ANI ensemble
        self.aens = ensemblemolecule(cnstfile, saefile, nnfprefix, Nnet, gpuid)

        self.edgepad = 1.0
        self.mindist = 2.0

    #def closeoutputs(self):
    #    self.mdcrd.close()
    #    self.traj.close()

    def __generategarbagebox__(self,Nm, Nembed, L, T):
        self.X = np.empty((0, 3), dtype=np.float32)
        self.S = []
        self.C = np.zeros((Nm, 3), dtype=np.float32)

        rint = np.random.randint(len(self.solv_list), size=Nm)
        self.Na = np.zeros(Nm, dtype=np.int32)
        self.ctd = []

        pos = 0
        plc = 0
        for idx, j in enumerate(rint):
            if idx < Nembed:
                ri = np.random.randint(len(self.solu_list), size=1)
                s   = self.solu_list[ri[0]]['species']
                xyz = self.solu_list[ri[0]]['coordinates']
                nmc = self.solu_list[ri[0]]['nmdisplacements']
                frc = self.solu_list[ri[0]]['forceconstant']

                nms = nmt.nmsgenerator(xyz,nmc,frc,s,T,minfc=5.0E-2)
                x = nms.get_Nrandom_structures(1)[0]

            else:
                s   = self.solv_list[j]['species']
                xyz = self.solv_list[j]['coordinates']
                nmc = self.solv_list[j]['nmdisplacements']
                frc = self.solv_list[j]['forceconstant']

                nms = nmt.nmsgenerator(xyz,nmc,frc,s,T,minfc=5.0E-2)
                x = nms.get_Nrandom_structures(1)[0]

            # Apply a random rotation
            M = rand_rotation_matrix()
            x = np.dot(x,M.T)

            maxd = hdn.generatedmatsd3(x).flatten().max() / 2.0

            Nf = 0
            fail = True
            while fail:
                ctr = np.random.uniform(2.0*maxd + self.edgepad, L - 2.0*maxd - self.edgepad, (3))
                fail = False
                for cid,c in enumerate(self.ctd):
                    if np.linalg.norm(c[0] - ctr) < maxd + c[1] + 8.0:
                        # search for atoms within r angstroms
                        minv = 10000.0
                        for xi in self.X[c[2]:c[2]+self.Na[cid],:]:
                            for xj in x + ctr:
                                dij = np.linalg.norm(xi-xj)
                                if dij < minv:
                                    minv = dij
                        if minv < self.mindist:
                            fail = True
                            Nf += 1

                if not fail:
                    plc += 1
                    #print('Added:',plc)
                    self.ctd.append((ctr, maxd, pos))
                    self.X = np.vstack([self.X, x + ctr])
                    self.Na[idx] = len(s)
                    pos += len(s)
                    #print('Placement Complete...',plc,'-',Nf)
                    self.S.extend(s)
        print('Mols filled:',plc,'/',Nm,'Density:',(plc/np.power(L,3))*29.8896980969,'Temp:',T)

    def init_dynamics(self, Nm, Nembed, V, L, dt, T):
        self.L = L

        # Generate the box of junk
        self.__generategarbagebox__(Nm, Nembed, L, T)

        # Make mol
        self.mol = Atoms(symbols=self.S, positions=self.X)

        # Set box and PBC
        self.mol.set_cell(([[L, 0, 0],
                            [0, L, 0],
                            [0, 0, L]]))

        self.mol.set_pbc((True, True, True))

        # Set ANI calculator
        self.mol.set_calculator(ANIENS(self.aens))

        # Open MD output
        #self.mdcrd = open(xyzfile, 'w')

        # Open MD output
        #self.traj = open(trjfile, 'w')

        # Set the velocities corresponding to a boltzmann dist @ T/5.0
        MaxwellBoltzmannDistribution(self.mol, T/5.0 * units.kB)

        # Declare Dyn
        self.dyn = Langevin(self.mol, dt * units.fs, T * units.kB, 0.01)

        # Define the printer
        #def printenergy(a=self.mol, d=self.dyn, b=self.mdcrd, t=self.traj):  # store a reference to atoms in the definition.
        #    """Function to print the potential, kinetic and total energy."""
        #    epot = a.get_potential_energy() / len(a)
        #    ekin = a.get_kinetic_energy() / len(a)
        #
        #    stddev = hdn.evtokcal * a.calc.stddev
        #
        #    t.write(str(d.get_number_of_steps()) + ' ' + str(ekin / (1.5 * units.kB)) + ' ' + str(epot) + ' ' + str(
        #        ekin) + ' ' + str(epot + ekin) + '\n')
        #    b.write(str(len(a)) + '\n' + str(ekin / (1.5 * units.kB)) + ' Step: ' + str(d.get_number_of_steps()) + '\n')
        #    c = a.get_positions(wrap=True)
        #    for j, i in zip(a, c):
        #        b.write(str(j.symbol) + ' ' + str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + '\n')
        #
        #    print('Step: %d Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
        #          'Etot = %.3feV' ' StdDev = %.3fkcal/mol/sqrt(Na)' % (
        #          d.get_number_of_steps(), epot, ekin, ekin / (1.5 * units.kB), epot + ekin, stddev))

        # Attach the printer
        #self.dyn.attach(printenergy, interval=1)

    def run_dynamics(self, Ni):
        for id, nc in enumerate(self.aens.ncl):
            nc.setMolecule(coords=np.array(self.mol.get_positions(), dtype=np.float32), types=self.mol.get_chemical_symbols())

        self.dyn.run(Ni) # Do Ni steps

    def __fragmentbox__(self, file, sighat):
        self.X = self.mol.get_positions()

        self.frag_list = []

        self.Nd = 0
        self.Nt = 0

        self.maxsig = 0

        CUT = 7.0

        for i in range(len(self.Na)):
            si = self.ctd[i][2]
            di = self.ctd[i][1]
            Nai = self.Na[i]
            Xi = self.X[si:si + Nai, :]
            Xci = np.sum(Xi, axis=0)/ Nai

            if np.all( Xci > CUT+di ) and np.all( Xci <= self.L-(CUT+di) ):

                #if np.all(Xci > di) and np.all(Xci <= self.L-di):
                Xf = Xi
                Sf = self.S[si:si + Nai]

                Nmax = np.random.randint(100, 101)
                Nmol = 0
                ridx = np.arange(len(self.Na))
                np.random.shuffle(ridx)
                for j in ridx:
                    if i != j:
                        sj = self.ctd[j][2]
                        dj = self.ctd[j][1]
                        Naj = self.Na[j]
                        Xcj = np.sum(self.X[sj:sj+Naj,:], axis=0)/ Naj
                        Xj = self.X[sj:sj+Naj,:]

                        if np.all(Xcj > dj) and np.all(Xcj < self.L - dj):
                            dc = np.linalg.norm(Xci - Xcj)
                            if dc < di + dj + CUT:
                                minl = 10.0
                                for ii in range(Nai):
                                    Xiii = Xi[ii]
                                    for jj in range(Naj):
                                        Xjjj = Xj[jj]
                                        v = np.linalg.norm(Xiii-Xjjj)
                                        if v < minl:
                                            minl = v

                                #inc = np.random.uniform(0.0, 4.0)
                                #if minl < 2.0+inc and minl > 0.0+inc:
                                if minl < CUT and minl > 0.5:
                                    Xf = np.vstack([Xf, Xj])
                                    Sf.extend(self.S[sj:sj+Naj])
                                    Nmol += 1
                        if Nmol >= Nmax:
                            break

                Xcf = np.sum(Xf, axis=0) / float(len(Sf))
                Xf = Xf

                E = np.empty(len(self.aens.ncl), dtype=np.float64)
                for idx,nc in enumerate(self.aens.ncl):
                    nc.setMolecule(coords=np.array(Xf,dtype=np.float32), types=Sf)
                    E[idx] = nc.energy()[0]

                sig = np.std(hdn.hatokcal*E)/np.sqrt(Nai+Naj)
                #print('Mol(',i,'): sig=',sig)

                self.Nt += 1
                if sig > sighat:
                    if sig > self.maxsig:
                        self.maxsig = sig
                    self.Nd += 1
                    hdn.writexyzfilewc(file+str(i).zfill(4)+'.xyz', Xf.reshape(1,Xf.shape[0],3), Sf, cmt=['sigma:'+str(sig)])
                    self.frag_list.append(dict({'coords': Xf,'spec': Sf}))
                    #print(dc, Sf, sig)

    def generate_clusters(self, gcmddict, mol_sizes, mol_temps, id):

        self.edgepad = gcmddict['edgepad']
        self.mindist = gcmddict['mindist']

        dstore = gcmddict['dstore']

        difo = open(dstore + 'info_data_mdcluster-' + str(id).zfill(2) + '.nfo', 'w')
        difo.write('Beginning dimer generation...\n')

        Nt = 0
        Nd = 0
        for i in range(gcmddict['Nr']):
            difo.write(str(i)+') Density='+"{:.2f}".format(29.8896980969*(mol_sizes[i]/np.power(30.0,3)))+' Set Temp: ' + "{:.2f}".format(mol_temps[i]) + '\n')
            self.init_dynamics(mol_sizes[i],
                               gcmddict['Nembed'],
                               gcmddict['V'],
                               gcmddict['L'],
                               gcmddict['dt'],
                               mol_temps[i],
                               )

            for j in range(gcmddict['Ns']):
                #if j is 0:
                #    Ni = 0
                #else:
                Ni = gcmddict['Ni']
                self.run_dynamics(Ni)
                self.__fragmentbox__(gcmddict['molfile'] + '-'  + str(id).zfill(2) + '-' +str(i).zfill(2) + '-' + str(j).zfill(4) + '_', gcmddict['sig'])
                #print('Step (',i,',',j,') [', str(self.Nd), '/', str(self.Nt),'] generated ',len(self.frag_list),' maxsig: ', self.maxsig,' dimers...')
                difo.write('Step (' + str(i) + ',' + str(i) + ') [' + str(self.Nd) + '/' + str(self.Nt) + '] generated ' + str(len(self.frag_list)) + ' clusters. ' + ' maxsig: ' + "{:.2f}".format(self.maxsig) + '\n')
                Nt += self.Nt
                Nd += self.Nd
                if self.maxsig > gcmddict['maxsig']:
                    difo.write("Terminated after: " + "{:.2f}".format(Ni*j*gcmddict['dt']) + 'fs for maxsig\n')
                    break
                difo.flush()


            #self.closeoutputs()

        difo.write('Generated '+str(Nd)+' of '+str(Nt)+' tested dimers. Percent: ' + "{:.2f}".format(100.0*Nd/float(Nt)))
        difo.close()
