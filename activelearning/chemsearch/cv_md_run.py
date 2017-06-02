import sys
import time

# Numpy
import numpy as np

# Neuro Chem
from ase_interface import ANI
import pyNeuroChem as pync
import hdnntools as hdt
import nmstools as nm

import  ase
#from ase.build import molecule
#from ase.neb import NEB
#from ase.calculators.mopac import MOPAC
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase.io.trajectory import Trajectory
from ase import units

from ase.vibrations import Vibrations

from ase.optimize.fire import FIRE as QuasiNewton

from ase.md.nvtberendsen import NVTBerendsen
from ase.md import MDLogger

#from ase.neb import NEBtools
from ase.io import read, write
from ase.optimize import BFGS, LBFGS

#import matplotlib
#import matplotlib as mpl

#import matplotlib.pyplot as plt

#import seaborn as sns
#%matplotlib inline
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

#--------------Parameters------------------
wkdir = '/home/jujuman/Dropbox/ChemSciencePaper.AER/networks/ANI-c08f09dd-ntwk-cv/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile = wkdir + 'sae_6-31gd.dat'

At = ['C', 'O', 'N'] # Hydrogens added after check

T = 300.0
dt = 0.25

stdir = '/home/jujuman/Research/CrossValidation/MD_CV/'

#-------------------------------------------

# Construct pyNeuroChem classes
print('Constructing CV network list...')
ncl =  [pync.molecule(cnstfile, saefile, wkdir + 'cv_c08e_ntw_' + str(l) + '/networks/', 0) for l in range(5)]
print('Complete.')

# Set required files for pyNeuroChem
#anipath  = '/home/jujuman/Dropbox/ChemSciencePaper.AER/ANI-c08e-ccdissotest1-ntwk'
#cnstfile = anipath + '/rHCNO-4.6A_16-3.1A_a4-8.params'
#saefile  = anipath + '/sae_6-31gd.dat'
#nnfdir   = anipath + '/networks/'

# Construct pyNeuroChem class
print('Constructing MD network...')
nc = ncl[1]
#nc = pync.molecule(cnstfile, saefile, nnfdir, 0)
print('FINISHED')

#mol = read('/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_testdata/specialtest/test.xyz')
#mol = read('/home/jujuman/Dropbox/ChemSciencePaper.AER/TestCases/water.pdb')
mol = read('/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_begdb/begdb-h2oclusters/xyz/4179_water2Cs.xyz')
#mol = read('/home/jujuman/Research/CrossValidation/MD_CV/benzene.xyz')
#mol = read('/home/jujuman/Dropbox/ChemSciencePaper.AER/TestCases/Retinol/opt_test_NO.xyz')

print(mol)
#L = 16.0
#bz.set_cell(([[L,0,0],[0,L,0],[0,0,L]]))
#bz.set_pbc((True, True, True))

mol.set_calculator(ANI(False))
mol.calc.setnc(nc)

start_time = time.time()
dyn = LBFGS(mol)
dyn.run(fmax=1.0)
print('[ANI Total time:', time.time() - start_time, 'seconds]')

# We want to run MD with constant energy using the Langevin algorithm
# with a time step of 5 fs, the temperature T and the friction
# coefficient to 0.02 atomic units.
dyn = Langevin(mol, dt * units.fs, T * units.kB, 0.01)

mdcrd = open(stdir + "mdcrd.xyz",'w')
temp = open(stdir + "temp.dat",'w')

dyn.get_time()
def printenergy(a=mol,b=mdcrd,d=dyn,t=temp):  # store a reference to atoms in the
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    #print('Step %i - Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
    #      'Etot = %.3feV' % (d.get_number_of_steps(),epot, ekin, ekin / (1.5 * units.kB), epot + ekin))
    t.write(str(d.get_number_of_steps()) + ' ' + str(d.get_time()) + ' ' + str(ekin / (1.5 * units.kB)) + ' ' + str(epot) + ' ' +  str(ekin) + ' ' + str(epot + ekin) + '\n')
    b.write('\n' + str(len(a)) + '\n')
    c=a.get_positions(wrap=True)
    for j,i in zip(a,c):
        b.write(str(j.symbol) + ' ' + str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + '\n')

dyn.attach(printenergy, interval=50)
dyn.set_temperature(600.0 * units.kB)
start_time2 = time.time()

# get the chemical symbols
spc = mol.get_chemical_symbols()

#xo = open(stdir + 'data/md-peptide-cvnms.xyz', 'w')
f = open(stdir + 'md-peptide-cv.dat','w')
l_sigma = []

for i in range(10000):
    dyn.run(1)  # Do 100 steps of MD

    xyz = np.array(mol.get_positions(), dtype=np.float32).reshape(len(spc), 3)
    energies = np.zeros((5), dtype=np.float64)
    forces = []
    N = 0
    for comp in ncl:
        comp.setMolecule(coords=xyz, types=list(spc))
        energies[N] = comp.energy()[0]
        forces.append(comp.force().reshape(1, len(list(spc)), 3))
        N = N + 1

    #print(np.vstack(forces))

    csl = []
    for j in range(0, len(list(spc))):
        print(np.vstack(forces)[:,j,:])
        print(np.sum(np.power(np.vstack(forces)[:,j,:],2.0),axis=0))
        csm = cosine_similarity(np.vstack(forces)[:,j,:])
        cse = np.mean(np.asarray(csm[np.triu_indices_from(csm,1)]))
        csl.append((spc[j],cse))

    energies = hdt.hatokcal * energies

    sigma = np.std(energies) / float(len(spc))

    f.write("{:.7f}".format((dyn.get_number_of_steps() * dt)/1000.0) +
            ' ' + "{:.7f}".format(energies[0]) +
            ' ' + "{:.7f}".format(energies[1]) +
            ' ' + "{:.7f}".format(energies[2]) +
            ' ' + "{:.7f}".format(energies[3]) +
            ' ' + "{:.7f}".format(energies[4]) +
            ' ' + "{:.7f}".format(sigma) + '\n')

    ekin = mol.get_kinetic_energy() / len(mol)

    output = '  ' + str(i) + ' (' + str(len(spc)) + ',', "{:.4f}".format(ekin / (1.5 * units.kB)),'K) : stps=' + str(dyn.get_number_of_steps()) + ' : std(kcal/mol)=' + "{:.4f}".format(sigma)
    print(output,csl)

    '''
    if sigma > 0.5:
        vib = Vibrations(mol)
        vib.run()
        vib.summary()

        nmo = vib.modes[6:].reshape(3*len(spc)-6, len(spc), 3)
        fcc = np.ones((3*len(spc)-6),dtype=np.float32)

        gen = nm.nmsgenerator(xyz, nmo, fcc, spc, 800.0)

        N = 4
        gen_crd = np.zeros((N, len(spc), 3), dtype=np.float32)
        for j in range(N):
            gen_crd[j] = gen.get_random_structure()

        np.vstack([xyz.reshape(1, xyz.shape[0], xyz.shape[1]), gen_crd])
        #hdt.writexyzfile(stdir + 'data/md-peptide-cv-' + str(i) + '.xyz', gen_crd, spc)

        Na = len(spc)
        for j,m in enumerate(gen_crd):
            xo.write(str(Na) + '\n')
            xo.write('      stddev: ' + str(sigma) + ' conf: ' + str(j) + '\n')
            for k,at in enumerate(m):
                    x = at[0]
                    y = at[1]
                    z = at[2]
                    xo.write(spc[k] + ' ' + "{:.7f}".format(x) + ' ' + "{:.7f}".format(y) + ' ' + "{:.7f}".format(z) + '\n')

    elif sigma > 0.1:
        xo.write(str(len(spc)) + '\n')
        xo.write('      stddev: ' + str(sigma) + ' step: ' + str(i) + '\n')
        for k, at in enumerate(xyz):
            x = at[0]
            y = at[1]
            z = at[2]
            xo.write(spc[k] + ' ' + "{:.7f}".format(x) + ' ' + "{:.7f}".format(y) + ' ' + "{:.7f}".format(
                z) + '\n')

            #xo.write('\n')
    '''

#xo.close()
f.close()
end_time2 = time.time()
print('CV MD Total Time:', end_time2 - start_time2)
mdcrd.close()
temp.close()
