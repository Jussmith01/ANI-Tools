from ase_interface import ANIENS
from ase_interface import ensemblemolecule

import pyNeuroChem as pync
import pyaniasetools as pya
import hdnntools as hdt

import numpy as np
import  ase
import time
#from ase.build import molecule
#from ase.neb import NEB
#from ase.calculators.mopac import MOPAC
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase import units

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ase.optimize.fire import FIRE as QuasiNewton

from ase.md.nvtberendsen import NVTBerendsen
from ase.md import MDLogger

#from ase.neb import NEBtools
from ase.io import read, write
from ase.optimize import BFGS, LBFGS

import matplotlib
import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

#----------------Parameters--------------------

dir = '/home/jujuman/Research/hard_const_test/'

# Molecule file
molfile = dir + 'AA4cap.xyz'

# Dynamics file
xyzfile = dir + 'mdcrd.xyz'

# Trajectory file
trajfile = dir + 'traj.dat'

# Optimized structure out:
optfile = dir + 'optmol.xyz'

T = 600.0 # Temperature
dt = 0.2
C = 0.0001 # Optimization convergence
steps = 40000

wkdir = '/home/jujuman/Research/DataReductionMethods/train_test/ANI-9.0.4_netarch8/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile  = wkdir + 'sae_6-31gd.dat'
nnfdir   = wkdir + '/train'
Nn = 5
#nnfdir   = wkdir + 'networks/'

#----------------------------------------------

# Load molecule
mol = read(molfile)
#print('test')
#L = 30.0
#mol.set_cell(([[L, 0, 0],
#               [0, L, 0],
#               [0, 0, L]]))

#mol.set_pbc((True, True, True))

#print(mol.get_chemical_symbols())

# Set NC
aens = ensemblemolecule(cnstfile, saefile, nnfdir, Nn, 0)

# Set ANI calculator
mol.set_calculator(ANIENS(aens,sdmx=20000000.0))

# Optimize molecule
start_time = time.time()
dyn = LBFGS(mol)
dyn.run(fmax=C)
print('[ANI Total time:', time.time() - start_time, 'seconds]')

print(hdt.evtokcal*mol.get_potential_energy())
print(hdt.evtokcal*mol.get_forces())

# Save optimized mol
spc = mol.get_chemical_symbols()
pos = mol.get_positions(wrap=False).reshape(1,len(spc),3)

hdt.writexyzfile(optfile, pos, spc)

# Open MD output
mdcrd = open(xyzfile,'w')

# Open MD output
traj = open(trajfile,'w')

# We want to run MD with constant energy using the Langevin algorithm
# with a time step of 0.5 fs, the temperature T and the friction
# coefficient to 0.02 atomic units.
dyn = Langevin(mol, dt * units.fs, T * units.kB, 0.02)

# Run equilibration
#print('Running equilibration...')
#start_time = time.time()
#dyn.run(10000) # Run 100ps equilibration dynamics
#print('[ANI Total time:', time.time() - start_time, 'seconds]')

# Set the momenta corresponding to T=300K
#MaxwellBoltzmannDistribution(mol, T * units.kB)
# Print temp
ekin = mol.get_kinetic_energy() / len(mol)
print('Temp: ', ekin / (1.5 * units.kB))

# Define the printer
def storeenergy(a=mol, d=dyn, b=mdcrd, t=traj):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)

    stddev =  hdt.evtokcal*a.calc.stddev

    t.write(str(d.get_number_of_steps()) + ' ' + str(ekin / (1.5 * units.kB)) + ' ' + str(epot) + ' ' + str(ekin) + ' ' + str(epot+ekin) + '\n')
    b.write(str(len(a)) + '\n' + str(ekin / (1.5 * units.kB)) + ' Step: ' + str(d.get_number_of_steps()) + '\n')
    c = a.get_positions(wrap=True)
    for j, i in zip(a, c):
        b.write(str(j.symbol) + ' ' + str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + '\n')

    print('Step: %d Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' ' StdDev = %.3fKcal/mol/atom' % (d.get_number_of_steps(), epot, ekin, ekin / (1.5 * units.kB), epot + ekin, stddev))

# Define the printer
def printenergy(a=mol, d=dyn, b=mdcrd, t=traj):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)

    stddev =  hdt.evtokcal*a.calc.stddev

    print('Step: %d Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' ' StdDev = %.3fKcal/mol/atom' % (d.get_number_of_steps(), epot, ekin, ekin / (1.5 * units.kB), epot + ekin, stddev))


# Attach the printer
dyn.attach(storeenergy, interval=1)
dyn.attach(printenergy, interval=1)

# Run production
print('Running production...')
#start_time = time.time()
#for i in range(int(T)):
#    print('Set temp:',i,'K')
#    dyn.set_temperature(float(i) * units.kB)
#    dyn.run(50)

dyn.set_temperature(T * units.kB)
dyn.run(steps)
print('[ANI Total time:', time.time() - start_time, 'seconds]')
mdcrd.close()
traj.close()
print('Finished.')
