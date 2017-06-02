from ase_interface import ANI
import pyNeuroChem as pync
import hdnntools as hdt
import pymolfrag as pmf
import numpy as np

# ASE Stuff
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.optimize import BFGS, LBFGS
from ase import units

#wkdir = '/home/jujuman/Dropbox/ChemSciencePaper.AER/networks/ANI-c08f-ntwk-cv/'
wkdir = '/home/jujuman/Scratch/Research/WaterData/CV2/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile = wkdir + 'sae_6-31gd.dat'

stdir = '/home/jujuman/Research/WaterData/cv_md_2/'

T = 10.0
dt = 0.25

mol = read('/home/jujuman/Dropbox/ChemSciencePaper.AER/TestCases/water.pdb')
Nnt = 4

# Construct pyNeuroChem classes
print('Constructing CV network list...')
ncl =  [pync.molecule(cnstfile, saefile, wkdir + 'cv_train_' + str(l) + '/networks/', 0, True) for l in range(Nnt)]
print('Complete.')

L1 = 16.761
L2 = 16.592
L3 = 16.566
mol.set_cell(([[L1,0,0],[0,L2,0],[0,0,L3]]))
mol.set_pbc((True, True, True))

# Set the calculator
nc = pync.molecule(cnstfile, saefile, wkdir + 'cv_train_' + str(0) + '/networks/', 0, True)
mol.set_calculator(ANI(False))
mol.calc.setnc(nc)

dyn = LBFGS(mol)
dyn.run(fmax=1.0)

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
    print('Step %i - Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (d.get_number_of_steps(),epot, ekin, ekin / (1.5 * units.kB), epot + ekin))
    t.write(str(d.get_number_of_steps()) + ' ' + str(d.get_time()) + ' ' + str(ekin / (1.5 * units.kB)) + ' ' + str(epot) + ' ' +  str(ekin) + ' ' + str(epot + ekin) + '\n')
    b.write('\n' + str(len(a)) + '\n')
    c=a.get_positions(wrap=True)
    for j,i in zip(a,c):
        b.write(str(j.symbol) + ' ' + str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + '\n')

dyn.attach(printenergy, interval=10)
dyn.set_temperature(T * units.kB)

data = dict()
#dyn.run(4000)  # Do 100 steps of MD
for i in range(100):

    spc = mol.get_chemical_symbols().copy()
    xyz = np.array(mol.get_positions(wrap=False).copy(), dtype=np.float32).reshape(len(spc), 3)

    mf = pmf.molfrag(xyz,spc)

    ofrags1 = mf.get_all_frags('O', 2.7, 3.8)
    ofrags2 = mf.get_all_frags('O', 4.5, 5.6)

    ofrags = ofrags1 + ofrags2
    Nb = 0
    Nf = 0
    Ne = 0
    Ftot = 0.0
    Etot = 0.0
    for i,of in enumerate(ofrags):
        spc_l = of[0]
        xyz_l = of[1]

        energies = np.zeros((Nnt), dtype=np.float64)
        forces = []
        N = 0
        for comp in ncl:
            comp.setMolecule(coords=xyz_l, types=list(spc_l))
            energies[N] = comp.energy()[0]
            forces.append(comp.force().reshape(1, len(list(spc_l)), 3))
            N = N + 1

        Ftot = Ftot + np.mean(np.std(np.concatenate(forces),axis=0))

        energies = hdt.hatokcal * energies

        sigma = np.std(energies) / float(len(spc_l))
        Etot = Etot + sigma

        No = len([s for s in spc_l if s == 'O'])
        Nh = len([s for s in spc_l if s == 'H'])

        Nf = Nf + 1
        if Nh == 2 * No:

            key = str(No).zfill(3) + str(Nh).zfill(3)
            if sigma > 0.06:
                Nb = Nb + 1
                if key not in data.keys():
                    data[key] = (spc_l,[xyz_l])
                else:
                    #print(type())
                    p_xyz = data[key][1]
                    p_xyz.append(xyz_l)
                    data[key] = (spc_l, p_xyz)

                '''
                print('H2O_frag' + str(i).zfill(3) + '.xyz : O=' + str(No).zfill(3) + ' : H=' + str(Nh).zfill(3) + ' : '
                            + "{:13.3f}".format(energies[0]) +
                        ' ' + "{:13.3f}".format(energies[1]) +
                        ' ' + "{:13.3f}".format(energies[2]) +
                        ' ' + "{:13.3f}".format(energies[3]) +
                        ' ' + "{:13.3f}".format(energies[4]) +
                        ' ' + "{:.3f}".format(sigma))
                '''
        else:
            Ne = Ne + 1
            print('Warning: fragmentation failed! No:',No,'Nh:',Nh)
    print('Bad Frags:', Nb, 'of',Nf,'Error:',Ne,'Estd:',Etot/float(Nf), ' Avg. Frc. Std.: ', Ftot/float(Nf))
    dyn.run(40)  # Do 100 steps of MD

for key in data:
    xyz = np.array(data[key][1])
    spc = data[key][0]
    print(key,xyz.shape[0])
    hdt.writexyzfile(stdir + 'frag' + key + '.xyz', xyz, spc)
