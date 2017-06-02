# Import pyNeuroChem
from __future__ import print_function

# Neuro Chem
from ase_interface import ANI
import pyNeuroChem as pync

import hdnntools as gt
import numpy as np
import matplotlib.pyplot as plt
import time as tm
from scipy import stats as st
import time

import hdnntools as hdt

from rdkit import Chem
from rdkit.Chem import AllChem

# ASE
import  ase
from ase.io import read, write
from ase.optimize import BFGS, LBFGS
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
from ase.units import Bohr
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms

def formatsmilesfile(file):
    ifile = open(file, 'r')
    contents = ifile.read()
    ifile.close()

    p = re.compile('([^\s]*).*\n')
    smiles = p.findall(contents)

    ofile = open(file, 'w')
    for mol in smiles:
        ofile.write(mol + '\n')
    ofile.close()

#def make_atoms

#--------------Parameters------------------
smfile = '/home/jujuman/Research/RawGDB11Database/gdb11_size06.smi' # Smiles file

wkdir1 = '/home/jujuman/Dropbox/ChemSciencePaper.AER/networks/ANI-c08f-ntwk-cv/'
wkdir2 = '/home/jujuman/Dropbox/ChemSciencePaper.AER/networks/ANI-c08f09bad-ntwk-cv/'
wkdir3 = '/home/jujuman/Dropbox/ChemSciencePaper.AER/networks/ANI-c08f09dd-ntwk-cv/'
wkdir4 = '/home/jujuman/Dropbox/ChemSciencePaper.AER/networks/ANI-c08f09div-ntwk-cv/'

cnstfile = 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile = 'sae_6-31gd.dat'

At = ['C', 'O', 'N'] # Hydrogens added after check

Nnc = 5

#-------------------------------------------
#nnfdir   = wkdir + 'cv_c08e_ntw_' + str(0) + '/networks/'

# Construct pyNeuroChem classes
nc1 =  [pync.molecule(wkdir1 + cnstfile, wkdir1 + saefile, wkdir1 + 'cv_c08e_ntw_' + str(l) + '/networks/', 0) for l in range(Nnc)]
nc2 =  [pync.molecule(wkdir2 + cnstfile, wkdir2 + saefile, wkdir2 + 'cv_c08e_ntw_' + str(l) + '/networks/', 0) for l in range(Nnc)]
nc3 =  [pync.molecule(wkdir3 + cnstfile, wkdir3 + saefile, wkdir3 + 'cv_c08e_ntw_' + str(l) + '/networks/', 0) for l in range(Nnc)]
nc4 =  [pync.molecule(wkdir4 + cnstfile, wkdir4 + saefile, wkdir4 + 'cv_c08e_ntw_' + str(l) + '/networks/', 0) for l in range(Nnc)]

molecules = Chem.SmilesMolSupplier(smfile, nameColumn=0)

total_mol = 0
total_bad = 0

#mols = [molecules[i] for i in range(217855,217865)]
f1 = open('/home/jujuman/Research/CrossValidation/GDB-06-High-sdev/gdb-06-cvsdev_c08f.dat','w')
f2 = open('/home/jujuman/Research/CrossValidation/GDB-06-High-sdev/gdb-06-cvsdev_c08f09bad.dat','w')
f3 = open('/home/jujuman/Research/CrossValidation/GDB-06-High-sdev/gdb-06-cvsdev_c08f09dd.dat','w')
f4 = open('/home/jujuman/Research/CrossValidation/GDB-06-High-sdev/gdb-06-cvsdev_c08f09div.dat','w')
for k,m in enumerate(molecules):
    if m is None: continue

    typecount = 0

    #print (Chem.MolToSmiles(m))

    typecheck = False
    for a in m.GetAtoms():
        sym = str(a.GetSymbol())
        count = 0

        for i in At:
            if i is sym:
                count = 1

        if count is 0:
            typecheck = True

    if typecheck is False:
        total_mol = total_mol + 1
        #print('total_mol: ',total_mol)
        m = Chem.AddHs(m) # Add Hydrogens
        embed = AllChem.EmbedMolecule(m, useRandomCoords=True)
        if embed is 0: # Embed in 3D Space was successful
            check = AllChem.MMFFOptimizeMolecule(m, maxIters=1000)  # Classical Optimization

            xyz = np.zeros((m.GetNumAtoms(),3),dtype=np.float32)
            spc = []

            Na = m.GetNumAtoms()
            for i in range (0,Na):
                pos = m.GetConformer().GetAtomPosition(i)
                sym = m.GetAtomWithIdx(i).GetSymbol()

                spc.append(sym)
                xyz[i, 0] = pos.x
                xyz[i, 1] = pos.y
                xyz[i, 2] = pos.z

            mol = Atoms(symbols=spc, positions=xyz)
            #mol.set_calculator(ANI(False))
            #mol.calc.setnc(nc1[0])

            xyzi = np.array(mol.get_positions(),dtype=np.float32).reshape(xyz.shape[0],3)

            #dyn = LBFGS(mol,logfile='logfile.txt')
            #dyn.run(fmax=0.001,steps=10000)
            # = True if dyn.get_number_of_steps() == 10000 else False
            #stps = dyn.get_number_of_steps()
            stps = 0

            xyz = np.array(mol.get_positions(),dtype=np.float32).reshape(xyz.shape[0],3)

            #if conv:
            #    print('Failed to converge!!!')
            energies = np.zeros((Nnc),dtype=np.float64)

            #------------ CV NETWORKS 1 -----------
            N = 0
            for comp in nc1:
                comp.setMolecule(coords=xyz, types=list(spc))
                energies[N] = comp.energy()[0]
                N = N + 1

            if np.std(hdt.hatokcal * energies) > 5.0:
                hdt.writexyzfile('/home/jujuman/Research/CrossValidation/GDB-06-High-sdev/CV1bmol-'+str(total_mol)+'.xyz',xyz.reshape(1,xyz.shape[0],xyz.shape[1]),spc)
                total_bad = total_bad + 1

            perc = int(100.0 * total_bad / float(total_mol))
            output = '  ' + str(k) + ' ' + str(total_bad) + '/' + str(total_mol) + ' ' + str(perc) + '% (' + str(
                Na) + ') : stps=' + str(stps) + ' : ' + str(energies) + ' : std(kcal/mol)=' + str(
                np.std(hdt.hatokcal * energies)) + ' : ' + Chem.MolToSmiles(m)

            if np.std(hdt.hatokcal*energies) > 5.0:
                print("CV1:", output)

            f1.write(output + '\n')

            #------------ CV NETWORKS 2 -----------
            N = 0
            for comp in nc2:
                comp.setMolecule(coords=xyz, types=list(spc))
                energies[N] = comp.energy()[0]
                N = N + 1

            if np.std(hdt.hatokcal * energies) > 5.0:
                hdt.writexyzfile('/home/jujuman/Research/CrossValidation/GDB-06-High-sdev/CV2bmol-'+str(total_mol)+'.xyz',xyz.reshape(1,xyz.shape[0],xyz.shape[1]),spc)
                total_bad = total_bad + 1

            perc = int(100.0 * total_bad / float(total_mol))
            output = '  ' + str(k) + ' ' + str(total_bad) + '/' + str(total_mol) + ' ' + str(perc) + '% (' + str(
                Na) + ') : stps=' + str(stps) + ' : ' + str(energies) + ' : std(kcal/mol)=' + str(
                np.std(hdt.hatokcal * energies)) + ' : ' + Chem.MolToSmiles(m)

            if np.std(hdt.hatokcal*energies) > 5.0:
                print("CV2:", output)

            f2.write(output + '\n')

            #------------ CV NETWORKS 3 -----------
            N = 0
            for comp in nc3:
                comp.setMolecule(coords=xyz, types=list(spc))
                energies[N] = comp.energy()[0]
                N = N + 1

            if np.std(hdt.hatokcal * energies) > 5.0:
                hdt.writexyzfile('/home/jujuman/Research/CrossValidation/GDB-06-High-sdev/CV3bmol-'+str(total_mol)+'.xyz',xyz.reshape(1,xyz.shape[0],xyz.shape[1]),spc)
                total_bad = total_bad + 1

            perc = int(100.0 * total_bad / float(total_mol))
            output = '  ' + str(k) + ' ' + str(total_bad) + '/' + str(total_mol) + ' ' + str(perc) + '% (' + str(
                Na) + ') : stps=' + str(stps) + ' : ' + str(energies) + ' : std(kcal/mol)=' + str(
                np.std(hdt.hatokcal * energies)) + ' : ' + Chem.MolToSmiles(m)

            if np.std(hdt.hatokcal*energies) > 5.0:
                print("CV3:", output)

            f3.write(output + '\n')

            #------------ CV NETWORKS 4 -----------
            N = 0
            for comp in nc4:
                comp.setMolecule(coords=xyz, types=list(spc))
                energies[N] = comp.energy()[0]
                N = N + 1

            if np.std(hdt.hatokcal * energies) > 5.0:
                hdt.writexyzfile('/home/jujuman/Research/CrossValidation/GDB-06-High-sdev/CV4bmol-'+str(total_mol)+'.xyz',xyz.reshape(1,xyz.shape[0],xyz.shape[1]),spc)
                total_bad = total_bad + 1

            perc = int(100.0 * total_bad / float(total_mol))
            output = '  ' + str(k) + ' ' + str(total_bad) + '/' + str(total_mol) + ' ' + str(perc) + '% (' + str(
                Na) + ') : stps=' + str(stps) + ' : ' + str(energies) + ' : std(kcal/mol)=' + str(
                np.std(hdt.hatokcal * energies)) + ' : ' + Chem.MolToSmiles(m)

            if np.std(hdt.hatokcal*energies) > 5.0:
                print("CV4:", output)

            f4.write(output + '\n')

print('Total Molecs: ', total_mol)
print('Total Bad 1.0:    ', total_bad)
print('Percent Bad:  ', int(100.0 * total_bad/float(total_mol)), '%')
f1.close()
f2.close()
f3.close()
f4.close()
#print('End...')

