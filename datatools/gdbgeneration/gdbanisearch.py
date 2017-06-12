import gdbsearchtools as gdb
import pyaniasetools as aat
import hdnntools as hdn

from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np

import os

fpf = 'gdb11_s03' #Filename prefix
wdir = '/home/jujuman/Research/GDB-11-test-LOT/dnntsgdb11_03/' #working directory
smfile = '/home/jujuman/Research/RawGDB11Database/gdb11_size03.smi' # Smiles file

Nc = 10

LOT='MP2/6-311++g**' # Level of theory
SCF='Tight' #

wkdir = '/home/jujuman/Gits/ANI-Networks/networks/ANI-c08f-ntwk/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile  = wkdir + 'sae_6-31gd.dat'
nnfdir   = wkdir + 'networks/'

if not os.path.exists(wdir):
    os.mkdir(wdir)

if not os.path.exists(wdir+'inputs'):
    os.mkdir(wdir+'inputs')

ani = aat.anicomputetool(cnstfile, saefile, nnfdir)

gdb.formatsmilesfile(smfile)
molecules = Chem.SmilesMolSupplier(smfile, nameColumn=0)

for n,m in enumerate(molecules):
    if not ('F' in Chem.MolToSmiles(m)):
        # Add hydrogens
        m = Chem.AddHs(m)

        # generate Nc conformers
        cids = AllChem.EmbedMultipleConfs(m, Nc, useRandomCoords=False)

        # Classical Optimization
        for cid in cids:
            _ = AllChem.MMFFOptimizeMolecule(m, confId=cid, maxIters=1000)

        # ANI Optimization
        for cid in cids:
            ani.optimize_rdkit_molecule(m,cid)

        # Detect unique conformers by energy (will fail to select degenerate energy molecules)
        ani.detect_unique_rdkitconfs(m, cids)

        # Align all conformers
        Chem.rdMolAlign.AlignMolConformers(m)

        # Get all conformers
        X = []
        for c in m.GetConformers():
            x =  np.empty((m.GetNumAtoms(),3),dtype=np.float32)
            for i in range(m.GetNumAtoms()):
                r = c.GetAtomPosition(i)
                x[i] = [r.x, r.y, r.z]
            X.append(x)
        X = np.stack(X)
        S = gdb.get_symbols_rdkitmol(m)

        for i,x in enumerate(X):
            id = int(str(n)+str(i))
            hdn.write_rcdb_input(x,S,id,wdir,fpf,100,LOT,'500.0',fill=8,comment='smiles: '+Chem.MolToSmiles(m))
            print(str(id).zfill(8))


