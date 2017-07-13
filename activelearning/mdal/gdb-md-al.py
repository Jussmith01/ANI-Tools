import pyaniasetools as pya
import gdbsearchtools as gdb

from rdkit import Chem
from rdkit.Chem import AllChem

#--------------Parameters------------------
wkdir = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb01-06_red03-08/cv4/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile = wkdir + 'sae_6-31gd.dat'

At = ['C', 'O', 'N'] # Hydrogens added after check

T = 1000.0
dt = 0.25

smfile = '/home/jujuman/Research/RawGDB11Database/gdb11_size06.smi'

#-------------------------------------------

activ = pya.moldynactivelearning(cnstfile, saefile, wkdir+'train', 5)

gdb.formatsmilesfile(smfile)
molecules = Chem.SmilesMolSupplier(smfile, nameColumn=0)

for n,m in enumerate(molecules):
    if not ('F' in Chem.MolToSmiles(m)):
        print(n,') Working on',Chem.MolToSmiles(m),'...')

        # Add hydrogens
        m = Chem.AddHs(m)

        # generate Nc conformers
        cids = AllChem.EmbedMultipleConfs(m, 100, useRandomCoords=True)

        # Classical Optimization
        #for cid in cids:
        #    _ = AllChem.MMFFOptimizeMolecule(m, confId=cid, maxIters=250)

        # Set mols
        activ.setrdkitmol(m,cids)

        # Generate conformations
        activ.generate_conformations(100, 800.0, 0.25, 400, 10)