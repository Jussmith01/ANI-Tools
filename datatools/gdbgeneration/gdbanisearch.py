import gdbsearchtools as gdb
import pyaniasetools as aat
import hdnntools as hdn

from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np

import os

<<<<<<< HEAD
fpf = 'gdb11_s07' #Filename prefix
wdir = '/home/jujuman/Research/GDB-11-wB97X-6-31gd/dnnts_red/dnntsgdb11_07_red/' #working directory
smfile = '/home/jujuman/Research/RawGDB11Database/gdb11_size07.smi' # Smiles file

Nc = 10

LOT='wb97x/6-31g*' # Level of theory
=======
fpf = 'gdb11_s03' #Filename prefix
wdir = '/home/jujuman/Research/GDB-11-test-LOT/dnntsgdb11_03/' #working directory
smfile = '/home/jujuman/Research/RawGDB11Database/gdb11_size03.smi' # Smiles file

Nc = 10

LOT='MP2/6-311++g**' # Level of theory
>>>>>>> 78d7b18b8841b3d3735cbb5f6f7c1d02acb69808
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

<<<<<<< HEAD
#wkdircv = '/home/jujuman/Research/DataReductionMethods/model6/model0.05me/cv/cv1/'
wkdircv = '/home/jujuman/Research/DataReductionMethods/model6r/model3-5/cv5/'
cnstfilecv = wkdircv + 'rHCNO-4.6A_16-3.1A_a4-8.params'
#wkdircv = '/home/jujuman/Research/DataReductionMethods/model6r/model-6-ext/model-6e1/'
#cnstfilecv = wkdircv + 'rHCNO-3.9A_16-3.0A_a4-8.params'
saefilecv  = wkdircv + 'sae_6-31gd.dat'
nnfprefix   = wkdircv + 'train'

anicv = aat.anicrossvalidationconformer(cnstfilecv,saefilecv,nnfprefix,5,0,False)

gdb.formatsmilesfile(smfile)
molecules = Chem.SmilesMolSupplier(smfile, nameColumn=0)

Nd = 0
Nt = 0
for n,m in enumerate(molecules):
    if not ('F' in Chem.MolToSmiles(m)):
        print(n,') Working on',Chem.MolToSmiles(m),'...')

=======
gdb.formatsmilesfile(smfile)
molecules = Chem.SmilesMolSupplier(smfile, nameColumn=0)

for n,m in enumerate(molecules):
    if not ('F' in Chem.MolToSmiles(m)):
>>>>>>> 78d7b18b8841b3d3735cbb5f6f7c1d02acb69808
        # Add hydrogens
        m = Chem.AddHs(m)

        # generate Nc conformers
        cids = AllChem.EmbedMultipleConfs(m, Nc, useRandomCoords=False)

        # Classical Optimization
        for cid in cids:
            _ = AllChem.MMFFOptimizeMolecule(m, confId=cid, maxIters=1000)

        # ANI Optimization
        for cid in cids:
<<<<<<< HEAD
            ani.optimize_rdkit_molecule(m,cid,fmax=0.001)
=======
            ani.optimize_rdkit_molecule(m,cid)
>>>>>>> 78d7b18b8841b3d3735cbb5f6f7c1d02acb69808

        # Detect unique conformers by energy (will fail to select degenerate energy molecules)
        ani.detect_unique_rdkitconfs(m, cids)

        # Align all conformers
        Chem.rdMolAlign.AlignMolConformers(m)

<<<<<<< HEAD
        # get number of confs
        Nu = len(m.GetConformers())

        # Cross test network cross validation on the conformers
        sigma = anicv.compute_stddev_rdkitconfs(m)
        print('    -sigma:',sigma)

        # Get all conformers
        X = []
        for s,c in zip(sigma,m.GetConformers()):
            if s > 0.1:
                x =  np.empty((m.GetNumAtoms(),3),dtype=np.float32)
                for i in range(m.GetNumAtoms()):
                    r = c.GetAtomPosition(i)
                    x[i] = [r.x, r.y, r.z]
                X.append(x)

        Nd += len(X)
        Nt += Nu
        #if len(X) > 0:
        #    Nd += 1
        print('    -kept', len(X),'of',Nu)

        if len(X) > 0:
            X = np.stack(X)
=======
        # Get all conformers
        X = []
        for c in m.GetConformers():
            x =  np.empty((m.GetNumAtoms(),3),dtype=np.float32)
            for i in range(m.GetNumAtoms()):
                r = c.GetAtomPosition(i)
                x[i] = [r.x, r.y, r.z]
            X.append(x)
        X = np.stack(X)
>>>>>>> 78d7b18b8841b3d3735cbb5f6f7c1d02acb69808
        S = gdb.get_symbols_rdkitmol(m)

        for i,x in enumerate(X):
            id = int(str(n)+str(i))
            hdn.write_rcdb_input(x,S,id,wdir,fpf,100,LOT,'500.0',fill=8,comment='smiles: '+Chem.MolToSmiles(m))
<<<<<<< HEAD
            hdn.writexyzfile(wdir+fpf+'-'+str(id).zfill(8)+'.xyz',x.reshape(1,x.shape[0],x.shape[1]),S)
            #print(str(id).zfill(8))

print('Total mols:',Nd,'of',Nt,'percent:',"{:.2f}".format(100.0*Nd/float(Nt)))
=======
            print(str(id).zfill(8))


>>>>>>> 78d7b18b8841b3d3735cbb5f6f7c1d02acb69808
