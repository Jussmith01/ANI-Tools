import gdbsearchtools as gdb
import pyaniasetools as aat
import hdnntools as hdn

from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np

import os

fpf = 'gdbSFCl_chembl' #Filename prefix
wdir = '/home/jujuman/Research/GDB-11-AL-wB97x631gd/elements_SFCl/ANI-AL-SFCl/ANI-AL-0808/ANI-AL-0808.0003/ANI-AL-0808.0003.0401/config_1/' #working directory
#smfile = '/home/jujuman/Research/RawGDB11Database/SFCl/gdb11SFClsize08.smi' # Smiles file
smfile = '/home/jujuman/Research/Drug_moles_raw/chembl_22_clean_1576904_sorted_std_final.smi'
Nc = 2
Pr = 0.5
GPU = 0

LOT='wb97x/6-31g*' # Level of theory
SCF='Tight' #

wkdir = '/home/jujuman/Research/DataReductionMethods/al_working_network/ANI-AL-0808.0303.0400/'
cnstfile = wkdir + 'train0/rHCNOSFCl-4.6A_16-3.1A_a4-8.params'
saefile  = wkdir + 'train0/sae_wb97x-631gd.dat'
nnfdir   = wkdir + 'train0/networks/'

if not os.path.exists(wdir):
    os.mkdir(wdir)

if not os.path.exists(wdir+'inputs'):
    os.mkdir(wdir+'inputs')

ani = aat.anicomputetool(cnstfile, saefile, nnfdir, gpuid=GPU)

wkdircv = '/home/jujuman/Research/DataReductionMethods/al_working_network/ANI-AL-0808.0303.0400/'
cnstfilecv = wkdircv + 'train0/rHCNOSFCl-4.6A_16-3.1A_a4-8.params'
saefilecv  = wkdircv + 'train0/sae_wb97x-631gd.dat'
nnfprefix   = wkdircv + 'train'

anicv = aat.anicrossvalidationconformer(cnstfilecv,saefilecv,nnfprefix,5,GPU,False)

gdb.formatsmilesfile(smfile)
molecules = Chem.SmilesMolSupplier(smfile, nameColumn=0)

Nd = 0
Nt = 0
for n,m in enumerate(molecules):

    s = ''
    try:
        s = Chem.MolToSmiles(m)
    except Exception as ex:
        print('Error:',ex)

    if not any(x in s for x in ['Br', 'I', 'P']) and s != '':
        print(n,') Working on', s,'... Na: ',m.GetNumAtoms())

        # Add hydrogens
        m = Chem.AddHs(m)

        # generate Nc conformers
        cids = AllChem.EmbedMultipleConfs(m, Nc, useRandomCoords=True)

        # Classical Optimization
        for cid in cids:
            _ = AllChem.MMFFOptimizeMolecule(m, confId=cid, maxIters=1000)

        # ANI Optimization
        for cid in cids:
            ani.optimize_rdkit_molecule(m,cid,fmax=0.0001)


        # Detect unique conformers by energy (will fail to select degenerate energy molecules)
        ani.detect_unique_rdkitconfs(m, cids)

        # Align all conformers
        Chem.rdMolAlign.AlignMolConformers(m)

        # get number of confs
        Nu = len(m.GetConformers())

        # Cross test network cross validation on the conformers
        sigma = anicv.compute_stddev_rdkitconfs(m)
        print('    -sigma:',sigma)

        # Get all conformers
        X = []
        for s,c in zip(sigma,m.GetConformers()):
            if s > 0.34:
                x =  np.empty((m.GetNumAtoms(),3),dtype=np.float32)
                for i in range(m.GetNumAtoms()):
                    r = c.GetAtomPosition(i)
                    x[i] = [r.x, r.y, r.z]
                X.append(x)

        #Nd += len(X)
        #Nt += Nu
        Nt += 1
        Ns = len(X)
        if len(X) > 0:
            Nd += 1
            X = np.stack(X)
            #X = X[0].reshape(1, X.shape[1], 3) # keep only the first guy

        print('    -kept', len(X),'of',Nu)

        if len(X) > 0:
            S = gdb.get_symbols_rdkitmol(m)

        P = np.random.binomial(1, Pr, Ns)
        for i,(x,p) in enumerate(zip(X,P)):
            print('       -Keep:', p)
            if p:
                id = int(str(n)+str(i))
                hdn.write_rcdb_input(x,S,id,wdir,fpf,100,LOT,'500.0',fill=8,comment='smiles: '+Chem.MolToSmiles(m))
                hdn.writexyzfile(wdir+fpf+'-'+str(id).zfill(8)+'.xyz',x.reshape(1,x.shape[0],x.shape[1]),S)
                #print(str(id).zfill(8))

print('Total mols:',Nd,'of',Nt,'percent:',"{:.2f}".format(100.0*Nd/float(Nt)))

