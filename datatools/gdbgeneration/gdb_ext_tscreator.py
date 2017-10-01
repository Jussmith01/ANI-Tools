import gdbsearchtools as gdb
import pyaniasetools as aat
import hdnntools as hdn

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

from rdkit.ForceField.rdForceField import *

m = Chem.MolFromSmiles("C=C")


exit (0)
import numpy as np

import os

import linecache
import mmap
import random

def mapcount(filename):
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines

fpf = 'gdb11_s13' #Filename prefix
wdir = '/home/jujuman/Research/extensibility_test_sets/gdb-13/' #working directory
smfile = '/home/jujuman/Research/RawGDB13Database/13.smi' # Smiles file

Nc = 10

LOT='wb97x/6-31g*' # Level of theory
SCF='Tight' #

if not os.path.exists(wdir):
    os.mkdir(wdir)

if not os.path.exists(wdir+'inputs'):
    os.mkdir(wdir+'inputs')

if not os.path.exists(wdir+'images'):
    os.mkdir(wdir+'images')

#print('Formatting...')
#gdb.formatsmilesfile(smfile)

nl = mapcount(smfile)
print('Total smiles:',nl)

#print('Smile supplier...')
#molecules = Chem.SmilesMolSupplier(smfile, nameColumn=0)
#print('Generate list...')
#molecules = [m for m in molecules if not 'F' in Chem.MolToSmiles(m)]
#print('Moles left:',len(molecules))
#ridx = np.arange(0,len(molecules),dtype=np.int32)
#np.random.shuffle(ridx)
#molecules = np.array(molecules)[ridx[0:500]]



Nd = 0
Nt = 0
n = 0

ind = np.full(1000, nl+1, dtype=np.int64)

f = open(smfile,'r+')
mm = mmap.mmap(f.fileno(),0)

molnfo = open(wdir+'molecule_information.nfo','w')
while Nt < 1000:


    ridx = np.random.randint( 0, nl, 1 )[0]
    mm.seek(0)
    print('Finding line...')

    sz = 0
    while True:
        b = mm.read(100000000)
        lines = b.count(str.encode('\n'))
        sz += lines
        if ridx < sz:
            sz = sz - lines
            id = ridx - sz
            #print(id,lines, sz)
            smstr = str(b.split(str.encode("\n"))[id],'utf-8')
            break

    print(smstr)

    #smstr = linecache.getline(smfile, ridx)
    #print(Nt,smstr)

    if 'S' not in smstr and 'Cl' not in smstr and ridx not in ind:
        ind[Nt] = ridx
        m = Chem.MolFromSmiles(smstr)

        print(str(Nt).zfill(4),') Working on',Chem.MolToSmiles(m),'...')
        id = str(Nt)

        # Check ring data
        ri = m.GetRingInfo()
        ring = ''
        for i in range(3,10):
            ring += str(ri.NumAtomRings(i))

        print(ring)
        molnfo.write(str(Nt).zfill(4)+' '+Chem.MolToSmiles(m)+' '+ ring +'\n')

        # Generate images
        AllChem.Compute2DCoords(m)
        #Draw.MolToFile(m, wdir+'images/' + fpf + '-' + id.zfill(4) + '.png')

        # Add hydrogens
        m = Chem.AddHs(m)

        # generate Nc conformers
        cids = AllChem.EmbedMultipleConfs(m, Nc, useRandomCoords=False)

        # Classical Optimization
        for cid in cids:
            _ = AllChem.MMFFOptimizeMolecule(m, confId=cid, maxIters=1000)



        # Align all conformers
        Chem.rdMolAlign.AlignMolConformers(m)

        # get number of confs
        Nu = len(m.GetConformers())

        # Get all conformers
        X = []
        for c in m.GetConformers():
            x =  np.empty((m.GetNumAtoms(),3),dtype=np.float32)
            for i in range(m.GetNumAtoms()):
                r = c.GetAtomPosition(i)
                x[i] = [r.x, r.y, r.z]
            X.append(x)

        #Nd += len(X)
        #Nt += Nu
        Nt += 1
        if len(X) > 0:
            Nd += 1
            X = np.stack(X)
            x = X[0].reshape(1, X.shape[1], 3) # keep only the first guy

        print('    -kept', len(x),'of',Nu)

        if len(X) > 0:
            S = gdb.get_symbols_rdkitmol(m)

        hdn.write_rcdb_input(x[0],S,int(id),wdir,fpf,100,LOT,'500.0',fill=4,comment='smiles: ' + Chem.MolToSmiles(m) + ' GDB-ID: '+str(ridx))
        hdn.writexyzfile(wdir+fpf+'-'+str(id).zfill(4)+'.xyz',x.reshape(1,x.shape[1],x.shape[2]),S)
        #print(str(id).zfill(8))

molnfo.close()
print('Total mols:',Nd,'of',Nt,'percent:',"{:.2f}".format(100.0*Nd/float(Nt)))

