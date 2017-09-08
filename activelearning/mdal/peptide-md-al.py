import pyaniasetools as pya
import gdbsearchtools as gdb
import hdnntools as hdt

from rdkit import Chem
from rdkit.Chem import AllChem

def gendipeptidelist(AAlist):
    fasta = []
    nlist = []

    for i in AAlist:
        for j in AAlist:
            #for k in AAlist:
            fasta.append( ">1AKI:A|PDBID|CHAIN|SEQUENCE\n" + i + j )
            nlist.append("aminoacid_" + i + j )

    for i in AAlist:
        fasta.append( ">1AKI:A|PDBID|CHAIN|SEQUENCE\n" + i )
        nlist.append("aminoacid_" + i )

    return fasta,nlist

#--------------Parameters------------------
wkdir = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb_r06_comb08_3/cv3/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile = wkdir + 'sae_6-31gd.dat'

# Store dir
sdir = '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_3/confs_4/'

At = ['C', 'O', 'N'] # Hydrogens added after check

T = 800.0
dt = 0.5
N = 25

#-------------------------------------------

AAlist = ['A','D','E','F','G','H','I','K','L','N','P','Q','R','S','T','V','W','Y']

fasta,namelist = gendipeptidelist(AAlist)

activ = pya.moldynactivelearning(cnstfile, saefile, wkdir+'train', 5)

print('Nmol:',len(fasta))
difo = open(sdir + 'info_data_mdaa.nfo', 'w')
for n,(a,l) in enumerate(zip(fasta, namelist)):
    m = Chem.MolFromFASTA(a)
    if not ('F' in Chem.MolToSmiles(m)):
        print(n,') Working on',l,Chem.MolToSmiles(m),'...')

        # Add hydrogens
        m = Chem.AddHs(m)

        # generate Nc conformers
        cids = AllChem.EmbedMultipleConfs(m, N, useRandomCoords=True)

        # Classical Optimization
        for cid in cids:
            _ = AllChem.MMFFOptimizeMolecule(m, confId=cid, maxIters=250)

        # Set mols
        activ.setrdkitmol(m,cids)

        # Generate conformations
        X = activ.generate_conformations(N, T, dt, 2000, 10, dS = 0.3)

        nfo = activ._infostr_
        difo.write('  -'+l+': '+nfo+'\n')
        print(nfo)
        difo.flush()

        # Set chemical symbols
        Xi, S = pya.__convert_rdkitmol_to_nparr__(m)

        # Store structures
        hdt.writexyzfile(sdir+l+'-'+str(n).zfill(2)+'.xyz', X, S)
difo.close()
