import hdnntools as hdt
import pyaniasetools as pya

from rdkit import Chem
from rdkit.Chem import AllChem

storedir = '/home/jujuman/Research/extensibility_test_sets/drugbank/'

suppl = Chem.SDMolSupplier('/home/jujuman/Dropbox/ChemSciencePaper.AER/Benchmark_Datasets/drugbank/drugbank_3d_1564.sdf',removeHs=False)
for id,m in enumerate(suppl):
    if m is None: continue

    name = m.GetProp('_Name')
    xyz, spc = pya.__convert_rdkitconfs_to_nparr__(m)

    print(xyz.shape)

    print(name, id, spc)

    hdt.writexyzfile(storedir+'xyz/drugbank_'+str(id).zfill(4)+'.xyz', xyz, spc)
    hdt.write_rcdb_input(xyz[0], spc, id, storedir, 'drugbank', 10, 'wb97x/6-31g*','300.0',fill=4,comment='Name: ' + name)

    # Print size
    #print(m.GetNumAtoms(), m.GetNumHeavyAtoms())

