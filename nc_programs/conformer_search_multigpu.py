import pyaniasetools as pya
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
import pybel

import hdnntools as hdt
import numpy as np

from multiprocessing import Process
import multiprocessing

import time

#################PARAMETERS#######################
#netdir = '/home/jsmith48/Libraries/ANI-Networks/networks/al_networks/ANI-AL-0808.0303.0400/'
netdir = '/nh/nest/u/jsmith/scratch/Gits/ANI-Networks/networks/al_networks/ANI-AL-0808.0303.0400/'
cns = netdir+'train0/rHCNOSFCl-4.6A_16-3.1A_a4-8.params'
sae = netdir+'train0/sae_wb97x-631gd.dat'
nnf = netdir+'train'
Nn = 5 # Number of models in the ensemble

num_consumers = 1 # This is the number of threads to be spawned
NGPUS = 1 # Number of GPUs on the node (num_consumers/NGPUS jobs will run on each GPU at the same time)
NCONF = 1000 # Number of conformations to embed
Ew = 25.0 # kcal/mol window for optimization selection

## SMILES file (actually each line should be formatted: "[Unique Ident.] [Smiles string]" without brakets)
#smiles = '/home/jsmith48/Chembl_opt/chembl_23_CHNOSFCl_neutral.smi'
smiles = '/scratch/Research/confs_test/Chembl_opt/chembl_23_CHNOSFCl_neutral.smi'

#optd = '/home/jsmith48/Chembl_opt/opt_pdb/' # pdb file output
#datd = '/home/jsmith48/Chembl_opt/opt_dat/' # conformer data output

optd = '/scratch/Research/confs_test/Chembl_opt/opt_pdb/' # pdb file output
datd = '/scratch/Research/confs_test/Chembl_opt/opt_dat/' # conformer data output
#################PARAMETERS#######################

## Roman's code!
## More Roman's code!!

def EmbedConformers(mol, nrot_min=4, nrot_max=10, pool_mult=50):
    nrot = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
    nrot = max(nrot_min, min(nrot_max, nrot))
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, useExpTorsionAnglePrefs=True, useBasicKnowledge=True,
                               numConfs=nrot*pool_mult, pruneRmsThresh=-1., numThreads=-1)
    return mol


def OptimizeConformersFF(mol):
    mol.energies = list()
    AllChem.MMFFSanitizeMolecule(mol)
    AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=-1)
    for conf in mol.GetConformers():
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf.GetId())
        mol.energies.append(ff.CalcEnergy())
    return mol


def CleanConformers(mol, max_keep=1, rmsd_thresh=0.5, max_energy=10.0, max_delta_e_compare=0.5):
    assert isinstance(mol, Chem.Mol)
    N = mol.GetNumConformers()
    if not N:
        # nothing to do
        return
    conformer_ids = [x.GetId() for x in mol.GetConformers()]
    obmols = [rdmol2obmol(mol, x) for x in conformer_ids]
    has_energies = hasattr(mol, 'energies')
    # 1000 is the safe max for conformer energy difference
    energies = mol.energies if has_energies else [i * 1000 for i in range(N)]
    d_energies = np.subtract(energies, min(energies))
    keep = [0] * len(conformer_ids)

    # sorted by energy
    for iconf in np.argsort(energies):
        _id = conformer_ids[iconf]
        # max number of conformers reached -> delete
        if d_energies[iconf] > max_energy:
            mol.RemoveConformer(_id)
            continue
        # make list of conformer ids in keep within max_delta_e_compare window
        targets = [obmols[x] for x in range(len(keep)) if
                   keep[x] and (energies[iconf] - energies[x]) < max_delta_e_compare]
        if sum(keep) != max_keep and is_uniq_conformer_ob(obmols[iconf], targets, rmsd_thresh):
            keep[iconf] = 1
        else:
            mol.RemoveConformer(_id)
    # update energies
    if has_energies:
        mol.energies = [e for e, i in zip(energies, keep) if i]
    return mol


def is_uniq_conformer_ob(reference, targets, rmsd_threshold=0.5):
    """
    Check if reference conformer is close to any of target conformers
    """
    aligner = pybel.ob.OBAlign(False, True)  # includeH, symmetry
    aligner.SetMethod(1)
    aligner.SetRefMol(reference)
    uniq = True
    for t in targets:
        aligner.SetTargetMol(t)
        aligner.Align()
        rmsd = aligner.GetRMSD()
        if rmsd < rmsd_threshold:
            uniq = False
            break
    return uniq


def rdmol2obmol(rdmol, conf_id=-1):
    pmol = pybel.readstring('mol', Chem.MolToMolBlock(rdmol, conf_id))
    return pmol.OBMol


def confsearchsmiles(name, smiles, Ew, NCONF, cmp, eout, optd):
    # Create RDKit MOL
    m = Chem.MolFromSmiles(smiles)
    print('Working on:', name, 'Heavyatoms(', m.GetNumHeavyAtoms(), ')')
    
    # Conf generation routine
    m = EmbedConformers(m)
    m = OptimizeConformersFF(m)
    nrot = Chem.rdMolDescriptors.CalcNumRotatableBonds(m)
    m = CleanConformers(m, max_keep=max(20, nrot*4), max_energy=20.0, rmsd_thresh=0.5)
    del nrot

    # Get new cids
    cids = [x.GetId() for x in m.GetConformers()]

    if len(cids) < 1:
        print('Skipping '+name+': not enough confs to continue.')
        return

    # ANI OPT
    ochk = np.zeros(len(cids), dtype=np.int64)
    for i, cid in enumerate(cids):
        #print('   -Optimizing confid:', cid)
        ochk[i] = cmp.optimize_rdkit_molecule(m, cid=cid, fmax=0.001)

    # Align conformers
    rmslist = []
    AllChem.AlignMolConformers(m, RMSlist=rmslist, maxIters=200)
    #print(rmslist)

    # Get energies and std dev.
    E, V = cmp.energy_rdkit_conformers(m, cids)
    E = hdt.hatokcal * E
    V = hdt.hatokcal * V

    # Sort by energy (low to high)
    idx = np.argsort(E)
    X, S = pya.__convert_rdkitconfs_to_nparr__(m)
    Xs = X[idx]
    Es = E[idx]
    Vs = V[idx]

    for cid, x in zip(cids, Xs):
        natm = m.GetNumAtoms()
        conf = m.GetConformer(cid)
        for i in range(natm):
            conf.SetAtomPosition(i, [float(x[i][0]), float(x[i][1]), float(x[i][2])])

    # Write out conformations
    sdf = AllChem.SDWriter(optd + name + '.sdf')
    for cid in cids:
        sdf.write(m,confId=cid)
    sdf.close()

    # Write out energy and sigma data
    eout.write(name + ' ' + str(Es) + ' ' + str(Vs) + ' ' + str(ochk) + '\n')
    eout.flush()

    return

class multianiconformersearch(multiprocessing.Process):

    def __init__(self, task_queue, Ew, ncon, ngpu, ntd, optd, datd):
        multiprocessing.Process.__init__(self)

        self.task_queue = task_queue # tasks

        self.Ew   = Ew   # Energy cut
        self.ncon = ncon # Number of conformers
        self.ngpu = ngpu # Number of GPUs
        self.optd = optd # Optimization file dir

        proc_name = self.name
        ID = int(proc_name.rsplit("-",1)[1])
        print('ThreadID:',ID)
        self.GPU = (ID-1) % self.ngpu # What GPU are we working on?

        print(proc_name,self.GPU)
        self.ntd = ntd

        self.eout = open(datd + proc_name + '-opnfo.dat', 'w')

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print ('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                self.eout.close()
                break
            gpuid = (int(proc_name[-1])-1) % self.ngpu
            #print (proc_name, next_task, gpuid)
            self.cmp = pya.anienscomputetool(self.ntd['cns'], self.ntd['sae'], self.ntd['nnf'], self.ntd['Nn'], self.GPU)
            confsearchsmiles(next_task['name'], next_task['smiles'], self.Ew, self.ncon, self.cmp, self.eout, self.optd)
            self.task_queue.task_done()
            self.eout.flush()
        return

netdict = {'cns': cns,
           'sae': sae,
           'nnf': nnf,
           'Nn' : Nn,}

tasks = multiprocessing.JoinableQueue()

# Start consumers
print ('Creating %d consumers' % num_consumers)
consumers = [ multianiconformersearch(tasks, Ew, NCONF, NGPUS, netdict, optd, datd) for i in range(num_consumers) ]

for w in consumers:
    w.start()

print('reading...')

data = (open(smiles , 'r').read()).split('\n')
for dat in data[0:3]:
    mol = dat.split(" ")
    if mol[0]:
        print(mol)
        tasks.put({'name'   : mol[0],
                   'smiles' : mol[1]})

# Add a poison pill for each consumer
for i in range(num_consumers):
    tasks.put(None)

# Wait for all of the tasks to finish
tasks.join()

print('COMPLETE!')

