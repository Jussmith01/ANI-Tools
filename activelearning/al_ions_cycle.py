import anialservertools as ast
import anitraintools as att
import anialtools as alt
from time import sleep
import os
import sys

#passfile = "/home/jsmith48/tkey.dat"

hostname = "comet.sdsc.xsede.org"
#hostname = "bridges.psc.edu"
#hostname = "moria.chem.ufl.edu"
username = "jsmith48"
#password = open(passfile,'r').read().strip()

root_dir = '/home/jsmith48/scratch/auto_ion_al/'

swkdir = '/home/jsmith48/scratch/auto_al_cycles/'# server working directory
datdir = 'ANI-AL-smallions-'

h5stor = root_dir + 'h5files/'# h5store location

optlfile = root_dir + 'optimized_input_files.dat'

#Comet
mae = 'module load gnu/4.9.2\n' +\
      'module load gaussian\n' +\
      'export PATH="/home/$USER/Gits/RCDBuilder/build/bin:$PATH"\n' +\
      'export LD_LIBRARY_PATH="/home/$USER/Gits/RCDBuilder/build/lib:$LD_LIBRARY_PATH"\n'

# Bridges
#mae = 'module load gcc/5.3.0\n' +\
#      'module load gaussian/09.D.01\n' +\
#      'export PATH="/home/$USER/Gits/RCDBuilder/build/bin:$PATH"\n' +\
#      'export LD_LIBRARY_PATH="/home/$USER/Gits/RCDBuilder/build/lib:$LD_LIBRARY_PATH"\n' +\
#      "export GAUSS_EXEDIR='/opt/packages/gaussian-RevD.01/g09/' \n"

fpatoms = ['C', 'N', 'O', 'S', 'F', 'Cl']
#fpatoms = ['C', 'N', 'O']

jtime = "0-12:00"

#---- Training Parameters ----
GPU = [2,3,4,5] # GPU IDs

M   = 0.3 # Max error per atom in kcal/mol
Nnets = 8 # networks in ensemble
Nblock = 16 # Number of blocks in split
Nbvald = 2 # number of valid blocks
Nbtest = 2 # number of test blocks
aevsize = 1008

#wkdir = '/home/jsmith48/scratch/auto_rxn_al/modelrxn/ANI-1x-RXN-0000/'
#iptfile = '/home/jsmith48/scratch/auto_rxn_al/modelrxn/inputtrain.ipt'
#saefile = '/home/jsmith48/scratch/auto_rxn_al/modelrxn/sae_linfit.dat'
#cstfile = '/home/jsmith48/scratch/auto_rxn_al/modelrxn/rHCNO-4.6R_16-3.1A_a4-8.params'

root = '/home/jsmith48/scratch/auto_ion_al'

wkdir   = root+'/modelions/ANI-AL-smallions/'
iptfile = root+'/modelions/inputtrain.ipt'
saefile = root+'/modelions/sae_linfit.dat'
cstfile = root+'/modelions/rHCNOSFCl-5.2R_16-3.8A_a4-8.params'

#-----------0---------

# Training varibles

#### Sampling parameters ####
nmsparams = {'T': 600.0, # Temperature
             'Ngen': 1500, # Confs to generate
             'Nkep': 1000, # Diverse confs to keep
             'maxd': 0.5, # Diverse confs to keep
             'sig' : M,
             }

mdsparams = {'N': 1, # trajectories to run
             'T1': 300,
             'T2': 1000,
             'dt': 0.5,
             'Nc': 3000,
             'Ns': 2,
             'sig': M,
             }

tsparams = {'T':200, # trajectories to run
             'n_samples' : 200,
             'n_steps': 10,
             'steps': 1500,
             'min_steps': 300,
             'sig' : M,
             'tsfiles': ['/home/jsmith48/scratch/auto_rxn_al/rxns/'],
             'nmfile':None,                          #path to gaussian log file containing the data
             'nm':0,                                 #id of normal mode
             'perc':0,                               #Move the molecules initial coordiantes along the mode by this amount. Negative numbers are ok. 
             }

dhparams = { 'Nmol': 250,
             'Nsamp': 2,
             'sig' : M,
             'rng' : 0.2,
             'MaxNa' : 25,
             #'smilefile': '/home/jsmith48/scratch/auto_dhl_al/dhl_files/dhl_genentech.smi',
             #'smilefile': '/home/jsmith48/scratch/Drug_moles_raw/chembl_22_clean_1576904_sorted_std_final.smi',
             'smilefile': '/home/jsmith48/scratch/Drug_moles_raw/chembl_22_clean_1576904_sorted_std_final.smi',
             }

dmrparams = {'mdselect' : [(4000,0),(1600,1),(400,2),(100,3),(25,4),(10,5),(1,6),(1,7),(1,8),(1,9),(1,10),(50,11),(1,12),(2,13),(2,14),],
             #'mdselect' : [(10,0), (1,11)],
             #'N' : 20,
             'maxNa' : 15, # Largest molecule to consider (for dimers max size is 2x maxNa)
             'T' : 400.0, # running temp 
             'L' : 25.0, # box length
             'V' : 0.04, # Random init velocities 
             'dt' : 0.25, # MD time step
             'Nm' : 100, # Molecules to embed
             #'Nm' : 160, # Molecules to embed
             'Nr' : 15, # Number of total boxes to embed and test
             'Ni' : 10, # Number of steps to run the dynamics before fragmenting
             'Ns' : 10,
             'sig': M,
            }

solv_file = '/home/jsmith48/scratch/GDB-11-AL-wB97x631gd/gdb11_s01/inputs/gdb11_s01-2.ipt'
solu_dirs = ''

gcmddict = {'edgepad': 0.8, # padding on the box edge
            'mindist': 1.6, # Minimum allow intermolecular distance
            'sig' : M, # sig hat for data selection
            'maxsig' : 3.0*M, # Max frag sig allowed to continue dynamics
            'Nr': 30, # Number of boxed to run
            'MolHigh': 910, #High number of molecules
            'MolLow': 820, #Low number of molecules
            'Ni': 50, #steps before checking frags
            'Ns': 100,
            'dt': 0.25,
            'V': 0.04,
            'L': 30.0,
            'T': 500.0,
            'Nembed' : 0,
            'solv_file' : solv_file,
            'solu_dirs' : solu_dirs,
            }

### BEGIN CONFORMATIONAL REFINEMENT LOOP HERE ###
#N = [20]
N = [1]

for i in N:
    #netdir = wkdir+'ANI-1x-RXN-0000.00'+str(i).zfill(2)+'/'
    netdir = wkdir + 'ANI-AL-smallions-' + str(i).zfill(2) + '/'
    ldtdir = root_dir  # local data directories

    if not os.path.exists(netdir):
        os.mkdir(netdir)

    nnfprefix   = netdir + 'train'

    netdict = {'iptfile' : iptfile,
               'cnstfile' : cstfile,
               'saefile': saefile,
               'nnfprefix': netdir+'train',
               'aevsize': aevsize,
               'num_nets': Nnets,
               'atomtyp' : ['H','C','N','O','S','F','Cl']
               }

    ## Train the ensemble ##
    aln = att.alaniensembletrainer(netdir, netdict, h5stor, Nnets)
    aln.build_strided_training_cache(Nblock,Nbvald,Nbtest,False)
    aln.train_ensemble(GPU)

    if not os.path.exists(root_dir + datdir + str(i+1).zfill(2)):
        os.mkdir(root_dir + datdir + str(i+1).zfill(2))

    ## Run active learning sampling ##
    acs = alt.alconformationalsampler(ldtdir, datdir + str(i+1).zfill(2), optlfile, fpatoms+['H'], netdict)
    #acs.run_sampling_cluster(gcmddict, GPU)
    #acs.run_sampling_dimer(dmrparams, GPU)
    acs.run_sampling_nms(nmsparams, GPU)
    #acs.run_sampling_md(mdsparams, perc=0.25, gpus=GPU+GPU+GPU)
    #acs.run_sampling_dhl(dhparams, gpus=GPU+GPU)
    #acs.run_sampling_TS(tsparams, gpus=GPU)

    ## Submit jobs, return and pack data
    ast.generateQMdata(hostname, username, swkdir, ldtdir, datdir + str(i+1).zfill(2), h5stor, mae, jtime,max_jobs=50)


