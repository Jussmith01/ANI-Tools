import anialservertools as ast
import anialtools as alt
from time import sleep
import os
import sys

passfile = "/home/jsmith48/tkey.dat"

#hostname = "comet.sdsc.xsede.org"
hostname = "bridges.psc.edu"
#hostname = "moria.chem.ufl.edu"
username = "jsmith8"
password = open(passfile,'r').read().strip()

root_dir = '/home/jsmith48/scratch/auto_al/'

swkdir = '/home/jsmith8/scratch/auto_al_cycles/'# server working directory
datdir = 'ANI-AL-0808.0302.04'

h5stor = root_dir + 'h5files/'# h5store location

optlfile = root_dir + 'optimized_input_files.dat'

#Comet
#mae = 'module load gnu/4.9.2\n' +\
#      'module load gaussian\n' +\
#      'export PATH="/home/$USER/Gits/RCDBuilder/build/bin:$PATH"\n' +\
#      'export LD_LIBRARY_PATH="/home/$USER/Gits/RCDBuilder/build/lib:$LD_LIBRARY_PATH"\n'

# Bridges
mae = 'module load gcc/5.3.0\n' +\
      'module load gaussian/09.D.01\n' +\
      'export PATH="/home/$USER/Gits/RCDBuilder/build/bin:$PATH"\n' +\
      'export LD_LIBRARY_PATH="/home/$USER/Gits/RCDBuilder/build/lib:$LD_LIBRARY_PATH"\n' +\
      "export GAUSS_EXEDIR='/opt/packages/gaussian-RevD.01/g09/' \n"

fpatoms = ['C', 'N', 'O', 'S', 'F', 'Cl']

jtime = "0-2:30"

#---- Training Parameters ----
GPU = [2,3,4,6,7] # GPU IDs

M   = 0.34 # Max error per atom in kcal/mol
Nnets = 5 # networks in ensemble
aevsize = 1008

wkdir = '/home/jsmith48/scratch/auto_al/modelCNOSFCl/ANI-AL-0808/ANI-AL-0808.0302/'
iptfile = '/home/jsmith48/scratch/auto_al/modelCNOSFCl/inputtrain.ipt'
saefile = '/home/jsmith48/scratch/auto_al/modelCNOSFCl/sae_wb97x-631gd.dat'
cstfile = '/home/jsmith48/scratch/auto_al/modelCNOSFCl/rHCNOSFCl-4.6A_16-3.1A_a4-8.params'
#-----------0---------

# Training varibles

#### Sampling parameters ####
nmsparams = {'T': 600.0, # Temperature
             'Ngen': 10, # Confs to generate
             'Nkep': 2, # Diverse confs to keep
             'sig' : M,
             }

mdsparams = {'N': 2, # trajectories to run
             'T1': 300,
             'T2': 1000,
             'dt': 0.5,
             'Nc': 3000,
             'Ns': 2,
             'sig': M,
             }

dmrparams = {#'mdselect' : [(400,0),(60,2),(40,3),(5,4)],
             'mdselect' : [(10,0), (1,11)],
             'N' : 20,
             'T' : 400.0, # running temp 
             'L' : 25.0, # box length
             'V' : 0.04, # Random init velocities 
             'dt' : 0.5, # MD time step
             'Nm' : 140, # Molecules to embed
             #'Nm' : 160, # Molecules to embed
             'Nr' : 50, # Number of total boxes to embed and test
             'Ni' : 10, # Number of steps to run the dynamics before fragmenting
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
N = [9]

for i in N:
    netdir = wkdir+'ANI-AL-0808.0302.04'+str(i).zfill(2)+'/'
    if not os.path.exists(netdir):
        os.mkdir(netdir)

    nnfprefix   = netdir + 'train'

    netdict = {'iptfile' : iptfile,
               'cnstfile' : cstfile,
               'saefile': saefile,
               'nnfprefix': netdir+'train',
               'aevsize': aevsize,
               'num_nets': Nnets,
               }

    ## Train the ensemble ##
    aet = alt.alaniensembletrainer(netdir, netdict, 'train', h5stor, Nnets)
    aet.build_training_cache()
    aet.train_ensemble(GPU)

    ldtdir = root_dir  # local data directories
    if not os.path.exists(root_dir + datdir + str(i+1).zfill(2)):
        os.mkdir(root_dir + datdir + str(i+1).zfill(2))

    ## Run active learning sampling ##
    acs = alt.alconformationalsampler(ldtdir, datdir + str(i+1).zfill(2), optlfile, fpatoms, netdict)
    acs.run_sampling_cluster(gcmddict, GPU)
    acs.run_sampling_dimer(dmrparams, GPU)
    acs.run_sampling_nms(nmsparams, GPU)
    acs.run_sampling_md(mdsparams, perc=0.5, gpus=GPU)

    ## Submit jobs, return and pack data
    ast.generateQMdata(hostname, username, swkdir, ldtdir, datdir + str(i+1).zfill(2), h5stor, mae, jtime, password=password)


