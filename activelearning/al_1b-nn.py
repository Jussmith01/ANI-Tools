import anialservertools as ast
import anitraintools as att
import anialtools as alt
from time import sleep,time
import os
import sys

#passfile = "/home/jsmith48/tkey.dat"

hostname = "comet.sdsc.xsede.org"
#hostname = "bridges.psc.edu"
#hostname = "moria.chem.ufl.edu"
username = "cruzeiro"
#password = open(passfile,'r').read().strip()

root_dir = '/server-home1/cruzeiro/auto_al/'

swkdir = '/home/cruzeiro/scratch/auto_al_cycles/'# server working directory
datdir = 'ANI-AL-n2o5-1b-'

h5stor = root_dir + 'h5files/'# h5store location

strucsfolder = root_dir + 'strucs/'# strucs location

optlfile = root_dir + 'input_files.dat'

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

#fpatoms = ['C', 'N', 'O']
fpatoms = ['N', 'O']

jtime = "0-12:00"

#---- Training Parameters ----
GPU = [1,2] # GPU IDs

M   = 0.06 # Max error per atom in kcal/mol
Nnets = 8 # networks in ensemble
Nblock = 16 # Number of blocks in split
Nbvald = 2 # number of valid blocks
Nbtest = 2 # number of test blocks
aevsize = 128

#wkdir = '/home/jsmith48/scratch/auto_rxn_al/modelrxn/ANI-1x-RXN-0000/'
#iptfile = '/home/jsmith48/scratch/auto_rxn_al/modelrxn/inputtrain.ipt'
#saefile = '/home/jsmith48/scratch/auto_rxn_al/modelrxn/sae_linfit.dat'
#cstfile = '/home/jsmith48/scratch/auto_rxn_al/modelrxn/rHCNO-4.6R_16-3.1A_a4-8.params'

root = '/server-home1/cruzeiro/auto_al'

wkdir   = root+'/models/ANI-AL-n2o5-1b/'
#iptfile = root+'/modelions/inputtrain.ipt'
saefile = root+'/models/ANI-AL-n2o5-1b/sae_linfit.dat'
cstfile = root+'/models/ANI-AL-n2o5-1b/rNO-5.2R_16-3.5A_a4-8.params'

ipt = alt.anitrainerinputdesigner()

ipt.set_parameter('atomEnergyFile','sae_linfit.dat')
ipt.set_parameter('sflparamsfile','rNO-5.2R_16-3.5A_a4-8.params')
ipt.set_parameter('eta','0.001')
ipt.set_parameter('energy','1')
ipt.set_parameter('force','0')
ipt.set_parameter('fmult','1.0')
ipt.set_parameter('feps','0.001')
ipt.set_parameter('dipole','0')
ipt.set_parameter('charge','0')
ipt.set_parameter('cdweight','2.0')
ipt.set_parameter('tolr','100')
ipt.set_parameter('tbtchsz','35')
ipt.set_parameter('vbtchsz','35')
ipt.set_parameter('nkde','2')


# Set network layers
#ipt.add_layer('H',{"nodes":160,"activation":9,"type":0,"l2norm":1,"l2valu":5.000e-3})
#ipt.add_layer('H',{"nodes":128,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-6})
#ipt.add_layer('H',{"nodes":96 ,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-6})

#ipt.add_layer('C',{"nodes":144,"activation":9,"type":0,"l2norm":1,"l2valu":5.000e-3})
#ipt.add_layer('C',{"nodes":112,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-6})
#ipt.add_layer('C',{"nodes":96 ,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-6})

#ipt.add_layer('N',{"nodes":128,"activation":9,"type":0,"l2norm":1,"l2valu":5.000e-3})
#ipt.add_layer('N',{"nodes":112,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-6})
#ipt.add_layer('N',{"nodes":96 ,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-6})

#ipt.add_layer('O',{"nodes":128,"activation":9,"type":0,"l2norm":1,"l2valu":5.000e-3})
#ipt.add_layer('O',{"nodes":112,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-6})
#ipt.add_layer('O',{"nodes":96 ,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-6})

ipt.add_layer('N',{"nodes":32,"activation":9,"type":0,"l2norm":1,"l2valu":5.000e-3})
ipt.add_layer('N',{"nodes":28,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-6})
ipt.add_layer('N',{"nodes":24,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-6})

ipt.add_layer('O',{"nodes":32,"activation":9,"type":0,"l2norm":1,"l2valu":5.000e-3})
ipt.add_layer('O',{"nodes":28,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-6})
ipt.add_layer('O',{"nodes":24,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-6})

ipt.print_layer_parameters()

#-----------0---------

# Training varibles

#### Sampling parameters ####
nmsparams = {'T': 700.0, # Temperature
             'Ngen': 300, # Confs to generate
             'Nkep': 200, # Diverse confs to keep
             'maxd': 0.5, # Diverse confs to keep
             'sig' : M,
             }

strucsparams = {'N': 50, # number of maximum structures to select before QBC from each XYZ file
                'sig': M,
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
#N = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
N = [41]

for i in N:
    #netdir = wkdir+'ANI-1x-RXN-0000.00'+str(i).zfill(2)+'/'
    netdir = wkdir + 'ANI-AL-n2o5-1b-' + str(i).zfill(2) + '/'
    ldtdir = root_dir  # local data directories

    if not os.path.exists(netdir):
        os.mkdir(netdir)

    nnfprefix   = netdir + 'train'

    netdict = {'cnstfile' : cstfile,
               'saefile': saefile,
               'iptsize': 128,
               'nnfprefix': netdir+'train',
               'aevsize': aevsize,
               'num_nets': Nnets,
               'atomtyp' : ['N','O']
               }

    ## Train the ensemble ##
    #aln = att.alaniensembletrainer(netdir, netdict, ipt, h5stor, Nnets)
    #aln.build_strided_training_cache(Nblock,Nbvald,Nbtest,Ekey='energies',forces=False,dipole=False,rmhighe=True)
    #aln.train_ensemble(GPU)

    if not os.path.exists(root_dir + datdir + str(i+1).zfill(2)):
        os.mkdir(root_dir + datdir + str(i+1).zfill(2))

    ## Run active learning sampling ##
    acs = alt.alconformationalsampler(ldtdir, datdir + str(i+1).zfill(2), optlfile, strucsfolder, fpatoms+['H'], netdict)
    #acs.run_sampling_cluster(gcmddict, GPU)
    #acs.run_sampling_dimer(dmrparams, GPU)
    #acs.run_sampling_nms(nmsparams, GPU)
    acs.run_sampling_strucs(strucsparams, GPU)
    #acs.run_sampling_md(mdsparams, perc=0.25, gpus=GPU+GPU+GPU)
    #acs.run_sampling_dhl(dhparams, gpus=GPU+GPU)
    #acs.run_sampling_TS(tsparams, gpus=GPU)

    ## Submit jobs, return and pack data
    ast.generateQMdata(hostname, username, swkdir, ldtdir, datdir + str(i+1).zfill(2), h5stor, mae, jtime,max_jobs=50)
