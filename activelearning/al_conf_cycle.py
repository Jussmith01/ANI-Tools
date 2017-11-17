import anialservertools as ast
import anialtools as alt
from time import sleep
import os

hostname = "comet.sdsc.xsede.org"
#hostname = "moria.chem.ufl.edu"
username = "jsmith48"

root_dir = '/home/jujuman/Research/test_auto_al/'

swkdir = '/home/jsmith48/scratch/test_autoal/'# server working directory
datdir = 'ANI-AL-0605.0001.00'

h5stor = root_dir + 'h5files/'# h5store location

optlfile = root_dir + 'optimized_input_files.dat'

mae = 'module load gnu/4.9.2\n' +\
      'module load gaussian\n' +\
      'export PATH="/home/$USER/Gits/RCDBuilder/build/bin:$PATH"\n' +\
      'export LD_LIBRARY_PATH="/home/$USER/Gits/RCDBuilder/build/lib:$LD_LIBRARY_PATH"\n'

fpatoms = ['C', 'N', 'O', 'S', 'F', 'Cl']
aevsize = 1008

wkdir = '/home/jujuman/Research/test_auto_al/modelCNOSFCl/ANI-AL-0605/ANI-AL-0605.0001/'

#---- Training Parameters ----
GPU = [0,1] # GPU IDs

trdict = dict({'learningrate' : 0.001,
               'lrannealing' : 0.5,
               'lrconvergence' : 1.0e-4,
               'ST' : 10,
               'printstep' : 1,
                })

M   = 0.34 # Max error per atom in kcal/mol
Nnets = 2 # networks in ensemble

saefile = '/home/jujuman/Research/test_auto_al/modelCNOSFCl/sae_wb97x-631gd.dat'
cstfile = '/home/jujuman/Research/test_auto_al/modelCNOSFCl/rHCNOSFCl-4.6A_16-3.1A_a4-8.params'
#-----------0---------

# Training varibles
d = dict({#'wkdir'         : wkdir,
          'sflparamsfile' : cstfile,
          #'ntwkStoreDir'  : wkdir+'networks/',
          'atomEnergyFile': saefile,
          #'datadir'       : datadir,
          'tbtchsz'       : '64',
          'vbtchsz'       : '64',
          #'gpuid'         : str(GPU),
          'ntwshr'        : '0',
          'nkde'          : '2',
          'force'         : '0',
          'fmult'         : '0.01',
          'runtype'       : 'ANNP_CREATE_HDNN_AND_TRAIN',
          'adptlrn'       : 'OFF',
          'moment'        : 'ADAM',})

l1 = dict({'nodes'      : '32',
           'activation' : '5',
           'maxnorm'    : '1',
           'norm'       : '3.0',
           'btchnorm'   : '0',})

l2 = dict({'nodes'      : '32',
           'activation' : '5',
           'maxnorm'    : '1',
           'norm'       : '3.0',
           'btchnorm'   : '0',})

l3 = dict({'nodes'      : '16',
           'activation' : '5',
           'maxnorm'    : '1',
           'norm'       : '3.0',
           'btchnorm'   : '0',})

l4 = dict({'nodes'      : '1',
           'activation' : '6',})

layers = [l1, l2, l3, l4,]

#### Sampling parameters ####
nmsparams = {'T': 1000.0,
             'Ngen': 100,
             'Nkep': 50,
             }

mdsparams = {'N': 10,
             'T': 800,
             'dt': 0.5,
             'Nc': 500,
             'Ns': 5,
             }

### BEGIN LOOP HERE ###
N = 2

for i in range(N):
    netdir = wkdir+'ANI-AL-0605.0001.'+str(i).zfill(4)+'/'
    if not os.path.exists(netdir):
        os.mkdir(netdir)

    nnfprefix   = netdir + 'train'

    netdict = {'cnstfile' : cstfile,
               'saefile': saefile,
               'nnfprefix': netdir+'train',
               'aevsize': aevsize,
               'num_nets': Nnets,
               }

    aet = alt.alaniensembletrainer(netdir, netdict, 'train', h5stor, Nnets)

    aet.build_training_cache()
    aet.train_ensemble(GPU, d, trdict, layers)

    ldtdir = root_dir  # local data directories
    if not os.path.exists(root_dir + datdir + str(i+1).zfill(2)):
        os.mkdir(root_dir + datdir + str(i+1).zfill(2))
    acs = alt.alconformationalsampler(ldtdir, datdir + str(i+1).zfill(2), optlfile, fpatoms, netdict)
    acs.run_sampling(nmsparams, mdsparams, GPU)

    ast.generateQMdata(hostname, username, swkdir, ldtdir, datdir + str(i+1).zfill(2), h5stor, mae)