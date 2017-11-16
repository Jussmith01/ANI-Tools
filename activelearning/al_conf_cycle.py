import anialservertools as ast
import anialtools as alt
from time import sleep

hostname = "comet.sdsc.xsede.org"
#hostname = "moria.chem.ufl.edu"
username = "jsmith48"

root_dir = '/home/jujuman/Research/test_auto_al/'

swkdir = '/home/jsmith48/scratch/test_autoal/'# server working directory
ldtdir = root_dir# local data directories
datdir = 'confs_1'

h5stor = root_dir + '/h5files/'# h5store location

optlfile = root_dir + '/optimized_input_files.dat'

mae = 'module load gnu/4.9.2\n' +\
      'module load gaussian\n' +\
      'export PATH="/home/$USER/Gits/RCDBuilder/build/bin:$PATH"\n' +\
      'export LD_LIBRARY_PATH="/home/$USER/Gits/RCDBuilder/build/lib:$LD_LIBRARY_PATH"\n'

fpatoms = ['C', 'N', 'O', 'S', 'F', 'Cl']
aevsize = 1008

wkdir = '/home/jujuman/Research/test_auto_al/modelCNOSFCl/ANI-AL-0605/ANI-AL-0605.0001/'

#---- Training Parameters ----
GPU = 0 # GPU ID
LR  = 0.001 # Initial learning rate
LA  = 0.25 # LR annealing
CV  = 1.0e-6 # LR converg
ST  = 100 # ????
M   = 0.34 # Max error per atom in kcal/mol
P   = 0.01 # Percent to keep
ps  = 20 # Print step
Naev = 384 #
sinet= False

saefile = '/home/jujuman/Research/test_auto_al/modelCNOSFCl/'
cstfile = '/home/jujuman/Research/test_auto_al/modelCNOSFCl/'
#-----------0---------

# Training varibles
d = dict({'wkdir'         : wkdir,
          'sflparamsfile' : cstfile,
          'ntwkStoreDir'  : wkdir+'networks/',
          'atomEnergyFile': saefile,
          'datadir'       : datadir,
          'tbtchsz'       : '256',
          'vbtchsz'       : '256',
          'gpuid'         : str(GPU),
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

l3 = dict({'nodes'      : '64',
           'activation' : '5',
           'maxnorm'    : '1',
           'norm'       : '3.0',
           'btchnorm'   : '0',})

l4 = dict({'nodes'      : '1',
           'activation' : '6',})

layers = [l1, l2, l3, l4,]

### BEGIN LOOP HERE ###

N = 1

for i in range(N):
    netdir = wkdir+'ANI-AL-0605.0001.'+str(N).zfill(4)
    nnfprefix   = netdir + 'train'

    nmsparams = {'T' : 1000.0,
                 'Ngen' : 400,
                 'Nkep' : 100,
                 }

    mdsparams = {'N' : 10,
                 'T' : 800,
                 'dt' : 0.5,
                 'Nc' : 500,
                 'Ns' : 5,
                 }



    netdict = {'cnstfile' : cnstfile,
               'saefile': saefile,
               'nnfprefix': nnfprefix,
               'aevsize': aevsize,
               'num_nets': 5,
               }

    alt.alaniensembletrainer()

    #acs = alt.alconformationalsampler(ldtdir, datdir, optlfile, fpatoms, netdict)
    #acs.run_sampling(nmsparams, mdsparams, [0, 1])

    #ast.generateQMdata(hostname, username, swkdir, ldtdir, datdir, h5stor, mae)