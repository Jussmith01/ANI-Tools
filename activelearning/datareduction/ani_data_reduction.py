import pyanitrainer as atr

import os

# Network 1 Files
wkdir = '/home/jujuman/Research/DataReductionMethods/models/train_test/'
cnstf = 'rHCNO-4.6A_16-3.1A_a4-8.params'
saenf = 'sae_6-31gd.dat'
nfdir = 'networks/'

opt = 'active_output.opt'

# Data Dir
datadir = '/home/jujuman/Research/DataReductionMethods/models/cachetest/'
testdata = datadir + 'testset/testset.h5'
trainh5 = wkdir + 'ani_red_c08f.h5'

# Test data
test_files = [#"/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c08f.h5",
              "/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c01.h5",
              "/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c02.h5",
              "/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c03.h5",
              "/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c04.h5",
              #"/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c05.h5",
              #"/home/jujuman/Research/ANI-DATASET/h5data/ani-gdb-c06.h5",
              ]

#---- Parameters ----
GPU = 0
LR  = 0.001
LA  = 0.25
CV  = 1.0e-6
ST  = 100
M   = 0.08 # Max error per atom in kcal/mol
P   = 0.025
ps  = 25
Naev = 388
#--------------------

# Training varibles
d = dict({'wkdir'         : wkdir,
          'sflparamsfile' : cnstf,
          'ntwkStoreDir'  : wkdir+'networks/',
          'atomEnergyFile': saenf,
          'datadir'       : datadir,
          'tbtchsz'       : '512',
          'vbtchsz'       : '512',
          'gpuid'         : str(GPU),
          'ntwshr'        : '1',
          'nkde'          : '2',
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

l3 = dict({'nodes'      : '1',
           'activation' : '6',})

layers = [l1, l2, l3,]

aani = atr.ActiveANI(test_files, wkdir+saenf, wkdir+opt, datadir, testdata, Naev)
aani.init_dataset(P)

inc = 0
while aani.get_percent_bad() > 2.0:
    # Remove existing network
    network_files = os.listdir(wkdir + 'networks/')
    for f in network_files:
        os.remove(wkdir + 'networks/' + f)

    # Setup trainer
    tr = atr.anitrainer(d,layers)

    # Train network
    tr.train_network(LR, LA, CV, ST, ps)

    # Write the learning curve
    tr.write_learning_curve(wkdir+'learning_curve_'+str(inc)+'.dat')

    # Test network
    ant = atr.anitester(wkdir+cnstf, wkdir+saenf, wkdir+nfdir, GPU, True)
    test_rmse = ant.compute_test(testdata)
    print('Test RMSE:',"{:.3f}".format(test_rmse),'kcal/mol')

    # Check for and add bad data
    aani.add_bad_data(wkdir+cnstf, wkdir+saenf, wkdir+nfdir, GPU, True, P=0.05 + inc * 0.025, M=M)

    inc = inc + 1

aani.add_bad_data(wkdir + cnstf, wkdir + saenf, wkdir + nfdir, GPU, True, P=1.0, M=M)
aani.store_train_h5(trainh5)

# Remove existing network
network_files = os.listdir(wkdir + 'networks/')
for f in network_files:
    os.remove(wkdir + 'networks/' + f)

# Setup trainer
tr = atr.anitrainer(d, layers)

# Train network
tr.train_network(LR, LA, CV, ST, ps)

# Test network
ant = atr.anitester(wkdir + cnstf, wkdir + saenf, wkdir + nfdir, GPU, True)
test_rmse = ant.compute_test(testdata)
print('Final Test RMSE:', "{:.3f}".format(test_rmse), 'kcal/mol')

o = open(wkdir + 'keep_info.dat', 'w')
for k in aani.get_keep_info():
    o.write(str(int(k[1])) + ' : ' + str(k[0]) + '\n')

f = open(wkdir + 'diffs.dat', 'w')
for K in aani.get_diff_kept (wkdir + cnstf, wkdir + saenf, wkdir + nfdir, GPU, True, M=M):
    string = ""
    for k in K:
        string = string + "{:.7f}".format(k) + ','
    f.write(string[:-1] + '\n')
