import pyanitrainer as atr

import os

# Network 1 Files
wkdir = '/home/jujuman/Research/transfer_learning/cv1/train0/'
cnstf = '../rHCNO-4.6A_16-3.1A_a4-8.params'
saenf = '../sae_ccsd_cbs.dat'
nfdir = 'networks/'

opt = 'active_output.opt'

# Data Dir
datadir = '/home/jujuman/Research/transfer_learning/cv1/cache-data-0/'
testdata = datadir + 'testset/testset.h5'

#---- Parameters ----
GPU = 0
LR  = 0.0001
LA  = 0.25
CV  = 1.0e-6
ST  = 100
M   = 0.08 # Max error per atom in kcal/mol
P   = 0.025
ps  = 1
Naev = 384
sinet= False
#--------------------

# Training varibles
d = dict({'wkdir'         : wkdir,
          'sflparamsfile' : cnstf,
          'ntwkStoreDir'  : wkdir+'networks/',
          'atomEnergyFile': saenf,
          'datadir'       : datadir,
          'tbtchsz'       : '1024',
          'vbtchsz'       : '1024',
          'gpuid'         : str(GPU),
          'ntwshr'        : '0',
          'nkde'          : '2',
          'runtype'       : 'ANNP_CREATE_HDNN_AND_TRAIN',
          'adptlrn'       : 'OFF',
          'moment'        : 'ADAM',})

l1 = dict({'nodes'      : '196',
           'activation' : '5',
           'maxnorm'    : '1',
           'norm'       : '3.0',
           'btchnorm'   : '0',})

l2 = dict({'nodes'      : '32',
           'activation' : '5',
           'maxnorm'    : '1',
           'norm'       : '3.0',
           'btchnorm'   : '0',})

l3 = dict({'nodes'      : '96',
           'activation' : '5',
           'maxnorm'    : '1',
           'norm'       : '3.0',
           'btchnorm'   : '0',})

l4 = dict({'nodes'      : '1',
           'activation' : '6',})

layers = [l1, l2, l3, l4,]



# Setup trainer
tr = atr.anitrainer(d,layers)

# Test network
ant = atr.anitester(wkdir+cnstf, wkdir+saenf, wkdir+nfdir, GPU, sinet)
test_rmse = ant.compute_test(testdata)
print('Test RMSE:',"{:.3f}".format(test_rmse),'kcal/mol')

# Train network
tr.train_network(LR, LA, CV, ST, ps)

# Write the learning curve
#tr.write_learning_curve(wkdir+'learning_curve_'+str(inc)+'.dat')

# Test network
ant = atr.anitester(wkdir+cnstf, wkdir+saenf, wkdir+nfdir, GPU, sinet)
test_rmse = ant.compute_test(testdata)
print('Test RMSE:',"{:.3f}".format(test_rmse),'kcal/mol')
