import pyanitrainer as atr

import os

# Network 1 Files
wkdir = '/home/jujuman/Research/DataReductionMethods/model_9.0.5_reduce/train/'
cnstf = 'rHCNO-4.6A_16-3.1A_a4-8.params'
saenf = 'sae_6-31gd.dat'
nfdir = 'networks/'

opt = 'active_output.opt'

# Data Dir
datadir = '/home/jujuman/Research/DataReductionMethods/model_9.0.5_reduce/cache/'
testdata = datadir + 'testset/testset.h5'
trainh5 = wkdir + 'ani_red9.0.5_ALfull.h5'

# Test data
test_files = ['/home/jujuman/Research/GDB_Dimer/dimers1_fix.h5',
              '/home/jujuman/Research/GDB_Dimer/dimers2_fix.h5',
              '/home/jujuman/Research/GDB_Dimer/dimers3_fix.h5',
              '/home/jujuman/Research/GDB_Dimer/dimers4_fix.h5',
              '/home/jujuman/Research/GDB_Dimer/dimers5_fix.h5',
              '/home/jujuman/Research/GDB_Dimer/dimers6_fix.h5',
              '/home/jujuman/Research/GDB_Dimer/dimer_gen_7/dimers7.h5',
              '/home/jujuman/Research/ReactionGeneration/reactiondata/DA_rxn_1/DA_rxn_1.h5',
              '/home/jujuman/Research/ReactionGeneration/reactiondata/DA_rxn_1/DA_rxn_1_2.h5',
              '/home/jujuman/Research/ReactionGeneration/reactiondata/comb_rxn_1/comb_rxn_1.h5',
              '/home/jujuman/Research/ReactionGeneration/reactiondata/comb_rxn_1/comb_rxn_1_2.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb09_1/ani_al-9.0.5.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb09_1/ani_al-9.0.4.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb09_1/ani_al-9.0.3.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb09_1/ani_al-9.0.2.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb09_1/ani_al-9.0.1.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_3/gdb_r06_comb08_03_4.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_3/gdb_r06_comb08_03_3.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_3/gdb_r06_comb08_03_2.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_3/gdb_r06_comb08_03_1.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_2/gdb_r06_comb08_02_4.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_2/gdb_r06_comb08_02_3.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_2/gdb_r06_comb08_02_2.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_2/gdb_r06_comb08_02_1.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_1/gdb_r06_comb08_5.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_1/gdb_r06_comb08_4.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_1/gdb_r06_comb08_3.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_1/gdb_r06_comb08_2.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_comb_resample/gdb_r06_comb08_1/gdb_r06_comb08_1.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_mdal_resample/mdal.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/h2o_cluster/h2o_nms_clusters.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-05_red03-05/confs_cv_gdb01-05_rs1.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-05_red03-05/confs_cv_gdb01-05_rs2.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-05_red03-05/confs_cv_gdb01-05_rs3.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-05_red03-05/confs_cv_gdb01-05_rs4.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-06/confs_cv_gdb01-06_rs1.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-06/confs_cv_gdb01-06_rs2.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-06/confs_cv_gdb01-06_rs3.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-06/confs_cv_gdb01-06_rs4.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-07/confs_cv_gdb01-07_rs1.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-07/confs_cv_gdb01-07_rs2.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-07/confs_cv_gdb01-07_rs3.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-07/confs_cv_gdb01-07_rs4.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-08/confs_cv_gdb01-08_rs1.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-08/confs_cv_gdb01-08_rs2.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-08/confs_cv_gdb01-08_rs3.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/dnnts_nms_resample/confs_cv_gdb01-06_red03-08/confs_cv_gdb01-08_rs4.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S01_06r.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S02_06r.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S03_06r.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S04_06r.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S05_06r.h5',
              '/home/jujuman/Research/GDB-11-AL-wB97x631gd/gdb11_h5/gdb11_S06_06r.h5',
            ]

#---- Parameters ----
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
          'force'         : '0',
          'fmult'         : '0.01',
          'runtype'       : 'ANNP_CREATE_HDNN_AND_TRAIN',
          'adptlrn'       : 'OFF',
          'moment'        : 'ADAM',})

l1 = dict({'nodes'      : '128',
           'activation' : '5',
           'maxnorm'    : '1',
           'norm'       : '3.0',
           'btchnorm'   : '0',})

l2 = dict({'nodes'      : '128',
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

aani = atr.ActiveANI(test_files, wkdir+saenf, wkdir+opt, datadir, testdata, Naev)
aani.init_dataset(P)

inc = 0
while aani.get_percent_bad() > 5.0:
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
    ant = atr.anitester(wkdir+cnstf, wkdir+saenf, wkdir+nfdir, GPU, sinet)
    test_rmse_e, test_rmse_f = ant.compute_test(testdata)
    print('Test E RMSE:', "{:.3f}".format(test_rmse_e), 'kcal/mol')
    print('Test F RMSE:', "{:.3f}".format(test_rmse_f), 'kcal/mol/A')
    
    # Check for and add bad data
    aani.add_bad_data(wkdir+cnstf, wkdir+saenf, wkdir+nfdir, GPU, sinet, P=0.01 + inc * 0.01, M=M)

    inc = inc + 1

aani.add_bad_data(wkdir + cnstf, wkdir + saenf, wkdir + nfdir, GPU, sinet, P=1.0, M=M)
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
ant = atr.anitester(wkdir + cnstf, wkdir + saenf, wkdir + nfdir, GPU, sinet)
test_rmse = ant.compute_test(testdata)
print('Final Test RMSE:', "{:.3f}".format(test_rmse), 'kcal/mol')

o = open(wkdir + 'keep_info.dat', 'w')
for k in aani.get_keep_info():
    o.write(str(int(k[1])) + ' : ' + str(k[0]) + '\n')

f = open(wkdir + 'diffs.dat', 'w')
for K in aani.get_diff_kept (wkdir + cnstf, wkdir + saenf, wkdir + nfdir, GPU, sinet, M=M):
    string = ""
    for k in K:
        string = string + "{:.7f}".format(k) + ','
    f.write(string[:-1] + '\n')
