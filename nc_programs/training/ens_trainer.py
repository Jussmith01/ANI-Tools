import anialtools as alt

nwdir = '/home/jsmith48/scratch/ANI-1x_retrain/large_ens/'
h5dir = '/home/jsmith48/scratch/ANI-1x_dataset/'

Nnets = 16 # networks in ensemble
Nblock = 16 # Number of blocks in split
Nbvald = 2 # number of valid blocks
Nbtest = 1 # number of test blocks

netdict = {'iptfile' :nwdir+'inputtrain.ipt',
           'cnstfile':nwdir+'rHCNO-4.6R_16-3.1A_a4-8.params',
           'saefile' :nwdir+'sae_linfit.dat',}

GPU = [0]

## Train the ensemble ##
aet = alt.alaniensembletrainer(nwdir, netdict, h5dir, Nnets)
aet.build_strided_training_cache(Nblock,Nbvald,Nbtest,False)
aet.train_ensemble(GPU)
