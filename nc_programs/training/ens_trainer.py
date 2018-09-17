import anialtools as alt

nwdir = '/home/jsmith48/scratch/ccsd_extrapolation/ccsd_train/tl_train_dhl_7/'
h5dir = '/home/jsmith48/scratch/ccsd_extrapolation/h5files/train/cmb/'

Nnets = 8 # networks in ensemble
Nblock = 16 # Number of blocks in split
Nbvald = 2 # number of valid blocks
Nbtest = 1 # number of test blocks

netdict = {'iptfile' :nwdir+'inputtrain.ipt',
           'cnstfile':nwdir+'rHCNO-5.2R_16-3.5A_a4-8.params',
           'saefile' :nwdir+'sae_linfit.dat',
           'atomtyp' :['H','C','N','O']}

GPU = [2,3,4,5]

## Train the ensemble ##
aet = alt.alaniensembletrainer(nwdir, netdict, h5dir, Nnets)
aet.build_strided_training_cache(Nblock,Nbvald,Nbtest,build_test=False,forces=False)
aet.train_ensemble(GPU)
