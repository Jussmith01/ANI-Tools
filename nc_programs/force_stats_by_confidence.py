import pyanitools as pyt
import pyaniasetools as aat

#--------------Parameters------------------
#wkdir = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb01-06_red03-08_mdal01/cv1/'
wkdir = '/home/jujuman/Gits/ANI-Networks/networks/ANI-c08f-ntwk-cv/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile = wkdir + 'sae_6-31gd.dat'
nnfprefix  = wkdir + 'cv_train_'
Ncv = 5

h5file = '/home/jujuman/Scratch/Research/extensibility_test_sets/gdb-07/gdb11_07_test500.h5'

#-------------------------------------------

# Define the conformer cross validator class
anicv = aat.anicrossvalidationconformer(cnstfile,saefile,nnfprefix,Ncv,0,False)

# Declare data loader
adl = pyt.anidataloader(h5file)

for data in adl:
    # Extract the data
    X  = data['coordinates']
    S  = data['species']
    Edft = data['energies']
    Fdft = data['forces']#/(0.52917724900001*0.52917724900001)
    path = data['path']

    # Calculate std. dev. per atom for all conformers
    sigma = anicv.compute_stddev_conformations(X, S)

    print(sigma.shape)

    # Calculate energy deltas
    Eani, Fani = anicv.compute_energy_conformations(X, S)

    # Convert to kcal/mol and reshape if needed
    Eani = hdn.hatokcal * Eani
    Edft = hdn.hatokcal * Edft

    Fani = hdn.hatokcal * Fani.reshape(Ncv, -1)
    Fdft = hdn.hatokcal * Fdft.reshape(-1)