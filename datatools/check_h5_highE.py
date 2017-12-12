import pyanitools as pyt
#import pyaniasetools as aat
import numpy as np
import hdnntools as hdt
import os

#import matplotlib.pyplot as plt

file_old = '/home/jsmith48/scratch/auto_al/h5files/ANI-AL-0707.0000.0408.h5'
file_new = '/home/jsmith48/scratch/auto_al/h5files_fix/ANI-AL-0707.0000.0408.h5'

print('Working on file:',file_old)
adl = pyt.anidataloader(file_old)

# Data storage
dpack = pyt.datapacker(file_new, mode='w')

for i,data in enumerate(adl):
    #if i == 20:
    #    break
    X = data['coordinates']
    S = data['species']
    Edft = data['energies']
    path = data['path']
    del data['path']

    #Eani, Fani = anicv.compute_energy_conformations(X=np.array(X,dtype=np.float32),S=S)

    Esae = hdt.compute_sae('/home/jsmith48/scratch/auto_al/modelCNOSFCl/sae_wb97x-631gd.dat',S)

    idx = np.where(np.abs(Edft-Esae)<5.0)
    bidx = np.where(np.abs(Edft-Esae)>=5.0)
    if bidx[0].size > 0:
        # SAE Check
        print(S)
        print(bidx,np.abs(Edft-Esae))
        #hdt.writexyzfile(file_new+'file_'+str(i).zfill(5)+'.xyz', X[bidx], S)

    #Eani_m = np.mean(Eani, axis=0)
    #Fani = np.mean(Fani, axis=0)

    #err = Eani_m - Edft 
    #pae = np.abs(err)/np.sqrt(float(len(S)))
    #idx = np.where(pae > 0.15)
    
    #Nt += err.size
    #Nk += idx[0].size
    
    klist = ['cm5', 'hirshfeld', 'hirdipole', 'forces', 'coordinates', 'spindensities', 'energies']
    #klist = ['CM5', 'hirshfeld', 'forces', 'coordinates', 'energies']
    #print(data.keys())
    
    data_new = data.copy()
    for key in klist:
        data_new[key] = data[key][idx]
        #print(key,type(data[key][0]),type(data_new[key][0]),type(data[key][gidx][0]))
        
    dpack.store_data(path, **data_new)

dpack.cleanup()
