import pyanitools as pyt

adl = pyt.anidataloader('/home/jujuman/Research/ANI-DATASET/h5data/r10_ccsd.h5')

for i,data in enumerate(adl):
    print(data['energies'])
