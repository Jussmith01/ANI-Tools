import numpy as np
import hdnntools as gt
import pyanitools as pyt
import os

lfile = '/home/jujuman/DataTesting/gdb9-2500-div-dim.h5'
sfile = '/home/jujuman/DataTesting/gdb9-2500-div-dim_35.h5'

if os.path.exists(sfile):
    os.remove(sfile)

adl = pyt.anidataloader(lfile)
dpk = pyt.datapacker(sfile)

for i,x in enumerate(adl):
    print(i)
    xyz = np.asarray(x['coordinates'],dtype=np.float32)
    erg = x['energies']
    spc = x['species']

    dpk.store_data('/gdb-09-DIV/mol'+str(i), coordinates=xyz.reshape(erg.shape[0],len(spc)*3), energies=erg, species=spc)

adl.cleanup()
dpk.cleanup()