# Import pyNeuroChem
import pyNeuroChem as pync
import hdnntools as hdt
import numpy as np
import matplotlib.pyplot as plt

molfile = '/home/jujuman/Research/MD_TEST/trim.dat'

wkdir    = '/home/jujuman/Scratch/Research/DataReductionMethods/model6r/model-gdb01-06_red03-08/cv1/train4/'
cnstfile = wkdir + '../rHCNO-4.6A_16-3.1A_a4-8.params'
saefile  = wkdir + '../sae_6-31gd.dat'
nnfdir   = wkdir + 'networks/'

wkdir = '/home/jujuman/Gits/ANI-Networks/networks/ANI-c08f-ntwk-cv/'
cnstfile = wkdir + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefile  = wkdir + 'sae_6-31gd.dat'
nnfdir   = wkdir + 'cv_train_4/networks/'

nc = pync.conformers(cnstfile, saefile, nnfdir, 0, False)

data = hdt.readncdatall(molfile)
Edft = data["energies"]

nc.setConformers(confs=data["coordinates"],types=data["species"])
Eani = nc.energy()

print(Eani)

import matplotlib.pyplot as plt

plt.plot(hdt.hatokcal*(Edft),label='DFT')
plt.plot(hdt.hatokcal*(Eani),label='ANI')
plt.ylabel('$E_t$ (kcal/mol)')
plt.xlabel('Frame')

plt.legend(bbox_to_anchor=(0.01, 0.99), loc=2, borderaxespad=0.,fontsize=14)
plt.show()