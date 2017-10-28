import hdnntools as hdt
import pyaniasetools as pya
import numpy as np

molfile = '/home/jujuman/Research/MD_TEST/trim.xyz'
newfile = '/home/jujuman/Research/MD_TEST/trim.xyz'

wkdircv = '/home/jujuman/Gits/ANI-Networks/networks/ANI-c08f-ntwk-cv/'
cnstfilecv = wkdircv + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefilecv  = wkdircv + 'sae_6-31gd.dat'
nnfprefix   = wkdircv + 'cv_train_'

anicv = pya.anicrossvalidationmolecule(cnstfilecv,saefilecv,nnfprefix,5,0,False)

xyz,spc,Na = hdt.readxyz2(molfile)
#xyz = xyz[72700:72900]
N = xyz.shape[0]

#hdt.writexyzfile(newfile,xyz,spc)

print('Number of steps:',N)

anicv.set_molecule(xyz[0],spc)

sa = np.empty(N,dtype=np.float32)
for i,x in enumerate(xyz):
    sigma = anicv.compute_stddev_molecule(x)
    sa[i] = sigma

    if i%100 == 0:
        print(i,'):',sigma)



import matplotlib.pyplot as plt

plt.plot(sa)
plt.ylabel('$Std. Dev./atom$ (kcal/mol)')
plt.xlabel('Frame')
plt.show()
