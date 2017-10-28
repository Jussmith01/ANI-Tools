# pyneurochem
import pyNeuroChem as pync
import pyaniasetools as aat
import hdnntools as hdn

# Define file
xyzfile = '/home/jujuman/gdb11_s08-2213_preandpostopt.xyz'

# Define cross validation networks
wkdircv = '/home/jujuman/Gits/ANI-Networks/networks/ANI-c08f-ntwk-cv/'
cnstfilecv = wkdircv + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefilecv  = wkdircv + 'sae_6-31gd.dat'
nnfprefix   = wkdircv + 'cv_train_'

# Define the conformer cross validator class
anicv = aat.anicrossvalidationconformer(cnstfilecv,saefilecv,nnfprefix,5,0,False)

# Read structure/s
X, S, Na = hdn.readxyz2(xyzfile)

# Calculate std. dev. per atom for all conformers
sigma = anicv.compute_stddev_conformations(X,S)

# Print result
print(sigma)