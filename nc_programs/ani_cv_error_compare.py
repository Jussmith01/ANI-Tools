# pyneurochem
import pyNeuroChem as pync
import pyanitools as pyt
import pyaniasetools as aat
import hdnntools as hdn

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

# Define test file
#h5file = '/home/jujuman/Research/ForceNMPaper/polypeptide/tripeptide_full.h5'
h5file = '/home/jujuman/Research/ANI-DATASET/ANI-1_release/ani_gdb_s07.h5'

# Define cross validation networks
wkdircv = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb01-06_red03-06/cv4/'
cnstfilecv = wkdircv + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefilecv  = wkdircv + 'sae_6-31gd.dat'
nnfprefix  = wkdircv + 'train'


# Define the conformer cross validator class
anicv1 = aat.anicrossvalidationconformer(cnstfilecv,saefilecv,nnfprefix,5,0,False)

# Define cross validation networks
wkdircv = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb01-06_red03-07/cv_hyper_search/models/cv_LR_test_32-32-32-1/'
cnstfilecv = wkdircv + 'rHCNO-4.6A_16-4.6A_a4-8.params'
saefilecv  = wkdircv + 'sae_6-31gd.dat'
nnfprefix  = wkdircv + 'train'

# Define the conformer cross validator class
#anicv2 = aat.anicrossvalidationconformer(cnstfilecv,saefilecv,nnfprefix,5,0,False)

# Declare data loader
adl = pyt.anidataloader(h5file)

means = []
sigms = []

# Iterate data set
for i,data in enumerate(adl):
    # Extract the data
    X  = data['coordinates'][0:10]
    S  = data['species']
    energies_t = data['energies'][0:10]

    #minid = np.argmin(energies_t)

    #X  = X[minid].reshape(1,X.shape[1],3)
    #energies_t = np.array(energies_t[minid])

    # Calculate std. dev. per atom for all conformers
    sigma1 = anicv1.compute_stddev_conformations(X,S)
    #sigma2 = anicv2.compute_stddev_conformations(X,S)

    # Calculate energy deltas
    delta, energies_c = anicv1.compute_energy_delta_conformations(X,energies_t,S)

    # Print result
    print('----------Result----------')
    #print(np.mean(delta,axis=0))
    print(sigma1)
    #print(sigma2)

    #rmse = np.array(np.sqrt(np.mean(np.power(delta,2.0))) / float(len(S)))
    #means.append(rmse)

    energies_t = hdn.hatokcal * (energies_t - energies_t.min())
    mean = hdn.hatokcal * np.mean(energies_c, axis=0)
    #delta = np.abs(mean - energies_t) / float(len(S))
    sigma = hdn.hatokcal * sigma1
    mdelt = np.abs(np.max(energies_c,axis=0)-np.min(energies_c,axis=0))

    #print(delta)

    #means.append(delta)
    means.append(np.max(np.abs(hdn.hatokcal * delta),axis=0) / float(len(S)))

    sigms.append(sigma)

    idx_hs = np.array(np.where(sigma >= 0.08))
    idx_ls = np.array(np.where(sigma < 0.08))

    idx_gd = np.intersect1d(np.where(delta < 0.08), idx_ls)
    idx_bd = np.intersect1d(np.where(delta >= 0.08), idx_ls)

    '''
    plt.plot(sigma-sigma.mean())
    plt.plot(delta-delta.mean())
    plt.plot(mdelt-mdelt.mean())
    plt.show()

    plt.plot(energies_t / float(len(S)),energies_t / float(len(S)))
    plt.scatter(energies_t[idx_gd] / float(len(S)), mean[idx_gd] / float(len(S)), color='green', label='Hit : ')
    plt.scatter(energies_t[idx_bd] / float(len(S)), mean[idx_bd] / float(len(S)), color='red', label='Hit : ')
    plt.scatter(energies_t[idx_hs] / float(len(S)), mean[idx_hs] / float(len(S)), color='blue', label='Hit : ')
    plt.show()

    plt.plot(energies_t / float(len(S)), energies_t / float(len(S)))
    plt.scatter(energies_t / float(len(S)), hdn.hatokcal * energies_c[0,:] / float(len(S)), color='blue', label='Hit : ')
    plt.scatter(energies_t / float(len(S)), hdn.hatokcal * energies_c[1,:] / float(len(S)), color='red', label='Hit : ')
    plt.scatter(energies_t / float(len(S)), hdn.hatokcal * energies_c[2,:] / float(len(S)), color='green', label='Hit : ')
    plt.scatter(energies_t / float(len(S)), hdn.hatokcal * energies_c[3,:] / float(len(S)), color='yellow', label='Hit : ')
    plt.scatter(energies_t / float(len(S)), hdn.hatokcal * energies_c[4,:] / float(len(S)), color='orange', label='Hit : ')

    plt.show()
    '''

    if i == 500:
        break

means = np.concatenate(means)
sigms = np.concatenate(sigms)

Nt = sigms.size

idx_hs = np.array(np.where(sigms >= 0.02))
idx_ls = np.array(np.where(sigms < 0.02))

print(type(idx_hs))

idx_gd = np.intersect1d(np.where(means <  0.1),idx_ls)
idx_bd = np.intersect1d(np.where(means >= 0.1),idx_ls)

print(type(idx_gd))

plt.scatter(means[idx_hs],sigms[idx_hs],color='blue' ,label='Hit : ' + "{:.1f}".format(100*idx_hs.size/float(Nt)))
plt.scatter(means[idx_gd],sigms[idx_gd],color='green',label='Good: ' + "{:.1f}".format(100*idx_gd.size/float(Nt)))
plt.scatter(means[idx_bd],sigms[idx_bd],color='red'  ,label='Miss: ' + "{:.1f}".format(100*idx_bd.size/float(Nt)))

plt.xlabel('Max Error (kcal/mol/atoms)')
plt.ylabel('Model std. dev. (kcal/mol/atoms)')
plt.legend(bbox_to_anchor=(0.01, 0.9), loc=2, borderaxespad=0.)
plt.show()