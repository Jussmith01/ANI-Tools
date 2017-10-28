import pyaniasetools as aat
import pyanitools as pyt
import hdnntools as hdn
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# ----------------------------------
# Plot force histogram
# ----------------------------------
def plot_corr_dist(Xa, Xp, inset=True, figsize=[13,10]):
    Fmx1 = Xa.max()
    Fmn1 = 0.0

    Fmx2 = Xp.max()
    Fmn2 = 0.0

    label_size = 14
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size

    fig, ax = plt.subplots(figsize=figsize)

    # Plot ground truth line
    #ax.plot([Fmn1, Fmx1], [Fmn2, Fmx2], '--', c='r', linewidth=3)

    # Set labels
    ax.set_ylabel('$dE$', fontsize=22)
    ax.set_xlabel('$\sigma / N$', fontsize=22)

    cmap = mpl.cm.viridis

    # Plot 2d Histogram
    #bins = ax.hist2d(np.log10(Xa), np.log10(Xp), bins=300, norm=LogNorm(), cmap=cmap)
    bins = ax.hist2d(Xa, Xp, bins=300, norm=LogNorm(), cmap=cmap)

    # Build color bar
    #cbaxes = fig.add_axes([0.91, 0.1, 0.03, 0.8])
    cb1 = fig.colorbar(bins[-1], cmap=cmap)
    cb1.set_label('Log(count)', fontsize=16)

    # Annotate with errors
    #PMAE = hdn.calculatemeanabserror(Xa, Xp)
    #PRMS = hdn.calculaterootmeansqrerror(Xa, Xp)
    #ax.text(0.75*((Fmx1-Fmn1))+Fmn1, 0.43*((Fmx2-Fmn2))+Fmn2, 'MAE='+"{:.1f}".format(PMAE)+'\nRMSE='+"{:.1f}".format(PRMS), fontsize=20,
    #        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})

    #plt.draw()
    plt.show()

#h5file = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb_r06_comb08_3/cv3/cache-data-0/testset/testset.h5'
h5file = ['/home/jujuman/Research/extensibility_test_sets/gdb-07/gdb11_07_test500.h5',
          '/home/jujuman/Research/extensibility_test_sets/gdb-08/gdb11_08_test500.h5',
          '/home/jujuman/Research/extensibility_test_sets/gdb-09/gdb11_09_test500.h5',
          '/home/jujuman/Research/extensibility_test_sets/gdb-10/gdb11_10_test500.h5',
          '/home/jujuman/Research/ForceNMPaper/polypeptide/tripeptide_full.h5',
          ]

wkdircv = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb_r06_comb08_3/cv3/'
cnstfilecv = wkdircv + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefilecv  = wkdircv + 'sae_6-31gd.dat'
nnfprefix  = wkdircv + 'train'

# Number of cv networks
Ncv = 5

# Define the conformer cross validator class
anicv = aat.anicrossvalidationconformer(cnstfilecv,saefilecv,nnfprefix,Ncv,0,False)

# ddict
ddict = dict({'de1': [],
              'de2': [],
              'df1': [],
              'fs1': [],
              'sn1': [],
              'sn2': [],
              'na': []})

for fn,h5 in enumerate(h5file):
    print('working on file',fn,'...')
    # Declare data loader
    adl = pyt.anidataloader(h5)

    for i,data in enumerate(adl):
        #if i == 10:
        #    break

        X  = np.ndarray.astype(data['coordinates'], dtype=np.float32)
        S  = data['species']
        Edft = data['energies']
        Fdft =  -data['forces']#/(0.52917724900001*0.52917724900001)
        path = data['path']

        # Calculate std. dev. per atom for all conformers
        sigma1, sigma2 = anicv.compute_stddev_conformations(X,S)

        # Calculate energy deltas
        Eani, Fani = anicv.compute_energy_conformations(X,S)
        #Eani = np.mean(Eani,axis=0)
        Fstd = np.std(Fani, axis=0)
        Fani = np.mean(Fani,axis=0)

        Edft = hdn.hatokcal * Edft
        Fdft = hdn.hatokcal * Fdft

        Na = np.full(Eani.size,len(S),dtype=np.int64)
        ddict['na'].append(Na)

        df = np.mean(np.abs(Fani-Fdft).reshape(Fani.shape[0], -1), axis=1)
        sf = np.max(Fstd.reshape(Fani.shape[0], -1), axis=1)

        ddict['sn1'].append(sigma1)
        ddict['sn2'].append(sigma2)
        ddict['df1'].append(df)
        ddict['fs1'].append(sf)
        ddict['de1'].append(np.max(np.abs(Eani-Edft), axis=0)/np.sqrt(float(len(S))))
        ddict['de2'].append(np.max(np.abs(Eani-Edft), axis=0)/float(len(S)))

    adl.cleanup()

ddict['na'] = np.concatenate(ddict['na'])
ddict['sn1'] = np.concatenate(ddict['sn1'])
ddict['sn2'] = np.concatenate(ddict['sn2'])
ddict['de1'] = np.concatenate(ddict['de1'])
ddict['de2'] = np.concatenate(ddict['de2'])
ddict['df1'] = np.concatenate(ddict['df1'])
ddict['fs1'] = np.concatenate(ddict['fs1'])

demax = 1.0
dsig = ddict['sn1'].max()
dx = dsig/1000.0

term = False
i=0
while not term:
    ds = ddict['sn1'].max() - i*dx

    Nds1 = np.where(ddict['sn1'] > ds)[0].size

    idx = np.where(ddict['sn1'] < ds)
    i += 1
    print(ds, ddict['de1'][idx].max(),Nds1, 'of', ddict['de1'].size)
    if ddict['de1'][idx].max() < demax:
        term = True

Nds1 = np.where(ddict['sn1'] > ds)[0].size
print(Nds1,'of',ddict['de1'].size)

plot_corr_dist(ddict['sn1'], ddict['de1'], False)
#plot_corr_dist(ddict['de1'], ddict['fs1'], False)
#plot_corr_dist(ddict['de'], ddict['sn2'], False)
