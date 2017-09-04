# pyneurochem
import pyNeuroChem as pync
import pyanitools as pyt
import pyaniasetools as aat
import hdnntools as hdn

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# Define test file
#h5file = '/home/jujuman/Research/ForceNMPaper/polypeptide/tripeptide_full.h5'
h5file = '/home/jujuman/Research/extensibility_test_sets/drugbank/drugbank_testset.h5'
#h5file = '/home/jujuman/Research/extensibility_test_sets/gdb-10/gdb11_10_test500.h5'
#h5file = '/home/jujuman/Research/extensibility_test_sets/gdb-09/gdb11_09_test500.h5'
#h5file = '/home/jujuman/Research/extensibility_test_sets/gdb-08/gdb11_08_test500.h5'
#h5file = '/home/jujuman/Research/extensibility_test_sets/gdb-07/gdb11_07_test500.h5'
#h5file = '/home/jujuman/Research/GDB_Dimer/dimer_gen_test/dimers_test.h5'
#h5file = '/home/jujuman/Research/ForceTrainTesting/train3/cache-data-0/testset/testset.h5'
#h5file = '/home/jujuman/Research/IR_MD/methanol/methanol_traj_rsub.h5'

# Define cross validation networks
#wkdircv = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb_r06_comb08/cv6/'
#wkdircv = '/home/jujuman/Scratch/Research/DataReductionMethods/model6r/model-gdb06r/org_cv/cv/'
#wkdircv = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb_r06_comb08_2/cv4/'
wkdircv = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb_r06_comb08_3/cv1/'
#wkdircv = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb06r/org_cv/cv/'
#wkdircv = '/home/jujuman/Research/ForceTrainTesting/train_full_al1/'
#wkdircv = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb01-06_red03-06/cv4/'
#wkdircv = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb01-06_red03-08_mdal01/cv2/'
#wkdircv = '/home/jujuman/Gits/ANI-Networks/networks/ANI-c08f-ntwk-cv/'
#wkdircv = '/home/jujuman/Scratch/Research/DataReductionMethods/model6r/model-gdb06r/org_cv/cv/'
#wkdircv = '/home/jujuman/Research/ForceTrainTesting/train_e_comp/'
#wkdircv = '/home/jujuman/Research/ForceTrainTesting/train/'
cnstfilecv = wkdircv + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefilecv  = wkdircv + 'sae_6-31gd.dat'
nnfprefix  = wkdircv + 'train'

# Number of cv networks
Ncv = 5

# Confidence list
clist = [0.03,0.05,0.08,0.12,0.2,0.4,0.6]

# Energy list
elist = [0.01, 10.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0]

# Define the conformer cross validator class
anicv = aat.anicrossvalidationconformer(cnstfilecv,saefilecv,nnfprefix,Ncv,0,False)

# Declare data loader
adl = pyt.anidataloader(h5file)

# Declare energy data storage dict
Cdat = dict({'Sigm' : [],
             'Natm' : [],
             'Eani' : [],
             'Edft' : [],
             'Emin' : [],
             'dEani': [],
             'dEdft': [],
             'EMAE' : [],
             'ERMSE': [],
             'dEMAE' : [],
             'dERMSE': [],
             'Fani' : [],
             'Fdft' : [],
             'FMAE' : [],
             'FRMSE': [],})

Emax = [0.0,0.0,0.0]
Fmax = [0.0,0.0,0.0]

Ferr = []
# Iterate data set
for i,data in enumerate(adl):
    #if (i==10):
    #    break

    # Extract the data
    print(data['path'])
    X  = np.ndarray.astype(data['coordinates'], dtype=np.float32)
    S  = data['species']
    Edft = data['energies']
    Fdft =  -data['forces']#/(0.52917724900001*0.52917724900001)
    path = data['path']

    # Calculate std. dev. per atom for all conformers
    sigma = anicv.compute_stddev_conformations(X,S)

    # Calculate energy deltas
    Eani, Fani = anicv.compute_energy_conformations(X,S)

    # Convert to kcal/mol and reshape if needed
    #Eani = hdn.hatokcal * Eani
    Edft = hdn.hatokcal * Edft

    #print(Edft-Eani)

    #Fani = hdn.hatokcal * Fani#.reshape(Ncv, -1)
    Fdft = hdn.hatokcal * Fdft#.reshape(-1)

    idx = np.asarray(np.where(sigma < 0.08))[0]
    #print(idx,Fani[0].shape,Fdft.shape)
    Ferr.append((Fani[0][idx]-Fdft[idx]).flatten())

    # Calculate full dE
    dEani = hdn.calculateKdmat(Ncv, Eani)
    dEdft = hdn.calculatedmat(Edft)

    # Calculate per molecule errors
    FMAE  = hdn.calculatemeanabserror(Fani.reshape(Ncv, -1), Fdft.reshape(-1), axis=1)
    FRMSE = hdn.calculaterootmeansqrerror(Fani.reshape(Ncv, -1), Fdft.reshape(-1), axis=1)

    #plt.hist((Fani-Fdft).flatten(),bins=100)
    # plt.show()

    '''
    if Emax[0] < np.abs((Eani-Edft)).max():
        ind = np.argmax(np.abs((Eani-Edft)).flatten())
        Emax[0] = (Eani-Edft).flatten()[ind]
        Emax[1] = Eani.flatten()[ind]
        Emax[2] = Edft.flatten()[ind]

    if Fmax[0] < np.abs((Fani-Fdft)).max():
        ind = np.argmax(np.abs((Fani-Fdft)).flatten())
        Fmax[0] = (Fani-Fdft).flatten()[ind]
        Fmax[1] = Fani.flatten()[ind]
        Fmax[2] = Fdft.flatten()[ind]
    '''

    #if FRMSE.mean() > 45.0:
    #    print("!!!!!!!!!!!!!!!!!!!!!FRMSE: ", FRMSE)
    #    print(Fani-Fdft)
    #    exit(0)

    # Calculate per molecule errors
    EMAE  = hdn.calculatemeanabserror (Eani,Edft,axis=1)
    ERMSE = hdn.calculaterootmeansqrerror(Eani,Edft,axis=1)

    # Calculate per molecule errors
    dEMAE  = hdn.calculatemeanabserror (dEani,dEdft,axis=1)
    dERMSE = hdn.calculaterootmeansqrerror(dEani,dEdft,axis=1)

    # Get min energy information
    index_min = np.argmin(Edft)
    #print(Eani[:,index_min])
    Emin = np.array([Eani[:,index_min].mean(), Edft[index_min]])

    # Store data for later
    Cdat['Sigm'].append(sigma)
    Cdat['Natm'].append(len(S))
    Cdat['Eani'].append(Eani)
    Cdat['Edft'].append(Edft)
    Cdat['Emin'].append(Emin)
    Cdat['dEani'].append(dEani)
    Cdat['dEdft'].append(dEdft)
    Cdat['EMAE'].append(EMAE)
    Cdat['ERMSE'].append(ERMSE)
    Cdat['dEMAE'].append(dEMAE)
    Cdat['dERMSE'].append(dERMSE)
    Cdat['Fani'].append(Fani)
    Cdat['Fdft'].append(Fdft)
    Cdat['FMAE'].append(FMAE)
    Cdat['FRMSE'].append(FRMSE)

    # Print per molecule information
    np.set_printoptions(precision=2)
    print(i,'):',path,'Atoms:',str(len(S)))
    print('   -EMAE:  ',   EMAE, ':', "{:.2f}".format(EMAE.mean()))
    print('   -ERMSE: ',  ERMSE, ':', "{:.2f}".format(ERMSE.mean()))
    print('   -dEMAE: ', dEMAE , ':', "{:.2f}".format(dEMAE.mean()))
    print('   -dERMSE:', dERMSE, ':', "{:.2f}".format(dERMSE.mean()))
    print('   -FMAE:  ',   FMAE, ':', "{:.2f}".format(FMAE.mean()))
    print('   -FRMSE: ',  FRMSE, ':', "{:.2f}".format(FRMSE.mean()))

#print("\n  MAX ENERGY DELTA:",Emax)
#"  MAX FORCE DELTA:",Fmax)

Ferr = np.concatenate(Ferr)
#plt.hist(Ferr, bins=250)
#plt.show()

dfe = pd.DataFrame()
print('\nPrinting stats...')
for id, e in enumerate(elist):
    #print('   Building data for energy range:',e)

    Fani, Fdft, Nd, Nt = aat.getenergyconformerdata(Ncv, Cdat['Fani'], Cdat['Fdft'], Cdat['Edft'], e)
    Eani, Edft, Nd, Nt = aat.getenergyconformerdata(Ncv, Cdat['Eani'], Cdat['Edft'], Cdat['Edft'], e)

    Fmt = hdn.calculatemeanabserror(Fani, Fdft,axis=1)
    Frt = hdn.calculaterootmeansqrerror(Fani, Fdft,axis=1)

    Emt = hdn.calculatemeanabserror(Eani, Edft,axis=1)
    Ert = hdn.calculaterootmeansqrerror(Eani, Edft,axis=1)

    #dff = pd.DataFrame(data=Frt.reshape(1,-1), index=[e], columns=['Frmse1','Frmse2', 'Frmse3','Frmse4','Frmse5',])
    dff = pd.DataFrame(index=[e])
    dff = dff.assign(Nd=pd.Series([Nd]).values)
    dff = dff.assign(Nt=pd.Series([Nt]).values)
    dff = dff.assign(EMAEm=pd.Series([np.mean(Emt)]).values)
    dff = dff.assign(EMAEs=pd.Series([Emt.std()]).values)
    dff = dff.assign(ERMSm=pd.Series([Ert.mean()]).values)
    dff = dff.assign(ERMSs=pd.Series([Ert.std()]).values)
    dff = dff.assign(FMAEm=pd.Series([Fmt.mean()]).values)
    dff = dff.assign(FMAEs=pd.Series([Fmt.std()]).values)
    dff = dff.assign(FRMSm=pd.Series([Frt.mean()]).values)
    dff = dff.assign(FRMSs=pd.Series([Frt.std()]).values)
    dfe = dfe.append(dff)

print('Energy level performance: ')
print(dfe)

# ----------------------------------
# Plot force histogram
# ----------------------------------
def plot_corr_dist(Xa, Xp, inset=True, figsize=[13,10]):
    Fmx = Xa.max()
    Fmn = Xa.min()

    label_size = 14
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size

    fig, ax = plt.subplots(figsize=figsize)

    # Plot ground truth line
    ax.plot([Fmn, Fmx], [Fmn, Fmx], '--', c='r', linewidth=3)

    # Set labels
    ax.set_xlabel('$F_{dft}$' + r' $(kcal \times mol^{-1} \times \AA^{-1})$', fontsize=22)
    ax.set_ylabel('$F_{ani}$' + r' $(kcal \times mol^{-1} \times \AA^{-1})$', fontsize=22)

    cmap = mpl.cm.viridis

    # Plot 2d Histogram
    bins = ax.hist2d(Xa, Xp, bins=200, norm=LogNorm(), range= [[Fmn, Fmx], [Fmn, Fmx]], cmap=cmap)

    # Build color bar
    #cbaxes = fig.add_axes([0.91, 0.1, 0.03, 0.8])
    cb1 = fig.colorbar(bins[-1], cmap=cmap)
    cb1.set_label('Log(count)', fontsize=16)

    # Annotate with errors
    PMAE = hdn.calculatemeanabserror(Xa, Xp)
    PRMS = hdn.calculaterootmeansqrerror(Xa, Xp)
    ax.text(0.75*((Fmx-Fmn))+Fmn, 0.43*((Fmx-Fmn))+Fmn, 'MAE='+"{:.1f}".format(PMAE)+'\nRMSE='+"{:.1f}".format(PRMS), fontsize=20,
            bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})

    if inset:
        axins = zoomed_inset_axes(ax, 2.2, loc=2) # zoom = 6

        sz = 6
        axins.hist2d(Xa, Xp,bins=50, range=[[Fmn/sz, Fmx/sz], [Fmn/sz, Fmx/sz]], norm=LogNorm(), cmap=cmap)
        axins.plot([Xa.min(), Xa.max()], [Xa.min(), Xa.max()], '--', c='r', linewidth=3)

        # sub region of the original image
        x1, x2, y1, y2 = Fmn/sz, Fmx/sz, Fmn/sz, Fmx/sz
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.yaxis.tick_right()

        plt.xticks(visible=True)
        plt.yticks(visible=True)

        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

        Ferr = Xa - Xp
        std = np.std(Ferr)
        men = np.mean(Ferr)
        axh = plt.axes([.49, .14, .235, .235])
        axh.hist(Ferr, bins=75, range=[men-4*std, men+4*std], normed=True)
        axh.set_title('Difference distribution')

    #plt.draw()
    plt.show()

Fani, Fdft, Nd, Nt = aat.getcvconformerdata(Ncv, Cdat['Fani'], Cdat['Fdft'], Cdat['Sigm'], 300.0)
plot_corr_dist(Fdft, np.mean(Fani, axis=0), True)

exit(0)
# ----------------------------------

dfc = pd.DataFrame()
for c in clist:
    #print('   Building data for lower sigma:',c)

    Fani, Fdft, Nd, Nt = aat.getcvconformerdata(Ncv, Cdat['Fani'], Cdat['Fdft'], Cdat['Sigm'], c)
    Eani, Edft, Nd, Nt = aat.getcvconformerdata(Ncv, Cdat['Eani'], Cdat['Edft'], Cdat['Sigm'], c)

    Fmt = hdn.calculatemeanabserror(Fani, Fdft, axis=1)
    Frt = hdn.calculaterootmeansqrerror(Fani, Fdft, axis=1)

    Emt = hdn.calculatemeanabserror(Eani, Edft,axis=1)
    Ert = hdn.calculaterootmeansqrerror(Eani, Edft,axis=1)

    #dff = pd.DataFrame(data=Frt.reshape(1,-1), index=[c], columns=['Frmse1','Frmse2', 'Frmse3','Frmse4','Frmse5'])
    dff = pd.DataFrame(index=[c])
    dff = dff.assign(Nd=pd.Series([Nd]).values)
    dff = dff.assign(Nt=pd.Series([Nt]).values)
    dff = dff.assign(EMAEm=pd.Series([np.mean(Emt)]).values)
    dff = dff.assign(EMAEs=pd.Series([Emt.std()]).values)
    dff = dff.assign(ERMSm=pd.Series([Ert.mean()]).values)
    dff = dff.assign(ERMSs=pd.Series([Ert.std()]).values)
    dff = dff.assign(FMAEm=pd.Series([Fmt.mean()]).values)
    dff = dff.assign(FMAEs=pd.Series([Fmt.std()]).values)
    dff = dff.assign(FRMSm=pd.Series([Frt.mean()]).values)
    dff = dff.assign(FRMSs=pd.Series([Frt.std()]).values)
    dfc = dfc.append(dff)

    #print('   -F  MAE:', Fmt, Fmt.mean(), Fmt.std())
    #print('   -F RMSE:', Frt, Frt.mean(), Frt.std())
print('Confidence level performance: ')
print(dfc)

# with errorbars: clip non-positive values
ax = plt.subplot(221)
ax.set_title('Force error vs Max CV Std. Dev')
#ax.set_xscale("log", nonposx='clip')
#ax.set_yscale("log", nonposy='clip')
ax.set_xlabel('Energy confidence (Max Std. Dev. kcal/mol)')
ax.set_ylabel('Force error (kcal/mol/$\AA$)')

plt.errorbar(clist, dfc['FMAEm'], yerr=dfc['FMAEs'], fmt='--o',label='MAE' )
plt.errorbar(clist, dfc['FRMSm'], yerr=dfc['FRMSs'], fmt='--o',label='RMSE')
plt.legend(loc=4)

ax = plt.subplot(222)
ax.set_title('Energy error vs Max CV Std. Dev')
#ax.set_xscale("log", nonposx='clip')
#ax.set_yscale("log", nonposy='clip')
ax.set_xlabel('Energy confidence (Max Std. Dev. kcal/mol)')
ax.set_ylabel('Energy error (kcal/mol)')

plt.errorbar(clist, dfc['EMAEm'], yerr=dfc['EMAEs'], fmt='--o',label='MAE' )
plt.errorbar(clist, dfc['ERMSm'], yerr=dfc['ERMSs'], fmt='--o',label='RMSE')
plt.legend(loc=4)

ax = plt.subplot(223)
ax.set_title('Force error vs Max energy from min')
#ax.set_xscale("log", nonposx='clip')
#ax.set_yscale("log", nonposy='clip')
ax.set_xlabel('Max energy from min (kcal/mol)')
ax.set_ylabel('Force error (kcal/mol/$\AA$)')

plt.errorbar(elist, dfe['FMAEm'], yerr=dfe['FMAEs'], fmt='--o',label='MAE' )
plt.errorbar(elist, dfe['FRMSm'], yerr=dfe['FRMSs'], fmt='--o',label='RMSE')
plt.legend(loc=4)

ax = plt.subplot(224)
ax.set_title('Energy error vs Max energy from min')
#ax.set_xscale("log", nonposx='clip')
#ax.set_yscale("log", nonposy='clip')
ax.set_xlabel('Max energy from min (kcal/mol)')
ax.set_ylabel('Energy error (kcal/mol)')

plt.errorbar(elist, dfe['EMAEm'], yerr=dfe['EMAEs'], fmt='--o',label='MAE' )
plt.errorbar(elist, dfe['ERMSm'], yerr=dfe['ERMSs'], fmt='--o',label='RMSE')
plt.legend(loc=4)
plt.show()

#plt.errorbar(clist, np.array(dfc['Frmean']), yerr=np.array(dfc['Frstdd']), fmt='-o')
#plt.errorbar(clist, np.array(dfc['Fmmean']), yerr=np.array(dfc['Fmstdd']), fmt='-o')
#plt.show()

#plt.errorbar(clist, np.array(dfc['ERMSm']), yerr=np.array(dfc['ERMSs']), fmt='-o')
#plt.errorbar(clist, np.array(dfc['EMAEm']), yerr=np.array(dfc['EMAEs']), fmt='-o')
#plt.show()

# Convert arrays
Cdat['Sigm']   = np.concatenate(Cdat['Sigm'])
Cdat['Natm']   = np.array(Cdat['Natm'], dtype=np.int)
Cdat['Eani']   = np.hstack(Cdat['Eani'])
Cdat['Edft']   = np.concatenate(Cdat['Edft'])
Cdat['Emin']   = np.vstack(Cdat['Emin'])
Cdat['dEani']  = np.hstack(Cdat['dEani'])
Cdat['dEdft']  = np.concatenate(Cdat['dEdft'])
Cdat['EMAE']   = np.vstack(Cdat['EMAE'])
Cdat['ERMSE']  = np.vstack(Cdat['ERMSE'])
Cdat['dEMAE']  = np.vstack(Cdat['dEMAE'])
Cdat['dERMSE'] = np.vstack(Cdat['dERMSE'])
#Cdat['Fani']   = np.hstack(Cdat['Fani'].reshape(Ncv,-1))
#Cdat['Fdft']   = np.concatenate(Cdat['Fdft'].reshape(-1))
Cdat['FMAE']   = np.vstack(Cdat['FMAE'])
Cdat['FRMSE']  = np.vstack(Cdat['FRMSE'])

#print(Cdat['Eani'][:, bidx].shape,Cdat['Sigm'].shape, bidx.shape)

Emt = hdn.calculatemeanabserror(Cdat['Eani'], Cdat['Edft'],axis=1)
Ert = hdn.calculaterootmeansqrerror(Cdat['Eani'], Cdat['Edft'],axis=1)

print('\n')
print('E  MAE:', Emt, Emt.mean(), Emt.std())
print('E RMSE:', Ert, Ert.mean(), Ert.std())

#print(Cdat['Fani'].shape)
#Fmt = hdn.calculatemeanabserror(Cdat['Fani'],Cdat['Fdft'],axis=1)
#Frt = hdn.calculaterootmeansqrerror(Cdat['Fani'],Cdat['Fdft'],axis=1)

#print('   -', c,'F  MAE:', Fmt, Fmt.mean(), Fmt.std())
#print('   -', c,'F RMSE:', Frt, Frt.mean(), Frt.std())


dEmt = hdn.calculatemeanabserror(Cdat['dEani'],Cdat['dEdft'],axis=1)
dErt = hdn.calculaterootmeansqrerror(Cdat['dEani'],Cdat['dEdft'],axis=1)

print('dE  MAE:', dEmt, dEmt.mean(), dEmt.std())
print('dE RMSE:', dErt, dErt.mean(), dErt.std())

Emtpa = np.mean(Cdat['EMAE'], axis=1)/Cdat['Natm']
Ertpa = np.mean(Cdat['ERMSE'], axis=1)/Cdat['Natm']

print('A. E  MAE/atom:', np.mean(Emtpa))
print('A. E RMSE/atom:', np.mean(Ertpa))

dEmtpa = np.mean(Cdat['dEMAE'], axis=1)/Cdat['Natm']
dErtpa = np.mean(Cdat['dERMSE'], axis=1)/Cdat['Natm']

print('A. dE  MAE/atom:', np.mean(dEmtpa))
print('A. dE RMSE/atom:', np.mean(dErtpa))

Emm = hdn.calculatemeanabserror(Cdat['Emin'][:, 0], Cdat['Emin'][:, 1])
Erm = hdn.calculaterootmeansqrerror(Cdat['Emin'][:, 0], Cdat['Emin'][:, 1])

print('Emin  MAE:', Emm)
print('Emin RMSE:', Erm)

dEa = hdn.calculatedmat(Cdat['Emin'][:, 0])
dEd = hdn.calculatedmat(Cdat['Emin'][:, 1])

dEtMAE  = hdn.calculatemeanabserror(dEa, dEd)
dEtRMSE = hdn.calculaterootmeansqrerror(dEa, dEd)

print('C Emin  MAE:', dEtMAE)
print('C Emin RMSE:', dEtRMSE)

# Calculate complete distance matricies
#print('Calculating distance matrices...')
#dEa = hdn.calculateKdmat(Ncv, Cdat['Eani'])
#dEd = hdn.calculatedmat(Cdat['Edft'])

#dEtMAE  = hdn.calculatemeanabserror(dEa, dEd, axis=1)
#dEtRMSE = hdn.calculaterootmeansqrerror(dEa, dEd, axis=1)

#print('Complete dE MAE :',  dEtMAE,  dEtMAE.mean())
#print('Complete dE RMSE:', dEtRMSE, dEtRMSE.mean())

print('--------INLINE DATA--------')
print(Emt.mean())
print(Ert.mean())
print(np.mean(Emtpa))
print(np.mean(Ertpa))
print(dEmt.mean())
print(dErt.mean())
print(np.mean(dEmtpa))
print(np.mean(dErtpa))
print(Emm)
print(Erm)
print(dEtMAE)
print(dEtRMSE)
print(Fmt.mean())
print(Frt.mean())
print('---------------------------')

#print(Cdat['Sigm'].shape,.shape)
#plt.scatter(Cdat['Sigm'], dErtpa, s=1)

#plt.xlabel('# of atoms')
#plt.ylabel('RMSE (kcal/mol/atoms)')
#plt.legend(bbox_to_anchor=(0.01, 0.9), loc=2, borderaxespad=0.)
#plt.show()
#plt.savefig('foo.png', bbox_inches='tight')

plt.scatter(Cdat['Natm'], dErtpa, s=1)
plt.xlabel('# of atoms')
plt.ylabel('RMSE (kcal/mol/atoms)')
#plt.legend(bbox_to_anchor=(0.01, 0.9), loc=2, borderaxespad=0.)
plt.show()
