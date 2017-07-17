# pyneurochem
import pyNeuroChem as pync
import pyanitools as pyt
import pyaniasetools as aat
import hdnntools as hdn

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

# Define test file
#h5file = '/home/jujuman/Research/ForceNMPaper/polypeptide/tripeptide_full.h5'
#h5file = '/home/jujuman/Scratch/Research/extensibility_test_sets/drugbank/drugbank_testset.h5'
#h5file = '/home/jujuman/Scratch/Research/extensibility_test_sets/gdb-10/gdb11_10_test500.h5'
#h5file = '/home/jujuman/Scratch/Research/extensibility_test_sets/gdb-09/gdb11_09_test500.h5'
#h5file = '/home/jujuman/Scratch/Research/extensibility_test_sets/gdb-08/gdb11_08_test500.h5'
h5file = '/home/jujuman/Scratch/Research/extensibility_test_sets/gdb-07/gdb11_07_test500.h5'

# Define cross validation networks
wkdircv = '/home/jujuman/Research/DataReductionMethods/model6r/model-gdb01-06_red03-08_mdal01/cv2/'
#wkdircv = '/home/jujuman/Gits/ANI-Networks/networks/ANI-c08f-ntwk-cv/'
#wkdircv = '/home/jujuman/Scratch/Research/DataReductionMethods/model6r/model-gdb06r/org_cv/cv/'
cnstfilecv = wkdircv + 'rHCNO-4.6A_16-3.1A_a4-8.params'
saefilecv  = wkdircv + 'sae_6-31gd.dat'
nnfprefix  = wkdircv + 'train'

# Number of cv networks
Ncv = 5

# Confidence list
clist = [0.03,0.05,0.08,0.12,0.2,0.4,0.6]

# Energy list
elist = [10.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0]

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

# Iterate data set
for i,data in enumerate(adl):
    #if (i==10):
    #    break
    # Extract the data
    X  = data['coordinates']
    S  = data['species']
    Edft = data['energies']
    Fdft = data['forces']#/(0.52917724900001*0.52917724900001)
    path = data['path']

    # Calculate std. dev. per atom for all conformers
    sigma = anicv.compute_stddev_conformations(X,S)

    # Calculate energy deltas
    Eani, Fani = anicv.compute_energy_conformations(X,S)

    # Convert to kcal/mol and reshape if needed
    Eani = hdn.hatokcal * Eani
    Edft = hdn.hatokcal * Edft

    Fani = hdn.hatokcal * Fani#.reshape(Ncv, -1)
    Fdft = hdn.hatokcal * Fdft#.reshape(-1)

    # Calculate full dE
    dEani = hdn.calculateKdmat(Ncv, Eani)
    dEdft = hdn.calculatedmat(Edft)

    # Calculate per molecule errors
    FMAE  = hdn.calculatemeanabserror(Fani.reshape(Ncv, -1), Fdft.reshape(-1), axis=1)
    FRMSE = hdn.calculaterootmeansqrerror(Fani.reshape(Ncv, -1), Fdft.reshape(-1), axis=1)

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
#ax.set_xscale("log", nonposx='clip')
#ax.set_yscale("log", nonposy='clip')

plt.errorbar(clist, dfc['FMAEm'], yerr=dfc['FMAEs'], fmt='--o')
plt.errorbar(clist, dfc['FRMSm'], yerr=dfc['FRMSs'], fmt='--o')

ax = plt.subplot(222)
#ax.set_xscale("log", nonposx='clip')
#ax.set_yscale("log", nonposy='clip')

plt.errorbar(clist, dfc['EMAEm'], yerr=dfc['EMAEs'], fmt='--o')
plt.errorbar(clist, dfc['ERMSm'], yerr=dfc['ERMSs'], fmt='--o')

ax = plt.subplot(223)
#ax.set_xscale("log", nonposx='clip')
#ax.set_yscale("log", nonposy='clip')

plt.errorbar(elist, dfe['FMAEm'], yerr=dfe['FMAEs'], fmt='--o')
plt.errorbar(elist, dfe['FRMSm'], yerr=dfe['FRMSs'], fmt='--o')

ax = plt.subplot(224)
#ax.set_xscale("log", nonposx='clip')
#ax.set_yscale("log", nonposy='clip')

plt.errorbar(elist, dfe['EMAEm'], yerr=dfe['EMAEs'], fmt='--o')
plt.errorbar(elist, dfe['ERMSm'], yerr=dfe['ERMSs'], fmt='--o')

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

print('E  MAE:', dEmt, dEmt.mean(), dEmt.std())
print('E RMSE:', dErt, dErt.mean(), dErt.std())

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


plt.scatter(Cdat['Natm'], dErtpa, s=1)

plt.xlabel('# of atoms')
plt.ylabel('RMSE (kcal/mol/atoms)')
#plt.legend(bbox_to_anchor=(0.01, 0.9), loc=2, borderaxespad=0.)
plt.show()
#plt.savefig('foo.png', bbox_inches='tight')