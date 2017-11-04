import pyaniasetools as aat
import pyanitools as ant
import hdnntools as hdt
import pandas as pd

import sys

import numpy as np

import re
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

#import seaborn as sns
pd.options.display.float_format = '{:.2f}'.format

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
    cb1.set_label('Count', fontsize=16)

    # Annotate with errors
    PMAE = hdt.calculatemeanabserror(Xa, Xp)
    PRMS = hdt.calculaterootmeansqrerror(Xa, Xp)
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
        axh = plt.axes([.49, .16, .235, .235])
        axh.hist(Ferr, bins=75, range=[men-4*std, men+4*std], normed=True)
        axh.set_title('Difference distribution')

    #plt.draw()
    plt.show()

class generate_ensemble_data(aat.anicrossvalidationconformer):

    '''Constructor'''
    def __init__(self, networks, tsfiles, gpu=0):
        super().__init__(networks['cns'], networks['sae'], networks['nnf'], networks['nts'], gpu )
        self.tsfiles = tsfiles
        self.Nn = networks['nts']

    '''Stat generator'''
    def generate_stats(self, maxe = sys.float_info.max):
        self.tdata = dict()
        for key in self.tsfiles.keys():
            print('   -Working on',key,'...')

            cdata = dict({'Eani': [],
                          'Edft': [],
                          'Fani': [],
                          'Fdft': [],
                          'dEani': [],
                          'dEdft': [],
                          'Na': [],})

            for file in self.tsfiles[key]:
                adl = ant.anidataloader(file)
                for i, data in enumerate(adl):
                    #if i > 5:
                    #    break

                    Eani, Fani = self.compute_energy_conformations(data['coordinates'], data['species'])

                    midx = np.where( data['energies'] - data['energies'].min() < maxe/hdt.hatokcal )[0]

                    Eani = Eani[:,midx]
                    Edft = data['energies'][midx]
                    Fani = -Fani[:,midx,:,:]
                    Fdft = data['forces'][midx]

                    #Eestd = np.std(Eani, axis=0)/np.sqrt(len(data['species']))
                    Eeani = np.mean(Eani, axis=0).reshape(1,-1)
                    Feani = np.mean(Fani, axis=0).flatten().reshape(1,-1)

                    Fani = Fani.reshape(Fani.shape[0],-1)

                    Eani = np.vstack([Eani, Eeani])
                    Fani = np.vstack([Fani, Feani])

                    Edft = hdt.hatokcal * Edft
                    Fdft = hdt.hatokcal * Fdft.flatten()

                    cdata['Na'].append(np.full(Eani.size, len(data['species']), dtype=np.int32))

                    cdata['Eani'].append(Eani)
                    cdata['Edft'].append(Edft)

                    cdata['Fani'].append(Fani)
                    cdata['Fdft'].append(Fdft)

                    #cdata['Frmse'].append(np.sqrt(np.mean((Fani-Fdft).reshape(Fdft.shape[0], -1)**2, axis=1)))
                    #cdata['Frmae'].append(np.sqrt(np.mean(np.abs((Fani - Fdft).reshape(Fdft.shape[0], -1)), axis=1)))

                    cdata['dEani'].append(hdt.calculateKdmat(self.Nn+1, Eani))
                    cdata['dEdft'].append(hdt.calculatedmat(Edft))

                    #cdata['Erani'].append(Eani-Eani.min())
                    #cdata['Erdft'].append(Edft-Edft.min())

            for k in ['Na', 'Edft', 'Fdft', 'dEdft']:
                cdata[k] = np.concatenate(cdata[k])

            for k in ['Eani', 'Fani', 'dEani']:
                cdata[k] = np.hstack(cdata[k])

            self.tdata.update({key: cdata})

    ''' Generate total errors '''
    def store_data(self, filename):
        if os.path.exists(filename):
            os.remove(filename)

        dpack = ant.datapacker(filename)
        for k in self.tdata.keys():
            dpack.store_data(k,**(self.tdata[k]))
        dpack.cleanup()

class evaluate_ensemble_data(aat.anicrossvalidationconformer):

    '''Constructor'''
    def __init__(self, datafile):
        self.fdata = dict()
        for df in datafile:
            adl = ant.anidataloader(df)
            tdata = dict()
            for data in adl:
                tdata.update({data['path'].split('/')[-1] : data})
            adl.cleanup()
            self.fdata[df.split('tsdata_')[-1].split('.h5')[0]] = tdata

    ''' Generate total errors '''
    def generate_total_errors(self, ntkey, tskey):
        #idx = np.nonzero(self.fdata[ntkey][tskey]['Erdft'])
        Nn = self.fdata[ntkey][tskey]['Eani'].shape[0]-1
        return {'EMAEm': hdt.calculatemeanabserror(self.fdata[ntkey][tskey]['Eani'][Nn,:], self.fdata[ntkey][tskey]['Edft']),
                'EMAEs': np.std(hdt.calculatemeanabserror(self.fdata[ntkey][tskey]['Eani'][0:Nn,:], self.fdata[ntkey][tskey]['Edft'], axis=1)),
                'ERMSm': hdt.calculaterootmeansqrerror(self.fdata[ntkey][tskey]['Eani'][Nn,:], self.fdata[ntkey][tskey]['Edft']),
                'ERMSs': np.std(hdt.calculaterootmeansqrerror(self.fdata[ntkey][tskey]['Eani'][0:Nn,:], self.fdata[ntkey][tskey]['Edft'], axis=1)),
                'dEMAEm': hdt.calculatemeanabserror(self.fdata[ntkey][tskey]['dEani'][Nn,:], self.fdata[ntkey][tskey]['dEdft']),
                'dEMAEs': np.std(hdt.calculatemeanabserror(self.fdata[ntkey][tskey]['dEani'][0:Nn,:], self.fdata[ntkey][tskey]['dEdft'], axis=1)),
                'dERMSm': hdt.calculaterootmeansqrerror(self.fdata[ntkey][tskey]['dEani'][Nn,:], self.fdata[ntkey][tskey]['dEdft']),
                'dERMSs': np.std(hdt.calculaterootmeansqrerror(self.fdata[ntkey][tskey]['dEani'][0:Nn,:], self.fdata[ntkey][tskey]['dEdft'], axis=1)),
                'FMAEm': hdt.calculatemeanabserror(self.fdata[ntkey][tskey]['Fani'][Nn,:], self.fdata[ntkey][tskey]['Fdft']),
                'FMAEs': np.std(hdt.calculatemeanabserror(self.fdata[ntkey][tskey]['Fani'][0:Nn,:], self.fdata[ntkey][tskey]['Fdft'], axis=1)),
                'FRMSm': hdt.calculaterootmeansqrerror(self.fdata[ntkey][tskey]['Fani'][Nn,:], self.fdata[ntkey][tskey]['Fdft']),
                'FRMSs': np.std(hdt.calculaterootmeansqrerror(self.fdata[ntkey][tskey]['Fani'][0:Nn, :],self.fdata[ntkey][tskey]['Fdft'], axis=1)),
                #'dEMAE': hdt.calculatemeanabserror(self.fdata[ntkey][tskey]['dEani'], self.fdata[ntkey][tskey]['dEdft']),
                #'dERMS': hdt.calculaterootmeansqrerror(self.fdata[ntkey][tskey]['dEani'], self.fdata[ntkey][tskey]['dEdft']),
                #'ERMAE': hdt.calculatemeanabserror(self.fdata[ntkey][tskey]['Erani'][idx], self.fdata[ntkey][tskey]['Erdft'][idx]),
                #'ERRMS': hdt.calculaterootmeansqrerror(self.fdata[ntkey][tskey]['Erani'][idx], self.fdata[ntkey][tskey]['rdft'][idx]),
                }

    def get_net_keys(self):
        return self.fdata.keys()

    def get_error_table(self, tskey):
        errors = dict()
        for k in self.fdata.keys():
            errors[k] = pd.Series(self.generate_total_errors(k,tskey))
        pd.set_option('expand_frame_repr', False)
        edat = pd.DataFrame(errors).transpose()
        return edat

    def generate_correlation_plot(self, ntkey, tskey, prop1, prop2, figsize=[13,10]):
        Nn = self.fdata[ntkey][tskey][prop1].shape[0]-1
        plot_corr_dist(self.fdata[ntkey][tskey][prop1][Nn,:], self.fdata[ntkey][tskey][prop2], True, figsize)

    def generate_violin_distribution(self, tskey, maxstd=0.34):
        import seaborn as sns

        dsset = []
        huset = []
        vlset = []

        for k in self.fdata.keys():
            #np.where

            p1p = self.fdata[k][tskey]['Eani'][5,:]
            p1a = self.fdata[k][tskey]['Edft']

            p2p = self.fdata[k][tskey]['Fani'][5,:]
            p2a = self.fdata[k][tskey]['Fdft']

            vlset.append(np.concatenate([np.abs(p1p-p1a), np.abs(p2p-p2a)]))

            dsset.extend([k for s in range(vlset[-1].size)])

            huset.extend(["Eani" for s in range(p1p.size)])
            huset.extend(["Fani" for s in range(p2p.size)])

        longform = {'Property_value': np.concatenate(vlset),
                    'Network': dsset,
                    'Properties': huset,}

        ddat = pd.DataFrame(longform)

        stddev = np.std(longform['Property_value'])

        fig, ax = plt.subplots(figsize=(12.0, 8.0))
        order = [k for k in self.fdata.keys()]
        order.sort()
        print(order)
        ax = sns.violinplot(ax=ax, x="Network", y="Property_value", hue="Properties",
                            data = ddat[ddat.Property_value < 2.0 * stddev], split=True, scale="count", order=order, bw=.075)

        #ax.set_ylim([-1,20])
        plt.show()

    def get_size(self, ntkey, tskey):
        return self.fdata[ntkey][tskey]['Eani'].shape

    def plot_2d_error(self, ntkey, tskey, maxstd=10.0):
        x = self.fdata[ntkey][tskey]['Eani']-self.fdata[ntkey][tskey]['Edft']
        y = self.fdata[ntkey][tskey]['Frmse']

        id = np.where(self.fdata[ntkey][tskey]['Estd'] < maxstd)

        label_size = 14
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size

        fig, ax = plt.subplots(figsize=(12.0, 8.0))

        # Plot ground truth line
        #ax.plot([Fmn, Fmx], [Fmn, Fmx], '--', c='r', linewidth=3)

        # Set labels
        ax.set_xlabel('$E_{ani} - E_{dft}$' + r' $(kcal \times mol^{-1})$', fontsize=22)
        ax.set_ylabel('Force RMSE' + r' $(kcal \times mol^{-1} \times \AA^{-1})$', fontsize=22)

        cmap = mpl.cm.viridis

        # Plot 2d Histogram , range=[[Fmn, Fmx], [Fmn, Fmx]]
        print('Total mols:',id[0].size)
        bins = ax.hist2d(x[id], y[id], bins=100, norm=LogNorm(), range=[[-25.0, 25.0], [0.0, 25]], cmap=cmap)


        # Build color bar
        # cbaxes = fig.add_axes([0.91, 0.1, 0.03, 0.8])
        cb1 = fig.colorbar(bins[-1], cmap=cmap)
        cb1.set_label('Count', fontsize=16)

        plt.show()

    def plot_bar_propsbynet(self, props, dsets, ntwks=[], fontsize=14, bbox_to_anchor=(1.0, 1.1), figsize=(15.0, 12.0), ncol=1, errortype='MAE'):

        N = len(dsets)
        ind = np.arange(N)  # the x locations for the groups
        rects = []
        nets = []

        label_size = fontsize
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size

        fig, axes = plt.subplots(len(props), 1, figsize=(30.0, 24.0))

        if len(ntwks) == 0:
            keys = list(self.fdata.keys())
            keys.sort()
        else:
            keys=ntwks

        for j,(p,ax) in enumerate(zip(props, axes.flatten())):
            bars = dict()
            errs = dict()

            width = 0.85/len(keys)  # the width of the bars

            colors = cm.viridis(np.linspace(0, 1, len(keys)))
            if j == len(keys)-1:
                colors='r'

            for i,(k,c) in enumerate(zip(keys,colors)):
                bars.update({k : []})
                errs.update({k : []})

                for tk in dsets:
                    if errortype is 'MAE':
                        height = hdt.calculatemeanabserror(self.fdata[k][tk][p[2]][5, :], self.fdata[k][tk][p[3]])
                        error = np.std(hdt.calculatemeanabserror(self.fdata[k][tk][p[2]], self.fdata[k][tk][p[3]], axis=1))

                        #if error > height:
                        #    error = height

                        bars[k].append(height)
                        errs[k].append(error)
                    elif errortype is 'RMSE':
                        height = hdt.calculaterootmeansqrerror(self.fdata[k][tk][p[2]][5, :], self.fdata[k][tk][p[3]])
                        error = np.std(hdt.calculaterootmeansqrerror(self.fdata[k][tk][p[2]], self.fdata[k][tk][p[3]], axis=1))

                        #if error > height:
                        #    error = height

                        bars[k].append(height)
                        errs[k].append(error)

                rects.append(ax.bar(ind+i*width, bars[k], width, color=c, bottom=0.0))
                ax.errorbar(ind+i*width+width/2.0,
                            bars[k],
                            errs[k],
                            fmt='.',
                            capsize=8,
                            elinewidth=3,
                            color='red',
                            ecolor='red',
                            markeredgewidth=2)
                ax.set_ylim(p[4])

            # add some text for labels, title and axes ticks
            ax.set_ylabel(p[1], fontsize=fontsize)
            ax.set_title(p[0], fontsize=fontsize)
            ax.set_xticks(ind + ((len(keys)+3)*width) / len(props))
            ax.set_xticklabels([d for d in dsets])
            if j == 0:
                ax.legend(rects, keys, fontsize=fontsize, bbox_to_anchor=bbox_to_anchor, ncol=ncol)

        plt.show()

    def plot_error_by_net(self, props, dsets, ntwks=[], fontsize=14, bbox_to_anchor=(1.0, 1.1), figsize=(15.0, 12.0), ncol=1, errortype='MAE'):

        N = len(dsets)
        ind = np.arange(N)  # the x locations for the groups
        rects = []
        nets = []

        label_size = fontsize
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size

        colors = cm.viridis(np.linspace(0, 1, len(props)))

        fig, axes = plt.subplots(3, 2, figsize=figsize)

        if len(ntwks) == 0:
            keys = list(self.fdata.keys())
            keys.sort()
        else:
            keys=ntwks

        for j,(ds,ax) in enumerate(zip(dsets, axes.flatten())):
            higt = dict()
            errs = dict()



            for i,(tk,c) in enumerate(zip(props,colors)):
                higt.update({tk[0] : []})
                errs.update({tk[0] : []})

                for k in keys:
                    if errortype is 'MAE':
                        Nn = self.fdata[k][ds][tk[2]].shape[0]-1
                        height = hdt.calculatemeanabserror(self.fdata[k][ds][tk[2]][Nn, :], self.fdata[k][ds][tk[3]])
                        error = np.std(hdt.calculatemeanabserror(self.fdata[k][ds][tk[2]], self.fdata[k][ds][tk[3]], axis=1))

                        higt[tk[0]].append(height)
                        errs[tk[0]].append(error)
                    elif errortype is 'RMSE':
                        Nn = self.fdata[k][ds][tk[2]].shape[0] - 1
                        height = hdt.calculaterootmeansqrerror(self.fdata[k][ds][tk[2]][Nn, :], self.fdata[k][ds][tk[3]])
                        error = np.std(hdt.calculaterootmeansqrerror(self.fdata[k][ds][tk[2]], self.fdata[k][ds][tk[3]], axis=1))

                        higt[tk[0]].append(height)
                        errs[tk[0]].append(error)

                x_axis = np.arange(len(higt[tk[0]][:-1]))
                #ax.set_yscale("log", nonposy='clip')
                rects.append(ax.plot(x_axis,
                                     higt[tk[0]][:-1],
                                     '-o',
                                     color=c,
                                     linewidth=5,
                                     label=tk[0]))
                ax.errorbar(x_axis,
                            higt[tk[0]][:-1],
                            yerr=errs[tk[0]][:-1],
                            fmt='.',
                            capsize=8,
                            elinewidth=3,
                            color=c,
                            ecolor=c,
                            markeredgewidth=2)

                ax.plot([-0.1,len(higt[tk[0]][:-1])-1+0.1],
                        [higt[tk[0]][-1],higt[tk[0]][-1]],
                        '--',
                        color=c,
                        linewidth=5)

                ax.legend(fontsize=fontsize, bbox_to_anchor=bbox_to_anchor, ncol=ncol)

                ax.set_title(ds, fontsize=fontsize+2)
                ax.set_xticks(x_axis)
                ax.set_xticklabels([d for d in keys[:-1]])
                ax.set_ylabel(errortype, fontsize=fontsize)

                #ax.set_ylim([0.1,100])

            # add some text for labels, title and axes ticks
            #ax.set_title(p[0], fontsize=fontsize)
            #ax.set_xticks(ind + ((len(keys)+3)*width) / len(props))

            #if j == 0:
            #ax.legend(rects, keys, fontsize=fontsize, bbox_to_anchor=bbox_to_anchor, ncol=ncol)

        plt.show()
