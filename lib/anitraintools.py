import hdnntools as hdt
import pyanitools as pyt

import pyNeuroChem as pync
from pyNeuroChem import cachegenerator as cg

import numpy as np

from scipy.integrate import quad
import pandas as pd

from time import sleep
import subprocess
import random
import re
import os

from multiprocessing import Process
import shutil
import copy

conv_au_ev = 27.21138505

def interval(v, S):
    ps = 0.0
    ds = 1.0 / float(S)
    for s in range(S):
        if v > ps and v <= ps + ds:
            return s
        ps = ps + ds


def get_train_stats(Nn,train_root):
    # rerr = re.compile('EPOCH\s+?(\d+?)\n[\s\S]+?E \(kcal\/mol\)\s+?(\d+?\.\d+?)\s+?(\d+?\.\d+?)\s+?(\d+?\.\d+?)\n\s+?dE \(kcal\/mol\)\s+?(\d+?\.\d+?)\s+?(\d+?\.\d+?)\s+?(\d+?\.\d+?)\n[\s\S]+?Current best:\s+?(\d+?)\n[\s\S]+?Learning Rate:\s+?(\S+?)\n[\s\S]+?TotalEpoch:\s+([\s\S]+?)\n')
    # rerr = re.compile('EPOCH\s+?(\d+?)\s+?\n[\s\S]+?E \(kcal\/mol\)\s+?(\S+?)\s+?(\S+?)\s+?(\S+?)\n\s+?dE \(kcal\/mol\)\s+?(\S+?)\s+?(\S+?)\s+?(\S+?)\n')
    rblk = re.compile('=+?\n([\s\S]+?=+?\n[\s\S]+?(?:=|Deleting))')
    repo = re.compile('EPOCH\s+?(\d+?)\s+?\n')
    rerr = re.compile('\s+?(\S+?\s+?\(\S+?)\s+?((?:\d|inf)\S*?)\s+?((?:\d|inf)\S*?)\s+?((?:\d|inf)\S*?)\n')
    rtme = re.compile('TotalEpoch:\s+?(\d+?)\s+?dy\.\s+?(\d+?)\s+?hr\.\s+?(\d+?)\s+?mn\.\s+?(\d+?\.\d+?)\s+?sc\.')
    comp = re.compile('Termination Criterion Met')

    allnets = []
    completed = []
    for index in range(Nn):
        #print('reading:', train_root + 'train' + str(index) + '/' + 'output.opt')
        if os.path.isfile(train_root + 'train' + str(index) + '/' + 'output.opt'):
            optfile = open(train_root + 'train' + str(index) + '/' + 'output.opt', 'r').read()
            matches = re.findall(rblk, optfile)

            run = dict({'EPOCH': [], 'RTIME': [], 'ERROR': dict()})
            for i, data in enumerate(matches):
                run['EPOCH'].append(int(re.search(repo, data).group(1)))

                m = re.search(rtme, data)
                run['RTIME'].append(86400.0 * float(m.group(1)) +
                                    3600.0 * float(m.group(2)) +
                                    60.0 * float(m.group(3)) +
                                    float(m.group(4)))

                err = re.findall(rerr, data)
                for e in err:
                    if e[0] in run['ERROR']:
                        run['ERROR'][e[0]].append(np.array([float(e[1]), float(e[2]), float(e[3])], dtype=np.float64))
                    else:
                        run['ERROR'].update(
                            {e[0]: [np.array([float(e[1]), float(e[2]), float(e[3])], dtype=np.float64)]})

            for key in run['ERROR'].keys():
                run['ERROR'][key] = np.vstack(run['ERROR'][key])

            if re.match(comp, optfile):
                completed.append(True)
            else:
                completed.append(False)

            allnets.append(run)
        else:
            completed.append(False)
    return allnets, completed

def get_train_stats_ind(index,train_root):
    # rerr = re.compile('EPOCH\s+?(\d+?)\n[\s\S]+?E \(kcal\/mol\)\s+?(\d+?\.\d+?)\s+?(\d+?\.\d+?)\s+?(\d+?\.\d+?)\n\s+?dE \(kcal\/mol\)\s+?(\d+?\.\d+?)\s+?(\d+?\.\d+?)\s+?(\d+?\.\d+?)\n[\s\S]+?Current best:\s+?(\d+?)\n[\s\S]+?Learning Rate:\s+?(\S+?)\n[\s\S]+?TotalEpoch:\s+([\s\S]+?)\n')
    # rerr = re.compile('EPOCH\s+?(\d+?)\s+?\n[\s\S]+?E \(kcal\/mol\)\s+?(\S+?)\s+?(\S+?)\s+?(\S+?)\n\s+?dE \(kcal\/mol\)\s+?(\S+?)\s+?(\S+?)\s+?(\S+?)\n')
    rblk = re.compile('=+?\n([\s\S]+?=+?\n[\s\S]+?(?:=|Deleting))')
    repo = re.compile('EPOCH\s+?(\d+?)\s+?\n')
    rerr = re.compile('\s+?(\S+?\s+?\(\S+?)\s+?((?:\d|inf)\S*?)\s+?((?:\d|inf)\S*?)\s+?((?:\d|inf)\S*?)\n')
    rtme = re.compile('TotalEpoch:\s+?(\d+?)\s+?dy\.\s+?(\d+?)\s+?hr\.\s+?(\d+?)\s+?mn\.\s+?(\d+?\.\d+?)\s+?sc\.')
    comp = re.compile('Termination Criterion Met')

    allnets = []
    completed = False
    #print('reading:', train_root + 'train' + str(index) + '/' + 'output.opt')
    if os.path.isfile(train_root + 'train' + str(index) + '/' + 'output.opt'):
        optfile = open(train_root + 'train' + str(index) + '/' + 'output.opt', 'r').read()
        matches = re.findall(rblk, optfile)

        run = dict({'EPOCH': [], 'RTIME': [], 'ERROR': dict()})
        for i, data in enumerate(matches):
            run['EPOCH'].append(int(re.search(repo, data).group(1)))

            m = re.search(rtme, data)
            run['RTIME'].append(86400.0 * float(m.group(1)) +
                                3600.0 * float(m.group(2)) +
                                60.0 * float(m.group(3)) +
                                float(m.group(4)))

            err = re.findall(rerr, data)
            for e in err:
                if e[0] in run['ERROR']:
                    run['ERROR'][e[0]].append(np.array([float(e[1]), float(e[2]), float(e[3])], dtype=np.float64))
                else:
                    run['ERROR'].update(
                        {e[0]: [np.array([float(e[1]), float(e[2]), float(e[3])], dtype=np.float64)]})

        for key in run['ERROR'].keys():
            run['ERROR'][key] = np.vstack(run['ERROR'][key])

        if re.match(comp, optfile):
            completed = True
        else:
            completed = False

        allnets.append(run)
    else:
        completed = False
    return allnets, True

class ANITesterTool:
    
    def load_models(self):
        self.ncl = [pync.molecule(self.cnstfile, self.saefile, self.model_path + 'train' + str(i) + '/networks/', self.gpuid, False) for i in range(self.ens_size)]

    
    def __init__(self,model_path,ens_size,gpuid):
        self.model_path = model_path
        self.ens_size = ens_size
        self.gpuid = gpuid
        self.cnstfile = model_path+[f for f in os.listdir(self.model_path) if f[-7:] == '.params'][0]
        self.saefile  = model_path+[f for f in os.listdir(self.model_path) if f[-4:] == '.dat'][0]
        
        self.load_models()
        
    def evaluate_individual_testset(self,energy_key='energies',force_key='forces',forces=False,pbc=True,remove_sae=True):
        self.Evals = []
        self.Fvals = []
        for i,nc in enumerate(self.ncl):
            adl = pyt.anidataloader(self.model_path+'/testset/testset'+str(i)+'.h5')
            
            Evals_ind = []
            Fvals_ind = []
            for data in adl:
                S = data['species']
            
                X = data['coordinates']
                C = data['cell']
                
                E = conv_au_ev*data[energy_key]
                F = conv_au_ev*data[force_key]

                if remove_sae:
                    Esae = conv_au_ev*hdt.compute_sae(self.saefile,S)
                else:
                    Esae = 0.0
                
                for x,c,e,f in zip(X,C,E,F):
                    if pbc is True:
                        pbc_inv = np.linalg.inv(c).astype(np.float64)
                    
                        nc.setMolecule(coords=np.array(x,dtype=np.float64), types=list(S))
                        nc.setPBC(bool(True), bool(True), bool(True))
                        nc.setCell(np.array(c,dtype=np.float64),pbc_inv)
                    else:
                        nc.setMolecule(coords=np.array(x, dtype=np.float64), types=list(S))

                    Eani = conv_au_ev*nc.energy().copy()[0]
                    if forces:
                        Fani = conv_au_ev*nc.force().copy()
                    else:
                        Fani = f

                    if pbc is True:
                        Evals_ind.append(np.array([Eani-Esae,e-Esae])/len(S))
                    else:
                        Evals_ind.append(np.array([Eani-Esae,e-Esae]))
                    Fvals_ind.append(np.stack([Fani.flatten(),f.flatten()]).T)
                    
            self.Evals.append(np.stack(Evals_ind))
            self.Fvals.append(np.vstack(Fvals_ind))

        return self.Evals,self.Fvals

    def evaluate_individual_dataset(self,dataset_file,energy_key='energies',force_key='forces',forces=False,pbc=True,remove_sae=True):
        self.Evals = []
        self.Fvals = []
        for i,nc in enumerate(self.ncl):
            adl = pyt.anidataloader(dataset_file)

            Evals_ind = []
            Fvals_ind = []
            for data in adl:
                S = data['species']

                X = data['coordinates']

                E = conv_au_ev*data[energy_key]

                if pbc:
                    C = data['cell']
                else:
                    C = np.zeros(shape=(E.size,3,3),dtype=np.float64)

                if forces:
                    F = conv_au_ev*data[force_key]
                else:
                    F = np.zeros(shape=X.shape,dtype=np.float64)

                if remove_sae:
                    Esae = conv_au_ev*hdt.compute_sae(self.saefile,S)
                else:
                    Esae = 0.0

                for x,c,e,f in zip(X,C,E,F):
                    if pbc is True:
                        pbc_inv = np.linalg.inv(c).astype(np.float64)

                        nc.setMolecule(coords=np.array(x,dtype=np.float64), types=list(S))
                        nc.setPBC(bool(True), bool(True), bool(True))
                        nc.setCell(np.array(c,dtype=np.float64),pbc_inv)
                    else:
                        nc.setMolecule(coords=np.array(x, dtype=np.float64), types=list(S))

                    Eani = conv_au_ev*nc.energy().copy()[0]
                    if forces:
                        Fani = conv_au_ev*nc.force().copy()
                    else:
                        Fani = f

                    if pbc is True:
                        Evals_ind.append(np.array([Eani-Esae,e-Esae])/len(S))
                    else:
                        Evals_ind.append(np.array([Eani-Esae,e-Esae]))
                    Fvals_ind.append(np.stack([Fani.flatten(),f.flatten()]).T)

            self.Evals.append(np.stack(Evals_ind))
            self.Fvals.append(np.vstack(Fvals_ind))

        return self.Evals,self.Fvals
 
    def build_ind_error_dataframe(self):
        d = {'Emae':[],'Erms':[],'Fmae':[],'Frms':[],}
        for i,(e,f) in enumerate(zip(self.Evals,self.Fvals)):
            d['Emae'].append(1000.0*hdt.calculatemeanabserror(e[:,0],e[:,1]))
            d['Erms'].append(1000.0*hdt.calculaterootmeansqrerror(e[:,0],e[:,1]))
            d['Fmae'].append(hdt.calculatemeanabserror(f[:,0],f[:,1]))
            d['Frms'].append(hdt.calculaterootmeansqrerror(f[:,0],f[:,1]))
        df = pd.DataFrame(data=d)
        df.loc['Avg.'] = df.mean()
        return df
            
    
    def plot_corr_dist(self, Xa, Xp, inset=True,linfit=True, xlabel='$F_{dft}$' + r' $(kcal \times mol^{-1} \times \AA^{-1})$', ylabel='$F_{dft}$' + r' $(kcal \times mol^{-1} \times \AA^{-1})$', figsize=[13,10]):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        import matplotlib.cm as cm

        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        from matplotlib.backends.backend_pdf import PdfPages

        cmap = mpl.cm.viridis

        Fmx = Xa.max()
        Fmn = Xa.min()
    
        label_size = 14
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size
    
        fig, ax = plt.subplots(figsize=figsize)
    
        # Plot ground truth line
        if linfit:
            ax.plot([Fmn, Fmx], [Fmn, Fmx], '--', c='r', linewidth=3)
    
        # Set labels
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
    
        #cmap = mpl.cm.viridis
        #cmap = mpl.cm.brg
    
        # Plot 2d Histogram
        if linfit:
            bins = ax.hist2d(Xa, Xp, bins=200, norm=LogNorm(), range= [[Xa.min(), Xa.max()], [Xp.min(), Xp.max()]], cmap=cmap)
        else:
            bins = ax.hist2d(Xa, Xp, bins=200, norm=LogNorm(), cmap=cmap)
    
        # Build color bar
        #cbaxes = fig.add_axes([0.91, 0.1, 0.03, 0.8])
        cb1 = fig.colorbar(bins[-1], cmap=cmap)
        cb1.set_label('Count', fontsize=16)
    
        # Annotate with errors
        PMAE = hdt.calculatemeanabserror(Xa, Xp)
        PRMS = hdt.calculaterootmeansqrerror(Xa, Xp)
        if linfit:
            ax.text(0.75*((Fmx-Fmn))+Fmn, 0.43*((Fmx-Fmn))+Fmn, 'MAE='+"{:.3f}".format(PMAE)+'\nRMSE='+"{:.3f}".format(PRMS), fontsize=20,
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
    
        if not linfit:
            plt.vlines(x=0.0,ymin=130,ymax=300,linestyle='--',color='red')
            
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

def plot_corr_dist_ax(ax, Xa, Xp, errors=False,linfit=True):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import matplotlib.cm as cm

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from matplotlib.backends.backend_pdf import PdfPages

    cmap = mpl.cm.viridis

    Fmx = Xa.max()
    Fmn = Xa.min()

    label_size = 14

    # Plot ground truth line
    if linfit:
        ax.plot([Fmn, Fmx], [Fmn, Fmx], '--', c='r', linewidth=1)

    # Plot 2d Histogram
    if linfit:
        bins = ax.hist2d(Xa, Xp, bins=200, norm=LogNorm(), range= [[Xa.min(), Xa.max()], [Xp.min(), Xp.max()]], cmap=cmap)
    else:
        bins = ax.hist2d(Xa, Xp, bins=200, norm=LogNorm(), cmap=cmap)
        # Build color bar
        #cbaxes = fig.add_axes([0.91, 0.1, 0.03, 0.8])
        #cb1 = ax.colorbar(bins[-1], cmap=cmap)
        #cb1.set_label('Count', fontsize=16)

    # Annotate with errors
    PMAE = hdt.calculatemeanabserror(Xa, Xp)
    PRMS = hdt.calculaterootmeansqrerror(Xa, Xp)
    if errors:
        #ax.text(0.55*((Fmx-Fmn))+Fmn, 0.2*((Fmx-Fmn))+Fmn, 'MAE='+"{:.3f}".format(PMAE)+'\nRMSE='+"{:.3f}".format(PRMS), fontsize=20, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
        ax.text(0.60, 0.05, 'MAE='+"{:.3f}".format(PMAE)+'\nRMSE='+"{:.3f}".format(PRMS), transform=ax.transAxes, fontsize=20, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})

    #if not linfit:
    #    plt.vlines(x=0.0,ymin=130,ymax=300,linestyle='--',color='red')        
 
class anitrainerparamsdesigner():
    def __init__(self, elements, Nrr, Rcr, Nar=0, Nzt=0, Rca=3.5, Xst=0.7, Charge=False, Repuls=False, ACA=False, descriptor="ANI_NORMAL"):
        self.params = {"elm":elements,
                       "Nrr":Nrr,
                       "Rcr":Rcr,
                       "Nar":Nar,
                       "Nzt":Nzt,
                       "Rca":Rca,
                       "Xst":Xst,
                       "ACA":ACA,
                       "Crg":Charge,
                       "Rps":Repuls,
                       "Dsc":descriptor
                      }

    # ------------------------------------------
    #           Radial Function Cos
    # ------------------------------------------
    def cutoffcos(self, X, Rc):
        Xt = X
        for i in range(0, Xt.shape[0]):
            if Xt[i] > Rc:
                Xt[i] = Rc

        return 0.5 * (np.cos((np.pi * Xt) / Rc) + 1.0)

    # ------------------------------------------
    #           Radial Function Cos
    # ------------------------------------------
    def radialfunction(self, X, eta, Rs):
        return np.exp(-eta * (X - Rs) ** 2.0)

    # ------------------------------------------
    #          Radial Build Functions
    # ------------------------------------------
    def radialfunctioncos(self, X, eta, Rc, Rs):
        return self.radialfunction(X, eta, Rs) * self.cutoffcos(X, Rc)

    def compute_overlap(self, eta, Rs1, Rs2):
        func1 = lambda x: self.radialfunction(x, eta, Rs1)
        func2 = lambda x: self.radialfunction(x, eta, Rs2)
        funcC = lambda x: min(func1(x),func2(x))

        i_f1 = quad(func1, -10, 20)[0]
        i_fC = quad(funcC, -10, 20)

        return i_fC[0]/i_f1

    def determine_eta(self, req_olap, Rs1, Rs2, dx = 0.1):
        eta = 0.1
        olap = 1.0
        while olap > req_olap:
            eta = eta + 0.1
            olap = self.compute_overlap(eta, Rs1, Rs2)
        return eta

    def obtain_radial_parameters(self, Nrr, Xst, Rcr):
        ShfR = np.zeros(Nrr)
        for i in range(0, Nrr):
            stepsize = (Rcr - Xst) / float(Nrr)
            step = i * stepsize + Xst
            ShfR[i] = step
        eta = self.determine_eta(0.4, ShfR[0], ShfR[1])
        return ShfR, eta

    def get_Rradial_parameters(self):
        Nrr = self.params['Nrr']
        Xst = self.params['Xst']
        Rcr = self.params['Rcr']
        return self.obtain_radial_parameters(Nrr, Xst, Rcr)

    def get_Aradial_parameters(self):
        Nar = self.params['Nar']
        Xst = self.params['Xst']
        Rca = self.params['Rca']
        return self.obtain_radial_parameters(Nar, Xst, Rca)

    def plot_radial_funcs(self, Shf, Eta, Rc):
        for sr in Shf:
            X = np.linspace(0, Rc, 1000, endpoint=True)
            Y = self.radialfunctioncos(X, Eta, Rc, sr)
            plt.plot(X, Y, color='red', linewidth=2)
        plt.show()

    def plot_Rradial_funcs(self):
        ShfR, EtaR = self.obtain_Rradial_parameters()
        self.plot_radial_funcs(ShfR, EtaR, self.params['Rcr'])

    def plot_Aradial_funcs(self):
        ShfA, EtaA = self.obtain_Aradial_parameters()
        self.plot_radial_funcs(ShfA, EtaA, self.params['Rca'])

    # ------------------------------------------
    #          Angular Build Functions
    # ------------------------------------------
    def angularfunction(self, T, zeta, lam, Ts):
        F = 0.5 * (2.0 ** (1.0 - zeta)) * ((1.0 + lam * np.cos(T - Ts)) ** zeta)
        return F

    def compute_overlap_angular(self, zeta, Zs1, Zs2):
        func1 = lambda x: self.angularfunction(x, zeta, 1, Zs1)
        func2 = lambda x: self.angularfunction(x, zeta, 1, Zs2)
        funcC = lambda x: min(func1(x),func2(x))

        i_f1 = quad(func1, -6, 6)[0]
        i_fC = quad(funcC, -6, 6)

        return i_fC[0]/i_f1

    def determine_zeta(self, req_olap, Zs1, Zs2, dx = 0.1):
        zeta = 4.0
        olap = 1.0
        while olap > req_olap:
            zeta = zeta + dx
            olap = self.compute_overlap_angular(zeta, Zs1, Zs2)
        return zeta

    def obtain_angular_parameters(self, Nzt):
        ShfZ = np.zeros(Nzt)
        for i in range(0, Nzt):
            stepsize = np.pi / float(Nzt)
            step = i * stepsize + stepsize/2.0
            ShfZ[i] = step
        zeta = self.determine_zeta(0.35, ShfZ[0], ShfZ[1])
        return ShfZ, zeta

    def get_angular_parameters(self):
        Nzt = self.params['Nzt']
        return self.obtain_angular_parameters(Nzt)

    def build_angular_plots(self, ShfZ, Zeta):
        for sz in ShfZ:
            X = np.linspace(0, np.pi, 1000, endpoint=True)
            Y = self.angularfunction(X, Zeta, 1, sz)
            plt.plot(X, Y, color='red', linewidth=2)
        plt.show()

    def plot_angular_funcs(self):
        ShfZ, Zeta = self.get_angular_parameters()
        self.build_angular_plots(ShfZ, Zeta)


    # ------------------------------------------
    #             Get a file name
    # ------------------------------------------
    def get_filename(self):
        return "r"+"".join(self.params["elm"])+"-" + "{0:.1f}".format(self.params["Rcr"]) + "R_" \
                                                   + str(self.params["Nrr"]) + "-"\
                                                   + "{0:.1f}".format(self.params["Rca"]) + "A_a" \
                                                   + str(self.params["Nar"]) + "-" \
                                                   + str(self.params["Nzt"]) + ".params"

    # ------------------------------------------
    #            Print data to file
    # ------------------------------------------
    def printdatatofile(self, f, title, X, N):
        f.write(title + ' = [')
        for i in range(0, N):
            if i < N - 1:
                s = "{:.7e}".format(X[i]) + ','
            else:
                s = "{:.7e}".format(X[i])
            f.write(s)
        f.write(']\n')

    def get_aev_size(self):
        Na = len(self.params['elm'])
        Nar = self.params['Nar']
        Nzt = self.params['Nzt']
        Nrr = self.params['Nrr']

        Nat = Nar * (Na * (Na + 1) / 2) * Nzt
        Nrt = Nrr * Na
        return int(Nat + Nrt)

    # ------------------------------------------
    #             Create params file
    # ------------------------------------------
    def create_params_file(self, path):
        ShfR,EtaR = self.get_Rradial_parameters()

        if self.params["Nzt"] is not 0 or self.params["Nar"] is not 0:
            ShfA,EtaA = self.get_Aradial_parameters()
            ShfZ,Zeta = self.get_angular_parameters()

        Rcr = self.params['Rcr']
        Rca = self.params['Rca']

        f = open(path+"/"+self.get_filename(),"w")

        f.write('DESC = ' + self.params['Dsc'] + '\n')
        f.write('TM = ' + str(1) + '\n')
        f.write('CG = ' + str(1 if self.params['Crg'] else 0) + '\n')
        f.write('RP = ' + str(1 if self.params['Rps'] else 0) + '\n')
        f.write('AC = ' + str(1 if self.params['ACA'] else 0) + '\n')
        f.write('Rcr = ' + "{:.4e}".format(Rcr) + '\n')
        f.write('Rca = ' + "{:.4e}".format(Rca) + '\n')
        self.printdatatofile(f, 'EtaR', [EtaR], 1)
        self.printdatatofile(f, 'ShfR', ShfR, ShfR.size)

        if self.params["Nzt"] is not 0 or self.params["Nar"] is not 0:
            self.printdatatofile(f, 'Zeta', [Zeta], 1)
            self.printdatatofile(f, 'ShfZ', ShfZ, ShfZ.size)
            self.printdatatofile(f, 'EtaA', [EtaA], 1)
            self.printdatatofile(f, 'ShfA', ShfA, ShfA.size)

        f.write('Atyp = [' + ",".join(self.params['elm']) + ']\n')
        f.close()

class anitrainerinputdesigner:
    def __init__(self):
        self.params = {"sflparamsfile": None,  # AEV parameters file
                       "ntwkStoreDir": "networks/",  # Store network dir
                       "atomEnergyFile": None,  # Atomic energy shift file
                       "nmax": 0,  # Max training iterations
                       "tolr": 50,  # Annealing tolerance (patience)
                       "emult": 0.5,  # Annealing multiplier
                       "eta": 0.001,  # Learning rate
                       "tcrit": 1.0e-5,  # eta termination crit.
                       "tmax": 0,  # Maximum time (0 = inf)
                       "tbtchsz": 2048,  # training batch size
                       "vbtchsz": 2048,  # validation batch size
                       "gpuid": 0,  # Default GPU id (is overridden by -g flag for HDAtomNNP-Trainer exe)
                       "ntwshr": 0,  # Use a single network for all types... (THIS IS BROKEN, DO NOT USE)
                       "nkde": 2,  # Energy delta regularization
                       "energy": 1,  # Enable/disable energy training
                       "force": 0,  # Enable/disable force training
                       "dipole": 0,  # Enable/disable dipole training
                       "charge": 0,  # Enable/disable charge training
                       "acachg": 0,  # Enable/disable ACA charge training
                       "fmult": 1.0,  # Multiplier of force cost
                       "pbc": 0,  # Use PBC in training (Warning, this only works for data with a single rect. box size)
                       "cmult": 1.0,  # Charge cost multiplier (CHARGE TRAINING BROKEN IN CURRENT VERSION)
                       "runtype": "ANNP_CREATE_HDNN_AND_TRAIN",  # DO NOT CHANGE - For NeuroChem backend
                       "adptlrn": "OFF",
                       "decrate": 0.9,
                       "moment": "ADAM",
                       "mu": 0.99
                       }

        self.layers = dict()

    def add_layer(self, atomtype, layer_dict):
        layer_dict.update({"type": 0})
        if atomtype not in self.layers:
            self.layers[atomtype] = [layer_dict]
        else:
            self.layers[atomtype].append(layer_dict)

    def set_parameter(self, key, value):
        self.params[key] = value

    def print_layer_parameters(self):
        for ak in self.layers.keys():
            print('Species:', ak)
            for l in self.layers[ak]:
                print('  -', l)

    def print_training_parameters(self):
        print(self.params)

    def __get_value_string__(self, value):

        if type(value) == float:
            string = "{0:10.7e}".format(value)
        else:
            string = str(value)

        return string

    def __build_network_str__(self, iptsize):

        network = "network_setup {\n"
        network += "    inputsize=" + str(iptsize) + ";\n"

        for ak in self.layers.keys():
            network += "    atom_net " + ak + " $\n"

            if int(self.params["dipole"]) != 0 or int(self.params["charge"]) != 0:
                self.layers[ak].append({"nodes": 12, "activation": 6, "type": 0})
                #self.layers[ak].append({"nodes": 2, "activation": 6, "type": 0})
            elif int(self.params["acachg"]) != 0:
                self.layers[ak].append({"nodes": 2, "activation": 6, "type": 0})
            else:
                self.layers[ak].append({"nodes": 1, "activation": 6, "type": 0})

            for l in self.layers[ak]:
                network += "        layer [\n"
                for key in l.keys():
                    network += "            " + key + "=" + self.__get_value_string__(l[key]) + ";\n"
                network += "        ]\n"
            network += "    $\n"
        network += "}\n"
        return network

    def write_input_file(self, file, iptsize):
        f = open(file, 'w')
        for key in self.params.keys():
            f.write(key + '=' + self.__get_value_string__(self.params[key]) + '\n')
        f.write(self.__build_network_str__(iptsize))
        f.close()


class alaniensembletrainer():
    def __init__(self, train_root, netdict, input_builder, h5dir, Nn, random_seed=-1):
        
        if random_seed != -1:
            np.random.seed(random_seed)

        self.train_root = train_root
        # self.train_pref = train_pref
        self.h5dir = h5dir
        self.Nn = Nn
        self.netdict = netdict
        self.iptbuilder = input_builder

        if h5dir is not None:
            self.h5file = [f for f in os.listdir(self.h5dir) if f.rsplit('.', 1)[1] == 'h5']
        # print(self.h5dir,self.h5file)

    def build_training_cache(self, forces=True):
        store_dir = self.train_root + "cache-data-"
        N = self.Nn

        for i in range(N):
            if not os.path.exists(store_dir + str(i)):
                os.mkdir(store_dir + str(i))

            if os.path.exists(store_dir + str(i) + '/../testset/testset' + str(i) + '.h5'):
                os.remove(store_dir + str(i) + '/../testset/testset' + str(i) + '.h5')

            if not os.path.exists(store_dir + str(i) + '/../testset'):
                os.mkdir(store_dir + str(i) + '/../testset')

        cachet = [cg('_train', self.netdict['saefile'], store_dir + str(r) + '/', False) for r in range(N)]
        cachev = [cg('_valid', self.netdict['saefile'], store_dir + str(r) + '/', False) for r in range(N)]
        testh5 = [pyt.datapacker(store_dir + str(r) + '/../testset/testset' + str(r) + '.h5') for r in range(N)]

        Nd = np.zeros(N, dtype=np.int32)
        Nbf = 0
        for f, fn in enumerate(self.h5file):
            print('Processing file(' + str(f + 1) + ' of ' + str(len(self.h5file)) + '):', fn)
            adl = pyt.anidataloader(self.h5dir + fn)

            To = adl.size()
            Ndc = 0
            Fmt = []
            Emt = []
            for c, data in enumerate(adl):
                Pn = data['path'] + '_' + str(f).zfill(6) + '_' + str(c).zfill(6)

                # Progress indicator
                # sys.stdout.write("\r%d%% %s" % (int(100 * c / float(To)), Pn))
                # sys.stdout.flush()

                # print(data.keys())

                # Extract the data
                X = data['coordinates']
                E = data['energies']
                S = data['species']

                # 0.0 forces if key doesnt exist
                if forces:
                    F = data['forces']
                else:
                    F = 0.0 * X

                Fmt.append(np.max(np.linalg.norm(F, axis=2), axis=1))
                Emt.append(E)
                Mv = np.max(np.linalg.norm(F, axis=2), axis=1)

                index = np.where(Mv > 10.5)[0]
                indexk = np.where(Mv <= 10.5)[0]

                Nbf += index.size

                # CLear forces
                X = X[indexk]
                F = F[indexk]
                E = E[indexk]

                Esae = hdt.compute_sae(self.netdict['saefile'], S)

                hidx = np.where(np.abs(E - Esae) > 10.0)
                lidx = np.where(np.abs(E - Esae) <= 10.0)
                if hidx[0].size > 0:
                    print('  -(' + str(c).zfill(3) + ')High energies detected:\n    ', E[hidx])

                X = X[lidx]
                E = E[lidx]
                F = F[lidx]

                Ndc += E.size

                if (set(S).issubset(self.netdict['atomtyp'])):
                    # if (set(S).issubset(['C', 'N', 'O', 'H', 'F', 'S', 'Cl'])):

                    # Random mask
                    R = np.random.uniform(0.0, 1.0, E.shape[0])
                    idx = np.array([interval(r, N) for r in R])

                    # Build random split lists
                    split = []
                    for j in range(N):
                        split.append([i for i, s in enumerate(idx) if s == j])
                        nd = len([i for i, s in enumerate(idx) if s == j])
                        Nd[j] = Nd[j] + nd

                    # Store data
                    for i, t, v, te in zip(range(N), cachet, cachev, testh5):
                        ## Store training data
                        X_t = np.array(np.concatenate([X[s] for j, s in enumerate(split) if j != i]), order='C',
                                       dtype=np.float32)
                        F_t = np.array(np.concatenate([F[s] for j, s in enumerate(split) if j != i]), order='C',
                                       dtype=np.float32)
                        E_t = np.array(np.concatenate([E[s] for j, s in enumerate(split) if j != i]), order='C',
                                       dtype=np.float64)

                        if E_t.shape[0] != 0:
                            t.insertdata(X_t, F_t, E_t, list(S))

                        ## Split test/valid data and store\
                        # tv_split = np.array_split(split[i], 2)

                        ## Store Validation
                        if np.array(split[i]).size > 0:
                            X_v = np.array(X[split[i]], order='C', dtype=np.float32)
                            F_v = np.array(F[split[i]], order='C', dtype=np.float32)
                            E_v = np.array(E[split[i]], order='C', dtype=np.float64)
                            if E_v.shape[0] != 0:
                                v.insertdata(X_v, F_v, E_v, list(S))

                                ## Store testset
                                # if tv_split[1].size > 0:
                                # X_te = np.array(X[split[i]], order='C', dtype=np.float32)
                                # F_te = np.array(F[split[i]], order='C', dtype=np.float32)
                                # E_te = np.array(E[split[i]], order='C', dtype=np.float64)
                                # if E_te.shape[0] != 0:
                                #    te.store_data(Pn, coordinates=X_te, forces=F_te, energies=E_te, species=list(S))


                                # sys.stdout.write("\r%d%%" % int(100))
                                # print(" Data Kept: ", Ndc, 'High Force: ', Nbf)
                                # sys.stdout.flush()
                                # print("")

        # Print some stats
        print('Data count:', Nd)
        print('Data split:', 100.0 * Nd / np.sum(Nd), '%')

        # Save train and valid meta file and cleanup testh5
        for t, v, th in zip(cachet, cachev, testh5):
            t.makemetadata()
            v.makemetadata()
            th.cleanup()

    def sae_linear_fitting(self, Ekey='energies', energy_unit=1.0, Eax0sum=False):
        from sklearn import linear_model
        print('Performing linear fitting...')

        datadir = self.h5dir
        sae_out = self.netdict['saefile']

        smap = dict()
        for i, Z in enumerate(self.netdict['atomtyp']):
            smap.update({Z: i})

        Na = len(smap)
        files = os.listdir(datadir)

        X = []
        y = []
        for f in files[0:20]:
            print(f)
            adl = pyt.anidataloader(datadir + f)
            for data in adl:
                # print(data['path'])
                S = data['species']

                if data[Ekey].size > 0:
                    if Eax0sum:
                        E = energy_unit * np.sum(np.array(data[Ekey], order='C', dtype=np.float64), axis=1)
                    else:
                        E = energy_unit * np.array(data[Ekey], order='C', dtype=np.float64)

                    S = S[0:data['coordinates'].shape[1]]
                    unique, counts = np.unique(S, return_counts=True)
                    x = np.zeros(Na, dtype=np.float64)
                    for u, c in zip(unique, counts):
                        x[smap[u]] = c

                    for e in E:
                        X.append(np.array(x))
                        y.append(np.array(e))

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        lin = linear_model.LinearRegression(fit_intercept=False)
        lin.fit(X, y)

        coef = lin.coef_
        print(coef)

        sae = open(sae_out, 'w')
        for i, c in enumerate(coef[0]):
            sae.write(next(key for key, value in smap.items() if value == i) + ',' + str(i) + '=' + str(c) + '\n')

        sae.close()

        print('Linear fitting complete.')

    def build_strided_training_cache(self, Nblocks, Nvalid, Ntest, build_test=True,
                                     Ekey='energies', energy_unit=1.0,
                                     forces=True, grad=False, Fkey='forces', forces_unit=1.0,
                                     dipole=False, dipole_unit=1.0, Dkey='dipoles',
                                     charge=False, charge_unit=1.0, Ckey='charges',
                                     solvent=False,
                                     pbc=False,
                                     Eax0sum=False, rmhighe=True,rmhighf=False,force_exact_split=False):
        if not os.path.isfile(self.netdict['saefile']):
            self.sae_linear_fitting(Ekey=Ekey, energy_unit=energy_unit, Eax0sum=Eax0sum)

        h5d = self.h5dir

        store_dir = self.train_root + "cache-data-"
        N = self.Nn
        Ntrain = Nblocks - Nvalid - Ntest

        if Nblocks % N != 0:
            raise ValueError('Error: number of networks must evenly divide number of blocks.')

        Nstride = Nblocks / N

        for i in range(N):
            if not os.path.exists(store_dir + str(i)):
                os.mkdir(store_dir + str(i))

            if build_test:
                if os.path.exists(store_dir + str(i) + '/../testset/testset' + str(i) + '.h5'):
                    os.remove(store_dir + str(i) + '/../testset/testset' + str(i) + '.h5')

                if not os.path.exists(store_dir + str(i) + '/../testset'):
                    os.mkdir(store_dir + str(i) + '/../testset')

        cachet = [cg('_train', self.netdict['saefile'], store_dir + str(r) + '/', False) for r in range(N)]
        cachev = [cg('_valid', self.netdict['saefile'], store_dir + str(r) + '/', False) for r in range(N)]

        if build_test:
            testh5 = [pyt.datapacker(store_dir + str(r) + '/../testset/testset' + str(r) + '.h5') for r in range(N)]

        if rmhighe:
            dE = []
            for f in self.h5file:
                adl = pyt.anidataloader(h5d + f)
                for data in adl:
                    S = data['species']
                    E = data[Ekey]
                    X = data['coordinates']

                    Esae = hdt.compute_sae(self.netdict['saefile'], S)

                    dE.append((E - Esae) / np.sqrt(len(S)))

            dE = np.concatenate(dE)
            cidx = np.where(np.abs(dE) < 15.0)
            std = dE[cidx].std()
            men = np.mean(dE[cidx])

            print(men, std, men + std)
            idx = np.intersect1d(np.where(dE >= -np.abs(8 * std + men))[0], np.where(dE <= np.abs(8 * std + men))[0])
            cnt = idx.size
            print('DATADIST: ', dE.size, cnt, (dE.size - cnt), 100.0 * ((dE.size - cnt) / dE.size))

        E = []
        data_count = np.zeros((N, 3), dtype=np.int32)
        for f in self.h5file:
            adl = pyt.anidataloader(h5d + f)
            for data in adl:
                # print(data['path'],data['energies'].size)

                S = data['species']

                if data[Ekey].size > 0 and (set(S).issubset(self.netdict['atomtyp'])):

                    X = np.array(data['coordinates'], order='C', dtype=np.float32)

                    if Eax0sum:
                        E = energy_unit * np.sum(np.array(data[Ekey], order='C', dtype=np.float64), axis=1)
                    else:
                        E = energy_unit * np.array(data[Ekey], order='C', dtype=np.float64)

                    Sv = np.zeros((E.size,7),dtype=np.float32)
                    if solvent:
                        Sv = np.array(data['solvent'], order='C', dtype=np.float32)

                    if forces and not grad:
                        F = forces_unit * np.array(data[Fkey], order='C', dtype=np.float32)
                    elif forces and grad:
                        F = -forces_unit * np.array(data[Fkey], order='C', dtype=np.float32)
                    else:
                        F = 0.0 * X

                    D = np.zeros((E.size,3),dtype=np.float32)
                    if dipole:
                        D = dipole_unit * np.array(data[Dkey], order='C', dtype=np.float32).reshape(E.size,3)
                    else:
                        D = 0.0 * D

                    P = np.zeros((E.size,3,3),dtype=np.float32)
                    if pbc:
                        P = np.array(data['cell'], order='C', dtype=np.float32).reshape(E.size,3,3)
                    else:
                        P = 0.0 * P

                    C = np.zeros((E.size,X.shape[1]),dtype=np.float32)
                    if charge:
                        C = charge_unit * np.array(data[Ckey], order='C', dtype=np.float32).reshape(E.size,len(S))
                    else:
                        C = 0.0 * C

                    if rmhighe:
                        Esae = hdt.compute_sae(self.netdict['saefile'], S)

                        ind_dE = (E - Esae) / np.sqrt(len(S))

                        hidx = np.union1d(np.where(ind_dE < -(9.0 * std + men))[0],
                                          np.where(ind_dE >  (9.0 * std + men))[0])

                        lidx = np.intersect1d(np.where(ind_dE >= -(9.0 * std + men))[0],
                                              np.where(ind_dE <=  (9.0 * std + men))[0])

                        if hidx.size > 0:
                            print('  -(' + f + ':' + data['path'] + ')High energies detected:\n    ',
                                  (E[hidx] - Esae) / np.sqrt(len(S)))

                        X = X[lidx]
                        E = E[lidx]
                        F = F[lidx]
                        D = D[lidx]
                        C = C[lidx]
                        P = P[lidx]
                        #Sv = Sv[lidx]

                    if rmhighf:
                        hfidx = np.where(np.abs(F) > 2.0)
                        if hfidx[0].size > 0:
                            print('High force:',hfidx)
                            hfidx = np.all(np.abs(F).reshape(E.size,-1) <= 2.0,axis=1)
                            X = X[hfidx]
                            E = E[hfidx]
                            F = F[hfidx]
                            D = D[hfidx]
                            C = C[hfidx]
                            P = P[hfidx]
                            #Sv = Sv[hfidx]

                    # Build random split index
                    ridx = np.random.randint(0, Nblocks, size=E.size)
                    Didx = [np.argsort(ridx)[np.where(ridx == i)] for i in range(Nblocks)]

                    # Build training cache
                    for nid, cache in enumerate(cachet):
                        set_idx = np.concatenate(
                            [Didx[((bid + nid * int(Nstride)) % Nblocks)] for bid in range(Ntrain)])
                        if set_idx.size != 0:
                            data_count[nid, 0] += set_idx.size
                            #print("Py tDIPOLE1:\n",D[set_idx][0:3],D.shape)
                            #print("Py tDIPOLE2:\n",D[set_idx][-3:],D.shape)
                            #cache.insertdata(X[set_idx], F[set_idx], C[set_idx], D[set_idx], E[set_idx], list(S))
                            #cache.insertdata(X[set_idx], F[set_idx], C[set_idx], D[set_idx], P[set_idx], E[set_idx], Sv[set_idx], list(S))
                            cache.insertdata(X[set_idx], F[set_idx], C[set_idx], D[set_idx], P[set_idx], E[set_idx], list(S))

                    for nid, cache in enumerate(cachev):
                        set_idx = np.concatenate(
                            [Didx[(Ntrain + bid + nid * int(Nstride)) % Nblocks] for bid in range(Nvalid)])
                        if set_idx.size != 0:
                            data_count[nid, 1] += set_idx.size
                            #print("Py vDIPOLE1:\n",D[set_idx][0:3],D.shape)
                            #print("Py vDIPOLE2:\n",D[set_idx][-3:],D.shape)
                            #cache.insertdata(X[set_idx], F[set_idx], C[set_idx], D[set_idx], E[set_idx], list(S))
                            #cache.insertdata(X[set_idx], F[set_idx], C[set_idx], D[set_idx], P[set_idx], E[set_idx],Sv[set_idx], list(S))
                            cache.insertdata(X[set_idx], F[set_idx], C[set_idx], D[set_idx], P[set_idx], E[set_idx], list(S))

                    if build_test:
                        for nid, th5 in enumerate(testh5):
                            set_idx = np.concatenate(
                                [Didx[(Ntrain + Nvalid + bid + nid * int(Nstride)) % Nblocks] for bid in range(Ntest)])
                            if set_idx.size != 0:
                                data_count[nid, 2] += set_idx.size
                                #th5.store_data(f + data['path'], coordinates=X[set_idx], forces=F[set_idx], charges=C[set_idx], dipoles=D[set_idx], cell=P[set_idx],energies=E[set_idx], species=list(S))
                                #th5.store_data(f + data['path'], coordinates=X[set_idx], forces=F[set_idx], charges=C[set_idx], dipoles=D[set_idx],
                                th5.store_data(f + data['path'], coordinates=X[set_idx], forces=F[set_idx], charges=C[set_idx], dipoles=D[set_idx], cell=P[set_idx],energies=E[set_idx], species=list(S))
                                #th5.store_data(f + data['path'], coordinates=X[set_idx], forces=F[set_idx], charges=C[set_idx], dipoles=D[set_idx], cell=P[set_idx],energies=E[set_idx],solvent=Sv[set_idx], species=list(S))

        # Save train and valid meta file and cleanup testh5
        for t, v in zip(cachet, cachev):
            t.makemetadata()
            v.makemetadata()

        if build_test:
            for th in testh5:
                th.cleanup()

        print(' Train ', ' Valid ', ' Test ')
        print(data_count)
        print('Training set built.')

    def build_strided_training_cache_ind(self, ids, rseed, Nblocks, Nvalid, Ntest, build_test=True,
                                         Ekey='energies', energy_unit=1.0,
                                         forces=True, grad=False, Fkey='forces', forces_unit=1.0,
                                         dipole=False, dipole_unit=1.0, Dkey='dipoles',
                                         charge=False, charge_unit=1.0, Ckey='charges',
                                         pbc=False,
                                         Eax0sum=False, rmhighe=True,rmhighf=False,force_exact_split=False):
        np.random.seed(rseed)

        if not os.path.isfile(self.netdict['saefile']):
            self.sae_linear_fitting(Ekey=Ekey, energy_unit=energy_unit, Eax0sum=Eax0sum)

        h5d = self.h5dir

        store_dir = self.train_root + "cache-data-"
        N = self.Nn
        Ntrain = Nblocks - Nvalid - Ntest

        if Nblocks % N != 0:
            raise ValueError('Error: number of networks must evenly divide number of blocks.')

        Nstride = Nblocks / N

        if not os.path.exists(store_dir + str(ids)):
            os.mkdir(store_dir + str(ids))

        if build_test:
            if os.path.exists(store_dir + str(ids) + '/../testset/testset' + str(ids) + '.h5'):
                os.remove(store_dir + str(ids) + '/../testset/testset' + str(ids) + '.h5')

            if not os.path.exists(store_dir + str(ids) + '/../testset'):
                os.mkdir(store_dir + str(ids) + '/../testset')

        cachet = cg('_train', self.netdict['saefile'], store_dir + str(ids) + '/', False)
        cachev = cg('_valid', self.netdict['saefile'], store_dir + str(ids) + '/', False)

        if build_test:
            testh5 = pyt.datapacker(store_dir + str(ids) + '/../testset/testset' + str(ids) + '.h5')

        if rmhighe:
            dE = []
            for f in self.h5file:
                adl = pyt.anidataloader(h5d + f)
                for data in adl:
                    S = data['species']
                    E = data[Ekey]
                    X = data['coordinates']

                    Esae = hdt.compute_sae(self.netdict['saefile'], S)

                    dE.append((E - Esae) / np.sqrt(len(S)))

            dE = np.concatenate(dE)
            cidx = np.where(np.abs(dE) < 15.0)
            std = np.abs(dE[cidx]).std()
            men = np.mean(dE[cidx])

            print(men, std, men + std)
            idx = np.intersect1d(np.where(dE >= -np.abs(15 * std + men))[0], np.where(dE <= np.abs(11 * std + men))[0])
            cnt = idx.size
            print('DATADIST: ', dE.size, cnt, (dE.size - cnt), 100.0 * ((dE.size - cnt) / dE.size))

        E = []
        data_count = np.zeros((N, 3), dtype=np.int32)
        for f in self.h5file:
            adl = pyt.anidataloader(h5d + f)
            for data in adl:
                # print(data['path'],data['energies'].size)

                S = data['species']

                if data[Ekey].size > 0 and (set(S).issubset(self.netdict['atomtyp'])):

                    X = np.array(data['coordinates'], order='C', dtype=np.float32)

                    if Eax0sum:
                        E = energy_unit * np.sum(np.array(data[Ekey], order='C', dtype=np.float64), axis=1)
                    else:
                        E = energy_unit * np.array(data[Ekey], order='C', dtype=np.float64)

                    Sv = np.zeros((E.size,7),dtype=np.float32)
                    if solvent:
                        Sv = np.array(data['solvent'], order='C', dtype=np.float32)

                    if forces and not grad:
                        F = forces_unit * np.array(data[Fkey], order='C', dtype=np.float32)
                    elif forces and grad:
                        F = -forces_unit * np.array(data[Fkey], order='C', dtype=np.float32)
                    else:
                        F = 0.0 * X

                    D = np.zeros((E.size,3),dtype=np.float32)
                    if dipole:
                        D = dipole_unit * np.array(data[Dkey], order='C', dtype=np.float32).reshape(E.size,3)
                    else:
                        D = 0.0 * D

                    P = np.zeros((E.size,3,3),dtype=np.float32)
                    if pbc:
                        P = np.array(data['cell'], order='C', dtype=np.float32).reshape(E.size,3,3)
                    else:
                        P = 0.0 * P

                    C = np.zeros((E.size,X.shape[1]),dtype=np.float32)
                    if charge:
                        C = charge_unit * np.array(data[Ckey], order='C', dtype=np.float32).reshape(E.size,len(S))
                    else:
                        C = 0.0 * C

                    if rmhighe:
                        Esae = hdt.compute_sae(self.netdict['saefile'], S)

                        ind_dE = (E - Esae) / np.sqrt(len(S))

                        hidx = np.union1d(np.where(ind_dE < -(15.0 * std + men))[0],
                                          np.where(ind_dE > (11.0 * std + men))[0])

                        lidx = np.intersect1d(np.where(ind_dE >= -(15.0 * std + men))[0],
                                              np.where(ind_dE <= (11.0 * std + men))[0])

                        if hidx.size > 0:
                            print('  -(' + f + ':' + data['path'] + ') High energies detected:\n    ',
                                  (E[hidx] - Esae) / np.sqrt(len(S)))

                        X = X[lidx]
                        E = E[lidx]
                        F = F[lidx]
                        D = D[lidx]
                        C = C[lidx]
                        P = P[lidx]
                        Sv = Sv[lidx]

                    if rmhighf:
                        hfidx = np.where(np.abs(F) > rmhighf)
                        if hfidx[0].size > 0:
                            print('High force:',hfidx)
                            hfidx = np.all(np.abs(F).reshape(E.size,-1) <= rmhighf,axis=1)
                            X = X[hfidx]
                            E = E[hfidx]
                            F = F[hfidx]
                            D = D[hfidx]
                            C = C[hfidx]
                            P = P[hfidx]
                            Sv = Sv[hfidx]

                    # Build random split index
                    ridx = np.random.randint(0, Nblocks, size=E.size)
                    Didx = [np.argsort(ridx)[np.where(ridx == i)] for i in range(Nblocks)]

                    # Build training cache
                    #for nid, cache in enumerate(cachet):

                    set_idx = np.concatenate(
                        [Didx[((bid + ids * int(Nstride)) % Nblocks)] for bid in range(Ntrain)])
                    if set_idx.size != 0:
                        data_count[ids, 0] += set_idx.size
                        cachet.insertdata(X[set_idx], F[set_idx], C[set_idx], D[set_idx], P[set_idx], E[set_idx], Sv[set_idx], list(S))

                    #for nid, cache in enumerate(cachev):

                    set_idx = np.concatenate(
                        [Didx[(Ntrain + bid + ids * int(Nstride)) % Nblocks] for bid in range(Nvalid)])
                    if set_idx.size != 0:
                        data_count[ids, 1] += set_idx.size
                        cachev.insertdata(X[set_idx], F[set_idx], C[set_idx], D[set_idx], P[set_idx], E[set_idx], Sv[set_idx], list(S))

                    if build_test:
                        #for nid, th5 in enumerate(testh5):

                        set_idx = np.concatenate(
                            [Didx[(Ntrain + Nvalid + bid + ids * int(Nstride)) % Nblocks] for bid in range(Ntest)])
                        if set_idx.size != 0:
                            data_count[ids, 2] += set_idx.size
                            testh5.store_data(f + data['path'], coordinates=X[set_idx], forces=F[set_idx], charges=C[set_idx], dipoles=D[set_idx], cell=P[set_idx], energies=E[set_idx], solvent=Sv[set_idx], species=list(S))

        # Save train and valid meta file and cleanup testh5
        cachet.makemetadata()
        cachev.makemetadata()

        if build_test:
            testh5.cleanup()

        #print(' Train ', ' Valid ', ' Test ')
        #print(data_count[ids])
        #print(ids,'Training set built.')

    def train_ensemble(self, GPUList, remove_existing=False):
        print('Training Ensemble...')
        processes = []
        indicies = np.array_split(np.arange(self.Nn), len(GPUList))
        seeds = np.array_split(np.random.randint(low=0,high=2**32,size=self.Nn), len(GPUList))
        for gpu, (idc,seedl) in enumerate(zip(indicies,seeds)):
            processes.append(Process(target=self.train_network, args=(GPUList[gpu], idc, seedl, remove_existing)))
            processes[-1].start()
            # self.train_network(pyncdict, trdict, layers, id, i)

        for p in processes:
            p.join()
        print('Training Complete.')

    def train_ensemble_single(self, gpuid, ntwkids, remove_existing=False, random_seed = 0):
        print('Training Single Model From Ensemble...')

        np.random.seed(random_seed)
        random_seeds = np.random.randint(0,2**32,size=len(ntwkids))
        self.train_network(gpuid, ntwkids, random_seeds, remove_existing)

        print('Training Complete.')

    def train_network(self, gpuid, indicies, seeds, remove_existing=False):
        for index,seed in zip(indicies,seeds):
            pyncdict = dict()
            pyncdict['wkdir'] = self.train_root + 'train' + str(index) + '/'
            pyncdict['ntwkStoreDir'] = self.train_root + 'train' + str(index) + '/' + 'networks/'
            pyncdict['datadir'] = self.train_root + "cache-data-" + str(index) + '/'
            pyncdict['gpuid'] = str(gpuid)

            if not os.path.exists(pyncdict['wkdir']):
                os.mkdir(pyncdict['wkdir'])

            if remove_existing:
                shutil.rmtree(pyncdict['ntwkStoreDir'])

            if not os.path.exists(pyncdict['ntwkStoreDir']):
                os.mkdir(pyncdict['ntwkStoreDir'])

            outputfile = pyncdict['wkdir'] + 'output.opt'

            ibuild = copy.deepcopy(self.iptbuilder)
            ibuild.set_parameter('seed',str(seed))

            nfile = pyncdict['wkdir']+'inputtrain.ipt'
            ibuild.write_input_file(nfile,iptsize=self.netdict["iptsize"])

            shutil.copy2(self.netdict['cnstfile'], pyncdict['wkdir'])
            shutil.copy2(self.netdict['saefile'], pyncdict['wkdir'])

            if "/" in nfile:
                nfile = nfile.rsplit("/", 1)[1]

            command = "cd " + pyncdict['wkdir'] + " && HDAtomNNP-Trainer -i " + nfile + " -d " + pyncdict[
                'datadir'] + " -p 1.0 -m -g " + pyncdict['gpuid'] + " > output.opt"
            proc = subprocess.Popen(command, shell=True)
            proc.communicate()

            if 'Termination Criterion Met!' not in open(pyncdict['wkdir']+'output.opt','r').read():
                with open(pyncdict['wkdir']+"output.opt",'a+') as output:
                    output.write("\n!!!TRAINING FAILED TO COMPLETE!!!\n")

            print('  -Model', index, 'complete')

