__author__ = 'jujuman'

import numpy as np
#import statsmodels.api as sm
import re
import os.path
import math
import time as tm
import linecache
import pandas as pd

hatokcal = 627.509469
hatoev = 27.21138602
evtokcal = 23.06054195
AtoBohr = 1.88973

convert = hatokcal  # Ha to Kcal/mol

def get_na_from_xyz(f):
    o = open(f,'r')
    value = int(o.readline())
    o.close()
    return value

def convertatomicnumber(X):
    X = int(X)
    if X == 1:
        return 'H'
    elif X == 6:
        return 'C'
    elif X == 7:
        return 'N'
    elif X == 8:
        return 'O'
    elif X == 16:
        return 'S'

def get_spc_idx(X1, X2):
    if X1 == 'H' and X2 == 'H':
        return 0 # HH
    elif (X1 == 'H' and X2 == 'C') or (X1 == 'C' and X2 == 'H'):
        return 1 # HC
    elif (X1 == 'H' and X2 == 'N') or (X1 == 'N' and X2 == 'H'):
        return 2 # HN
    elif (X1 == 'H' and X2 == 'O') or (X1 == 'O' and X2 == 'H'):
        return 3 # HO
    elif X1 == 'C' and X2 == 'C':
        return 4 # CC
    elif (X1 == 'C' and X2 == 'N') or (X1 == 'N' and X2 == 'C'):
        return 5 # CN
    elif (X1 == 'C' and X2 == 'O') or (X1 == 'O' and X2 == 'C'):
        return 6 # CO
    elif X1 == 'O' and X2 == 'O':
        return 7 # OO
    elif (X1 == 'O' and X2 == 'N') or (X1 == 'N' and X2 == 'O'):
        return 8 # ON
    elif X1 == 'N' and X2 == 'N':
        return 9 # NN


class ncdata():
    def __init__(self, mat, spc, Na):
        self.mat = mat
        self.spc = spc

        self.num_data  = mat.shape[0]
        self.num_atoms = Na

def buckingham_pot(x, *p):
    cp = (0.1645, 0.74,# HH
          0.1573, 1.09,# HC
          0.1489, 1.01,# HN
          0.1779, 0.96,# HO
          0.2339, 1.20,# CC
          0.2342, 1.16,# CN
          0.2838, 1.13,# CO
          0.0556, 1.21,# OO
          0.1143, 1.21,# ON
          0.1592, 1.10,)# NN

    s = []
    print('Params: ', p)
    for i in x:
        E = np.zeros(i.num_data)

        for e, R in enumerate(i.mat):
            didx = 0
            for A1 in range(i.num_atoms):
                for A2 in range(A1+1, i.num_atoms):
                    #print(get_spc_idx(i.spc[A1], i.spc[A2]), X[didx])
                    sidx = get_spc_idx(i.spc[A1], i.spc[A2])
                    #E[e] = E[e] + -p[2*sidx] * (np.exp(-X[didx]/p0[2*sidx+1]) - np.power(p0[2*sidx+1]/X[didx], 6)) + p[2*sidx+1]

                    De = cp[2*sidx]
                    r0 = cp[2*sidx+1]
                    a = p[2*sidx]
                    S = p[2*sidx+1]

                    E[e] = E[e] + De * (1.0 - np.exp(-a * AtoBohr * (R[didx] - r0)))**2 + S

                    didx = didx + 1
        #print(E)
        s.append(E)

    return np.concatenate(s)

# ------------------------------------------
#          Radial Function Cos
# ------------------------------------------
def cutoffcos(X,Rc):
    Xt = X

    if Xt > Rc:
        Xt = Rc

    return 0.5 * (np.cos((np.pi * Xt)/Rc) + 1.0)

def src_pot(x):
    cp = (0.1645, 0.50,# HH
          0.1573, 0.50,# HC
          0.1489, 0.50,# HN
          0.1779, 0.50,# HO
          15.0, 0.96,# CC
          0.2342, 0.50,# CN
          0.2838, 0.50,# CO
          0.0556, 0.50,# OO
          0.1143, 0.50,# ON
          0.1592, 0.50,)# NN

    s = []
    for i in x:
        E = np.zeros(i.num_data)

        for e, R in enumerate(i.mat):
            didx = 0
            for A1 in range(i.num_atoms):
                for A2 in range(A1+1, i.num_atoms):
                    #print(get_spc_idx(i.spc[A1], i.spc[A2]), X[didx])
                    sidx = get_spc_idx(i.spc[A1], i.spc[A2])
                    #E[e] = E[e] + -p[2*sidx] * (np.exp(-X[didx]/p0[2*sidx+1]) - np.power(p0[2*sidx+1]/X[didx], 6)) + p[2*sidx+1]

                    H = cp[2*sidx]
                    Rc = cp[2*sidx+1]

                    E[e] = E[e] + H * cutoffcos(R[didx],Rc)
                    didx = didx + 1
        #print(E)
        s.append(E)

    return np.concatenate(s)

def readxyz (file):
    xyz = []
    typ = []
    Na  = []

    fd = open(file, 'r').read()

    #rb = re.compile('\s*?\n?\s*?(\d+?)\s*?\n((?:\s*?[A-Z][a-z]?.+(?:\n|))+)')
    rb = re.compile('(\d+)[\s\S]+?(?=[A-Z])((?:\s*?[A-Z][a-z]?\s+[-+]?\d+?\.\d+?\s+?[-+]?\d+?\.\d+?\s+?[-+]?\d+?\.\d+?\s.+(?:\n|))+)')
    ra = re.compile('([A-Z][a-z]?)\s+?(\S+?)\s+?(\S+?)\s+?(\S+)')

    s = rb.findall(fd)

    for i in s:
        Na.append(int(i[0]))
        atm = ra.findall(i[1])

        print(atm)

        ntyp = []
        nxyz = []
        for j in range(0, int(i[0])):
            ntyp.append(atm[j][0])
            nxyz.append(float(atm[j][1]))
            nxyz.append(float(atm[j][2]))
            nxyz.append(float(atm[j][3]))

        xyz.append(nxyz)
        typ.append(ntyp)

    xyz = np.asarray(xyz,dtype=np.float32)
    xyz = xyz.reshape((xyz.shape[0],len(typ[0]),3))

    return xyz,typ[0],Na

def readxyz2 (file):
    xyz = []
    typ = []
    Na  = []
    ct = []

    fd = open(file, 'r').read()

    #print(fd)

    #rb = re.compile('\s*?\n?\s*?(\d+?)\s*?\n((?:\s*?[A-Z][a-z]?.+(?:\n|))+)')
    #rb = re.compile('((?:[A-Z][a-z]? +?[-+]?\d+?\.\S+? +?[-+]?\d+?\.\S+? +?[-+]?\d+?\.\S+?\s*?(?:\n|$))+)')
    rb = re.compile('(\d+?)\n(.*?)\n((?:[A-Z][a-z]?.+?(?:\n|$))+)')
    #rb = re.compile('(\d+?)(\n)((?:[A-Z][a-z]?.+?(?:\n|$))+)')
    ra = re.compile('([A-Z][a-z]?)\s+?([-+]?\d+?\.\S+?)\s+?([-+]?\d+?\.\S+?)\s+?([-+]?\d+?\.\S+?)\s*?(?:\n|$)')

    s = rb.findall(fd)
    Nc = len(s)
    if Nc == 0:
        raise ValueError('No coordinates found in file. Check formatting of '+file+'.')

    for i in s:
        ct.append(i[1])

        c = ra.findall(i[2])
        Na = len(c)
        for j in c:
            typ.append(j[0])
            xyz.append(j[1])
            xyz.append(j[2])
            xyz.append(j[3])

    xyz = np.asarray(xyz,dtype=np.float32)
    xyz = xyz.reshape((Nc,Na,3))

    return xyz,typ[0:Na],Na, ct

def readxyz3 (file):           #XYZ file reader for RXN
	xyz = []
	typ = []
	Na  = []
	ct = []
	fd = open(file, 'r').read()
	rb = re.compile('(\d+?)\n(.*?)\n((?:[A-Z][a-z]?.+?(?:\n|$))+)')
	ra = re.compile('([A-Z][a-z]?)\s+?([-+]?\d+?\.\S+?)\s+?([-+]?\d+?\.\S+?)\s+?([-+]?\d+?\.\S+?)\s*?(?:\n|$)')
	s = rb.findall(fd)
	Nc = len(s)
	if Nc == 0:
       		raise ValueError('No coordinates found in file. Check formatting of '+file+'.')
	for i in s:
		X=[]
		T=[]
		ct.append(i[1])
		c = ra.findall(i[2])
		Na.append(len(c))
		for j in c:
			T.append(j[0])
			X.append(j[1])
			X.append(j[2])
			X.append(j[3])
		X=np.array(X, dtype=np.float32)
		X=X.reshape(len(T),3)
		xyz.append(X)
		typ.append(T)

	return xyz,typ,Na,ct


def readrcdbstruct (file):
    xyz = []
    typ = []
    Na  = []

    fd = open(file, 'r').read()

    ro = re.compile('optimize=\s*?(\d+?)\s*?[!|\n]')
    o = ro.findall(fd)
    opt = bool(int(o[0]))

    ra = re.compile('\$coordinates\s*?\n([\s\S]+?)&')
    rb = re.compile('([A-Z][a-z]?)\s+?[A-Z][a-z]?\s+?([+,-]?\d+?\.\d+?)\s+?([+,-]?\d+?\.\d+?)\s+?([+,-]?\d+?\.\d+?)\s*?(\n|$)')

    s = ra.findall(fd)[0]
    c = rb.findall(s)

    Na = len(c)
    for l in c:
        typ.append(l[0])
        xyz.append(float(l[1]))
        xyz.append(float(l[2]))
        xyz.append(float(l[3]))

    xyz = np.asarray(xyz,dtype=np.float32)
    xyz = xyz.reshape((Na,3))

    return xyz,typ[0:Na],Na,opt

def writexyzfile (fn,xyz,typ,cmt=''):
    f = open(fn, 'w')
    N = len(typ)
    #print('N ATOMS: ',typ)
    for m in xyz:
        f.write(str(N)+'\n comment:' + cmt + '\n')
        #print(m)
        for i in range(N):
            x = m[i,0]
            y = m[i,1]
            z = m[i,2]
            f.write(typ[i] + ' ' + "{:.7f}".format(x) + ' ' + "{:.7f}".format(y) + ' ' + "{:.7f}".format(z) + '\n')
        #f.write('\n')
    f.close()

def writexyzfilewc (fn,xyz,typ,cmt):
    f = open(fn, 'w')
    N = len(typ)
    #print('N ATOMS: ',typ)
    for m,c in zip(xyz,cmt):
        f.write(str(N)+'\n' + c + '\n')
        #print(m)
        for i in range(N):
            x = m[i,0]
            y = m[i,1]
            z = m[i,2]
            f.write(typ[i] + ' ' + "{:.7f}".format(x) + ' ' + "{:.7f}".format(y) + ' ' + "{:.7f}".format(z) + '\n')
        #f.write('\n')
    f.close()

def readncdatall(file,N = 0):
    rdat = dict()
    if os.path.isfile(file):
        fd = open(file, 'r').read()
        fd = fd.split("\n",3)
        lot = fd[0]
        nda = int(fd[1])
        nat = int(fd[2].split(",")[0])
        spc = fd[2].split(",")[1:-1]

        elm = nat*3*3 + nat*4 + 1

        rdat.update({'lot':[lot],'species':spc})

        np.empty((nda,elm))

        Xi = []
        Ei = []
        Fi = []
        C1 = []
        C2 = []
        SD = []
        DP = []
        #print('File: ',file)
        for i,l in enumerate(fd[3].split("\n")[0:-1]):
            
            data = l.split(",")[0:-1]
            #if True:
            if len(data) == 3*3*nat+1+3*nat+3+3 or len(data) == 3*3*nat+1+3*nat+3+3+2:
               if len(data) == 3*3*nat+1+3*nat+3+3+2:
                  data = data[0:-2]
               #print(i,np.array(data).shape,3*3*nat+1+3*nat+3+3,l.count(","))
               Xi.append(np.array(data[0:nat * 3], dtype=np.float32).reshape(nat,3))
               Ei.append(data[nat*3])
               Fi.append(np.array(data[nat*3+1:nat*3+nat*3+1],dtype=np.float32).reshape(nat,3))
               C1.append(np.array(data[nat*3+nat*3+1:nat*3+nat*3+(nat+1)+1],dtype=np.float32))
               C2.append(np.array(data[nat*3+nat*3+(nat+1)+1:nat*3+nat*3+2*(nat+1)+1],dtype=np.float32))
               SD.append(np.array(data[nat*3+nat*3+2*(nat+1)+1:nat*3+nat*3+3*(nat+1)+1],dtype=np.float32))
               DP.append(np.array(data[nat*3+nat*3+3*(nat+1)+1:nat*3+nat*3+4*(nat+1)+(nat+1)*3+1],dtype=np.float32).reshape(nat+1,3))
               #DP.append(np.array(data[nat*3+nat*3+3*(nat+1)+1:nat*3+nat*3+4*(nat+1)+(nat+1)*3+1],dtype=np.float32).reshape(nat+1,3))
            else:
               print(i,np.array(data).shape,3*3*nat+1+3*nat+3+3,l.count(","))
               print('Line size does not match expected!')
               print('File:',file)
    else:
        exit(FileNotFoundError)

    if len(Xi) > 0:
        Xi = np.stack(Xi)
        Ei = np.array(Ei,dtype=np.float64)
        Fi = np.stack(Fi)
        C1 = np.stack(C1)
        C2 = np.stack(C2)
        SD = np.stack(SD)
        DP = np.stack(DP)
        rdat.update({'coordinates':Xi,
                     'energies':Ei,
                     'forces':Fi,
                     'hirshfeld':C1,
                     'cm5':C2,
                     'spindensities':SD,
                     'hirdipole':DP})

    return rdat



def readncdatwforce (file,N = 0):
    xyz = []
    typ = []
    frc = []
    Eact = []

    readf = False

    if os.path.isfile(file):
        readf = True

        fd = open(file, 'r')

        fd.readline()
        fd.readline()

        types=fd.readline().split(",")

        Na = int(types[0])
        typ = types[1:Na+1]

        cnt = 0

        for i in fd.readlines():
            cnt += 1
            sd = i.strip().split(",")
            #print(sd)
            xyz.append(list(map(float, sd[0:3*Na])))
            Eact.append(float(sd[3*Na]))
            frc.append(list(map(float,sd[3*Na+1:2*3*Na+1])))
            if cnt >= N and N > 0:
                break

    Eact = np.array(Eact, dtype = np.float64)
    xyz = np.array(xyz,dtype=np.float32).reshape(Eact.size,Na,3)
    frc = np.array(frc,dtype=np.float32).reshape(Eact.size,Na,3)


    return xyz,frc,typ,Eact,readf

def readncdat (file,type = np.float):

    if os.path.isfile(file):

        fd = open(file, 'r')

        fd.readline()
        Nconf = int(fd.readline())

        types=fd.readline().split(",")
        Na = int(types[0])
        spc = np.asarray(types[1:Na+1],dtype=bytes)

        if Nconf > 0:
            data = np.loadtxt(fd, delimiter=',',usecols=range(Na*3+1),dtype=type)

            if Nconf is 1:
                data = data.reshape(1,Na*3+1)
        else:
            data = np.empty((0,Na*3+1))

        if Nconf != data.shape[0]:
            print('Error: shapes dont match for file: ',file)
            print('Sizes: ',Nconf, '!=', data.shape[0])
            print('Data: ', data)
            exit(1)

        xyz    = data[:Nconf,0:3*Na].reshape(Nconf,Na,3).copy()
        energy = data[:Nconf,3*Na].flatten()

        return [xyz,spc,energy]
    else :
        raise (FileNotFoundError("File not found: " + file))

def writencdat (file,xyz,spec,energy,comment=''):
    f = open(file,'w')
    f.write(comment+'\n')
    f.write(str(xyz.shape[0])+'\n')
    f.write(str(len(spec)) + ',')
    for i in spec:
        f.write(i + ',')
    f.write('\n')

    for m,e in zip(xyz,energy):
        for a in m:
            for c in a:
                f.write("{:.7f}".format(c) + ',')
        f.write("{:.7f}".format(e) + ',\n')
    f.close()



def readg09trajdat (file,type):
    xyz = []
    typ = []
    Enr = []

    fd = open(file, 'r').read()

    # Get energies
    rE = re.compile('SCF Done:\s+?E\(\S+\)\s+?=\s+?([+,-]?\d+?\.\d+?E?[+,-]?\d+?)\s')
    s = rE.findall(fd)

    for i in s:
        Enr.append(float(i))

    # Get coords
    rB = re.compile('Input orientation:([\S\s]+?)(?=Distance|Rotational)')
    b = rB.findall(fd)

    for i in b:
        rX = re.compile('\s+?\d+?\s+?(\d+?)\s+?\d+?\s+?([+,-]?\d+?\.\d+?)\s+?([+,-]?\d+?\.\d+?)\s+?([+,-]?\d+?\.\d+?)\s')
        c = rX.findall(i)

        t_xyz = []
        t_typ = []

        for j in c:
            t_typ.append(convertatomicnumber(j[0]))
            t_xyz.append(float(j[1]))
            t_xyz.append(float(j[2]))
            t_xyz.append(float(j[3]))

        typ.append(t_typ)
        xyz.append(t_xyz)

    #typ.pop(len(typ)-1)
    #xyz.pop(len(xyz)-1)
    #typ.pop(0)
    #xyz.pop(0)

    Enr.pop(0)
    #typ.pop(0)
    #xyz.pop(0)

    #print (len(typ[0]))
    xyz = np.asarray(xyz,dtype=type)
    xyz = xyz.reshape((xyz.shape[0],len(typ[0]),3))

    Enr = np.asarray(Enr,dtype=type)
    return xyz,typ,Enr

# -----------------------
# readfile into np array
# -----------------------
def getfltsfromfile(file, delim, cols):
    # Open and read the data file
    infile = open(file, 'r')

    infile_s = []

    for line in infile:
        row = line.strip().split(delim)
        infile_s.append(row)

    # Truncate and convert to numpy array
    nparray = np.array(infile_s)
    data = nparray[:, cols]
    data = np.array(data, dtype='f8')
    return data

def getfltsfromfileprob(file, delim, col1, col2, prob):
    # Open and read the data file
    infile = open(file, 'r')

    infile_s = []

    for line in infile:
        if np.random.binomial(1,prob):
            row = line.strip().split(delim)
            infile_s.append(row)

    # Truncate and convert to numpy array
    nparray = np.array(infile_s)
    data1 = nparray[:, col1]
    data2 = nparray[:, col2]

    data1 = np.array(data1, dtype='f8')
    data2 = np.array(data2, dtype='f8')
    return data1,data2

# -----------------------

def generatedmat(crds,Na):
    dmat = []

    for i in range(0,Na):
        for j in range(i+1, Na):
            dmat.append(((crds[i*3] - crds[j*3])**2+(crds[i*3+1] - crds[j*3+1])**2+(crds[i*3+2] - crds[j*3+2])**2)**0.5)

    return np.array(dmat)

def generatedmat(crds,Na):
    dmat = []

    for i in range(0,Na):
        for j in range(i+1, Na):
            dmat.append(((crds[i*3] - crds[j*3])**2+(crds[i*3+1] - crds[j*3+1])**2+(crds[i*3+2] - crds[j*3+2])**2)**0.5)

    return dmat

def generatedmats(crds,Na):
    print(crds.shape[0])
    dmat = np.zeros([crds.shape[0],int((Na*(Na+1))/2)],np.float)

    for a in dmat:
        count = 0
        for i in range(0,Na):
            for j in range(i+1, Na):
                a[count] = ((crds[i*3] - crds[j*3])**2+(crds[i*3+1] - crds[j*3+1])**2+(crds[i*3+2] - crds[j*3+2])**2)**0.5
                count += 1

    return dmat

def generatedmatsd3(crds):
    Na = crds.shape[1]
    dmat = np.zeros([crds.shape[0],int((Na*(Na-1))/2)],np.float)

    for s,a in enumerate(dmat):
        count = 0
        for i in range(0,Na):
            for j in range(i+1, Na):
                a[count] = np.linalg.norm(crds[s,i]-crds[s,j])
                #print(a[count],crds[s,i],crds[s,i],crds[s,i]-crds[s,j])
                count += 1

    return dmat

def compute_sae(file, spc):
    f = open(file,'r').read()

    p = re.compile('([A-Z][a-z]?),\d+?=(.+?)(?=\n|$)')
    m = dict(re.findall(p,f))

    sae = 0.0
    for s in spc:
        sae = sae + float(m[s])

    return sae



# ----------------------------
# Calculate Mean Squared Diff
# ----------------------------

def calculatemeansqrerror(data1, data2):
    data = np.power(data1 - data2, 2)
    return np.mean(data)

def calculaterootmeansqrerror(data1, data2, axis=0):
    data = np.power(data1 - data2, 2)
    return np.sqrt(np.mean(data, axis=axis))

def calculatemeanabserror(data1, data2, axis=0):
    data = np.abs(data1 - data2)
    return np.mean(data,axis=axis)

# ----------------------------
# Calculate Mean Squared Diff
# ----------------------------

def calculatenumderiv(data1, dx):
    C = np.shape(data1)[0]
    data = np.zeros((C, 2))
    for i in range(1, C-1):
        data[i,0] = i
        data[i,1] = (data1[i-1] - data1[i+1]) / (2.0*dx)

    return data

def calculateelementdiff(data1):
    C = np.shape(data1)[0]
    data = np.zeros((C-1, 2))
    for i in range(0, C-1):
        data[i,0] = i
        data[i,1] = (data1[i] - data1[i+1])

    return data

def calculateelementdiff2D(data1):
    C = np.shape(data1)[0]
    x = np.zeros((C*C))
    y = np.zeros((C*C))
    z = np.zeros((C*C))
    d = np.zeros(int(C*(C-1)/2))
    cnt = 0
    for i in range(0, C):
        for j in range(0, C):
            x[i+j*C] = i
            y[i+j*C] = j
            z[i+j*C] = (data1[i] - data1[j])
            if i > j:
                d[cnt] = z[i+j*C]
                cnt += 1

    return x,y,z,d

def calculatecompareelementdiff2D(data1,data2):
    C = np.shape(data1)[0]
    x = np.zeros(C*C)
    y = np.zeros(C*C)
    z = np.zeros(C*C)
    d = np.zeros(C*(C+1)/2)
    cnt = 0
    for i in range(0, C):
        for j in range(0, C):

            if i > j:
                x[i+j*C] = i
                y[i+j*C] = j
                z[i+j*C] = (data1[i] - data1[j])

            if j > i:
                x[i+j*C] = i
                y[i+j*C] = j
                z[i+j*C] = -(data2[i] - data2[j])

            if j == i:
                x[i+j*C] = i
                y[i+j*C] = j
                z[i+j*C] = 0.0

            if i > j:
                d[cnt] = z[i+j*C]
                cnt += 1

    return x,y,z,d

def calculatemean(data1):
    C = np.shape(data1)[0]
    sum = 0.0
    for i in data1:
        sum += i

    return sum/float(C)

def calculateabsdiff(data1):
    C = int(np.shape(data1)[0]/2)
    data = np.zeros((C, 2))
    for i in range(1, C-1):
        data[i,0] = i
        data[i,1] = data1[i*2+1] - data1[i*2]

    return data

def nsum(n):
    return int(( n * ( n - 1 ) ) / 2)

def index_triangle(i, j, n):
    if j < i:
        return int(nsum(n) - nsum(n-j) + i - j - 1)
    else:
        return int(nsum(n) - nsum(n-i) + j - i - 1)

def calculatedmat(data):
    N = data.size
    d = np.empty(nsum(N))

    for i in range(N):
        for j in range(i+1,N):
            idx = index_triangle(i,j,N)
            d[idx] = data[i] - data[j]

    return d

def calculateKdmat(K, data):
    N = data.shape[1]
    d = np.empty((K, nsum(N)))

    for k in range(K):
        for i in range(N):
            for j in range(i+1,N):
                idx = index_triangle(i,j,N)
                d[k,idx] = data[k,i] - data[k,j]

    return d

def calculateKdmat_sm(K, data, sigma):
    N = data.shape[1]
    d = np.empty((K, nsum(N)))
    s = np.empty((K, nsum(N)))

    for k in range(K):
        for i in range(N):
            for j in range(i+1,N):
                idx = index_triangle(i,j,N)
                d[k,idx] = data[k,i] - data[k,j]
                s[k,idx] = np.max([sigma[k,i],sigma[k,j]])

    return d,s

# -----------------------

# ----------------------------
# Linear-ize Data
# ----------------------------
def makedatalinear(datain):
    data = np.zeros((np.shape(datain)[0]*np.shape(datain)[1],1))
    #data = datain[:,0]

    C = np.shape(datain)[0]
    R = np.shape(datain)[1]
    print (C, "X", R)

    i = j = 0
    for i in range(0, C):
        for j in range(0, R):
            data[i*3+j] = datain[i, j]

    return data

def to_precision(x,p):
    """
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)

def read_rcdb_coordsandnm(file):
    f = open(file,'r').read()

    rc = re.compile("(?:\$coordinates)\s*?\n([\s\S]+?)&")
    prm = re.compile("(.+?)=(.+?)(?: |\n)")
    
    output = dict()
    for p in prm.findall(f):
        output[p[0]] = p[1]
    
    S = rc.search(f)

    if S:
        elements = np.vstack([np.array(list(filter(None,i.split(" ")))) for i in S.group(1).split("\n")][0:-1])
        spc = [str(e) for e in elements[:, 0]]
        xyz = np.array(elements[:,2:5],dtype=np.float32)
        output.update({"coordinates":xyz,"species":spc})

    rn = re.compile("FREQUEN=(.+)\nREDMASS=(.+)\nFRCCNST=(.+)\s*?{\s*?\n([\s\S]+?)}")
    M = rn.findall(f)

    if M:
        freq = np.empty(len(M) ,dtype=np.float32)
        rmas = np.empty(len(M), dtype=np.float32)
        frcn = np.empty(len(M), dtype=np.float32)

        nmcd = []
        for i,m in enumerate(M):
            freq[i] = float(m[0])
            rmas[i] = float(m[1])
            frcn[i] = float(m[2])
            nmcd.append(np.vstack([np.array(list(filter(None, x.split(" "))),dtype=np.float32) for x in [s for s in m[3].split("\n") if s != '']]))

        nmcd = np.stack(nmcd)
        output.update({"frequency":freq,"reducedmass":rmas,"forceconstant":frcn,"nmdisplacements":nmcd})

    return output

def write_rcdb_input (xyz,typ,Nc,wkdir,fpf,LOT,modes=None,mode_props=None,charge='0',multip='1',TSS=10,Temp='300.0',rdm='uniform',type='nmrandom',SCF='Tight',freq='1',opt='1',fill=1,comment=""):

    f = open(wkdir + 'inputs/' + fpf + '-' + str(Nc).zfill(fill) + '.ipt', 'w')

    Na = len(typ)

    # ---------- Write Input Variables ------------
    dfname = fpf + '-' + str(Nc) + '_train.dat'
    vdfname = fpf + '-' + str(Nc) + '_valid.dat'
    edfname = fpf + '-' + str(Nc) + '_test.dat'

    V = 6
    if Na is 2:
        V = 5

    DOF = (3 * Na - V)

    f.write('TSS=' + str(int(TSS * DOF)) + ' \n')
    f.write('VSS=' + str(int((TSS * DOF) / 10.0)) + ' \n')
    f.write('ESS=' + str(int((TSS * DOF) / 10.0)) + ' \n')
    f.write('LOT=' + LOT + ' \n')
    f.write('rdm=' + rdm + '\n')
    f.write('type=' + type + '\n')
    f.write('Temp=' + Temp + '\n')
    f.write('mem=' + '16000' + '\n')
    f.write('SCF=' + SCF + '\n')
    f.write('dfname=' + dfname + ' \n')
    f.write('vdfname=' + vdfname + ' \n')
    f.write('edfname=' + edfname + ' \n')
    f.write('optimize='+ opt + ' \n')
    f.write('frequency='+ freq + ' \n')
    f.write('charge='+ charge + ' \n')
    f.write('multip='+ multip + ' \n')

    f.write('\n#'+comment+'\n')

    f.write('\n\n')
    f.write('$coordinates\n')
    for x, t in zip(xyz,typ):
        f.write(' ' + t + ' ' + t + ' ' + "{:.7f}".format(x[0]) + ' ' + "{:.7f}".format(x[1]) + ' ' + "{:.7f}".format(x[2]) + '\n')
    f.write('&\n\n')

    f.write('$connectivity\n')
    f.write(' NONE\n')
    f.write('&\n\n')

    f.write('$normalmodes\n')
    if modes is None:
        f.write('  NONE\n')
    else:
        for c,p in zip(modes,mode_props):
            f.write('FREQUEN='+"{:.7E}".format(p[0]) + '\n')
            f.write('REDMASS='+"{:.7E}".format(p[1]) + '\n')
            f.write('FRCCNST='+"{:.7E}".format(p[2])+' {\n')
            for x in c:
                f.write(' '+"{:.7E}".format(x[0])+' '+"{:.7E}".format(x[1])+' '+"{:.7E}".format(x[2])+'\n')

            f.write('}\n')
    f.write('&\n\n')

