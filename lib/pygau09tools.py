import numpy as np
import re
import os

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

def read_charge (file, type='mulliken'):

    fd = open(file, 'r').read()

    if type == 'mulliken':
        ra = re.compile('(?:Mulliken atomic charges:)\n +?\d\n([\s\S]+?)(?:Sum)')
    elif type == 'esp':
        ra = re.compile('(?:Fitting point charges to electrostatic potential)\n.+?\n.+?\n +?\d\n([\s\S]+?)(?:Charges)')

    rb = re.compile(' +?\d+? +?([A-Z][a-z]?) +?([+-]?\d+?\.\S+?)\n')

    species = []
    charges = []
    block = ra.findall(fd)
    for b in block:
        lines = rb.findall(b)
        for l in lines:
            species.append(l[0])
            charges.append(l[1])

    return charges, species

def read_irc (file):
    f = open(file,'r').read()

    ty = []
    en = []
    cd = []

    r1 = re.compile('Corrected End Point Energy =\s+?([+-]?\d+?\.\d+?)\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n\s+?(?:Delta-x Convergence Met)')
    r2 = re.compile('Cartesian Coordinates \(Ang\):.+\n.+\n.+\n.+\n.+\n([\S\s]+?)(?:---)')
    r3 = re.compile('\d+?\s+?(\d+?)\s+?(\S+?)\s+?(\S+?)\s+?(\S+)')
    r32 = re.compile('\d+?\s+?(\d+?)\s+?\d+?\s+?(\S+?)\s+?(\S+?)\s+?(\S+)')
    r4 = re.compile('(?:Point Number)\s+?(\d+?)\s+?(?:in FORWARD path direction.)')
    rRc= re.compile('\s+?\d+?\s+?([-+]?\d+?\.\d+?)\s+?([-+]?\d+?\.\d+?)\s*?\n')

    r1s = re.compile('SCF Done:\s+?E\(.+?\)\s+?=\s+?(.+?)A.U.')
    r2s = re.compile('Input orientation:.+\n.+\n.+\n.+\n.+\n([\S\s]+?)(?:---)')

    tse = r1s.search(f)
    tsx = r2s.search(f)

    en.append(tse.group(1))

    xyz = []
    sm = r32.findall(tsx.group(1))
    for j in sm:
        xyz.append(float(j[1]))
        xyz.append(float(j[2]))
        xyz.append(float(j[3]))
    cd.append(xyz)

    s1 = r1.findall(f)
    s2 = r2.findall(f)
    s3 = r4.findall(f)

    for i in s1:
        en.append(i.strip())

    Nc = 1
    for i in s2:

        sm = r3.findall(i.strip())
        Na = len(sm)

        ty = []
        xyz = []

        for j in sm:
            ty.append(convertatomicnumber(j[0]))
            xyz.append(float(j[1]))
            xyz.append(float(j[2]))
            xyz.append(float(j[3]))

        cd.append(xyz)
        Nc += 1

    en = np.asarray(en, dtype=np.float32)
    cd = np.asarray(cd, dtype=np.float32).reshape(Nc, Na, 3)

    Nf = int(s3[-1])

    en_t = np.concatenate([en[:Nf+1][::-1], en[Nf+1:]])
    cd_t = np.vstack([cd[:Nf+1,:,:][::-1,:,:], cd[Nf+1:,:,:]])

    Rc = np.array(rRc.findall(f), dtype=np.float32)

    return [en_t, cd_t, ty[0:Na], Rc]

def get_irc_data(fwd_file,bkw_file,trs_file):

    en1, cd1, ty1 = read_irc(fwd_file)
    en2, cd2, ty2 = read_irc(bkw_file)
    en3, cd3, ty3 = read_irc(trs_file)

    Et = np.array(en3[-1],dtype=np.float64).reshape(1)
    Ct = np.array(cd3[-1],dtype=np.float32).reshape(1,len(ty1),3)

    #print(cd2[::-1].shape,Ct.shape)

    #en = np.concatenate([en2[::-1], Et, en1])
    #cd = np.vstack([cd1[::-1], Ct, cd2])
    en = np.concatenate([en2[::-1], en1])
    cd = np.vstack([cd1[::-1], cd2])

    return en, cd, ty1

def read_scan (file):
    f = open(file,'r').read()

    ty = []
    ls = []
    en = []
    cd = []

    for i in range(0, 10):
        ls.append([])

    r0 = re.compile('(?:Optimization completed\.)([\s\S]+?)(?:\*\*\*\*\* Axes restored to original set \*\*\*\*\*)')
    r1 = re.compile('SCF Done:\s+?E\(.+?\)\s+?=\s+?(.+?)A.U.')
    r2 = re.compile('Input orientation:.+\n.+\n.+\n.+\n.+\n([\S\s]+?)(?:---)')
    r3 = re.compile('\d+?\s+?(\d+?)\s+?\d+?\s+?(\S+?)\s+?(\S+?)\s+?(\S+)')

    s0 = r0.findall(f)

    for i in s0:
        en.append(r1.findall(i)[0])
        B = r2.findall(i)

        rx = r3.findall(B[0])
        for i in rx:
            ty.append(convertatomicnumber(i[0]))
            cd.append(np.asarray([i[1],i[2],i[3]], dtype=np.float32))

    en = np.asarray(en, dtype=np.float32)
    cd = np.concatenate(cd).reshape(en.shape[0],int(len(cd)/en.shape[0]),3)

    return [en, cd, ty[0:cd.shape[1]]]

def create_input_str(wkdir,lot,options,chkpt,spc,xyz,mult,charge,nproc,mb):
    if len(spc) != xyz.shape[0]:
        print('spc and xyz shapes do not match!')
        exit(1)

    input = ''

    input += "%\n%Mem=" + str(mb) + "mb" + "\n"
    input += "%NProcShared=" + str(nproc) + "\n"
    input += "%chk=" + wkdir + chkpt + "\n"
    input += "# " + lot + " " + options + "\n\n"
    input += "COMMENT LINE\n\n"
    input += str(charge) + "  " + str(mult) + "\n"
    for t,r in zip(spc,xyz):
        input += t + ' ' + str(r[0]) + ' ' + str(r[1]) + ' ' + str(r[2]) + '\n'
    input += '\n'
    print(input)
    return input

def execg09(input):
    command = "g09 << " + input
    output = os.popen(command).read()
    return output

def read_energy(output):
    import re
    er = re.compile('SCF Done: +?E\(.+?\) +?= +?([-+]?\d+?\.\d+?) +?A\.U\.')
    value = er.findall(output)
    return np.asarray(value,dtype=np.float64)

def read_normalmodes_fromchkpt(chkptfile):
    import re

    sscmd = "formchk  " + chkptfile + " " + chkptfile + ".fchk"
    pipe = os.popen(sscmd).read()
    print(pipe)

    output = open(chkptfile + ".fchk").read()
    print(output)

    er = re.compile('SCF Done: +?E\(.+?\) +?= +?([-+]?\d+?\.\d+?) +?A\.U\.')
    value = er.findall(output)
    print(value)


def g09_compute_normmodes(wkdir,lot,spc,xyz):

    input = create_input_str(wkdir,lot,'freq','nmchk.chk',spc,xyz,1,0,8,1024)
    output = execg09(input)

    #print(output)
    E = read_energy(output)
    print(E)

    read_normalmodes_fromchkpt(wkdir+'nmchk.chk')

    #print('Output: ', output)

