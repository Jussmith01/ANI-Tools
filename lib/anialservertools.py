import hdnntools as hdt
import numpy as np

import pyanitools as pyt

from time import sleep
import subprocess
import pyssh
import re
import os

def convert_eformula(S):
    Z = set(S)
    rtn = str()
    for z in Z:
        N = list(S).count(z)
        rtn += z+str(N)
    return rtn

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

class alQMserversubmission():
    def __init__(self, hostname, username, swkdir, ldtdir, datdir, port=22):
        self.server = pyssh.session.Session(hostname=hostname, username=username, port=str(port))
        self.swkdir = swkdir
        self.ldtdir = ldtdir
        self.datdir = datdir

        self.hostname = hostname
        self.username = username
        self.port = port

        self.job_ids = set({})
        self.job_list = []
        self.mae = ''

    def set_optional_submission_command(self, mae):
        self.mae = mae

    def submit_ssh_command(self, command):
        r = self.server.execute(command)
        return r.as_str()

    def submit_job(self):
        re.compile('Submitted batch job\s(\d+?)(\s|\n|$)')

    def get_job_status(self):
        r = self.submit_ssh_command("squeue -O jobid,state,numcpus,timeused -u " + self.username)
        r = r.split("\n")[1:-1]

        running_ids = set()
        for i in r:
            data = [j for j in i.split(" ") if j != '']
            running_ids.add(data[0])

        print('jid:',self.job_ids)
        print('rid:',running_ids)

        return len(self.job_ids.intersection(running_ids))

    def create_submission_script(self, file, cores):
        f = file.rsplit('.',1)[0]
        fname = self.ldtdir+self.datdir+'/working/'+f+'.sh'
        self.job_list.append(f+'.sh')
        sf = open(fname, 'w')

        parti = 'shared'
        times = '0-24:00'
        Nmemr = 2048
        lot = 'wb97x/6-31g*'

        sf.write('#!/bin/sh\n')
        sf.write('#SBATCH --job-name=\"' + 'test' + '\"\n')
        sf.write('#SBATCH --partition=' + parti + '\n')
        sf.write('#SBATCH -N 1 # number of nodes\n')
        sf.write('#SBATCH -n ' + str(cores) + ' # number of cores\n')
        sf.write('#SBATCH --mem-per-cpu=' + str(Nmemr) + '\n')
        sf.write('#SBATCH -t ' + times + ' # time (D-HH:MM)\n')
        sf.write('#SBATCH -o slurm.cgfp.%j.out # STDOUT\n')
        sf.write('#SBATCH -e slurm.cgfp.%j.err # STDERR\n')
        sf.write('#SBATCH --mail-type=END,FAIL # notifications for job done & fail\n')
        sf.write('#SBATCH --mail-user=jsmith48@ufl.edu # send-to address\n')
        sf.write('#SBATCH -A jsu101\n')
        sf.write('\n')
        sf.write('cd ' + self.swkdir+self.datdir+'/working/' + '\n\n')

        sf.write(self.mae)

        sf.write('export OMP_STACKSIZE=64m\n')
        sf.write('export OMP_NUM_THREADS='+ str(cores) +'\n')

        g09scr = self.ldtdir+self.datdir+'/working/' + f + '_scratch'
        if not os.path.exists(g09scr):
            os.makedirs(g09scr)
        sf.write('export GAUSS_SCRDIR=' + self.swkdir+self.datdir+'/working/' + '\n\n')

        sf.write('gcdata -i ' + self.swkdir+self.datdir+'/confs_iso/'+f+'.xyz' + ' -o ' + self.swkdir+self.datdir+'/data/' + f.split(".")[0] + '.dat -l ' + lot + ' -m ' + str(
            Nmemr) + ' -s -p -f > ' + self.swkdir+self.datdir+'/output/' + f.split(".")[0] + '.opt')

        sf.close()

        #sf.write('cd ' + wrkdir + '\n\n')

    def prepare_confs_iso(self):
        prefix = 'anidata'
        confs_dir = self.ldtdir + self.datdir + '/confs/'
        isoms_dir = self.ldtdir + self.datdir + '/confs_iso/'

        files = os.listdir(confs_dir)
        files = [f for f in files if f.rsplit('.', maxsplit=1)[-1] == 'xyz']
        #print(len(files))

        ds = dict()
        of = open(self.ldtdir + self.datdir + '/info_confstoiso_map.dat', 'w')
        for i, f in enumerate(files):
            #print(confs_dir + f)
            X, S, N, C = hdt.readxyz2(confs_dir + f)
            S = np.array(S)

            idx = sorted(range(len(S)), key=lambda k: S[k])
            S = S[np.array(idx)]

            for j, x in enumerate(X):
                X[j] = x[idx]

            id = "".join(S)

            if id in ds:
                sid = len(ds[id])
                of.write(f + ' ' + convert_eformula(S) + ' ' + str(sid) + ' ' + str(X.shape[0]) + '\n')
                ds[id].append((X, S))
            else:
                of.write(f + ' ' + convert_eformula(S) + ' ' + str(0) + ' ' + str(X.shape[0]) + '\n')
                ds.update({id: [(X, S)]})
        of.close()

        Nt = 0
        for i in ds.keys():
            X = []
            S = []
            for j in ds[i]:
                X.append(j[0])
                S.append(j[1])

            X = np.vstack(X)
            S = list(S[0])
            N = X.shape[0]

            Nt += N

            #print(type(S), S)
            fn = prefix + '_' + convert_eformula(S) + '-' + str(N).zfill(5) + '.xyz'
            #print('Writing: ', fn)
            hdt.writexyzfile(isoms_dir + '/' + fn, X, S)
        #print('Total data:', Nt)

    def prepare_data_dir(self):
        iptdir = self.ldtdir + self.datdir + '/confs_iso/'
        optdir = self.ldtdir + self.datdir + '/output/'
        datdir = self.ldtdir + self.datdir + '/data/'
        wrkdir = self.ldtdir + self.datdir + '/working/'
        chkdir = self.ldtdir + self.datdir + '/checkpoints/'

        if not os.path.exists(iptdir):
            os.mkdir(iptdir)
        if not os.path.exists(optdir):
            os.mkdir(optdir)
        if not os.path.exists(datdir):
            os.mkdir(datdir)
        if not os.path.exists(wrkdir):
            os.mkdir(wrkdir)
        if not os.path.exists(chkdir):
            os.mkdir(chkdir)

        print('Preparing iso confs...')
        self.prepare_confs_iso()

        fextn = 'xyz'
        self.files = os.listdir(iptdir)
        self.files = [f for f in self.files if f.split(".")[1] == fextn]
        self.files = sorted(self.files, key=lambda x: int(x.rsplit("-", 1)[1].split(".")[0]), reverse=True)

        print('Building scripts...')
        for f in self.files:
            Nc = int(f.rsplit('-', 1)[1].split('.')[0])
            if Nc > 512:
                Nproc = 8
            elif Nc > 128:
                Nproc = 4
            elif Nc > 48:
                Nproc = 2
            elif Nc > 12:
                Nproc = 1
            else:
                Nproc = 1
            print(f)
            self.create_submission_script(f, Nproc)

        sf = open(self.ldtdir+self.datdir+'/working/runall.sh', 'w')
        sf.write('#!/bin/sh\n')
        for j in self.job_list:
            sf.write('sbatch ' + j + '\n')
        sf.close()


    def load_to_server(self):
        command = 'rsync -a --delete ' + self.ldtdir + self.datdir + ' ' + self.username  + '@' + self.hostname + ':' + self.swkdir
        proc = subprocess.Popen (command, shell=True)
        r = proc.communicate()
        print('Wait')
        proc.wait()
        print(r)

    def load_from_server(self):
        command = 'rsync -a --delete ' + self.username  + '@' + self.hostname + ':' + self.swkdir + self.datdir + ' ' + self.ldtdir
        print('Execute transfer from server...')
        proc = subprocess.Popen (command, shell=True)
        r = proc.communicate()
        proc.wait()
        print(r)

    def run_all_jobs(self):
        print("cd " + self.swkdir + self.datdir + '/working' + " && bash runall.sh")
        r = self.submit_ssh_command("cd " + self.swkdir + self.datdir + '/working' + " && bash runall.sh")
        print(r)
        reg = re.compile(r'(?:Submitted batch job )(\d+?)(?:\n|$)')

        ids = reg.findall(r)
        self.job_ids = set(ids)

    def cleanup_server(self):
        r = self.submit_ssh_command("rm -r " + self.swkdir+self.datdir+'/working && mv '
                                    + self.swkdir+self.datdir+'/data/*.fchk ' + self.swkdir+self.datdir+'/checkpoints/ && '
                                    + 'rm ' + self.swkdir + self.datdir+'/data/*.chk')

    def generate_h5(self, path):
        # open an HDF5 for compressed storage.
        # Note that if the path exists, it will open whatever is there.
        dpack = pyt.datapacker(path)

        d = self.ldtdir+self.datdir+'/data/'
        files = [f for f in os.listdir(d) if ".dat" in f]
        files.sort()
        Nf = len(files)
        Nd = 0
        for n, f in enumerate(files):
            L = file_len(d + f)

            if L >= 4:
                print(d + f)

                data = hdt.readncdatall(d + f)

                Ne = data['energies'].size
                Nd += Ne

                f = f.rsplit("-", 1)

                fn = f[0] + "/mol" + f[1].split(".")[0]

                dpack.store_data(fn, **data)
        dpack.cleanup()


    def disconnect(self):
        self.server.close()

def generateQMdata(hostname, username, swkdir, ldtdir, datdir, h5stor, mae):
    # Declare server submission class and connect to ssh
    alserv = alQMserversubmission(hostname, username, swkdir, ldtdir, datdir)

    # Set optional server information
    alserv.set_optional_submission_command(mae)

    # Prepare the working directories and submission files
    alserv.prepare_data_dir()

    # Load all prepared files to the server
    alserv.load_to_server()

    # Run all jobs at once
    alserv.run_all_jobs()

    # Monitor jobs
    print('Listening...')
    Nj = 1
    while Nj != 0:
        sleep(120)
        Nj = alserv.get_job_status()
        print("Running (" + str(Nj) + ")...")

    # CLeanup working files on server
    alserv.cleanup_server()

    # Load all data from server
    alserv.load_from_server()

    # Create h5 file
    print('Packing...')
    alserv.generate_h5(h5stor + datdir + '.h5')

    print('END')

    alserv.disconnect()