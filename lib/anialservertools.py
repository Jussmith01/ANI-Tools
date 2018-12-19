import hdnntools as hdt
import numpy as np

import pyanitools as pyt

from traceback import print_exc
from time import sleep

import subprocess
import pyssh
import math
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
    i = 0
    with open(fname) as f:
        for l in f:
            i += 1
            pass
    return i + 1

class alQMserversubmission():
    def __init__(self, hostname, username, swkdir, ldtdir, datdir, jtime, max_jobs = 40, port=22, password=''):
        if password:
            self.server = pyssh.session.Session(hostname=hostname, username=username, password=password, port=str(port))
        else:	
            self.server = pyssh.session.Session(hostname=hostname, username=username, port=str(port))

        self.swkdir = swkdir
        self.ldtdir = ldtdir
        self.datdir = datdir
        self.jtime = jtime

        self.hostname = hostname
        self.username = username
        self.password = password
        self.port = port

        self.job_ids = set({})
        self.job_list = []
        self.mae = ''
        self.ucount = 0

        self.jobs_cnt = 0
        self.max_jobs = max_jobs

        self.cmp_count = 0

        self.job_ids = set()
        self.iter = 0

    def reconnect(self):
        self.server.close()
        if self.password:
            self.server = pyssh.session.Session(hostname=self.hostname, username=self.username, password=self.password, port=str(self.port))
        else:
            self.server = pyssh.session.Session(hostname=self.hostname, username=self.username, port=str(self.port))

    def set_optional_submission_command(self, mae):
        self.mae = mae

    def submit_ssh_command(self, command):
        if self.ucount > 8:
            self.reconnect()
            self.ucount = 0

        connect = False
        r = ""
        while connect is False:
            try:
                r = self.server.execute(command, lazy=True)
                connect = True
            except Exception as e:
                ext = e.__class__.__name__
                print('Error raised:', ext)
                if ext == "AuthenticationError" or ext == "ConnectionError":
                    print("Attempting connection again in 120s...")
                    connect = False
                    sleep(120)
                    self.reconnect()
                    self.ucount = 0
                else:
                    print("Another error occured which I cannot handle.")
                    connect = True
                    print_exc()
                    exit(1)

        self.ucount += 1
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

        #print('jid:',self.job_ids)
        #print('rid:',running_ids)

        return len(self.job_ids.intersection(running_ids))

    def create_submission_script(self, file, cores, time):
        f = file.rsplit('.',1)[0]
        fname = self.ldtdir+self.datdir+'/working/'+f+'.sh'
        self.job_list.append(f+'.sh')
        sf = open(fname, 'w')

        parti = 'shared'
        times = time 
        #times = '0-1:30'
        Nmemr = 3000
        lot = 'wb97x/6-31g*'

        sf.write('#!/bin/sh\n')
        sf.write('#SBATCH --job-name=\"' + f + '\"\n')
        sf.write('#SBATCH --partition=' + parti + '\n')
        sf.write('#SBATCH -N 1 # number of nodes\n')
        sf.write('#SBATCH -n ' + str(cores) + ' # number of cores\n')
        sf.write('#SBATCH --mem ' + str(cores*Nmemr) + '\n')
        sf.write('#SBATCH -t ' + times + ' # time (D-HH:MM)\n')
        sf.write('#SBATCH -o slurm.cgfp.%j.out # STDOUT\n')
        sf.write('#SBATCH -e slurm.cgfp.%j.err # STDERR\n')
        #sf.write('#SBATCH --mail-type=END,FAIL # notifications for job done & fail\n')
        #sf.write('#SBATCH --mail-user=jsmith48@ufl.edu # send-to address\n')
        #sf.write('#SBATCH -A mr3bdtp\n') #BRIDGES
        sf.write('#SBATCH -A jsu101\n')
        sf.write('\n')
        sf.write('cd ' + self.swkdir+self.datdir+'/working/' + '\n\n')

        sf.write(self.mae)

        sf.write('export OMP_STACKSIZE=64m\n')
        #sf.write('export OMP_NUM_THREADS='+ str(cores) +'\n')
        sf.write('export OMP_NUM_THREADS=1\n')

        #g09scr = self.ldtdir+self.datdir+'/working/' + f + '_scratch'
        #if not os.path.exists(g09scr):
        #    os.makedirs(g09scr)
        #sf.write('mkdir /scratch/$USER/$SLURM_JOBID/')
        sf.write('export GAUSS_SCRDIR=/scratch/$USER/$SLURM_JOBID/\n\n')

        sf.write('gcdata -i ' + self.swkdir+self.datdir+'/confs_iso/'+f+'.xyz' + ' -o ' + self.swkdir+self.datdir+'/data/' + f.split(".")[0] + '.dat -l ' + lot + ' -c ' + str(cores) + ' -m ' + str(
            Nmemr) + ' -s -p -f > ' + self.swkdir+self.datdir+'/output/' + f.split(".")[0] + '.opt')

        sf.close()

        #sf.write('cd ' + wrkdir + '\n\n')

    def prepare_confs_iso(self):
        prefix = 'ad'
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
                ds[id].append((X, S, C))
            else:
                of.write(f + ' ' + convert_eformula(S) + ' ' + str(0) + ' ' + str(X.shape[0]) + '\n')
                ds.update({id: [(X, S, C)]})
        of.close()

        Nt = 0
        for i in ds.keys():
            X = []
            S = []
            C = []
            for j in ds[i]:
                X.append(j[0])
                S.append(j[1])
                C.extend(j[2])

            X = np.vstack(X)
            S = list(S[0])
            N = X.shape[0]

            Nt += N

            if N < 40:
                #print(type(S), S)
                fn = prefix + '_' + convert_eformula(S) + '-' + str(N).zfill(3) + '.xyz'
                #print('Writing: ', fn)
                hdt.writexyzfilewc(isoms_dir + '/' + fn, X, S, C)
            else:
                Nsplit = int(math.ceil(N/float(40)))
                X = np.array_split(X, Nsplit)
                C = np.array_split(np.array(C), Nsplit)
                for l,(x,c) in enumerate(zip(X,C)):
                    fn = prefix + '_' + convert_eformula(S) + '_'  + str(l).zfill(2)  + '-' + str(x.shape[0]).zfill(3) + '.xyz'
                    hdt.writexyzfilewc(isoms_dir + '/' + fn, x, S, c)
                
        #print('Total data:', Nt)

    def compress_orginal_confs(self):
        command = 'cd ' + self.ldtdir+self.datdir + ' && tar --remove-files -czf confs.tar.gz confs'
        proc = subprocess.Popen (command, shell=True)
        r = proc.communicate()
        print('Compressing confs...')
        proc.wait()
        print(r)


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
        self.compress_orginal_confs()

        fextn = 'xyz'
        self.files = os.listdir(iptdir)
        self.files = [f for f in self.files if f.split(".")[1] == fextn]
        self.files = sorted(self.files, key=lambda x: int(x.rsplit("-", 1)[1].split(".")[0]), reverse=True)#[415:]

        rgx = re.compile('(?:[A-G]|[I-Z])[a-z]?(\d+)')

        print('Building scripts...')
        for f in self.files:
            As = f.rsplit('-', 1)[0].split('_')[1]
            Na = np.sum(np.array(rgx.findall(As),dtype=np.int32))

            Nc = int(f.rsplit('-', 1)[1].split('.')[0])
            if Nc > 32:
                Nproc = 8
            elif Nc > 24:
                Nproc = 8
            elif Nc > 12:
                Nproc = 6
            else:
                Nproc = 4

            #if Na > 18:
            #    Nproc = 2 * Nproc

            #if Na <= 7 and Nproc > 1:
            #    Nproc = int(Nproc/2)

            self.create_submission_script(f, Nproc, self.jtime)

        #self.jobs_left = len(self.job_list)

        #sf = open(self.ldtdir+self.datdir+'/working/runall.sh', 'w')
        #sf.write('#!/bin/sh\n')
        #for j in self.job_list:
        #    sf.write('sbatch ' + j + '\n')
        #sf.close()

    def check_and_submit_jobs(self):
        Nj = self.get_job_status()
        sleep(2)

        Nl = len(self.job_list)-self.jobs_cnt

        N = min([self.max_jobs-Nj, Nl])
        
        command = "cd " + self.swkdir + self.datdir + '/working && '
        idx = self.jobs_cnt
        #print('N:',N,'Nl:',Nl,'Nj:',Nj,'Jtot:',len(self.job_list))
        if N > 0:
            for j in self.job_list[idx:idx+N]:
                command += 'sbatch ' + j + ' && '
            command = command[:-4]
            #print('Command:', command)
            
            r = self.submit_ssh_command(command)
            print(r)
            reg = re.compile(r'(?:Submitted batch job )(\d+?)(?:\n|$)')

            ids = reg.findall(r)
            self.job_ids = self.job_ids.union(set(ids))

            self.jobs_cnt += N

            if self.iter > 0:
                sleep(2)
                self.compress_fchks()
        
        self.iter += 1
        complete = (Nj == 0) and (Nl == 0)
        return complete,Nj,N,len(self.job_list)-self.jobs_cnt

    def load_to_server(self):
        #pwd = " sshpass -p "+self.password
        command = 'rsync -a ' + self.ldtdir + self.datdir + ' ' + self.username  + '@' + self.hostname + ':' + self.swkdir
        proc = subprocess.Popen (command, shell=True)
        r = proc.communicate()
        print('Wait')
        proc.wait()
        print(r)

    def load_from_server(self):
        #pwd = " sshpass -p "+self.password
        command = 'rsync -a --delete ' + self.username  + '@' + self.hostname + ':' + self.swkdir + self.datdir + ' ' + self.ldtdir
        print('Execute transfer from server...')
        proc = subprocess.Popen (command, shell=True)
        r = proc.communicate()
        proc.wait()
        print(r)

    def compress_fchks(self):
        r = self.submit_ssh_command('cd ' + self.swkdir+self.datdir+'/data/' + ' && tar --remove-files -czf '+ self.swkdir+self.datdir+'/checkpoints/checkpoints'+str(self.cmp_count).zfill(3)+'.tar.gz ' + '*.fchk')
        print(r)
        self.cmp_count += 1

    def cleanup_server(self):
        r = self.submit_ssh_command('cd ' + self.swkdir+self.datdir+'/data/' + " && rm -r " + self.swkdir+self.datdir+'/working && tar --remove-files -czf ' + self.swkdir+self.datdir+'/checkpoints/checkpoints'+str(self.cmp_count).zfill(3)+'.tar.gz ' + '*.fchk && ' + 'rm ' + '*.chk')

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
            #print(d + f)
            L = file_len(d + f)

            if L >= 4:
                #print(d + f)

                data = hdt.readncdatall(d + f)

                if 'energies' in data:
                    Ne = data['energies'].size
                    Nd += Ne

                    f = f.rsplit("-", 1)

                    fn = f[0] + "/mol" + f[1].split(".")[0]

                    dpack.store_data(fn, **data)
        dpack.cleanup()


    def disconnect(self):
        self.server.close()

def generateQMdata(hostname, username, swkdir, ldtdir, datdir, h5stor, mae, jtime, password='', max_jobs = 40):
    # Declare server submission class and connect to ssh
    print('Connecting...')
    alserv = alQMserversubmission(hostname, username, swkdir, ldtdir, datdir, jtime, password=password, max_jobs=max_jobs)

    # Set optional server information
    alserv.set_optional_submission_command(mae)

    # Prepare the working directories and submission files
    alserv.prepare_data_dir()
    #exit(0)
    # Load all prepared files to the server
    print('Loading to server...')
    alserv.load_to_server()

    # Submit and monitor jobs
    print('Generating Data...')
    complete = False
    while True:
        complete,Nr,Nj,Nl = alserv.check_and_submit_jobs()
        print("Running (" + str(Nr) + " : " + str(Nj) + " : " + str(Nl) + " : " + str(len(alserv.job_list)) + ")...")
        if complete:
            break
        sleep(120)

    # CLeanup working files on server
    print('Cleaing up server...')
    sleep(5) # pyssh seems to freeze sometimes when there are fast back to back commands
    alserv.cleanup_server()

    # Load all data from server
    print('Loading from server...')
    sleep(5) # pyssh seems to freeze sometimes when there are fast back to back commands
    alserv.load_from_server()

    # Create h5 file
    print('Packing data...')
    alserv.generate_h5(h5stor + datdir + '.h5')

    print('--Cycle Complete--')
    alserv.disconnect()
