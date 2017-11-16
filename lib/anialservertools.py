import subprocess
import pyssh
import re
import os

class alserversubmission():
    def __init__(self, hostname, username, swkdir, ldtdir, port=22):
        self.server = pyssh.session.Session(hostname=hostname, username=username, port=str(port))
        self.swkdir = swkdir
        self.ldtdir = ldtdir

        self.hostname = hostname
        self.username = username
        self.port = port

        self.job_ids = set({})
        self.job_list = []

    def submit_ssh_command(self, command):
        r = self.server.execute(command)
        return r.as_str()

    def submit_job(self):
        re.compile('Submitted batch job\s(\d+?)(\s|\n|$)')

    def get_job_status(self):
        r = self.submit_ssh_command("squeue -O jobid,state,numcpus,timeused -u jsmith48")
        r = r.split("\n")[1:-1]

        running_ids = set()
        for i in r:
            data = [j for j in i.split(" ") if j != '']
            running_ids.add(data[0])

        return len(self.job_ids.intersection(running_ids))

    def create_submission_script(self, idx):
        fname = self.ldtdir+'file'+idx+'.sh'
        self.job_list.append('file'+idx+'.sh')
        sf = open(fname, 'w')

        parti = 'shared'
        times = '0-24:00'
        Nmemr = 2048

        sf.write('#!/bin/sh\n')
        sf.write('#SBATCH --job-name=\"' + 'test' + '\"\n')
        sf.write('#SBATCH --partition=' + parti + '\n')
        sf.write('#SBATCH -N 1 # number of nodes\n')
        sf.write('#SBATCH -n ' + str(1) + ' # number of cores\n')
        sf.write('#SBATCH --mem-per-cpu=' + str(Nmemr) + '\n')
        sf.write('#SBATCH -t ' + times + ' # time (D-HH:MM)\n')
        sf.write('#SBATCH -o slurm.cgfp.%j.out # STDOUT\n')
        sf.write('#SBATCH -e slurm.cgfp.%j.err # STDERR\n')
        sf.write('#SBATCH --mail-type=END,FAIL # notifications for job done & fail\n')
        sf.write('#SBATCH --mail-user=jsmith48@ufl.edu # send-to address\n')
        sf.write('#SBATCH -A jsu101\n')
        sf.write('\n')
        sf.write('sleep 1m')
        sf.write('\n')

        sf.close()

        #sf.write('cd ' + wrkdir + '\n\n')

    def load_to_server(self):
        command = 'rsync -a ' + self.ldtdir + '* ' + self.username  + '@' + self.hostname + ':' + self.swkdir
        print('    -Executing:',command)
        proc = subprocess.Popen (command, shell=True)
        r = proc.communicate()
        print(r)

    def run_all_jobs(self):
        for j in self.job_list:
            r = self.submit_ssh_command("cd " + self.swkdir + " && sbatch " + j)
            jid = [i for i in r.split(' ') if i != ''][-1][:-1]
            self.job_ids.add(jid)
        print(self.job_ids)

    def disconnect(self):
        self.server.close()