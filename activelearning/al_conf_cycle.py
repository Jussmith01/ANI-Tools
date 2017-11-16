import anialservertools as ast
from time import sleep

hostname = "comet.sdsc.xsede.org"
username = "jsmith48"
password = "Tiger-123"

swkdir = '/home/jsmith48/scratch/test_autoal/'# server working directory
ldtdir = '/home/jujuman/Research/test_auto_al/'# local data directories


alserv = ast.alserversubmission(hostname, username, swkdir, ldtdir)

alserv.create_submission_script("1")
alserv.create_submission_script("2")

alserv.load_to_server()

alserv.run_all_jobs()

Nj = 1
while Nj != 0:
    sleep(30)
    Nj = alserv.get_job_status()
    print("Running... ",Nj)
    
print('END')

alserv.disconnect()