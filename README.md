# ANI-Tools
1.Obtain training binaries HDAtomNNP-Trainer

	•Contact us for this file
		i.Justin Smith: jsmith48@lanl.gov
		ii.Christian Devereux: cdever01@ufl.edu
		iii.Kavindri Ranasingrhe: kdrana.uf@chem.ufl.edu

	•Make a directory for HDAtomNNP-Trainer
		i.There may be permission errors with this file. If these occur go to its location and type “chmod +x HDAtomNNP-Trainer”

	•Add the following to your .bashrc
		i.export PATH="/path/to/HDAtomNNP-Traininer:$PATH"


2.Clone ANI-Tools repository

	•git clone https://github.com/Jussmith01/ANI-Tools.git

	•Change to the branch training
		i.cd to ANI-Tools
		ii.Enter “git checkout training” into the command line

	•Add the following to your .bashrc:
		i.export PYTHONPATH="/path/to/ANI-Tools/lib"


3.Clone ASE_ANI repository

	•git clone https://github.com/isayev/ASE_ANI.git
		i.Be sure to check the requirements in the README page

	•Change to the branch python_36_cuda92_nightly
		i.cd to ASE_ANI
		ii.Enter “git checkout python_36_cuda92_nightly” into the command line

	•Add the following to your .bashrc:
		i.export NC_ROOT="/path/to/ASE_ANI"
		ii.export LD_LIBRARY_PATH="$NC_ROOT/lib:$LD_LIBRARY_PATH"
		iii.export PYTHONPATH="$NC_ROOT/lib:$PYTHONPATH"


4.Set up the training script

	•ANI-Tools/NeuroChem_training/ens_trainer.py  

	•Change nwdir (line 6) to the path where you want the new ensemble to be located
		i.This path must already exist
		ii.nwdir must be set to the full path

	•Change h5dir (line 7) to the path containing your training data
		i.h5dir must be set to the full path
		ii.A sample training set is in ANI-Tools/Neurochem_training/h5files

	•Set the size of the ensemble to be trained and the portion of the data set to be used for validation and testing (lines 10-13)

	•Set the parameters to be used when generating the atomic environment vectors (par.set_params) (lines 28-37)

	•Determine the network architecture for each atom type (lines 55-69)

	•Set the training parameters (lines 88-103)

	•Change GPU to the ID numbers of the GPUs to be used for training (line 118) 


5.Execute the script

	•python ens_trainer.py

MSE vs. training epoch for sample network
![Alt text](NeuroChem_training/images/MSE.png?raw=true "MSE")


RMSE vs. training epoch for sample network 
![Alt text](NeuroChem_training/images/RMSE.png?raw=true "RMSE")
