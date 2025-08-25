The purpose of the various files and programs in this github are as follows:
"SrCs_mol.py" is the file that contains the training procedure for the NN-mol model. 
"SrCs_mol_predict.py" is the file that is used to make predictions with the model NN-mol.
"SrCs_upto10K.py" and "SrCs_upto10K_predict,py" are the analagous files corresponding to the NN-ion model.
"features_SrCs_mol.txt" and "targets_SrCs_mol.txt" are the files containing the training data for NN-mol. 
"features_upto10K_SrCs.txt" and "targets_upto10K_SrCs.txt" are the files containing the training data for NN-ion. 
All files starting with "model" contain the model parameters, with names containing "mol" corresponding to NN-mol and those containing "upto10K" corresponding to NN-ion.
In the features files, the first column contains impact parameters (in units of bohr) and the second column contains the (natural log of) the collision energies, originally given in Kelvin.
