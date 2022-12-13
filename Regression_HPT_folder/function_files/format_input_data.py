import shutil
import sys
import os
import pickle
import csv
import pandas as pd
import numpy as np
from itertools import combinations

# IMPORT FILES AND FUNCTIONS *******************************************************************************************
pickle_file_decoded = []
with (open('input_file', 'rb')) as f:
    while True:
        try:
            pickle_file_decoded.append(pickle.load(f))
        except EOFError:
            break
input_dict = pickle_file_decoded[0]

main_path = input_dict['main_path']
current_path = os.getcwd()
sys.path.append(os.path.abspath(main_path))
from config_inputs import parm_var, locData_var, num_fts_per_sublist
sys.path.append(os.path.abspath(main_path+'/function_files/'))
from Regression import Regression
from Heatmaps import heatmaps
sys.path.append(os.path.abspath(current_path))
# **********************************************************************************************************************


Nk = input_dict['Nk']
N = input_dict['N']
Y_int = input_dict['Y_inp']
X_list = input_dict['X_list']
parm = input_dict['parm']
locDat = input_dict['locData']
goodIds = input_dict['goodIDs']
models_use = input_dict['models_use']
model_type = input_dict['model_type']
seed = input_dict['seed']
combo_array = input_dict['combo_array']
ftSet_names = input_dict['feature_names']
print("FEATURE NAMES")
print(ftSet_names)
numTopFeatures = input_dict['numTopFeatures']
current_path = os.getcwd()

numFeatSets = len(X_list)

X_full_list = []
X_names_list = []

for combo_length in combo_array:
    X_set_combo_temp_list = list(combinations(X_list, combo_length))  # Creates all the combinations of the features
    if ftSet_names == []:  # Tests to see if the feature names were input beforehand
        X_ftName_list = np.arange(start=0, stop=numFeatSets, step=1)
        X_names_combo_temp_list = list(combinations(X_ftName_list, combo_length))
    else:
        X_names_combo_temp_list = list(combinations(ftSet_names, combo_length))

    Npts_i = len(X_set_combo_temp_list)
    for i in range(Npts_i):
        X_full_list.append(X_set_combo_temp_list[i])
        X_names_list.append(X_names_combo_temp_list[i])

s = os.path.sep
file_1_dest = os.getcwd() + s + 'input_file'
file_2_dest = os.getcwd() + s + 'run_single_inp_subset.py'
file_3_dest = os.getcwd() + s + 'run_single_inp_subset.sub'

os.mkdir('input_feature_set_folder')
os.chdir('input_feature_set_folder')
Ncombos = len(X_full_list)
count = 0
f_i = 0
while count < Ncombos-1:
    sublist = []
    subnames = []
    for i in range(num_fts_per_sublist):
        print("COUNT: ", count)
        if count != Ncombos:
            sublist.append(X_full_list[count])
            subnames.append(X_names_list[count])
            count += 1
        else:
            break
    os.mkdir(str(f_i))
    os.chdir(str(f_i))
    with open('input.txt', 'w') as f:
        for nm in range(len(subnames)):
            f.write(str(subnames[nm]) + "\n")
        f.close
    input_set = dict()
    input_set['features'] = sublist
    input_set['names'] = subnames
    file_for_inputs = open('input_set', 'wb')
    pickle.dump(input_set, file_for_inputs)
    file_for_inputs.close
    shutil.copyfile(file_1_dest, 'input_file')
    shutil.copyfile(file_2_dest, 'run_single_inp_subset.py')
    shutil.copyfile(file_3_dest, 'run_single_inp_subset.sub')
    f_i += 1
    os.chdir('..')
os.chdir('..')

