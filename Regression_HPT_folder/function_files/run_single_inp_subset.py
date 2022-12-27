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
pickle_file_decoded = []
with (open('input_set', 'rb')) as f:
    while True:
        try:
            pickle_file_decoded.append(pickle.load(f))
        except EOFError:
            break
input_set = pickle_file_decoded[0]

main_path = input_dict['main_path']
current_path = os.getcwd()
sys.path.append(os.path.abspath(main_path))
from config_inputs import parm_var, locData_var, num_fts_per_sublist
sys.path.append(os.path.abspath(main_path+'/function_files/'))
from Regression import Regression
from Heatmaps_old import heatmaps
sys.path.append(os.path.abspath(current_path))
# **********************************************************************************************************************


Nk = input_dict['Nk']
N = input_dict['N']
Y_int = input_dict['Y_inp']
parm = input_dict['parm']
locDat = input_dict['locData']
goodIds = input_dict['goodIDs']
models_use = input_dict['models_use']
model_type = input_dict['model_type']
seed = input_dict['seed']
current_path = os.getcwd()

X_list = input_set['features']
X_full_list = []
X_names_inp = input_set['names']

if parm_var:
    X_names = []
    for i in range(len(X_names_inp)):
        X_names_temp = X_names_inp[i]
        X_names_new = []
        for n in X_names_temp:
            X_names_new.append(n)
        X_names_new.append('Parm')
        X_names.append(tuple(X_names_new))
else:
    X_names = X_names_inp

NftSets = len(X_list)

RMSE_list = []
R2_list = []

for i in range(NftSets):
    ft_set = X_list[i]
    ft_names = X_names[i]

    if len(ft_set) == 1:
        X_use = ft_set
    elif len(ft_set) > 1:
        X_use = np.concatenate(ft_set, axis=1)
    else:
        print("ERROR IN FEATURE-SET LENGTH ~ Charlie")

    if parm_var:
        X_use = np.concatenate([X_use, parm], axis=1)

    X_full_list.append(X_use)
    print(str(i + 1) + "/" + str(NftSets) + ": ", ft_names)

    reg = Regression(X_use, Y_int, Nk=Nk, N=N,
                     seed=seed, goodIDs=goodIds, RemoveNaN=True, StandardizeX=True,
                     models_use=models_use, giveKFdata=False)
    results, bestPred = reg.RegressionCVMult()

    # Collects bestPred Data
    rmse_temp = bestPred['rmse']
    RMSE_list.append(rmse_temp)
    r2_temp = bestPred['r2']
    R2_list.append(r2_temp)

fname = os.path.basename(os.getcwd())
sorting_var_start = num_fts_per_sublist * int(fname)
sorting_var_end = sorting_var_start + NftSets
sorting_var_list = np.arange(start=sorting_var_start, stop=sorting_var_end, step=1)


sublist_df = pd.DataFrame(columns=X_names)
sublist_df.loc['feature_names'] = X_names
sublist_df.loc['RMSE'] = RMSE_list
sublist_df.loc["R2"] = R2_list
sublist_df.loc["sorting_var"] = sorting_var_list

sublist_df.to_csv('data_file.csv')

final_X_data = dict()
final_X_data['features'] = X_full_list
final_X_data['names'] = X_names
file_final_X_data = open('final_X_data', 'wb')
pickle.dump(final_X_data, file_final_X_data)
file_final_X_data.close()

