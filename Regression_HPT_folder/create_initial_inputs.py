import os

import numpy as np
from scipy.io import loadmat
import shutil

# IMPORT FUNCTIONS FROM "FUNCTION_FILES"****************************************************************************** #
from function_files.data_preprocessing import data_preprocessing
from config_inputs import *
from import_and_organize_data import *

main_path = os.getcwd()

# ******************************************************************************************************************** #

# CREATE INPUT DICT ****************************************************************************************************

input_dict = dict()
input_dict['N'] = N
input_dict['Nk'] = Nk
input_dict['numLayers'] = numLayers
input_dict['numZooms'] = numZooms
input_dict['gridLength'] = gridLength
input_dict['main_path'] = main_path
input_dict['combo_array'] = combo_array
input_dict['numTopFeatures'] = numTopFeatures
input_dict['test_train_split_var'] = test_train_split_var
input_dict['split_decimal'] = split_decimal
model_use_logArray = np.identity(len(model_names))

# ******************************************************************************************************************** #

# CREATES AND SAVES DATA TO FOLDERS & SUBFOLDERS ***********************************************************************
numProps = len(prop_names)
numModels = len(model_names)
if 'data_processing_folder' not in os.listdir():
    os.mkdir('data_processing_folder')
os.chdir('data_processing_folder')
for case in case_use:
    case_current = 'Case_'+case
    if case_current in os.listdir():
        shutil.rmtree(case_current)
    os.mkdir(case_current)
    os.chdir(case_current)
    # Imports the correct 'case' data ----------------------------------------------------------------------------------
    X_case = feature_set_list[case]
    if case == 'org':
        Y_case = property_data
        goodIDs = goodId_org
    elif case == 'diff':
        Y_case = property_diff
        goodIDs = goodId_res
    elif case == 'ratio':
        Y_case = property_ratio
        goodIDs = goodId_res
    else:
        print("ERROR: No case input ~ Charlie")

    for prop_idx in props_to_run:
        # Creates Property Folder --------------------------------------------------------------------------------------
        prop_current = prop_names[prop_idx]
        folder_prop = str(prop_idx) + "-" + str(prop_current)
        os.mkdir(folder_prop)
        os.chdir(folder_prop)
        for mdl_idx in models_to_run:
            # Creates Model Folder -------------------------------------------------------------------------------------
            mdl_current = model_names[mdl_idx]
            folder_model = str(mdl_idx) + "-" + str(mdl_current)
            os.mkdir(folder_model)
            os.chdir(folder_model)

            # Adds correct data files to input folder ------------------------------------------------------------------
            Y_inp = Y_case[:, prop_idx]
            goodIDs_prop = goodIDs[:, prop_idx]
            hyperparameters = full_HP_list[str(prop_current)][str(mdl_current)]
            X_names = feature_name_list[case][prop_keys[prop_idx]]
            X_data = feature_set_list[case][prop_keys[prop_idx]]
            if parm_var == True:
                X_names.append('Parm')
                X_data.append(parmData)
            if locData_var == True:
                X_names.append('locData')
                X_data.append(locData)
            models_use_current = model_use_logArray[mdl_idx, :]
            model_type_current = model_types[mdl_idx]
            model_name_current = model_names[mdl_idx]
    
            input_dict['Y_inp'] = Y_inp
            input_dict['X_list'] = X_data
            input_dict['seed'] = seeds[prop_idx]
            input_dict['tr_ts_seed'] = tr_ts_seed[prop_idx]
            input_dict['goodIDs'] = goodIDs_prop
            input_dict['feature_names'] = X_names
            input_dict['hyperparameters'] = hyperparameters
            input_dict['models_use'] = models_use_current
            input_dict['model_name'] = model_name_current
            input_dict['model_type'] = model_type_current

            parm_use = parmData
            locData_use = locData

            input_dict['parm'] = parm_use
            input_dict['locData'] = locData_use

            # Takes in 'input_dict' and adds it to the current folder
            data_pre = data_preprocessing(input_dict=input_dict)
            data_pre.copy_input_files()
            os.chdir('..')
        os.chdir('..')
    os.chdir('..')

# ******************************************************************************************************************** #
