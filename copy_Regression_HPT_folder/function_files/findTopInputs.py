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
from config_inputs import parm_var, locData_var
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

numFeatSets = len(X_list)  # Number of features input

# OUTPUT LISTS
FtSetNames = []
RMSE_list = []
R2_list = []
sorting_var_list = []
list_of_input_feature_sets_fullData = []
sorting_var = 0

for combo_length in combo_array:  # Itterates over the array 'combo_array'
    """ METHOD """
    """
    This script is used to determine the top 'n' set of input features. It does so via two possible methods.
    1. Looking at only the default C, Epsilon, Gamma, and Coef0 cases and finding the top 'n'
    2. Running a coarse HPT-test on each set of input features to determine the top 'n'
        - This method has the pro of being able to distingusih between feature-sets that are near their 
          optimal zone at the default, but don't have much room for imporvment, and those that are not near
          their optimal zone at the defualt (and would therefor be ranked lower in case 1), but have room for
          imporovment, possibly above the former set. It runs at the con of running much slower, having to run
          m^(#-hps) cases (m=meshsize, #-hps=the number of hyperparameters being studied) vs just 1 run in
          case 1.
    For each case, all but the actual running of the regessions is the same. The method for everything around this
    part is as follows:
    First, for each 'combo-length' (which is represented as an element in the 'combo_array'), all sets of 
    input-feature combonations are put in a list, and each set is run and the RMSE and R^2 is stored in a list.
    The full list is then sorted and the top 'n' features are chosen based on the lowest RMSE. These feature sets 
    are then combined into a CSV file, folders are created for each case, and their respective CSV files are copied
    into the folders. A txt-file is also created, which is just a list of input-feature-set names.
    """

    current_CL_all_combonations = list(
        combinations(X_list, combo_length))  # Creates all the combinations of the features
    if ftSet_names == []:  # Tests to see if the feature names were input beforehand
        X_ftName_list = np.arange(start=0, stop=numFeatSets, step=1)
        X_ftName_combonations = list(combinations(X_ftName_list, combo_length))
    else:
        X_ftName_combonations = list(combinations(ftSet_names, combo_length))

    numCombos_current = len(current_CL_all_combonations)

    for i in range(numCombos_current):
        X_feat_temp = current_CL_all_combonations[i]
        FtSetNames.append(X_ftName_combonations[i])
        sorting_var_list.append(sorting_var)
        sorting_var += 1

        # BELOW: Checks the length of the current feature combonation. If only one, no change. If greater than 1,
        #        combine the arrays into one large input
        if len(X_feat_temp) == 1:
            X_temp = X_feat_temp[0]
        elif len(X_feat_temp) > 1:
            X_temp = np.concatenate(X_feat_temp, axis=1)
        else:
            print("ERROR: NEED ONE OR MORE INPUT FEATURES")
        # BELOW: Checks to see if a parm or locDat array have been input. If not, continue. If they are, combine
        #        them for a larger input
        """
        if (parm_var and locData_var):
            X_feats_temp = np.concatenate((X_temp, parm), axis=1)
            X_feats_temp = np.concatenate((X_feats_temp, locDat), axis=1)
        elif parm_var:
            X_feats_temp = np.concatenate((X_temp, parm), axis=1)
        elif locData_var:
            X_feats_temp = np.concatenate((X_temp, locDat), axis=1)
        else:
            X_feats_temp = X_temp
        """
        X_feats_temp = X_temp

        list_of_input_feature_sets_fullData.append(X_feats_temp)
        # Removes NaN values and runs regressions
        print("COMBO-LENGTH: " + str(combo_length) + ", " + str(i + 1) + "/" + str(numCombos_current) + ": ",
              X_ftName_combonations[i])

        #progress_file = open('progress_file.txt', 'a')

        reg = Regression(X_feats_temp, Y_int, Nk=Nk, N=N,
                         seed=seed, goodIDs=goodIds, RemoveNaN=True, StandardizeX=True,
                         models_use=models_use, giveKFdata=False)

        results, bestPred = reg.RegressionCVMult()

        # Collects bestPred Data
        rmse_temp = bestPred['rmse']
        RMSE_list.append(rmse_temp)
        r2_temp = bestPred['r2']
        R2_list.append(r2_temp)
        """
        # RUN COARSE HEATMAP TO DETERMINE BEST INPUT FEATURES
        ht = heatmaps(X_feats_temp, Y_int, numLayers=2, numZooms=2, gridLength=5,
                      Nk=Nk, N=N,
                      goodIDs=goodIds, RemoveNaN=True, seed=seed,
                      models_use=models_use, save_csv_files=False)
        if models_use[0]:  # Linear
            C_input_data = [0.001, 100]
            e_input_data = [0.001, 1]
            g_input_data = [0.01, 2]
            storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_SVM_C_Epsilon_Gamma(C_input_data, e_input_data, g_input_data)
        elif models_use[1]:  # Poly2
            C_input_data = [0.001, 100]
            e_input_data = [0.001, 1]
            g_input_data = [0.01, 2]
            c0_input_data = [0.01, 10]
            storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_SVM_C_Epsilon_Gamma_Coef0(C_input_data, e_input_data, g_input_data, c0_input_data)
        elif models_use[2]:  # Poly3
            C_input_data = [0.001, 100]
            e_input_data = [0.001, 1]
            g_input_data = [0.01, 2]
            c0_input_data = [0.01, 10]
            storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_SVM_C_Epsilon_Gamma_Coef0(C_input_data, e_input_data, g_input_data, c0_input_data)
        elif models_use[3]:  # RBF
            C_input_data = [0.001, 100]
            e_input_data = [0.001, 1]
            g_input_data = [0.01, 2]
            storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_SVM_C_Epsilon_Gamma(C_input_data, e_input_data, g_input_data)
        elif models_use[4]:  # RatQuad
            noise_input_data = [0.001, 10]
            sigF_input_data = [0.01, 100]
            length_input_data = [0.01, 10]
            alpha_input_data = [0.01, 10]
            storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_GPR_Noise_SigF_Length_Alpha(noise_input_data, sigF_input_data, length_input_data, alpha_input_data)
        elif models_use[5]:  # RBF
            noise_input_data = [0.001, 10]
            sigF_input_data = [0.01, 100]
            length_input_data = [0.01, 10]
            storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_GPR_Noise_SigF_Length(noise_input_data, sigF_input_data, length_input_data)
        elif models_use[6]:  # Matern 3/2
            noise_input_data = [0.001, 10]
            sigF_input_data = [0.01, 100]
            length_input_data = [0.01, 10]
            storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_GPR_Noise_SigF_Length(noise_input_data, sigF_input_data, length_input_data)
        elif models_use[7]:  # Matern 5/2
            noise_input_data = [0.001, 10]
            sigF_input_data = [0.01, 100]
            length_input_data = [0.01, 10]
            storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_GPR_Noise_SigF_Length(noise_input_data, sigF_input_data, length_input_data)

        top_version = storage_df_sorted.iloc[0, :]
        rmse_temp = top_version['RMSE']
        RMSE_list.append(rmse_temp)
        r2_temp = top_version['R^2']
        R2_list.append(r2_temp)
        """
full_data_df = pd.DataFrame(columns=FtSetNames)
full_data_df.loc['ftSetNames'] = FtSetNames
full_data_df.loc['RMSE'] = RMSE_list
full_data_df.loc['R2'] = R2_list
full_data_df.loc['sorting_var'] = sorting_var_list

full_data_df.to_csv('full_input_feature_data.csv')

sorted_output = full_data_df.sort_values(by=['RMSE'], axis=1, ascending=True)
top_features_sorting_var = sorted_output.loc['sorting_var'][0:numTopFeatures]
top_features_ftNames = sorted_output.loc['ftSetNames'][0:numTopFeatures]
top_features_full_data = []
for i in top_features_sorting_var:
    top_features_full_data.append(list_of_input_feature_sets_fullData[i])

for i in range(len(top_features_full_data)):
    os.chdir(current_path)
    feature_data_current = top_features_full_data[i]
    feature_name_current = top_features_ftNames.iloc[i]
    folder_name_ftName = str(i) + "-" + str(feature_name_current)
    os.mkdir(folder_name_ftName)
    os.chdir(current_path + os.path.sep + folder_name_ftName)
    with open('feature_data.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(feature_data_current)
    csvfile.close()

os.chdir(current_path)
with open('top_ftSet_names.txt', 'w') as txtfile:
    for i in top_features_ftNames:
        txtfile.write(str(i) + " \n")
txtfile.close()



