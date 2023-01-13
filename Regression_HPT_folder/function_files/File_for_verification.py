# *********************** VERIRICATION OF HEATMAT OUTPUTS BASED ON THE INPUT DATA ************************************ #
""" METHOD """
import shutil

"""
This verification test is of the outputs given by the 'Heatmaps' script. 

"""

# Importing libraries --------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import numpy as np
import os
import sys
import pickle
# SkLearn
import sklearn
from sklearn import svm, tree, linear_model, metrics, pipeline, preprocessing
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RationalQuadratic, RBF
from scipy.stats import iqr

from Regression import *
from Heatmaps_old import *
from Heatmaps import *
from miscFunctions import *
os.chdir('..')
from import_and_organize_data import *
from config_inputs import *

property_num = props_to_run[0]
model_num = models_to_run[0]

X_list = feature_set_list['org'][prop_keys[property_num]]

#X1 = np.concatenate((X_list[0], X_list[2]), axis=1)
#X1 = np.concatenate((X1, parmData), axis=1)
#X1 = np.concatenate(X_list[(0,1,2,12), :, :], axis=1)
X1 = np.concatenate([X_list[0], X_list[1], X_list[2], X_list[4]], axis=1)
Y = property_data[:, property_num]
goodIDs = goodId_org[:, property_num]
sId_use = sId[goodIDs.astype(bool)]
sTpInt_use = sTpInt[goodIDs.astype(bool)]

mdl_num = 5

mdl_SVM_linear = [1, 0, 0, 0, 0, 0, 0, 0]
mdl_SVM_poly2 =  [0, 1, 0, 0, 0, 0, 0, 0]
mdl_SVM_poly3 =  [0, 0, 1, 0, 0, 0, 0, 0]
mdl_SVM_rbf =    [0, 0, 0, 1, 0, 0, 0, 0]
mdl_GPR_rq =     [0, 0, 0, 0, 1, 0, 0, 0]
mdl_GPR_rbf =    [0, 0, 0, 0, 0, 1, 0, 0]
mdl_GPR_m32 =    [0, 0, 0, 0, 0, 0, 1, 0]
mdl_GPR_m52 =    [0, 0, 0, 0, 0, 0, 0, 1]
model_use_list = [mdl_SVM_linear, mdl_SVM_poly2, mdl_SVM_poly3, mdl_SVM_rbf, mdl_GPR_rq, mdl_GPR_rbf, mdl_GPR_m32, mdl_GPR_m52]

models_use = model_use_list[model_num]
model_name_use = model_names[model_num]
seed = seeds[property_num]

numLayers = 2
numZooms = 2
gridLength = 20
# SVM
C_input_data = (0.001, 0.01)
C_range = np.linspace(C_input_data[0], C_input_data[1], gridLength)
epsilon_input_data = (0.001, 0.01)
epsilon_range = np.linspace(epsilon_input_data[0], epsilon_input_data[1], gridLength)
gamma_input_data = (0.001, 0.1)
coef0_input_data = (0.001, 10)
# GPR
noise_input_data = (0.001, 10)
sigF_input_data = (0.1, 100)
length_input_data = (0.001, 10)
alpha_input_data = (0.001, 10)

"""
os.chdir('../../new_AL_folder')
folder_name = 'SVM_RBF_GS_Update'
if folder_name in os.listdir():
    shutil.rmtree(folder_name)
os.mkdir(folder_name)
os.chdir(folder_name)

print("HERE: ", os.getcwd())

ht2 = heatmaps(X1, Y, Nk=Nk, N=N,
                num_HP_zones_AL=2, num_runs_AL=2,
                numLayers_GS=3, numZooms_GS=2,
                gridLength_AL=8, gridLength_GS=5,
                decimal_points_int=0.05, decimal_points_top=0.1,
                decimal_point_GS=0.1,
                RemoveNaN=True, goodIDs=goodIDs, seed=seed, models_use=models_use,
                save_csv_files=True,
                C_input='None', epsilon_input='None', gamma_input='None', coef0_input='None',
                noise_input=noise_input_data, sigmaF_input=sigF_input_data, length_input=length_input_data, alpha_input='None')

storage_df = ht2.runActiveLearning()

"""


#ht = heatmaps(X1, Y, numLayers=numLayers, numZooms=numZooms, Nk=5, N=1, gridLength=gridLength,
#              RemoveNaN=True, goodIDs=goodIDs, seed=seed,
#              decimal_points_int=0.1, decimal_points_top=0.1,
#              models_use=models_use, save_csv_files=False)

#storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_AL_SVM_C_Epsilon(C_input_data, epsilon_input_data)
#storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_AL_SVM_C_Epsilon_Gamma(C_input_data, epsilon_input_data, gamma_input_data)
#storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_AL_SVM_C_Epsilon_Gamma_Coef0(C_input_data, epsilon_input_data, gamma_input_data, coef0_input_data)
#storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_AL_GPR_Noise_SigF_Length(noise_input_data, sigF_input_data, length_input_data)
#storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_AL_GPR_Noise_SigF_Length_Alpha(noise_input_data, sigF_input_data, length_input_data, alpha_input_data)

#storage_df, arrays = ht.runSingleGridSearch_AL_SVM_C_Epsilon(C_range, epsilon_range, 'figure')


#best_data = storage_df_sorted.iloc[0, :]
#C_inp = best_data['C']
#epsilon_inp = best_data['Epsilon']
#gamma_inp = best_data['Gamma']
#coef0_inp = best_data['Coef0']
#if coef0_inp == 'N/A':
#    coef0_inp = 1

#C_inp = 1.13
#epsilon_inp = 1.935839
#gamma_inp = 0.06

# EXP1:

C = 0.243873205608901
Epsilon = 0.121475234981663
Gamma = 0.114004404100213
Coef0 = 6.2102011915197

Noise = 3.750925
SigmaF = 0.313923116995073
Scale_Length = 0.400375454433497
Alpha = 1


reg = Regression(X1, Y,
                 C=C, epsilon=Epsilon, gamma=Gamma, coef0=Coef0,
                 noise=Noise, sigma_F=SigmaF, scale_length=Scale_Length, alpha=Alpha,
                 Nk=5, N=1, goodIDs=goodIDs, seed=seed, RemoveNaN=True, StandardizeX=True, models_use=models_use,
                 giveKFdata=True, random_state=1000)

results, bestPred, kFold_data = reg.RegressionCVMult()

"""
hp_list = temp_hp_list[model_num]
reg = Regression(X1, Y,
                 C=hp_list[0], epsilon=hp_list[1], gamma=hp_list[2], coef0=hp_list[3],
                 noise=hp_list[0], sigma_F=hp_list[1], scale_length=hp_list[2], alpha=hp_list[3],
                 Nk=5, N=1, goodIDs=goodIDs, seed=seed, RemoveNaN=True, StandardizeX=True, models_use=models_use,
                 giveKFdata=True)
results, bestPred, kFold_data = reg.RegressionCVMult()


os.chdir('../../TR_TS_data')
filename = prop_names[property_num]+"_"+model_names[model_num]+'.csv'
zeros = np.zeros((5, 5))
ratio_value = np.average(kFold_data['tr']['results']['variation_#1']['rmse']) / results['rmse'].iloc[0,0]
col_names = ['TR_RMSE', "TR_R2", "TS_RMSE", "TS_R2", "Ratio"]
df = pd.DataFrame(data=zeros, columns=col_names)
df.iloc[:, 0] = kFold_data['tr']['results']['variation_#1']['rmse']
df.iloc[:, 1] = kFold_data['tr']['results']['variation_#1']['r2']
df.iloc[:, 2] = kFold_data['ts']['results']['variation_#1']['rmse']
df.iloc[:, 3] = kFold_data['ts']['results']['variation_#1']['r2']
df.iloc[:, 4] = [1/ratio_value, 'RMSE', results['rmse'].iloc[0,0], 'R2', results['r2'].iloc[0,0]]
df.to_csv(filename)
"""

"""
rmse_temp = results['rmse'].loc[model_name_use]
r2_temp = results['r2'].loc[model_name_use]
Yp_use = results['Yp'][model_name_use]['variation_#1']
Npts = len(Yp_use)

mf = miscFunctions()
train_idx, test_idx = mf.getKfoldSplits(X1, Nk=5, seed=seed)

# Create DataFrame with final data
col_names = ['sId', 'sTpInt', 'Partition', 'Y_true', 'Y_pred']
data_zeros = np.zeros((Npts+1, len(col_names)))
final_df = pd.DataFrame(data=data_zeros, columns=col_names)
for i in range(Npts):
    sId_temp = sId_use[i][0]
    sTpInt_temp = sTpInt_use[i][0]
    Yt_temp = Y[i][0]
    Yp_temp = Yp_use[i][0]
    for p in range(Nk):
        if i in test_idx[0]:
            partition_temp = 1
        elif i in test_idx[1]:
            partition_temp = 2
        elif i in test_idx[2]:
            partition_temp = 3
        elif i in test_idx[3]:
            partition_temp = 4
        elif i in test_idx[4]:
            partition_temp = 5

    res = [int(sId_temp), int(sTpInt_temp), int(partition_temp), Yt_temp, Yp_temp]

    final_df.iloc[i, :] = res

metrics_use = ['Metics: ', 'RMSE', float(rmse_temp),"R^2", float(r2_temp)]
print(metrics_use)
final_df.iloc[-1, :] = metrics_use

os.chdir('../../Final_pred-2022_12_23/'+str(prop_names[property_num]))
filename = prop_names[property_num]+"_and_"+model_names[model_num]+".csv"
final_df.to_csv(filename)

print("HERE: ", os.getcwd())
"""


print('Completed')