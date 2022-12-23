# *********************** VERIRICATION OF HEATMAT OUTPUTS BASED ON THE INPUT DATA ************************************ #
""" METHOD """
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
from Heatmaps import *
from miscFunctions import *
os.chdir('..')
from import_and_organize_data import *
from config_inputs import *

property_num = 3

X_list = feature_set_list['org'][prop_keys[property_num]]

X1 = np.concatenate((X_list[0], X_list[2]), axis=1)
X1 = np.concatenate((X1, parmData), axis=1)
Y = property_data[:, property_num]
goodIDs = goodId_org[:, property_num]

mdl_SVM_linear = [1, 0, 0, 0, 0, 0, 0, 0]
mdl_SVM_poly2 =  [0, 1, 0, 0, 0, 0, 0, 0]
mdl_SVM_poly3 =  [0, 0, 1, 0, 0, 0, 0, 0]
mdl_SVM_rbf =    [0, 0, 0, 1, 0, 0, 0, 0]
mdl_GPR_rq =     [0, 0, 0, 0, 1, 0, 0, 0]
mdl_GPR_rbf =    [0, 0, 0, 0, 0, 1, 0, 0]
mdl_GPR_m32 =    [0, 0, 0, 0, 0, 0, 1, 0]
mdl_GPR_m52 =    [0, 0, 0, 0, 0, 0, 0, 1]

models_use = mdl_SVM_linear
seed = seeds[property_num]

numLayers = 2
numZooms = 2
gridLength = 20
# SVM
C_input_data = (0.001, 100)
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

ht = heatmaps(X1, Y, numLayers=numLayers, numZooms=numZooms, Nk=5, N=1, gridLength=gridLength,
              RemoveNaN=True, goodIDs=goodIDs, seed=seed,
              decimal_points_int=0.1, decimal_points_top=0.1,
              models_use=models_use, save_csv_files=False)

storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_AL_SVM_C_Epsilon(C_input_data, epsilon_input_data)
#storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_AL_SVM_C_Epsilon_Gamma(C_input_data, epsilon_input_data, gamma_input_data)
#storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_AL_SVM_C_Epsilon_Gamma_Coef0(C_input_data, epsilon_input_data, gamma_input_data, coef0_input_data)
#storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_AL_GPR_Noise_SigF_Length(noise_input_data, sigF_input_data, length_input_data)
#storage_df_unsorted, storage_df_sorted = ht.runFullGridSearch_AL_GPR_Noise_SigF_Length_Alpha(noise_input_data, sigF_input_data, length_input_data, alpha_input_data)

#storage_df, arrays = ht.runSingleGridSearch_AL_SVM_C_Epsilon(C_range, epsilon_range, 'figure')


best_data = storage_df_sorted.iloc[0, :]
C_inp = best_data['C']
epsilon_inp = best_data['Epsilon']
gamma_inp = best_data['Gamma']
coef0_inp = best_data['Coef0']
if coef0_inp == 'N/A':
    coef0_inp = 1

#C_inp = 1.13
#epsilon_inp = 1.935839
#gamma_inp = 0.06

# EXP1:
# C=1.4513777518588715
# Epsilon=1.951526607377184
# Gamma = 0.05

reg = Regression(X1, Y, C=C_inp, epsilon=epsilon_inp, gamma=gamma_inp, coef0=coef0_inp,
                 Nk=5, N=1, goodIDs=goodIDs, seed=seed, RemoveNaN=True, StandardizeX=True, models_use=models_use,
                 giveKFdata=True)
results, bestPred, kFold_data = reg.RegressionCVMult()


print('Completed')