import random
import shutil
import sys
import os
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# IMPORT FUNCTIONS FROM "FUNCTION_FILES"****************************************************************************** #
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
sys.path.append(os.path.abspath(main_path + '/function_files/'))
from miscFunctions import miscFunctions
mf = miscFunctions()
from Regression import Regression
from Heatmaps import heatmaps
sys.path.append(os.path.abspath(current_path))

# ******************************************************************************************************************** #

X_int = np.genfromtxt('feature_data.csv', delimiter=',')
Y_int = input_dict['Y_inp']
model_use = input_dict['models_use']
model_type = input_dict['model_type']
model_name = input_dict['model_name']
N = input_dict['N']
Nk = input_dict['Nk']
seed = input_dict['seed']
goodIds = input_dict['goodIDs']
hyperparameters = input_dict['hyperparameters']

gridLength_GS = input_dict['gridLength_GS']
numZooms_GS = input_dict['numZooms_GS']
numLayers_GS = input_dict['numLayers_GS']
decimal_point_GS = input_dict['decimal_point_GS']
gridLength_AL = input_dict['gridLength_AL']
num_HP_zones_AL = input_dict['num_HP_zones_AL']
num_runs_AL = input_dict['num_runs_AL']
decimal_points_int = input_dict['decimal_points_int']
decimal_points_top = input_dict['decimal_points_top']


test_train_split_var = input_dict['test_train_split_var']
split_decimal = input_dict['split_decimal']
tr_ts_seed = input_dict['tr_ts_seed']

# For Test-Train Splitting
if test_train_split_var:

    X_noNaN, Y_noNaN = mf.RemoveNaN(X_int, Y_int, goodIds=goodIds)

    if tr_ts_seed == 'random':
        tr_ts_seed = random.randint(1, 2 ** 31)
        print("tr_ts_seed: ", tr_ts_seed)

    X_train, X_test, Y_train, Y_test = train_test_split(X_noNaN, Y_noNaN, test_size=split_decimal,
                                                        random_state=tr_ts_seed)

    X_input = X_train
    Y_input = Y_train

    RemoveNaN_var = False
    goodIds_var = None

else:
    X_input = X_int
    Y_input = Y_int

    RemoveNaN_var = True
    goodIds_var = goodIds

if 'Heatmaps' in os.listdir():
    shutil.rmtree('Heatmaps')
os.mkdir('Heatmaps')
os.chdir('Heatmaps')

# -------------------------------------- SVM --------------------------------------------------------------------------#
if model_use[0]:  # LINEAR
    C_input_data = hyperparameters['C']
    epsilon_input_data = hyperparameters['e']

    ht = heatmaps(X_input, Y_input, Nk=Nk, N=N,
                  num_HP_zones_AL=num_HP_zones_AL, num_runs_AL=num_runs_AL,
                  numLayers_GS=numLayers_GS, numZooms_GS=numZooms_GS,
                  gridLength_AL=gridLength_AL, gridLength_GS=gridLength_GS,
                  decimal_points_int=decimal_points_int, decimal_points_top=decimal_points_top,
                  decimal_point_GS=decimal_point_GS,
                  RemoveNaN=RemoveNaN_var, goodIDs=goodIds_var, seed=seed, models_use=model_use,
                  save_csv_files=True,
                  C_input=C_input_data, epsilon_input=epsilon_input_data, gamma_input='None', coef0_input='None',
                  noise_input='None', sigmaF_input='None', length_input='None', alpha_input='None')

    ht.runActiveLearning()

elif model_use[1]:  # POLY-2
    C_input_data = hyperparameters['C']
    epsilon_input_data = hyperparameters['e']
    gamma_input_data = hyperparameters['g']
    coef0_input_data = hyperparameters['c0']

    ht = heatmaps(X_input, Y_input, Nk=Nk, N=N,
                  num_HP_zones_AL=num_HP_zones_AL, num_runs_AL=num_runs_AL,
                  numLayers_GS=numLayers_GS, numZooms_GS=numZooms_GS,
                  gridLength_AL=gridLength_AL, gridLength_GS=gridLength_GS,
                  decimal_points_int=decimal_points_int, decimal_points_top=decimal_points_top,
                  decimal_point_GS=decimal_point_GS,
                  RemoveNaN=RemoveNaN_var, goodIDs=goodIds_var, seed=seed, models_use=model_use,
                  save_csv_files=True,
                  C_input=C_input_data, epsilon_input=epsilon_input_data, gamma_input=gamma_input_data, coef0_input=coef0_input_data,
                  noise_input='None', sigmaF_input='None', length_input='None', alpha_input='None')

    ht.runActiveLearning()

elif model_use[2]:  # POLY-3
    C_input_data = hyperparameters['C']
    epsilon_input_data = hyperparameters['e']
    gamma_input_data = hyperparameters['g']
    coef0_input_data = hyperparameters['c0']

    ht = heatmaps(X_input, Y_input, Nk=Nk, N=N,
                  num_HP_zones_AL=num_HP_zones_AL, num_runs_AL=num_runs_AL,
                  numLayers_GS=numLayers_GS, numZooms_GS=numZooms_GS,
                  gridLength_AL=gridLength_AL, gridLength_GS=gridLength_GS,
                  decimal_points_int=decimal_points_int, decimal_points_top=decimal_points_top,
                  decimal_point_GS=decimal_point_GS,
                  RemoveNaN=RemoveNaN_var, goodIDs=goodIds_var, seed=seed, models_use=model_use,
                  save_csv_files=True,
                  C_input=C_input_data, epsilon_input=epsilon_input_data, gamma_input=gamma_input_data,
                  coef0_input=coef0_input_data,
                  noise_input='None', sigmaF_input='None', length_input='None', alpha_input='None')

    ht.runActiveLearning()

elif model_use[3]:  # RBF
    C_input_data = hyperparameters['C']
    epsilon_input_data = hyperparameters['e']
    gamma_input_data = hyperparameters['g']

    ht = heatmaps(X_input, Y_input, Nk=Nk, N=N,
                  num_HP_zones_AL=num_HP_zones_AL, num_runs_AL=num_runs_AL,
                  numLayers_GS=numLayers_GS, numZooms_GS=numZooms_GS,
                  gridLength_AL=gridLength_AL, gridLength_GS=gridLength_GS,
                  decimal_points_int=decimal_points_int, decimal_points_top=decimal_points_top,
                  decimal_point_GS=decimal_point_GS,
                  RemoveNaN=RemoveNaN_var, goodIDs=goodIds_var, seed=seed, models_use=model_use,
                  save_csv_files=True,
                  C_input=C_input_data, epsilon_input=epsilon_input_data, gamma_input=gamma_input_data, coef0_input='None',
                  noise_input='None', sigmaF_input='None', length_input='None', alpha_input='None')

    ht.runActiveLearning()

# -------------------------------------- GPR --------------------------------------------------------------------------#

elif model_use[4]:  # RatQuad
    noise_input_data = hyperparameters['noise']
    sigF_input_data = hyperparameters['sigmaF']
    length_input_data = hyperparameters['scale_length']
    alpha_input_data = hyperparameters['alpha']

    ht = heatmaps(X_input, Y_input, Nk=Nk, N=N,
                  num_HP_zones_AL=num_HP_zones_AL, num_runs_AL=num_runs_AL,
                  numLayers_GS=numLayers_GS, numZooms_GS=numZooms_GS,
                  gridLength_AL=gridLength_AL, gridLength_GS=gridLength_GS,
                  decimal_points_int=decimal_points_int, decimal_points_top=decimal_points_top,
                  decimal_point_GS=decimal_point_GS,
                  RemoveNaN=RemoveNaN_var, goodIDs=goodIds_var, seed=seed, models_use=model_use,
                  save_csv_files=True,
                  C_input='None', epsilon_input='None', gamma_input='None', coef0_input='None',
                  noise_input=noise_input_data, sigmaF_input=sigF_input_data, length_input=length_input_data, alpha_input=alpha_input_data)

    ht.runActiveLearning()

elif model_use[5]:  # RBF
    noise_input_data = hyperparameters['noise']
    sigF_input_data = hyperparameters['sigmaF']
    length_input_data = hyperparameters['scale_length']

    ht = heatmaps(X_input, Y_input, Nk=Nk, N=N,
                  num_HP_zones_AL=num_HP_zones_AL, num_runs_AL=num_runs_AL,
                  numLayers_GS=numLayers_GS, numZooms_GS=numZooms_GS,
                  gridLength_AL=gridLength_AL, gridLength_GS=gridLength_GS,
                  decimal_points_int=decimal_points_int, decimal_points_top=decimal_points_top,
                  decimal_point_GS=decimal_point_GS,
                  RemoveNaN=RemoveNaN_var, goodIDs=goodIds_var, seed=seed, models_use=model_use,
                  save_csv_files=True,
                  C_input='None', epsilon_input='None', gamma_input='None', coef0_input='None',
                  noise_input=noise_input_data, sigmaF_input=sigF_input_data, length_input=length_input_data,
                  alpha_input='None')

    ht.runActiveLearning()

elif model_use[6]:  # Matern 3/2
    noise_input_data = hyperparameters['noise']
    sigF_input_data = hyperparameters['sigmaF']
    length_input_data = hyperparameters['scale_length']

    ht = heatmaps(X_input, Y_input, Nk=Nk, N=N,
                  num_HP_zones_AL=num_HP_zones_AL, num_runs_AL=num_runs_AL,
                  numLayers_GS=numLayers_GS, numZooms_GS=numZooms_GS,
                  gridLength_AL=gridLength_AL, gridLength_GS=gridLength_GS,
                  decimal_points_int=decimal_points_int, decimal_points_top=decimal_points_top,
                  decimal_point_GS=decimal_point_GS,
                  RemoveNaN=RemoveNaN_var, goodIDs=goodIds_var, seed=seed, models_use=model_use,
                  save_csv_files=True,
                  C_input='None', epsilon_input='None', gamma_input='None', coef0_input='None',
                  noise_input=noise_input_data, sigmaF_input=sigF_input_data, length_input=length_input_data,
                  alpha_input='None')

    ht.runActiveLearning()

elif model_use[7]:  # Matern 5/2
    noise_input_data = hyperparameters['noise']
    sigF_input_data = hyperparameters['sigmaF']
    length_input_data = hyperparameters['scale_length']

    ht = heatmaps(X_input, Y_input, Nk=Nk, N=N,
                  num_HP_zones_AL=num_HP_zones_AL, num_runs_AL=num_runs_AL,
                  numLayers_GS=numLayers_GS, numZooms_GS=numZooms_GS,
                  gridLength_AL=gridLength_AL, gridLength_GS=gridLength_GS,
                  decimal_points_int=decimal_points_int, decimal_points_top=decimal_points_top,
                  decimal_point_GS=decimal_point_GS,
                  RemoveNaN=RemoveNaN_var, goodIDs=goodIds_var, seed=seed, models_use=model_use,
                  save_csv_files=True,
                  C_input='None', epsilon_input='None', gamma_input='None', coef0_input='None',
                  noise_input=noise_input_data, sigmaF_input=sigF_input_data, length_input=length_input_data,
                  alpha_input='None')

    ht.runActiveLearning()

if test_train_split_var:

    if model_type == 'SVM':
        col_names = ['RMSE', 'R2', 'Cor', 'C', 'Epsilon', 'Gamma', 'Coef0',
                     'kF-Training-RMSE', 'kF-Training-R2', 'kF-Training-Cor',
                     'Average-Training-to-Final-Training-Ratio']
    elif model_type == 'GPR':
        col_names = ['RMSE', 'R2', 'Cor', 'Noise', 'Length', 'SigmaF', 'Alpha',
                     'kF-Training-RMSE', 'kF-Training-R2', 'kF-Training-Cor',
                     'Average-Training-to-Final-Training-Ratio']

    top_models_df = pd.DataFrame(columns=col_names)

    for i in range(2):
        sorted_data_df = pd.read_csv("0-Data_" + model_name + "_Sorted.csv")
        best_model_data = sorted_data_df.iloc[i, :]

        BM_RMSE = best_model_data['RMSE']
        BM_R2 = best_model_data['R^2']
        BM_Cor = best_model_data['Cor']
        BM_avgTR_to_finalTR_Error = best_model_data['avgTR to Final Error']

        if model_type == 'SVM':
            C = best_model_data['C']
            Epsilon = best_model_data['Epsilon']
            Coef0 = best_model_data['Coef0']
            Gamma = best_model_data['Gamma']

            Alpha = 1
            Scale_Length = 1
            Noise = 1
            Sigma_F = 1

        if model_type == 'GPR':
            if model_use[4]:
                Alpha = best_model_data['Alpha']
            else:
                Alpha = 1
            Scale_Length = best_model_data['Length']
            Noise = best_model_data['Noise']
            Sigma_F = best_model_data['Sigma_F']

            C = 1
            Epsilon = 1
            Coef0 = 1
            Gamma = 1

        reg = Regression(X_test, Y_test, models_use=model_use, seed=seed, RemoveNaN=False,
                         C=C, epsilon=Epsilon, gamma=Gamma, coef0=Coef0, noise=Noise, sigma_F=Sigma_F,
                         scale_length=Scale_Length, alpha=Alpha)

        results = reg.Regression_test_multProp(X_train, X_test, Y_train, Y_test)

        rmse_temp = results['metrics'].loc[model_name]["RMSE"]
        r2_temp = results['metrics'].loc[model_name]["R2"]
        cor_temp = results['metrics'].loc[model_name]["Cor"]

        if model_type == 'SVM':
            row_data = [rmse_temp, r2_temp, cor_temp, C, Epsilon, Gamma, Coef0, BM_RMSE, BM_R2, BM_Cor,
                        BM_avgTR_to_finalTR_Error]
        if model_type == 'GPR':
            row_data = [rmse_temp, r2_temp, cor_temp, Noise, Scale_Length, Sigma_F, Alpha, BM_RMSE, BM_R2, BM_Cor,
                        BM_avgTR_to_finalTR_Error]

        top_models_df.loc[str(i)] = row_data

    top_models_df = top_models_df.sort_values(by=["RMSE"], ascending=True)
    top_models_df.to_csv('Best_Models.csv')

os.chdir('..')
