import shutil
import sys
import os
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd

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
sys.path.append(os.path.abspath(main_path+'/function_files/'))
from miscFunctions import *
from Regression import *
from Heatmaps import *
sys.path.append(os.path.abspath(current_path))

# ******************************************************************************************************************** #


X_int = np.genfromtxt('feature_data.csv', delimiter=',')
Y_int = input_dict['Y_inp']
model_use = input_dict['models_use']
model_type = input_dict['model_type']
model_name = input_dict['model_name']
numLayers = input_dict['numLayers']
numZooms = input_dict['numZooms']
N = input_dict['N']
Nk = input_dict['Nk']
gridLength = input_dict['gridLength']
seed = input_dict['seed']
goodIds = input_dict['goodIDs']
hyperparameters = input_dict['hyperparameters']

test_train_split_var = input_dict['test_train_split_var']
split_decimal = input_dict['split_decimal']
tr_ts_seed = input_dict['tr_ts_seed']

# For Test-Train Splitting
if test_train_split_var:
    X_train, X_test, Y_train, Y_test = test_train_split(X_int, Y_int, test_size=split_decimal, random_state=tr_ts_seed)
    
    X_input = X_train
    Y_input = Y_train
else:
    X_input = X_int
    Y_input = Y_int


ht = heatmaps(X_input, Y_input,
              N=N, Nk=Nk,
              numLayers=numLayers, numZooms=numZooms, gridLength=gridLength,
              RemoveNaN=True, goodIDs=goodIds, seed=seed, models_use=model_use)

if 'Heatmaps' in os.listdir():
    shutil.rmtree('Heatmaps')
os.mkdir('Heatmaps')
os.chdir('Heatmaps')

# -------------------------------------- SVM --------------------------------------------------------------------------#
if model_use[0]:  # LINEAR
    C_input_data = hyperparameters['C']
    epsilon_input_data = hyperparameters['e']

    ht.runFullGridSearch_SVM_C_Epsilon(C_input_data, epsilon_input_data)

elif model_use[1]:  # POLY-2
    C_input_data = hyperparameters['C']
    epsilon_input_data = hyperparameters['e']
    gamma_input_data = hyperparameters['g']
    coef0_input_data = hyperparameters['c0']

    ht.runFullGridSearch_SVM_C_Epsilon_Gamma_Coef0(C_input_data, epsilon_input_data, gamma_input_data, coef0_input_data)

elif model_use[2]:  # POLY-3
    C_input_data = hyperparameters['C']
    epsilon_input_data = hyperparameters['e']
    gamma_input_data = hyperparameters['g']
    coef0_input_data = hyperparameters['c0']

    ht.runFullGridSearch_SVM_C_Epsilon_Gamma_Coef0(C_input_data, epsilon_input_data, gamma_input_data, coef0_input_data)

elif model_use[3]:  # RBF
    C_input_data = hyperparameters['C']
    epsilon_input_data = hyperparameters['e']
    gamma_input_data = hyperparameters['g']

    ht.runFullGridSearch_SVM_C_Epsilon_Gamma(C_input_data, epsilon_input_data, gamma_input_data)

# -------------------------------------- GPR --------------------------------------------------------------------------#

elif model_use[4]:  # RatQuad
    noise_input_data = hyperparameters['noise']
    sigF_input_data = hyperparameters['sigmaF']
    length_input_data = hyperparameters['scale_length']
    alpha_input_data = hyperparameters['alpha']

    ht.runFullGridSearch_GPR_Noise_SigF_Length_Alpha(noise_input_data, sigF_input_data, length_input_data, alpha_input_data)

elif model_use[5]:  # RBF
    noise_input_data = hyperparameters['noise']
    sigF_input_data = hyperparameters['sigmaF']
    length_input_data = hyperparameters['scale_length']

    ht.runFullGridSearch_GPR_Noise_SigF_Length(noise_input_data, sigF_input_data, length_input_data)

elif model_use[6]:  # Matern 3/2
    noise_input_data = hyperparameters['noise']
    sigF_input_data = hyperparameters['sigmaF']
    length_input_data = hyperparameters['scale_length']

    ht.runFullGridSearch_GPR_Noise_SigF_Length(noise_input_data, sigF_input_data, length_input_data)

elif model_use[7]:  # Matern 5/2
    noise_input_data = hyperparameters['noise']
    sigF_input_data = hyperparameters['sigmaF']
    length_input_data = hyperparameters['scale_length']

    ht.runFullGridSearch_GPR_Noise_SigF_Length(noise_input_data, sigF_input_data, length_input_data)

if test_train_split_var:

    if model_type == 'SVM':
        col_names = ['RMSE', 'R2', 'Cor', 'C', 'Epsilon', 'Gamma', 'Coef0', 'kF-Training-RMSE', 'kF-Training-R2', 'kF-Training-Cor']
    elif model_type == 'GPR':
        col_names = ['RMSE', 'R2', 'Cor', 'Noise', 'Length', 'SigmaF', 'Alpha']
    
    top_models_df = pd.DataFrame(columns=col_names)
    
    for i in range(5):
        sorted_data_df = pd.read_csv("0-Data_"+model_name+"_Sorted.csv")
        best_model_data = sorted_data_df.iloc[i,:]
        
        BM_RMSE = best_model_data['RMSE']
        BM_R2 = best_model_data['R2']
        BM_Cor = best_model_data['cor']
        
        
        if model_type == 'SVM':
            C = best_model_data['C']
            Epsilon = best_model_data['Epsilon']
            Coef0 = best_model_data['Coef0']
            Gamma = best_model_data['Gamma']
            
            Alpha = 'N/A'
            Scale_Length = 'N/A'
            Noise = 'N/A'
            Sigma_F = 'N/A'
            
        if model_type == 'GPR':
            if model_use[4]:
                Alpha = best_model_data['Alpha']
            Scale_Length = best_model_data['Length']
            Noise = best_model_data['Noise']
            Sigma_F = best_model_data['Sigma_F']
            
            C = 'N/A'
            Epsilon = 'N/A'
            Coef0 = 'N/A'
            Gamma = 'N/A'
        
        reg = Regression(X_input, Y_input, models_use=model_use, seed=seed, RemoveNaN=False, 
                         C=C, epsilon=Epsilon, gamma=Gamma, coef0=Coef0, noise=Noise, sigma_F=Sigma_F, scale_length=Scale_Length, alpha=Alpha)
        
        results = Regression_test_multProp(X_train, X_test, Y_train, Y_test)
        
        rmse_temp = results['metrics'][model_name][0]
        r2_temp = results['metrics'][model_name][1]
        cor_temp = results['metrics'][model_name][2]
        
        if model_type == 'SVM':
            row_data = [rmse_temp, r2_temp, cor_temp, C, Epsilon, Gamma, Coef0, BM_RMSE, BM_R2, BM_Cor]
        if model_type == 'GPR':
            row_data = [rmse_temp, r2_temp, cor_temp, Noise, Scale_Length, Sigma_F, Alpha, BM_RMSE, BM_R2, BM_Cor]
        
        top_models_df.loc[str(i)]  = row_data
    
    top_models_df = top_models_df.sort_values(by=["RMSE"], ascending=True)
    top_models_df.to_csv('Best_Models.csv')
    
    
os.chdir('..')
