import shutil
import sys
import os
import pickle

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


X_input = np.genfromtxt('feature_data.csv', delimiter=',')
Y_input = input_dict['Y_inp']
model_use = input_dict['models_use']
model_type = input_dict['model_type']
numLayers = input_dict['numLayers']
numZooms = input_dict['numZooms']
N = input_dict['N']
Nk = input_dict['Nk']
gridLength = input_dict['gridLength']
seed = input_dict['seed']
goodIds = input_dict['goodIDs']
hyperparameters = input_dict['hyperparameters']

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
    gamma_input_data = hyperparameters['g']

    ht.runFullGridSearch_SVM_C_Epsilon_Gamma(C_input_data, epsilon_input_data, gamma_input_data)

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

os.chdir('..')
