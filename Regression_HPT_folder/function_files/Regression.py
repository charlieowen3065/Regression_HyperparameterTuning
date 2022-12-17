# Basic
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
from sklearn.model_selection import train_test_split
from scipy.stats import iqr

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
sys.path.append(os.path.abspath(current_path))

# ******************************************************************************************************************** #

import warnings

warnings.filterwarnings("ignore")

print("MAIN PATH: ", os.getcwd())

class Regression():
    def __init__(self, X_inp, Y_inp,
                 C='default', epsilon='default', gamma=0.2, coef0=1, noise=0.5, sigma_F=0.1, scale_length=1, alpha=1,
                 Nk=5, N=1, goodIDs=None, seed='random',
                 RemoveNaN=True, StandardizeX=True, models_use='all', giveKFdata=False):
        """ INPUT VARIABLES """
        ################################################################################################################
        # X & Y inputs:
        # - X_inp: (n x m) array of input features
        # - Y_inp: (n x 1) array of input features
        # Hyperparameters:
        #   {SVMs}
        # - C: [SVM] // BoxConstraint
        # - epsilon: [SVM]
        # - gamma: [SVM]
        # - coef0: [SVM]
        #   {GPRs}
        # - noise: [GPR]
        # - sigma_F: [GPR]
        # - scale_length: [GPR]
        # - alpha: [GPR]
        # Non-Choice Variables:
        # - Nk: number of kFold-cross-validations
        # - N: number of itterations
        # - goodIDs: logical array indicating if a row needs to be removed due to NaN/other situations
        # - seed: either 'random' or list of seed values for the kFold-data split
        # Choice Variables:
        # - RemovesNaN: boolean, determines if the 'RemoveNaN' function is called on X & Y
        # - StandardizeX: boolean, determines if X is standardized
        # - models_use: logical-array, determines which models to run
        # - giveKFdata: determines is the kF-data is output
        ################################################################################################################
        """ CLASS ATRIBUTES """
        ################################################################################################################
        # - mF == class of functions, 'miscFunctions'
        #
        #
        ################################################################################################################

        self.mF = miscFunctions()
        self.giveKFdata = giveKFdata

        # Reworks X & Y ------------------------------------------------------------------------------------------------
        if RemoveNaN == True:
            X_original, self.Y = self.mF.RemoveNaN(X_inp, Y_inp, goodIds=goodIDs)
        else:
            X_original, self.Y = self.X_inp, self.Y_inp

        if StandardizeX == True:
            scaler = preprocessing.StandardScaler()
            self.X = scaler.fit_transform(X_original)
        else:
            self.X = X_original

        self.Npts = len(self.Y)
        self.Nfts = len(self.X[0, :])

        # HYPERPARAMETERS ----------------------------------------------------------------------------------------------
        # Default Values
        rv = iqr(self.Y)
        C_def = rv / 1.349
        e_def = rv/13.49
        # SVM-Hyperparameters
        if C == 'default':
            self.C = C_def
        else:
            self.C = C
        if epsilon == 'default':
            self.epsilon = e_def
        else:
            self.epsilon = epsilon
        self.gamma = gamma
        self.coef0 = coef0
        # GPR-Hyperparameters
        self.noise = noise
        self.scale_length = scale_length
        self.alpha = alpha
        self.sigma_F = sigma_F

        # MODELS -------------------------------------------------------------------------------------------------------
        self.Nk = Nk
        self.N = N

        self.models_use = models_use
        self.model_names, self.model_list = self.getModels()
        self.numMdls = len(self.model_list)

        # TRAIN-TEST INDICES -------------------------------------------------------------------------------------------
        # If seed is an integer, we need to create a list N long starting with this value
        if type(seed) == int:
            seed_int = seed
            seed = []
            for i in range(N):
                seed.append(seed_int + i) # makes sure all the seeds are different
        # Else if seed is 'random', we will need to create a list 'N' long full of 'random'
        elif seed == 'random':
            seed = []
            for i in range(N):
                seed.append('random')
        # Else, seed = seed

        # Goes through each seed value and adds the set of train-test pairs to a list
        self.train_idx = []
        self.test_idx = []
        for i in range(N):
            tr_idx, ts_idx = self.mF.getKfoldSplits(self.X, Nk=Nk, seed=seed[i])
            self.train_idx.append(tr_idx)
            self.test_idx.append(ts_idx)

    def getModels(self):
        """ METHODS, INPUTS, & OUTPUTS"""
        ################################################################################################################
        # METHODS:
        # calls the atributes of the class:
        #   - models_use
        #   - C, epsilon, gamma, coef 0
        #   - noise, sigmaF, scale_length, alpha
        # and creates a list with the specified models:
        #   - SVM_Linear
        #   - SVM_Poly2
        #   - SVM_Poly3
        #   - SVM_RBG
        #   - GPR_RationalQuadratic
        #   - GPR_SquaredExponential (RBF)
        #   - GPR_Matern_3/2
        #   - GPR_Matern_5/2
        # then goes through 'models_use' to determine which models to keep
        # INPUTS:
        # - none (only calls on class atributes)
        # OUTPUTS:
        # - model_names: list of the model names that are used
        # - model_list: list of the actual models being used
        ################################################################################################################

        model_list = []
        model_names = []

        # GETTING THE MODELS -------------------------------------------------------------------------------------------
        """ SVM MODELS """
        model_names.append('SVM_Linear')
        model_list.append(svm.SVR(kernel='linear', C=self.C, epsilon=self.epsilon, coef0=self.coef0))

        model_names.append("SVM_Poly2")
        model_list.append(
            svm.SVR(kernel='poly', degree=2, C=self.C, gamma=self.gamma, epsilon=self.epsilon, coef0=self.coef0))

        model_names.append("SVM_Poly3")
        model_list.append(
            svm.SVR(kernel='poly', degree=3, C=self.C, gamma=self.gamma, epsilon=self.epsilon, coef0=self.coef0))

        model_names.append('SVM_RBF')
        model_list.append(svm.SVR(kernel='rbf', C=self.C, gamma=self.gamma, epsilon=self.epsilon, coef0=self.coef0))

        sig_F_sqrd = ConstantKernel(self.sigma_F ** 2)
        noise = WhiteKernel(noise_level=self.noise)

        """ GAUSSIAN MODELS """
        model_names.append("GPR_RationalQuadratic")
        kernel_use = (RationalQuadratic(length_scale=self.scale_length, alpha=self.alpha) * sig_F_sqrd) + noise
        model_list.append(gaussian_process.GaussianProcessRegressor(kernel=kernel_use))

        model_names.append("GPR_RBF")
        kernel_use = (RBF(length_scale=self.scale_length) * sig_F_sqrd) + noise
        model_list.append(gaussian_process.GaussianProcessRegressor(kernel=kernel_use))

        model_names.append("GPR_Matern32")
        kernel_use = (Matern(nu=3 / 2, length_scale=self.scale_length) * sig_F_sqrd) + noise
        model_list.append(gaussian_process.GaussianProcessRegressor(kernel=kernel_use))

        model_names.append("GPR_Matern52")
        kernel_use = (Matern(nu=5 / 2, length_scale=self.scale_length) * sig_F_sqrd) + noise
        model_list.append(gaussian_process.GaussianProcessRegressor(kernel=kernel_use))

        # CHECKS TO SEE WHICH MODELS TO USE IN THE RUN -----------------------------------------------------------------
        if self.models_use != 'all':
            model_names_temp = []
            model_list_temp = []
            for i in range(len(self.models_use)):
                if self.models_use[i] == 1:
                    model_names_temp.append(model_names[i])
                    model_list_temp.append(model_list[i])
            model_names = model_names_temp
            model_list = model_list_temp

        return model_names, model_list

    def runSingleModel(self, model, id):
        """ METHODS, INPUTS, & OUTPUTS"""
        ################################################################################################################
        # METHODS:
        # Import the correct set of train-test indices, then split the X and Y respectivly.
        # This function runs a Nk-Fold Cross Validation on the data. It stores the prediction
        # of each of the Nk-testing sets in their respective spot in the "Yp" array. Then,
        # both the testing and training metric for each set in 'kf_data'
        # INPUTS:
        # - model: this is the specific model that is being used
        # - id: the variation number (self.N)
        # OUTPUTS:
        # - Yp: full vector of predictions from the Nk-Fold CV
        # - kf_data: holds all the training and testing metrics
        ################################################################################################################

        Y = self.Y
        X = self.X

        numPts = self.Npts

        Yp = np.zeros((numPts, 1))
        kf_data = dict()
        kf_data['tr'] = np.zeros((3, self.Nk))
        kf_data['ts'] = np.zeros((3, self.Nk))

        for i in range(self.Nk):
            # Sets up train-test data ----------------------------------------------------------------------------------
            tr_idx = self.train_idx[id][i]
            ts_idx = self.test_idx[id][i]

            Ytr = Y[tr_idx]
            Yts = Y[ts_idx]
            Xtr = X[tr_idx]
            Xts = X[ts_idx]

            numTrPts = len(Ytr)  # number of training data points
            numTsPts = len(Yts)  # number of testing data points

            Ytr.shape = (numTrPts,)

            # Runs model -----------------------------------------------------------------------------------------------
            mdl = model.fit(Xtr, Ytr)  # fits the training data
            Yp_temp = mdl.predict(Xts)  # predicts using X-testing data
            Yp_temp.shape = (numTsPts, 1)  # reshapes Yp_temp to the correct size (n x 1)

            Yp[ts_idx, 0] = Yp_temp[:, 0]  # adds predicted-Y to Yp array

            # kFold Data Storage ---------------------------------------------------------------------------------------
            Yp_temp_tr = mdl.predict(Xtr)  # finds the predicted Y from the training data
            Yp_temp_ts = Yp_temp  # finds the predicted Y from the testing data

            shape_ts = (numTsPts, 1)
            shape_tr = (numTrPts, 1)

            Yts.shape = shape_ts  # reshapes Yts
            Yp_temp_ts.shape = shape_ts  # reshapes Yp_ts
            Ytr.shape = shape_tr  # reshapes Ytr
            Yp_temp_tr.shape = shape_tr  # reshapes Yp_tr

            rmse_kF_ts, r2_kF_ts, cor_kF_ts = self.mF.getPredMetrics(Yts, Yp_temp_ts)  # metrics of the testing data
            rmse_kF_tr, r2_kF_tr, cor_kF_tr = self.mF.getPredMetrics(Ytr, Yp_temp_tr)  # metrics of the training data
            kf_data['ts'][0, i] = rmse_kF_ts
            kf_data['ts'][1, i] = r2_kF_ts
            kf_data['ts'][2, i] = cor_kF_ts
            kf_data['tr'][0, i] = rmse_kF_tr
            kf_data['tr'][1, i] = r2_kF_tr
            kf_data['tr'][2, i] = cor_kF_tr

        return Yp, kf_data

    def RegressionCV(self, id):
        """ METHODS, INPUTS, & OUTPUTS"""
        ################################################################################################################
        # METHODS:

        Y = self.Y
        X = self.X

        # Storage for Prediction Metrics -------------------------------------------------------------------------------
        rmse = np.zeros((self.numMdls, 1))
        r2 = np.zeros((self.numMdls, 1))
        cor = np.zeros((self.numMdls, 1))
        Yp = dict()

        # Storage for kFold-Data ---------------------------------------------------------------------------------------
        # Top level
        kF_data = dict()
        kF_data['ts'] = dict()
        kF_data['tr'] = dict()
        # Temporary Arrays
        # *** TESTING *** #
        kF_rmse_ts_array = np.zeros((self.Nk, self.numMdls))  # temp array for testing-RMSE
        kF_r2_ts_array = np.zeros((self.Nk, self.numMdls))  # temp array for testing-R2
        kF_cor_ts_array = np.zeros((self.Nk, self.numMdls))  # temp array for testing-Cor
        # *** TRAINING *** #
        kF_rmse_tr_array = np.zeros((self.Nk, self.numMdls))  # temp array for training-RMSE
        kF_r2_tr_array = np.zeros((self.Nk, self.numMdls))  # temp array for training-R2
        kF_cor_tr_array = np.zeros((self.Nk, self.numMdls))  # temp array for training-Cor

        # RUNNING THE MODELS -------------------------------------------------------------------------------------------
        for i in range(self.numMdls):  # Iterates over all models
            # Run Current Model
            model = self.model_list[i]  # current model
            model_name = self.model_names[i]  # current model name

            Yp_temp, kf_data_temp = self.runSingleModel(model, id)  # runs current model

            Y_shape = (self.Npts, 1)  # Reshapes/ensures that Y and Yp are the same shape
            Y.shape = Y_shape
            Yp_temp.shape = Y_shape

            rmse_temp, r2_temp, cor_temp = self.mF.getPredMetrics(Y, Yp_temp)  # get the metrics from model

            # Adds data to final/temporary arrays
            # *** Model-Metircs ** #
            rmse[i, 0] = rmse_temp
            r2[i, 0] = r2_temp
            cor[i, 0] = cor_temp
            Yp[str(model_name)] = Yp_temp
            # *** kFold-Data *** #
            # Testing Data
            kF_rmse_ts_array[:, i] = kf_data_temp['ts'][0, :]
            kF_r2_ts_array[:, i] = kf_data_temp['ts'][1, :]
            kF_cor_ts_array[:, i] = kf_data_temp['ts'][2, :]
            # Training Data
            kF_rmse_tr_array[:, i] = kf_data_temp['tr'][0, :]
            kF_r2_tr_array[:, i] = kf_data_temp['tr'][1, :]
            kF_cor_tr_array[:, i] = kf_data_temp['tr'][2, :]

        kF_data['ts']['rmse'] = pd.DataFrame(data=kF_rmse_ts_array, columns=self.model_names)
        kF_data['ts']['r2'] = pd.DataFrame(data=kF_r2_ts_array, columns=self.model_names)
        kF_data['ts']['cor'] = pd.DataFrame(data=kF_cor_ts_array, columns=self.model_names)
        # Training
        kF_data['tr']['rmse'] = pd.DataFrame(data=kF_rmse_tr_array, columns=self.model_names)
        kF_data['tr']['r2'] = pd.DataFrame(data=kF_r2_tr_array, columns=self.model_names)
        kF_data['tr']['cor'] = pd.DataFrame(data=kF_cor_tr_array, columns=self.model_names)

        return rmse, r2, cor, Yp, kF_data

    def RegressionCVMult(self):
        """ METHODS, INPUTS, & OUTPUTS"""
        ################################################################################################################
        # METHODS:

        # Storage Arrays -----------------------------------------------------------------------------------------------
        rmse_np = np.zeros((self.numMdls, self.N))
        r2_np = np.zeros((self.numMdls, self.N))
        cor_np = np.zeros((self.numMdls, self.N))
        Yp_int = dict()

        kFold_data_int = dict()
        kFold_data_int['ts'] = dict()
        kFold_data_int['tr'] = dict()

        # Runs the models 'N' times ------------------------------------------------------------------------------------
        for id in range(self.N):
            var_name = "variation_#"+str(id + 1)  # variation name

            rmse_temp, r2_temp, cor_temp, Yp_temp, kF_data_temp = self.RegressionCV(id)

            # Puts data into proper storage arrays
            rmse_np[:, id] = rmse_temp[:, 0]
            r2_np[:, id] = r2_temp[:, 0]
            cor_np[:, id] = cor_temp[:, 0]
            Yp_int[var_name] = Yp_temp

            kFold_data_int['ts'][var_name] = kF_data_temp['ts']
            kFold_data_int['tr'][var_name] = kF_data_temp['tr']

        # RESTRUCTURES DATA --------------------------------------------------------------------------------------------
        col_names = []
        row_names = []
        for i in range(self.N):
            col_names.append(str("variation_#"+str(i + 1)))
        for i in range(self.numMdls):
            row_names.append(self.model_names[i])

        rmse_df = pd.DataFrame(data=rmse_np, columns=col_names, index=row_names)
        r2_df = pd.DataFrame(data=r2_np, columns=col_names, index=row_names)
        cor_df = pd.DataFrame(data=cor_np, columns=col_names, index=row_names)

        Yp = dict()
        for mdl in row_names:
            Yp[mdl] = dict()
        # Switches from Yp_int[var][mdl] to Yp[mdl[var]
        for var in col_names:
            for mdl in row_names:
                Yp[mdl][var] = Yp_int[var][mdl]

        # RESULTS ------------------------------------------------------------------------------------------------------
        results = dict()
        results['rmse'] = rmse_df
        results['r2'] = r2_df
        results['cor'] = cor_df
        results['Yp'] = Yp

        # BEST PREDICTION ----------------------------------------------------------------------------------------------
        bestPred = dict()
        minId_temp = np.where(rmse_np == np.min(rmse_np))
        minId = [minId_temp[0][0], minId_temp[1][0]]
        model_id = minId[0]
        variation_id = minId[1]
        rmse_min = rmse_np[model_id, variation_id]
        r2_max = r2_np[model_id, variation_id]
        cor_max = cor_np[model_id, variation_id]
        Yp_best = Yp[self.model_names[model_id]]["variation_#"+str(variation_id+1)]

        bestPred['rmse'] = rmse_min
        bestPred['r2'] = r2_max
        bestPred['cor'] = cor_max
        bestPred['model_type'] = self.model_names[model_id]
        bestPred['Yp'] = Yp_best
        bestPred['minID'] = minId

        # K-FOLD DATA --------------------------------------------------------------------------------------------------

        kFold_data = dict()  # initalizes the dict that will store all the kFold data
        kFold_data['ts'] = dict()  # adds a section for testing data
        kFold_data['tr'] = dict()  # adds a section for training data

        # *** RESULTS *** #
        kFold_data['ts']['results'] = kFold_data_int['ts']
        kFold_data['tr']['results'] = kFold_data_int['tr']

        # *** BEST PRED *** #
        # Testing
        kF_TOPVARIATION_ts = kFold_data_int['ts']["variation_#" + str(variation_id + 1)]  # Top variation
        kF_TOPMODEL_rmse_ts = kF_TOPVARIATION_ts['rmse'].iloc[:, model_id]  # top RMSE-list
        kF_TOPMODEL_r2_ts = kF_TOPVARIATION_ts['r2'].iloc[:, model_id]  # top R2-list
        kF_TOPMODEL_cor_ts = kF_TOPVARIATION_ts['cor'].iloc[:, model_id]  # top Cor-list
        kFold_data['ts']['bestPred'] = {'rmse': kF_TOPMODEL_rmse_ts, 'r2': kF_TOPMODEL_r2_ts,
                                        'cor': kF_TOPMODEL_cor_ts}
        # training
        kF_TOPVARIATION_tr = kFold_data_int['tr']["variation_#" + str(variation_id + 1)]  # Top variation
        kF_TOPMODEL_rmse_tr = kF_TOPVARIATION_tr['rmse'].iloc[:, model_id]  # top RMSE-list
        kF_TOPMODEL_r2_tr = kF_TOPVARIATION_tr['r2'].iloc[:, model_id]  # top R2-list
        kF_TOPMODEL_cor_tr = kF_TOPVARIATION_tr['cor'].iloc[:, model_id]  # top Cor-list
        kFold_data['tr']['bestPred'] = {'rmse': kF_TOPMODEL_rmse_tr, 'r2': kF_TOPMODEL_r2_tr,
                                        'cor': kF_TOPMODEL_cor_tr}


        if self.giveKFdata:
            return results, bestPred, kFold_data
        else:
            return results, bestPred

    def Regression_test_singleModel(self, model, X_train, X_test, Y_train, Y_test):
        
        mdl = model.fit(X_train, Y_train)
        Yp = mdl.predict(X_test)
        
        rmse, r2, cor = self.mF.getPredMetrics(Y_test, Yp)
        
        return Yp, rmse, r2, cor
    
    def Regression_test_multProp(self, X_train, X_test, Y_train, Y_test):
        
        results = dict()
        
        Yp_colNames = []
        Yp_zeros = np.zeros((len(X_test), self.numMdls))
        for mdl_nm in self.model_names:
            Yp_colNames.append(mdl_nm)
        metrics_colNames = ['RMSE', 'R2', 'Cor']
        
        Yp_df = pd.DataFrame(data=Yp_zeros, columns=Yp_colNames)
        metrics_df = pd.DataFrame(columns=metrics_colNames)
        
        for model in range(self.numMdls):
            model_current = self.model_list[model]
            model_name_current = self.model_names[model]
        
            Yp_temp, rmse_temp, r2_temp, cor_temp = Regression_test_singleModel(model_current, X_train, X_test, Y_train, Y_test)
            
            Yp_df.iloc[:, model] = Yp_temp
            metrics_df.loc[model_name_current] = [rmse_temp, r2_temp, cor_temp]
            
        results['Yp'] = Yp_df
        results['metrics'] = metrics_df

        return results