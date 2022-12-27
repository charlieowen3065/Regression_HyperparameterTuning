import random
import sys
import os
import pickle

import pandas as pd

import numpy as np
from scipy.stats import iqr
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RationalQuadratic, RBF


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
from miscFunctions import *
from Regression import *

sys.path.append(os.path.abspath(current_path))


# ******************************************************************************************************************** #

class heatmaps2():
    def __init__(self, X_inp, Y_inp, Nk=5, N=1,
                 num_HP_zones_AL=3, num_runs_AL=3,
                 numLayers_GS=3, numZooms_GS=3,
                 gridLength_AL=10, gridLength_GS=10,
                 decimal_points_int=0.25, decimal_points_top=0.25,
                 decimal_point_GS=0.10,
                 RemoveNaN=True, goodIDs=None, seed='random', models_use='all',
                 save_csv_files=True,
                 C_input='None', epsilon_input='None', gamma_input='None', coef0_input='None',
                 noise_input='None', sigmaF_input='None', length_input='None', alpha_input='None'):


        self.mf = miscFunctions()
        reg = Regression(X_inp, Y_inp, models_use=models_use)
        model_names, model_list = reg.getModels()

        self.X_inp = X_inp
        self.Y_inp = Y_inp
        self.Nk = Nk
        self.N = N

        # Active Learning
        self.num_HP_zones_AL = num_HP_zones_AL
        self.num_runs_AL = num_runs_AL
        self.gridLength_AL = gridLength_AL
        self.decimal_points_int = decimal_points_int
        self.decimal_points_top = decimal_points_top

        # Grid Search
        self.numLayers_GS = numLayers_GS
        self.numZooms_GS = numZooms_GS
        self.gridLength_GS = gridLength_GS
        self.decimal_point_GS = decimal_point_GS


        self.RemoveNaN_var = RemoveNaN
        self.goodIDs = goodIDs
        self.seed = seed

        self.models_use = models_use
        self.mdl_name = model_names[0]

        self.save_csv_files = save_csv_files

        self.C_input = C_input
        self.epsilon_input = epsilon_input
        self.gamma_input = gamma_input
        self.coef0_input = coef0_input

        self.noise_input = noise_input
        self.sigmaF_input = sigmaF_input
        self.length_input = length_input
        self.alpha_input = alpha_input

        self.model_type = self.determine_model_type()


    def inputs_dist_function(self, numHPs, dec, model_type, ItoV_eq_const, model_data_old):
        Npts_tot = self.gridLength ** numHPs
        num_points = Npts_tot * dec
        num_idx_pts = int(num_points ** (1/numHPs))
        maxDist = float(self.gridLength) / float(num_idx_pts)
        indices = np.linspace(0, self.gridLength - 1, num_idx_pts, dtype=int)
        if model_type == "SVM_Type1":
            points = [(i, j) for i in indices for j in indices]
        elif (model_type == "SVM_Type2") | (model_type == "GPR_Type1"):
            points = [(i, j, k) for i in indices for j in indices for k in indices]
        elif (model_type == "SVM_Type3") | (model_type == "GPR_Type2"):
            points = [(i, j, k, l) for i in indices for j in indices for k in indices for l in indices]

        N = len(points[0])
        length_mdl_data = len(model_data_old[:,0])

        new_input_idx = []
        for pt in points:
            counter = 0
            for mdl_d in model_data_old:
                sqr_sum = 0
                for i in range(N):
                    a = ItoV_eq_const[i][0]
                    b = ItoV_eq_const[i][1]
                    dp = (a * mdl_d[i]) - b
                    sqr_sum += (pt[i] - dp) ** 2
                dist = np.sqrt(sqr_sum)
                if dist < maxDist:
                    counter += 1
                    break
            if counter == 0:
                new_input_idx.append(pt)

        return new_input_idx

    def convert_to_base_b(self, n, base):
        # Converts n from base 10 to base 'b', and the output is the new value in a tring format
        # ** only valid for b <= 10 **

        truth_var = False
        current_n = n
        new_n = []
        while truth_var == False:
            int_n = int(current_n / base)
            remainder_n = current_n % base
            new_n.append(remainder_n)
            current_n = int_n
            if current_n == 0:
                truth_var = True
        new_n.reverse()
        new_n_string = ''
        for i in new_n:
            new_n_string = new_n_string + str(i)

        return new_n_string

    def figureNumbers(self, numLayers, numZooms):
        figure_names_list = []

        name_start = "Figure "
        o = '.'
        base = numZooms + 1

        # First layer = 1.0.0.(n).0
        # Starts with a hard-code of the first value
        first_value = 'Figure 1'
        for i in range(numLayers - 1):
            first_value += o + '0'
        figure_names_list.append(first_value)

        # The rest of the layers
        for layer in range(numLayers - 1):  # Iterates over the number of layers - 1
            layer += 1  # starts on layer 2

            base_10_values = []

            starting_value = 0  # this value will be 11..(n)..11
            for i in range(layer + 1):
                starting_value += (base ** i)  # 11..(n)..11 = [b^0 + b^1 + b^2 + ... + b^n]
                # This value will be in base10 -> all values will be kept in base10 until
                # they are added to the list, along with the trailing zeros.
            base_10_values.append(starting_value)
            j = 1
            # for i in range(1, numZooms**layer):
            while True:
                base10_value_temp = starting_value + j
                j += 1
                base_b_value_temp = self.convert_to_base_b(base10_value_temp, base)
                if int(base_b_value_temp[0]) > 1:
                    break
                if '0' not in base_b_value_temp:
                    base_10_values.append(base10_value_temp)

            for base10_value in base_10_values:
                base_b_value = self.convert_to_base_b(base10_value, base)
                new_fig_name = 'Figure '
                for str_i in base_b_value:
                    new_fig_name += str_i + o
                if numLayers - layer != 0:
                    for i in range(numLayers - layer - 1):
                        new_fig_name += '0' + o
                figure_names_list.append(new_fig_name)

        return figure_names_list

    """ New Active Learning Model """
    """
    if model_type == 'SVM_Type1':
        # Initialize Ranges
        num_HPs = 2
        C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength)
        epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength)
        numC = len(C_range)
        numE = len(epsilon_range)
        # Running All Combinations
        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                epsilon = epsilon_range[e_idx]

    elif model_type == 'SVM_Type2':
        # Initialize Ranges
        num_HPs = 3
        C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength)
        epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength)
        gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], self.gridLength)
        numC = len(C_range)
        numE = len(epsilon_range)
        numG = len(gamma_range)
        # Running All Combinations 
        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                epsilon = epsilon_range[e_idx]
                for g_idx in range(numG):
                    gamma = gamma_range[g_idx]

    elif model_type == 'SVM_Type3':
        # Initialize Ranges
        num_HPs = 4
        C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength)
        epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength)
        gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], self.gridLength)
        coef0_range = np.linspace(self.coef0_input[0], self.coef0_input[1], self.gridLength)
        numC = len(C_range)
        numE = len(epsilon_range)
        numG = len(gamma_range)
        numC0 = len(coef0_range)
        # Running All Combinations
        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                epsilon = epsilon_range[e_idx]
                for g_idx in range(numG):
                    gamma = gamma_range[g_idx]
                    for c0_idx in range(numC0):
                        c0 = coef0_range[c0_idx]

    elif model_type == 'GPR_Type1':
        # Initialize Ranges
        num_HPs = 3
        noise_range = np.linspace(self.noise_input[0], self.noise_input[1], self.gridLength)
        sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], self.gridLength)
        length_range = np.linspace(self.length_input[0], self.length_input[1], self.gridLength)
        numN = len(noise_range)
        numS = len(sigmaF_range)
        numL = len(length_range)
        # Running All Combinations
        for n_idx in range(numN):
            noise = noise_range[n_idx]
            for s_idx in range(numS):
                epsilon = sigmaF_range[s_idx]
                for l_idx in range(numL):
                    scale_length = length_range[l_idx]

    elif model_type == 'GPR_Type2':
        # Initialize Ranges
        num_HPs = 4
        noise_range = np.linspace(self.noise_input[0], self.noise_input[1], self.gridLength)
        sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], self.gridLength)
        length_range = np.linspace(self.length_input[0], self.length_input[1], self.gridLength)
        alpha_range = np.linspace(self.alpha_input[0], self.alpha_input[1], self.gridLength)
        numN = len(noise_range)
        numS = len(sigmaF_range)
        numL = len(length_range)
        numA = len(alpha_range)
        # Running All Combinations
        for n_idx in range(numN):
            noise = noise_range[n_idx]
            for s_idx in range(numS):
                epsilon = sigmaF_range[s_idx]
                for l_idx in range(numL):
                    scale_length = length_range[l_idx]
                    for a_idx in range(numA):
                        alpha = alpha_range[a_idx]

    else:
        print("Error in Model-Type: ", model_type)
            """
    """ PART A: Functions for the Active Learning Section of the model """

    def determine_model_type(self):
        """ Method """
        """
        This function is called at the initialization of the class (__init__) and determines the 'model type' for the 
        rest of the code, based off the following list:
                Models Used                             Hyperparameters
         - SVM_Type1: SVM_Linear ------------------- C, Epsilon
         - SVM_Type2: SVM_RBF ---------------------- C, Epsilon, Gamma
         - SVM_Type3: SVM_Poly2, Poly3 ------------- C, Epsilon, Gamma, Coef0
         - GPR_Type1: RBF, Matern 3/2, Matern 5/2 -- Noise, SigmaF, Scale Length
         - GPR_Type2: Rational Quadratic ----------- Noise, SigmaF, Scale Length, Alpha
        And determines this by looking at which hyperparameter-inputs were set and which were left as 'None'.
        """
        # Initializes Boolean Variables --------------------------------------------------------------------------------
        C_var = False  # HP: C
        E_var = False  # HP: Epsilon
        G_var = False  # HP: Gamma
        C0_var = False  # HP: Coef0
        N_var = False  # HP: Noise
        S_var = False  # HP: Sigma_F
        L_var = False  # HP: Scale-Length
        A_var = False  # HP: Alpha

        # Checks each range-value --------------------------------------------------------------------------------------
        # SVM
        if self.C_input != 'None':
            C_var = True
        if self.epsilon_input != 'None':
            E_var = True
        if self.gamma_input != 'None':
            G_var = True
        if self.coef0_input != 'None':
            C0_var = True
        # GPR
        if self.noise_input != 'None':
            N_var = True
        if self.sigmaF_input != 'None':
            S_var = True
        if self.length_input != 'None':
            L_var = True
        if self.alpha_input != 'None':
            A_var = True

        # Determines model type ----------------------------------------------------------------------------------------
        model_type = 'None'
        # SVM
        if (C_var) & (E_var) & (not G_var) & (not C0_var):
            model_type = "SVM_Type1"  # SVM_Linear
        elif (C_var) & (E_var) & (G_var) & (not C0_var):
            model_type = "SVM_Type2"  # SVM_RBF
        elif (C_var) & (E_var) & (G_var) & (C0_var):
            model_type = "SVM_Type3"  # SVM_Poly2, SVM_Poly3
        # GPR
        elif (N_var) & (S_var) & (L_var) & (not A_var):
            model_type = "GPR_Type1"  # GPR RBF, Matern 3/2, Matern 5/2
        elif (N_var) & (S_var) & (L_var) & (A_var):
            model_type = "GPR_Type2"  # GPR_RatQuad
        else:
            print("LOGICAL ERROR ~ Charlie")

        return model_type

    def combine_model_data(self, model_data_1, model_data_2):
        """ Method """
        """
        This function takes in data from two models in the form:
         - model_data_(1 or 2) = [model_inputs, model_outputs]
        and then combines the inputs and outputs to create one long array from each.
        """
        inputs_1 = model_data_1[0]
        outputs_1 = model_data_1[1]

        inputs_2 = model_data_2[0]
        outputs_2 = model_data_2[1]

        Npts_1 = len(inputs_1[:,0])
        NHPs_1 = len(inputs_1[0,:])

        Npts_2 = len(inputs_2[:, 0])
        NHPs_2 = len(inputs_2[0, :])

        if NHPs_1 != NHPs_2:
            print("ERROR: Number of Hyperparameters is not equal")
            return

        Npts_f = Npts_1 + Npts_2
        inputs_f = np.zeros((Npts_f, NHPs_1))
        outputs_f = np.zeros((Npts_f, 1))

        counter = 0
        for i in range(Npts_1):
            inputs_f[counter, :] = inputs_1[i, :]
            outputs_f[counter, 0] = outputs_1[i, 0]
            counter += 1
        for i in range(Npts_2):
            inputs_f[counter, :] = inputs_2[i, :]
            outputs_f[counter, 0] = outputs_2[i, 0]
            counter += 1

        return [inputs_f, outputs_f]

    def PartI_intital_calculations(self, ranges, model_data_int):
        """ Method """
        """
        This function is part 1 of 3 for the 'High-Density' function.
        In this method, a small percentage of the of the full range of values in the specified region (with
        this area being specified by the 'ranges' input) are calulated and added to the full model (this new
        data is combined with the 'model_data_int' input) to make better predictions on the space in Part III.
        
        """

        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name
        decimal_points_int = self.decimal_points_int
        gridLength = self.gridLength_AL
        model_type = self.model_type

        print("Part I -- Started")

        if model_type == 'SVM_Type1':
            # Initialize Ranges
            num_HPs = 2
            C_range = ranges[0]
            epsilon_range = ranges[1]
            numC = len(C_range)
            numE = len(epsilon_range)
            # Getting index points
            Npts = numC * numE
            indices = np.linspace(0, gridLength - 1, int((Npts * decimal_points_int) ** (1 / num_HPs)),
                                  dtype=int)
            points = [(i, j) for i in indices for j in indices]
            Npts_int_calc = len(points)

            model_inputs_new = np.zeros((Npts_int_calc, 2))
            model_outputs_new = np.zeros((Npts_int_calc, 1))

            # Running All Combinations
            count = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]

                    index_current = tuple((C_idx, e_idx))
                    if index_current in points:
                        reg = Regression(X_use, Y_use,
                                         C=C, epsilon=epsilon,
                                         Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                         RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                         giveKFdata=True)
                        results, bestPred, kFold_data = reg.RegressionCVMult()

                        error = float(results['rmse'].loc[str(mdl_name)])

                        model_inputs_new[count, 0] = C
                        model_inputs_new[count, 1] = epsilon
                        model_outputs_new[count, 0] = error

                        count += 1
                        print("Count: "+str(count)+" / "+str(Npts_int_calc))

            model_data = self.combine_model_data(model_data_int, [model_inputs_new, model_outputs_new])
            return model_data

        elif model_type == 'SVM_Type2':
            # Initialize Ranges
            num_HPs = 3
            C_range = ranges[0]
            epsilon_range = ranges[1]
            gamma_range = ranges[2]
            numC = len(C_range)
            numE = len(epsilon_range)
            numG = len(gamma_range)

            # Getting index points
            Npts = numC * numE * numG
            indices = np.linspace(0, gridLength - 1, int((Npts * decimal_points_int) ** (1 / num_HPs)),
                                  dtype=int)
            points = [(i, j, k) for i in indices for j in indices for k in indices]
            Npts_int_calc = len(points)

            model_inputs_new = np.zeros((Npts_int_calc, 3))
            model_outputs_new = np.zeros((Npts_int_calc, 1))

            # Running All Combinations
            count = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]
                    for g_idx in range(numG):
                        gamma = gamma_range[g_idx]

                        index_current = tuple((C_idx, e_idx, g_idx))
                        if index_current in points:
                            reg = Regression(X_use, Y_use,
                                             C=C, epsilon=epsilon, gamma=gamma,
                                             Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                             RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                             giveKFdata=True)
                            results, bestPred, kFold_data = reg.RegressionCVMult()

                            error = float(results['rmse'].loc[str(mdl_name)])

                            model_inputs_new[count, 0] = C
                            model_inputs_new[count, 1] = epsilon
                            model_inputs_new[count, 2] = gamma
                            model_outputs_new[count, 0] = error

                            count += 1
                            print("Count: " + str(count) + " / " + str(Npts_int_calc))

            model_data = self.combine_model_data(model_data_int, [model_inputs_new, model_outputs_new])
            return model_data

        elif model_type == 'SVM_Type3':
            # Initialize Ranges
            num_HPs = 4
            C_range = ranges[0]
            epsilon_range = ranges[1]
            gamma_range = ranges[2]
            coef0_range = ranges[3]
            numC = len(C_range)
            numE = len(epsilon_range)
            numG = len(gamma_range)
            numC0 = len(coef0_range)

            # Getting index points
            Npts = numC * numE * numG * numC0
            indices = np.linspace(0, gridLength - 1, int((Npts * decimal_points_int) ** (1 / num_HPs)),
                                  dtype=int)
            points = [(i, j, k, m) for i in indices for j in indices for k in indices for m in indices]
            Npts_int_calc = len(points)

            model_inputs_new = np.zeros((Npts_int_calc, 4))
            model_outputs_new = np.zeros((Npts_int_calc, 1))

            # Running All Combinations
            count = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]
                    for g_idx in range(numG):
                        gamma = gamma_range[g_idx]
                        for c0_idx in range(numC0):
                            c0 = coef0_range[c0_idx]

                            index_current = tuple((C_idx, e_idx, g_idx))
                            if index_current in points:
                                reg = Regression(X_use, Y_use,
                                                 C=C, epsilon=epsilon, gamma=gamma,
                                                 Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                                 RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                                 giveKFdata=True)
                                results, bestPred, kFold_data = reg.RegressionCVMult()

                                error = float(results['rmse'].loc[str(mdl_name)])

                                model_inputs_new[count, 0] = C
                                model_inputs_new[count, 1] = epsilon
                                model_inputs_new[count, 2] = gamma
                                model_inputs_new[count, 3] = c0
                                model_outputs_new[count, 0] = error

                                count += 1
                                print("Count: " + str(count) + " / " + str(Npts_int_calc))

            model_data = self.combine_model_data(model_data_int, [model_inputs_new, model_outputs_new])
            return model_data

        elif model_type == 'GPR_Type1':
            # Initialize Ranges
            num_HPs = 3
            noise_range = ranges[0]
            sigmaF_range = ranges[1]
            length_range = ranges[2]
            numN = len(noise_range)
            numS = len(sigmaF_range)
            numL = len(length_range)

            # Getting index points
            Npts = numN * numS * numL
            indices = np.linspace(0, gridLength - 1, int((Npts * decimal_points_int) ** (1 / num_HPs)),
                                  dtype=int)
            points = [(i, j, k) for i in indices for j in indices for k in indices]
            Npts_int_calc = len(points)

            model_inputs_new = np.zeros((Npts_int_calc, 3))
            model_outputs_new = np.zeros((Npts_int_calc, 1))

            # Running All Combinations
            count = 0
            for n_idx in range(numN):
                noise = noise_range[n_idx]
                for s_idx in range(numS):
                    sigmaF = sigmaF_range[s_idx]
                    for l_idx in range(numL):
                        scale_length = length_range[l_idx]

                        index_current = tuple((n_idx, s_idx, l_idx))
                        if index_current in points:
                            reg = Regression(X_use, Y_use,
                                             noise=noise, sigma_F=sigmaF, scale_length=scale_length,
                                             Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                             RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                             giveKFdata=True)
                            results, bestPred, kFold_data = reg.RegressionCVMult()

                            error = float(results['rmse'].loc[str(mdl_name)])

                            model_inputs_new[count, 0] = noise
                            model_inputs_new[count, 1] = sigmaF
                            model_inputs_new[count, 2] = scale_length
                            model_outputs_new[count, 0] = error

                            count += 1
                            print("Count: " + str(count) + " / " + str(Npts_int_calc))

            model_data = self.combine_model_data(model_data_int, [model_inputs_new, model_outputs_new])
            return model_data

        elif model_type == 'GPR_Type2':
            # Initialize Ranges
            num_HPs = 4
            noise_range = ranges[0]
            sigmaF_range = ranges[1]
            length_range = ranges[2]
            alpha_range = ranges[3]
            numN = len(noise_range)
            numS = len(sigmaF_range)
            numL = len(length_range)
            numA = len(alpha_range)

            # Getting index points
            Npts = numN * numS * numL * numA
            indices = np.linspace(0, gridLength - 1, int((Npts * decimal_points_int) ** (1 / num_HPs)),
                                  dtype=int)
            points = [(i, j, k) for i in indices for j in indices for k in indices]
            Npts_int_calc = len(points)

            model_inputs_new = np.zeros((Npts_int_calc, 4))
            model_outputs_new = np.zeros((Npts_int_calc, 1))

            # Running All Combinations
            count = 0
            for n_idx in range(numN):
                noise = noise_range[n_idx]
                for s_idx in range(numS):
                    sigmaF = sigmaF_range[s_idx]
                    for l_idx in range(numL):
                        scale_length = length_range[l_idx]
                        for a_idx in range(numA):
                            alpha = alpha_range[a_idx]

                            index_current = tuple((n_idx, s_idx, l_idx))
                            if index_current in points:
                                reg = Regression(X_use, Y_use,
                                                 noise=noise, sigma_F=sigmaF, scale_length=scale_length,
                                                 Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                                 RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                                 giveKFdata=True)
                                results, bestPred, kFold_data = reg.RegressionCVMult()

                                error = float(results['rmse'].loc[str(mdl_name)])

                                model_inputs_new[count, 0] = noise
                                model_inputs_new[count, 1] = sigmaF
                                model_inputs_new[count, 2] = scale_length
                                model_inputs_new[count, 3] = alpha
                                model_outputs_new[count, 0] = error

                                count += 1
                                print("Count: " + str(count) + " / " + str(Npts_int_calc))

            model_data = self.combine_model_data(model_data_int, [model_inputs_new, model_outputs_new])
            return model_data

        else:
            print("Error in Model-Type: ", model_type)

    def PartII_predictions(self, ranges, model_int, model_data, run_number):
        """ Method """
        """
        This function is part 2 of 3 for the 'High-Density' function.
        This method takes in the current, wanted range in the hyperparmeter space, as well as the model up to this
        point, to make predictions on the full given space.
        """

        model_type = self.model_type

        model = model_int.fit(model_data[0], model_data[1])

        print("Part II -- Started")

        if model_type == 'SVM_Type1':
            # Initialize Ranges
            num_HPs = 2
            C_range = ranges[0]
            epsilon_range = ranges[1]
            numC = len(C_range)
            numE = len(epsilon_range)

            # Setting up data arrays
            zeros = np.zeros((numC, numE))

            pred_error_array = zeros.copy()
            pred_std_array = zeros.copy()
            pred_min_error_array = zeros.copy()

            X_ts = np.zeros((1, 2))
            pred_error_df = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon'])

            # Running All Combinations
            counter = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]

                    X_ts[0, :] = [C, epsilon]
                    error_pred, error_std = model.predict(X_ts, return_std=True)

                    if run_number != 'None':
                        min_error = (error_pred[0] - ((1 / run_number) * error_std[0]))
                    else:
                        min_error = error_pred[0]

                    pred_error_df.loc[counter] = [min_error, C, epsilon]
                    counter += 1

                    idx_tuple = (C_idx, e_idx)

                    pred_error_array[idx_tuple] = error_pred
                    pred_std_array[idx_tuple] = error_std
                    pred_min_error_array[idx_tuple] = min_error

            pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)
            return pred_min_error_array, pred_error_df_sorted

        elif model_type == 'SVM_Type2':
            # Initialize Ranges
            num_HPs = 3
            C_range = ranges[0]
            epsilon_range = ranges[1]
            gamma_range = ranges[2]
            numC = len(C_range)
            numE = len(epsilon_range)
            numG = len(gamma_range)

            # Setting up data arrays
            zeros = np.zeros((numC, numE, numG))

            pred_error_array = zeros.copy()
            pred_std_array = zeros.copy()
            pred_min_error_array = zeros.copy()

            X_ts = np.zeros((1, 3))
            pred_error_df = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma'])

            # Running All Combinations
            counter = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]
                    for g_idx in range(numG):
                        gamma = gamma_range[g_idx]

                        X_ts[0, :] = [C, epsilon, gamma]
                        error_pred, error_std = model.predict(X_ts, return_std=True)

                        if run_number != 'None':
                            min_error = (error_pred[0] - ((1 / run_number) * error_std[0]))
                        else:
                            min_error = error_pred[0]

                        pred_error_df.loc[counter] = [min_error, C, epsilon, gamma]
                        counter += 1

                        idx_tuple = (C_idx, e_idx, g_idx)

                        pred_error_array[idx_tuple] = error_pred
                        pred_std_array[idx_tuple] = error_std
                        pred_min_error_array[idx_tuple] = min_error

            pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)
            return pred_min_error_array, pred_error_df_sorted

        elif model_type == 'SVM_Type3':
            # Initialize Ranges
            num_HPs = 4
            C_range = ranges[0]
            epsilon_range = ranges[1]
            gamma_range = ranges[2]
            coef0_range = ranges[3]
            numC = len(C_range)
            numE = len(epsilon_range)
            numG = len(gamma_range)
            numC0 = len(coef0_range)

            # Setting up data arrays
            zeros = np.zeros((numC, numE, numG, numC0))

            pred_error_array = zeros.copy()
            pred_std_array = zeros.copy()
            pred_min_error_array = zeros.copy()

            X_ts = np.zeros((1, 4))
            pred_error_df = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma', 'Coef0'])

            # Running All Combinations
            counter = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]
                    for g_idx in range(numG):
                        gamma = gamma_range[g_idx]
                        for c0_idx in range(numC0):
                            c0 = coef0_range[c0_idx]

                            X_ts[0, :] = [C, epsilon, gamma, c0]
                            error_pred, error_std = model.predict(X_ts, return_std=True)

                            if run_number != 'None':
                                min_error = (error_pred[0] - ((1 / run_number) * error_std[0]))
                            else:
                                min_error = error_pred[0]

                            pred_error_df.loc[counter] = [min_error, C, epsilon, gamma, c0]
                            counter += 1

                            idx_tuple = (C_idx, e_idx, g_idx, c0_idx)

                            pred_error_array[idx_tuple] = error_pred
                            pred_std_array[idx_tuple] = error_std
                            pred_min_error_array[idx_tuple] = min_error

            pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)
            return pred_min_error_array, pred_error_df_sorted

        elif model_type == 'GPR_Type1':
            # Initialize Ranges
            num_HPs = 3
            noise_range = ranges[0]
            sigmaF_range = ranges[1]
            length_range = ranges[2]
            numN = len(noise_range)
            numS = len(sigmaF_range)
            numL = len(length_range)

            # Setting up data arrays
            zeros = np.zeros((numN, numS, numL))

            pred_error_array = zeros.copy()
            pred_std_array = zeros.copy()
            pred_min_error_array = zeros.copy()

            X_ts = np.zeros((1, 3))
            pred_error_df = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length'])

            # Running All Combinations
            counter = 0
            for n_idx in range(numN):
                noise = noise_range[n_idx]
                for s_idx in range(numS):
                    sigmaF = sigmaF_range[s_idx]
                    for l_idx in range(numL):
                        scale_length = length_range[l_idx]

                        X_ts[0, :] = [noise, sigmaF, scale_length]
                        error_pred, error_std = model.predict(X_ts, return_std=True)

                        if run_number != 'None':
                            min_error = (error_pred[0] - ((1 / run_number) * error_std[0]))
                        else:
                            min_error = error_pred[0]

                        pred_error_df.loc[counter] = [min_error, noise, sigmaF, scale_length]
                        counter += 1

                        idx_tuple = (n_idx, s_idx, l_idx)

                        pred_error_array[idx_tuple] = error_pred
                        pred_std_array[idx_tuple] = error_std
                        pred_min_error_array[idx_tuple] = min_error

            pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)
            return pred_min_error_array, pred_error_df_sorted

        elif model_type == 'GPR_Type2':
            # Initialize Ranges
            num_HPs = 4
            noise_range = ranges[0]
            sigmaF_range = ranges[1]
            length_range = ranges[2]
            alpha_range = ranges[3]
            numN = len(noise_range)
            numS = len(sigmaF_range)
            numL = len(length_range)
            numA = len(alpha_range)

            # Setting up data arrays
            zeros = np.zeros((numN, numS, numL, numA))

            pred_error_array = zeros.copy()
            pred_std_array = zeros.copy()
            pred_min_error_array = zeros.copy()

            X_ts = np.zeros((1, 4))
            pred_error_df = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length', 'Alpha'])

            # Running All Combinations
            counter = 0
            for n_idx in range(numN):
                noise = noise_range[n_idx]
                for s_idx in range(numS):
                    sigmaF = sigmaF_range[s_idx]
                    for l_idx in range(numL):
                        scale_length = length_range[l_idx]
                        for a_idx in range(numA):
                            alpha = alpha_range[a_idx]

                            X_ts[0, :] = [noise, sigmaF, scale_length, alpha]
                            error_pred, error_std = model.predict(X_ts, return_std=True)

                            if run_number != 'None':
                                min_error = (error_pred[0] - ((1 / run_number) * error_std[0]))
                            else:
                                min_error = error_pred[0]

                            pred_error_df.loc[counter] = [min_error, noise, sigmaF, scale_length, alpha]
                            counter += 1

                            idx_tuple = (n_idx, s_idx, l_idx, a_idx)

                            pred_error_array[idx_tuple] = error_pred
                            pred_std_array[idx_tuple] = error_std
                            pred_min_error_array[idx_tuple] = min_error

            pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)
            return pred_min_error_array, pred_error_df_sorted

        else:
            print("Error in Model-Type: ", model_type)

    def PartIII_final_calculations(self, sorted_pred_dataframe, model_data_int, stor_num_counter):
        """ Method """
        """
        This function is part 1 of 3 for the 'High-Density' function.
        This function takes in the predictions from Part II, then selects the top points from this precition
        and actually calulates them. The data from these calculations is saved into the full model.
        
        """
        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name
        decimal_points_top = self.decimal_points_top
        gridLength = self.gridLength_AL
        model_type = self.model_type

        Npts = sorted_pred_dataframe.shape[0]
        num_new_calc_points = int(Npts * decimal_points_top)

        print("Part III -- Started")

        # DataFrame
        if (model_type == 'SVM_Type1') | (model_type == 'SVM_Type2') | (model_type == 'SVM_Type3'):
            df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Gamma', 'Coef0',
                            'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                            'avgTR to avgTS', 'avgTR to Final Error']
        elif (model_type == 'GPR_Type1') | (model_type == 'GPR_Type2'):
            df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'Noise', 'SigmaF', 'Length', 'Alpha',
                            'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                            'avgTR to avgTS', 'avgTR to Final Error']
        else:
            print("MODEL TYPE ERROR: ", model_type)
            return
        df_numRows = num_new_calc_points
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        # New Model Inputs
        if model_type == 'SVM_Type1':
            model_new_inputs = np.zeros((num_new_calc_points, 2))
            model_new_outputs = np.zeros((num_new_calc_points, 1))
        elif model_type == 'SVM_Type2':
            model_new_inputs = np.zeros((num_new_calc_points, 3))
            model_new_outputs = np.zeros((num_new_calc_points, 1))
        elif model_type == 'SVM_Type3':
            model_new_inputs = np.zeros((num_new_calc_points, 4))
            model_new_outputs = np.zeros((num_new_calc_points, 1))
        elif model_type == 'GPR_Type1':
            model_new_inputs = np.zeros((num_new_calc_points, 3))
            model_new_outputs = np.zeros((num_new_calc_points, 1))
        elif model_type == 'GPR_Type3':
            model_new_inputs = np.zeros((num_new_calc_points, 4))
            model_new_outputs = np.zeros((num_new_calc_points, 1))
        else:
            print("MODEL TYPE ERROR: ", model_type)
            return

        for pt in range(num_new_calc_points):
            HP1 = sorted_pred_dataframe.iloc[pt, 1]
            HP2 = sorted_pred_dataframe.iloc[pt, 2]
            if (model_type == 'SVM_Type2') | (model_type == 'GPR_Type1'):
                HP3 = sorted_pred_dataframe.iloc[pt, 3]
                HP4 = 1
            elif (model_type == 'SVM_Type3') | (model_type == 'GPR_Type2'):
                HP3 = sorted_pred_dataframe.iloc[pt, 3]
                HP4 = sorted_pred_dataframe.iloc[pt, 4]
            else:
                HP3 = 1
                HP4 = 1

            reg = Regression(X_use, Y_use,
                             C=HP1, epsilon=HP2, gamma=HP3, coef0=HP4,
                             noise=HP1, sigma_F=HP2, scale_length=HP3, alpha=HP4,
                             Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                             RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use, giveKFdata=True)
            results, bestPred, kFold_data = reg.RegressionCVMult()

            # Extracts data
            error = float(results['rmse'].loc[str(mdl_name)])
            avg_tr_error = np.mean(list(kFold_data['tr']['results']['variation_#1']['rmse'][str(mdl_name)]))
            avg_ts_error = np.mean(list(kFold_data['ts']['results']['variation_#1']['rmse'][str(mdl_name)]))
            ratio_trAvg_tsAvg = float(avg_ts_error / avg_tr_error)
            ratio_trAvg_final = float(error / avg_tr_error)

            rmse = error
            r2 = float(results['r2'].loc[str(mdl_name)])
            cor = float(results['cor'].loc[str(mdl_name)])

            avg_tr_rmse = float(
                np.mean(list(kFold_data['tr']['results']['variation_#1']['rmse'][str(mdl_name)])))
            avg_tr_r2 = float(np.mean(list(kFold_data['tr']['results']['variation_#1']['r2'][str(mdl_name)])))
            avg_tr_cor = float(np.mean(list(kFold_data['tr']['results']['variation_#1']['cor'][str(mdl_name)])))

            # Puts data into storage array
            storage_df.loc[pt] = ['run_'+str(stor_num_counter), rmse, r2, cor, HP1, HP2, HP3, HP4,
                                  avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                  ratio_trAvg_tsAvg, ratio_trAvg_final]

            print("Part III: "+str(pt+1)+" / "+str(num_new_calc_points)+": RMSE, R2 -> ("+str(rmse)+", "+str(r2)+")")

            # New model inputs
            if model_type == 'SVM_Type1':
                model_new_inputs[pt, :] = [HP1, HP2]
                model_new_outputs[pt, 0] = error
            elif model_type == 'SVM_Type2':
                model_new_inputs[pt, :] = [HP1, HP2, HP3]
                model_new_outputs[pt, 0] = error
            elif model_type == 'SVM_Type3':
                model_new_inputs[pt, :] = [HP1, HP2, HP3, HP4]
                model_new_outputs[pt, 0] = error
            elif model_type == 'GPR_Type1':
                model_new_inputs[pt, :] = [HP1, HP2, HP3]
                model_new_outputs[pt, 0] = error
            elif model_type == 'GPR_Type2':
                model_new_inputs[pt, :] = [HP1, HP2, HP3, HP4]
                model_new_outputs[pt, 0] = error

        model_data = self.combine_model_data(model_data_int, [model_new_inputs, model_new_outputs])
        storage_sorted_df = storage_df.sort_values(by=["RMSE"], ascending=True)
        return storage_sorted_df, model_data

    def initial_predictions(self):
        """ Method """
        """
        This function is the initial function. It serves multiple purposes.
        First, the function runs an initial small percent of the full hyperparameter space and calulates the errors.
        Then, using this data, it trains 4 GPR models (Rational Quadratic, RBF, Matern 3/2, and Matern 5/2) and picks
        the one with the lowest total (summed) standard deviation from the 4, over predictions across the whole space.
        It saves the best model and its data.
        
        """

        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name
        decimal_points_int = self.decimal_points_int
        decimal_points_top = self.decimal_points_top
        gridLength = self.gridLength_AL
        model_type = self.model_type
        print("Model: ", model_type)

        print("initial_predictions --  Started")

        # RUN INITIAL PREDICTIONS --------------------------------------------------------------------------------------
        if model_type == 'SVM_Type1':
            # Initialize Ranges
            num_HPs = 2
            C_range = np.linspace(self.C_input[0], self.C_input[1], gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], gridLength)
            numC = len(C_range)
            numE = len(epsilon_range)
            # Getting index points
            Npts = numC * numE
            indices = np.linspace(0, gridLength-1, int((Npts * decimal_points_int) ** (1 / num_HPs)), dtype=int)
            points = [(i, j) for i in indices for j in indices]
            Npts_int_calc = len(points)

            model_inputs = np.zeros((Npts_int_calc, 2))
            model_outputs = np.zeros((Npts_int_calc, 1))

            # Running All
            """ PART I: Initial Calculations """
            count = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]

                    index_current = tuple((C_idx, e_idx))
                    if index_current in points:
                        reg = Regression(X_use, Y_use,
                                         C=C, epsilon=epsilon,
                                         Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                         RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                         giveKFdata=True)
                        results, bestPred, kFold_data = reg.RegressionCVMult()

                        error = float(results['rmse'].loc[str(mdl_name)])

                        model_inputs[count, 0] = C
                        model_inputs[count, 1] = epsilon
                        model_outputs[count, 0] = error

                        count += 1
                        print("initial Count: "+str(count)+" / "+str(Npts_int_calc))


            kernel_1 = RationalQuadratic()
            kernel_2 = RBF()
            kernel_3 = Matern(nu=3 / 2)
            kernel_4 = Matern(nu=5 / 2)

            model_1 = gaussian_process.GaussianProcessRegressor(kernel=kernel_1, copy_X_train=False)
            model_2 = gaussian_process.GaussianProcessRegressor(kernel=kernel_2, copy_X_train=False)
            model_3 = gaussian_process.GaussianProcessRegressor(kernel=kernel_3, copy_X_train=False)
            model_4 = gaussian_process.GaussianProcessRegressor(kernel=kernel_4, copy_X_train=False)

            model_1.fit(model_inputs, model_outputs)
            model_2.fit(model_inputs, model_outputs)
            model_3.fit(model_inputs, model_outputs)
            model_4.fit(model_inputs, model_outputs)

            """ PART II: Predictions """
            zeros = np.zeros((numC, numE))

            pred_error_array_1 = zeros.copy()
            pred_std_array_1 = zeros.copy()
            pred_min_error_array_1 = zeros.copy()

            pred_error_array_2 = zeros.copy()
            pred_std_array_2 = zeros.copy()
            pred_min_error_array_2 = zeros.copy()

            pred_error_array_3 = zeros.copy()
            pred_std_array_3 = zeros.copy()
            pred_min_error_array_3 = zeros.copy()

            pred_error_array_4 = zeros.copy()
            pred_std_array_4 = zeros.copy()
            pred_min_error_array_4 = zeros.copy()

            X_ts = np.zeros((1, 2))
            pred_error_df_1 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon'])
            pred_error_df_2 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon'])
            pred_error_df_3 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon'])
            pred_error_df_4 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon'])
            counter = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]

                    X_ts[0, :] = [C, epsilon]
                    error_pred_1, error_std_1 = model_1.predict(X_ts, return_std=True)
                    error_pred_2, error_std_2 = model_2.predict(X_ts, return_std=True)
                    error_pred_3, error_std_3 = model_3.predict(X_ts, return_std=True)
                    error_pred_4, error_std_4 = model_4.predict(X_ts, return_std=True)

                    min_error_1 = (error_pred_1[0] - error_std_1[0])
                    min_error_2 = (error_pred_2[0] - error_std_2[0])
                    min_error_3 = (error_pred_3[0] - error_std_3[0])
                    min_error_4 = (error_pred_4[0] - error_std_4[0])

                    pred_error_df_1.loc[counter] = [min_error_1, C, epsilon]
                    pred_error_df_2.loc[counter] = [min_error_2, C, epsilon]
                    pred_error_df_3.loc[counter] = [min_error_3, C, epsilon]
                    pred_error_df_4.loc[counter] = [min_error_4, C, epsilon]
                    counter += 1

                    idx_tuple = (C_idx, e_idx)

                    pred_error_array_1[idx_tuple] = error_pred_1
                    pred_std_array_1[idx_tuple] = error_std_1
                    pred_min_error_array_1[idx_tuple] = min_error_1

                    pred_error_array_2[idx_tuple] = error_pred_2
                    pred_std_array_2[idx_tuple] = error_std_2
                    pred_min_error_array_2[idx_tuple] = min_error_2

                    pred_error_array_3[idx_tuple] = error_pred_3
                    pred_std_array_3[idx_tuple] = error_std_3
                    pred_min_error_array_3[idx_tuple] = min_error_3

                    pred_error_array_4[idx_tuple] = error_pred_4
                    pred_std_array_4[idx_tuple] = error_std_4
                    pred_min_error_array_4[idx_tuple] = min_error_4

            tot_error_1 = np.concatenate(pred_std_array_1).sum()
            tot_error_2 = np.concatenate(pred_std_array_2).sum()
            tot_error_3 = np.concatenate(pred_std_array_3).sum()
            tot_error_4 = np.concatenate(pred_std_array_4).sum()
            list_of_total_errors = [tot_error_1, tot_error_2, tot_error_3, tot_error_4]
            print("total error list: ", list_of_total_errors)
            mdl_num = np.where(list_of_total_errors == np.min(list_of_total_errors))[0][0]
            if mdl_num == 0:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RationalQuadratic())
                pred_error_df = pred_error_df_1
                pred_min_error_array = pred_min_error_array_1
            elif mdl_num == 1:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RBF())
                pred_error_df = pred_error_df_2
                pred_min_error_array = pred_min_error_array_2
            elif mdl_num == 2:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3/2))
                pred_error_df = pred_error_df_3
                pred_min_error_array = pred_min_error_array_3
            elif mdl_num == 3:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3/2))
                pred_error_df = pred_error_df_4
                pred_min_error_array = pred_min_error_array_4
            else:
                print("Error in 'mdl_num': ", mdl_num)
                return
            pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)
            pred_error_df_sorted.to_csv("run_0_initial_predictions.csv")

        elif model_type == 'SVM_Type2':
            # Initialize Ranges
            num_HPs = 3
            C_range = np.linspace(self.C_input[0], self.C_input[1], gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], gridLength)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], gridLength)
            numC = len(C_range)
            numE = len(epsilon_range)
            numG = len(gamma_range)
            # Getting index points
            Npts = numC * numE * numG
            indices = np.linspace(0, gridLength - 1, int((Npts * decimal_points_int) ** (1 / num_HPs)),
                                  dtype=int)
            points = [(i, j, k) for i in indices for j in indices for k in indices]
            Npts_int_calc = len(points)

            model_inputs = np.zeros((Npts_int_calc, 3))
            model_outputs = np.zeros((Npts_int_calc, 1))

            # Running All Combinations
            """ PART I: Initial Calculations """
            count = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]
                    for g_idx in range(numG):
                        gamma = gamma_range[g_idx]

                        index_current = tuple((C_idx, e_idx, g_idx))
                        if index_current in points:
                            reg = Regression(X_use, Y_use,
                                             C=C, epsilon=epsilon, gamma=gamma,
                                             Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                             RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                             giveKFdata=True)
                            results, bestPred, kFold_data = reg.RegressionCVMult()

                            error = float(results['rmse'].loc[str(mdl_name)])

                            model_inputs[count, 0] = C
                            model_inputs[count, 1] = epsilon
                            model_inputs[count, 2] = gamma
                            model_outputs[count, 0] = error

                            count += 1
                            print("initial Count: " + str(count) + " / " + str(Npts_int_calc))

            kernel_1 = RationalQuadratic()
            kernel_2 = RBF()
            kernel_3 = Matern(nu=3 / 2)
            kernel_4 = Matern(nu=5 / 2)

            model_1 = gaussian_process.GaussianProcessRegressor(kernel=kernel_1, copy_X_train=False)
            model_2 = gaussian_process.GaussianProcessRegressor(kernel=kernel_2, copy_X_train=False)
            model_3 = gaussian_process.GaussianProcessRegressor(kernel=kernel_3, copy_X_train=False)
            model_4 = gaussian_process.GaussianProcessRegressor(kernel=kernel_4, copy_X_train=False)

            model_1.fit(model_inputs, model_outputs)
            model_2.fit(model_inputs, model_outputs)
            model_3.fit(model_inputs, model_outputs)
            model_4.fit(model_inputs, model_outputs)

            """ PART II: Predictions """
            zeros = np.zeros((numC, numE, numG))

            pred_error_array_1 = zeros.copy()
            pred_std_array_1 = zeros.copy()
            pred_min_error_array_1 = zeros.copy()

            pred_error_array_2 = zeros.copy()
            pred_std_array_2 = zeros.copy()
            pred_min_error_array_2 = zeros.copy()

            pred_error_array_3 = zeros.copy()
            pred_std_array_3 = zeros.copy()
            pred_min_error_array_3 = zeros.copy()

            pred_error_array_4 = zeros.copy()
            pred_std_array_4 = zeros.copy()
            pred_min_error_array_4 = zeros.copy()

            X_ts = np.zeros((1, 3))
            pred_error_df_1 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma'])
            pred_error_df_2 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma'])
            pred_error_df_3 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma'])
            pred_error_df_4 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma'])
            counter = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]
                    for g_idx in range(numG):
                        gamma = gamma_range[g_idx]

                        X_ts[0, :] = [C, epsilon, gamma]
                        error_pred_1, error_std_1 = model_1.predict(X_ts, return_std=True)
                        error_pred_2, error_std_2 = model_2.predict(X_ts, return_std=True)
                        error_pred_3, error_std_3 = model_3.predict(X_ts, return_std=True)
                        error_pred_4, error_std_4 = model_4.predict(X_ts, return_std=True)

                        min_error_1 = (error_pred_1[0] - error_std_1[0])
                        min_error_2 = (error_pred_2[0] - error_std_2[0])
                        min_error_3 = (error_pred_3[0] - error_std_3[0])
                        min_error_4 = (error_pred_4[0] - error_std_4[0])

                        pred_error_df_1.loc[counter] = [min_error_1, C, epsilon, gamma]
                        pred_error_df_2.loc[counter] = [min_error_2, C, epsilon, gamma]
                        pred_error_df_3.loc[counter] = [min_error_3, C, epsilon, gamma]
                        pred_error_df_4.loc[counter] = [min_error_4, C, epsilon, gamma]
                        counter += 1

                        idx_tuple = (C_idx, e_idx, g_idx)

                        pred_error_array_1[idx_tuple] = error_pred_1
                        pred_std_array_1[idx_tuple] = error_std_1
                        pred_min_error_array_1[idx_tuple] = min_error_1

                        pred_error_array_2[idx_tuple] = error_pred_2
                        pred_std_array_2[idx_tuple] = error_std_2
                        pred_min_error_array_2[idx_tuple] = min_error_2

                        pred_error_array_3[idx_tuple] = error_pred_3
                        pred_std_array_3[idx_tuple] = error_std_3
                        pred_min_error_array_3[idx_tuple] = min_error_3

                        pred_error_array_4[idx_tuple] = error_pred_4
                        pred_std_array_4[idx_tuple] = error_std_4
                        pred_min_error_array_4[idx_tuple] = min_error_4

            tot_error_1 = np.concatenate(pred_std_array_1).sum()
            tot_error_2 = np.concatenate(pred_std_array_2).sum()
            tot_error_3 = np.concatenate(pred_std_array_3).sum()
            tot_error_4 = np.concatenate(pred_std_array_4).sum()
            list_of_total_errors = [tot_error_1, tot_error_2, tot_error_3, tot_error_4]
            print("total error list: ", list_of_total_errors)
            mdl_num = np.where(list_of_total_errors == np.min(list_of_total_errors))[0][0]
            if mdl_num == 0:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RationalQuadratic())
                pred_error_df = pred_error_df_1
                pred_min_error_array = pred_min_error_array_1
            elif mdl_num == 1:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RBF())
                pred_error_df = pred_error_df_2
                pred_min_error_array = pred_min_error_array_2
            elif mdl_num == 2:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_3
                pred_min_error_array = pred_min_error_array_3
            elif mdl_num == 3:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_4
                pred_min_error_array = pred_min_error_array_4
            else:
                print("Error in 'mdl_num': ", mdl_num)
                return
            pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)
            pred_error_df_sorted.to_csv("run_0_initial_predictions.csv")

        elif model_type == 'SVM_Type3':
            # Initialize Ranges
            num_HPs = 4
            C_range = np.linspace(self.C_input[0], self.C_input[1], gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], gridLength)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], gridLength)
            coef0_range = np.linspace(self.coef0_input[0], self.coef0_input[1], gridLength)
            numC = len(C_range)
            numE = len(epsilon_range)
            numG = len(gamma_range)
            numC0 = len(coef0_range)

            # Getting index points
            Npts = numC * numE * numG * numC0
            indices = np.linspace(0, gridLength - 1, int((Npts * decimal_points_int) ** (1 / num_HPs)),
                                  dtype=int)
            points = [(i, j, k, l) for i in indices for j in indices for k in indices for l in indices]
            Npts_int_calc = len(points)

            model_inputs = np.zeros((Npts_int_calc, 4))
            model_outputs = np.zeros((Npts_int_calc, 1))

            # Running All Combinations
            """ PART I: Initial Calculations """
            count = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]
                    for g_idx in range(numG):
                        gamma = gamma_range[g_idx]
                        for c0_idx in range(numC0):
                            c0 = coef0_range[c0_idx]

                            index_current = tuple((C_idx, e_idx, g_idx, c0_idx))
                            if index_current in points:
                                reg = Regression(X_use, Y_use,
                                                 C=C, epsilon=epsilon, gamma=gamma, coef0=c0,
                                                 Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                                 RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                                 giveKFdata=True)
                                results, bestPred, kFold_data = reg.RegressionCVMult()

                                error = float(results['rmse'].loc[str(mdl_name)])

                                model_inputs[count, 0] = C
                                model_inputs[count, 1] = epsilon
                                model_inputs[count, 2] = gamma
                                model_inputs[count, 3] = c0
                                model_outputs[count, 0] = error

                                count += 1
                                print("initial Count: " + str(count) + " / " + str(Npts_int_calc))

            kernel_1 = RationalQuadratic()
            kernel_2 = RBF()
            kernel_3 = Matern(nu=3 / 2)
            kernel_4 = Matern(nu=5 / 2)

            model_1 = gaussian_process.GaussianProcessRegressor(kernel=kernel_1, copy_X_train=False)
            model_2 = gaussian_process.GaussianProcessRegressor(kernel=kernel_2, copy_X_train=False)
            model_3 = gaussian_process.GaussianProcessRegressor(kernel=kernel_3, copy_X_train=False)
            model_4 = gaussian_process.GaussianProcessRegressor(kernel=kernel_4, copy_X_train=False)

            model_1.fit(model_inputs, model_outputs)
            model_2.fit(model_inputs, model_outputs)
            model_3.fit(model_inputs, model_outputs)
            model_4.fit(model_inputs, model_outputs)

            """ PART II: Predictions """
            zeros = np.zeros((numC, numE, numG, numC0))

            pred_error_array_1 = zeros.copy()
            pred_std_array_1 = zeros.copy()
            pred_min_error_array_1 = zeros.copy()

            pred_error_array_2 = zeros.copy()
            pred_std_array_2 = zeros.copy()
            pred_min_error_array_2 = zeros.copy()

            pred_error_array_3 = zeros.copy()
            pred_std_array_3 = zeros.copy()
            pred_min_error_array_3 = zeros.copy()

            pred_error_array_4 = zeros.copy()
            pred_std_array_4 = zeros.copy()
            pred_min_error_array_4 = zeros.copy()

            X_ts = np.zeros((1, 3))
            pred_error_df_1 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma', 'Coef0'])
            pred_error_df_2 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma', 'Coef0'])
            pred_error_df_3 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma', 'Coef0'])
            pred_error_df_4 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma', 'Coef0'])
            counter = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]
                    for g_idx in range(numG):
                        gamma = gamma_range[g_idx]
                        for c0_idx in range(numC0):
                            c0 = coef0_range[c0_idx]

                            X_ts[0, :] = [C, epsilon, gamma, c0]
                            error_pred_1, error_std_1 = model_1.predict(X_ts, return_std=True)
                            error_pred_2, error_std_2 = model_2.predict(X_ts, return_std=True)
                            error_pred_3, error_std_3 = model_3.predict(X_ts, return_std=True)
                            error_pred_4, error_std_4 = model_4.predict(X_ts, return_std=True)

                            min_error_1 = (error_pred_1[0] - error_std_1[0])
                            min_error_2 = (error_pred_2[0] - error_std_2[0])
                            min_error_3 = (error_pred_3[0] - error_std_3[0])
                            min_error_4 = (error_pred_4[0] - error_std_4[0])

                            pred_error_df_1.loc[counter] = [min_error_1, C, epsilon, gamma, c0]
                            pred_error_df_2.loc[counter] = [min_error_2, C, epsilon, gamma, c0]
                            pred_error_df_3.loc[counter] = [min_error_3, C, epsilon, gamma, c0]
                            pred_error_df_4.loc[counter] = [min_error_4, C, epsilon, gamma, c0]
                            counter += 1

                            idx_tuple = (C_idx, e_idx, g_idx, c0_idx)

                            pred_error_array_1[idx_tuple] = error_pred_1
                            pred_std_array_1[idx_tuple] = error_std_1
                            pred_min_error_array_1[idx_tuple] = min_error_1

                            pred_error_array_2[idx_tuple] = error_pred_2
                            pred_std_array_2[idx_tuple] = error_std_2
                            pred_min_error_array_2[idx_tuple] = min_error_2

                            pred_error_array_3[idx_tuple] = error_pred_3
                            pred_std_array_3[idx_tuple] = error_std_3
                            pred_min_error_array_3[idx_tuple] = min_error_3

                            pred_error_array_4[idx_tuple] = error_pred_4
                            pred_std_array_4[idx_tuple] = error_std_4
                            pred_min_error_array_4[idx_tuple] = min_error_4

            tot_error_1 = np.concatenate(pred_std_array_1).sum()
            tot_error_2 = np.concatenate(pred_std_array_2).sum()
            tot_error_3 = np.concatenate(pred_std_array_3).sum()
            tot_error_4 = np.concatenate(pred_std_array_4).sum()
            list_of_total_errors = [tot_error_1, tot_error_2, tot_error_3, tot_error_4]
            print("total error list: ", list_of_total_errors)
            mdl_num = np.where(list_of_total_errors == np.min(list_of_total_errors))[0][0]
            if mdl_num == 0:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RationalQuadratic())
                pred_error_df = pred_error_df_1
                pred_min_error_array = pred_min_error_array_1
            elif mdl_num == 1:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RBF())
                pred_error_df = pred_error_df_2
                pred_min_error_array = pred_min_error_array_2
            elif mdl_num == 2:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_3
                pred_min_error_array = pred_min_error_array_3
            elif mdl_num == 3:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_4
                pred_min_error_array = pred_min_error_array_4
            else:
                print("Error in 'mdl_num': ", mdl_num)
                return
            pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)
            pred_error_df_sorted.to_csv("run_0_initial_predictions.csv")

        elif model_type == 'GPR_Type1':
            # Initialize Ranges
            num_HPs = 3
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], gridLength)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], gridLength)
            length_range = np.linspace(self.length_input[0], self.length_input[1], gridLength)
            numN = len(noise_range)
            numS = len(sigmaF_range)
            numL = len(length_range)

            # Getting index points
            Npts = numN * numS * numL
            indices = np.linspace(0, gridLength - 1, int((Npts * decimal_points_int) ** (1 / num_HPs)),
                                  dtype=int)
            points = [(i, j, k) for i in indices for j in indices for k in indices]
            Npts_int_calc = len(points)

            model_inputs = np.zeros((Npts_int_calc, 3))
            model_outputs = np.zeros((Npts_int_calc, 1))

            # Running All Combinations
            """ PART I: Initial Calculations """
            count = 0
            for n_idx in range(numN):
                noise = noise_range[n_idx]
                for s_idx in range(numS):
                    sigmaF = sigmaF_range[s_idx]
                    for l_idx in range(numL):
                        scale_length = length_range[l_idx]

                        index_current = tuple((n_idx, s_idx, l_idx))
                        if index_current in points:
                            reg = Regression(X_use, Y_use,
                                             noise=noise, sigma_F=sigmaF, scale_length=scale_length,
                                             Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                             RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                             giveKFdata=True)
                            results, bestPred, kFold_data = reg.RegressionCVMult()

                            error = float(results['rmse'].loc[str(mdl_name)])

                            model_inputs[count, 0] = noise
                            model_inputs[count, 1] = sigmaF
                            model_inputs[count, 2] = scale_length
                            model_outputs[count, 0] = error

                            count += 1
                            print("initial Count: " + str(count) + " / " + str(Npts_int_calc))

            kernel_1 = RationalQuadratic()
            kernel_2 = RBF()
            kernel_3 = Matern(nu=3 / 2)
            kernel_4 = Matern(nu=5 / 2)

            model_1 = gaussian_process.GaussianProcessRegressor(kernel=kernel_1, copy_X_train=False)
            model_2 = gaussian_process.GaussianProcessRegressor(kernel=kernel_2, copy_X_train=False)
            model_3 = gaussian_process.GaussianProcessRegressor(kernel=kernel_3, copy_X_train=False)
            model_4 = gaussian_process.GaussianProcessRegressor(kernel=kernel_4, copy_X_train=False)

            model_1.fit(model_inputs, model_outputs)
            model_2.fit(model_inputs, model_outputs)
            model_3.fit(model_inputs, model_outputs)
            model_4.fit(model_inputs, model_outputs)

            """ PART II: Predictions """
            zeros = np.zeros((numN, numS, numL))

            pred_error_array_1 = zeros.copy()
            pred_std_array_1 = zeros.copy()
            pred_min_error_array_1 = zeros.copy()

            pred_error_array_2 = zeros.copy()
            pred_std_array_2 = zeros.copy()
            pred_min_error_array_2 = zeros.copy()

            pred_error_array_3 = zeros.copy()
            pred_std_array_3 = zeros.copy()
            pred_min_error_array_3 = zeros.copy()

            pred_error_array_4 = zeros.copy()
            pred_std_array_4 = zeros.copy()
            pred_min_error_array_4 = zeros.copy()

            X_ts = np.zeros((1, 3))
            pred_error_df_1 = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length'])
            pred_error_df_2 = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length'])
            pred_error_df_3 = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length'])
            pred_error_df_4 = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length'])
            counter = 0
            for n_idx in range(numN):
                noise = noise_range[n_idx]
                for s_idx in range(numS):
                    sigmaF = sigmaF_range[s_idx]
                    for l_idx in range(numL):
                        scale_length = length_range[l_idx]

                        X_ts[0, :] = [noise, sigmaF, scale_length]
                        error_pred_1, error_std_1 = model_1.predict(X_ts, return_std=True)
                        error_pred_2, error_std_2 = model_2.predict(X_ts, return_std=True)
                        error_pred_3, error_std_3 = model_3.predict(X_ts, return_std=True)
                        error_pred_4, error_std_4 = model_4.predict(X_ts, return_std=True)

                        min_error_1 = (error_pred_1[0] - error_std_1[0])
                        min_error_2 = (error_pred_2[0] - error_std_2[0])
                        min_error_3 = (error_pred_3[0] - error_std_3[0])
                        min_error_4 = (error_pred_4[0] - error_std_4[0])

                        pred_error_df_1.loc[counter] = [min_error_1, noise, sigmaF, scale_length]
                        pred_error_df_2.loc[counter] = [min_error_2, noise, sigmaF, scale_length]
                        pred_error_df_3.loc[counter] = [min_error_3, noise, sigmaF, scale_length]
                        pred_error_df_4.loc[counter] = [min_error_4, noise, sigmaF, scale_length]
                        counter += 1

                        idx_tuple = (n_idx, s_idx, l_idx)

                        pred_error_array_1[idx_tuple] = error_pred_1
                        pred_std_array_1[idx_tuple] = error_std_1
                        pred_min_error_array_1[idx_tuple] = min_error_1

                        pred_error_array_2[idx_tuple] = error_pred_2
                        pred_std_array_2[idx_tuple] = error_std_2
                        pred_min_error_array_2[idx_tuple] = min_error_2

                        pred_error_array_3[idx_tuple] = error_pred_3
                        pred_std_array_3[idx_tuple] = error_std_3
                        pred_min_error_array_3[idx_tuple] = min_error_3

                        pred_error_array_4[idx_tuple] = error_pred_4
                        pred_std_array_4[idx_tuple] = error_std_4
                        pred_min_error_array_4[idx_tuple] = min_error_4

            tot_error_1 = np.concatenate(pred_std_array_1).sum()
            tot_error_2 = np.concatenate(pred_std_array_2).sum()
            tot_error_3 = np.concatenate(pred_std_array_3).sum()
            tot_error_4 = np.concatenate(pred_std_array_4).sum()
            list_of_total_errors = [tot_error_1, tot_error_2, tot_error_3, tot_error_4]
            print("total error list: ", list_of_total_errors)
            mdl_num = np.where(list_of_total_errors == np.min(list_of_total_errors))[0][0]
            if mdl_num == 0:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RationalQuadratic())
                pred_error_df = pred_error_df_1
                pred_min_error_array = pred_min_error_array_1
            elif mdl_num == 1:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RBF())
                pred_error_df = pred_error_df_2
                pred_min_error_array = pred_min_error_array_2
            elif mdl_num == 2:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_3
                pred_min_error_array = pred_min_error_array_3
            elif mdl_num == 3:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_4
                pred_min_error_array = pred_min_error_array_4
            else:
                print("Error in 'mdl_num': ", mdl_num)
                return
            pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)
            pred_error_df_sorted.to_csv("run_0_initial_predictions.csv")

        elif model_type == 'GPR_Type2':
            # Initialize Ranges
            num_HPs = 4
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], gridLength)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], gridLength)
            length_range = np.linspace(self.length_input[0], self.length_input[1], gridLength)
            alpha_range = np.linspace(self.alpha_input[0], self.alpha_input[1], gridLength)
            numN = len(noise_range)
            numS = len(sigmaF_range)
            numL = len(length_range)
            numA = len(alpha_range)

            # Getting index points
            Npts = numN * numS * numL * numA
            indices = np.linspace(0, gridLength - 1, int((Npts * decimal_points_int) ** (1 / num_HPs)),
                                  dtype=int)
            points = [(i, j, k, l) for i in indices for j in indices for k in indices for l in indices]
            Npts_int_calc = len(points)

            model_inputs = np.zeros((Npts_int_calc, 3))
            model_outputs = np.zeros((Npts_int_calc, 1))

            # Running All Combinations
            """ PART I: Initial Calculations """
            count = 0
            for n_idx in range(numN):
                noise = noise_range[n_idx]
                for s_idx in range(numS):
                    sigmaF = sigmaF_range[s_idx]
                    for l_idx in range(numL):
                        scale_length = length_range[l_idx]
                        for a_idx in range(numA):
                            alpha = alpha_range[a_idx]

                            index_current = tuple((n_idx, s_idx, l_idx))
                            if index_current in points:
                                reg = Regression(X_use, Y_use,
                                                 noise=noise, sigma_F=sigmaF, scale_length=scale_length, alpha=alpha,
                                                 Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                                 RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                                 giveKFdata=True)
                                results, bestPred, kFold_data = reg.RegressionCVMult()

                                error = float(results['rmse'].loc[str(mdl_name)])

                                model_inputs[count, 0] = noise
                                model_inputs[count, 1] = sigmaF
                                model_inputs[count, 2] = scale_length
                                model_inputs[count, 3] = alpha
                                model_outputs[count, 0] = error

                                count += 1
                                print("initial Count: " + str(count) + " / " + str(Npts_int_calc))

            kernel_1 = RationalQuadratic()
            kernel_2 = RBF()
            kernel_3 = Matern(nu=3 / 2)
            kernel_4 = Matern(nu=5 / 2)

            model_1 = gaussian_process.GaussianProcessRegressor(kernel=kernel_1, copy_X_train=False)
            model_2 = gaussian_process.GaussianProcessRegressor(kernel=kernel_2, copy_X_train=False)
            model_3 = gaussian_process.GaussianProcessRegressor(kernel=kernel_3, copy_X_train=False)
            model_4 = gaussian_process.GaussianProcessRegressor(kernel=kernel_4, copy_X_train=False)

            model_1.fit(model_inputs, model_outputs)
            model_2.fit(model_inputs, model_outputs)
            model_3.fit(model_inputs, model_outputs)
            model_4.fit(model_inputs, model_outputs)

            """ PART II: Predictions """
            zeros = np.zeros((numN, numS, numL))

            pred_error_array_1 = zeros.copy()
            pred_std_array_1 = zeros.copy()
            pred_min_error_array_1 = zeros.copy()

            pred_error_array_2 = zeros.copy()
            pred_std_array_2 = zeros.copy()
            pred_min_error_array_2 = zeros.copy()

            pred_error_array_3 = zeros.copy()
            pred_std_array_3 = zeros.copy()
            pred_min_error_array_3 = zeros.copy()

            pred_error_array_4 = zeros.copy()
            pred_std_array_4 = zeros.copy()
            pred_min_error_array_4 = zeros.copy()

            X_ts = np.zeros((1, 4))
            pred_error_df_1 = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length', 'Alpha'])
            pred_error_df_2 = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length', 'Alpha'])
            pred_error_df_3 = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length', 'Alpha'])
            pred_error_df_4 = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length', 'Alpha'])
            counter = 0
            for n_idx in range(numN):
                noise = noise_range[n_idx]
                for s_idx in range(numS):
                    sigmaF = sigmaF_range[s_idx]
                    for l_idx in range(numL):
                        scale_length = length_range[l_idx]
                        for a_idx in range(numA):
                            alpha = alpha_range[a_idx]

                            X_ts[0, :] = [noise, sigmaF, scale_length, alpha]
                            error_pred_1, error_std_1 = model_1.predict(X_ts, return_std=True)
                            error_pred_2, error_std_2 = model_2.predict(X_ts, return_std=True)
                            error_pred_3, error_std_3 = model_3.predict(X_ts, return_std=True)
                            error_pred_4, error_std_4 = model_4.predict(X_ts, return_std=True)

                            min_error_1 = (error_pred_1[0] - error_std_1[0])
                            min_error_2 = (error_pred_2[0] - error_std_2[0])
                            min_error_3 = (error_pred_3[0] - error_std_3[0])
                            min_error_4 = (error_pred_4[0] - error_std_4[0])

                            pred_error_df_1.loc[counter] = [min_error_1, noise, sigmaF, scale_length, alpha]
                            pred_error_df_2.loc[counter] = [min_error_2, noise, sigmaF, scale_length, alpha]
                            pred_error_df_3.loc[counter] = [min_error_3, noise, sigmaF, scale_length, alpha]
                            pred_error_df_4.loc[counter] = [min_error_4, noise, sigmaF, scale_length, alpha]
                            counter += 1

                            idx_tuple = (n_idx, s_idx, l_idx, a_idx)

                            pred_error_array_1[idx_tuple] = error_pred_1
                            pred_std_array_1[idx_tuple] = error_std_1
                            pred_min_error_array_1[idx_tuple] = min_error_1

                            pred_error_array_2[idx_tuple] = error_pred_2
                            pred_std_array_2[idx_tuple] = error_std_2
                            pred_min_error_array_2[idx_tuple] = min_error_2

                            pred_error_array_3[idx_tuple] = error_pred_3
                            pred_std_array_3[idx_tuple] = error_std_3
                            pred_min_error_array_3[idx_tuple] = min_error_3

                            pred_error_array_4[idx_tuple] = error_pred_4
                            pred_std_array_4[idx_tuple] = error_std_4
                            pred_min_error_array_4[idx_tuple] = min_error_4

            tot_error_1 = np.concatenate(pred_std_array_1).sum()
            tot_error_2 = np.concatenate(pred_std_array_2).sum()
            tot_error_3 = np.concatenate(pred_std_array_3).sum()
            tot_error_4 = np.concatenate(pred_std_array_4).sum()
            list_of_total_errors = [tot_error_1, tot_error_2, tot_error_3, tot_error_4]
            print("total error list: ", list_of_total_errors)
            mdl_num = np.where(list_of_total_errors == np.min(list_of_total_errors))[0][0]
            if mdl_num == 0:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RationalQuadratic())
                pred_error_df = pred_error_df_1
                pred_min_error_array = pred_min_error_array_1
            elif mdl_num == 1:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RBF())
                pred_error_df = pred_error_df_2
                pred_min_error_array = pred_min_error_array_2
            elif mdl_num == 2:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_3
                pred_min_error_array = pred_min_error_array_3
            elif mdl_num == 3:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_4
                pred_min_error_array = pred_min_error_array_4
            else:
                print("Error in 'mdl_num': ", mdl_num)
                return
            pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)
            pred_error_df_sorted.to_csv("run_0_initial_predictions.csv")

        else:
            print("MODEL TYPE ERROR: ", model_type)
            return

        print("initial_predictions --  Complete")

        return model_use, [model_inputs, model_outputs], pred_min_error_array

    def initial_predictions_and_calculations(self):
        """ Method """
        """
        This function is the initial function. It serves multiple purposes.
        First, the function runs an initial small percent of the full hyperparameter space and calulates the errors.
        Then, using this data, it trains 4 GPR models (Rational Quadratic, RBF, Matern 3/2, and Matern 5/2) and picks
        the one with the lowest total (summed) standard deviation from the 4, over predictions across the whole space.
        It saves the best model and its data.

        """

        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name
        decimal_points_int = self.decimal_points_int
        decimal_points_top = self.decimal_points_top
        gridLength = self.gridLength_AL
        model_type = self.model_type
        print("Model: ", model_type)

        print("initial_predictions --  Started")

        # RUN INITIAL PREDICTIONS --------------------------------------------------------------------------------------
        if model_type == 'SVM_Type1':
            # Initialize Ranges
            num_HPs = 2
            C_range = np.linspace(self.C_input[0], self.C_input[1], gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], gridLength)
            numC = len(C_range)
            numE = len(epsilon_range)
            # Getting index points
            Npts = numC * numE
            indices = np.linspace(0, gridLength - 1, int((Npts * decimal_points_int) ** (1 / num_HPs)), dtype=int)
            points = [(i, j) for i in indices for j in indices]
            Npts_int_calc = len(points)

            model_inputs = np.zeros((Npts_int_calc, 2))
            model_outputs = np.zeros((Npts_int_calc, 1))

            # Running All
            """ PART I: Initial Calculations """
            count = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]

                    index_current = tuple((C_idx, e_idx))
                    if index_current in points:
                        reg = Regression(X_use, Y_use,
                                         C=C, epsilon=epsilon,
                                         Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                         RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                         giveKFdata=True)
                        results, bestPred, kFold_data = reg.RegressionCVMult()

                        error = float(results['rmse'].loc[str(mdl_name)])

                        model_inputs[count, 0] = C
                        model_inputs[count, 1] = epsilon
                        model_outputs[count, 0] = error

                        count += 1
                        print("initial Count: " + str(count) + " / " + str(Npts_int_calc))

            kernel_1 = RationalQuadratic()
            kernel_2 = RBF()
            kernel_3 = Matern(nu=3 / 2)
            kernel_4 = Matern(nu=5 / 2)

            model_1 = gaussian_process.GaussianProcessRegressor(kernel=kernel_1, copy_X_train=False)
            model_2 = gaussian_process.GaussianProcessRegressor(kernel=kernel_2, copy_X_train=False)
            model_3 = gaussian_process.GaussianProcessRegressor(kernel=kernel_3, copy_X_train=False)
            model_4 = gaussian_process.GaussianProcessRegressor(kernel=kernel_4, copy_X_train=False)

            model_1.fit(model_inputs, model_outputs)
            model_2.fit(model_inputs, model_outputs)
            model_3.fit(model_inputs, model_outputs)
            model_4.fit(model_inputs, model_outputs)

            """ PART II: Predictions """
            zeros = np.zeros((numC, numE))

            pred_error_array_1 = zeros.copy()
            pred_std_array_1 = zeros.copy()
            pred_min_error_array_1 = zeros.copy()

            pred_error_array_2 = zeros.copy()
            pred_std_array_2 = zeros.copy()
            pred_min_error_array_2 = zeros.copy()

            pred_error_array_3 = zeros.copy()
            pred_std_array_3 = zeros.copy()
            pred_min_error_array_3 = zeros.copy()

            pred_error_array_4 = zeros.copy()
            pred_std_array_4 = zeros.copy()
            pred_min_error_array_4 = zeros.copy()

            X_ts = np.zeros((1, 2))
            pred_error_df_1 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon'])
            pred_error_df_2 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon'])
            pred_error_df_3 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon'])
            pred_error_df_4 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon'])
            counter = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]

                    X_ts[0, :] = [C, epsilon]
                    error_pred_1, error_std_1 = model_1.predict(X_ts, return_std=True)
                    error_pred_2, error_std_2 = model_2.predict(X_ts, return_std=True)
                    error_pred_3, error_std_3 = model_3.predict(X_ts, return_std=True)
                    error_pred_4, error_std_4 = model_4.predict(X_ts, return_std=True)

                    min_error_1 = (error_pred_1[0] - error_std_1[0])
                    min_error_2 = (error_pred_2[0] - error_std_2[0])
                    min_error_3 = (error_pred_3[0] - error_std_3[0])
                    min_error_4 = (error_pred_4[0] - error_std_4[0])

                    pred_error_df_1.loc[counter] = [min_error_1, C, epsilon]
                    pred_error_df_2.loc[counter] = [min_error_2, C, epsilon]
                    pred_error_df_3.loc[counter] = [min_error_3, C, epsilon]
                    pred_error_df_4.loc[counter] = [min_error_4, C, epsilon]
                    counter += 1

                    idx_tuple = (C_idx, e_idx)

                    pred_error_array_1[idx_tuple] = error_pred_1
                    pred_std_array_1[idx_tuple] = error_std_1
                    pred_min_error_array_1[idx_tuple] = min_error_1

                    pred_error_array_2[idx_tuple] = error_pred_2
                    pred_std_array_2[idx_tuple] = error_std_2
                    pred_min_error_array_2[idx_tuple] = min_error_2

                    pred_error_array_3[idx_tuple] = error_pred_3
                    pred_std_array_3[idx_tuple] = error_std_3
                    pred_min_error_array_3[idx_tuple] = min_error_3

                    pred_error_array_4[idx_tuple] = error_pred_4
                    pred_std_array_4[idx_tuple] = error_std_4
                    pred_min_error_array_4[idx_tuple] = min_error_4

            tot_error_1 = np.concatenate(pred_std_array_1).sum()
            tot_error_2 = np.concatenate(pred_std_array_2).sum()
            tot_error_3 = np.concatenate(pred_std_array_3).sum()
            tot_error_4 = np.concatenate(pred_std_array_4).sum()
            list_of_total_errors = [tot_error_1, tot_error_2, tot_error_3, tot_error_4]
            print("total error list: ", list_of_total_errors)
            mdl_num = np.where(list_of_total_errors == np.min(list_of_total_errors))[0][0]
            if mdl_num == 0:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RationalQuadratic())
                pred_error_df = pred_error_df_1
                pred_min_error_array = pred_min_error_array_1
            elif mdl_num == 1:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RBF())
                pred_error_df = pred_error_df_2
                pred_min_error_array = pred_min_error_array_2
            elif mdl_num == 2:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_3
                pred_min_error_array = pred_min_error_array_3
            elif mdl_num == 3:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_4
                pred_min_error_array = pred_min_error_array_4
            else:
                print("Error in 'mdl_num': ", mdl_num)
                return
            pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)
            pred_error_df_sorted.to_csv("run_0_initial_predictions.csv")

        elif model_type == 'SVM_Type2':
            # Initialize Ranges
            num_HPs = 3
            C_range = np.linspace(self.C_input[0], self.C_input[1], gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], gridLength)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], gridLength)
            numC = len(C_range)
            numE = len(epsilon_range)
            numG = len(gamma_range)
            # Getting index points
            Npts = numC * numE * numG
            indices = np.linspace(0, gridLength - 1, int((Npts * decimal_points_int) ** (1 / num_HPs)),
                                  dtype=int)
            points = [(i, j, k) for i in indices for j in indices for k in indices]
            Npts_int_calc = len(points)

            model_inputs = np.zeros((Npts_int_calc, 3))
            model_outputs = np.zeros((Npts_int_calc, 1))

            # Running All Combinations
            """ PART I: Initial Calculations """
            count = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]
                    for g_idx in range(numG):
                        gamma = gamma_range[g_idx]

                        index_current = tuple((C_idx, e_idx, g_idx))
                        if index_current in points:
                            reg = Regression(X_use, Y_use,
                                             C=C, epsilon=epsilon, gamma=gamma,
                                             Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                             RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                             giveKFdata=True)
                            results, bestPred, kFold_data = reg.RegressionCVMult()

                            error = float(results['rmse'].loc[str(mdl_name)])

                            model_inputs[count, 0] = C
                            model_inputs[count, 1] = epsilon
                            model_inputs[count, 2] = gamma
                            model_outputs[count, 0] = error

                            count += 1
                            print("initial Count: " + str(count) + " / " + str(Npts_int_calc))

            kernel_1 = RationalQuadratic()
            kernel_2 = RBF()
            kernel_3 = Matern(nu=3 / 2)
            kernel_4 = Matern(nu=5 / 2)

            model_1 = gaussian_process.GaussianProcessRegressor(kernel=kernel_1, copy_X_train=False)
            model_2 = gaussian_process.GaussianProcessRegressor(kernel=kernel_2, copy_X_train=False)
            model_3 = gaussian_process.GaussianProcessRegressor(kernel=kernel_3, copy_X_train=False)
            model_4 = gaussian_process.GaussianProcessRegressor(kernel=kernel_4, copy_X_train=False)

            model_1.fit(model_inputs, model_outputs)
            model_2.fit(model_inputs, model_outputs)
            model_3.fit(model_inputs, model_outputs)
            model_4.fit(model_inputs, model_outputs)

            """ PART II: Predictions """
            zeros = np.zeros((numC, numE, numG))

            pred_error_array_1 = zeros.copy()
            pred_std_array_1 = zeros.copy()
            pred_min_error_array_1 = zeros.copy()

            pred_error_array_2 = zeros.copy()
            pred_std_array_2 = zeros.copy()
            pred_min_error_array_2 = zeros.copy()

            pred_error_array_3 = zeros.copy()
            pred_std_array_3 = zeros.copy()
            pred_min_error_array_3 = zeros.copy()

            pred_error_array_4 = zeros.copy()
            pred_std_array_4 = zeros.copy()
            pred_min_error_array_4 = zeros.copy()

            X_ts = np.zeros((1, 3))
            pred_error_df_1 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma'])
            pred_error_df_2 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma'])
            pred_error_df_3 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma'])
            pred_error_df_4 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma'])
            counter = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]
                    for g_idx in range(numG):
                        gamma = gamma_range[g_idx]

                        X_ts[0, :] = [C, epsilon, gamma]
                        error_pred_1, error_std_1 = model_1.predict(X_ts, return_std=True)
                        error_pred_2, error_std_2 = model_2.predict(X_ts, return_std=True)
                        error_pred_3, error_std_3 = model_3.predict(X_ts, return_std=True)
                        error_pred_4, error_std_4 = model_4.predict(X_ts, return_std=True)

                        min_error_1 = (error_pred_1[0] - error_std_1[0])
                        min_error_2 = (error_pred_2[0] - error_std_2[0])
                        min_error_3 = (error_pred_3[0] - error_std_3[0])
                        min_error_4 = (error_pred_4[0] - error_std_4[0])

                        pred_error_df_1.loc[counter] = [min_error_1, C, epsilon, gamma]
                        pred_error_df_2.loc[counter] = [min_error_2, C, epsilon, gamma]
                        pred_error_df_3.loc[counter] = [min_error_3, C, epsilon, gamma]
                        pred_error_df_4.loc[counter] = [min_error_4, C, epsilon, gamma]
                        counter += 1

                        idx_tuple = (C_idx, e_idx, g_idx)

                        pred_error_array_1[idx_tuple] = error_pred_1
                        pred_std_array_1[idx_tuple] = error_std_1
                        pred_min_error_array_1[idx_tuple] = min_error_1

                        pred_error_array_2[idx_tuple] = error_pred_2
                        pred_std_array_2[idx_tuple] = error_std_2
                        pred_min_error_array_2[idx_tuple] = min_error_2

                        pred_error_array_3[idx_tuple] = error_pred_3
                        pred_std_array_3[idx_tuple] = error_std_3
                        pred_min_error_array_3[idx_tuple] = min_error_3

                        pred_error_array_4[idx_tuple] = error_pred_4
                        pred_std_array_4[idx_tuple] = error_std_4
                        pred_min_error_array_4[idx_tuple] = min_error_4

            tot_error_1 = np.concatenate(pred_std_array_1).sum()
            tot_error_2 = np.concatenate(pred_std_array_2).sum()
            tot_error_3 = np.concatenate(pred_std_array_3).sum()
            tot_error_4 = np.concatenate(pred_std_array_4).sum()
            list_of_total_errors = [tot_error_1, tot_error_2, tot_error_3, tot_error_4]
            print("total error list: ", list_of_total_errors)
            mdl_num = np.where(list_of_total_errors == np.min(list_of_total_errors))[0][0]
            if mdl_num == 0:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RationalQuadratic())
                pred_error_df = pred_error_df_1
                pred_min_error_array = pred_min_error_array_1
            elif mdl_num == 1:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RBF())
                pred_error_df = pred_error_df_2
                pred_min_error_array = pred_min_error_array_2
            elif mdl_num == 2:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_3
                pred_min_error_array = pred_min_error_array_3
            elif mdl_num == 3:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_4
                pred_min_error_array = pred_min_error_array_4
            else:
                print("Error in 'mdl_num': ", mdl_num)
                return
            pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)
            pred_error_df_sorted.to_csv("run_0_initial_predictions.csv")

        elif model_type == 'SVM_Type3':
            # Initialize Ranges
            num_HPs = 4
            C_range = np.linspace(self.C_input[0], self.C_input[1], gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], gridLength)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], gridLength)
            coef0_range = np.linspace(self.coef0_input[0], self.coef0_input[1], gridLength)
            numC = len(C_range)
            numE = len(epsilon_range)
            numG = len(gamma_range)
            numC0 = len(coef0_range)

            # Getting index points
            Npts = numC * numE * numG * numC0
            indices = np.linspace(0, gridLength - 1, int((Npts * decimal_points_int) ** (1 / num_HPs)),
                                  dtype=int)
            points = [(i, j, k, l) for i in indices for j in indices for k in indices for l in indices]
            Npts_int_calc = len(points)

            model_inputs = np.zeros((Npts_int_calc, 4))
            model_outputs = np.zeros((Npts_int_calc, 1))

            # Running All Combinations
            """ PART I: Initial Calculations """
            count = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]
                    for g_idx in range(numG):
                        gamma = gamma_range[g_idx]
                        for c0_idx in range(numC0):
                            c0 = coef0_range[c0_idx]

                            index_current = tuple((C_idx, e_idx, g_idx, c0_idx))
                            if index_current in points:
                                reg = Regression(X_use, Y_use,
                                                 C=C, epsilon=epsilon, gamma=gamma, coef0=c0,
                                                 Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                                 RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                                 giveKFdata=True)
                                results, bestPred, kFold_data = reg.RegressionCVMult()

                                error = float(results['rmse'].loc[str(mdl_name)])

                                model_inputs[count, 0] = C
                                model_inputs[count, 1] = epsilon
                                model_inputs[count, 2] = gamma
                                model_inputs[count, 3] = c0
                                model_outputs[count, 0] = error

                                count += 1
                                print("initial Count: " + str(count) + " / " + str(Npts_int_calc))

            kernel_1 = RationalQuadratic()
            kernel_2 = RBF()
            kernel_3 = Matern(nu=3 / 2)
            kernel_4 = Matern(nu=5 / 2)

            model_1 = gaussian_process.GaussianProcessRegressor(kernel=kernel_1, copy_X_train=False)
            model_2 = gaussian_process.GaussianProcessRegressor(kernel=kernel_2, copy_X_train=False)
            model_3 = gaussian_process.GaussianProcessRegressor(kernel=kernel_3, copy_X_train=False)
            model_4 = gaussian_process.GaussianProcessRegressor(kernel=kernel_4, copy_X_train=False)

            model_1.fit(model_inputs, model_outputs)
            model_2.fit(model_inputs, model_outputs)
            model_3.fit(model_inputs, model_outputs)
            model_4.fit(model_inputs, model_outputs)

            """ PART II: Predictions """
            zeros = np.zeros((numC, numE, numG, numC0))

            pred_error_array_1 = zeros.copy()
            pred_std_array_1 = zeros.copy()
            pred_min_error_array_1 = zeros.copy()

            pred_error_array_2 = zeros.copy()
            pred_std_array_2 = zeros.copy()
            pred_min_error_array_2 = zeros.copy()

            pred_error_array_3 = zeros.copy()
            pred_std_array_3 = zeros.copy()
            pred_min_error_array_3 = zeros.copy()

            pred_error_array_4 = zeros.copy()
            pred_std_array_4 = zeros.copy()
            pred_min_error_array_4 = zeros.copy()

            X_ts = np.zeros((1, 3))
            pred_error_df_1 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma', 'Coef0'])
            pred_error_df_2 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma', 'Coef0'])
            pred_error_df_3 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma', 'Coef0'])
            pred_error_df_4 = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma', 'Coef0'])
            counter = 0
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]
                    for g_idx in range(numG):
                        gamma = gamma_range[g_idx]
                        for c0_idx in range(numC0):
                            c0 = coef0_range[c0_idx]

                            X_ts[0, :] = [C, epsilon, gamma, c0]
                            error_pred_1, error_std_1 = model_1.predict(X_ts, return_std=True)
                            error_pred_2, error_std_2 = model_2.predict(X_ts, return_std=True)
                            error_pred_3, error_std_3 = model_3.predict(X_ts, return_std=True)
                            error_pred_4, error_std_4 = model_4.predict(X_ts, return_std=True)

                            min_error_1 = (error_pred_1[0] - error_std_1[0])
                            min_error_2 = (error_pred_2[0] - error_std_2[0])
                            min_error_3 = (error_pred_3[0] - error_std_3[0])
                            min_error_4 = (error_pred_4[0] - error_std_4[0])

                            pred_error_df_1.loc[counter] = [min_error_1, C, epsilon, gamma, c0]
                            pred_error_df_2.loc[counter] = [min_error_2, C, epsilon, gamma, c0]
                            pred_error_df_3.loc[counter] = [min_error_3, C, epsilon, gamma, c0]
                            pred_error_df_4.loc[counter] = [min_error_4, C, epsilon, gamma, c0]
                            counter += 1

                            idx_tuple = (C_idx, e_idx, g_idx, c0_idx)

                            pred_error_array_1[idx_tuple] = error_pred_1
                            pred_std_array_1[idx_tuple] = error_std_1
                            pred_min_error_array_1[idx_tuple] = min_error_1

                            pred_error_array_2[idx_tuple] = error_pred_2
                            pred_std_array_2[idx_tuple] = error_std_2
                            pred_min_error_array_2[idx_tuple] = min_error_2

                            pred_error_array_3[idx_tuple] = error_pred_3
                            pred_std_array_3[idx_tuple] = error_std_3
                            pred_min_error_array_3[idx_tuple] = min_error_3

                            pred_error_array_4[idx_tuple] = error_pred_4
                            pred_std_array_4[idx_tuple] = error_std_4
                            pred_min_error_array_4[idx_tuple] = min_error_4

            tot_error_1 = np.concatenate(pred_std_array_1).sum()
            tot_error_2 = np.concatenate(pred_std_array_2).sum()
            tot_error_3 = np.concatenate(pred_std_array_3).sum()
            tot_error_4 = np.concatenate(pred_std_array_4).sum()
            list_of_total_errors = [tot_error_1, tot_error_2, tot_error_3, tot_error_4]
            print("total error list: ", list_of_total_errors)
            mdl_num = np.where(list_of_total_errors == np.min(list_of_total_errors))[0][0]
            if mdl_num == 0:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RationalQuadratic())
                pred_error_df = pred_error_df_1
                pred_min_error_array = pred_min_error_array_1
            elif mdl_num == 1:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RBF())
                pred_error_df = pred_error_df_2
                pred_min_error_array = pred_min_error_array_2
            elif mdl_num == 2:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_3
                pred_min_error_array = pred_min_error_array_3
            elif mdl_num == 3:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_4
                pred_min_error_array = pred_min_error_array_4
            else:
                print("Error in 'mdl_num': ", mdl_num)
                return
            pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)
            pred_error_df_sorted.to_csv("run_0_initial_predictions.csv")

        elif model_type == 'GPR_Type1':
            # Initialize Ranges
            num_HPs = 3
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], gridLength)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], gridLength)
            length_range = np.linspace(self.length_input[0], self.length_input[1], gridLength)
            numN = len(noise_range)
            numS = len(sigmaF_range)
            numL = len(length_range)

            # Getting index points
            Npts = numN * numS * numL
            indices = np.linspace(0, gridLength - 1, int((Npts * decimal_points_int) ** (1 / num_HPs)),
                                  dtype=int)
            points = [(i, j, k) for i in indices for j in indices for k in indices]
            Npts_int_calc = len(points)

            model_inputs = np.zeros((Npts_int_calc, 3))
            model_outputs = np.zeros((Npts_int_calc, 1))

            # Running All Combinations
            """ PART I: Initial Calculations """
            count = 0
            for n_idx in range(numN):
                noise = noise_range[n_idx]
                for s_idx in range(numS):
                    sigmaF = sigmaF_range[s_idx]
                    for l_idx in range(numL):
                        scale_length = length_range[l_idx]

                        index_current = tuple((n_idx, s_idx, l_idx))
                        if index_current in points:
                            reg = Regression(X_use, Y_use,
                                             noise=noise, sigma_F=sigmaF, scale_length=scale_length,
                                             Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                             RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                             giveKFdata=True)
                            results, bestPred, kFold_data = reg.RegressionCVMult()

                            error = float(results['rmse'].loc[str(mdl_name)])

                            model_inputs[count, 0] = noise
                            model_inputs[count, 1] = sigmaF
                            model_inputs[count, 2] = scale_length
                            model_outputs[count, 0] = error

                            count += 1
                            print("initial Count: " + str(count) + " / " + str(Npts_int_calc))

            kernel_1 = RationalQuadratic()
            kernel_2 = RBF()
            kernel_3 = Matern(nu=3 / 2)
            kernel_4 = Matern(nu=5 / 2)

            model_1 = gaussian_process.GaussianProcessRegressor(kernel=kernel_1, copy_X_train=False)
            model_2 = gaussian_process.GaussianProcessRegressor(kernel=kernel_2, copy_X_train=False)
            model_3 = gaussian_process.GaussianProcessRegressor(kernel=kernel_3, copy_X_train=False)
            model_4 = gaussian_process.GaussianProcessRegressor(kernel=kernel_4, copy_X_train=False)

            model_1.fit(model_inputs, model_outputs)
            model_2.fit(model_inputs, model_outputs)
            model_3.fit(model_inputs, model_outputs)
            model_4.fit(model_inputs, model_outputs)

            """ PART II: Predictions """
            zeros = np.zeros((numN, numS, numL))

            pred_error_array_1 = zeros.copy()
            pred_std_array_1 = zeros.copy()
            pred_min_error_array_1 = zeros.copy()

            pred_error_array_2 = zeros.copy()
            pred_std_array_2 = zeros.copy()
            pred_min_error_array_2 = zeros.copy()

            pred_error_array_3 = zeros.copy()
            pred_std_array_3 = zeros.copy()
            pred_min_error_array_3 = zeros.copy()

            pred_error_array_4 = zeros.copy()
            pred_std_array_4 = zeros.copy()
            pred_min_error_array_4 = zeros.copy()

            X_ts = np.zeros((1, 3))
            pred_error_df_1 = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length'])
            pred_error_df_2 = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length'])
            pred_error_df_3 = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length'])
            pred_error_df_4 = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length'])
            counter = 0
            for n_idx in range(numN):
                noise = noise_range[n_idx]
                for s_idx in range(numS):
                    sigmaF = sigmaF_range[s_idx]
                    for l_idx in range(numL):
                        scale_length = length_range[l_idx]

                        X_ts[0, :] = [noise, sigmaF, scale_length]
                        error_pred_1, error_std_1 = model_1.predict(X_ts, return_std=True)
                        error_pred_2, error_std_2 = model_2.predict(X_ts, return_std=True)
                        error_pred_3, error_std_3 = model_3.predict(X_ts, return_std=True)
                        error_pred_4, error_std_4 = model_4.predict(X_ts, return_std=True)

                        min_error_1 = (error_pred_1[0] - error_std_1[0])
                        min_error_2 = (error_pred_2[0] - error_std_2[0])
                        min_error_3 = (error_pred_3[0] - error_std_3[0])
                        min_error_4 = (error_pred_4[0] - error_std_4[0])

                        pred_error_df_1.loc[counter] = [min_error_1, noise, sigmaF, scale_length]
                        pred_error_df_2.loc[counter] = [min_error_2, noise, sigmaF, scale_length]
                        pred_error_df_3.loc[counter] = [min_error_3, noise, sigmaF, scale_length]
                        pred_error_df_4.loc[counter] = [min_error_4, noise, sigmaF, scale_length]
                        counter += 1

                        idx_tuple = (n_idx, s_idx, l_idx)

                        pred_error_array_1[idx_tuple] = error_pred_1
                        pred_std_array_1[idx_tuple] = error_std_1
                        pred_min_error_array_1[idx_tuple] = min_error_1

                        pred_error_array_2[idx_tuple] = error_pred_2
                        pred_std_array_2[idx_tuple] = error_std_2
                        pred_min_error_array_2[idx_tuple] = min_error_2

                        pred_error_array_3[idx_tuple] = error_pred_3
                        pred_std_array_3[idx_tuple] = error_std_3
                        pred_min_error_array_3[idx_tuple] = min_error_3

                        pred_error_array_4[idx_tuple] = error_pred_4
                        pred_std_array_4[idx_tuple] = error_std_4
                        pred_min_error_array_4[idx_tuple] = min_error_4

            tot_error_1 = np.concatenate(pred_std_array_1).sum()
            tot_error_2 = np.concatenate(pred_std_array_2).sum()
            tot_error_3 = np.concatenate(pred_std_array_3).sum()
            tot_error_4 = np.concatenate(pred_std_array_4).sum()
            list_of_total_errors = [tot_error_1, tot_error_2, tot_error_3, tot_error_4]
            print("total error list: ", list_of_total_errors)
            mdl_num = np.where(list_of_total_errors == np.min(list_of_total_errors))[0][0]
            if mdl_num == 0:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RationalQuadratic())
                pred_error_df = pred_error_df_1
                pred_min_error_array = pred_min_error_array_1
            elif mdl_num == 1:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RBF())
                pred_error_df = pred_error_df_2
                pred_min_error_array = pred_min_error_array_2
            elif mdl_num == 2:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_3
                pred_min_error_array = pred_min_error_array_3
            elif mdl_num == 3:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_4
                pred_min_error_array = pred_min_error_array_4
            else:
                print("Error in 'mdl_num': ", mdl_num)
                return
            pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)
            pred_error_df_sorted.to_csv("run_0_initial_predictions.csv")

        elif model_type == 'GPR_Type2':
            # Initialize Ranges
            num_HPs = 4
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], gridLength)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], gridLength)
            length_range = np.linspace(self.length_input[0], self.length_input[1], gridLength)
            alpha_range = np.linspace(self.alpha_input[0], self.alpha_input[1], gridLength)
            numN = len(noise_range)
            numS = len(sigmaF_range)
            numL = len(length_range)
            numA = len(alpha_range)

            # Getting index points
            Npts = numN * numS * numL * numA
            indices = np.linspace(0, gridLength - 1, int((Npts * decimal_points_int) ** (1 / num_HPs)),
                                  dtype=int)
            points = [(i, j, k, l) for i in indices for j in indices for k in indices for l in indices]
            Npts_int_calc = len(points)

            model_inputs = np.zeros((Npts_int_calc, 3))
            model_outputs = np.zeros((Npts_int_calc, 1))

            # Running All Combinations
            """ PART I: Initial Calculations """
            count = 0
            for n_idx in range(numN):
                noise = noise_range[n_idx]
                for s_idx in range(numS):
                    sigmaF = sigmaF_range[s_idx]
                    for l_idx in range(numL):
                        scale_length = length_range[l_idx]
                        for a_idx in range(numA):
                            alpha = alpha_range[a_idx]

                            index_current = tuple((n_idx, s_idx, l_idx))
                            if index_current in points:
                                reg = Regression(X_use, Y_use,
                                                 noise=noise, sigma_F=sigmaF, scale_length=scale_length, alpha=alpha,
                                                 Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                                 RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                                 giveKFdata=True)
                                results, bestPred, kFold_data = reg.RegressionCVMult()

                                error = float(results['rmse'].loc[str(mdl_name)])

                                model_inputs[count, 0] = noise
                                model_inputs[count, 1] = sigmaF
                                model_inputs[count, 2] = scale_length
                                model_inputs[count, 3] = alpha
                                model_outputs[count, 0] = error

                                count += 1
                                print("initial Count: " + str(count) + " / " + str(Npts_int_calc))

            kernel_1 = RationalQuadratic()
            kernel_2 = RBF()
            kernel_3 = Matern(nu=3 / 2)
            kernel_4 = Matern(nu=5 / 2)

            model_1 = gaussian_process.GaussianProcessRegressor(kernel=kernel_1, copy_X_train=False)
            model_2 = gaussian_process.GaussianProcessRegressor(kernel=kernel_2, copy_X_train=False)
            model_3 = gaussian_process.GaussianProcessRegressor(kernel=kernel_3, copy_X_train=False)
            model_4 = gaussian_process.GaussianProcessRegressor(kernel=kernel_4, copy_X_train=False)

            model_1.fit(model_inputs, model_outputs)
            model_2.fit(model_inputs, model_outputs)
            model_3.fit(model_inputs, model_outputs)
            model_4.fit(model_inputs, model_outputs)

            """ PART II: Predictions """
            zeros = np.zeros((numN, numS, numL))

            pred_error_array_1 = zeros.copy()
            pred_std_array_1 = zeros.copy()
            pred_min_error_array_1 = zeros.copy()

            pred_error_array_2 = zeros.copy()
            pred_std_array_2 = zeros.copy()
            pred_min_error_array_2 = zeros.copy()

            pred_error_array_3 = zeros.copy()
            pred_std_array_3 = zeros.copy()
            pred_min_error_array_3 = zeros.copy()

            pred_error_array_4 = zeros.copy()
            pred_std_array_4 = zeros.copy()
            pred_min_error_array_4 = zeros.copy()

            X_ts = np.zeros((1, 4))
            pred_error_df_1 = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length', 'Alpha'])
            pred_error_df_2 = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length', 'Alpha'])
            pred_error_df_3 = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length', 'Alpha'])
            pred_error_df_4 = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length', 'Alpha'])
            counter = 0
            for n_idx in range(numN):
                noise = noise_range[n_idx]
                for s_idx in range(numS):
                    sigmaF = sigmaF_range[s_idx]
                    for l_idx in range(numL):
                        scale_length = length_range[l_idx]
                        for a_idx in range(numA):
                            alpha = alpha_range[a_idx]

                            X_ts[0, :] = [noise, sigmaF, scale_length, alpha]
                            error_pred_1, error_std_1 = model_1.predict(X_ts, return_std=True)
                            error_pred_2, error_std_2 = model_2.predict(X_ts, return_std=True)
                            error_pred_3, error_std_3 = model_3.predict(X_ts, return_std=True)
                            error_pred_4, error_std_4 = model_4.predict(X_ts, return_std=True)

                            min_error_1 = (error_pred_1[0] - error_std_1[0])
                            min_error_2 = (error_pred_2[0] - error_std_2[0])
                            min_error_3 = (error_pred_3[0] - error_std_3[0])
                            min_error_4 = (error_pred_4[0] - error_std_4[0])

                            pred_error_df_1.loc[counter] = [min_error_1, noise, sigmaF, scale_length, alpha]
                            pred_error_df_2.loc[counter] = [min_error_2, noise, sigmaF, scale_length, alpha]
                            pred_error_df_3.loc[counter] = [min_error_3, noise, sigmaF, scale_length, alpha]
                            pred_error_df_4.loc[counter] = [min_error_4, noise, sigmaF, scale_length, alpha]
                            counter += 1

                            idx_tuple = (n_idx, s_idx, l_idx, a_idx)

                            pred_error_array_1[idx_tuple] = error_pred_1
                            pred_std_array_1[idx_tuple] = error_std_1
                            pred_min_error_array_1[idx_tuple] = min_error_1

                            pred_error_array_2[idx_tuple] = error_pred_2
                            pred_std_array_2[idx_tuple] = error_std_2
                            pred_min_error_array_2[idx_tuple] = min_error_2

                            pred_error_array_3[idx_tuple] = error_pred_3
                            pred_std_array_3[idx_tuple] = error_std_3
                            pred_min_error_array_3[idx_tuple] = min_error_3

                            pred_error_array_4[idx_tuple] = error_pred_4
                            pred_std_array_4[idx_tuple] = error_std_4
                            pred_min_error_array_4[idx_tuple] = min_error_4

            tot_error_1 = np.concatenate(pred_std_array_1).sum()
            tot_error_2 = np.concatenate(pred_std_array_2).sum()
            tot_error_3 = np.concatenate(pred_std_array_3).sum()
            tot_error_4 = np.concatenate(pred_std_array_4).sum()
            list_of_total_errors = [tot_error_1, tot_error_2, tot_error_3, tot_error_4]
            print("total error list: ", list_of_total_errors)
            mdl_num = np.where(list_of_total_errors == np.min(list_of_total_errors))[0][0]
            if mdl_num == 0:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RationalQuadratic())
                pred_error_df = pred_error_df_1
                pred_min_error_array = pred_min_error_array_1
            elif mdl_num == 1:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=RBF())
                pred_error_df = pred_error_df_2
                pred_min_error_array = pred_min_error_array_2
            elif mdl_num == 2:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_3
                pred_min_error_array = pred_min_error_array_3
            elif mdl_num == 3:
                model_use = gaussian_process.GaussianProcessRegressor(kernel=Matern(nu=3 / 2))
                pred_error_df = pred_error_df_4
                pred_min_error_array = pred_min_error_array_4
            else:
                print("Error in 'mdl_num': ", mdl_num)
                return
            pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)
            pred_error_df_sorted.to_csv("run_0_initial_predictions.csv")

        else:
            print("MODEL TYPE ERROR: ", model_type)
            return

        print("initial_predictions --  Complete")

        model_data = [model_inputs, model_outputs]

        # Calculations
        storage_df, model_data_final = self.PartIII_final_calculations(pred_error_df_sorted, model_data, 0)



        return model_use, model_data_final, storage_df

    def top_layer_predictions(self, model, model_data, run_number):
        """ Method """
        """
        Before and after running each set of 'High-Density' areas, we must zoom back out and examine the full area
        under the light of all new data added to the model and make new predictions. 
        With each new set of high-density calculations, our model becomes deeper and we must reevaluate where
        to place the new high density regions based off new predictions.
        """
        model_inputs = model_data[0]
        model_outputs = model_data[1]
        mdl = model.fit(model_inputs, model_outputs)

        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name
        decimal_points_int = self.decimal_points_int
        decimal_points_top = self.decimal_points_top
        gridLength = self.gridLength_AL
        model_type = self.model_type

        print("top_layer_predictions -- Started")

        if model_type == 'SVM_Type1':
            # Initialize Ranges
            C_range = np.linspace(self.C_input[0], self.C_input[1], gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], gridLength)
            numC = len(C_range)
            numE = len(epsilon_range)
            # Running All Combinations
            min_error_array = np.zeros((numC, numE))
            input_test = np.zeros((1,2))
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]

                    input_test[0, :] = [C, epsilon]
                    error_pred, error_std = mdl.predict(input_test, return_std=True)
                    if run_number != 'None':
                        min_error = (error_pred[0] - ((1/run_number) * error_std[0]))
                    else:
                        min_error = error_pred[0]
                    min_error_array[C_idx, e_idx] = min_error

        elif model_type == 'SVM_Type2':
            # Initialize Ranges
            C_range = np.linspace(self.C_input[0], self.C_input[1], gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], gridLength)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], gridLength)
            numC = len(C_range)
            numE = len(epsilon_range)
            numG = len(gamma_range)
            # Running All Combinations
            min_error_array = np.zeros((numC, numE, numG))
            input_test = np.zeros((1, 3))
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]
                    for g_idx in range(numG):
                        gamma = gamma_range[g_idx]

                        input_test[0, :] = [C, epsilon, gamma]
                        error_pred, error_std = mdl.predict(input_test, return_std=True)
                        if run_number != 'None':
                            min_error = (error_pred[0] - ((1 / run_number) * error_std[0]))
                        else:
                            min_error = error_pred[0]
                        min_error_array[C_idx, e_idx, g_idx] = min_error

        elif model_type == 'SVM_Type3':
            # Initialize Ranges
            C_range = np.linspace(self.C_input[0], self.C_input[1], gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], gridLength)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], gridLength)
            coef0_range = np.linspace(self.coef0_input[0], self.coef0_input[1], gridLength)
            numC = len(C_range)
            numE = len(epsilon_range)
            numG = len(gamma_range)
            numC0 = len(coef0_range)
            # Running All Combinations
            min_error_array = np.zeros((numC, numE, numG, numC0))
            input_test = np.zeros((1, 4))
            for C_idx in range(numC):
                C = C_range[C_idx]
                for e_idx in range(numE):
                    epsilon = epsilon_range[e_idx]
                    for g_idx in range(numG):
                        gamma = gamma_range[g_idx]
                        for c0_idx in range(numC0):
                            c0 = coef0_range[c0_idx]

                            input_test[0, :] = [C, epsilon, gamma, c0]
                            error_pred, error_std = mdl.predict(input_test, return_std=True)
                            if run_number != 'None':
                                min_error = (error_pred[0] - ((1 / run_number) * error_std[0]))
                            else:
                                min_error = error_pred[0]
                            min_error_array[C_idx, e_idx, g_idx, c0_idx] = min_error

        elif model_type == 'GPR_Type1':
            # Initialize Ranges
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], gridLength)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], gridLength)
            length_range = np.linspace(self.length_input[0], self.length_input[1], gridLength)
            numN = len(noise_range)
            numS = len(sigmaF_range)
            numL = len(length_range)
            # Running All Combinations
            min_error_array = np.zeros((numN, numS, numL))
            input_test = np.zeros((1, 3))
            for n_idx in range(numN):
                noise = noise_range[n_idx]
                for s_idx in range(numS):
                    sigmaF = sigmaF_range[s_idx]
                    for l_idx in range(numL):
                        scale_length = length_range[l_idx]

                        input_test[0, :] = [noise, sigmaF, scale_length]
                        error_pred, error_std = mdl.predict(input_test, return_std=True)
                        if run_number != 'None':
                            min_error = (error_pred[0] - ((1 / run_number) * error_std[0]))
                        else:
                            min_error = error_pred[0]
                        min_error_array[n_idx, s_idx, l_idx] = min_error

        elif model_type == 'GPR_Type2':
            # Initialize Ranges
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], gridLength)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], gridLength)
            length_range = np.linspace(self.length_input[0], self.length_input[1], gridLength)
            alpha_range = np.linspace(self.alpha_input[0], self.alpha_input[1], gridLength)
            numN = len(noise_range)
            numS = len(sigmaF_range)
            numL = len(length_range)
            numA = len(alpha_range)
            # Running All Combinations
            min_error_array = np.zeros((numN, numS, numL, numA))
            input_test = np.zeros((1, 4))
            for n_idx in range(numN):
                noise = noise_range[n_idx]
                for s_idx in range(numS):
                    sigmaF = sigmaF_range[s_idx]
                    for l_idx in range(numL):
                        scale_length = length_range[l_idx]
                        for a_idx in range(numA):
                            alpha = alpha_range[a_idx]

                            input_test[0, :] = [noise, sigmaF, scale_length, alpha]
                            error_pred, error_std = mdl.predict(input_test, return_std=True)
                            if run_number != 'None':
                                min_error = (error_pred[0] - ((1 / run_number) * error_std[0]))
                            else:
                                min_error = error_pred[0]
                            min_error_array[n_idx, s_idx, l_idx, a_idx] = min_error

        else:
            print("Error in Model-Type: ", model_type)
            return

        print("top_layer_predictions -- Complete")

        return min_error_array

    def top_layer_predictions_and_calculations(self, model, model_data, run_number, stor_num_counter):

        model_type = self.model_type
        gridLength = self.gridLength_AL

        if model_type == 'SVM_Type1':
            # Initialize Ranges
            C_input = self.C_input
            epsilon_input = self.epsilon_input
            C_range = np.linspace(C_input[0], C_input[1], gridLength)
            epsilon_range = np.linspace(epsilon_input[0], epsilon_input[1], gridLength)

            ranges = [C_range, epsilon_range]

        elif model_type == 'SVM_Type2':
            # Initialize Ranges
            C_input = self.C_input
            epsilon_input = self.epsilon_input
            gamma_input = self.gamma_input
            C_range = np.linspace(C_input[0], C_input[1], gridLength)
            epsilon_range = np.linspace(epsilon_input[0], epsilon_input[1], gridLength)
            gamma_range = np.linspace(gamma_input[0], gamma_input[1], gridLength)

            ranges = [C_range, epsilon_range, gamma_range]

        elif model_type == 'SVM_Type3':
            # Initialize Ranges
            C_input = self.C_input
            epsilon_input = self.epsilon_input
            gamma_input = self.gamma_input
            coef0_input = self.coef0_input
            C_range = np.linspace(C_input[0], C_input[1], gridLength)
            epsilon_range = np.linspace(epsilon_input[0], epsilon_input[1], gridLength)
            gamma_range = np.linspace(gamma_input[0], gamma_input[1], gridLength)
            coef0_range = np.linspace(coef0_input[0], coef0_input[1], gridLength)

            ranges = [C_range, epsilon_range, gamma_range, coef0_range]

        elif model_type == 'GPR_Type1':
            # Initialize Ranges
            noise_input = self.noise_input
            sigmaF_input = self.sigmaF_input
            length_input = self.length_input
            noise_range = np.linspace(noise_input[0], noise_input[1], gridLength)
            sigmaF_range = np.linspace(sigmaF_input[0], sigmaF_input[1], gridLength)
            length_range = np.linspace(length_input[0], length_input[1], gridLength)

            """ PART I """
            ranges = [noise_range, sigmaF_range, length_range]

        elif model_type == 'GPR_Type2':
            # Initialize Ranges
            noise_input = self.noise_input
            sigmaF_input = self.sigmaF_input
            length_input = self.length_input
            alpha_input = self.alpha_input
            noise_range = np.linspace(noise_input[0], noise_input[1], gridLength)
            sigmaF_range = np.linspace(sigmaF_input[0], sigmaF_input[1], gridLength)
            length_range = np.linspace(length_input[0], length_input[1], gridLength)
            alpha_range = np.linspace(alpha_input[0], alpha_input[1], gridLength)

            ranges = [noise_range, sigmaF_range, length_range, alpha_range]

        else:
            print("Error in Model-Type: ", model_type)
            return

        """ PART II """
        pred_min_error_array, pred_error_df_sorted = self.PartII_predictions(ranges, model, model_data,
                                                                             run_number)
        pred_error_df_sorted.to_csv("run_" + str(stor_num_counter) + "_TL_predictions.csv")

        """ PART III """
        storage_df, model_data_final = self.PartIII_final_calculations(pred_error_df_sorted, model_data,
                                                                       stor_num_counter)
        storage_df.to_csv("run_" + str(stor_num_counter) + "_TL_final_calcs.csv")

        return storage_df, model_data_final

    def intake_top_layer_predictions_and_determine_new_grids(self, min_error_array, list_of_verts=()):
        """ Method """
        """
        Once top-layer predictions have been made, this function finds the new high-density regions to examine.
        It aslo takes into account all previous high-density regions, and if a new point falls in an old area, 
        the model zooms in even futher into this area to get even higher density points.
        """

        # list_of_verts: When a top layer prediction is made and the top points are found, the region around these
        #                points is examined in as a 'high-density range' (referring to the high density of points being
        #                examined in this region). Once the area is investigated, the grid vertices are saved in this
        #                list. If a point later should fall in these grids, the given point should be zoomed in on more
        #                as much of the given region has already been investigated.

        model_type = self.model_type
        num_top_points = self.num_HP_zones_AL

        print("intake_top_layer_predictions_and_determine_new_grids --  Started (no end PS)")

        if model_type == 'SVM_Type1':
            # Initialize Ranges
            C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength_AL)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength_AL)

            mvmt_decimal = 0.3
            x_int_mvmt = (C_range[-1] - C_range[1]) * mvmt_decimal / 2
            y_int_mvmt = (epsilon_range[-1] - epsilon_range[1]) * mvmt_decimal / 2

            top_points = np.unravel_index(np.argsort(min_error_array.ravel())[:num_top_points], min_error_array.shape)
            list_of_vertices = []
            list_of_verts_rev = np.flip(list_of_verts)
            for pt in range(num_top_points):
                print("pt: ", pt)
                vertices = 'Empty'
                x_idx = top_points[0][pt]
                y_idx = top_points[1][pt]

                x = C_range[x_idx]
                y = epsilon_range[y_idx]

                print("(x, y): ("+str(x)+", "+str(y)+")")

                if len(list_of_verts) != 0:
                    for vert in list_of_verts_rev:
                        print("vert: ", vert)
                        x_values = vert[0]
                        y_values = vert[1]
                        if (((x_values[0] >= x) & (x >= x_values[1])) & ((y_values[0] >= y) & (y >= y_values[1]))):
                            x_mvmt = (x_values[0] - x_values[1]) * mvmt_decimal
                            y_mvmt = (y_values[0] - y_values[1]) * mvmt_decimal
                            x_max = x + x_mvmt
                            x_min = x - x_mvmt
                            if x_min < 0:
                                x_min = C_range[0] / 2
                            y_max = y + y_mvmt
                            y_min = y - y_mvmt
                            if y_min < 0:
                                y_min = epsilon_range[0] / 2

                            vertices = [(x_max, x_min), (y_max, y_min)]
                            print("vert_new: ", vertices)
                            list_of_vertices.append(vertices)
                            break
                if vertices == 'Empty':
                    x_max = x + x_int_mvmt
                    x_min = x - x_int_mvmt
                    if x_min < 0:
                        x_min = C_range[0] / 2
                    y_max = y + y_int_mvmt
                    y_min = y - y_int_mvmt
                    if y_min < 0:
                        y_min = epsilon_range[0] / 2

                    vertices = [(x_max, x_min),
                                (y_max, y_min)]
                    list_of_vertices.append(vertices)

            return list_of_vertices

        elif model_type == 'SVM_Type2':
            # Initialize Ranges
            C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength_AL)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength_AL)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], self.gridLength_AL)

            mvmt_decimal = 0.3
            x_int_mvmt = (C_range[-1] - C_range[1]) * mvmt_decimal / 2
            y_int_mvmt = (epsilon_range[-1] - epsilon_range[1]) * mvmt_decimal / 2
            z_int_mvmt = (gamma_range[-1] - gamma_range[1]) * mvmt_decimal / 2

            top_points = np.unravel_index(np.argsort(min_error_array.ravel())[:num_top_points], min_error_array.shape)
            list_of_vertices = []
            for pt in range(num_top_points):
                print("pt: ", pt)
                vertices = 'Empty'
                x_idx = top_points[0][pt]
                y_idx = top_points[1][pt]
                z_idx = top_points[2][pt]

                x = C_range[x_idx]
                y = epsilon_range[y_idx]
                z = gamma_range[z_idx]

                print("(x, y, z): (" + str(x) + ", " + str(y) + ", " + str(z) + ")")

                if len(list_of_verts) != 0:
                    for vert in list_of_verts:
                        print("vert: ", vert)
                        x_values = vert[0]
                        y_values = vert[1]
                        z_values = vert[2]
                        if (((x_values[0] >= x) & (x >= x_values[1])) & ((y_values[0] >= y) & (y >= y_values[1])) & ((z_values[0] >= z) & (z >= z_values[1]))):
                            x_mvmt = (x_values[0] - x_values[1]) * mvmt_decimal
                            y_mvmt = (y_values[0] - y_values[1]) * mvmt_decimal
                            z_mvmt = (z_values[0] - z_values[1]) * mvmt_decimal
                            x_max = x + x_mvmt
                            x_min = x - x_mvmt
                            if x_min < 0:
                                x_min = C_range[0] / 2
                            y_max = y + y_mvmt
                            y_min = y - y_mvmt
                            if y_min < 0:
                                y_min = epsilon_range[0] / 2
                            z_max = z + z_mvmt
                            z_min = z - z_mvmt
                            if z_min < 0:
                                z_min = gamma_range[0] / 2

                            vertices = [(x_max, x_min), (y_max, y_min), (z_max, z_min)]
                            list_of_vertices.append(vertices)
                            break
                if vertices == 'Empty':
                    x_max = x + x_int_mvmt
                    x_min = x - x_int_mvmt
                    if x_min < 0:
                        x_min = C_range[0] / 2
                    y_max = y + y_int_mvmt
                    y_min = y - y_int_mvmt
                    if y_min < 0:
                        y_min = epsilon_range[0] / 2
                    z_max = z + z_int_mvmt
                    z_min = z - z_int_mvmt
                    if z_min < 0:
                        z_min = gamma_range[0] / 2
                    vertices = [(x_max, x_min),
                                (y_max, y_min),
                                (z_max, z_min)]
                    list_of_vertices.append(vertices)

            return list_of_vertices

        elif model_type == 'SVM_Type3':
            # Initialize Ranges
            C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength_AL)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength_AL)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], self.gridLength_AL)
            coef0_range = np.linspace(self.coef0_input[0], self.coef0_input[1], self.gridLength_AL)

            mvmt_decimal = 0.3
            x_int_mvmt = (C_range[-1] - C_range[1]) * mvmt_decimal / 2
            y_int_mvmt = (epsilon_range[-1] - epsilon_range[1]) * mvmt_decimal / 2
            z_int_mvmt = (gamma_range[-1] - gamma_range[1]) * mvmt_decimal / 2
            i_int_mvmt = (coef0_range[-1] - coef0_range[1]) * mvmt_decimal / 2

            top_points = np.unravel_index(np.argsort(min_error_array.ravel())[:num_top_points], min_error_array.shape)
            list_of_vertices = []
            for pt in range(num_top_points):
                print("pt: ", pt)
                vertices = 'Empty'
                x_idx = top_points[0][pt]
                y_idx = top_points[1][pt]
                z_idx = top_points[2][pt]
                i_idx = top_points[3][pt]

                x = C_range[x_idx]
                y = epsilon_range[y_idx]
                z = gamma_range[z_idx]
                i = coef0_range[i_idx]

                print("(x, y, z, i): (" + str(x) + ", " + str(y) + ", " + str(z) + ", " + str(i) + ")")

                if len(list_of_verts) != 0:
                    for vert in list_of_verts:
                        print("vert: ", vert)
                        x_values = vert[0]
                        y_values = vert[1]
                        z_values = vert[2]
                        i_values = vert[3]
                        if (((x_values[0] >= x) & (x >= x_values[1])) & ((y_values[0] >= y) & (y >= y_values[1])) & ((z_values[0] >= z) & (z >= z_values[1])) & ((i_values[0] >= i) & (i >= i_values[1]))):
                            x_mvmt = (x_values[0] - x_values[1]) * mvmt_decimal
                            y_mvmt = (y_values[0] - y_values[1]) * mvmt_decimal
                            z_mvmt = (z_values[0] - z_values[1]) * mvmt_decimal
                            i_mvmt = (i_values[0] - i_values[1]) * mvmt_decimal
                            x_max = x + x_mvmt
                            x_min = x - x_mvmt
                            if x_min < 0:
                                x_min = C_range[0] / 2
                            y_max = y + y_mvmt
                            y_min = y - y_mvmt
                            if y_min < 0:
                                y_min = epsilon_range[0] / 2
                            z_max = z + z_mvmt
                            z_min = z - z_mvmt
                            if z_min < 0:
                                z_min = gamma_range[0] / 2
                            i_max = i + i_mvmt
                            i_min = i - i_mvmt
                            if i_min < 0:
                                i_min = coef0_range[0] / 2

                            vertices = [(x_max, x_min), (y_max, y_min), (z_max, z_min), (i_max, i_min)]
                            break
                if vertices == 'Empty':
                    x_max = x + x_int_mvmt
                    x_min = x - x_int_mvmt
                    if x_min < 0:
                        x_min = C_range[0] / 2
                    y_max = y + y_int_mvmt
                    y_min = y - y_int_mvmt
                    if y_min < 0:
                        y_min = epsilon_range[0] / 2
                    z_max = z + z_int_mvmt
                    z_min = z - z_int_mvmt
                    if z_min < 0:
                        z_min = gamma_range[0] / 2
                    i_max = i + i_int_mvmt
                    i_min = i - i_int_mvmt
                    if i_min < 0:
                        i_min = coef0_range[0] / 2
                    vertices = [(x_max, x_min),
                                (y_max, y_min),
                                (z_max, z_min),
                                (i_max, i_min)]
                    list_of_vertices.append(vertices)

            return list_of_vertices

        elif model_type == 'GPR_Type1':
            # Initialize Ranges
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], self.gridLength_AL)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], self.gridLength_AL)
            length_range = np.linspace(self.length_input[0], self.length_input[1], self.gridLength_AL)

            mvmt_decimal = 0.3
            x_int_mvmt = (noise_range[-1] - noise_range[1]) * mvmt_decimal / 2
            y_int_mvmt = (sigmaF_range[-1] - sigmaF_range[1]) * mvmt_decimal / 2
            z_int_mvmt = (length_range[-1] - length_range[1]) * mvmt_decimal / 2

            top_points = np.unravel_index(np.argsort(min_error_array.ravel())[:num_top_points], min_error_array.shape)
            list_of_vertices = []
            for pt in range(num_top_points):
                print("pt: ", pt)
                vertices = 'Empty'
                x_idx = top_points[0][pt]
                y_idx = top_points[1][pt]
                z_idx = top_points[2][pt]

                x = noise_range[x_idx]
                y = sigmaF_range[y_idx]
                z = length_range[z_idx]

                print("(x, y, z): (" + str(x) + ", " + str(y) + ", " + str(z) + ")")

                if len(list_of_verts) != 0:
                    for vert in list_of_verts:
                        print("vert: ", vert)
                        x_values = vert[0]
                        y_values = vert[1]
                        z_values = vert[2]
                        if (((x_values[0] >= x) & (x >= x_values[1])) & ((y_values[0] >= y) & (y >= y_values[1])) & ((z_values[0] >= z) & (z >= z_values[1]))):
                            x_mvmt = (x_values[0] - x_values[1]) * mvmt_decimal
                            y_mvmt = (y_values[0] - y_values[1]) * mvmt_decimal
                            z_mvmt = (z_values[0] - z_values[1]) * mvmt_decimal
                            x_max = x + x_mvmt
                            x_min = x - x_mvmt
                            if x_min < 0:
                                x_min = noise_range[0] / 2
                            y_max = y + y_mvmt
                            y_min = y - y_mvmt
                            if y_min < 0:
                                y_min = sigmaF_range[0] / 2
                            z_max = z + z_mvmt
                            z_min = z - z_mvmt
                            if z_min < 0:
                                z_min = length_range[0] / 2

                            vertices = [(x_max, x_min), (y_max, y_min), (z_max, z_min)]
                            break
                if vertices == 'Empty':
                    x_max = x + x_int_mvmt
                    x_min = x - x_int_mvmt
                    if x_min < 0:
                        x_min = noise_range[0] / 2
                    y_max = y + y_int_mvmt
                    y_min = y - y_int_mvmt
                    if y_min < 0:
                        y_min = sigmaF_range[0] / 2
                    z_max = z + z_int_mvmt
                    z_min = z - z_int_mvmt
                    if z_min < 0:
                        z_min = length_range[0] / 2
                    vertices = [(x_max, x_min),
                                (y_max, y_min),
                                (z_max, z_min),]
                    list_of_vertices.append(vertices)

            return list_of_vertices

        elif model_type == 'GPR_Type2':
            # Initialize Ranges
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], self.gridLength_AL)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], self.gridLength_AL)
            length_range = np.linspace(self.length_input[0], self.length_input[1], self.gridLength_AL)
            alpha_range = np.linspace(self.alpha_input[0], self.alpha_input[1], self.gridLength_AL)

            mvmt_decimal = 0.3
            x_int_mvmt = (noise_range[-1] - noise_range[1]) * mvmt_decimal / 2
            y_int_mvmt = (sigmaF_range[-1] - sigmaF_range[1]) * mvmt_decimal / 2
            z_int_mvmt = (length_range[-1] - length_range[1]) * mvmt_decimal / 2
            i_int_mvmt = (alpha_range[-1] - alpha_range[1]) * mvmt_decimal / 2

            top_points = np.unravel_index(np.argsort(min_error_array.ravel())[:num_top_points], min_error_array.shape)
            list_of_vertices = []
            for pt in range(num_top_points):
                print("pt: ", pt)
                vertices = 'Empty'
                x_idx = top_points[0][pt]
                y_idx = top_points[1][pt]
                z_idx = top_points[2][pt]
                i_idx = top_points[3][pt]

                x = noise_range[x_idx]
                y = sigmaF_range[y_idx]
                z = length_range[z_idx]
                i = alpha_range[i_idx]

                print("(x, y, z, i): (" + str(x) + ", " + str(y) + ", " + str(z) + ", " + str(i) + ")")

                if len(list_of_verts) != 0:
                    for vert in list_of_verts:
                        print("vert: ", vert)
                        x_values = vert[0]
                        y_values = vert[1]
                        z_values = vert[2]
                        i_values = vert[3]
                        if (((x_values[0] >= x) & (x >= x_values[1])) & ((y_values[0] >= y) & (y >= y_values[1])) & ((z_values[0] >= z) & (z >= z_values[1])) & ((i_values[0] >= i) & (i >= i_values[1]))):
                            x_mvmt = (x_values[0] - x_values[1]) * mvmt_decimal
                            y_mvmt = (y_values[0] - y_values[1]) * mvmt_decimal
                            z_mvmt = (z_values[0] - z_values[1]) * mvmt_decimal
                            i_mvmt = (i_values[0] - i_values[1]) * mvmt_decimal
                            x_max = x + x_mvmt
                            x_min = x - x_mvmt
                            if x_min < 0:
                                x_min = noise_range[0] / 2
                            y_max = y + y_mvmt
                            y_min = y - y_mvmt
                            if y_min < 0:
                                y_min = sigmaF_range[0] / 2
                            z_max = z + z_mvmt
                            z_min = z - z_mvmt
                            if z_min < 0:
                                z_min = length_range[0] / 2
                            i_max = i + i_mvmt
                            i_min = i - i_mvmt
                            if i_min < 0:
                                i_min = alpha_range[0] / 2

                            vertices = [(x_max, x_min), (y_max, y_min), (z_max, z_min), (i_max, i_min)]
                            break
                if vertices == 'Empty':
                    x_max = x + x_int_mvmt
                    x_min = x - x_int_mvmt
                    if x_min < 0:
                        x_min = noise_range[0] / 2
                    y_max = y + y_int_mvmt
                    y_min = y - y_int_mvmt
                    if y_min < 0:
                        y_min = sigmaF_range[0] / 2
                    z_max = z + z_int_mvmt
                    z_min = z - z_int_mvmt
                    if z_min < 0:
                        z_min = length_range[0] / 2
                    i_max = i + i_int_mvmt
                    i_min = i - i_int_mvmt
                    if i_min < 0:
                        i_min = alpha_range[0] / 2
                    vertices = [(x_max, x_min),
                                (y_max, y_min),
                                (z_max, z_min),
                                (i_max, i_min)]
                    list_of_vertices.append(vertices)

            return list_of_vertices

        else:
            print("Error in Model-Type: ", model_type)

    def intake_top_layer_calculations_and_determine_new_grids(self, storage_df, list_of_verts=()):
        """ Method """
        """
        Once top-layer predictions have been made, this function finds the new high-density regions to examine.
        It aslo takes into account all previous high-density regions, and if a new point falls in an old area, 
        the model zooms in even futher into this area to get even higher density points.
        """

        # list_of_verts: When a top layer prediction is made and the top points are found, the region around these
        #                points is examined in as a 'high-density range' (referring to the high density of points being
        #                examined in this region). Once the area is investigated, the grid vertices are saved in this
        #                list. If a point later should fall in these grids, the given point should be zoomed in on more
        #                as much of the given region has already been investigated.

        model_type = self.model_type
        num_top_points = self.num_HP_zones_AL

        print("intake_top_layer_predictions_and_determine_new_grids --  Started (no end PS)")

        if model_type == 'SVM_Type1':
            # Initialize Ranges
            C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength_AL)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength_AL)

            mvmt_decimal = 0.3
            x_int_mvmt = (C_range[-1] - C_range[1]) * mvmt_decimal / 2
            y_int_mvmt = (epsilon_range[-1] - epsilon_range[1]) * mvmt_decimal / 2


            list_of_vertices = []
            list_of_verts_rev = np.flip(list_of_verts)
            for pt in range(num_top_points):
                print("pt: ", pt)
                vertices = 'Empty'
                x = storage_df.iloc[pt, :]['C']
                y = storage_df.iloc[pt, :]['Epsilon']

                print("(x, y): ("+str(x)+", "+str(y)+")")

                if len(list_of_verts) != 0:
                    for vert in list_of_verts_rev:
                        print("vert: ", vert)
                        x_values = vert[0]
                        y_values = vert[1]
                        if (((x_values[0] >= x) & (x >= x_values[1])) & ((y_values[0] >= y) & (y >= y_values[1]))):
                            x_mvmt = (x_values[0] - x_values[1]) * mvmt_decimal
                            y_mvmt = (y_values[0] - y_values[1]) * mvmt_decimal
                            x_max = x + x_mvmt
                            x_min = x - x_mvmt
                            if x_min < 0:
                                x_min = C_range[0] / 2
                            y_max = y + y_mvmt
                            y_min = y - y_mvmt
                            if y_min < 0:
                                y_min = epsilon_range[0] / 2

                            vertices = [(x_max, x_min), (y_max, y_min)]
                            print("vert_new: ", vertices)
                            list_of_vertices.append(vertices)
                            break
                if vertices == 'Empty':
                    x_max = x + x_int_mvmt
                    x_min = x - x_int_mvmt
                    if x_min < 0:
                        x_min = C_range[0] / 2
                    y_max = y + y_int_mvmt
                    y_min = y - y_int_mvmt
                    if y_min < 0:
                        y_min = epsilon_range[0] / 2

                    vertices = [(x_max, x_min),
                                (y_max, y_min)]
                    list_of_vertices.append(vertices)

            return list_of_vertices

        elif model_type == 'SVM_Type2':
            # Initialize Ranges
            C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength_AL)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength_AL)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], self.gridLength_AL)

            mvmt_decimal = 0.3
            x_int_mvmt = (C_range[-1] - C_range[1]) * mvmt_decimal / 2
            y_int_mvmt = (epsilon_range[-1] - epsilon_range[1]) * mvmt_decimal / 2
            z_int_mvmt = (gamma_range[-1] - gamma_range[1]) * mvmt_decimal / 2


            list_of_vertices = []
            for pt in range(num_top_points):
                print("pt: ", pt)
                vertices = 'Empty'
                x = storage_df.iloc[pt, :]['C']
                y = storage_df.iloc[pt, :]['Epsilon']
                z = storage_df.iloc[pt, :]['Gamma']

                print("(x, y, z): (" + str(x) + ", " + str(y) + ", " + str(z) + ")")

                if len(list_of_verts) != 0:
                    for vert in list_of_verts:
                        print("vert: ", vert)
                        x_values = vert[0]
                        y_values = vert[1]
                        z_values = vert[2]
                        if (((x_values[0] >= x) & (x >= x_values[1])) & ((y_values[0] >= y) & (y >= y_values[1])) & ((z_values[0] >= z) & (z >= z_values[1]))):
                            x_mvmt = (x_values[0] - x_values[1]) * mvmt_decimal
                            y_mvmt = (y_values[0] - y_values[1]) * mvmt_decimal
                            z_mvmt = (z_values[0] - z_values[1]) * mvmt_decimal
                            x_max = x + x_mvmt
                            x_min = x - x_mvmt
                            if x_min < 0:
                                x_min = C_range[0] / 2
                            y_max = y + y_mvmt
                            y_min = y - y_mvmt
                            if y_min < 0:
                                y_min = epsilon_range[0] / 2
                            z_max = z + z_mvmt
                            z_min = z - z_mvmt
                            if z_min < 0:
                                z_min = gamma_range[0] / 2

                            vertices = [(x_max, x_min), (y_max, y_min), (z_max, z_min)]
                            list_of_vertices.append(vertices)
                            break
                if vertices == 'Empty':
                    x_max = x + x_int_mvmt
                    x_min = x - x_int_mvmt
                    if x_min < 0:
                        x_min = C_range[0] / 2
                    y_max = y + y_int_mvmt
                    y_min = y - y_int_mvmt
                    if y_min < 0:
                        y_min = epsilon_range[0] / 2
                    z_max = z + z_int_mvmt
                    z_min = z - z_int_mvmt
                    if z_min < 0:
                        z_min = gamma_range[0] / 2
                    vertices = [(x_max, x_min),
                                (y_max, y_min),
                                (z_max, z_min)]
                    list_of_vertices.append(vertices)

            return list_of_vertices

        elif model_type == 'SVM_Type3':
            # Initialize Ranges
            C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength_AL)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength_AL)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], self.gridLength_AL)
            coef0_range = np.linspace(self.coef0_input[0], self.coef0_input[1], self.gridLength_AL)

            mvmt_decimal = 0.3
            x_int_mvmt = (C_range[-1] - C_range[1]) * mvmt_decimal / 2
            y_int_mvmt = (epsilon_range[-1] - epsilon_range[1]) * mvmt_decimal / 2
            z_int_mvmt = (gamma_range[-1] - gamma_range[1]) * mvmt_decimal / 2
            i_int_mvmt = (coef0_range[-1] - coef0_range[1]) * mvmt_decimal / 2

            list_of_vertices = []
            for pt in range(num_top_points):
                print("pt: ", pt)
                vertices = 'Empty'
                x = storage_df.iloc[pt, :]['C']
                y = storage_df.iloc[pt, :]['Epsilon']
                z = storage_df.iloc[pt, :]['Gamma']
                i = storage_df.iloc[pt, :]['Coef0']

                print("(x, y, z, i): (" + str(x) + ", " + str(y) + ", " + str(z) + ", " + str(i) + ")")

                if len(list_of_verts) != 0:
                    for vert in list_of_verts:
                        print("vert: ", vert)
                        x_values = vert[0]
                        y_values = vert[1]
                        z_values = vert[2]
                        i_values = vert[3]
                        if (((x_values[0] >= x) & (x >= x_values[1])) & ((y_values[0] >= y) & (y >= y_values[1])) & ((z_values[0] >= z) & (z >= z_values[1])) & ((i_values[0] >= i) & (i >= i_values[1]))):
                            x_mvmt = (x_values[0] - x_values[1]) * mvmt_decimal
                            y_mvmt = (y_values[0] - y_values[1]) * mvmt_decimal
                            z_mvmt = (z_values[0] - z_values[1]) * mvmt_decimal
                            i_mvmt = (i_values[0] - i_values[1]) * mvmt_decimal
                            x_max = x + x_mvmt
                            x_min = x - x_mvmt
                            if x_min < 0:
                                x_min = C_range[0] / 2
                            y_max = y + y_mvmt
                            y_min = y - y_mvmt
                            if y_min < 0:
                                y_min = epsilon_range[0] / 2
                            z_max = z + z_mvmt
                            z_min = z - z_mvmt
                            if z_min < 0:
                                z_min = gamma_range[0] / 2
                            i_max = i + i_mvmt
                            i_min = i - i_mvmt
                            if i_min < 0:
                                i_min = coef0_range[0] / 2

                            vertices = [(x_max, x_min), (y_max, y_min), (z_max, z_min), (i_max, i_min)]
                            break
                if vertices == 'Empty':
                    x_max = x + x_int_mvmt
                    x_min = x - x_int_mvmt
                    if x_min < 0:
                        x_min = C_range[0] / 2
                    y_max = y + y_int_mvmt
                    y_min = y - y_int_mvmt
                    if y_min < 0:
                        y_min = epsilon_range[0] / 2
                    z_max = z + z_int_mvmt
                    z_min = z - z_int_mvmt
                    if z_min < 0:
                        z_min = gamma_range[0] / 2
                    i_max = i + i_int_mvmt
                    i_min = i - i_int_mvmt
                    if i_min < 0:
                        i_min = coef0_range[0] / 2
                    vertices = [(x_max, x_min),
                                (y_max, y_min),
                                (z_max, z_min),
                                (i_max, i_min)]
                    list_of_vertices.append(vertices)

            return list_of_vertices

        elif model_type == 'GPR_Type1':
            # Initialize Ranges
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], self.gridLength_AL)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], self.gridLength_AL)
            length_range = np.linspace(self.length_input[0], self.length_input[1], self.gridLength_AL)

            mvmt_decimal = 0.3
            x_int_mvmt = (noise_range[-1] - noise_range[1]) * mvmt_decimal / 2
            y_int_mvmt = (sigmaF_range[-1] - sigmaF_range[1]) * mvmt_decimal / 2
            z_int_mvmt = (length_range[-1] - length_range[1]) * mvmt_decimal / 2

            list_of_vertices = []
            for pt in range(num_top_points):
                print("pt: ", pt)
                vertices = 'Empty'
                x = storage_df.iloc[pt, :]['Noise']
                y = storage_df.iloc[pt, :]['SigmaF']
                z = storage_df.iloc[pt, :]['Length']

                print("(x, y, z): (" + str(x) + ", " + str(y) + ", " + str(z) + ")")

                if len(list_of_verts) != 0:
                    for vert in list_of_verts:
                        print("vert: ", vert)
                        x_values = vert[0]
                        y_values = vert[1]
                        z_values = vert[2]
                        if (((x_values[0] >= x) & (x >= x_values[1])) & ((y_values[0] >= y) & (y >= y_values[1])) & ((z_values[0] >= z) & (z >= z_values[1]))):
                            x_mvmt = (x_values[0] - x_values[1]) * mvmt_decimal
                            y_mvmt = (y_values[0] - y_values[1]) * mvmt_decimal
                            z_mvmt = (z_values[0] - z_values[1]) * mvmt_decimal
                            x_max = x + x_mvmt
                            x_min = x - x_mvmt
                            if x_min < 0:
                                x_min = noise_range[0] / 2
                            y_max = y + y_mvmt
                            y_min = y - y_mvmt
                            if y_min < 0:
                                y_min = sigmaF_range[0] / 2
                            z_max = z + z_mvmt
                            z_min = z - z_mvmt
                            if z_min < 0:
                                z_min = length_range[0] / 2

                            vertices = [(x_max, x_min), (y_max, y_min), (z_max, z_min)]
                            break
                if vertices == 'Empty':
                    x_max = x + x_int_mvmt
                    x_min = x - x_int_mvmt
                    if x_min < 0:
                        x_min = noise_range[0] / 2
                    y_max = y + y_int_mvmt
                    y_min = y - y_int_mvmt
                    if y_min < 0:
                        y_min = sigmaF_range[0] / 2
                    z_max = z + z_int_mvmt
                    z_min = z - z_int_mvmt
                    if z_min < 0:
                        z_min = length_range[0] / 2
                    vertices = [(x_max, x_min),
                                (y_max, y_min),
                                (z_max, z_min),]
                    list_of_vertices.append(vertices)

            return list_of_vertices

        elif model_type == 'GPR_Type2':
            # Initialize Ranges
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], self.gridLength_AL)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], self.gridLength_AL)
            length_range = np.linspace(self.length_input[0], self.length_input[1], self.gridLength_AL)
            alpha_range = np.linspace(self.alpha_input[0], self.alpha_input[1], self.gridLength_AL)

            mvmt_decimal = 0.3
            x_int_mvmt = (noise_range[-1] - noise_range[1]) * mvmt_decimal / 2
            y_int_mvmt = (sigmaF_range[-1] - sigmaF_range[1]) * mvmt_decimal / 2
            z_int_mvmt = (length_range[-1] - length_range[1]) * mvmt_decimal / 2
            i_int_mvmt = (alpha_range[-1] - alpha_range[1]) * mvmt_decimal / 2

            list_of_vertices = []
            for pt in range(num_top_points):
                print("pt: ", pt)
                vertices = 'Empty'
                x = storage_df.iloc[pt, :]['Noise']
                y = storage_df.iloc[pt, :]['SigmaF']
                z = storage_df.iloc[pt, :]['Length']
                i = storage_df.iloc[pt, :]['Alpha']

                print("(x, y, z, i): (" + str(x) + ", " + str(y) + ", " + str(z) + ", " + str(i) + ")")

                if len(list_of_verts) != 0:
                    for vert in list_of_verts:
                        print("vert: ", vert)
                        x_values = vert[0]
                        y_values = vert[1]
                        z_values = vert[2]
                        i_values = vert[3]
                        if (((x_values[0] >= x) & (x >= x_values[1])) & ((y_values[0] >= y) & (y >= y_values[1])) & ((z_values[0] >= z) & (z >= z_values[1])) & ((i_values[0] >= i) & (i >= i_values[1]))):
                            x_mvmt = (x_values[0] - x_values[1]) * mvmt_decimal
                            y_mvmt = (y_values[0] - y_values[1]) * mvmt_decimal
                            z_mvmt = (z_values[0] - z_values[1]) * mvmt_decimal
                            i_mvmt = (i_values[0] - i_values[1]) * mvmt_decimal
                            x_max = x + x_mvmt
                            x_min = x - x_mvmt
                            if x_min < 0:
                                x_min = noise_range[0] / 2
                            y_max = y + y_mvmt
                            y_min = y - y_mvmt
                            if y_min < 0:
                                y_min = sigmaF_range[0] / 2
                            z_max = z + z_mvmt
                            z_min = z - z_mvmt
                            if z_min < 0:
                                z_min = length_range[0] / 2
                            i_max = i + i_mvmt
                            i_min = i - i_mvmt
                            if i_min < 0:
                                i_min = alpha_range[0] / 2

                            vertices = [(x_max, x_min), (y_max, y_min), (z_max, z_min), (i_max, i_min)]
                            break
                if vertices == 'Empty':
                    x_max = x + x_int_mvmt
                    x_min = x - x_int_mvmt
                    if x_min < 0:
                        x_min = noise_range[0] / 2
                    y_max = y + y_int_mvmt
                    y_min = y - y_int_mvmt
                    if y_min < 0:
                        y_min = sigmaF_range[0] / 2
                    z_max = z + z_int_mvmt
                    z_min = z - z_int_mvmt
                    if z_min < 0:
                        z_min = length_range[0] / 2
                    i_max = i + i_int_mvmt
                    i_min = i - i_int_mvmt
                    if i_min < 0:
                        i_min = alpha_range[0] / 2
                    vertices = [(x_max, x_min),
                                (y_max, y_min),
                                (z_max, z_min),
                                (i_max, i_min)]
                    list_of_vertices.append(vertices)

            return list_of_vertices

        else:
            print("Error in Model-Type: ", model_type)


    def high_density_calculations(self, HP_vertex_list, model, model_data, run_number, stor_num_counter):
        """ Method """
        """
        This function combines Parts I, II, and III to run calculations on the high-density regions
        """
        gridLength = self.gridLength_AL
        model_type = self.model_type

        model_inputs_int = model_data[0]
        model_outputs_int = model_data[1]

        print("high_density_calculations -- Started")

        if model_type == 'SVM_Type1':
            # Initialize Ranges
            C_input = HP_vertex_list[0]
            epsilon_input = HP_vertex_list[1]
            C_range = np.linspace(C_input[0], C_input[1], gridLength)
            epsilon_range = np.linspace(epsilon_input[0], epsilon_input[1], gridLength)

            ranges = [C_range, epsilon_range]

        elif model_type == 'SVM_Type2':
            # Initialize Ranges
            C_input = HP_vertex_list[0]
            epsilon_input = HP_vertex_list[1]
            gamma_input = HP_vertex_list[1]
            C_range = np.linspace(C_input[0], C_input[1], gridLength)
            epsilon_range = np.linspace(epsilon_input[0], epsilon_input[1], gridLength)
            gamma_range = np.linspace(gamma_input[0], gamma_input[1], gridLength)

            ranges = [C_range, epsilon_range, gamma_range]

        elif model_type == 'SVM_Type3':
            # Initialize Ranges
            C_input = HP_vertex_list[0]
            epsilon_input = HP_vertex_list[1]
            gamma_input = HP_vertex_list[1]
            coef0_input = HP_vertex_list[2]
            C_range = np.linspace(C_input[0], C_input[1], gridLength)
            epsilon_range = np.linspace(epsilon_input[0], epsilon_input[1], gridLength)
            gamma_range = np.linspace(gamma_input[0], gamma_input[1], gridLength)
            coef0_range = np.linspace(coef0_input[0], coef0_input[1], gridLength)

            ranges = [C_range, epsilon_range, gamma_range, coef0_range]

        elif model_type == 'GPR_Type1':
            # Initialize Ranges
            noise_input = HP_vertex_list[0]
            sigmaF_input = HP_vertex_list[1]
            length_input = HP_vertex_list[1]
            noise_range = np.linspace(noise_input[0], noise_input[1], gridLength)
            sigmaF_range = np.linspace(sigmaF_input[0], sigmaF_input[1], gridLength)
            length_range = np.linspace(length_input[0], length_input[1], gridLength)

            """ PART I """
            ranges = [noise_range, sigmaF_range, length_range]

        elif model_type == 'GPR_Type2':
            # Initialize Ranges
            noise_input = HP_vertex_list[0]
            sigmaF_input = HP_vertex_list[1]
            length_input = HP_vertex_list[1]
            alpha_input = HP_vertex_list[2]
            noise_range = np.linspace(noise_input[0], noise_input[1], gridLength)
            sigmaF_range = np.linspace(sigmaF_input[0], sigmaF_input[1], gridLength)
            length_range = np.linspace(length_input[0], length_input[1], gridLength)
            alpha_range = np.linspace(alpha_input[0], alpha_input[1], gridLength)

            ranges = [noise_range, sigmaF_range, length_range, alpha_range]

        else:
            print("Error in Model-Type: ", model_type)
            return

        # Running All Combinations
        """ PART I """
        model_data_P1 = self.PartI_intital_calculations(ranges, model_data)

        """ PART II """
        pred_min_error_array, pred_error_df_sorted = self.PartII_predictions(ranges, model, model_data_P1,
                                                                             run_number)
        pred_error_df_sorted.to_csv("run_" + str(stor_num_counter) + "_HD_predictions.csv")

        """ PART III """
        storage_df, model_data_final = self.PartIII_final_calculations(pred_error_df_sorted, model_data_P1,
                                                                       stor_num_counter)
        storage_df.to_csv("run_" + str(stor_num_counter) + "_HD_final_calcs.csv")

        return storage_df, model_data_final

    """ PART B: Functions for the Full Grid Search portion of the model """
    def intake_dataframe_and_determine_new_grid_points(self, storage_df):
        """ Method """
        """
        Takes in the final top layer predictions from the active learning and gets the data ready for a full grid search
        
            To determine the new regions, the function first takes in the top layer predictions and does calculations on 
            the top points. Then the top points from that are selected. However, if two points in the top, say, 5 points
        are right next to each other in the hyperparameter space, it indicates that that region is dense in low-error
        HP-combos and should definitely be examined. But, if it were as simple as the top 5 points, if 3 were in the
        same close region, the same region would just be looked at 3 times. To avoid this, points in the same region
        should instead expand the singular region instead of making two separate regions. 
                    (NOTE: A region is defined as -> x +/- (0.1 * full_x_range))
            When two points are in the same region, the new region is defined as a box formed by the outermost points
        of the combined regions. This should only take effect in the first layer, and the mesh size should increase by
        some scalar amount.
            Points that are distinctly separate will be treated as such. Only 10 points should be examined, and the code
        ceases when: 5 regions are found, or 10 points are involved - whichever comes first.
        """

        model_type = self.model_type
        range_dec = self.decimal_point_GS
        gridLength = self.gridLength_GS
        numZooms = self.numZooms_GS

        mesh_increase = 1.5

        if model_type == 'SVM_Type1':
            C_int_range = np.linspace(self.C_input[0], self.C_input[1], gridLength)
            epsilon_int_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1])

            x_mvmt = (C_int_range[-1] - C_int_range[1]) * range_dec
            y_mvmt = (epsilon_int_range[-1] - epsilon_int_range[1]) * range_dec

            i = 0
            pt = 0
            list_of_points = []
            list_of_gL = []
            while (i < numZooms) & (pt < 2*numZooms):
                x = storage_df.iloc[pt, :]['C']
                y = storage_df.iloc[pt, :]['Epsilon']
                pt += 1

                x_max = x + x_mvmt
                x_min = x - x_mvmt
                y_max = y + y_mvmt
                y_min = y - y_mvmt

                if x_min <= 0:
                    x_min = C_int_range[-1] / 2
                if y_min <= 0:
                    y_min = epsilon_int_range[-1] / 2

                x_range = (x_max, x_min)
                y_range = (y_max, y_min)

                if list_of_points == []:
                    list_of_points.append([x_range, y_range])
                    list_of_gL.append(gridLength)
                    i += 1
                else:
                    place = 0
                    add_i = True
                    for old_ranges in list_of_points:
                        x_old_range = old_ranges[0]
                        y_old_range = old_ranges[1]
                        if ((x_old_range[0] >= x) & (x >= x_old_range[1])) & ((y_old_range[0] >= y) & (y >= y_old_range[1])):
                            x_max_range = np.max((x_old_range[0], x_range[0]))
                            x_min_range = np.min((x_old_range[1], x_range[1]))
                            y_max_range = np.max((y_old_range[0], y_range[0]))
                            y_min_range = np.min((y_old_range[1], y_range[1]))

                            x_range = (x_max_range, x_min_range)
                            y_range = (y_max_range, y_min_range)
                            add_i = False
                            break
                        else:
                            place += 1
                    if add_i:
                        list_of_points.append([x_range, y_range])
                        list_of_gL.append(gridLength)
                    else:
                        list_of_points[place] = [x_range, y_range]
                        list_of_gL[place] = mesh_increase * list_of_gL[place]

        elif model_type == 'SVM_Type2':
            C_int_range = np.linspace(self.C_input[0], self.C_input[1], gridLength)
            epsilon_int_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1])
            gamma_int_range = np.linspace(self.gamma_input[0], self.gamma_input[1])

            x_mvmt = (C_int_range[-1] - C_int_range[1]) * range_dec
            y_mvmt = (epsilon_int_range[-1] - epsilon_int_range[1]) * range_dec
            z_mvmt = (gamma_int_range[-1] - gamma_int_range[1]) * range_dec

            i = 0
            pt = 0
            list_of_points = []
            list_of_gL = []
            while (i < numZooms) & (pt < 2*numZooms):
                x = storage_df.iloc[pt, :]['C']
                y = storage_df.iloc[pt, :]['Epsilon']
                z = storage_df.iloc[pt, :]['Gamma']
                pt += 1

                x_max = x + x_mvmt
                x_min = x - x_mvmt
                y_max = y + y_mvmt
                y_min = y - y_mvmt
                z_max = z + z_mvmt
                z_min = z - z_mvmt

                if x_min <= 0:
                    x_min = C_int_range[-1] / 2
                if y_min <= 0:
                    y_min = epsilon_int_range[-1] / 2
                if z_min <= 0:
                    z_min = gamma_int_range[-1] / 2

                x_range = (x_max, x_min)
                y_range = (y_max, y_min)
                z_range = (z_max, z_min)

                if list_of_points == []:
                    list_of_points.append([x_range, y_range, z_range])
                    list_of_gL.append(gridLength)
                    i += 1
                else:
                    place = 0
                    add_i = True
                    for old_ranges in list_of_points:
                        x_old_range = old_ranges[0]
                        y_old_range = old_ranges[1]
                        z_old_range = old_ranges[2]
                        if ((x_old_range[0] >= x) & (x >= x_old_range[1])) & ((y_old_range[0] >= y) & (y >= y_old_range[1])) & ((z_old_range[0] >= z) & (z >= z_old_range[1])):
                            x_max_range = np.max((x_old_range[0], x_range[0]))
                            x_min_range = np.min((x_old_range[1], x_range[1]))
                            y_max_range = np.max((y_old_range[0], y_range[0]))
                            y_min_range = np.min((y_old_range[1], y_range[1]))
                            z_max_range = np.max((z_old_range[0], z_range[0]))
                            z_min_range = np.min((z_old_range[1], z_range[1]))

                            x_range = (x_max_range, x_min_range)
                            y_range = (y_max_range, y_min_range)
                            z_range = (z_max_range, z_min_range)
                            add_i = False
                            break
                        else:
                            place += 1
                    if add_i:
                        list_of_points.append([x_range, y_range, z_range])
                        list_of_gL.append(gridLength)
                    else:
                        list_of_points[place] = [x_range, y_range, z_range]
                        list_of_gL[place] = mesh_increase * list_of_gL[place]

        elif model_type == 'SVM_Type3':
            C_int_range = np.linspace(self.C_input[0], self.C_input[1], gridLength)
            epsilon_int_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1])
            gamma_int_range = np.linspace(self.gamma_input[0], self.gamma_input[1])
            coef0_int_range = np.linspace(self.coef0_input[0], self.coef0_input[1])

            x_mvmt = (C_int_range[-1] - C_int_range[1]) * range_dec
            y_mvmt = (epsilon_int_range[-1] - epsilon_int_range[1]) * range_dec
            z_mvmt = (gamma_int_range[-1] - gamma_int_range[1]) * range_dec
            u_mvmt = (coef0_int_range[-1] - coef0_int_range[1]) * range_dec

            i = 0
            pt = 0
            list_of_points = []
            list_of_gL = []
            while (i < numZooms) & (pt < 2*numZooms):
                x = storage_df.iloc[pt, :]['C']
                y = storage_df.iloc[pt, :]['Epsilon']
                z = storage_df.iloc[pt, :]['Gamma']
                u = storage_df.iloc[pt, :]['Coef0']
                pt += 1

                x_max = x + x_mvmt
                x_min = x - x_mvmt
                y_max = y + y_mvmt
                y_min = y - y_mvmt
                z_max = z + z_mvmt
                z_min = z - z_mvmt
                u_max = u + u_mvmt
                u_min = u - u_mvmt

                if x_min <= 0:
                    x_min = C_int_range[-1] / 2
                if y_min <= 0:
                    y_min = epsilon_int_range[-1] / 2
                if z_min <= 0:
                    z_min = gamma_int_range[-1] / 2
                if u_min <= 0:
                    u_min = coef0_int_range[-1] / 2

                x_range = (x_max, x_min)
                y_range = (y_max, y_min)
                z_range = (z_max, z_min)
                u_range = (u_max, u_min)

                if list_of_points == []:
                    list_of_points.append([x_range, y_range, z_range, u_range])
                    list_of_gL.append(gridLength)
                    i += 1
                else:
                    place = 0
                    add_i = True
                    for old_ranges in list_of_points:
                        x_old_range = old_ranges[0]
                        y_old_range = old_ranges[1]
                        z_old_range = old_ranges[2]
                        u_old_range = old_ranges[3]
                        if ((x_old_range[0] >= x) & (x >= x_old_range[1])) & ((y_old_range[0] >= y) & (y >= y_old_range[1])) & ((z_old_range[0] >= z) & (z >= z_old_range[1])) & ((u_old_range[0] >= u) & (u >= u_old_range[1])):
                            x_max_range = np.max((x_old_range[0], x_range[0]))
                            x_min_range = np.min((x_old_range[1], x_range[1]))
                            y_max_range = np.max((y_old_range[0], y_range[0]))
                            y_min_range = np.min((y_old_range[1], y_range[1]))
                            z_max_range = np.max((z_old_range[0], z_range[0]))
                            z_min_range = np.min((z_old_range[1], z_range[1]))
                            u_max_range = np.max((u_old_range[0], u_range[0]))
                            u_min_range = np.min((u_old_range[1], u_range[1]))

                            x_range = (x_max_range, x_min_range)
                            y_range = (y_max_range, y_min_range)
                            z_range = (z_max_range, z_min_range)
                            u_range = (u_max_range, u_min_range)
                            add_i = False
                            break
                        else:
                            place += 1
                    if add_i:
                        list_of_points.append([x_range, y_range, z_range, u_range])
                        list_of_gL.append(gridLength)
                    else:
                        list_of_points[place] = [x_range, y_range, z_range, u_range]
                        list_of_gL[place] = mesh_increase * list_of_gL[place]

        elif model_type == 'GPR_Type1':
            noise_int_range = np.linspace(self.noise_input[0], self.noise_input[1], gridLength)
            sigmaF_int_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1])
            length_int_range = np.linspace(self.length_input[0], self.length_input[1])

            x_mvmt = (noise_int_range[-1] - noise_int_range[1]) * range_dec
            y_mvmt = (sigmaF_int_range[-1] - sigmaF_int_range[1]) * range_dec
            z_mvmt = (length_int_range[-1] - length_int_range[1]) * range_dec

            i = 0
            pt = 0
            list_of_points = []
            list_of_gL = []
            while (i < numZooms) & (pt < 2*numZooms):
                x = storage_df.iloc[pt, :]['C']
                y = storage_df.iloc[pt, :]['Epsilon']
                z = storage_df.iloc[pt, :]['Gamma']
                pt += 1

                x_max = x + x_mvmt
                x_min = x - x_mvmt
                y_max = y + y_mvmt
                y_min = y - y_mvmt
                z_max = z + z_mvmt
                z_min = z - z_mvmt

                if x_min <= 0:
                    x_min = noise_int_range[-1] / 2
                if y_min <= 0:
                    y_min = sigmaF_int_range[-1] / 2
                if z_min <= 0:
                    z_min = length_int_range[-1] / 2

                x_range = (x_max, x_min)
                y_range = (y_max, y_min)
                z_range = (z_max, z_min)

                if list_of_points == []:
                    list_of_points.append([x_range, y_range, z_range])
                    list_of_gL.append(gridLength)
                    i += 1
                else:
                    place = 0
                    add_i = True
                    for old_ranges in list_of_points:
                        x_old_range = old_ranges[0]
                        y_old_range = old_ranges[1]
                        z_old_range = old_ranges[2]
                        if ((x_old_range[0] >= x) & (x >= x_old_range[1])) & ((y_old_range[0] >= y) & (y >= y_old_range[1])) & ((z_old_range[0] >= z) & (z >= z_old_range[1])):
                            x_max_range = np.max((x_old_range[0], x_range[0]))
                            x_min_range = np.min((x_old_range[1], x_range[1]))
                            y_max_range = np.max((y_old_range[0], y_range[0]))
                            y_min_range = np.min((y_old_range[1], y_range[1]))
                            z_max_range = np.max((z_old_range[0], z_range[0]))
                            z_min_range = np.min((z_old_range[1], z_range[1]))

                            x_range = (x_max_range, x_min_range)
                            y_range = (y_max_range, y_min_range)
                            z_range = (z_max_range, z_min_range)
                            add_i = False
                            break
                        else:
                            place += 1
                    if add_i:
                        list_of_points.append([x_range, y_range, z_range])
                        list_of_gL.append(gridLength)
                    else:
                        list_of_points[place] = [x_range, y_range, z_range]
                        list_of_gL[place] = mesh_increase * list_of_gL[place]

        elif model_type == 'GPR_Type2':
            noise_int_range = np.linspace(self.noise_input[0], self.noise_input[1], gridLength)
            sigmaF_int_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1])
            length_int_range = np.linspace(self.length_input[0], self.length_input[1])
            alpha_int_range = np.linspace(self.alpha_input[0], self.alpha_input[1])

            x_mvmt = (noise_int_range[-1] - noise_int_range[1]) * range_dec
            y_mvmt = (sigmaF_int_range[-1] - sigmaF_int_range[1]) * range_dec
            z_mvmt = (length_int_range[-1] - length_int_range[1]) * range_dec
            u_mvmt = (alpha_int_range[-1] - alpha_int_range[1]) * range_dec

            i = 0
            pt = 0
            list_of_points = []
            list_of_gL = []
            while (i < numZooms) & (pt < 2*numZooms):
                x = storage_df.iloc[pt, :]['C']
                y = storage_df.iloc[pt, :]['Epsilon']
                z = storage_df.iloc[pt, :]['Gamma']
                u = storage_df.iloc[pt, :]['Coef0']
                pt += 1

                x_max = x + x_mvmt
                x_min = x - x_mvmt
                y_max = y + y_mvmt
                y_min = y - y_mvmt
                z_max = z + z_mvmt
                z_min = z - z_mvmt
                u_max = u + u_mvmt
                u_min = u - u_mvmt

                if x_min <= 0:
                    x_min = noise_int_range[-1] / 2
                if y_min <= 0:
                    y_min = sigmaF_int_range[-1] / 2
                if z_min <= 0:
                    z_min = length_int_range[-1] / 2
                if u_min <= 0:
                    u_min = alpha_int_range[-1] / 2

                x_range = (x_max, x_min)
                y_range = (y_max, y_min)
                z_range = (z_max, z_min)
                u_range = (u_max, u_min)

                if list_of_points == []:
                    list_of_points.append([x_range, y_range, z_range, u_range])
                    list_of_gL.append(gridLength)
                    i += 1
                else:
                    place = 0
                    add_i = True
                    for old_ranges in list_of_points:
                        x_old_range = old_ranges[0]
                        y_old_range = old_ranges[1]
                        z_old_range = old_ranges[2]
                        u_old_range = old_ranges[3]
                        if ((x_old_range[0] >= x) & (x >= x_old_range[1])) & ((y_old_range[0] >= y) & (y >= y_old_range[1])) & ((z_old_range[0] >= z) & (z >= z_old_range[1])) & ((u_old_range[0] >= u) & (u >= u_old_range[1])):
                            x_max_range = np.max((x_old_range[0], x_range[0]))
                            x_min_range = np.min((x_old_range[1], x_range[1]))
                            y_max_range = np.max((y_old_range[0], y_range[0]))
                            y_min_range = np.min((y_old_range[1], y_range[1]))
                            z_max_range = np.max((z_old_range[0], z_range[0]))
                            z_min_range = np.min((z_old_range[1], z_range[1]))
                            u_max_range = np.max((u_old_range[0], u_range[0]))
                            u_min_range = np.min((u_old_range[1], u_range[1]))

                            x_range = (x_max_range, x_min_range)
                            y_range = (y_max_range, y_min_range)
                            z_range = (z_max_range, z_min_range)
                            u_range = (u_max_range, u_min_range)
                            add_i = False
                            break
                        else:
                            place += 1
                    if add_i:
                        list_of_points.append([x_range, y_range, z_range, u_range])
                        list_of_gL.append(gridLength)
                    else:
                        list_of_points[place] = [x_range, y_range, z_range, u_range]
                        list_of_gL[place] = mesh_increase * list_of_gL[place]

        else:
            print("Error in Model-Type: ", model_type)
            return

        return list_of_points, list_of_gL

    def run_single_gridSearch(self, ranges, gridLength, figure_name):

        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name
        model_type = self.model_type

        numHPs = len(ranges)

        if model_type == "SVM_Type1":
            HP1_range = np.linspace(ranges[0][0], ranges[0][1], gridLength)
            HP2_range = np.linspace(ranges[1][0], ranges[1][1], gridLength)
            HP3_range = [1]
            HP4_range = [1]

            np_zeros_tuple = (len(HP1_range), len(HP2_range))

            df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Gamma', 'Coef0',
                            'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                            'avgTR to avgTS', 'avgTR to Final Error']

        elif model_type == "SVM_Type2":
            HP1_range = np.linspace(ranges[0][0], ranges[0][1], gridLength)
            HP2_range = np.linspace(ranges[1][0], ranges[1][1], gridLength)
            HP3_range = np.linspace(ranges[2][0], ranges[2][1], gridLength)
            HP4_range = [1]

            np_zeros_tuple = (len(HP1_range), len(HP2_range), len(HP3_range))
            
            df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Gamma', 'Coef0',
                            'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                            'avgTR to avgTS', 'avgTR to Final Error']

        elif model_type == "SVM_Type3":
            HP1_range = np.linspace(ranges[0][0], ranges[0][1], gridLength)
            HP2_range = np.linspace(ranges[1][0], ranges[1][1], gridLength)
            HP3_range = np.linspace(ranges[2][0], ranges[2][1], gridLength)
            HP4_range = np.linspace(ranges[3][0], ranges[3][1], gridLength)

            np_zeros_tuple = (len(HP1_range), len(HP2_range), len(HP3_range), len(HP4_range))
            
            df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Gamma', 'Coef0',
                            'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                            'avgTR to avgTS', 'avgTR to Final Error']

        elif model_type == "GPR_Type1":
            HP1_range = np.linspace(ranges[0][0], ranges[0][1], gridLength)
            HP2_range = np.linspace(ranges[1][0], ranges[1][1], gridLength)
            HP3_range = np.linspace(ranges[2][0], ranges[2][1], gridLength)
            HP4_range = [1]

            np_zeros_tuple = (len(HP1_range), len(HP2_range), len(HP3_range))

            df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'Noise', 'SigmaF', 'Length', 'Alpha',
                            'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                            'avgTR to avgTS', 'avgTR to Final Error']

        elif model_type == "GPR_Type2":
            HP1_range = np.linspace(ranges[0][0], ranges[0][1], gridLength)
            HP2_range = np.linspace(ranges[1][0], ranges[1][1], gridLength)
            HP3_range = np.linspace(ranges[2][0], ranges[2][1], gridLength)
            HP4_range = np.linspace(ranges[3][0], ranges[3][1], gridLength)

            np_zeros_tuple = (len(HP1_range), len(HP2_range), len(HP3_range), len(HP4_range))

            df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'Noise', 'SigmaF', 'Length', 'Alpha',
                            'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                            'avgTR to avgTS', 'avgTR to Final Error']

        else:
            print("Error in Model-Type: ", model_type)
            return

        numHP1 = len(HP1_range)
        numHP2 = len(HP2_range)
        numHP3 = len(HP3_range)
        numHP4 = len(HP4_range)
        Npts = numHP1 * numHP2 * numHP3 * numHP4
        
        # Sets up arrays and dataframe
        storage_df = pd.DataFrame(columns = df_col_names)
        error_array = np.zeros(np_zeros_tuple)
        r2_array = np.zeros(np_zeros_tuple)
        cor_array = np.zeros(np_zeros_tuple)
        tr_error_array = np.zeros(np_zeros_tuple)
        tr_r2_array = np.zeros(np_zeros_tuple)
        tr_cor_array = np.zeros(np_zeros_tuple)
        ratio_array = np.zeros(np_zeros_tuple)

        df_pt = 0
        for HP1_idx in range(numHP1):
            HP1 = HP1_range[HP1_idx]
            for HP2_idx in range(numHP2):
                HP2 = HP2_range[HP2_idx]
                for HP3_idx in range(numHP3):
                    HP3 = HP3_range[HP3_idx]
                    for HP4_idx in range(numHP4):
                        HP4 = HP4_range[HP4_idx]

                        reg = Regression(X_use, Y_use,
                                         C=HP1, epsilon=HP2, gamma=HP3, coef0=HP4,
                                         noise=HP1, sigma_F=HP2, scale_length=HP3, alpha=HP4,
                                         Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                         RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                         giveKFdata=True)
                        results, bestPred, kFold_data = reg.RegressionCVMult()

                        # Extracts data
                        error = float(results['rmse'].loc[str(mdl_name)])
                        avg_tr_error = np.mean(list(kFold_data['tr']['results']['variation_#1']['rmse'][str(mdl_name)]))
                        avg_ts_error = np.mean(list(kFold_data['ts']['results']['variation_#1']['rmse'][str(mdl_name)]))
                        ratio_trAvg_tsAvg = float(avg_ts_error / avg_tr_error)
                        ratio_trAvg_final = float(error / avg_tr_error)

                        rmse = error
                        r2 = float(results['r2'].loc[str(mdl_name)])
                        cor = float(results['cor'].loc[str(mdl_name)])

                        avg_tr_rmse = float(
                            np.mean(list(kFold_data['tr']['results']['variation_#1']['rmse'][str(mdl_name)])))
                        avg_tr_r2 = float(
                            np.mean(list(kFold_data['tr']['results']['variation_#1']['r2'][str(mdl_name)])))
                        avg_tr_cor = float(
                            np.mean(list(kFold_data['tr']['results']['variation_#1']['cor'][str(mdl_name)])))

                        # Puts data into storage array
                        storage_df.loc[df_pt] = [str(figure_name), rmse, r2, cor, HP1, HP2, HP3, HP4,
                                              avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                              ratio_trAvg_tsAvg, ratio_trAvg_final]
                        df_pt += 1

                        if numHPs == 2:
                            idx_tuple = (HP1_idx, HP2_idx)
                        elif numHPs == 3:
                            idx_tuple = (HP1_idx, HP2_idx, HP3_idx)
                        elif numHPs == 4:
                            idx_tuple = (HP1_idx, HP2_idx, HP3_idx, HP4_idx)
                        else:
                            print("ERROR IN numHPs: ", numHPs)
                            return

                        error_array[idx_tuple] = rmse
                        r2_array[idx_tuple] = r2
                        cor_array[idx_tuple] = cor
                        tr_error_array[idx_tuple] = avg_tr_error
                        tr_r2_array[idx_tuple] = avg_tr_r2
                        tr_cor_array[idx_tuple] = avg_tr_cor
                        ratio_array[idx_tuple] = ratio_trAvg_final

                        HP1_name = df_col_names[4]
                        HP2_name = df_col_names[5]
                        HP3_name = df_col_names[6]
                        HP4_name = df_col_names[7]

                        print(figure_name+" -- "
                              +str(HP1_name)+'='+str(HP1)+" | "
                              +str(HP2_name)+'='+str(HP2)+" | "
                              +str(HP3_name)+'='+str(HP3)+" | "
                              +str(HP4_name)+'='+str(HP4)+" | "
                              +"RUN# "+str(df_pt)+" / "+str(Npts)
                              +"\n      |--> RMSE: "+str("%.6f" % rmse)+", R^2: "+str("%.6f" % r2)+", Cor: "+str(cor))


        np_arrays = [error_array,
                     r2_array,
                     cor_array,
                     tr_error_array,
                     tr_r2_array,
                     tr_cor_array,
                     ratio_array]

        return np_arrays, storage_df

    def np_array_to_CSV(self, np_array, row_names, col_names, filename):
        numCols = len(col_names)
        numRows = len(row_names)

        data_zeros = np.zeros((numRows+1, numCols+1))
        df = pd.DataFrame(data=data_zeros)

        df.iloc[0,0] = filename
        df.iloc[0, 1:] = col_names
        df.iloc[1:, 0] = row_names
        df.iloc[1:, 1:] = np_array

        df.to_csv(filename)

    def create_heatmap_from_np_arrays(self, np_arrays, ranges, gridLength, figure_name):
        model_type = self.model_type

        if model_type == "SVM_Type1":
            C_input = ranges[0]
            epsilon_input = ranges[1]

            C_range = np.linspace(C_input[0], C_input[1], gridLength)
            epsilon_range = np.linspace(epsilon_input[0], epsilon_input[1], gridLength)

            # Create names for the axis
            C_names = []
            for i in range(len(C_range)):
                name_temp = "C="+str("%.6f" % C_range[i])
                C_names.append(name_temp)
            epsilon_names = []
            for i in range(len(epsilon_range)):
                name_temp = "C=" + str("%.6f" % epsilon_range[i])
                epsilon_names.append(name_temp)

            # Sets up arrays
            error_array    = np_arrays[0]
            r2_array       = np_arrays[1]
            cor_array      = np_arrays[2]
            tr_error_array = np_arrays[3]
            tr_r2_array    = np_arrays[4]
            tr_cor_array   = np_arrays[5]
            ratio_array    = np_arrays[6]

            # sets up file names
            error_FN    = figure_name + "_Error_array.csv"
            r2_FN       = figure_name + "_R2_array.csv"
            cor_FN      = figure_name + "_Cor_array.csv"
            tr_error_FN = figure_name + "_TR_Error_array.csv"
            tr_r2_FN    = figure_name + "_TR_R2_array.csv"
            tr_cor_FN   = figure_name + "_TR_Cor_array.csv"
            ratio_FN    = figure_name + "_Ratio_array.csv"

            # Creates heatmaps
            self.np_array_to_CSV(error_array, C_names, epsilon_names, error_FN)
            self.np_array_to_CSV(r2_array, C_names, epsilon_names, r2_FN)
            self.np_array_to_CSV(cor_array, C_names, epsilon_names, cor_FN)
            self.np_array_to_CSV(tr_error_array, C_names, epsilon_names, tr_error_FN)
            self.np_array_to_CSV(tr_r2_array, C_names, epsilon_names, tr_r2_FN)
            self.np_array_to_CSV(tr_cor_array, C_names, epsilon_names, tr_cor_FN)
            self.np_array_to_CSV(ratio_array, C_names, epsilon_names, ratio_FN)

        if model_type == "SVM_Type2":
            C_input = ranges[0]
            epsilon_input = ranges[1]
            gamma_input = ranges[2]

            C_range = np.linspace(C_input[0], C_input[1], gridLength)
            epsilon_range = np.linspace(epsilon_input[0], epsilon_input[1], gridLength)
            gamma_range = np.linspace(gamma_input[0], gamma_input[1], gridLength)

            # Create names for the axis
            C_names = []
            for i in range(len(C_range)):
                name_temp = "C=" + str("%.6f" % C_range[i])
                C_names.append(name_temp)
            epsilon_names = []
            for i in range(len(epsilon_range)):
                name_temp = "C=" + str("%.6f" % epsilon_range[i])
                epsilon_names.append(name_temp)

            # Creates names for folder
            gamma_folders = []
            for i in range(len(gamma_range)):
                folder_name_temp = figure_name+"_gamma_"+str("%.6f" % gamma_range[i])
                gamma_folders.append(folder_name_temp)


            for i in range(len(gamma_range)):
                os.mkdir(gamma_folders[i])
                os.chdir(gamma_folders[i])

                # Sets up arrays
                error_array = np_arrays[0][:, :, i]
                r2_array = np_arrays[1][:, :, i]
                cor_array = np_arrays[2][:, :, i]
                tr_error_array = np_arrays[3][:, :, i]
                tr_r2_array = np_arrays[4][:, :, i]
                tr_cor_array = np_arrays[5][:, :, i]
                ratio_array = np_arrays[6][:, :, i]

                # sets up file names
                error_FN = figure_name + "_Error_array.csv"
                r2_FN = figure_name + "_R2_array.csv"
                cor_FN = figure_name + "_Cor_array.csv"
                tr_error_FN = figure_name + "_TR_Error_array.csv"
                tr_r2_FN = figure_name + "_TR_R2_array.csv"
                tr_cor_FN = figure_name + "_TR_Cor_array.csv"
                ratio_FN = figure_name + "_Ratio_array.csv"

                # Creates heatmaps
                self.np_array_to_CSV(error_array, C_names, epsilon_names, error_FN)
                self.np_array_to_CSV(r2_array, C_names, epsilon_names, r2_FN)
                self.np_array_to_CSV(cor_array, C_names, epsilon_names, cor_FN)
                self.np_array_to_CSV(tr_error_array, C_names, epsilon_names, tr_error_FN)
                self.np_array_to_CSV(tr_r2_array, C_names, epsilon_names, tr_r2_FN)
                self.np_array_to_CSV(tr_cor_array, C_names, epsilon_names, tr_cor_FN)
                self.np_array_to_CSV(ratio_array, C_names, epsilon_names, ratio_FN)

                os.chdir('..')

        if model_type == "SVM_Type3":
            C_input = ranges[0]
            epsilon_input = ranges[1]
            gamma_input = ranges[2]
            coef0_input = ranges[3]

            C_range = np.linspace(C_input[0], C_input[1], gridLength)
            epsilon_range = np.linspace(epsilon_input[0], epsilon_input[1], gridLength)
            gamma_range = np.linspace(gamma_input[0], gamma_input[1], gridLength)
            coef0_range = np.linspace(coef0_input[0], coef0_input[1], gridLength)

            # Create names for the axis
            C_names = []
            for i in range(len(C_range)):
                name_temp = "C=" + str("%.6f" % C_range[i])
                C_names.append(name_temp)
            epsilon_names = []
            for i in range(len(epsilon_range)):
                name_temp = "C=" + str("%.6f" % epsilon_range[i])
                epsilon_names.append(name_temp)

            # Creates names for folder
            gamma_folders = []
            for i in range(len(gamma_range)):
                folder_name_temp = figure_name + "_gamma_" + str("%.6f" % gamma_range[i])
                gamma_folders.append(folder_name_temp)
            coef0_folders = []
            for i in range(len(coef0_range)):
                folder_name_temp = figure_name + "_coef0_" + str("%.6f" % coef0_range[i])
                coef0_folders.append(folder_name_temp)

            for j in range(len(coef0_range)):
                os.mkdir(coef0_folders[j])
                os.chdir(coef0_folders[j])

                for i in range(len(gamma_range)):
                    os.mkdir(gamma_folders[i])
                    os.chdir(gamma_folders[i])

                    # Sets up arrays
                    error_array = np_arrays[0][:, :, i, j]
                    r2_array = np_arrays[1][:, :, i, j]
                    cor_array = np_arrays[2][:, :, i, j]
                    tr_error_array = np_arrays[3][:, :, i, j]
                    tr_r2_array = np_arrays[4][:, :, i, j]
                    tr_cor_array = np_arrays[5][:, :, i, j]
                    ratio_array = np_arrays[6][:, :, i, j]

                    # sets up file names
                    error_FN = figure_name + "_Error_array.csv"
                    r2_FN = figure_name + "_R2_array.csv"
                    cor_FN = figure_name + "_Cor_array.csv"
                    tr_error_FN = figure_name + "_TR_Error_array.csv"
                    tr_r2_FN = figure_name + "_TR_R2_array.csv"
                    tr_cor_FN = figure_name + "_TR_Cor_array.csv"
                    ratio_FN = figure_name + "_Ratio_array.csv"

                    # Creates heatmaps
                    self.np_array_to_CSV(error_array, C_names, epsilon_names, error_FN)
                    self.np_array_to_CSV(r2_array, C_names, epsilon_names, r2_FN)
                    self.np_array_to_CSV(cor_array, C_names, epsilon_names, cor_FN)
                    self.np_array_to_CSV(tr_error_array, C_names, epsilon_names, tr_error_FN)
                    self.np_array_to_CSV(tr_r2_array, C_names, epsilon_names, tr_r2_FN)
                    self.np_array_to_CSV(tr_cor_array, C_names, epsilon_names, tr_cor_FN)
                    self.np_array_to_CSV(ratio_array, C_names, epsilon_names, ratio_FN)

                    os.chdir('..')
                os.chdir('..')

        if model_type == "GPR_Type1":
            noise_input = ranges[0]
            sigmaF_input = ranges[1]
            length_input = ranges[2]

            noise_range = np.linspace(noise_input[0], noise_input[1], gridLength)
            sigmaF_range = np.linspace(sigmaF_input[0], sigmaF_input[1], gridLength)
            length_range = np.linspace(length_input[0], length_input[1], gridLength)

            # Create names for the axis
            noise_names = []
            for i in range(len(noise_range)):
                name_temp = "Noise=" + str("%.6f" % noise_range[i])
                noise_names.append(name_temp)
            sigmaF_names = []
            for i in range(len(sigmaF_range)):
                name_temp = "SigmaF=" + str("%.6f" % sigmaF_range[i])
                sigmaF_names.append(name_temp)

            # Creates names for folder
            length_folders = []
            for i in range(len(length_range)):
                folder_name_temp = figure_name + "_length_" + str("%.6f" % length_range[i])
                length_folders.append(folder_name_temp)

            for i in range(len(length_range)):
                os.mkdir(length_folders[i])
                os.chdir(length_folders[i])

                # Sets up arrays
                error_array = np_arrays[0][:, :, i]
                r2_array = np_arrays[1][:, :, i]
                cor_array = np_arrays[2][:, :, i]
                tr_error_array = np_arrays[3][:, :, i]
                tr_r2_array = np_arrays[4][:, :, i]
                tr_cor_array = np_arrays[5][:, :, i]
                ratio_array = np_arrays[6][:, :, i]

                # sets up file names
                error_FN = figure_name + "_Error_array.csv"
                r2_FN = figure_name + "_R2_array.csv"
                cor_FN = figure_name + "_Cor_array.csv"
                tr_error_FN = figure_name + "_TR_Error_array.csv"
                tr_r2_FN = figure_name + "_TR_R2_array.csv"
                tr_cor_FN = figure_name + "_TR_Cor_array.csv"
                ratio_FN = figure_name + "_Ratio_array.csv"

                # Creates heatmaps
                self.np_array_to_CSV(error_array, noise_names, sigmaF_names, error_FN)
                self.np_array_to_CSV(r2_array, noise_names, sigmaF_names, r2_FN)
                self.np_array_to_CSV(cor_array, noise_names, sigmaF_names, cor_FN)
                self.np_array_to_CSV(tr_error_array, noise_names, sigmaF_names, tr_error_FN)
                self.np_array_to_CSV(tr_r2_array, noise_names, sigmaF_names, tr_r2_FN)
                self.np_array_to_CSV(tr_cor_array, noise_names, sigmaF_names, tr_cor_FN)
                self.np_array_to_CSV(ratio_array, noise_names, sigmaF_names, ratio_FN)

                os.chdir('..')

        if model_type == "GPR_Type2":
            noise_input = ranges[0]
            sigmaF_input = ranges[1]
            length_input = ranges[2]
            alpha_input = ranges[3]

            noise_range = np.linspace(noise_input[0], noise_input[1], gridLength)
            sigmaF_range = np.linspace(sigmaF_input[0], sigmaF_input[1], gridLength)
            length_range = np.linspace(length_input[0], length_input[1], gridLength)
            alpha_range = np.linspace(alpha_input[0], alpha_input[1], gridLength)

            # Create names for the axis
            noise_names = []
            for i in range(len(noise_range)):
                name_temp = "Noise=" + str("%.6f" % noise_range[i])
                noise_names.append(name_temp)
            sigmaF_names = []
            for i in range(len(sigmaF_range)):
                name_temp = "SigmaF=" + str("%.6f" % sigmaF_range[i])
                sigmaF_names.append(name_temp)

            # Creates names for folder
            length_folders = []
            for i in range(len(length_range)):
                folder_name_temp = figure_name + "_length_" + str("%.6f" % length_range[i])
                length_folders.append(folder_name_temp)
            alpha_folders = []
            for i in range(len(alpha_range)):
                folder_name_temp = figure_name + "_length_" + str("%.6f" % alpha_range[i])
                alpha_folders.append(folder_name_temp)

            for j in range(len(alpha_range)):
                os.mkdir(alpha_folders[j])
                os.chdir(alpha_folders[j])

                for i in range(len(length_range)):
                    os.mkdir(length_folders[i])
                    os.chdir(length_folders[i])

                    # Sets up arrays
                    error_array = np_arrays[0][:, :, i]
                    r2_array = np_arrays[1][:, :, i]
                    cor_array = np_arrays[2][:, :, i]
                    tr_error_array = np_arrays[3][:, :, i]
                    tr_r2_array = np_arrays[4][:, :, i]
                    tr_cor_array = np_arrays[5][:, :, i]
                    ratio_array = np_arrays[6][:, :, i]

                    # sets up file names
                    error_FN = figure_name + "_Error_array.csv"
                    r2_FN = figure_name + "_R2_array.csv"
                    cor_FN = figure_name + "_Cor_array.csv"
                    tr_error_FN = figure_name + "_TR_Error_array.csv"
                    tr_r2_FN = figure_name + "_TR_R2_array.csv"
                    tr_cor_FN = figure_name + "_TR_Cor_array.csv"
                    ratio_FN = figure_name + "_Ratio_array.csv"

                    # Creates heatmaps
                    self.np_array_to_CSV(error_array, noise_names, sigmaF_names, error_FN)
                    self.np_array_to_CSV(r2_array, noise_names, sigmaF_names, r2_FN)
                    self.np_array_to_CSV(cor_array, noise_names, sigmaF_names, cor_FN)
                    self.np_array_to_CSV(tr_error_array, noise_names, sigmaF_names, tr_error_FN)
                    self.np_array_to_CSV(tr_r2_array, noise_names, sigmaF_names, tr_r2_FN)
                    self.np_array_to_CSV(tr_cor_array, noise_names, sigmaF_names, tr_cor_FN)
                    self.np_array_to_CSV(ratio_array, noise_names, sigmaF_names, ratio_FN)

                    os.chdir('..')
                os.chdir('..')

    #def intake_GS_dataframe(self, storage_df):

    def runActiveLearning(self):

        os.mkdir('Active_Learning')
        os.chdir('Active_Learning')

        """ RUNS THE INITIAL CALCULATIONS """

        full_list_of_verts = []
        model_use, model_data, storage_df_int = self.initial_predictions_and_calculations()
        list_of_vertices_current = self.intake_top_layer_calculations_and_determine_new_grids(storage_df_int, full_list_of_verts)
        df_storage_list = []

        stor_num_counter = 1
        for vert in list_of_vertices_current:
            storage_df_HD, model_data_final = self.high_density_calculations(vert, model_use, model_data, 1, stor_num_counter)
            stor_num_counter += 1
            full_list_of_verts.append(vert)
            df_storage_list.append(storage_df_HD)
        df_storage_unsorted_final = df_storage_list[0]
        for i in range(len(df_storage_list)-1):
            i += 1
            df_storage_unsorted_final = pd.concat([df_storage_unsorted_final, df_storage_list[i]], axis=0)

        for i in range(self.num_runs_AL):
            storage_df_TL, model_data = self.top_layer_predictions_and_calculations(model_use, model_data, i+2, stor_num_counter)
            list_of_vertices_current = self.intake_top_layer_calculations_and_determine_new_grids(storage_df_TL, full_list_of_verts)

            for vert in list_of_vertices_current:
                storage_df_HD, model_data = self.high_density_calculations(vert, model_use, model_data, i+2, stor_num_counter)
                stor_num_counter += 1
                full_list_of_verts.append(vert)
                df_storage_unsorted_final = pd.concat([df_storage_unsorted_final, storage_df_HD], axis=0)

        df_storage_sorted_final = df_storage_unsorted_final.sort_values(by=['RMSE'], ascending=True)
        filename = "final_data.csv"
        df_storage_sorted_final.to_csv(filename)

        storage_df_final_TL, model_data = self.top_layer_predictions_and_calculations(model_use, model_data, 'None', 'final')

        os.chdir('..')

        os.mkdir('Full_Heatmaps')
        os.chdir('Full_Heatmaps')

        figure_names_list = self.figureNumbers(self.numLayers_GS, self.numZooms_GS)
        list_of_int_points, list_of_int_gL = self.intake_dataframe_and_determine_new_grid_points(storage_df_final_TL)

        fig_idx = 0
        list_of_ranges_for_next_layer = list_of_int_points
        list_of_gL_for_next_layer = list_of_int_gL
        # Storage for the dataframes
        df_storage = []
        for layer in range(self.numLayers_GS):
            print("Layer: ", layer)
            # Ranges and gridLengths for the current layer, about to be run
            list_of_ranges_for_current_layer = list_of_ranges_for_next_layer.copy()
            list_of_gL_for_current_layer = list_of_gL_for_next_layer.copy()

            num_zooms_for_current_layer = len(list_of_ranges_for_current_layer)

            # Storage for the ranges and gridLengths for the layer next layer
            list_of_ranges_for_next_layer = []
            list_of_gL_for_next_layer = []

            for zoom in range(num_zooms_for_current_layer):
                print("Zoom: "+str(zoom)+" / "+str(num_zooms_for_current_layer))

                ranges_use = list_of_ranges_for_current_layer[zoom]
                gridLength_use = list_of_gL_for_current_layer[zoom]

                np_arrays, storage_df = self.run_single_gridSearch(ranges_use, gridLength_use, figure_names_list[fig_idx])
                self.create_heatmap_from_np_arrays(np_arrays, ranges_use, gridLength_use, figure_names_list[fig_idx])
                list_of_range_points, list_of_gL = self.intake_dataframe_and_determine_new_grid_points(storage_df)

                for range_pt in list_of_range_points:
                    list_of_ranges_for_next_layer.append(range_pt)
                for gL_pt in list_of_gL:
                    list_of_gL_for_next_layer.append(gL_pt)

                df_storage.append(storage_df)

                fig_idx += 1

        unsorted_df_int = df_storage[0]
        for i in range(len(df_storage[1:])):
            i+=1

            unsorted_df_int = pd.concat([unsorted_df_int, df_storage[i]], axis=0)

        sorted_df = unsorted_df_int.sort_values(by=["RMSE"], ascending=True)
        sorted_df_cutoff = sorted_df.iloc[0:50, :]
        sorted_filename = "0_Sorted_Data_"+self.mdl_name+".csv"
        sorted_df_cutoff.to_csv(sorted_filename)

        print("done")

        return df_storage_sorted_final




