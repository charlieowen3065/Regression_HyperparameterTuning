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
    def __init__(self, X_inp, Y_inp,
                 numLayers=3, numZooms=3, Nk=5, N=1, gridLength=10,
                 decimal_points_int=0.25, decimal_points_top=0.25,
                 RemoveNaN=True, goodIDs=None, seed='random', models_use='all',
                 save_csv_files=True,
                 C_input='None', epsilon_input='None', gamma_input='None', coef0_input='None',
                 noise_input='None', sigmaF_input='None', length_input='None', alpha_input='None'):


        self.mf = miscFunctions()
        reg = Regression(X_inp, Y_inp, models_use=models_use)
        model_names, model_list = reg.getModels()

        self.X_inp = X_inp
        self.Y_inp = Y_inp

        self.numLayers = numLayers
        self.numZooms = numZooms
        self.gridLength = gridLength
        self.Nk = Nk
        self.N = N

        self.RemoveNaN_var = RemoveNaN
        self.goodIDs = goodIDs
        self.seed = seed

        self.models_use = models_use
        self.mdl_name = model_names[0]

        self.save_csv_files = save_csv_files

        self.decimal_points_int = decimal_points_int
        self.decimal_points_top = decimal_points_top

        self.C_input = C_input
        self.epsilon_input = epsilon_input
        self.gamma_input = gamma_input
        self.coef0_input = coef0_input

        self.noise_input = noise_input
        self.sigmaF_input = sigmaF_input
        self.length_input = length_input
        self.alpha_input = alpha_input

        self.model_type = self.determine_model_type()

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

    def intake_initial_data(self, C_range, epsilon_range, gamma_range, coef0_range,
                            noise_range, sigF_range, length_range, alpha_range):

        gL = self.gridLength

        # Initializes Boolean Variables --------------------------------------------------------------------------------
        C_var = False   # HP: C
        E_var = False   # HP: Epsilon
        G_var = False   # HP: Gamma
        C0_var = False  # HP: Coef0
        N_var = False   # HP: Noise
        S_var = False   # HP: Sigma_F
        L_var = False   # HP: Scale-Length
        A_var = False   # HP: Alpha

        # Checks each range-value --------------------------------------------------------------------------------------
        # SVM
        if C_range != 'None':
            C_var = True
        if epsilon_range != 'None':
            E_var = True
        if gamma_range != 'None':
            G_var = True
        if coef0_range != 'None':
            C0_var = True
        # GPR
        if noise_range != 'None':
            N_var = True
        if sigF_range != 'None':
            S_var = True
        if length_range != 'None':
            L_var = True
        if alpha_range != 'None':
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

        # Distance Reference Values ------------------------------------------------------------------------------------
        """ OUTPUT """
        """
        [(tupleHP1), (tupleHP1), ... ]
         - tupleHP-N:
            (a, b)
            a: gridLength / (HP[-1] - HP[0])
            b: (gridLength - HP[-1]) / (HP[-1] - HP[0])        
        """
        ItoV_eq_const = []
        numHPs = 0
        if (model_type == "SVM_Type1") | (model_type == "SVM_Type2") | (model_type == "SVM_Type3"):
            a_C = gL / (C_range[-1] - C_range[0])
            b_C = (gL - C_range[0]) / (C_range[-1] - C_range[0])
            ItoV_eq_const.append((a_C, b_C))
            numHPs += 1
            a_e = gL / (epsilon_range[-1] - epsilon_range[0])
            b_e = (gL - epsilon_range[0]) / (epsilon_range[-1] - epsilon_range[0])
            ItoV_eq_const.append((a_e, b_e))
            numHPs += 1
            if (model_type == "SVM_Type2") | (model_type == "SVM_Type3"):
                a_g = gL / (gamma_range[-1] - gamma_range[0])
                b_g = (gL - gamma_range[0]) / (gamma_range[-1] - gamma_range[0])
                ItoV_eq_const.append((a_g, b_g))
                numHPs += 1
                if model_type == "SVM_Type3":
                    a_c0 = gL / (coef0_range[-1] - coef0_range[0])
                    b_c0 = (gL - coef0_range[0]) / (coef0_range[-1] - coef0_range[0])
                    ItoV_eq_const.append((a_c0, b_c0))
                    numHPs += 1

        elif (model_type == "GPR_Type1") | (model_type == "GPR_Type2"):
            a_n = gL / (noise_range[-1] - noise_range[0])
            b_n = (gL - noise_range[0]) / (noise_range[-1] - noise_range[0])
            ItoV_eq_const.append((a_n, b_n))
            numHPs += 1
            a_l = gL / (length_range[-1] - length_range[0])
            b_l = (gL - length_range[0]) / (length_range[-1] - length_range[0])
            ItoV_eq_const.append((a_l, b_l))
            numHPs += 1
            a_s = gL / (sigF_range[-1] - sigF_range[0])
            b_s = (gL - sigF_range[0]) / (sigF_range[-1] - sigF_range[0])
            ItoV_eq_const.append((a_s, b_s))
            numHPs += 1
            if model_type == "GPR_Type2":
                a_a = gL / (alpha_range[-1] - alpha_range[0])
                b_a = (gL - alpha_range[0]) / (alpha_range[-1] - alpha_range[0])
                ItoV_eq_const.append((a_a, b_a))
                numHPs += 1

        return model_type, ItoV_eq_const, numHPs

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

    def initial_calculations(self, model_type, inputs):

        False

    def runSingleGridSearch_AL(self, fig_idx, model_data, ):

        model_data_old_inputs = model_data[0]
        model_data_old_outputs = model_data[1]

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
        gridLength = self.gridLength

        # Detemines model type -----------------------------------------------------------------------------------------
        model_type, ItoV_eq_const, numHPs = self.intake_initial_data(self, C_range, epsilon_range, gamma_range, coef0_range, noise_range, sigF_range, length_range, alpha_range)

        new_input_idx = self.inputs_dist_function(numHPs, decimal_points_int, model_type, ItoV_eq_const, model_data_old_inputs)


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
    def determine_model_type(self):
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
        """
        SVM_Type1: SVM_Linear
        SVM_Type2: SVM_RBF
        SVM_Type3: SVM_Poly2, SVM_Poly3
        GPR_Type1: GPR_RBF, GPR_Matern32, GPR_Matern52
        GPR_Type2: GPR_RationalQuadratic
        """
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
        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name
        decimal_points_int = self.decimal_points_int
        gridLength = self.gridLength
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
            indices = np.linspace(0, self.gridLength - 1, int((Npts * self.decimal_points_int) ** (1 / num_HPs)),
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
            indices = np.linspace(0, self.gridLength - 1, int((Npts * self.decimal_points_int) ** (1 / num_HPs)),
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
            indices = np.linspace(0, self.gridLength - 1, int((Npts * self.decimal_points_int) ** (1 / num_HPs)),
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
            indices = np.linspace(0, self.gridLength - 1, int((Npts * self.decimal_points_int) ** (1 / num_HPs)),
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
            model_data = self.combine_model_data(model_data_int, [model_inputs_new, model_outputs_new])
            return model_data

        else:
            print("Error in Model-Type: ", model_type)

    def PartII_predictions(self, ranges, model_int, model_data, run_number):
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

                    min_error = (error_pred[0] - ((1/run_number) * error_std[0]))

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

                        min_error = (error_pred[0] - ((1 / run_number) * error_std[0]))

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

                            min_error = (error_pred[0] - ((1 / run_number) * error_std[0]))

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

                        min_error = (error_pred[0] - ((1 / run_number) * error_std[0]))

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

                            min_error = (error_pred[0] - ((1 / run_number) * error_std[0]))

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
        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name
        decimal_points_top = self.decimal_points_top
        gridLength = self.gridLength
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
        return storage_df, model_data

    def initial_predictions(self):

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
        gridLength = self.gridLength
        model_type = self.model_type
        print("Model: ", model_type)

        print("initial_predictions --  Started")

        # RUN INITIAL PREDICTIONS --------------------------------------------------------------------------------------
        if model_type == 'SVM_Type1':
            # Initialize Ranges
            num_HPs = 2
            C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength)
            numC = len(C_range)
            numE = len(epsilon_range)
            # Getting index points
            Npts = numC * numE
            indices = np.linspace(0, self.gridLength-1, int((Npts*self.decimal_points_int) ** (1 / num_HPs)), dtype=int)
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
            C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], self.gridLength)
            numC = len(C_range)
            numE = len(epsilon_range)
            numG = len(gamma_range)
            # Getting index points
            Npts = numC * numE * numG
            indices = np.linspace(0, self.gridLength - 1, int((Npts * self.decimal_points_int) ** (1 / num_HPs)),
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
            C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], self.gridLength)
            coef0_range = np.linspace(self.coef0_input[0], self.coef0_input[1], self.gridLength)
            numC = len(C_range)
            numE = len(epsilon_range)
            numG = len(gamma_range)
            numC0 = len(coef0_range)

            # Getting index points
            Npts = numC * numE * numG * numC0
            indices = np.linspace(0, self.gridLength - 1, int((Npts * self.decimal_points_int) ** (1 / num_HPs)),
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
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], self.gridLength)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], self.gridLength)
            length_range = np.linspace(self.length_input[0], self.length_input[1], self.gridLength)
            numN = len(noise_range)
            numS = len(sigmaF_range)
            numL = len(length_range)

            # Getting index points
            Npts = numN * numS * numL
            indices = np.linspace(0, self.gridLength - 1, int((Npts * self.decimal_points_int) ** (1 / num_HPs)),
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
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], self.gridLength)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], self.gridLength)
            length_range = np.linspace(self.length_input[0], self.length_input[1], self.gridLength)
            alpha_range = np.linspace(self.alpha_input[0], self.alpha_input[1], self.gridLength)
            numN = len(noise_range)
            numS = len(sigmaF_range)
            numL = len(length_range)
            numA = len(alpha_range)

            # Getting index points
            Npts = numN * numS * numL * numA
            indices = np.linspace(0, self.gridLength - 1, int((Npts * self.decimal_points_int) ** (1 / num_HPs)),
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

    def top_layer_predictions(self, model, model_data, run_number):

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
        gridLength = self.gridLength
        model_type = self.model_type

        print("top_layer_predictions -- Started")

        if model_type == 'SVM_Type1':
            # Initialize Ranges
            C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength)
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
                    min_error = (error_pred[0] - ((1/run_number) * error_std[0]))
                    min_error_array[C_idx, e_idx] = min_error

        elif model_type == 'SVM_Type2':
            # Initialize Ranges
            C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], self.gridLength)
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
                        min_error = (error_pred[0] - ((1 / run_number) * error_std[0]))
                        min_error_array[C_idx, e_idx, g_idx] = min_error

        elif model_type == 'SVM_Type3':
            # Initialize Ranges
            C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], self.gridLength)
            coef0_range = np.linspace(self.coef0_input[0], self.coef0_input[1], self.gridLength)
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
                            min_error = (error_pred[0] - ((1 / run_number) * error_std[0]))
                            min_error_array[C_idx, e_idx, g_idx, c0_idx] = min_error

        elif model_type == 'GPR_Type1':
            # Initialize Ranges
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], self.gridLength)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], self.gridLength)
            length_range = np.linspace(self.length_input[0], self.length_input[1], self.gridLength)
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
                        min_error = (error_pred[0] - ((1 / run_number) * error_std[0]))
                        min_error_array[n_idx, s_idx, l_idx] = min_error

        elif model_type == 'GPR_Type2':
            # Initialize Ranges
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], self.gridLength)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], self.gridLength)
            length_range = np.linspace(self.length_input[0], self.length_input[1], self.gridLength)
            alpha_range = np.linspace(self.alpha_input[0], self.alpha_input[1], self.gridLength)
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
                            min_error = (error_pred[0] - ((1 / run_number) * error_std[0]))
                            min_error_array[n_idx, s_idx, l_idx, a_idx] = min_error

        else:
            print("Error in Model-Type: ", model_type)
            return

        print("top_layer_predictions -- Complete")

        return min_error_array

    def intake_top_layer_predictions_and_determine_new_grids(self, min_error_array, list_of_verts=()):
        # list_of_verts: When a top layer prediction is made and the top points are found, the region around these
        #                points is examined in as a 'high-density range' (referring to the high density of points being
        #                examined in this region). Once the area is investigated, the grid vertices are saved in this
        #                list. If a point later should fall in these grids, the given point should be zoomed in on more
        #                as much of the given region has already been investigated.

        model_type = self.model_type
        num_top_points = self.numZooms

        print("intake_top_layer_predictions_and_determine_new_grids --  Started (no end PS)")

        if model_type == 'SVM_Type1':
            # Initialize Ranges
            C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength)

            mvmt_decimal = 0.3
            x_int_mvmt = (C_range[-1] - C_range[1]) * mvmt_decimal
            y_int_mvmt = (epsilon_range[-1] - epsilon_range[1]) * mvmt_decimal

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
            C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], self.gridLength)

            mvmt_decimal = 0.3
            x_int_mvmt = (C_range[-1] - C_range[1]) * mvmt_decimal
            y_int_mvmt = (epsilon_range[-1] - epsilon_range[1]) * mvmt_decimal
            z_int_mvmt = (gamma_range[-1] - gamma_range[1]) * mvmt_decimal

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
            C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], self.gridLength)
            coef0_range = np.linspace(self.coef0_input[0], self.coef0_input[1], self.gridLength)

            mvmt_decimal = 0.3
            x_int_mvmt = (C_range[-1] - C_range[1]) * mvmt_decimal
            y_int_mvmt = (epsilon_range[-1] - epsilon_range[1]) * mvmt_decimal
            z_int_mvmt = (gamma_range[-1] - gamma_range[1]) * mvmt_decimal
            i_int_mvmt = (coef0_range[-1] - coef0_range[1]) * mvmt_decimal

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
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], self.gridLength)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], self.gridLength)
            length_range = np.linspace(self.length_input[0], self.length_input[1], self.gridLength)

            mvmt_decimal = 0.3
            x_int_mvmt = (noise_range[-1] - noise_range[1]) * mvmt_decimal
            y_int_mvmt = (sigmaF_range[-1] - sigmaF_range[1]) * mvmt_decimal
            z_int_mvmt = (length_range[-1] - length_range[1]) * mvmt_decimal

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
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], self.gridLength)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], self.gridLength)
            length_range = np.linspace(self.length_input[0], self.length_input[1], self.gridLength)
            alpha_range = np.linspace(self.alpha_input[0], self.alpha_input[1], self.gridLength)

            mvmt_decimal = 0.3
            x_int_mvmt = (noise_range[-1] - noise_range[1]) * mvmt_decimal
            y_int_mvmt = (sigmaF_range[-1] - sigmaF_range[1]) * mvmt_decimal
            z_int_mvmt = (length_range[-1] - length_range[1]) * mvmt_decimal
            i_int_mvmt = (alpha_range[-1] - alpha_range[1]) * mvmt_decimal

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

    def high_density_calculations(self, HP_vertex_list, model, model_data, run_number, stor_num_counter):
        gridLength = self.gridLength
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
            C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], self.gridLength)

            ranges = [C_range, epsilon_range, gamma_range]

        elif model_type == 'SVM_Type3':
            # Initialize Ranges
            C_range = np.linspace(self.C_input[0], self.C_input[1], self.gridLength)
            epsilon_range = np.linspace(self.epsilon_input[0], self.epsilon_input[1], self.gridLength)
            gamma_range = np.linspace(self.gamma_input[0], self.gamma_input[1], self.gridLength)
            coef0_range = np.linspace(self.coef0_input[0], self.coef0_input[1], self.gridLength)

            ranges = [C_range, epsilon_range, gamma_range, coef0_range]

        elif model_type == 'GPR_Type1':
            # Initialize Ranges
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], self.gridLength)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], self.gridLength)
            length_range = np.linspace(self.length_input[0], self.length_input[1], self.gridLength)

            """ PART I """
            ranges = [noise_range, sigmaF_range, length_range]

        elif model_type == 'GPR_Type2':
            # Initialize Ranges
            noise_range = np.linspace(self.noise_input[0], self.noise_input[1], self.gridLength)
            sigmaF_range = np.linspace(self.sigmaF_input[0], self.sigmaF_input[1], self.gridLength)
            length_range = np.linspace(self.length_input[0], self.length_input[1], self.gridLength)
            alpha_range = np.linspace(self.alpha_input[0], self.alpha_input[1], self.gridLength)

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

    def runActiveLearning(self):

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
        gridLength = self.gridLength
        model_type = self.model_type

        os.mkdir('Active_Learning')
        os.chdir('Active_Learning')

        """ RUNS THE INITIAL CALCULATIONS """

        full_list_of_verts = []
        model_use, model_data, pred_min_error_array = self.initial_predictions()
        list_of_vertices_current = self.intake_top_layer_predictions_and_determine_new_grids(pred_min_error_array, full_list_of_verts)
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

        for i in range(3):
            min_error_array = self.top_layer_predictions(model_use, model_data, i+2)
            list_of_vertices_current = self.intake_top_layer_predictions_and_determine_new_grids(min_error_array, full_list_of_verts)

            for vert in list_of_vertices_current:
                storage_df_HD, model_data = self.high_density_calculations(vert, model_use, model_data, i+2, stor_num_counter)
                stor_num_counter += 1
                full_list_of_verts.append(vert)
                df_storage_unsorted_final = pd.concat([df_storage_unsorted_final, storage_df_HD], axis=0)

        df_storage_sorted_final = df_storage_unsorted_final.sort_values(by=['RMSE'], ascending=True)
        filename = "final_data.csv"
        df_storage_sorted_final.to_csv(filename)

        os.chdir('..')

        os.mkdir('Full_Heatmaps')

        print("done")

        return df_storage_sorted_final



