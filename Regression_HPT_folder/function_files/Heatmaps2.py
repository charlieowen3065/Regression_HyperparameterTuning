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

class heatmaps():
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

        if model_type == "SVM_Type1":
            for

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


    def runFullGridSearch_AL(self):


    #def runFullGridSearch_AL(self, C_input_data='None', epsilon_input_data='None', gamma_input_data='None', coef0_input_data='None',
    #                         noise_input_data='None', sigF_input_data='None', length_input_data='None', alpha_input_data='None'):
