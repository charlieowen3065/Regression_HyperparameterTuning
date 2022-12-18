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
                 save_csv_files=True):


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

    def determineTopHPs_2HP(self, data_matrix, HP1_array, HP2_array, numTopHP_Pairs):
        """ METHOD, INPUTS, AND OUTPUTS """
        """
        Method:
        Run "ii = np.unravel_index(np.argsort(data_matrix.ravel())[-numTopHP_Pairs:], data_matrix.shape)" from StackOverflow,
            https://stackoverflow.com/questions/42098093/get-indices-of-top-n-values-in-2d-numpy-ndarray-or-numpy-matrix, 
            which gives the top "n" index-values. These values will be stored in the form [(i,j)_1, (i,j)_2, ... , (i,j)_n].
            Then, for each tuple in the list, the "i" (row) resprents the position in the C-list, and "j" (column) represents
            the position in the epsilon-list. 
        What is then stored is the point to the left and right of the C and epsilon, and each of these tuples will be stored
            in the same list, and will be the output. 

        Inputs:
        1) data_matrix: Matrix constaining the metric-data currently being looked at
        2) HP1_array: List of the first set HPs
        3) HP2_array: List of the second set HPs
        4) numTopHP_Pair: the number of (C,epsilon) pairs that should be found a recorded

        Outputs:
        1) HP_new_list: This will be a list of tuple values in the form
                        [ ([C_min1, C_max1],[e_min1, e_max1]) , ([C_min2, C_max2],[e_min2, e_max2]) , ... , ([C_minN, C_maxN],[e_minN, e_maxN]) ]
                        and will be used in the form:

                            for tuple in HP_new_list:
                                tuple[0] = C_min_and_max
                                tuple[1] = e_min_and_max
                                C_range = np.linspace(C_min_and_max[0], C_min_and_max[1], 20)
                                e_range = np.linspace(e_min_and_max[0], e_min_and_max[1], 20)
                                for i in range(len(C_range)):
                                    C = C_range[i]
                                    for j in range(len(e_range)):
                                    e = e_range[j]
                                    ... RUN REGRESSION ...

        2) Index_Values: List of index values used
        """
        numHP1_points = len(HP1_array)
        numHP2_points = len(HP2_array)

        ii = np.unravel_index(np.argsort(data_matrix.ravel())[:numTopHP_Pairs], data_matrix.shape)

        Index_Values = []
        for i in range(numTopHP_Pairs):
            HP1_position = ii[0][-i]
            HP2_position = ii[1][-i]

            Index_temp = tuple([HP1_position, HP2_position])
            Index_Values.append(Index_temp)

        HP_new_list = []
        for i in Index_Values:
            HP1_position = i[0]
            HP2_position = i[1]
            # Checks to see if the point zoomed in on is in the middle of the heatmap or on the edge. If the point is on the
            #   edge, this would indicat that our mesh did not go deep enough in that direction. If that's the case, we should
            #   be extending the graph by a large amount in that direction to get a better view, since the current one was
            #   not good enough.

            # HP1
            if HP1_position == 0:
                HP1_min = HP1_array[0]
                HP1_new_range_temp = tuple([HP1_min / 10, HP1_array[HP1_position + 2]])
            elif HP1_position == 1:
                HP1_min = HP1_array[1]
                HP1_new_range_temp = tuple([HP1_min / np.sqrt(10), HP1_array[HP1_position + 2]])
            elif HP1_position == numHP1_points - 1:
                HP1_max = HP1_array[numHP1_points - 1]
                HP1_new_range_temp = tuple([HP1_array[HP1_position - 2], HP1_max * 10])
            elif HP1_position == numHP1_points - 2:
                HP1_max = HP1_array[numHP1_points - 2]
                HP1_new_range_temp = tuple([HP1_array[HP1_position - 2], HP1_max * np.sqrt(10)])
            else:
                HP1_new_range_temp = tuple([HP1_array[HP1_position - 2], HP1_array[HP1_position + 2]])
            # HP2
            if HP2_position == 0:
                HP2_min = HP2_array[0]
                HP2_new_range_temp = tuple([HP2_min / 10, HP2_array[HP2_position + 2]])
            elif HP2_position == 1:
                HP2_min = HP2_array[1]
                HP2_new_range_temp = tuple([HP2_min / np.sqrt(10), HP2_array[HP2_position + 2]])
            elif HP2_position == numHP2_points - 1:
                HP2_max = HP2_array[numHP2_points - 1]
                HP2_new_range_temp = tuple([HP2_array[HP2_position - 2], HP2_max * 10])
            elif HP2_position == numHP2_points - 2:
                HP2_max = HP2_array[numHP2_points - 2]
                HP2_new_range_temp = tuple([HP2_array[HP2_position - 2], HP2_max * np.sqrt(10)])
            else:
                HP2_new_range_temp = tuple([HP2_array[HP2_position - 2], HP2_array[HP2_position + 2]])
            HP_new_list.append([HP1_new_range_temp, HP2_new_range_temp])

        return HP_new_list, Index_Values

    def determineTopHPs_3HP(self, data_matrix, HP1_array, HP2_array, HP3_array, numTopHP_Pairs):
        """ METHOD, INPUTS, AND OUTPUTS """
        """
        Method:
        Run "ii = np.unravel_index(np.argsort(data_matrix.ravel())[-numTopHP_Pairs:], data_matrix.shape)" from StackOverflow,
            https://stackoverflow.com/questions/42098093/get-indices-of-top-n-values-in-2d-numpy-ndarray-or-numpy-matrix, 
            which gives the top "n" index-values. These values will be stored in the form [(i,j)_1, (i,j)_2, ... , (i,j)_n].
            Then, for each tuple in the list, the "i" (row) resprents the position in the C-list, and "j" (column) represents
            the position in the epsilon-list. 
        What is then stored is the point to the left and right of the C and epsilon, and each of these tuples will be stored
            in the same list, and will be the output. 

         Inputs:
        1) data_matrix: Matrix constaining the metric-data currently being looked at
        2) HP1_array: List of the first set HPs
        3) HP2_array: List of the second set HPs
        4) HP3_array: List of the third set HPs
        5) numTopHP_Pair: the number of (C,epsilon) pairs that should be found a recorded


        Outputs:
        1) HP_new_list: This will be a list of tuple values in the form
                        [ ([C_min1, C_max1],[e_min1, e_max1]) , ([C_min2, C_max2],[e_min2, e_max2]) , ... , ([C_minN, C_maxN],[e_minN, e_maxN]) ]
                        and will be used in the form:

                            for tuple in HP_new_list:
                                tuple[0] = C_min_and_max
                                tuple[1] = e_min_and_max
                                C_range = np.linspace(C_min_and_max[0], C_min_and_max[1], 20)
                                e_range = np.linspace(e_min_and_max[0], e_min_and_max[1], 20)
                                for i in range(len(C_range)):
                                    C = C_range[i]
                                    for j in range(len(e_range)):
                                    e = e_range[j]
                                    ... RUN REGRESSION ...

        2) Index_Values: List of index values used
        """
        numHP1_points = len(HP1_array)
        numHP2_points = len(HP2_array)
        numHP3_points = len(HP3_array)

        ii = np.unravel_index(np.argsort(data_matrix.ravel())[:numTopHP_Pairs], data_matrix.shape)

        Index_Values = []
        for i in range(numTopHP_Pairs):
            HP1_position = ii[0][-i]
            HP2_position = ii[1][-i]
            HP3_position = ii[2][-i]

            Index_temp = tuple([HP1_position, HP2_position, HP3_position])
            Index_Values.append(Index_temp)

        HP_new_list = []
        for i in Index_Values:
            HP1_position = i[0]
            HP2_position = i[1]
            HP3_position = i[2]
            # Checks to see if the point zoomed in on is in the middle of the heatmap or on the edge. If the point is on the
            #   edge, this would indicat that our mesh did not go deep enough in that direction. If that's the case, we should
            #   be extending the graph by a large amount in that direction to get a better view, since the current one was
            #   not good enough.

            # HP1
            if HP1_position == 0:
                HP1_min = HP1_array[0]
                HP1_new_range_temp = tuple([HP1_min / 10, HP1_array[HP1_position + 2]])
            elif HP1_position == 1:
                HP1_min = HP1_array[1]
                HP1_new_range_temp = tuple([HP1_min / np.sqrt(10), HP1_array[HP1_position + 2]])
            elif HP1_position == numHP1_points - 1:
                HP1_max = HP1_array[numHP1_points - 1]
                HP1_new_range_temp = tuple([HP1_array[HP1_position - 2], HP1_max * 10])
            elif HP1_position == numHP1_points - 2:
                HP1_max = HP1_array[numHP1_points - 2]
                HP1_new_range_temp = tuple([HP1_array[HP1_position - 2], HP1_max * np.sqrt(10)])
            else:
                HP1_new_range_temp = tuple([HP1_array[HP1_position - 2], HP1_array[HP1_position + 2]])
            # HP2
            if HP2_position == 0:
                HP2_min = HP2_array[0]
                HP2_new_range_temp = tuple([HP2_min / 10, HP2_array[HP2_position + 2]])
            elif HP2_position == 1:
                HP2_min = HP2_array[1]
                HP2_new_range_temp = tuple([HP2_min / np.sqrt(10), HP2_array[HP2_position + 2]])
            elif HP2_position == numHP2_points - 1:
                HP2_max = HP2_array[numHP2_points - 1]
                HP2_new_range_temp = tuple([HP2_array[HP2_position - 2], HP2_max * 10])
            elif HP2_position == numHP2_points - 2:
                HP2_max = HP2_array[numHP2_points - 2]
                HP2_new_range_temp = tuple([HP2_array[HP2_position - 2], HP2_max * np.sqrt(10)])
            else:
                HP2_new_range_temp = tuple([HP2_array[HP2_position - 2], HP2_array[HP2_position + 2]])
            # HP3
            if HP3_position == 0:
                HP3_min = HP3_array[0]
                HP3_new_range_temp = tuple([HP3_min / 10, HP3_array[HP3_position + 2]])
            elif HP3_position == 1:
                HP3_min = HP3_array[1]
                HP3_new_range_temp = tuple([HP3_min / np.sqrt(10), HP3_array[HP3_position + 2]])
            elif HP3_position == numHP3_points - 1:
                HP3_max = HP3_array[numHP3_points - 1]
                HP3_new_range_temp = tuple([HP3_array[HP3_position - 2], HP3_max * 10])
            elif HP3_position == numHP3_points - 2:
                HP3_max = HP3_array[numHP3_points - 2]
                HP3_new_range_temp = tuple([HP3_array[HP3_position - 2], HP3_max * np.sqrt(10)])
            else:
                HP3_new_range_temp = tuple([HP3_array[HP3_position - 2], HP3_array[HP3_position + 2]])
            HP_new_list.append([HP1_new_range_temp, HP2_new_range_temp, HP3_new_range_temp])

        return HP_new_list, Index_Values

    def determineTopHPs_4HP(self, data_matrix, HP1_array, HP2_array, HP3_array, HP4_array, numTopHP_Pairs):
        """ METHOD, INPUTS, AND OUTPUTS """
        """
        Method:
        Run "ii = np.unravel_index(np.argsort(data_matrix.ravel())[-numTopHP_Pairs:], data_matrix.shape)" from StackOverflow,
            https://stackoverflow.com/questions/42098093/get-indices-of-top-n-values-in-2d-numpy-ndarray-or-numpy-matrix, 
            which gives the top "n" index-values. These values will be stored in the form [(i,j)_1, (i,j)_2, ... , (i,j)_n].
            Then, for each tuple in the list, the "i" (row) resprents the position in the C-list, and "j" (column) represents
            the position in the epsilon-list. 
        What is then stored is the point to the left and right of the C and epsilon, and each of these tuples will be stored
            in the same list, and will be the output. 

         Inputs:
        1) data_matrix: Matrix constaining the metric-data currently being looked at
        2) HP1_array: List of the first set HPs
        3) HP2_array: List of the second set HPs
        4) HP3_array: List of the third set HPs
        5) numTopHP_Pair: the number of (C,epsilon) pairs that should be found a recorded


        Outputs:
        1) HP_new_list: This will be a list of tuple values in the form
                        [ ([C_min1, C_max1],[e_min1, e_max1]) , ([C_min2, C_max2],[e_min2, e_max2]) , ... , ([C_minN, C_maxN],[e_minN, e_maxN]) ]
                        and will be used in the form:

                            for tuple in HP_new_list:
                                tuple[0] = C_min_and_max
                                tuple[1] = e_min_and_max
                                C_range = np.linspace(C_min_and_max[0], C_min_and_max[1], 20)
                                e_range = np.linspace(e_min_and_max[0], e_min_and_max[1], 20)
                                for i in range(len(C_range)):
                                    C = C_range[i]
                                    for j in range(len(e_range)):
                                    e = e_range[j]
                                    ... RUN REGRESSION ...

        2) Index_Values: List of index values used
        """
        numHP1_points = len(HP1_array)
        numHP2_points = len(HP2_array)
        numHP3_points = len(HP3_array)
        numHP4_points = len(HP4_array)

        ii = np.unravel_index(np.argsort(data_matrix.ravel())[:numTopHP_Pairs], data_matrix.shape)

        Index_Values = []
        for i in range(numTopHP_Pairs):
            HP1_position = ii[0][-i]
            HP2_position = ii[1][-i]
            HP3_position = ii[2][-i]
            HP4_position = ii[3][-i]

            Index_temp = tuple([HP1_position, HP2_position, HP3_position, HP4_position])
            Index_Values.append(Index_temp)

        HP_new_list = []
        for i in Index_Values:
            HP1_position = i[0]
            HP2_position = i[1]
            HP3_position = i[2]
            HP2_position = i[3]
            # Checks to see if the point zoomed in on is in the middle of the heatmap or on the edge. If the point is on the
            #   edge, this would indicat that our mesh did not go deep enough in that direction. If that's the case, we should
            #   be extending the graph by a large amount in that direction to get a better view, since the current one was
            #   not good enough.

            # HP1
            if HP1_position == 0:
                HP1_min = HP1_array[0]
                HP1_new_range_temp = tuple([HP1_min / 10, HP1_array[HP1_position + 2]])
            elif HP1_position == 1:
                HP1_min = HP1_array[1]
                HP1_new_range_temp = tuple([HP1_min / np.sqrt(10), HP1_array[HP1_position + 2]])
            elif HP1_position == numHP1_points - 1:
                HP1_max = HP1_array[numHP1_points - 1]
                HP1_new_range_temp = tuple([HP1_array[HP1_position - 2], HP1_max * 10])
            elif HP1_position == numHP1_points - 2:
                HP1_max = HP1_array[numHP1_points - 2]
                HP1_new_range_temp = tuple([HP1_array[HP1_position - 2], HP1_max * np.sqrt(10)])
            else:
                HP1_new_range_temp = tuple([HP1_array[HP1_position - 2], HP1_array[HP1_position + 2]])
            # HP2
            if HP2_position == 0:
                HP2_min = HP2_array[0]
                HP2_new_range_temp = tuple([HP2_min / 10, HP2_array[HP2_position + 2]])
            elif HP2_position == 1:
                HP2_min = HP2_array[1]
                HP2_new_range_temp = tuple([HP2_min / np.sqrt(10), HP2_array[HP2_position + 2]])
            elif HP2_position == numHP2_points - 1:
                HP2_max = HP2_array[numHP2_points - 1]
                HP2_new_range_temp = tuple([HP2_array[HP2_position - 2], HP2_max * 10])
            elif HP2_position == numHP2_points - 2:
                HP2_max = HP2_array[numHP2_points - 2]
                HP2_new_range_temp = tuple([HP2_array[HP2_position - 2], HP2_max * np.sqrt(10)])
            else:
                HP2_new_range_temp = tuple([HP2_array[HP2_position - 2], HP2_array[HP2_position + 2]])
            # HP3
            if HP3_position == 0:
                HP3_min = HP3_array[0]
                HP3_new_range_temp = tuple([HP3_min / 10, HP3_array[HP3_position + 2]])
            elif HP3_position == 1:
                HP3_min = HP3_array[1]
                HP3_new_range_temp = tuple([HP3_min / np.sqrt(10), HP3_array[HP3_position + 2]])
            elif HP3_position == numHP3_points - 1:
                HP3_max = HP3_array[numHP3_points - 1]
                HP3_new_range_temp = tuple([HP3_array[HP3_position - 2], HP3_max * 10])
            elif HP3_position == numHP3_points - 2:
                HP3_max = HP3_array[numHP3_points - 2]
                HP3_new_range_temp = tuple([HP3_array[HP3_position - 2], HP3_max * np.sqrt(10)])
            else:
                HP3_new_range_temp = tuple([HP3_array[HP3_position - 2], HP3_array[HP3_position + 2]])
            # HP4
            if HP4_position == 0:
                HP4_min = HP4_array[0]
                HP4_new_range_temp = tuple([HP4_min / 10, HP4_array[HP4_position + 2]])
            elif HP4_position == 1:
                HP4_min = HP4_array[1]
                HP4_new_range_temp = tuple([HP4_min / np.sqrt(10), HP4_array[HP4_position + 2]])
            elif HP4_position == numHP4_points - 1:
                HP4_max = HP4_array[numHP4_points - 1]
                HP4_new_range_temp = tuple([HP4_array[HP4_position - 2], HP4_max * 10])
            elif HP4_position == numHP4_points - 2:
                HP4_max = HP4_array[numHP4_points - 2]
                HP4_new_range_temp = tuple([HP4_array[HP4_position - 2], HP4_max * np.sqrt(10)])
            else:
                HP4_new_range_temp = tuple([HP4_array[HP4_position - 2], HP4_array[HP4_position + 2]])

            HP_new_list.append([HP1_new_range_temp, HP2_new_range_temp, HP3_new_range_temp, HP4_new_range_temp])

        return HP_new_list, Index_Values

    """ SVMs """

    def determineBestGamma(self, gamma_range, C_input, e_input):

        X_use = self.X_inp
        Y_use = self.Y_inp
        goodId = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name

        coef0 = 1

        metric_list = []

        for gamma in gamma_range:
            reg = Regression(X_use, Y_use,
                             C=C_input, gamma=gamma, epsilon=e_input, coef0=coef0,
                             goodIDs=goodId, seed=seed,
                             RemoveNaN=True, StandardizeX=True, models_use=models_use, giveKFdata=False)

            results, bestPred = reg.RegressionCVMult()

            metric_list.append(float(results['rmse'].loc[str(mdl_name)]))

            print("Gamma_current: " + str(gamma) + ", RMSE: " + str( float(results['rmse'].loc[str(mdl_name)]) ) )

        min_metric = min(metric_list)
        mix_index = metric_list.index(min_metric)

        gamma_min = gamma_range[mix_index]

        data_zeros = np.zeros((len(gamma_range), 2))
        gamma_df = pd.DataFrame(data=data_zeros, columns=["gamma", 'rmse'])
        gamma_df["gamma"] = gamma_range
        gamma_df['rmse'] = metric_list
        if self.save_csv_files == True:
            gamma_df.to_csv("gamma_values.csv")

        return gamma_min

    def runSingleGridSearch_SVM(self, C_range, epsilon_range, gamma, fig_idx):
        """ INPUTS, METHOD, AND OUTPUTS """
        """
        Inputs:
        1) C_range: range for the C-Hyperparameter
        2) epsilon_range: range for the epsilon-Hyperparameter
        3) gamma_input: gamma to be used in the regression run
        4) models_use: logical array for regression class which tell the function which model-types to run
        5) mdl_name: name of the chosen model - used in the recall of data
        6) metric: should be 'r2', 'rmse', or 'cor' - determines which metric to use for comparisons
        7) heatmap_title: string that will in the title of the main heatmap (the one that shows just the metric data)
        8) figure_name: string indicating the current subset of heatmap:
            i.e. Fig 1     = original heatmap
                 Fig 1.2   = heatmap of the second best output from the original
                 Fig 1.2.3 = heatmap of the third best output from the second best output from the original.
            The number of digits represents what zoom-in layer the image is from
        9) output_type: 
        Method:
        First, the function runs setup. This starts with creating lists for the row and column names, which will be the C
            and epsilon values, respectivly. Then, the three (3) matrices that will store the data are initalized. 
            These are:
            1) the metric data, which wills store the data one the metrics alone, 
            2) the ratio of average testing metric data to average training metric data (each will be the average from the
                Nk-CV splits),
            3) and the ratio of the final testing metric to the average training metric data (where the final is from the
                combonation of all Nk-splits - and is the value in the first (1) matrix - and the average is again the 
                average of the kFold splits)
            The data is then iterated over both the C and Epsilon ranges and a regression is run using (C, epsilon, gamma).
            The data final metric data and the average values will then be recalled and the ratios made. The data will then
            be stored.

        Outputs:
        1) metric_data: matrix holding all of the metric data from the test
        """

        coef0 = 1
        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name

        # SETS UP FINAL STORAGE DATAFRAME ------------------------------------------------------------------------------
        df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Gamma',
                        'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                        'avgTR to avgTS', 'avgTR to Final Error']
        df_numRows = len(C_range) * len(epsilon_range)
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        # SETS UP STORAGE FOR EACH HEATMAP-CSV -------------------------------------------------------------------------
        row_names = []
        col_names = []
        numRows = len(C_range)
        numCols = len(epsilon_range)

        # Row and column names for the heatmap
        for r in range(numRows):
            row_names.append("C=" + str("%.6f" % C_range[r]))
        for c in range(numCols):
            col_names.append("e=" + str("%.6f" % epsilon_range[c]))

        total_points = numRows * numCols

        # SETS UP STORAGE FOR COLLECTED DATA ---------------------------------------------------------------------------
        error_array = np.zeros((numRows, numCols))
        r2_array = np.zeros((numRows, numCols))
        tr_error_array = np.zeros((numRows, numCols))
        tr_r2_array = np.zeros((numRows, numCols))
        ratio_trAvg_tsAvg_array = np.zeros((numRows, numCols))
        ratio_trAvg_array_final = np.zeros((numRows, numCols))

        # RUNS GRID SEARCH ---------------------------------------------------------------------------------------------
        place_counter = 1
        for C_idx in range(numRows):
            C = C_range[C_idx]
            for e_idx in range(numCols):
                epsilon = epsilon_range[e_idx]

                print("C: " + str(C) + ", e: " + str(epsilon) + ", gamma: " + str(gamma) + " (" + str(
                    place_counter) + "/" + str(total_points) + ")")

                # Runs Regressions using current [C, epsilon, gamma, coef0]
                reg = Regression(X_use, Y_use,
                                 C=C, gamma=gamma, epsilon=epsilon, coef0=coef0,
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

                avg_tr_rmse = float(np.mean(list(kFold_data['tr']['results']['variation_#1']['rmse'][str(mdl_name)])))
                avg_tr_r2 = float(np.mean(list(kFold_data['tr']['results']['variation_#1']['r2'][str(mdl_name)])))
                avg_tr_cor = float(np.mean(list(kFold_data['tr']['results']['variation_#1']['cor'][str(mdl_name)])))

                # Puts data into correct array
                error_array[C_idx, e_idx] = error
                r2_array[C_idx, e_idx] = r2
                tr_error_array[C_idx, e_idx] = avg_tr_error
                tr_r2_array[C_idx, e_idx] = avg_tr_r2
                ratio_trAvg_tsAvg_array[C_idx, e_idx] = ratio_trAvg_tsAvg
                ratio_trAvg_array_final[C_idx, e_idx] = ratio_trAvg_final

                # Puts data into storage array
                storage_df.loc[place_counter - 1] = [fig_idx, rmse, r2, cor, C, epsilon, gamma,
                                                     avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                                     ratio_trAvg_tsAvg, ratio_trAvg_final]
                place_counter += 1

        # SAVES HEATMAPS AS CSV-FILES ----------------------------------------------------------------------------------
        if self.save_csv_files == True:
            # Initialization of dataframes
            data_zeros = np.zeros((numRows + 1, numCols + 1))
            error_data_df = pd.DataFrame(data=data_zeros)
            R2_data_df = pd.DataFrame(data=data_zeros)
            tr_error_data_df = pd.DataFrame(data=data_zeros)
            tr_r2_data_df = pd.DataFrame(data=data_zeros)
            trAvg_tsAvg_data_df = pd.DataFrame(data=data_zeros)
            trAvg_final_data_df = pd.DataFrame(data=data_zeros)

            # File Names
            error_file_name = str(mdl_name) + " - Error Data - " + str(fig_idx) + ".csv"
            r2_file_name = str(mdl_name) + " - R2 Data - " + str(fig_idx) + ".csv"
            tr_error_file_name = str(mdl_name) + " - Training-Error Data - " + str(fig_idx) + ".csv"
            tr_r2_file_name = str(mdl_name) + " - Training-R2 Data - " + str(fig_idx) + ".csv"
            trAvg_tsAvg_file_name = str(mdl_name) + " - avgTR to avgTS - " + str(fig_idx) + ".csv"
            trAvg_final_file_name = str(mdl_name) + " - avgTR to Final - " + str(fig_idx) + ".csv"

            # Error Data
            error_data_df.iloc[0, 1:] = col_names
            error_data_df.iloc[1:, 0] = row_names
            error_data_df.iloc[1:, 1:] = error_array
            error_data_df.to_csv(error_file_name)

            # R^2 Data
            R2_data_df.iloc[0, 1:] = col_names
            R2_data_df.iloc[1:, 0] = row_names
            R2_data_df.iloc[1:, 1:] = r2_array
            R2_data_df.to_csv(r2_file_name)

            # Training-Error Data
            tr_error_data_df.iloc[0, 1:] = col_names
            tr_error_data_df.iloc[1:, 0] = row_names
            tr_error_data_df.iloc[1:, 1:] = tr_error_array
            tr_error_data_df.to_csv(tr_error_file_name)

            # Training-R^2 Data
            tr_r2_data_df.iloc[0, 1:] = col_names
            tr_r2_data_df.iloc[1:, 0] = row_names
            tr_r2_data_df.iloc[1:, 1:] = tr_r2_array
            tr_r2_data_df.to_csv(tr_r2_file_name)

            # Average training to average testing error
            trAvg_tsAvg_data_df.iloc[0, 1:] = col_names
            trAvg_tsAvg_data_df.iloc[1:, 0] = row_names
            trAvg_tsAvg_data_df.iloc[1:, 1:] = ratio_trAvg_tsAvg_array
            trAvg_tsAvg_data_df.to_csv(trAvg_tsAvg_file_name)

            # Average training to final error
            trAvg_final_data_df.iloc[0, 1:] = col_names
            trAvg_final_data_df.iloc[1:, 0] = row_names
            trAvg_final_data_df.iloc[1:, 1:] = ratio_trAvg_array_final
            trAvg_final_data_df.to_csv(trAvg_final_file_name)

        return error_array, storage_df

    def runFullGridSearch_SVM(self, C_input_data, epsilon_input_data):

        HP_data = [[C_input_data, epsilon_input_data]]

        fig_idx_list = self.figureNumbers(self.numLayers, self.numZooms)

        final_storage_df_list = []

        # GETS GAMMA VALUE ---------------------------------------------------------------------------------------------
        goodIds = self.goodIDs
        Y_inp = self.Y_inp
        Y_use = [Y_inp[i] for i in range(len(Y_inp)) if goodIds[i]]
        C_forGamma = iqr(Y_use) / 1.349
        e_forGamma = C_forGamma / 10
        # gamma_range = np.arange(0.01, 10, 0.01)
        gamma_range = np.arange(0.01, 1, 0.01)

        gamma = self.determineBestGamma(gamma_range, C_forGamma, e_forGamma)

        fig_counter = 0
        for layer in range(self.numLayers):
            HP_layer_data = []
            for position_idx in range(len(HP_data)):
                print("Figure: ", fig_idx_list[fig_counter])
                print("Current Layer: ", layer + 1)
                print("Current Position: ", position_idx + 1)

                HP_data_current = HP_data[position_idx]

                # Sets up C and Epsilon ranges
                C_data = HP_data_current[0]
                epsilon_data = HP_data_current[1]
                C_range = np.linspace(C_data[0], C_data[1], self.gridLength)
                epsilon_range = np.linspace(epsilon_data[0], epsilon_data[1], self.gridLength)

                # Runs the heatmap with the current C & Epsilon ranges
                error_matrix, storage_df_temp = self.runSingleGridSearch_SVM(C_range, epsilon_range, gamma,
                                                                             fig_idx_list[fig_counter])

                # Gets the new ranges from the error-matrix
                HP_new_list, Index_Values = self.determineTopHPs_2HP(error_matrix, C_range, epsilon_range,
                                                                     self.numZooms)

                if layer == self.numLayers - 1:
                    final_storage_df_list.append(storage_df_temp)

                for i in HP_new_list:
                    HP_layer_data.append(i)

                fig_counter += 1

            HP_data = HP_layer_data

        # Initalizes the final storage of the data on the final layer
        storage_df_current = pd.DataFrame(columns=['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Gamma',
                                                   'Average Training-RMSE', 'Average Training-R^2',
                                                   'Average Training-Cor',
                                                   'avgTR to avgTS', 'avgTR to Final Error'])
        for i in range(len(final_storage_df_list)):
            storage_df_new = pd.concat([storage_df_current, final_storage_df_list[i]])
            storage_df_current = storage_df_new

        storage_df_unsorted = storage_df_current
        storage_df_sorted = storage_df_unsorted.sort_values(by=["RMSE"], ascending=True)
        if self.save_csv_files == True:
            storage_df_unsorted.to_csv('0-Data_' + str(self.mdl_name) + '_Unsorted.csv')
            storage_df_sorted.to_csv('0-Data_' + str(self.mdl_name) + '_Sorted.csv')

        return storage_df_unsorted, storage_df_sorted

    """ GPRs """

    def runSingleGridSearch_GPR(self, noise_range, sigF_range, length_range, fig_idx):

        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name

        # SETS UP FINAL STORAGE DATAFRAME ------------------------------------------------------------------------------
        df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'Noise', 'Sigma_F', 'Length',
                        'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                        'avgTR to avgTS', 'avgTR to Final Error']
        df_numRows = len(noise_range) * len(sigF_range) * len(length_range)
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        # SETS UP STORAGE FOR EACH HEATMAP-CSV -------------------------------------------------------------------------
        noiseNames = []
        sigFNames = []
        lengthNames = []
        numNoise = len(noise_range)
        numSigF = len(sigF_range)
        numLength = len(length_range)

        # Row and column names for the heatmap
        for i in range(numNoise):
            noiseNames.append("noise=" + str("%.6f" % noise_range[i]))
        for i in range(numSigF):
            sigFNames.append("e=" + str("%.6f" % sigF_range[i]))
        for i in range(numLength):
            lengthNames.append("e=" + str("%.6f" % length_range[i]))

        total_points = numNoise * numSigF * numLength

        # SETS UP STORAGE FOR COLLECTED DATA ---------------------------------------------------------------------------
        error_array = np.zeros((numNoise, numSigF, numLength))
        r2_array = np.zeros((numNoise, numSigF, numLength))
        tr_error_array = np.zeros((numNoise, numSigF, numLength))
        tr_r2_array = np.zeros((numNoise, numSigF, numLength))
        ratio_trAvg_tsAvg_array = np.zeros((numNoise, numSigF, numLength))
        ratio_trAvg_final_array = np.zeros((numNoise, numSigF, numLength))

        place_counter = 1
        for i in range(len(noise_range)):
            for j in range(len(sigF_range)):
                for k in range(len(length_range)):
                    noise = noise_range[i]
                    sigF = sigF_range[j]
                    l = length_range[k]

                    print("noise: " + str(noise) + ", sigF: " + str(sigF) + ", length: " + str(l) + " (" + str(
                        place_counter) + "/" + str(total_points) + ")")

                    reg = Regression(X_use, Y_use,
                                     noise=noise, sigma_F=sigF, scale_length=l,
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

                    # Puts data into correct array
                    error_array[i, j, k] = error
                    r2_array[i, j, k] = r2
                    tr_error_array[i, j, k] = avg_tr_error
                    tr_r2_array[i, j, k] = avg_tr_r2
                    ratio_trAvg_tsAvg_array[i, j, k] = ratio_trAvg_tsAvg
                    ratio_trAvg_final_array[i, j, k] = ratio_trAvg_final

                    # Puts data into storage array
                    storage_df.loc[place_counter - 1] = [fig_idx, rmse, r2, cor, noise, sigF, l,
                                                         avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                                         ratio_trAvg_tsAvg, ratio_trAvg_final]
                    place_counter += 1

        # SAVES DATA TO CSV-FILES --------------------------------------------------------------------------------------
        if self.save_csv_files == True:
            for n in range(numNoise):
                noise_current = noise_range[n]
                folder_name = str(fig_idx) + "-noise_" + str(noise_current)
                os.mkdir(folder_name)
                os.chdir(folder_name)

                error_array_current = error_array[n, :, :]
                r2_array_current = r2_array[n, :, :]
                tr_error_array_current = tr_error_array[n, :, :]
                tr_r2_array_current = tr_r2_array[n, :, :]
                ratio_trAvg_tsAvg_array_current = ratio_trAvg_tsAvg_array[n, :, :]
                ratio_trAvg_array_final_current = ratio_trAvg_final_array[n, :, :]

                data_zeros = np.zeros((numSigF + 1, numLength + 1))
                error_data_df = pd.DataFrame(data=data_zeros)
                R2_data_df = pd.DataFrame(data=data_zeros)
                tr_error_data_df = pd.DataFrame(data=data_zeros)
                tr_r2_data_df = pd.DataFrame(data=data_zeros)
                trAvg_tsAvg_data_df = pd.DataFrame(data=data_zeros)
                trAvg_final_data_df = pd.DataFrame(data=data_zeros)

                # File Names
                error_file_name = str(mdl_name) + " - Error Data - " + str(fig_idx) + ".csv"
                r2_file_name = str(mdl_name) + " - R2 Data - " + str(fig_idx) + ".csv"
                tr_error_file_name = str(mdl_name) + " - Training-Error Data - " + str(fig_idx) + ".csv"
                tr_r2_file_name = str(mdl_name) + " - Training-R2 Data - " + str(fig_idx) + ".csv"
                trAvg_tsAvg_file_name = str(mdl_name) + " - avgTR to avgTS - " + str(fig_idx) + ".csv"
                trAvg_final_file_name = str(mdl_name) + " - avgTR to Final - " + str(fig_idx) + ".csv"

                # Error Data
                error_data_df.iloc[0, 1:] = lengthNames
                error_data_df.iloc[1:, 0] = sigFNames
                error_data_df.iloc[1:, 1:] = error_array_current
                error_data_df.to_csv(error_file_name)

                # R^2 Data
                R2_data_df.iloc[0, 1:] = lengthNames
                R2_data_df.iloc[1:, 0] = sigFNames
                R2_data_df.iloc[1:, 1:] = r2_array_current
                R2_data_df.to_csv(r2_file_name)

                # Training-Error Data
                tr_error_data_df.iloc[0, 1:] = lengthNames
                tr_error_data_df.iloc[1:, 0] = sigFNames
                tr_error_data_df.iloc[1:, 1:] = tr_error_array_current
                tr_error_data_df.to_csv(tr_error_file_name)

                # Training-R^2 Data
                tr_r2_data_df.iloc[0, 1:] = lengthNames
                tr_r2_data_df.iloc[1:, 0] = sigFNames
                tr_r2_data_df.iloc[1:, 1:] = tr_r2_array_current
                tr_r2_data_df.to_csv(tr_r2_file_name)

                # Average training to average testing error
                trAvg_tsAvg_data_df.iloc[0, 1:] = lengthNames
                trAvg_tsAvg_data_df.iloc[1:, 0] = sigFNames
                trAvg_tsAvg_data_df.iloc[1:, 1:] = ratio_trAvg_tsAvg_array_current
                trAvg_tsAvg_data_df.to_csv(trAvg_tsAvg_file_name)

                # Average training to final error
                trAvg_final_data_df.iloc[0, 1:] = lengthNames
                trAvg_final_data_df.iloc[1:, 0] = sigFNames
                trAvg_final_data_df.iloc[1:, 1:] = ratio_trAvg_array_final_current
                trAvg_final_data_df.to_csv(trAvg_final_file_name)

                os.chdir('..')

        return error_array, storage_df

    def runFullGridSearch_GPR(self, noise_input_data, sigF_input_data, length_input_data):

        HP_data = [[noise_input_data, sigF_input_data, length_input_data]]

        fig_idx_list = self.figureNumbers(self.numLayers, self.numZooms)

        final_storage_df_list = []

        fig_counter = 0
        for layer in range(self.numLayers):
            HP_layer_data = []
            for position_idx in range(len(HP_data)):
                print("Figure: ", fig_idx_list[fig_counter])
                print("Current Layer: ", layer + 1)
                print("Current Position: ", position_idx + 1)

                HP_data_current = HP_data[position_idx]

                # Sets up C and Epsilon ranges
                noise_data = HP_data_current[0]
                sigF_data = HP_data_current[1]
                length_data = HP_data_current[2]
                noise_range = np.linspace(noise_data[0], noise_data[1], self.gridLength)
                sigF_range = np.linspace(sigF_data[0], sigF_data[1], self.gridLength)
                length_range = np.linspace(length_data[0], length_data[1], self.gridLength)

                # Runs the heatmap with the current Noise, Sigma_F, & Scale-Length ranges
                error_matrix, storage_df_temp = self.runSingleGridSearch_GPR(noise_range, sigF_range, length_range,
                                                                             fig_idx_list[fig_counter])

                # Gets the new ranges from the error-matrix
                HP_new_list, Index_Values = self.determineTopHPs_3HP(error_matrix, noise_range, sigF_range,
                                                                     length_range, self.numZooms)

                if layer == self.numLayers - 1:
                    final_storage_df_list.append(storage_df_temp)

                for i in HP_new_list:
                    HP_layer_data.append(i)

                fig_counter += 1

            HP_data = HP_layer_data

        # Initalizes the final storage of the data on the final layer
        storage_df_current = pd.DataFrame(columns=['Figure', 'RMSE', 'R^2', 'Cor', 'Noise', 'Sigma_F', 'Length',
                                                   'Average Training-RMSE', 'Average Training-R^2',
                                                   'Average Training-Cor',
                                                   'avgTR to avgTS', 'avgTR to Final Error'])

        for i in range(len(final_storage_df_list)):
            storage_df_new = pd.concat([storage_df_current, final_storage_df_list[i]])
            storage_df_current = storage_df_new

        storage_df_unsorted = storage_df_current
        storage_df_sorted = storage_df_unsorted.sort_values(by=["RMSE"], ascending=True)
        if self.save_csv_files == True:
            storage_df_unsorted.to_csv('0-Data_' + str(self.mdl_name) + '_Unsorted.csv')
            storage_df_sorted.to_csv('0-Data_' + str(self.mdl_name) + '_Sorted.csv')

        return storage_df_unsorted, storage_df_sorted

    """ NEW VERSIONS """

    # Single Grid Search
    def runSingleGridSearch_SVM_2HP(self, C_range, epsilon_range, gamma, fig_idx):
        """ INPUTS, METHOD, AND OUTPUTS """
        """
        Inputs:
        1) C_range: range for the C-Hyperparameter
        2) epsilon_range: range for the epsilon-Hyperparameter
        3) gamma_input: gamma to be used in the regression run
        4) models_use: logical array for regression class which tell the function which model-types to run
        5) mdl_name: name of the chosen model - used in the recall of data
        6) metric: should be 'r2', 'rmse', or 'cor' - determines which metric to use for comparisons
        7) heatmap_title: string that will in the title of the main heatmap (the one that shows just the metric data)
        8) figure_name: string indicating the current subset of heatmap:
            i.e. Fig 1     = original heatmap
                 Fig 1.2   = heatmap of the second best output from the original
                 Fig 1.2.3 = heatmap of the third best output from the second best output from the original.
            The number of digits represents what zoom-in layer the image is from
        9) output_type: 
        Method:
        First, the function runs setup. This starts with creating lists for the row and column names, which will be the C
            and epsilon values, respectivly. Then, the three (3) matrices that will store the data are initalized. 
            These are:
            1) the metric data, which wills store the data one the metrics alone, 
            2) the ratio of average testing metric data to average training metric data (each will be the average from the
                Nk-CV splits),
            3) and the ratio of the final testing metric to the average training metric data (where the final is from the
                combonation of all Nk-splits - and is the value in the first (1) matrix - and the average is again the 
                average of the kFold splits)
            The data is then iterated over both the C and Epsilon ranges and a regression is run using (C, epsilon, gamma).
            The data final metric data and the average values will then be recalled and the ratios made. The data will then
            be stored.

        Outputs:
        1) metric_data: matrix holding all of the metric data from the test
        """

        coef0 = 1
        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name

        # SETS UP FINAL STORAGE DATAFRAME ------------------------------------------------------------------------------
        df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Gamma',
                        'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                        'avgTR to avgTS', 'avgTR to Final Error']
        df_numRows = len(C_range) * len(epsilon_range)
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        # SETS UP STORAGE FOR EACH HEATMAP-CSV -------------------------------------------------------------------------
        row_names = []
        col_names = []
        numRows = len(C_range)
        numCols = len(epsilon_range)

        # Row and column names for the heatmap
        for r in range(numRows):
            row_names.append("C=" + str("%.6f" % C_range[r]))
        for c in range(numCols):
            col_names.append("e=" + str("%.6f" % epsilon_range[c]))

        total_points = numRows * numCols

        # SETS UP STORAGE FOR COLLECTED DATA ---------------------------------------------------------------------------
        error_array = np.zeros((numRows, numCols))
        r2_array = np.zeros((numRows, numCols))
        tr_error_array = np.zeros((numRows, numCols))
        tr_r2_array = np.zeros((numRows, numCols))
        ratio_trAvg_tsAvg_array = np.zeros((numRows, numCols))
        ratio_trAvg_array_final = np.zeros((numRows, numCols))

        # RUNS GRID SEARCH ---------------------------------------------------------------------------------------------
        place_counter = 1
        for C_idx in range(numRows):
            C = C_range[C_idx]
            for e_idx in range(numCols):
                epsilon = epsilon_range[e_idx]

                print("C: " + str(C) + ", e: " + str(epsilon) + ", gamma: " + str(gamma) + " (" + str(
                    place_counter) + "/" + str(total_points) + ")")

                # Runs Regressions using current [C, epsilon, gamma, coef0]
                reg = Regression(X_use, Y_use,
                                 C=C, gamma=gamma, epsilon=epsilon, coef0=coef0,
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

                avg_tr_rmse = float(np.mean(list(kFold_data['tr']['results']['variation_#1']['rmse'][str(mdl_name)])))
                avg_tr_r2 = float(np.mean(list(kFold_data['tr']['results']['variation_#1']['r2'][str(mdl_name)])))
                avg_tr_cor = float(np.mean(list(kFold_data['tr']['results']['variation_#1']['cor'][str(mdl_name)])))

                # Puts data into correct array
                error_array[C_idx, e_idx] = error
                r2_array[C_idx, e_idx] = r2
                tr_error_array[C_idx, e_idx] = avg_tr_error
                tr_r2_array[C_idx, e_idx] = avg_tr_r2
                ratio_trAvg_tsAvg_array[C_idx, e_idx] = ratio_trAvg_tsAvg
                ratio_trAvg_array_final[C_idx, e_idx] = ratio_trAvg_final

                # Puts data into storage array
                storage_df.loc[place_counter - 1] = [fig_idx, rmse, r2, cor, C, epsilon, gamma,
                                                     avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                                     ratio_trAvg_tsAvg, ratio_trAvg_final]
                place_counter += 1

        # SAVES HEATMAPS AS CSV-FILES ----------------------------------------------------------------------------------
        if self.save_csv_files == True:
            # Initialization of dataframes
            data_zeros = np.zeros((numRows + 1, numCols + 1))
            error_data_df = pd.DataFrame(data=data_zeros)
            R2_data_df = pd.DataFrame(data=data_zeros)
            tr_error_data_df = pd.DataFrame(data=data_zeros)
            tr_r2_data_df = pd.DataFrame(data=data_zeros)
            trAvg_tsAvg_data_df = pd.DataFrame(data=data_zeros)
            trAvg_final_data_df = pd.DataFrame(data=data_zeros)

            # File Names
            error_file_name = str(mdl_name) + " - Error Data - " + str(fig_idx) + ".csv"
            r2_file_name = str(mdl_name) + " - R2 Data - " + str(fig_idx) + ".csv"
            tr_error_file_name = str(mdl_name) + " - Training-Error Data - " + str(fig_idx) + ".csv"
            tr_r2_file_name = str(mdl_name) + " - Training-R2 Data - " + str(fig_idx) + ".csv"
            trAvg_tsAvg_file_name = str(mdl_name) + " - avgTR to avgTS - " + str(fig_idx) + ".csv"
            trAvg_final_file_name = str(mdl_name) + " - avgTR to Final - " + str(fig_idx) + ".csv"

            # Error Data
            error_data_df.iloc[0, 1:] = col_names
            error_data_df.iloc[1:, 0] = row_names
            error_data_df.iloc[1:, 1:] = error_array
            error_data_df.to_csv(error_file_name)

            # R^2 Data
            R2_data_df.iloc[0, 1:] = col_names
            R2_data_df.iloc[1:, 0] = row_names
            R2_data_df.iloc[1:, 1:] = r2_array
            R2_data_df.to_csv(r2_file_name)

            # Training-Error Data
            tr_error_data_df.iloc[0, 1:] = col_names
            tr_error_data_df.iloc[1:, 0] = row_names
            tr_error_data_df.iloc[1:, 1:] = tr_error_array
            tr_error_data_df.to_csv(tr_error_file_name)

            # Training-R^2 Data
            tr_r2_data_df.iloc[0, 1:] = col_names
            tr_r2_data_df.iloc[1:, 0] = row_names
            tr_r2_data_df.iloc[1:, 1:] = tr_r2_array
            tr_r2_data_df.to_csv(tr_r2_file_name)

            # Average training to average testing error
            trAvg_tsAvg_data_df.iloc[0, 1:] = col_names
            trAvg_tsAvg_data_df.iloc[1:, 0] = row_names
            trAvg_tsAvg_data_df.iloc[1:, 1:] = ratio_trAvg_tsAvg_array
            trAvg_tsAvg_data_df.to_csv(trAvg_tsAvg_file_name)

            # Average training to final error
            trAvg_final_data_df.iloc[0, 1:] = col_names
            trAvg_final_data_df.iloc[1:, 0] = row_names
            trAvg_final_data_df.iloc[1:, 1:] = ratio_trAvg_array_final
            trAvg_final_data_df.to_csv(trAvg_final_file_name)

        return error_array, storage_df

    def runSingleGridSearch_SVM_3HP(self, C_range, epsilon_range, coef0_range, gamma, fig_idx):
        """ INPUTS, METHOD, AND OUTPUTS """
        """
        Inputs:
        1) C_range: range for the C-Hyperparameter
        2) epsilon_range: range for the epsilon-Hyperparameter
        3) gamma_input: gamma to be used in the regression run
        4) models_use: logical array for regression class which tell the function which model-types to run
        5) mdl_name: name of the chosen model - used in the recall of data
        6) metric: should be 'r2', 'rmse', or 'cor' - determines which metric to use for comparisons
        7) heatmap_title: string that will in the title of the main heatmap (the one that shows just the metric data)
        8) figure_name: string indicating the current subset of heatmap:
            i.e. Fig 1     = original heatmap
                 Fig 1.2   = heatmap of the second best output from the original
                 Fig 1.2.3 = heatmap of the third best output from the second best output from the original.
            The number of digits represents what zoom-in layer the image is from
        9) output_type: 
        Method:
        First, the function runs setup. This starts with creating lists for the row and column names, which will be the C
            and epsilon values, respectivly. Then, the three (3) matrices that will store the data are initalized. 
            These are:
            1) the metric data, which wills store the data one the metrics alone, 
            2) the ratio of average testing metric data to average training metric data (each will be the average from the
                Nk-CV splits),
            3) and the ratio of the final testing metric to the average training metric data (where the final is from the
                combonation of all Nk-splits - and is the value in the first (1) matrix - and the average is again the 
                average of the kFold splits)
            The data is then iterated over both the C and Epsilon ranges and a regression is run using (C, epsilon, gamma).
            The data final metric data and the average values will then be recalled and the ratios made. The data will then
            be stored.

        Outputs:
        1) metric_data: matrix holding all of the metric data from the test
        """

        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name

        # SETS UP FINAL STORAGE DATAFRAME ------------------------------------------------------------------------------
        df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                        'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                        'avgTR to avgTS', 'avgTR to Final Error']
        df_numRows = len(C_range) * len(epsilon_range) * len(coef0_range)
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        # SETS UP STORAGE FOR EACH HEATMAP-CSV -------------------------------------------------------------------------
        C_names = []
        e_names = []
        c0_names = []
        numC = len(C_range)
        numE = len(epsilon_range)
        numC0 = len(coef0_range)

        # Row and column names for the heatmap
        for C_i in range(numC):
            C_names.append("C=" + str("%.6f" % C_range[C_i]))
        for e_i in range(numE):
            e_names.append("e=" + str("%.6f" % epsilon_range[e_i]))
        for c0_i in range(numC0):
            c0_names.append('coef0=' + str("%.6f" % coef0_range[c0_i]))

        total_points = numC * numE * numC0

        # SETS UP STORAGE FOR COLLECTED DATA ---------------------------------------------------------------------------
        error_array = np.zeros((numC, numE, numC0))
        r2_array = np.zeros((numC, numE, numC0))
        tr_error_array = np.zeros((numC, numE, numC0))
        tr_r2_array = np.zeros((numC, numE, numC0))
        ratio_trAvg_tsAvg_array = np.zeros((numC, numE, numC0))
        ratio_trAvg_array_final = np.zeros((numC, numE, numC0))

        # RUNS GRID SEARCH ---------------------------------------------------------------------------------------------
        place_counter = 1
        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                epsilon = epsilon_range[e_idx]
                for c0_idx in range(numC0):
                    coef0 = coef0_range[c0_idx]

                    print("C: " + str(C) + ", e: " + str(epsilon) + "ceof0: " + str(coef0) + ", gamma: " + str(
                        gamma) + " (" + str(place_counter) + "/" + str(total_points) + ")")

                    # Runs Regressions using current [C, epsilon, gamma, coef0]
                    reg = Regression(X_use, Y_use,
                                     C=C, gamma=gamma, epsilon=epsilon, coef0=coef0,
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

                    # Puts data into correct array
                    error_array[C_idx, e_idx] = error
                    r2_array[C_idx, e_idx] = r2
                    tr_error_array[C_idx, e_idx] = avg_tr_error
                    tr_r2_array[C_idx, e_idx] = avg_tr_r2
                    ratio_trAvg_tsAvg_array[C_idx, e_idx] = ratio_trAvg_tsAvg
                    ratio_trAvg_array_final[C_idx, e_idx] = ratio_trAvg_final

                    # Puts data into storage array
                    storage_df.loc[place_counter - 1] = [fig_idx, rmse, r2, cor, C, epsilon, coef0, gamma,
                                                         avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                                         ratio_trAvg_tsAvg, ratio_trAvg_final]
                    place_counter += 1

        # SAVES HEATMAPS AS CSV-FILES ----------------------------------------------------------------------------------
        if self.save_csv_files == True:
            for f in range(numC0):
                coef0_current = coef0_range[f]
                folder_name = str(fig_idx) + "-coef0_" + str(coef0_current)
                os.mkdir(folder_name)
                os.chdir(folder_name)

                error_current = error_array[:, :, f]
                R2_current = r2_array[:, :, f]
                tr_error_current = tr_error_array[:, :, f]
                tr_r2_current = tr_r2_array[:, :, f]
                trAvg_tsAvg_current = ratio_trAvg_tsAvg_array[:, :, f]
                trAvg_final_current = ratio_trAvg_array_final[:, :, f]

                # Initialization of dataframes
                data_zeros = np.zeros((numC + 1, numE + 1))
                error_data_df = pd.DataFrame(data=data_zeros)
                R2_data_df = pd.DataFrame(data=data_zeros)
                tr_error_data_df = pd.DataFrame(data=data_zeros)
                tr_r2_data_df = pd.DataFrame(data=data_zeros)
                trAvg_tsAvg_data_df = pd.DataFrame(data=data_zeros)
                trAvg_final_data_df = pd.DataFrame(data=data_zeros)

                # File Names
                error_file_name = str(mdl_name) + " - Error Data - " + str(fig_idx) + ".csv"
                r2_file_name = str(mdl_name) + " - R2 Data - " + str(fig_idx) + ".csv"
                tr_error_file_name = str(mdl_name) + " - Training-Error Data - " + str(fig_idx) + ".csv"
                tr_r2_file_name = str(mdl_name) + " - Training-R2 Data - " + str(fig_idx) + ".csv"
                trAvg_tsAvg_file_name = str(mdl_name) + " - avgTR to avgTS - " + str(fig_idx) + ".csv"
                trAvg_final_file_name = str(mdl_name) + " - avgTR to Final - " + str(fig_idx) + ".csv"

                # Error Data
                error_data_df.iloc[0, 1:] = C_names
                error_data_df.iloc[1:, 0] = e_names
                error_data_df.iloc[1:, 1:] = error_current
                error_data_df.to_csv(error_file_name)

                # R^2 Data
                R2_data_df.iloc[0, 1:] = C_names
                R2_data_df.iloc[1:, 0] = e_names
                R2_data_df.iloc[1:, 1:] = R2_current
                R2_data_df.to_csv(r2_file_name)

                # Training-Error Data
                tr_error_data_df.iloc[0, 1:] = C_names
                tr_error_data_df.iloc[1:, 0] = e_names
                tr_error_data_df.iloc[1:, 1:] = tr_error_current
                tr_error_data_df.to_csv(tr_error_file_name)

                # Training-R^2 Data
                tr_r2_data_df.iloc[0, 1:] = C_names
                tr_r2_data_df.iloc[1:, 0] = e_names
                tr_r2_data_df.iloc[1:, 1:] = tr_r2_current
                tr_r2_data_df.to_csv(tr_r2_file_name)

                # Average training to average testing error
                trAvg_tsAvg_data_df.iloc[0, 1:] = C_names
                trAvg_tsAvg_data_df.iloc[1:, 0] = e_names
                trAvg_tsAvg_data_df.iloc[1:, 1:] = trAvg_tsAvg_current
                trAvg_tsAvg_data_df.to_csv(trAvg_tsAvg_file_name)

                # Average training to final error
                trAvg_final_data_df.iloc[0, 1:] = C_names
                trAvg_final_data_df.iloc[1:, 0] = e_names
                trAvg_final_data_df.iloc[1:, 1:] = trAvg_final_current
                trAvg_final_data_df.to_csv(trAvg_final_file_name)

        return error_array, storage_df

    def runSingleGridSearch_SVM_2HP_and_gamma(self, C_range, epsilon_range, coef0, gamma_range, fig_idx):
        """ INPUTS, METHOD, AND OUTPUTS """
        """
        Inputs:
        1) C_range: range for the C-Hyperparameter
        2) epsilon_range: range for the epsilon-Hyperparameter
        3) gamma_input: gamma to be used in the regression run
        4) models_use: logical array for regression class which tell the function which model-types to run
        5) mdl_name: name of the chosen model - used in the recall of data
        6) metric: should be 'r2', 'rmse', or 'cor' - determines which metric to use for comparisons
        7) heatmap_title: string that will in the title of the main heatmap (the one that shows just the metric data)
        8) figure_name: string indicating the current subset of heatmap:
            i.e. Fig 1     = original heatmap
                 Fig 1.2   = heatmap of the second best output from the original
                 Fig 1.2.3 = heatmap of the third best output from the second best output from the original.
            The number of digits represents what zoom-in layer the image is from
        9) output_type: 
        Method:
        First, the function runs setup. This starts with creating lists for the row and column names, which will be the C
            and epsilon values, respectivly. Then, the three (3) matrices that will store the data are initalized. 
            These are:
            1) the metric data, which wills store the data one the metrics alone, 
            2) the ratio of average testing metric data to average training metric data (each will be the average from the
                Nk-CV splits),
            3) and the ratio of the final testing metric to the average training metric data (where the final is from the
                combonation of all Nk-splits - and is the value in the first (1) matrix - and the average is again the 
                average of the kFold splits)
            The data is then iterated over both the C and Epsilon ranges and a regression is run using (C, epsilon, gamma).
            The data final metric data and the average values will then be recalled and the ratios made. The data will then
            be stored.

        Outputs:
        1) metric_data: matrix holding all of the metric data from the test
        """

        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name

        # SETS UP FINAL STORAGE DATAFRAME ------------------------------------------------------------------------------
        df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                        'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                        'avgTR to avgTS', 'avgTR to Final Error']
        df_numRows = len(C_range) * len(epsilon_range) * len(gamma_range)
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        # SETS UP STORAGE FOR EACH HEATMAP-CSV -------------------------------------------------------------------------
        C_names = []
        e_names = []
        c0_names = []
        numC = len(C_range)
        numE = len(epsilon_range)
        numG = len(gamma_range)

        # Row and column names for the heatmap
        for C_i in range(numC):
            C_names.append("C=" + str("%.6f" % C_range[C_i]))
        for e_i in range(numE):
            e_names.append("e=" + str("%.6f" % epsilon_range[e_i]))
        for g_i in range(numG):
            c0_names.append('gamma=' + str("%.6f" % gamma_range[g_i]))

        total_points = numC * numE * numG

        # SETS UP STORAGE FOR COLLECTED DATA ---------------------------------------------------------------------------
        error_array = np.zeros((numC, numE, numG))
        r2_array = np.zeros((numC, numE, numG))
        tr_error_array = np.zeros((numC, numE, numG))
        tr_r2_array = np.zeros((numC, numE, numG))
        ratio_trAvg_tsAvg_array = np.zeros((numC, numE, numG))
        ratio_trAvg_array_final = np.zeros((numC, numE, numG))

        # RUNS GRID SEARCH ---------------------------------------------------------------------------------------------
        place_counter = 1
        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                epsilon = epsilon_range[e_idx]
                for g_idx in range(numG):
                    gamma = gamma_range[g_idx]

                    # Runs Regressions using current [C, epsilon, gamma, gamma]
                    reg = Regression(X_use, Y_use,
                                     C=C, gamma=gamma, epsilon=epsilon, coef0=coef0,
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

                    # Puts data into correct array
                    error_array[C_idx, e_idx] = error
                    r2_array[C_idx, e_idx] = r2
                    tr_error_array[C_idx, e_idx] = avg_tr_error
                    tr_r2_array[C_idx, e_idx] = avg_tr_r2
                    ratio_trAvg_tsAvg_array[C_idx, e_idx] = ratio_trAvg_tsAvg
                    ratio_trAvg_array_final[C_idx, e_idx] = ratio_trAvg_final

                    # Puts data into storage array
                    storage_df.loc[place_counter - 1] = [fig_idx, rmse, r2, cor, C, epsilon, coef0, gamma,
                                                         avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                                         ratio_trAvg_tsAvg, ratio_trAvg_final]
                    place_counter += 1

                    print("C: " + str(C) + ", e: " + str(epsilon) + ", ceof0: " + str(coef0) + ", gamma: " + str(
                        gamma) + " (" + str(place_counter) + "/" + str(total_points) + ")" + " || Error: "+str(error)+", R^2: "+str(r2))

        # SAVES HEATMAPS AS CSV-FILES ----------------------------------------------------------------------------------
        if self.save_csv_files == True:
            for f in range(numG):
                gamma_current = gamma_range[f]
                folder_name = str(fig_idx) + "-gamma_" + str(gamma_current)
                os.mkdir(folder_name)
                os.chdir(folder_name)

                error_current = error_array[:, :, f]
                R2_current = r2_array[:, :, f]
                tr_error_current = tr_error_array[:, :, f]
                tr_r2_current = tr_r2_array[:, :, f]
                trAvg_tsAvg_current = ratio_trAvg_tsAvg_array[:, :, f]
                trAvg_final_current = ratio_trAvg_array_final[:, :, f]

                # Initialization of dataframes
                data_zeros = np.zeros((numC + 1, numE + 1))
                error_data_df = pd.DataFrame(data=data_zeros)
                R2_data_df = pd.DataFrame(data=data_zeros)
                tr_error_data_df = pd.DataFrame(data=data_zeros)
                tr_r2_data_df = pd.DataFrame(data=data_zeros)
                trAvg_tsAvg_data_df = pd.DataFrame(data=data_zeros)
                trAvg_final_data_df = pd.DataFrame(data=data_zeros)

                # File Names
                error_file_name = str(mdl_name) + " - Error Data - " + str(fig_idx) + ".csv"
                r2_file_name = str(mdl_name) + " - R2 Data - " + str(fig_idx) + ".csv"
                tr_error_file_name = str(mdl_name) + " - Training-Error Data - " + str(fig_idx) + ".csv"
                tr_r2_file_name = str(mdl_name) + " - Training-R2 Data - " + str(fig_idx) + ".csv"
                trAvg_tsAvg_file_name = str(mdl_name) + " - avgTR to avgTS - " + str(fig_idx) + ".csv"
                trAvg_final_file_name = str(mdl_name) + " - avgTR to Final - " + str(fig_idx) + ".csv"

                # Error Data
                error_data_df.iloc[0, 1:] = C_names
                error_data_df.iloc[1:, 0] = e_names
                error_data_df.iloc[1:, 1:] = error_current
                error_data_df.to_csv(error_file_name)

                # R^2 Data
                R2_data_df.iloc[0, 1:] = C_names
                R2_data_df.iloc[1:, 0] = e_names
                R2_data_df.iloc[1:, 1:] = R2_current
                R2_data_df.to_csv(r2_file_name)

                # Training-Error Data
                tr_error_data_df.iloc[0, 1:] = C_names
                tr_error_data_df.iloc[1:, 0] = e_names
                tr_error_data_df.iloc[1:, 1:] = tr_error_current
                tr_error_data_df.to_csv(tr_error_file_name)

                # Training-R^2 Data
                tr_r2_data_df.iloc[0, 1:] = C_names
                tr_r2_data_df.iloc[1:, 0] = e_names
                tr_r2_data_df.iloc[1:, 1:] = tr_r2_current
                tr_r2_data_df.to_csv(tr_r2_file_name)

                # Average training to average testing error
                trAvg_tsAvg_data_df.iloc[0, 1:] = C_names
                trAvg_tsAvg_data_df.iloc[1:, 0] = e_names
                trAvg_tsAvg_data_df.iloc[1:, 1:] = trAvg_tsAvg_current
                trAvg_tsAvg_data_df.to_csv(trAvg_tsAvg_file_name)

                # Average training to final error
                trAvg_final_data_df.iloc[0, 1:] = C_names
                trAvg_final_data_df.iloc[1:, 0] = e_names
                trAvg_final_data_df.iloc[1:, 1:] = trAvg_final_current
                trAvg_final_data_df.to_csv(trAvg_final_file_name)

        return error_array, storage_df

    # Full Grid Search
    def runFullGridSearch_SVM_2HP(self, C_input_data, epsilon_input_data):

        HP_data = [[C_input_data, epsilon_input_data]]

        fig_idx_list = self.figureNumbers(self.numLayers, self.numZooms)

        final_storage_df_list = []

        # GETS GAMMA VALUE ---------------------------------------------------------------------------------------------
        goodIds = self.goodIDs
        Y_inp = self.Y_inp
        Y_use = [Y_inp[i] for i in range(len(Y_inp)) if goodIds[i]]
        C_forGamma = iqr(Y_use) / 1.349
        e_forGamma = C_forGamma / 10
        # gamma_range = np.arange(0.01, 10, 0.01)
        gamma_range = np.arange(0.01, 1, 0.01)

        gamma = self.determineBestGamma(gamma_range, C_forGamma, e_forGamma)

        fig_counter = 0
        for layer in range(self.numLayers):
            HP_layer_data = []
            for position_idx in range(len(HP_data)):
                print("Figure: ", fig_idx_list[fig_counter])
                print("Current Layer: ", layer + 1)
                print("Current Position: ", position_idx + 1)

                HP_data_current = HP_data[position_idx]

                # Sets up C and Epsilon ranges
                C_data = HP_data_current[0]
                epsilon_data = HP_data_current[1]
                C_range = np.linspace(C_data[0], C_data[1], self.gridLength)
                epsilon_range = np.linspace(epsilon_data[0], epsilon_data[1], self.gridLength)

                # Runs the heatmap with the current C & Epsilon ranges
                error_matrix, storage_df_temp = self.runSingleGridSearch_SVM(C_range, epsilon_range, gamma,
                                                                             fig_idx_list[fig_counter])

                # Gets the new ranges from the error-matrix
                HP_new_list, Index_Values = self.determineTopHPs_2HP(error_matrix, C_range, epsilon_range,
                                                                     self.numZooms)

                if layer == self.numLayers - 1:
                    final_storage_df_list.append(storage_df_temp)

                for i in HP_new_list:
                    HP_layer_data.append(i)

                fig_counter += 1

            HP_data = HP_layer_data

        # Initalizes the final storage of the data on the final layer
        storage_df_current = pd.DataFrame(columns=['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Gamma',
                                                   'Average Training-RMSE', 'Average Training-R^2',
                                                   'Average Training-Cor',
                                                   'avgTR to avgTS', 'avgTR to Final Error'])
        for i in range(len(final_storage_df_list)):
            storage_df_new = pd.concat([storage_df_current, final_storage_df_list[i]])
            storage_df_current = storage_df_new

        storage_df_unsorted = storage_df_current
        storage_df_sorted = storage_df_unsorted.sort_values(by=["RMSE"], ascending=True)
        if self.save_csv_files == True:
            storage_df_unsorted.to_csv('0-Data_' + str(self.mdl_name) + '_Unsorted.csv')
            storage_df_sorted.to_csv('0-Data_' + str(self.mdl_name) + '_Sorted.csv')

        return storage_df_unsorted, storage_df_sorted

    def runFullGridSearch_SVM_3HP(self, C_input_data, epsilon_input_data, coef0_input_data):

        HP_data = [[C_input_data, epsilon_input_data, coef0_input_data]]

        fig_idx_list = self.figureNumbers(self.numLayers, self.numZooms)

        final_storage_df_list = []

        # GETS GAMMA VALUE ---------------------------------------------------------------------------------------------
        goodIds = self.goodIDs
        Y_inp = self.Y_inp
        Y_use = [Y_inp[i] for i in range(len(Y_inp)) if goodIds[i]]
        C_forGamma = iqr(Y_use) / 1.349
        e_forGamma = C_forGamma / 10
        # gamma_range = np.arange(0.01, 10, 0.01)
        gamma_range = np.arange(0.01, 1, 0.01)

        gamma = self.determineBestGamma(gamma_range, C_forGamma, e_forGamma)

        fig_counter = 0
        for layer in range(self.numLayers):
            HP_layer_data = []
            for position_idx in range(len(HP_data)):
                print("Figure: ", fig_idx_list[fig_counter])
                print("Current Layer: ", layer + 1)
                print("Current Position: ", position_idx + 1)

                HP_data_current = HP_data[position_idx]

                # Sets up C and Epsilon ranges
                C_data = HP_data_current[0]
                epsilon_data = HP_data_current[1]
                coef0_data = HP_data_current[2]
                C_range = np.linspace(C_data[0], C_data[1], self.gridLength)
                epsilon_range = np.linspace(epsilon_data[0], epsilon_data[1], self.gridLength)
                coef0_range = np.linspace(coef0_data[0], coef0_data[1], self.gridLength)

                # Runs the heatmap with the current C & Epsilon ranges
                error_matrix, storage_df_temp = self.runSingleGridSearch_SVM(C_range, epsilon_range, coef0_range, gamma,
                                                                             fig_idx_list[fig_counter])

                # Gets the new ranges from the error-matrix
                HP_new_list, Index_Values = self.determineTopHPs_3HP(error_matrix, C_range, epsilon_range, coef0_range,
                                                                     self.numZooms)

                if layer == self.numLayers - 1:
                    final_storage_df_list.append(storage_df_temp)

                for i in HP_new_list:
                    HP_layer_data.append(i)

                fig_counter += 1

            HP_data = HP_layer_data

        # Initalizes the final storage of the data on the final layer
        storage_df_current = pd.DataFrame(columns=['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                                                   'Average Training-RMSE', 'Average Training-R^2',
                                                   'Average Training-Cor',
                                                   'avgTR to avgTS', 'avgTR to Final Error'])
        for i in range(len(final_storage_df_list)):
            storage_df_new = pd.concat([storage_df_current, final_storage_df_list[i]])
            storage_df_current = storage_df_new

        storage_df_unsorted = storage_df_current
        storage_df_sorted = storage_df_unsorted.sort_values(by=["RMSE"], ascending=True)
        if self.save_csv_files == True:
            storage_df_unsorted.to_csv('0-Data_' + str(self.mdl_name) + '_Unsorted.csv')
            storage_df_sorted.to_csv('0-Data_' + str(self.mdl_name) + '_Sorted.csv')

        return storage_df_unsorted, storage_df_sorted

    def runFullGridSearch_SVM_2HP_and_gamma(self, C_input_data, epsilon_input_data, gamma_input_data):

        HP_data = [[C_input_data, epsilon_input_data, gamma_input_data]]

        fig_idx_list = self.figureNumbers(self.numLayers, self.numZooms)

        final_storage_df_list = []

        # GETS GAMMA VALUE ---------------------------------------------------------------------------------------------
        goodIds = self.goodIDs
        Y_inp = self.Y_inp
        Y_use = [Y_inp[i] for i in range(len(Y_inp)) if goodIds[i]]

        coef0 = 1
        fig_counter = 0
        for layer in range(self.numLayers):
            HP_layer_data = []
            for position_idx in range(len(HP_data)):
                print("Figure: ", fig_idx_list[fig_counter])
                print("Current Layer: ", layer + 1)
                print("Current Position: ", position_idx + 1)

                HP_data_current = HP_data[position_idx]

                # Sets up C and Epsilon ranges
                C_data = HP_data_current[0]
                epsilon_data = HP_data_current[1]
                gamma_data = HP_data_current[2]
                C_range = np.linspace(C_data[0], C_data[1], self.gridLength)
                epsilon_range = np.linspace(epsilon_data[0], epsilon_data[1], self.gridLength)
                gamma_range = np.linspace(gamma_data[0], gamma_data[1], self.gridLength)

                # Runs the heatmap with the current C & Epsilon ranges
                error_matrix, storage_df_temp = self.runSingleGridSearch_SVM_2HP_and_gamma(C_range, epsilon_range, coef0, gamma_range,
                                                                             fig_idx_list[fig_counter])

                # Gets the new ranges from the error-matrix
                HP_new_list, Index_Values = self.determineTopHPs_3HP(error_matrix, C_range, epsilon_range, gamma_range,
                                                                     self.numZooms)

                if layer == self.numLayers - 1:
                    final_storage_df_list.append(storage_df_temp)

                for i in HP_new_list:
                    HP_layer_data.append(i)

                fig_counter += 1

            HP_data = HP_layer_data

        # Initalizes the final storage of the data on the final layer
        storage_df_current = pd.DataFrame(columns=['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                                                   'Average Training-RMSE', 'Average Training-R^2',
                                                   'Average Training-Cor',
                                                   'avgTR to avgTS', 'avgTR to Final Error'])
        for i in range(len(final_storage_df_list)):
            storage_df_new = pd.concat([storage_df_current, final_storage_df_list[i]])
            storage_df_current = storage_df_new

        storage_df_unsorted = storage_df_current
        storage_df_sorted = storage_df_unsorted.sort_values(by=["RMSE"], ascending=True)
        if self.save_csv_files == True:
            storage_df_unsorted.to_csv('0-Data_' + str(self.mdl_name) + '_Unsorted.csv')
            storage_df_sorted.to_csv('0-Data_' + str(self.mdl_name) + '_Sorted.csv')

        return storage_df_unsorted, storage_df_sorted

    """ TRY 2 """
    # IDEA: Sort all models by which hyperparameters effect them, then create individaul scripts for each case

    # CASE I: SVM - C, Epsilon -- Linear
    def runSingleGridSearch_SVM_C_Epsilon(self, C_range, epsilon_range, fig_idx):
        """ INPUTS, METHOD, AND OUTPUTS """
        """
        Inputs:
        1) C_range: range for the C-Hyperparameter
        2) epsilon_range: range for the epsilon-Hyperparameter
        3) gamma_input: gamma to be used in the regression run
        4) models_use: logical array for regression class which tell the function which model-types to run
        5) mdl_name: name of the chosen model - used in the recall of data
        6) metric: should be 'r2', 'rmse', or 'cor' - determines which metric to use for comparisons
        7) heatmap_title: string that will in the title of the main heatmap (the one that shows just the metric data)
        8) figure_name: string indicating the current subset of heatmap:
            i.e. Fig 1     = original heatmap
                 Fig 1.2   = heatmap of the second best output from the original
                 Fig 1.2.3 = heatmap of the third best output from the second best output from the original.
            The number of digits represents what zoom-in layer the image is from
        9) output_type: 
        Method:
        First, the function runs setup. This starts with creating lists for the row and column names, which will be the C
            and epsilon values, respectivly. Then, the three (3) matrices that will store the data are initalized. 
            These are:
            1) the metric data, which wills store the data one the metrics alone, 
            2) the ratio of average testing metric data to average training metric data (each will be the average from the
                Nk-CV splits),
            3) and the ratio of the final testing metric to the average training metric data (where the final is from the
                combonation of all Nk-splits - and is the value in the first (1) matrix - and the average is again the 
                average of the kFold splits)
            The data is then iterated over both the C and Epsilon ranges and a regression is run using (C, epsilon, gamma).
            The data final metric data and the average values will then be recalled and the ratios made. The data will then
            be stored.

        Outputs:
        1) metric_data: matrix holding all of the metric data from the test
        """

        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name

        # SETS UP FINAL STORAGE DATAFRAME ------------------------------------------------------------------------------
        df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                        'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                        'avgTR to avgTS', 'avgTR to Final Error']
        df_numRows = len(C_range) * len(epsilon_range)
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        # SETS UP STORAGE FOR EACH HEATMAP-CSV -------------------------------------------------------------------------
        C_names = []
        e_names = []
        numC = len(C_range)
        numE = len(epsilon_range)

        # Row and column names for the heatmap
        for C_i in range(numC):
            C_names.append("C=" + str("%.6f" % C_range[C_i]))
        for e_i in range(numE):
            e_names.append("e=" + str("%.6f" % epsilon_range[e_i]))

        total_points = numC * numE

        # SETS UP STORAGE FOR COLLECTED DATA ---------------------------------------------------------------------------
        error_array = np.zeros((numC, numE))
        r2_array = np.zeros((numC, numE))
        tr_error_array = np.zeros((numC, numE))
        tr_r2_array = np.zeros((numC, numE))
        ratio_trAvg_tsAvg_array = np.zeros((numC, numE))
        ratio_trAvg_array_final = np.zeros((numC, numE))

        # RUNS GRID SEARCH ---------------------------------------------------------------------------------------------
        place_counter = 1
        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                epsilon = epsilon_range[e_idx]

                # Runs Regressions using current [C, epsilon]
                reg = Regression(X_use, Y_use,
                                 C=C, epsilon=epsilon,
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

                # Puts data into correct array
                error_array[C_idx, e_idx] = error
                r2_array[C_idx, e_idx] = r2
                tr_error_array[C_idx, e_idx] = avg_tr_error
                tr_r2_array[C_idx, e_idx] = avg_tr_r2
                ratio_trAvg_tsAvg_array[C_idx, e_idx] = ratio_trAvg_tsAvg
                ratio_trAvg_array_final[C_idx, e_idx] = ratio_trAvg_final

                # Puts data into storage array
                storage_df.loc[place_counter - 1] = [fig_idx, rmse, r2, cor, C, epsilon, 'N/A', 'N/A',
                                                     avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                                     ratio_trAvg_tsAvg, ratio_trAvg_final]
                place_counter += 1

                print("C: " + str(C) + ", e: " + str(epsilon) + ", ceof0: " + str('N/A') + ", gamma: " + str(
                    'N/A') + " (" + str(place_counter) + "/" + str(total_points) + ")" + " || Error: "+str(error)+", R^2: "+str(r2))

        # SAVES HEATMAPS AS CSV-FILES ----------------------------------------------------------------------------------
        if self.save_csv_files == True:
        
            error_current = error_array[:, :]
            R2_current = r2_array[:, :]
            tr_error_current = tr_error_array[:, :]
            tr_r2_current = tr_r2_array[:, :]
            trAvg_tsAvg_current = ratio_trAvg_tsAvg_array[:, :]
            trAvg_final_current = ratio_trAvg_array_final[:, :]

            # Initialization of dataframes
            data_zeros = np.zeros((numC + 1, numE + 1))
            error_data_df = pd.DataFrame(data=data_zeros)
            R2_data_df = pd.DataFrame(data=data_zeros)
            tr_error_data_df = pd.DataFrame(data=data_zeros)
            tr_r2_data_df = pd.DataFrame(data=data_zeros)
            trAvg_tsAvg_data_df = pd.DataFrame(data=data_zeros)
            trAvg_final_data_df = pd.DataFrame(data=data_zeros)

            # File Names
            error_file_name = str(mdl_name) + " - Error Data - " + str(fig_idx) + ".csv"
            r2_file_name = str(mdl_name) + " - R2 Data - " + str(fig_idx) + ".csv"
            tr_error_file_name = str(mdl_name) + " - Training-Error Data - " + str(fig_idx) + ".csv"
            tr_r2_file_name = str(mdl_name) + " - Training-R2 Data - " + str(fig_idx) + ".csv"
            trAvg_tsAvg_file_name = str(mdl_name) + " - avgTR to avgTS - " + str(fig_idx) + ".csv"
            trAvg_final_file_name = str(mdl_name) + " - avgTR to Final - " + str(fig_idx) + ".csv"

            # Error Data
            error_data_df.iloc[0, 1:] = C_names
            error_data_df.iloc[1:, 0] = e_names
            error_data_df.iloc[1:, 1:] = error_current
            error_data_df.to_csv(error_file_name)

            # R^2 Data
            R2_data_df.iloc[0, 1:] = C_names
            R2_data_df.iloc[1:, 0] = e_names
            R2_data_df.iloc[1:, 1:] = R2_current
            R2_data_df.to_csv(r2_file_name)

            # Training-Error Data
            tr_error_data_df.iloc[0, 1:] = C_names
            tr_error_data_df.iloc[1:, 0] = e_names
            tr_error_data_df.iloc[1:, 1:] = tr_error_current
            tr_error_data_df.to_csv(tr_error_file_name)

            # Training-R^2 Data
            tr_r2_data_df.iloc[0, 1:] = C_names
            tr_r2_data_df.iloc[1:, 0] = e_names
            tr_r2_data_df.iloc[1:, 1:] = tr_r2_current
            tr_r2_data_df.to_csv(tr_r2_file_name)

            # Average training to average testing error
            trAvg_tsAvg_data_df.iloc[0, 1:] = C_names
            trAvg_tsAvg_data_df.iloc[1:, 0] = e_names
            trAvg_tsAvg_data_df.iloc[1:, 1:] = trAvg_tsAvg_current
            trAvg_tsAvg_data_df.to_csv(trAvg_tsAvg_file_name)

            # Average training to final error
            trAvg_final_data_df.iloc[0, 1:] = C_names
            trAvg_final_data_df.iloc[1:, 0] = e_names
            trAvg_final_data_df.iloc[1:, 1:] = trAvg_final_current
            trAvg_final_data_df.to_csv(trAvg_final_file_name)

        return error_array, storage_df

    def runFullGridSearch_SVM_C_Epsilon(self, C_input_data, epsilon_input_data):

        HP_data = [[C_input_data, epsilon_input_data]]

        fig_idx_list = self.figureNumbers(self.numLayers, self.numZooms)

        final_storage_df_list = []

        fig_counter = 0
        for layer in range(self.numLayers):
            HP_layer_data = []
            for position_idx in range(len(HP_data)):
                print("Figure: ", fig_idx_list[fig_counter])
                print("Current Layer: ", layer + 1)
                print("Current Position: ", position_idx + 1)

                HP_data_current = HP_data[position_idx]

                # Sets up C and Epsilon ranges
                C_data = HP_data_current[0]
                epsilon_data = HP_data_current[1]
                C_range = np.linspace(C_data[0], C_data[1], self.gridLength)
                epsilon_range = np.linspace(epsilon_data[0], epsilon_data[1], self.gridLength)

                # Runs the heatmap with the current C & Epsilon ranges
                error_matrix, storage_df_temp = self.runSingleGridSearch_SVM_C_Epsilon(C_range, epsilon_range, fig_idx_list[fig_counter])

                # Gets the new ranges from the error-matrix
                HP_new_list, Index_Values = self.determineTopHPs_2HP(error_matrix, C_range, epsilon_range, self.numZooms)

                if layer == self.numLayers - 1:
                    final_storage_df_list.append(storage_df_temp)

                for i in HP_new_list:
                    HP_layer_data.append(i)

                fig_counter += 1

            HP_data = HP_layer_data

        # Initalizes the final storage of the data on the final layer
        storage_df_current = pd.DataFrame(columns=['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                                                   'Average Training-RMSE', 'Average Training-R^2',
                                                   'Average Training-Cor',
                                                   'avgTR to avgTS', 'avgTR to Final Error'])
        for i in range(len(final_storage_df_list)):
            storage_df_new = pd.concat([storage_df_current, final_storage_df_list[i]])
            storage_df_current = storage_df_new

        storage_df_unsorted = storage_df_current
        storage_df_sorted = storage_df_unsorted.sort_values(by=["RMSE"], ascending=True)
        if self.save_csv_files == True:
            storage_df_unsorted.to_csv('0-Data_' + str(self.mdl_name) + '_Unsorted.csv')
            storage_df_sorted.to_csv('0-Data_' + str(self.mdl_name) + '_Sorted.csv')

        return storage_df_unsorted, storage_df_sorted

    # CASE II: SVM - C, Epsilon, Gamma -- RBF
    def runSingleGridSearch_SVM_C_Epsilon_Gamma(self, C_range, epsilon_range, gamma_range, fig_idx):
        """ INPUTS, METHOD, AND OUTPUTS """
        """
        Inputs:
        1) C_range: range for the C-Hyperparameter
        2) epsilon_range: range for the epsilon-Hyperparameter
        3) gamma_input: gamma to be used in the regression run
        4) models_use: logical array for regression class which tell the function which model-types to run
        5) mdl_name: name of the chosen model - used in the recall of data
        6) metric: should be 'r2', 'rmse', or 'cor' - determines which metric to use for comparisons
        7) heatmap_title: string that will in the title of the main heatmap (the one that shows just the metric data)
        8) figure_name: string indicating the current subset of heatmap:
            i.e. Fig 1     = original heatmap
                 Fig 1.2   = heatmap of the second best output from the original
                 Fig 1.2.3 = heatmap of the third best output from the second best output from the original.
            The number of digits represents what zoom-in layer the image is from
        9) output_type: 
        Method:
        First, the function runs setup. This starts with creating lists for the row and column names, which will be the C
            and epsilon values, respectivly. Then, the three (3) matrices that will store the data are initalized. 
            These are:
            1) the metric data, which wills store the data one the metrics alone, 
            2) the ratio of average testing metric data to average training metric data (each will be the average from the
                Nk-CV splits),
            3) and the ratio of the final testing metric to the average training metric data (where the final is from the
                combonation of all Nk-splits - and is the value in the first (1) matrix - and the average is again the 
                average of the kFold splits)
            The data is then iterated over both the C and Epsilon ranges and a regression is run using (C, epsilon, gamma).
            The data final metric data and the average values will then be recalled and the ratios made. The data will then
            be stored.

        Outputs:
        1) metric_data: matrix holding all of the metric data from the test
        """

        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name

        # SETS UP FINAL STORAGE DATAFRAME ------------------------------------------------------------------------------
        df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                        'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                        'avgTR to avgTS', 'avgTR to Final Error']
        df_numRows = len(C_range) * len(epsilon_range) * len(gamma_range)
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        # SETS UP STORAGE FOR EACH HEATMAP-CSV -------------------------------------------------------------------------
        C_names = []
        e_names = []
        g_names = []
        numC = len(C_range)
        numE = len(epsilon_range)
        numG = len(gamma_range)

        # Row and column names for the heatmap
        for C_i in range(numC):
            C_names.append("C=" + str("%.6f" % C_range[C_i]))
        for e_i in range(numE):
            e_names.append("e=" + str("%.6f" % epsilon_range[e_i]))
        for g_i in range(numG):
            g_names.append('gamma=' + str("%.6f" % gamma_range[g_i]))

        total_points = numC * numE * numG

        # SETS UP STORAGE FOR COLLECTED DATA ---------------------------------------------------------------------------
        error_array = np.zeros((numC, numE, numG))
        r2_array = np.zeros((numC, numE, numG))
        tr_error_array = np.zeros((numC, numE, numG))
        tr_r2_array = np.zeros((numC, numE, numG))
        ratio_trAvg_tsAvg_array = np.zeros((numC, numE, numG))
        ratio_trAvg_array_final = np.zeros((numC, numE, numG))

        # RUNS GRID SEARCH ---------------------------------------------------------------------------------------------
        place_counter = 1
        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                epsilon = epsilon_range[e_idx]
                for g_idx in range(numG):
                    gamma = gamma_range[g_idx]

                    # Runs Regressions using current [C, epsilon, gamma, gamma]
                    reg = Regression(X_use, Y_use,
                                     C=C, gamma=gamma, epsilon=epsilon,
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

                    # Puts data into correct array
                    error_array[C_idx, e_idx, g_idx] = error
                    r2_array[C_idx, e_idx, g_idx] = r2
                    tr_error_array[C_idx, e_idx, g_idx] = avg_tr_error
                    tr_r2_array[C_idx, e_idx, g_idx] = avg_tr_r2
                    ratio_trAvg_tsAvg_array[C_idx, e_idx, g_idx] = ratio_trAvg_tsAvg
                    ratio_trAvg_array_final[C_idx, e_idx, g_idx] = ratio_trAvg_final

                    # Puts data into storage array
                    storage_df.loc[place_counter - 1] = [fig_idx, rmse, r2, cor, C, epsilon, 'N/A', gamma,
                                                         avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                                         ratio_trAvg_tsAvg, ratio_trAvg_final]
                    place_counter += 1

                    print("C: " + str(C) + ", e: " + str(epsilon) + ", ceof0: " + str('N/A') + ", gamma: " + str(
                        gamma) + " (" + str(place_counter) + "/" + str(total_points) + ")" + " || Error: "+str(error)+", R^2: "+str(r2))

        # SAVES HEATMAPS AS CSV-FILES ----------------------------------------------------------------------------------
        if self.save_csv_files == True:
            for f in range(numG):
                gamma_current = gamma_range[f]
                folder_name = str(fig_idx) + "-gamma_" + str(gamma_current)
                os.mkdir(folder_name)
                os.chdir(folder_name)

                error_current = error_array[:, :, f]
                R2_current = r2_array[:, :, f]
                tr_error_current = tr_error_array[:, :, f]
                tr_r2_current = tr_r2_array[:, :, f]
                trAvg_tsAvg_current = ratio_trAvg_tsAvg_array[:, :, f]
                trAvg_final_current = ratio_trAvg_array_final[:, :, f]

                # Initialization of dataframes
                data_zeros = np.zeros((numC + 1, numE + 1))
                error_data_df = pd.DataFrame(data=data_zeros)
                R2_data_df = pd.DataFrame(data=data_zeros)
                tr_error_data_df = pd.DataFrame(data=data_zeros)
                tr_r2_data_df = pd.DataFrame(data=data_zeros)
                trAvg_tsAvg_data_df = pd.DataFrame(data=data_zeros)
                trAvg_final_data_df = pd.DataFrame(data=data_zeros)

                # File Names
                error_file_name = str(mdl_name) + " - Error Data - " + str(fig_idx) + ".csv"
                r2_file_name = str(mdl_name) + " - R2 Data - " + str(fig_idx) + ".csv"
                tr_error_file_name = str(mdl_name) + " - Training-Error Data - " + str(fig_idx) + ".csv"
                tr_r2_file_name = str(mdl_name) + " - Training-R2 Data - " + str(fig_idx) + ".csv"
                trAvg_tsAvg_file_name = str(mdl_name) + " - avgTR to avgTS - " + str(fig_idx) + ".csv"
                trAvg_final_file_name = str(mdl_name) + " - avgTR to Final - " + str(fig_idx) + ".csv"

                # Error Data
                error_data_df.iloc[0, 1:] = C_names
                error_data_df.iloc[1:, 0] = e_names
                error_data_df.iloc[1:, 1:] = error_current
                error_data_df.to_csv(error_file_name)

                # R^2 Data
                R2_data_df.iloc[0, 1:] = C_names
                R2_data_df.iloc[1:, 0] = e_names
                R2_data_df.iloc[1:, 1:] = R2_current
                R2_data_df.to_csv(r2_file_name)

                # Training-Error Data
                tr_error_data_df.iloc[0, 1:] = C_names
                tr_error_data_df.iloc[1:, 0] = e_names
                tr_error_data_df.iloc[1:, 1:] = tr_error_current
                tr_error_data_df.to_csv(tr_error_file_name)

                # Training-R^2 Data
                tr_r2_data_df.iloc[0, 1:] = C_names
                tr_r2_data_df.iloc[1:, 0] = e_names
                tr_r2_data_df.iloc[1:, 1:] = tr_r2_current
                tr_r2_data_df.to_csv(tr_r2_file_name)

                # Average training to average testing error
                trAvg_tsAvg_data_df.iloc[0, 1:] = C_names
                trAvg_tsAvg_data_df.iloc[1:, 0] = e_names
                trAvg_tsAvg_data_df.iloc[1:, 1:] = trAvg_tsAvg_current
                trAvg_tsAvg_data_df.to_csv(trAvg_tsAvg_file_name)

                # Average training to final error
                trAvg_final_data_df.iloc[0, 1:] = C_names
                trAvg_final_data_df.iloc[1:, 0] = e_names
                trAvg_final_data_df.iloc[1:, 1:] = trAvg_final_current
                trAvg_final_data_df.to_csv(trAvg_final_file_name)

                os.chdir('..')

        return error_array, storage_df

    def runFullGridSearch_SVM_C_Epsilon_Gamma(self, C_input_data, epsilon_input_data, gamma_input_data):

        HP_data = [[C_input_data, epsilon_input_data, gamma_input_data]]

        fig_idx_list = self.figureNumbers(self.numLayers, self.numZooms)

        final_storage_df_list = []

        # GETS GAMMA VALUE ---------------------------------------------------------------------------------------------
        goodIds = self.goodIDs
        Y_inp = self.Y_inp
        Y_use = [Y_inp[i] for i in range(len(Y_inp)) if goodIds[i]]

        fig_counter = 0
        for layer in range(self.numLayers):
            HP_layer_data = []
            for position_idx in range(len(HP_data)):
                print("Figure: ", fig_idx_list[fig_counter])
                print("Current Layer: ", layer + 1)
                print("Current Position: ", position_idx + 1)

                HP_data_current = HP_data[position_idx]

                # Sets up C and Epsilon ranges
                C_data = HP_data_current[0]
                epsilon_data = HP_data_current[1]
                gamma_data = HP_data_current[2]
                C_range = np.linspace(C_data[0], C_data[1], self.gridLength)
                epsilon_range = np.linspace(epsilon_data[0], epsilon_data[1], self.gridLength)
                gamma_range = np.linspace(gamma_data[0], gamma_data[1], self.gridLength)

                # Runs the heatmap with the current C & Epsilon ranges
                error_matrix, storage_df_temp = self.runSingleGridSearch_SVM_C_Epsilon_Gamma(C_range, epsilon_range,
                                                                                             gamma_range, fig_idx_list[fig_counter])

                # Gets the new ranges from the error-matrix
                HP_new_list, Index_Values = self.determineTopHPs_3HP(error_matrix, C_range, epsilon_range, gamma_range,
                                                                     self.numZooms)

                if layer == self.numLayers - 1:
                    final_storage_df_list.append(storage_df_temp)

                for i in HP_new_list:
                    HP_layer_data.append(i)

                fig_counter += 1

            HP_data = HP_layer_data

        # Initalizes the final storage of the data on the final layer
        storage_df_current = pd.DataFrame(columns=['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                                                   'Average Training-RMSE', 'Average Training-R^2',
                                                   'Average Training-Cor',
                                                   'avgTR to avgTS', 'avgTR to Final Error'])
        for i in range(len(final_storage_df_list)):
            storage_df_new = pd.concat([storage_df_current, final_storage_df_list[i]])
            storage_df_current = storage_df_new

        storage_df_unsorted = storage_df_current
        storage_df_sorted = storage_df_unsorted.sort_values(by=["RMSE"], ascending=True)
        if self.save_csv_files == True:
            storage_df_unsorted.to_csv('0-Data_' + str(self.mdl_name) + '_Unsorted.csv')
            storage_df_sorted.to_csv('0-Data_' + str(self.mdl_name) + '_Sorted.csv')

        return storage_df_unsorted, storage_df_sorted

    # CASE III: SVM - C, Epsilon, Gamma, Coef0 -- Poly2, Poly3
    def runSingleGridSearch_SVM_C_Epsilon_Gamma_Coef0(self, C_range, epsilon_range, gamma_range, coef0_range, fig_idx):
        """ INPUTS, METHOD, AND OUTPUTS """
        """
        Inputs:
        1) C_range: range for the C-Hyperparameter
        2) epsilon_range: range for the epsilon-Hyperparameter
        3) gamma_input: gamma to be used in the regression run
        4) models_use: logical array for regression class which tell the function which model-types to run
        5) mdl_name: name of the chosen model - used in the recall of data
        6) metric: should be 'r2', 'rmse', or 'cor' - determines which metric to use for comparisons
        7) heatmap_title: string that will in the title of the main heatmap (the one that shows just the metric data)
        8) figure_name: string indicating the current subset of heatmap:
            i.e. Fig 1     = original heatmap
                 Fig 1.2   = heatmap of the second best output from the original
                 Fig 1.2.3 = heatmap of the third best output from the second best output from the original.
            The number of digits represents what zoom-in layer the image is from
        9) output_type: 
        Method:
        First, the function runs setup. This starts with creating lists for the row and column names, which will be the C
            and epsilon values, respectivly. Then, the three (3) matrices that will store the data are initalized. 
            These are:
            1) the metric data, which wills store the data one the metrics alone, 
            2) the ratio of average testing metric data to average training metric data (each will be the average from the
                Nk-CV splits),
            3) and the ratio of the final testing metric to the average training metric data (where the final is from the
                combonation of all Nk-splits - and is the value in the first (1) matrix - and the average is again the 
                average of the kFold splits)
            The data is then iterated over both the C and Epsilon ranges and a regression is run using (C, epsilon, gamma).
            The data final metric data and the average values will then be recalled and the ratios made. The data will then
            be stored.

        Outputs:
        1) metric_data: matrix holding all of the metric data from the test
        """

        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name

        # SETS UP FINAL STORAGE DATAFRAME ------------------------------------------------------------------------------
        df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                        'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                        'avgTR to avgTS', 'avgTR to Final Error']
        df_numRows = len(C_range) * len(epsilon_range) * len(gamma_range)
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        # SETS UP STORAGE FOR EACH HEATMAP-CSV -------------------------------------------------------------------------
        C_names = []
        e_names = []
        g_names = []
        c0_names = []
        numC = len(C_range)
        numE = len(epsilon_range)
        numG = len(gamma_range)
        numC0 = len(coef0_range)

        # Row and column names for the heatmap
        for C_i in range(numC):
            C_names.append("C=" + str("%.6f" % C_range[C_i]))
        for e_i in range(numE):
            e_names.append("e=" + str("%.6f" % epsilon_range[e_i]))
        for g_i in range(numG):
            g_names.append('gamma=' + str("%.6f" % gamma_range[g_i]))
        for c0_i in range(numC0):
            c0_names.append('coef0=' + str("%.6f" % coef0_range[c0_i]))

        total_points = numC * numE * numG * numC0

        # SETS UP STORAGE FOR COLLECTED DATA ---------------------------------------------------------------------------
        error_array = np.zeros((numC, numE, numG, numC0))
        r2_array = np.zeros((numC, numE, numG, numC0))
        tr_error_array = np.zeros((numC, numE, numG, numC0))
        tr_r2_array = np.zeros((numC, numE, numG, numC0))
        ratio_trAvg_tsAvg_array = np.zeros((numC, numE, numG, numC0))
        ratio_trAvg_array_final = np.zeros((numC, numE, numG, numC0))

        # RUNS GRID SEARCH ---------------------------------------------------------------------------------------------
        place_counter = 1
        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                epsilon = epsilon_range[e_idx]
                for g_idx in range(numG):
                    gamma = gamma_range[g_idx]
                    for c0_idx in range(numC0):
                        coef0 = coef0_range[c0_idx]

                        # Runs Regressions using current [C, epsilon, gamma, gamma]
                        reg = Regression(X_use, Y_use,
                                         C=C, gamma=gamma, epsilon=epsilon, coef0=coef0,
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

                        # Puts data into correct array
                        error_array[C_idx, e_idx, g_idx, c0_idx] = error
                        r2_array[C_idx, e_idx, g_idx, c0_idx] = r2
                        tr_error_array[C_idx, e_idx, g_idx, c0_idx] = avg_tr_error
                        tr_r2_array[C_idx, e_idx, g_idx, c0_idx] = avg_tr_r2
                        ratio_trAvg_tsAvg_array[C_idx, e_idx, g_idx, c0_idx] = ratio_trAvg_tsAvg
                        ratio_trAvg_array_final[C_idx, e_idx, g_idx, c0_idx] = ratio_trAvg_final

                        # Puts data into storage array
                        storage_df.loc[place_counter - 1] = [fig_idx, rmse, r2, cor, C, epsilon, coef0, gamma,
                                                             avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                                             ratio_trAvg_tsAvg, ratio_trAvg_final]
                        place_counter += 1

                        print("C: " + str(C) + ", e: " + str(epsilon) + ", ceof0: " + str(coef0) + ", gamma: " + str(
                            gamma) + " (" + str(place_counter) + "/" + str(total_points) + ")" + " || Error: " + str(
                            error) + ", R^2: " + str(r2))

        # SAVES HEATMAPS AS CSV-FILES ----------------------------------------------------------------------------------
        if self.save_csv_files == True:
            for c in range(numC0):
                coef0_current = coef0_range[c]
                folder_name = str(fig_idx) +"-coef0_"+str(coef0_current)
                os.mkdir(folder_name)
                os.chdir(folder_name)

                error_coef0 = error_array[:, :, :, c]
                r2_coef0 = r2_array[:, :, :, c]
                tr_error_coef0 = tr_error_array[:, :, :, c]
                tr_r2_coef0 = tr_r2_array[:, :, :, c]
                ratio_trAvg_tsAvg_coef0 = ratio_trAvg_tsAvg_array[:, :, :, c]
                ratio_trAvg_coef0_final = ratio_trAvg_array_final[:, :, :, c]

                for g in range(numG):
                    gamma_current = gamma_range[g]
                    folder_name = str(fig_idx) + "-gamma_" + str(gamma_current)
                    os.mkdir(folder_name)
                    os.chdir(folder_name)

                    error_current = error_coef0[:, :, g]
                    R2_current = r2_coef0[:, :, g]
                    tr_error_current = tr_error_coef0[:, :, g]
                    tr_r2_current = tr_r2_coef0[:, :, g]
                    trAvg_tsAvg_current = ratio_trAvg_tsAvg_coef0[:, :, g]
                    trAvg_final_current = ratio_trAvg_coef0_final[:, :, g]

                    # Initialization of dataframes
                    data_zeros = np.zeros((numC + 1, numE + 1))
                    error_data_df = pd.DataFrame(data=data_zeros)
                    R2_data_df = pd.DataFrame(data=data_zeros)
                    tr_error_data_df = pd.DataFrame(data=data_zeros)
                    tr_r2_data_df = pd.DataFrame(data=data_zeros)
                    trAvg_tsAvg_data_df = pd.DataFrame(data=data_zeros)
                    trAvg_final_data_df = pd.DataFrame(data=data_zeros)

                    # File Names
                    error_file_name = str(mdl_name) + " - Error Data - " + str(fig_idx) + ".csv"
                    r2_file_name = str(mdl_name) + " - R2 Data - " + str(fig_idx) + ".csv"
                    tr_error_file_name = str(mdl_name) + " - Training-Error Data - " + str(fig_idx) + ".csv"
                    tr_r2_file_name = str(mdl_name) + " - Training-R2 Data - " + str(fig_idx) + ".csv"
                    trAvg_tsAvg_file_name = str(mdl_name) + " - avgTR to avgTS - " + str(fig_idx) + ".csv"
                    trAvg_final_file_name = str(mdl_name) + " - avgTR to Final - " + str(fig_idx) + ".csv"

                    # Error Data
                    error_data_df.iloc[0, 1:] = C_names
                    error_data_df.iloc[1:, 0] = e_names
                    error_data_df.iloc[1:, 1:] = error_current
                    error_data_df.to_csv(error_file_name)

                    # R^2 Data
                    R2_data_df.iloc[0, 1:] = C_names
                    R2_data_df.iloc[1:, 0] = e_names
                    R2_data_df.iloc[1:, 1:] = R2_current
                    R2_data_df.to_csv(r2_file_name)

                    # Training-Error Data
                    tr_error_data_df.iloc[0, 1:] = C_names
                    tr_error_data_df.iloc[1:, 0] = e_names
                    tr_error_data_df.iloc[1:, 1:] = tr_error_current
                    tr_error_data_df.to_csv(tr_error_file_name)

                    # Training-R^2 Data
                    tr_r2_data_df.iloc[0, 1:] = C_names
                    tr_r2_data_df.iloc[1:, 0] = e_names
                    tr_r2_data_df.iloc[1:, 1:] = tr_r2_current
                    tr_r2_data_df.to_csv(tr_r2_file_name)

                    # Average training to average testing error
                    trAvg_tsAvg_data_df.iloc[0, 1:] = C_names
                    trAvg_tsAvg_data_df.iloc[1:, 0] = e_names
                    trAvg_tsAvg_data_df.iloc[1:, 1:] = trAvg_tsAvg_current
                    trAvg_tsAvg_data_df.to_csv(trAvg_tsAvg_file_name)

                    # Average training to final error
                    trAvg_final_data_df.iloc[0, 1:] = C_names
                    trAvg_final_data_df.iloc[1:, 0] = e_names
                    trAvg_final_data_df.iloc[1:, 1:] = trAvg_final_current
                    trAvg_final_data_df.to_csv(trAvg_final_file_name)
                    os.chdir('..')
                os.chdir('..')

        return error_array, storage_df

    def runFullGridSearch_SVM_C_Epsilon_Gamma_Coef0(self, C_input_data, epsilon_input_data, gamma_input_data, coef0_input_data):

        HP_data = [[C_input_data, epsilon_input_data, gamma_input_data, coef0_input_data]]

        fig_idx_list = self.figureNumbers(self.numLayers, self.numZooms)

        final_storage_df_list = []

        # GETS GAMMA VALUE ---------------------------------------------------------------------------------------------
        goodIds = self.goodIDs
        Y_inp = self.Y_inp
        Y_use = [Y_inp[i] for i in range(len(Y_inp)) if goodIds[i]]

        fig_counter = 0
        for layer in range(self.numLayers):
            HP_layer_data = []
            for position_idx in range(len(HP_data)):
                print("Figure: ", fig_idx_list[fig_counter])
                print("Current Layer: ", layer + 1)
                print("Current Position: ", position_idx + 1)

                HP_data_current = HP_data[position_idx]

                # Sets up C and Epsilon ranges
                C_data = HP_data_current[0]
                epsilon_data = HP_data_current[1]
                gamma_data = HP_data_current[2]
                coef0_data = HP_data_current[3]
                C_range = np.linspace(C_data[0], C_data[1], self.gridLength)
                epsilon_range = np.linspace(epsilon_data[0], epsilon_data[1], self.gridLength)
                gamma_range = np.linspace(gamma_data[0], gamma_data[1], self.gridLength)
                coef0_range = np.linspace(coef0_data[0], coef0_data[1], self.gridLength)

                # Runs the heatmap with the current C & Epsilon ranges
                error_matrix, storage_df_temp = self.runSingleGridSearch_SVM_C_Epsilon_Gamma_Coef0(C_range, epsilon_range,
                                                                                             gamma_range, coef0_range, fig_idx_list[fig_counter])

                # Gets the new ranges from the error-matrix
                HP_new_list, Index_Values = self.determineTopHPs_4HP(error_matrix, C_range, epsilon_range, gamma_range, coef0_range,
                                                                     self.numZooms)

                if layer == self.numLayers - 1:
                    final_storage_df_list.append(storage_df_temp)

                for i in HP_new_list:
                    HP_layer_data.append(i)

                fig_counter += 1

            HP_data = HP_layer_data

        # Initalizes the final storage of the data on the final layer
        storage_df_current = pd.DataFrame(columns=['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                                                   'Average Training-RMSE', 'Average Training-R^2',
                                                   'Average Training-Cor',
                                                   'avgTR to avgTS', 'avgTR to Final Error'])
        for i in range(len(final_storage_df_list)):
            storage_df_new = pd.concat([storage_df_current, final_storage_df_list[i]])
            storage_df_current = storage_df_new

        storage_df_unsorted = storage_df_current
        storage_df_sorted = storage_df_unsorted.sort_values(by=["RMSE"], ascending=True)
        if self.save_csv_files == True:
            storage_df_unsorted.to_csv('0-Data_' + str(self.mdl_name) + '_Unsorted.csv')
            storage_df_sorted.to_csv('0-Data_' + str(self.mdl_name) + '_Sorted.csv')

        return storage_df_unsorted, storage_df_sorted

    # CASE IV: GPR - Noise, SigF, Length -- RBF, Matern 3/2, Matern 5/2
    def runSingleGridSearch_GPR_Noise_SigF_Length(self, noise_range, sigF_range, length_range, fig_idx):

        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name

        # SETS UP FINAL STORAGE DATAFRAME ------------------------------------------------------------------------------
        df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'Noise', 'Sigma_F', 'Length',
                        'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                        'avgTR to avgTS', 'avgTR to Final Error']
        df_numRows = len(noise_range) * len(sigF_range) * len(length_range)
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        # SETS UP STORAGE FOR EACH HEATMAP-CSV -------------------------------------------------------------------------
        noiseNames = []
        sigFNames = []
        lengthNames = []
        numNoise = len(noise_range)
        numSigF = len(sigF_range)
        numLength = len(length_range)

        # Row and column names for the heatmap
        for i in range(numNoise):
            noiseNames.append("noise=" + str("%.6f" % noise_range[i]))
        for i in range(numSigF):
            sigFNames.append("e=" + str("%.6f" % sigF_range[i]))
        for i in range(numLength):
            lengthNames.append("e=" + str("%.6f" % length_range[i]))

        total_points = numNoise * numSigF * numLength

        # SETS UP STORAGE FOR COLLECTED DATA ---------------------------------------------------------------------------
        error_array = np.zeros((numNoise, numSigF, numLength))
        r2_array = np.zeros((numNoise, numSigF, numLength))
        tr_error_array = np.zeros((numNoise, numSigF, numLength))
        tr_r2_array = np.zeros((numNoise, numSigF, numLength))
        ratio_trAvg_tsAvg_array = np.zeros((numNoise, numSigF, numLength))
        ratio_trAvg_final_array = np.zeros((numNoise, numSigF, numLength))

        place_counter = 1
        for i in range(len(noise_range)):
            for j in range(len(sigF_range)):
                for k in range(len(length_range)):
                    noise = noise_range[i]
                    sigF = sigF_range[j]
                    l = length_range[k]

                    print("noise: " + str(noise) + ", sigF: " + str(sigF) + ", length: " + str(l) + " (" + str(
                        place_counter) + "/" + str(total_points) + ")")

                    reg = Regression(X_use, Y_use,
                                     noise=noise, sigma_F=sigF, scale_length=l,
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

                    # Puts data into correct array
                    error_array[i, j, k] = error
                    r2_array[i, j, k] = r2
                    tr_error_array[i, j, k] = avg_tr_error
                    tr_r2_array[i, j, k] = avg_tr_r2
                    ratio_trAvg_tsAvg_array[i, j, k] = ratio_trAvg_tsAvg
                    ratio_trAvg_final_array[i, j, k] = ratio_trAvg_final

                    # Puts data into storage array
                    storage_df.loc[place_counter - 1] = [fig_idx, rmse, r2, cor, noise, sigF, l,
                                                         avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                                         ratio_trAvg_tsAvg, ratio_trAvg_final]
                    place_counter += 1

        # SAVES DATA TO CSV-FILES --------------------------------------------------------------------------------------
        if self.save_csv_files == True:
            for n in range(numNoise):
                noise_current = noise_range[n]
                folder_name = str(fig_idx) + "-noise_" + str(noise_current)
                os.mkdir(folder_name)
                os.chdir(folder_name)

                error_array_current = error_array[n, :, :]
                r2_array_current = r2_array[n, :, :]
                tr_error_array_current = tr_error_array[n, :, :]
                tr_r2_array_current = tr_r2_array[n, :, :]
                ratio_trAvg_tsAvg_array_current = ratio_trAvg_tsAvg_array[n, :, :]
                ratio_trAvg_array_final_current = ratio_trAvg_final_array[n, :, :]

                data_zeros = np.zeros((numSigF + 1, numLength + 1))
                error_data_df = pd.DataFrame(data=data_zeros)
                R2_data_df = pd.DataFrame(data=data_zeros)
                tr_error_data_df = pd.DataFrame(data=data_zeros)
                tr_r2_data_df = pd.DataFrame(data=data_zeros)
                trAvg_tsAvg_data_df = pd.DataFrame(data=data_zeros)
                trAvg_final_data_df = pd.DataFrame(data=data_zeros)

                # File Names
                error_file_name = str(mdl_name) + " - Error Data - " + str(fig_idx) + ".csv"
                r2_file_name = str(mdl_name) + " - R2 Data - " + str(fig_idx) + ".csv"
                tr_error_file_name = str(mdl_name) + " - Training-Error Data - " + str(fig_idx) + ".csv"
                tr_r2_file_name = str(mdl_name) + " - Training-R2 Data - " + str(fig_idx) + ".csv"
                trAvg_tsAvg_file_name = str(mdl_name) + " - avgTR to avgTS - " + str(fig_idx) + ".csv"
                trAvg_final_file_name = str(mdl_name) + " - avgTR to Final - " + str(fig_idx) + ".csv"

                # Error Data
                error_data_df.iloc[0, 1:] = lengthNames
                error_data_df.iloc[1:, 0] = sigFNames
                error_data_df.iloc[1:, 1:] = error_array_current
                error_data_df.to_csv(error_file_name)

                # R^2 Data
                R2_data_df.iloc[0, 1:] = lengthNames
                R2_data_df.iloc[1:, 0] = sigFNames
                R2_data_df.iloc[1:, 1:] = r2_array_current
                R2_data_df.to_csv(r2_file_name)

                # Training-Error Data
                tr_error_data_df.iloc[0, 1:] = lengthNames
                tr_error_data_df.iloc[1:, 0] = sigFNames
                tr_error_data_df.iloc[1:, 1:] = tr_error_array_current
                tr_error_data_df.to_csv(tr_error_file_name)

                # Training-R^2 Data
                tr_r2_data_df.iloc[0, 1:] = lengthNames
                tr_r2_data_df.iloc[1:, 0] = sigFNames
                tr_r2_data_df.iloc[1:, 1:] = tr_r2_array_current
                tr_r2_data_df.to_csv(tr_r2_file_name)

                # Average training to average testing error
                trAvg_tsAvg_data_df.iloc[0, 1:] = lengthNames
                trAvg_tsAvg_data_df.iloc[1:, 0] = sigFNames
                trAvg_tsAvg_data_df.iloc[1:, 1:] = ratio_trAvg_tsAvg_array_current
                trAvg_tsAvg_data_df.to_csv(trAvg_tsAvg_file_name)

                # Average training to final error
                trAvg_final_data_df.iloc[0, 1:] = lengthNames
                trAvg_final_data_df.iloc[1:, 0] = sigFNames
                trAvg_final_data_df.iloc[1:, 1:] = ratio_trAvg_array_final_current
                trAvg_final_data_df.to_csv(trAvg_final_file_name)

                os.chdir('..')

        return error_array, storage_df

    def runFullGridSearch_GPR_Noise_SigF_Length(self, noise_input_data, sigF_input_data, length_input_data):

        HP_data = [[noise_input_data, sigF_input_data, length_input_data]]

        fig_idx_list = self.figureNumbers(self.numLayers, self.numZooms)

        final_storage_df_list = []

        fig_counter = 0
        for layer in range(self.numLayers):
            HP_layer_data = []
            for position_idx in range(len(HP_data)):
                print("Figure: ", fig_idx_list[fig_counter])
                print("Current Layer: ", layer + 1)
                print("Current Position: ", position_idx + 1)

                HP_data_current = HP_data[position_idx]

                # Sets up C and Epsilon ranges
                noise_data = HP_data_current[0]
                sigF_data = HP_data_current[1]
                length_data = HP_data_current[2]
                noise_range = np.linspace(noise_data[0], noise_data[1], self.gridLength)
                sigF_range = np.linspace(sigF_data[0], sigF_data[1], self.gridLength)
                length_range = np.linspace(length_data[0], length_data[1], self.gridLength)

                # Runs the heatmap with the current Noise, Sigma_F, & Scale-Length ranges
                error_matrix, storage_df_temp = self.runSingleGridSearch_GPR_Noise_SigF_Length(noise_range, sigF_range, length_range,
                                                                             fig_idx_list[fig_counter])

                # Gets the new ranges from the error-matrix
                HP_new_list, Index_Values = self.determineTopHPs_3HP(error_matrix, noise_range, sigF_range,
                                                                     length_range, self.numZooms)

                if layer == self.numLayers - 1:
                    final_storage_df_list.append(storage_df_temp)

                for i in HP_new_list:
                    HP_layer_data.append(i)

                fig_counter += 1

            HP_data = HP_layer_data

        # Initalizes the final storage of the data on the final layer
        storage_df_current = pd.DataFrame(columns=['Figure', 'RMSE', 'R^2', 'Cor', 'Noise', 'Sigma_F', 'Length',
                                                   'Average Training-RMSE', 'Average Training-R^2',
                                                   'Average Training-Cor',
                                                   'avgTR to avgTS', 'avgTR to Final Error'])

        for i in range(len(final_storage_df_list)):
            storage_df_new = pd.concat([storage_df_current, final_storage_df_list[i]])
            storage_df_current = storage_df_new

        storage_df_unsorted = storage_df_current
        storage_df_sorted = storage_df_unsorted.sort_values(by=["RMSE"], ascending=True)
        if self.save_csv_files == True:
            storage_df_unsorted.to_csv('0-Data_' + str(self.mdl_name) + '_Unsorted.csv')
            storage_df_sorted.to_csv('0-Data_' + str(self.mdl_name) + '_Sorted.csv')

        return storage_df_unsorted, storage_df_sorted

    # CASE V: GPR - Noise, SigF, Length, Alpha -- Rational Quadratic
    def runSingleGridSearch_GPR_Noise_SigF_Length_Alpha(self, noise_range, sigF_range, length_range, alpha_range, fig_idx):

        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name

        # SETS UP FINAL STORAGE DATAFRAME ------------------------------------------------------------------------------
        df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'Noise', 'Sigma_F', 'Length', 'Alpha',
                        'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                        'avgTR to avgTS', 'avgTR to Final Error']
        df_numRows = len(noise_range) * len(sigF_range) * len(length_range) * len(alpha_range)
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        # SETS UP STORAGE FOR EACH HEATMAP-CSV -------------------------------------------------------------------------
        noiseNames = []
        sigFNames = []
        lengthNames = []
        alphaNames = []
        numNoise = len(noise_range)
        numSigF = len(sigF_range)
        numLength = len(length_range)
        numAlpha = len(alpha_range)

        # Row and column names for the heatmap
        for i in range(numNoise):
            noiseNames.append("noise=" + str("%.6f" % noise_range[i]))
        for i in range(numSigF):
            sigFNames.append("sigF=" + str("%.6f" % sigF_range[i]))
        for i in range(numLength):
            lengthNames.append("Length=" + str("%.6f" % length_range[i]))
        for i in range(numAlpha):
            alphaNames.append("Alpha=" + str("%.6f" % alpha_range[i]))

        total_points = numNoise * numSigF * numLength * numAlpha

        # SETS UP STORAGE FOR COLLECTED DATA ---------------------------------------------------------------------------
        error_array = np.zeros((numNoise, numSigF, numLength, numAlpha))
        r2_array = np.zeros((numNoise, numSigF, numLength, numAlpha))
        tr_error_array = np.zeros((numNoise, numSigF, numLength, numAlpha))
        tr_r2_array = np.zeros((numNoise, numSigF, numLength, numAlpha))
        ratio_trAvg_tsAvg_array = np.zeros((numNoise, numSigF, numLength, numAlpha))
        ratio_trAvg_final_array = np.zeros((numNoise, numSigF, numLength, numAlpha))

        place_counter = 1
        for i in range(len(noise_range)):
            for j in range(len(sigF_range)):
                for k in range(len(length_range)):
                    for a in range(len(alpha_range)):

                        noise = noise_range[i]
                        sigF = sigF_range[j]
                        l = length_range[k]
                        alpha = alpha_range[a]

                        print("noise: " + str(noise) + ", sigF: " + str(sigF) + ", length: " + str(l) + ", alpha: " + str(alpha) + " (" + str(
                            place_counter) + "/" + str(total_points) + ")")

                        reg = Regression(X_use, Y_use,
                                         noise=noise, sigma_F=sigF, scale_length=l, alpha=alpha,
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

                        # Puts data into correct array
                        error_array[i, j, k, a] = error
                        r2_array[i, j, k, a] = r2
                        tr_error_array[i, j, k, a] = avg_tr_error
                        tr_r2_array[i, j, k, a] = avg_tr_r2
                        ratio_trAvg_tsAvg_array[i, j, k, a] = ratio_trAvg_tsAvg
                        ratio_trAvg_final_array[i, j, k, a] = ratio_trAvg_final

                        # Puts data into storage array
                        storage_df.loc[place_counter - 1] = [fig_idx, rmse, r2, cor, noise, sigF, l, alpha,
                                                             avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                                             ratio_trAvg_tsAvg, ratio_trAvg_final]
                        place_counter += 1

        # SAVES DATA TO CSV-FILES --------------------------------------------------------------------------------------
        if self.save_csv_files == True:
            for a in range(numAlpha):
                alpha_current = alpha_range[a]
                folder_name = str(fig_idx) + "-alpha_" + str(alpha_current)
                os.mkdir(folder_name)
                os.chdir(folder_name)

                error_array_alpha = error_array[:, :, :, a]
                r2_array_alpha = r2_array[:, :, :, a]
                tr_error_array_alpha = tr_error_array[:, :, :, a]
                tr_r2_array_alpha = tr_r2_array[:, :, :, a]
                ratio_trAvg_tsAvg_array_alpha = ratio_trAvg_tsAvg_array[:, :, :, a]
                ratio_trAvg_array_final_alpha = ratio_trAvg_final_array[:, :, :, a]

                for n in range(numNoise):
                    noise_current = noise_range[n]
                    folder_name = str(fig_idx) + "-noise_" + str(noise_current)
                    os.mkdir(folder_name)
                    os.chdir(folder_name)

                    error_array_current = error_array_alpha[n, :, :]
                    r2_array_current = r2_array_alpha[n, :, :]
                    tr_error_array_current = tr_error_array_alpha[n, :, :]
                    tr_r2_array_current = tr_r2_array_alpha[n, :, :]
                    ratio_trAvg_tsAvg_array_current = ratio_trAvg_tsAvg_array_alpha[n, :, :]
                    ratio_trAvg_array_final_current = ratio_trAvg_array_final_alpha[n, :, :]

                    data_zeros = np.zeros((numSigF + 1, numLength + 1))
                    error_data_df = pd.DataFrame(data=data_zeros)
                    R2_data_df = pd.DataFrame(data=data_zeros)
                    tr_error_data_df = pd.DataFrame(data=data_zeros)
                    tr_r2_data_df = pd.DataFrame(data=data_zeros)
                    trAvg_tsAvg_data_df = pd.DataFrame(data=data_zeros)
                    trAvg_final_data_df = pd.DataFrame(data=data_zeros)

                    # File Names
                    error_file_name = str(mdl_name) + " - Error Data - " + str(fig_idx) + ".csv"
                    r2_file_name = str(mdl_name) + " - R2 Data - " + str(fig_idx) + ".csv"
                    tr_error_file_name = str(mdl_name) + " - Training-Error Data - " + str(fig_idx) + ".csv"
                    tr_r2_file_name = str(mdl_name) + " - Training-R2 Data - " + str(fig_idx) + ".csv"
                    trAvg_tsAvg_file_name = str(mdl_name) + " - avgTR to avgTS - " + str(fig_idx) + ".csv"
                    trAvg_final_file_name = str(mdl_name) + " - avgTR to Final - " + str(fig_idx) + ".csv"

                    # Error Data
                    error_data_df.iloc[0, 1:] = lengthNames
                    error_data_df.iloc[1:, 0] = sigFNames
                    error_data_df.iloc[1:, 1:] = error_array_current
                    error_data_df.to_csv(error_file_name)

                    # R^2 Data
                    R2_data_df.iloc[0, 1:] = lengthNames
                    R2_data_df.iloc[1:, 0] = sigFNames
                    R2_data_df.iloc[1:, 1:] = r2_array_current
                    R2_data_df.to_csv(r2_file_name)

                    # Training-Error Data
                    tr_error_data_df.iloc[0, 1:] = lengthNames
                    tr_error_data_df.iloc[1:, 0] = sigFNames
                    tr_error_data_df.iloc[1:, 1:] = tr_error_array_current
                    tr_error_data_df.to_csv(tr_error_file_name)

                    # Training-R^2 Data
                    tr_r2_data_df.iloc[0, 1:] = lengthNames
                    tr_r2_data_df.iloc[1:, 0] = sigFNames
                    tr_r2_data_df.iloc[1:, 1:] = tr_r2_array_current
                    tr_r2_data_df.to_csv(tr_r2_file_name)

                    # Average training to average testing error
                    trAvg_tsAvg_data_df.iloc[0, 1:] = lengthNames
                    trAvg_tsAvg_data_df.iloc[1:, 0] = sigFNames
                    trAvg_tsAvg_data_df.iloc[1:, 1:] = ratio_trAvg_tsAvg_array_current
                    trAvg_tsAvg_data_df.to_csv(trAvg_tsAvg_file_name)

                    # Average training to final error
                    trAvg_final_data_df.iloc[0, 1:] = lengthNames
                    trAvg_final_data_df.iloc[1:, 0] = sigFNames
                    trAvg_final_data_df.iloc[1:, 1:] = ratio_trAvg_array_final_current
                    trAvg_final_data_df.to_csv(trAvg_final_file_name)

                    os.chdir('..')
                os.chdir('..')
        return error_array, storage_df

    def runFullGridSearch_GPR_Noise_SigF_Length_Alpha(self, noise_input_data, sigF_input_data, length_input_data, alpha_input_data):

        HP_data = [[noise_input_data, sigF_input_data, length_input_data, alpha_input_data]]

        fig_idx_list = self.figureNumbers(self.numLayers, self.numZooms)

        final_storage_df_list = []

        fig_counter = 0
        for layer in range(self.numLayers):
            HP_layer_data = []
            for position_idx in range(len(HP_data)):
                print("Figure: ", fig_idx_list[fig_counter])
                print("Current Layer: ", layer + 1)
                print("Current Position: ", position_idx + 1)

                HP_data_current = HP_data[position_idx]

                # Sets up C and Epsilon ranges
                noise_data = HP_data_current[0]
                sigF_data = HP_data_current[1]
                length_data = HP_data_current[2]
                alpha_data = HP_data_current[3]
                noise_range = np.linspace(noise_data[0], noise_data[1], self.gridLength)
                sigF_range = np.linspace(sigF_data[0], sigF_data[1], self.gridLength)
                length_range = np.linspace(length_data[0], length_data[1], self.gridLength)
                alpha_range = np.linspace(alpha_data[0], alpha_data[1], self.gridLength)

                # Runs the heatmap with the current Noise, Sigma_F, & Scale-Length ranges
                error_matrix, storage_df_temp = self.runSingleGridSearch_GPR_Noise_SigF_Length_Alpha(noise_range, sigF_range, length_range, alpha_range,
                                                                             fig_idx_list[fig_counter])

                # Gets the new ranges from the error-matrix
                HP_new_list, Index_Values = self.determineTopHPs_4HP(error_matrix, noise_range, sigF_range, length_range, alpha_range, self.numZooms)

                if layer == self.numLayers - 1:
                    final_storage_df_list.append(storage_df_temp)

                for i in HP_new_list:
                    HP_layer_data.append(i)

                fig_counter += 1

            HP_data = HP_layer_data

        # Initalizes the final storage of the data on the final layer
        storage_df_current = pd.DataFrame(columns=['Figure', 'RMSE', 'R^2', 'Cor', 'Noise', 'Sigma_F', 'Length', 'Alpha',
                                                   'Average Training-RMSE', 'Average Training-R^2',
                                                   'Average Training-Cor',
                                                   'avgTR to avgTS', 'avgTR to Final Error'])

        for i in range(len(final_storage_df_list)):
            storage_df_new = pd.concat([storage_df_current, final_storage_df_list[i]])
            storage_df_current = storage_df_new

        storage_df_unsorted = storage_df_current
        storage_df_sorted = storage_df_unsorted.sort_values(by=["RMSE"], ascending=True)
        if self.save_csv_files == True:
            storage_df_unsorted.to_csv('0-Data_' + str(self.mdl_name) + '_Unsorted.csv')
            storage_df_sorted.to_csv('0-Data_' + str(self.mdl_name) + '_Sorted.csv')

        return storage_df_unsorted, storage_df_sorted

    """ TRY 3: With Active Learning """

    def find_pred_values_SVM_C_Epsilon(self, C_range, epsilon_range, fig_idx):

        int_calc_precent = 0.1
        top_percent_pts = 0.1
        num_HPs = 2

        # INPUTS FROM CLASS ATRIBUTES ----------------------------------------------------------------------------------
        X_use = self.X_inp
        Y_use = self.Y_inp
        removeNaN_var = self.RemoveNaN_var
        goodIDs = self.goodIDs
        seed = self.seed
        models_use = self.models_use
        mdl_name = self.mdl_name

        """ PART I: Initial Calculations"""
        # Create full grid of hyperparameter space
        numC = len(C_range)
        numE = len(epsilon_range)

        """
        Npts = numC*numE
        Npts_calc = int(Npts*int_calc_precent)
        calc_pts_list = []
        count = 0
        while count < Npts_calc:
            rand_num = random.randint(0, Npts)
            if rand_num not in calc_pts_list:
                calc_pts_list.append(rand_num)
                count += 1
            else:
                pass
        calc_pts_list.sort()
        """

        Npts = numC * numE
        num_points = Npts*int_calc_precent

        indices = np.linspace(0, 19, int(num_points ** (1/num_HPs)), dtype=int)
        points = [(i, j) for i in indices for j in indices]
        Npts_calc = len(points)

        model_inputs = np.zeros((Npts_calc, 2))
        model_outputs = np.zeros((Npts_calc, 1))

        #HP_space = np.zeros((numC, numE))
        counter = 0
        pos_var = 0
        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                epsilon = epsilon_range[e_idx]

                #if counter in calc_pts_list:
                if tuple((C_idx, e_idx)) in points:
                    reg = Regression(X_use, Y_use,
                                     C=C, epsilon=epsilon,
                                     Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                     RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use, giveKFdata=True)
                    results, bestPred, kFold_data = reg.RegressionCVMult()

                    error = float(results['rmse'].loc[str(mdl_name)])

                    model_inputs[pos_var, 0] = C
                    model_inputs[pos_var, 1] = epsilon
                    model_outputs[pos_var, 0] = error

                    pos_var += 1
                    print("POS: ", pos_var)

                counter += 1

        kernel_use = Matern(nu=3/2)
        model = gaussian_process.GaussianProcessRegressor(kernel=kernel_use)
        model.fit(model_inputs, model_outputs)

        """ PART II: Predictions """
        pred_error_array = np.zeros((numC, numE))
        pred_std_array = np.zeros((numC, numE))
        pred_min_error_array = np.zeros((numC, numE))

        X_ts = np.zeros((1, 2))
        pred_error_df = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon'])
        counter = 0
        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                epsilon = epsilon_range[e_idx]

                X_ts[0, :] = [C, epsilon]
                error_pred, error_std = model.predict(X_ts, return_std=True)
                min_error = (error_pred[0] - error_std[0]*0.1)

                pred_error_df.loc[counter] = [min_error, C, epsilon]
                counter += 1

                pred_error_array[C_idx, e_idx] = error_pred
                pred_std_array[C_idx, e_idx] = error_std
                pred_min_error_array[C_idx, e_idx] = min_error


        pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)

        """ PART III: Running best points """
        num_calc_points = int(Npts*top_percent_pts)

        df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                        'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                        'avgTR to avgTS', 'avgTR to Final Error']
        df_numRows = num_calc_points
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        for pt in range(num_calc_points):
            C = pred_error_df_sorted.iloc[pt, 1]
            epsilon = pred_error_df_sorted.iloc[pt, 2]

            reg = Regression(X_use, Y_use,
                             C=C, epsilon=epsilon,
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
            storage_df.loc[pt] = [fig_idx, rmse, r2, cor, C, epsilon, 'N/A', 'N/A',
                                                 avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                                 ratio_trAvg_tsAvg, ratio_trAvg_final]

            print("C: " + str(C) + ", e: " + str(epsilon) + ", ceof0: " + str('N/A') + ", gamma: " + str(
                'N/A') + " (" + str(pt+1) + "/" + str(num_calc_points) + ")" + " || Error: " + str(
                error) + ", R^2: " + str(r2))

        storage_sorted_df = storage_df.sort_values(by=['RMSE'], ascending=True)

        return storage_sorted_df, [pred_error_array, pred_std_array, pred_min_error_array]

    def runSingleGridSearch_AL_SVM_C_Epsilon(self, C_range, epsilon_range, fig_idx):
        num_HPs = 2

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

        """ PART I: Initial Calculations """
        numC = len(C_range)
        numE = len(epsilon_range)

        Npts = numC*numE
        num_points = Npts*decimal_points_int

        indices = np.linspace(0, 19, int(num_points ** (1 / num_HPs)), dtype=int)
        points = [(i, j) for i in indices for j in indices]
        Npts_calc = len(points)

        model_inputs = np.zeros((Npts_calc, 2))
        model_outputs = np.zeros((Npts_calc, 1))

        count = 0
        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                epsilon = epsilon_range[e_idx]

                index_current = tuple((C_idx, e_idx))
                if index_current in indices:
                    reg = Regression(X_use, Y_use,
                                     C=C, epsilon=epsilon,
                                     Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                     RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use, giveKFdata=True)
                    results, bestPred, kFold_data = reg.RegressionCVMult()

                    error = float(results['rmse'].loc[str(mdl_name)])

                    model_inputs[count, 0] = C
                    model_inputs[count, 1] = epsilon
                    model_outputs[count, 0] = error

                    count += 1
                    print("Count: ", count)

        kernel_use = Matern(nu=3 / 2)
        model = gaussian_process.GaussianProcessRegressor(kernel=kernel_use)
        model.fit(model_inputs, model_outputs)

        """ PART II: Predictions """
        pred_error_array = np.zeros((numC, numE))
        pred_std_array = np.zeros((numC, numE))
        pred_min_error_array = np.zeros((numC, numE))

        X_ts = np.zeros((1, 2))
        pred_error_df = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon'])
        counter = 0
        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                epsilon = epsilon_range[e_idx]

                X_ts[0, :] = [C, epsilon]
                error_pred, error_std = model.predict(X_ts, return_std=True)
                min_error = (error_pred[0] - error_std[0] * 0.1)

                pred_error_df.loc[counter] = [min_error, C, epsilon]
                counter += 1

                pred_error_array[C_idx, e_idx] = error_pred
                pred_std_array[C_idx, e_idx] = error_std
                pred_min_error_array[C_idx, e_idx] = min_error

        pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)

        """ PART III: Running best points """
        num_calc_points = int(Npts * decimal_points_top)

        df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                        'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                        'avgTR to avgTS', 'avgTR to Final Error']
        df_numRows = num_calc_points
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        for pt in range(num_calc_points):
            C = pred_error_df_sorted.iloc[pt, 1]
            epsilon = pred_error_df_sorted.iloc[pt, 2]

            reg = Regression(X_use, Y_use,
                             C=C, epsilon=epsilon,
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
            storage_df.loc[pt] = [fig_idx, rmse, r2, cor, C, epsilon, 'N/A', 'N/A',
                                  avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                  ratio_trAvg_tsAvg, ratio_trAvg_final]

            print("C: " + str(C) + ", e: " + str(epsilon) + ", ceof0: " + str('N/A') + ", gamma: " + str(
                'N/A') + " (" + str(pt + 1) + "/" + str(num_calc_points) + ")" + " || Error: " + str(
                error) + ", R^2: " + str(r2))

        storage_sorted_df = storage_df.sort_values(by=['RMSE'], ascending=True)

        """ TEMP: Create 'Error Array' """
        """
        This should only be thought of as a temporay fix as to not have to replace the current
        'determineTopHPs_XHP' functions. To do so, an array is created that holds double the maximum
        error from the 'storage_df' error values, and then the values that are found in PART III are put
        in their proper place in this array. This is done to ensure that the top values (the ones that would
        be picked up by the 'determineTopHPs_XHP') are still the best ones in this new array
        """
        rmse_list = list(storage_df['RMSE'])
        max_rmse = np.max(rmse_list)
        rmse_bad_input = 2 * max_rmse

        # Create list of good HP-pairs
        tup_list = []
        num_rows_df = len(storage_df['RMSE'])
        for i in range(num_rows_df):
            C_temp = storage_df['C'][i]
            e_temp = storage_df['Epsilon'][i]
            tup_list.append(tuple((C_temp, e_temp)))

        error_array = np.zeros((numC, numE))

        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                e = epsilon_range[e_idx]

                if tuple((C, e)) not in tup_list:
                    error_temp = rmse_bad_input
                else:
                    error_temp = float(storage_df.loc[(storage_df['C'] == C) & (storage_df['Epsilon'] == e)]['RMSE'])

                error_array[C_idx, e_idx] = error_temp


        return error_array, storage_sorted_df



    # CASE I: SVM - C, Epsilon -- Linear
    def runSingleGridSearch_AL_SVM_C_Epsilon(self, C_range, epsilon_range, fig_idx):
        num_HPs = 2

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

        """ PART I: Initial Calculations """
        numC = len(C_range)
        numE = len(epsilon_range)

        Npts = numC * numE
        num_points = Npts * decimal_points_int

        indices = np.linspace(0, 19, int(num_points ** (1 / num_HPs)), dtype=int)
        points = [(i, j) for i in indices for j in indices]
        Npts_calc = len(points)

        model_inputs = np.zeros((Npts_calc, 2))
        model_outputs = np.zeros((Npts_calc, 1))

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
                                     RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use, giveKFdata=True)
                    results, bestPred, kFold_data = reg.RegressionCVMult()

                    error = float(results['rmse'].loc[str(mdl_name)])

                    model_inputs[count, 0] = C
                    model_inputs[count, 1] = epsilon
                    model_outputs[count, 0] = error

                    count += 1
                    print("Count: ", count)

        kernel_use = Matern(nu=3 / 2)
        model = gaussian_process.GaussianProcessRegressor(kernel=kernel_use)
        model.fit(model_inputs, model_outputs)

        """ PART II: Predictions """
        pred_error_array = np.zeros((numC, numE))
        pred_std_array = np.zeros((numC, numE))
        pred_min_error_array = np.zeros((numC, numE))

        X_ts = np.zeros((1, 2))
        pred_error_df = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon'])
        counter = 0
        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                epsilon = epsilon_range[e_idx]

                X_ts[0, :] = [C, epsilon]
                error_pred, error_std = model.predict(X_ts, return_std=True)
                min_error = (error_pred[0] - error_std[0] * 0.1)

                pred_error_df.loc[counter] = [min_error, C, epsilon]
                counter += 1

                pred_error_array[C_idx, e_idx] = error_pred
                pred_std_array[C_idx, e_idx] = error_std
                pred_min_error_array[C_idx, e_idx] = min_error

        pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)

        """ PART III: Running best points """
        num_calc_points = int(Npts * decimal_points_top)

        df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                        'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                        'avgTR to avgTS', 'avgTR to Final Error']
        df_numRows = num_calc_points
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        for pt in range(num_calc_points):
            C = pred_error_df_sorted.iloc[pt, 1]
            epsilon = pred_error_df_sorted.iloc[pt, 2]

            reg = Regression(X_use, Y_use,
                             C=C, epsilon=epsilon,
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
            storage_df.loc[pt] = [fig_idx, rmse, r2, cor, C, epsilon, 'N/A', 'N/A',
                                  avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                  ratio_trAvg_tsAvg, ratio_trAvg_final]

            print("C: " + str(C) + ", e: " + str(epsilon) + ", ceof0: " + str('N/A') + ", gamma: " + str(
                'N/A') + " (" + str(pt + 1) + "/" + str(num_calc_points) + ")" + " || Error: " + str(
                error) + ", R^2: " + str(r2))

        storage_sorted_df = storage_df.sort_values(by=['RMSE'], ascending=True)

        """ TEMP: Create 'Error Array' """
        """
        This should only be thought of as a temporay fix as to not have to replace the current
        'determineTopHPs_XHP' functions. To do so, an array is created that holds double the maximum
        error from the 'storage_df' error values, and then the values that are found in PART III are put
        in their proper place in this array. This is done to ensure that the top values (the ones that would
        be picked up by the 'determineTopHPs_XHP') are still the best ones in this new array
        """
        rmse_list = list(storage_df['RMSE'])
        max_rmse = np.max(rmse_list)
        rmse_bad_input = 2 * max_rmse

        # Create list of good HP-pairs
        tup_list = []
        num_rows_df = len(storage_df['RMSE'])
        for i in range(num_rows_df):
            C_temp = storage_df['C'][i]
            e_temp = storage_df['Epsilon'][i]
            tup_list.append(tuple((C_temp, e_temp)))

        error_array = np.zeros((numC, numE))

        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                e = epsilon_range[e_idx]

                if tuple((C, e)) not in tup_list:
                    error_temp = rmse_bad_input
                else:
                    error_temp = float(storage_df.loc[(storage_df['C'] == C) & (storage_df['Epsilon'] == e)]['RMSE'])

                error_array[C_idx, e_idx] = error_temp

        return error_array, storage_sorted_df

    def runFullGridSearch_AL_SVM_C_Epsilon(self, C_input_data, epsilon_input_data):

        HP_data = [[C_input_data, epsilon_input_data]]

        fig_idx_list = self.figureNumbers(self.numLayers, self.numZooms)

        final_storage_df_list = []

        fig_counter = 0
        for layer in range(self.numLayers):
            HP_layer_data = []
            for position_idx in range(len(HP_data)):
                print("Figure: ", fig_idx_list[fig_counter])
                print("Current Layer: ", layer + 1)
                print("Current Position: ", position_idx + 1)

                HP_data_current = HP_data[position_idx]

                # Sets up C and Epsilon ranges
                C_data = HP_data_current[0]
                epsilon_data = HP_data_current[1]
                C_range = np.linspace(C_data[0], C_data[1], self.gridLength)
                epsilon_range = np.linspace(epsilon_data[0], epsilon_data[1], self.gridLength)

                # Runs the heatmap with the current C & Epsilon ranges
                error_matrix, storage_df_temp = self.runSingleGridSearch_AL_SVM_C_Epsilon(C_range, epsilon_range,
                                                                                       fig_idx_list[fig_counter])

                # Gets the new ranges from the error-matrix
                HP_new_list, Index_Values = self.determineTopHPs_2HP(error_matrix, C_range, epsilon_range,
                                                                     self.numZooms)

                if layer == self.numLayers - 1:
                    final_storage_df_list.append(storage_df_temp)

                for i in HP_new_list:
                    HP_layer_data.append(i)

                fig_counter += 1

            HP_data = HP_layer_data

        # Initalizes the final storage of the data on the final layer
        storage_df_current = pd.DataFrame(columns=['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                                                   'Average Training-RMSE', 'Average Training-R^2',
                                                   'Average Training-Cor',
                                                   'avgTR to avgTS', 'avgTR to Final Error'])
        for i in range(len(final_storage_df_list)):
            storage_df_new = pd.concat([storage_df_current, final_storage_df_list[i]])
            storage_df_current = storage_df_new

        storage_df_unsorted = storage_df_current
        storage_df_sorted = storage_df_unsorted.sort_values(by=["RMSE"], ascending=True)
        if self.save_csv_files == True:
            storage_df_unsorted.to_csv('0-Data_' + str(self.mdl_name) + '_Unsorted.csv')
            storage_df_sorted.to_csv('0-Data_' + str(self.mdl_name) + '_Sorted.csv')

        return storage_df_unsorted, storage_df_sorted

    # CASE II: SVM - C, Epsilon, Gamma -- RBF
    def runSingleGridSearch_AL_SVM_C_Epsilon_Gamma(self, C_range, epsilon_range, gamma_range, fig_idx):
        num_HPs = 3

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

        """ PART I: Initial Calculations """
        numC = len(C_range)
        numE = len(epsilon_range)
        numG = len(gamma_range)

        Npts = numC * numE * numG
        num_points = Npts * decimal_points_int

        indices = np.linspace(0, 19, int(num_points ** (1 / num_HPs)), dtype=int)
        points = [(i, j, k) for i in indices for j in indices for k in indices]
        Npts_calc = len(points)

        model_inputs = np.zeros((Npts_calc, 3))
        model_outputs = np.zeros((Npts_calc, 1))

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
                                         RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use, giveKFdata=True)
                        results, bestPred, kFold_data = reg.RegressionCVMult()

                        error = float(results['rmse'].loc[str(mdl_name)])

                        model_inputs[count, 0] = C
                        model_inputs[count, 1] = epsilon
                        model_inputs[count, 2] = gamma
                        model_outputs[count, 0] = error

                        count += 1
                        print("Count: ", count)

        kernel_use = Matern(nu=3 / 2)
        model = gaussian_process.GaussianProcessRegressor(kernel=kernel_use)
        model.fit(model_inputs, model_outputs)

        """ PART II: Predictions """
        pred_error_array = np.zeros((numC, numE, numG))
        pred_std_array = np.zeros((numC, numE, numG))
        pred_min_error_array = np.zeros((numC, numE, numG))

        X_ts = np.zeros((1, 3))
        pred_error_df = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma'])
        counter = 0
        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                epsilon = epsilon_range[e_idx]
                for g_idx in range(numG):
                    gamma = gamma_range[g_idx]

                    X_ts[0, :] = [C, epsilon, gamma]
                    error_pred, error_std = model.predict(X_ts, return_std=True)
                    min_error = (error_pred[0] - error_std[0] * 0.1)

                    pred_error_df.loc[counter] = [min_error, C, epsilon, gamma]
                    counter += 1

                    pred_error_array[C_idx, e_idx, g_idx] = error_pred
                    pred_std_array[C_idx, e_idx, g_idx] = error_std
                    pred_min_error_array[C_idx, e_idx, g_idx] = min_error

        pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)

        """ PART III: Running best points """
        num_calc_points = int(Npts * decimal_points_top)

        df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                        'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                        'avgTR to avgTS', 'avgTR to Final Error']
        df_numRows = num_calc_points
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        for pt in range(num_calc_points):
            C = pred_error_df_sorted.iloc[pt, 1]
            epsilon = pred_error_df_sorted.iloc[pt, 2]
            gamma = pred_error_df_sorted.iloc[pt, 3]

            reg = Regression(X_use, Y_use,
                             C=C, epsilon=epsilon, gamma=gamma,
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
            storage_df.loc[pt] = [fig_idx, rmse, r2, cor, C, epsilon, 'N/A', gamma,
                                  avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                  ratio_trAvg_tsAvg, ratio_trAvg_final]

            print("C: " + str(C) + ", e: " + str(epsilon) + ", ceof0: " + str('N/A') + ", gamma: " + str(
                gamma) + " (" + str(pt + 1) + "/" + str(num_calc_points) + ")" + " || Error: " + str(
                error) + ", R^2: " + str(r2))

        storage_sorted_df = storage_df.sort_values(by=['RMSE'], ascending=True)

        """ TEMP: Create 'Error Array' """
        """
        This should only be thought of as a temporay fix as to not have to replace the current
        'determineTopHPs_XHP' functions. To do so, an array is created that holds double the maximum
        error from the 'storage_df' error values, and then the values that are found in PART III are put
        in their proper place in this array. This is done to ensure that the top values (the ones that would
        be picked up by the 'determineTopHPs_XHP') are still the best ones in this new array
        """
        rmse_list = list(storage_df['RMSE'])
        max_rmse = np.max(rmse_list)
        rmse_bad_input = 2 * max_rmse

        # Create list of good HP-pairs
        tup_list = []
        num_rows_df = len(storage_df['RMSE'])
        for i in range(num_rows_df):
            C_temp = storage_df['C'][i]
            e_temp = storage_df['Epsilon'][i]
            g_temp = storage_df['Gamma'][i]
            tup_list.append(tuple((C_temp, e_temp, g_temp)))

        error_array = np.zeros((numC, numE, numG))

        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                e = epsilon_range[e_idx]
                for g_idx in range(numG):
                    gamma = gamma_range[g_idx]

                    if tuple((C, e, gamma)) not in tup_list:
                        error_temp = rmse_bad_input
                    else:
                        error_temp = float(storage_df.loc[(storage_df['C'] == C) & (storage_df['Epsilon'] == e) & (storage_df['Gamma'] == gamma)]['RMSE'])

                    error_array[C_idx, e_idx, g_idx] = error_temp

        return error_array, storage_sorted_df

    def runFullGridSearch_AL_SVM_C_Epsilon_Gamma(self, C_input_data, epsilon_input_data, gamma_input_data):

        HP_data = [[C_input_data, epsilon_input_data, gamma_input_data]]

        fig_idx_list = self.figureNumbers(self.numLayers, self.numZooms)

        final_storage_df_list = []

        # GETS GAMMA VALUE ---------------------------------------------------------------------------------------------
        goodIds = self.goodIDs
        Y_inp = self.Y_inp
        Y_use = [Y_inp[i] for i in range(len(Y_inp)) if goodIds[i]]

        fig_counter = 0
        for layer in range(self.numLayers):
            HP_layer_data = []
            for position_idx in range(len(HP_data)):
                print("Figure: ", fig_idx_list[fig_counter])
                print("Current Layer: ", layer + 1)
                print("Current Position: ", position_idx + 1)

                HP_data_current = HP_data[position_idx]

                # Sets up C and Epsilon ranges
                C_data = HP_data_current[0]
                epsilon_data = HP_data_current[1]
                gamma_data = HP_data_current[2]
                C_range = np.linspace(C_data[0], C_data[1], self.gridLength)
                epsilon_range = np.linspace(epsilon_data[0], epsilon_data[1], self.gridLength)
                gamma_range = np.linspace(gamma_data[0], gamma_data[1], self.gridLength)

                # Runs the heatmap with the current C & Epsilon ranges
                error_matrix, storage_df_temp = self.runSingleGridSearch_AL_SVM_C_Epsilon_Gamma(C_range, epsilon_range,
                                                                                             gamma_range,
                                                                                             fig_idx_list[
                                                                                                 fig_counter])

                # Gets the new ranges from the error-matrix
                HP_new_list, Index_Values = self.determineTopHPs_3HP(error_matrix, C_range, epsilon_range,
                                                                     gamma_range,
                                                                     self.numZooms)

                if layer == self.numLayers - 1:
                    final_storage_df_list.append(storage_df_temp)

                for i in HP_new_list:
                    HP_layer_data.append(i)

                fig_counter += 1

            HP_data = HP_layer_data

        # Initalizes the final storage of the data on the final layer
        storage_df_current = pd.DataFrame(columns=['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                                                   'Average Training-RMSE', 'Average Training-R^2',
                                                   'Average Training-Cor',
                                                   'avgTR to avgTS', 'avgTR to Final Error'])
        for i in range(len(final_storage_df_list)):
            storage_df_new = pd.concat([storage_df_current, final_storage_df_list[i]])
            storage_df_current = storage_df_new

        storage_df_unsorted = storage_df_current
        storage_df_sorted = storage_df_unsorted.sort_values(by=["RMSE"], ascending=True)
        if self.save_csv_files == True:
            storage_df_unsorted.to_csv('0-Data_' + str(self.mdl_name) + '_Unsorted.csv')
            storage_df_sorted.to_csv('0-Data_' + str(self.mdl_name) + '_Sorted.csv')

        return storage_df_unsorted, storage_df_sorted

    # CASE III: SVM - C, Epsilon, Gamma, Coef0 -- Poly2, Poly3
    def runSingleGridSearch_AL_SVM_C_Epsilon_Gamma_Coef0(self, C_range, epsilon_range, gamma_range, coef0_range, fig_idx):
        num_HPs = 4

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

        """ PART I: Initial Calculations """
        numC = len(C_range)
        numE = len(epsilon_range)
        numG = len(gamma_range)
        numC0 = len(coef0_range)

        Npts = numC * numE * numG * numC0
        num_points = Npts * decimal_points_int

        indices = np.linspace(0, 19, int(num_points ** (1 / num_HPs)), dtype=int)
        points = [(i, j, k, l) for i in indices for j in indices for k in indices for l in indices]
        Npts_calc = len(points)

        model_inputs = np.zeros((Npts_calc, 4))
        model_outputs = np.zeros((Npts_calc, 1))

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
                            print("Count: ", count)

        kernel_use = Matern(nu=3 / 2)
        model = gaussian_process.GaussianProcessRegressor(kernel=kernel_use)
        model.fit(model_inputs, model_outputs)

        """ PART II: Predictions """
        pred_error_array = np.zeros((numC, numE, numG, numC0))
        pred_std_array = np.zeros((numC, numE, numG, numC0))
        pred_min_error_array = np.zeros((numC, numE, numG, numC0))

        X_ts = np.zeros((1, 4))
        pred_error_df = pd.DataFrame(columns=['Min-Error', 'C', 'Epsilon', 'Gamma', 'Coef0'])
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
                        min_error = (error_pred[0] - error_std[0] * 0.1)

                        pred_error_df.loc[counter] = [min_error, C, epsilon, gamma, c0]
                        counter += 1

                        pred_error_array[C_idx, e_idx, g_idx, c0_idx] = error_pred
                        pred_std_array[C_idx, e_idx, g_idx, c0_idx] = error_std
                        pred_min_error_array[C_idx, e_idx, g_idx, c0_idx] = min_error

        pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)

        """ PART III: Running best points """
        num_calc_points = int(Npts * decimal_points_top)

        df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                        'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                        'avgTR to avgTS', 'avgTR to Final Error']
        df_numRows = num_calc_points
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        for pt in range(num_calc_points):
            C = pred_error_df_sorted.iloc[pt, 1]
            epsilon = pred_error_df_sorted.iloc[pt, 2]
            gamma = pred_error_df_sorted.iloc[pt, 3]
            c0 = pred_error_df_sorted.iloc[pt, 4]

            reg = Regression(X_use, Y_use,
                             C=C, epsilon=epsilon, gamma=gamma, coef0=c0,
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
            storage_df.loc[pt] = [fig_idx, rmse, r2, cor, C, epsilon, c0, gamma,
                                  avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                  ratio_trAvg_tsAvg, ratio_trAvg_final]

            print("C: " + str(C) + ", e: " + str(epsilon) + ", ceof0: " + str(c0) + ", gamma: " + str(
                gamma) + " (" + str(pt + 1) + "/" + str(num_calc_points) + ")" + " || Error: " + str(
                error) + ", R^2: " + str(r2))

        storage_sorted_df = storage_df.sort_values(by=['RMSE'], ascending=True)

        """ TEMP: Create 'Error Array' """
        """
        This should only be thought of as a temporay fix as to not have to replace the current
        'determineTopHPs_XHP' functions. To do so, an array is created that holds double the maximum
        error from the 'storage_df' error values, and then the values that are found in PART III are put
        in their proper place in this array. This is done to ensure that the top values (the ones that would
        be picked up by the 'determineTopHPs_XHP') are still the best ones in this new array
        """
        rmse_list = list(storage_df['RMSE'])
        max_rmse = np.max(rmse_list)
        rmse_bad_input = 2 * max_rmse

        # Create list of good HP-pairs
        tup_list = []
        num_rows_df = len(storage_df['RMSE'])
        for i in range(num_rows_df):
            C_temp = storage_df['C'][i]
            e_temp = storage_df['Epsilon'][i]
            g_temp = storage_df['Gamma'][i]
            c0_temp = storage_df['Coef0'][i]
            tup_list.append(tuple((C_temp, e_temp, g_temp, c0_temp)))

        error_array = np.zeros((numC, numE, numG, numC0))

        for C_idx in range(numC):
            C = C_range[C_idx]
            for e_idx in range(numE):
                e = epsilon_range[e_idx]
                for g_idx in range(numG):
                    gamma = gamma_range[g_idx]
                    for c0_idx in range(numC0):
                        c0 = coef0_range[c0_idx]

                        if tuple((C, e, gamma, c0)) not in tup_list:
                            error_temp = rmse_bad_input
                        else:
                            error_temp = float(storage_df.loc[(storage_df['C'] == C) & (storage_df['Epsilon'] == e) & (
                                        storage_df['Gamma'] == gamma) & (storage_df['Coef0'] == c0)]['RMSE'])

                        error_array[C_idx, e_idx, g_idx, c0_idx] = error_temp

        return error_array, storage_sorted_df

    def runFullGridSearch_AL_SVM_C_Epsilon_Gamma_Coef0(self, C_input_data, epsilon_input_data, gamma_input_data,
                                                    coef0_input_data):

        HP_data = [[C_input_data, epsilon_input_data, gamma_input_data, coef0_input_data]]

        fig_idx_list = self.figureNumbers(self.numLayers, self.numZooms)

        final_storage_df_list = []

        # GETS GAMMA VALUE ---------------------------------------------------------------------------------------------
        goodIds = self.goodIDs
        Y_inp = self.Y_inp
        Y_use = [Y_inp[i] for i in range(len(Y_inp)) if goodIds[i]]

        fig_counter = 0
        for layer in range(self.numLayers):
            HP_layer_data = []
            for position_idx in range(len(HP_data)):
                print("Figure: ", fig_idx_list[fig_counter])
                print("Current Layer: ", layer + 1)
                print("Current Position: ", position_idx + 1)

                HP_data_current = HP_data[position_idx]

                # Sets up C and Epsilon ranges
                C_data = HP_data_current[0]
                epsilon_data = HP_data_current[1]
                gamma_data = HP_data_current[2]
                coef0_data = HP_data_current[3]
                C_range = np.linspace(C_data[0], C_data[1], self.gridLength)
                epsilon_range = np.linspace(epsilon_data[0], epsilon_data[1], self.gridLength)
                gamma_range = np.linspace(gamma_data[0], gamma_data[1], self.gridLength)
                coef0_range = np.linspace(coef0_data[0], coef0_data[1], self.gridLength)

                # Runs the heatmap with the current C & Epsilon ranges
                error_matrix, storage_df_temp = self.runSingleGridSearch_AL_SVM_C_Epsilon_Gamma_Coef0(C_range,
                                                                                                   epsilon_range,
                                                                                                   gamma_range,
                                                                                                   coef0_range,
                                                                                                   fig_idx_list[
                                                                                                       fig_counter])

                # Gets the new ranges from the error-matrix
                HP_new_list, Index_Values = self.determineTopHPs_4HP(error_matrix, C_range, epsilon_range,
                                                                     gamma_range, coef0_range,
                                                                     self.numZooms)

                if layer == self.numLayers - 1:
                    final_storage_df_list.append(storage_df_temp)

                for i in HP_new_list:
                    HP_layer_data.append(i)

                fig_counter += 1

            HP_data = HP_layer_data

        # Initalizes the final storage of the data on the final layer
        storage_df_current = pd.DataFrame(columns=['Figure', 'RMSE', 'R^2', 'Cor', 'C', 'Epsilon', 'Coef0', 'Gamma',
                                                   'Average Training-RMSE', 'Average Training-R^2',
                                                   'Average Training-Cor',
                                                   'avgTR to avgTS', 'avgTR to Final Error'])
        for i in range(len(final_storage_df_list)):
            storage_df_new = pd.concat([storage_df_current, final_storage_df_list[i]])
            storage_df_current = storage_df_new

        storage_df_unsorted = storage_df_current
        storage_df_sorted = storage_df_unsorted.sort_values(by=["RMSE"], ascending=True)
        if self.save_csv_files == True:
            storage_df_unsorted.to_csv('0-Data_' + str(self.mdl_name) + '_Unsorted.csv')
            storage_df_sorted.to_csv('0-Data_' + str(self.mdl_name) + '_Sorted.csv')

        return storage_df_unsorted, storage_df_sorted

    # CASE IV: GPR - Noise, SigF, Length -- RBF, Matern 3/2, Matern 5/2
    def runSingleGridSearch_AL_GPR_Noise_SigF_Length(self, noise_range, sigF_range, length_range, fig_idx):
        num_HPs = 3

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

        """ PART I: Initial Calculations """
        numN = len(noise_range)
        numS = len(sigF_range)
        numL = len(length_range)

        Npts = numN * numS * numL
        num_points = Npts * decimal_points_int

        indices = np.linspace(0, 19, int(num_points ** (1 / num_HPs)), dtype=int)
        points = [(i, j, k) for i in indices for j in indices for k in indices]
        Npts_calc = len(points)

        model_inputs = np.zeros((Npts_calc, 3))
        model_outputs = np.zeros((Npts_calc, 1))

        count = 0
        for n_idx in range(numN):
            noise = noise_range[n_idx]
            for s_idx in range(numS):
                sigF = sigF_range[s_idx]
                for l_idx in range(numL):
                    length = length_range[l_idx]

                    index_current = tuple((n_idx, s_idx, l_idx))
                    if index_current in points:
                        reg = Regression(X_use, Y_use,
                                         noise=noise, sigma_F=sigF, scale_length=length,
                                         Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                         RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                         giveKFdata=True)
                        results, bestPred, kFold_data = reg.RegressionCVMult()

                        error = float(results['rmse'].loc[str(mdl_name)])

                        model_inputs[count, 0] = noise
                        model_inputs[count, 1] = sigF
                        model_inputs[count, 2] = length
                        model_outputs[count, 0] = error

                        count += 1
                        print("Count: ", count)

        kernel_use = Matern(nu=3 / 2)
        model = gaussian_process.GaussianProcessRegressor(kernel=kernel_use)
        model.fit(model_inputs, model_outputs)

        """ PART II: Predictions """
        pred_error_array = np.zeros((numN, numS, numL))
        pred_std_array = np.zeros((numN, numS, numL))
        pred_min_error_array = np.zeros((numN, numS, numL))

        X_ts = np.zeros((1, 3))
        pred_error_df = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length'])
        counter = 0
        for n_idx in range(numN):
            noise = noise_range[n_idx]
            for s_idx in range(numS):
                sigF = sigF_range[s_idx]
                for l_idx in range(numL):
                    length = length_range[l_idx]

                    X_ts[0, :] = [noise, sigF, length]
                    error_pred, error_std = model.predict(X_ts, return_std=True)
                    min_error = (error_pred[0] - error_std[0] * 0.1)

                    pred_error_df.loc[counter] = [min_error, noise, sigF, length]
                    counter += 1

                    pred_error_array[n_idx, s_idx, l_idx] = error_pred
                    pred_std_array[n_idx, s_idx, l_idx] = error_std
                    pred_min_error_array[n_idx, s_idx, l_idx] = min_error

        pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)

        """ PART III: Running best points """
        num_calc_points = int(Npts * decimal_points_top)

        df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'Noise', 'Sigma_F', 'Length',
                        'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                        'avgTR to avgTS', 'avgTR to Final Error']
        df_numRows = num_calc_points
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        for pt in range(num_calc_points):
            noise = pred_error_df_sorted.iloc[pt, 1]
            sigF = pred_error_df_sorted.iloc[pt, 2]
            length = pred_error_df_sorted.iloc[pt, 3]

            reg = Regression(X_use, Y_use,
                             noise=noise, sigma_F=sigF, scale_length=length,
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
            storage_df.loc[pt] = [fig_idx, rmse, r2, cor, noise, sigF, length,
                                  avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                  ratio_trAvg_tsAvg, ratio_trAvg_final]

            print("noise: " + str(noise) + ", sigF: " + str(sigF) + ", length: " + str(length) + " (" + str(
                pt+1) + "/" + str(num_calc_points) + ")")

        storage_sorted_df = storage_df.sort_values(by=['RMSE'], ascending=True)

        """ TEMP: Create 'Error Array' """
        """
        This should only be thought of as a temporay fix as to not have to replace the current
        'determineTopHPs_XHP' functions. To do so, an array is created that holds double the maximum
        error from the 'storage_df' error values, and then the values that are found in PART III are put
        in their proper place in this array. This is done to ensure that the top values (the ones that would
        be picked up by the 'determineTopHPs_XHP') are still the best ones in this new array
        """
        rmse_list = list(storage_df['RMSE'])
        max_rmse = np.max(rmse_list)
        rmse_bad_input = 2 * max_rmse

        # Create list of good HP-pairs
        tup_list = []
        num_rows_df = len(storage_df['RMSE'])
        for i in range(num_rows_df):
            noise_temp = storage_df['Noise'][i]
            sigF_temp = storage_df['Sigma_F'][i]
            length_temp = storage_df['Length'][i]
            tup_list.append(tuple((noise_temp, sigF_temp, length_temp)))

        error_array = np.zeros((numN, numS, numL))

        for n_idx in range(numN):
            noise = noise_range[n_idx]
            for s_idx in range(numS):
                sigF = sigF_range[s_idx]
                for l_idx in range(numL):
                    length = length_range[l_idx]

                    if tuple((noise, sigF, length)) not in tup_list:
                        error_temp = rmse_bad_input
                    else:
                        error_temp = float(storage_df.loc[(storage_df['Noise'] == noise) & (storage_df['Sigma_F'] == sigF) & (
                                    storage_df['Length'] == length)]['RMSE'])

                    error_array[n_idx, s_idx, l_idx] = error_temp

        return error_array, storage_sorted_df

    def runFullGridSearch_AL_GPR_Noise_SigF_Length(self, noise_input_data, sigF_input_data, length_input_data):

        HP_data = [[noise_input_data, sigF_input_data, length_input_data]]

        fig_idx_list = self.figureNumbers(self.numLayers, self.numZooms)

        final_storage_df_list = []

        fig_counter = 0
        for layer in range(self.numLayers):
            HP_layer_data = []
            for position_idx in range(len(HP_data)):
                print("Figure: ", fig_idx_list[fig_counter])
                print("Current Layer: ", layer + 1)
                print("Current Position: ", position_idx + 1)

                HP_data_current = HP_data[position_idx]

                # Sets up C and Epsilon ranges
                noise_data = HP_data_current[0]
                sigF_data = HP_data_current[1]
                length_data = HP_data_current[2]
                noise_range = np.linspace(noise_data[0], noise_data[1], self.gridLength)
                sigF_range = np.linspace(sigF_data[0], sigF_data[1], self.gridLength)
                length_range = np.linspace(length_data[0], length_data[1], self.gridLength)

                # Runs the heatmap with the current Noise, Sigma_F, & Scale-Length ranges
                error_matrix, storage_df_temp = self.runSingleGridSearch_AL_GPR_Noise_SigF_Length(noise_range,
                                                                                               sigF_range,
                                                                                               length_range,
                                                                                               fig_idx_list[
                                                                                                   fig_counter])

                # Gets the new ranges from the error-matrix
                HP_new_list, Index_Values = self.determineTopHPs_3HP(error_matrix, noise_range, sigF_range,
                                                                     length_range, self.numZooms)

                if layer == self.numLayers - 1:
                    final_storage_df_list.append(storage_df_temp)

                for i in HP_new_list:
                    HP_layer_data.append(i)

                fig_counter += 1

            HP_data = HP_layer_data

        # Initalizes the final storage of the data on the final layer
        storage_df_current = pd.DataFrame(columns=['Figure', 'RMSE', 'R^2', 'Cor', 'Noise', 'Sigma_F', 'Length',
                                                   'Average Training-RMSE', 'Average Training-R^2',
                                                   'Average Training-Cor',
                                                   'avgTR to avgTS', 'avgTR to Final Error'])

        for i in range(len(final_storage_df_list)):
            storage_df_new = pd.concat([storage_df_current, final_storage_df_list[i]])
            storage_df_current = storage_df_new

        storage_df_unsorted = storage_df_current
        storage_df_sorted = storage_df_unsorted.sort_values(by=["RMSE"], ascending=True)
        if self.save_csv_files == True:
            storage_df_unsorted.to_csv('0-Data_' + str(self.mdl_name) + '_Unsorted.csv')
            storage_df_sorted.to_csv('0-Data_' + str(self.mdl_name) + '_Sorted.csv')

        return storage_df_unsorted, storage_df_sorted

    # CASE V: GPR - Noise, SigF, Length, Alpha -- Rational Quadratic
    def runSingleGridSearch_AL_GPR_Noise_SigF_Length_Alpha(self, noise_range, sigF_range, length_range, alpha_range, fig_idx):
        num_HPs = 4

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

        """ PART I: Initial Calculations """
        numN = len(noise_range)
        numS = len(sigF_range)
        numL = len(length_range)
        numA = len(alpha_range)

        Npts = numN * numS * numL * numA
        num_points = Npts * decimal_points_int

        indices = np.linspace(0, 19, int(num_points ** (1 / num_HPs)), dtype=int)
        points = [(i, j, k, l) for i in indices for j in indices for k in indices for l in indices]
        Npts_calc = len(points)

        model_inputs = np.zeros((Npts_calc, 4))
        model_outputs = np.zeros((Npts_calc, 1))

        count = 0
        for n_idx in range(numN):
            noise = noise_range[n_idx]
            for s_idx in range(numS):
                sigF = sigF_range[s_idx]
                for l_idx in range(numL):
                    length = length_range[l_idx]
                    for a_idx in range(numA):
                        alpha = alpha_range[a_idx]

                        index_current = tuple((n_idx, s_idx, l_idx))
                        if index_current in points:
                            reg = Regression(X_use, Y_use,
                                             noise=noise, sigma_F=sigF, scale_length=length, alpha=alpha,
                                             Nk=5, N=1, goodIDs=goodIDs, seed=seed,
                                             RemoveNaN=removeNaN_var, StandardizeX=True, models_use=models_use,
                                             giveKFdata=True)
                            results, bestPred, kFold_data = reg.RegressionCVMult()

                            error = float(results['rmse'].loc[str(mdl_name)])

                            model_inputs[count, 0] = noise
                            model_inputs[count, 1] = sigF
                            model_inputs[count, 2] = length
                            model_inputs[count, 3] = alpha
                            model_outputs[count, 0] = error

                            count += 1
                            print("Count: ", count)

        kernel_use = Matern(nu=3 / 2)
        model = gaussian_process.GaussianProcessRegressor(kernel=kernel_use)
        model.fit(model_inputs, model_outputs)

        """ PART II: Predictions """
        pred_error_array = np.zeros((numN, numS, numL, numA))
        pred_std_array = np.zeros((numN, numS, numL, numA))
        pred_min_error_array = np.zeros((numN, numS, numL, numA))

        X_ts = np.zeros((1, 3))
        pred_error_df = pd.DataFrame(columns=['Min-Error', 'Noise', 'SigmaF', 'Length', 'Alpha'])
        counter = 0
        for n_idx in range(numN):
            noise = noise_range[n_idx]
            for s_idx in range(numS):
                sigF = sigF_range[s_idx]
                for l_idx in range(numL):
                    length = length_range[l_idx]
                    for a_idx in range(numA):
                        alpha = alpha_range[a_idx]

                        X_ts[0, :] = [noise, sigF, length, alpha]
                        error_pred, error_std = model.predict(X_ts, return_std=True)
                        min_error = (error_pred[0] - error_std[0] * 0.1)

                        pred_error_df.loc[counter] = [min_error, noise, sigF, length, alpha]
                        counter += 1

                        pred_error_array[n_idx, s_idx, l_idx, a_idx] = error_pred
                        pred_std_array[n_idx, s_idx, l_idx, a_idx] = error_std
                        pred_min_error_array[n_idx, s_idx, l_idx, a_idx] = min_error

        pred_error_df_sorted = pred_error_df.sort_values(by=['Min-Error'], ascending=True)

        """ PART III: Running best points """
        num_calc_points = int(Npts * decimal_points_top)

        df_col_names = ['Figure', 'RMSE', 'R^2', 'Cor', 'Noise', 'Sigma_F', 'Length', 'Alpha',
                        'Average Training-RMSE', 'Average Training-R^2', 'Average Training-Cor',
                        'avgTR to avgTS', 'avgTR to Final Error']
        df_numRows = num_calc_points
        storage_df = pd.DataFrame(data=np.zeros((df_numRows, len(df_col_names))), columns=df_col_names)

        for pt in range(num_calc_points):
            noise = pred_error_df_sorted.iloc[pt, 1]
            sigF = pred_error_df_sorted.iloc[pt, 2]
            length = pred_error_df_sorted.iloc[pt, 3]
            alpha = pred_error_df_sorted.iloc[pt, 4]

            reg = Regression(X_use, Y_use,
                             noise=noise, sigma_F=sigF, scale_length=length, alpha=alpha,
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
            storage_df.loc[pt] = [fig_idx, rmse, r2, cor, noise, sigF, length, alpha,
                                  avg_tr_rmse, avg_tr_r2, avg_tr_cor,
                                  ratio_trAvg_tsAvg, ratio_trAvg_final]

            print("noise: " + str(noise) + ", sigF: " + str(sigF) + ", length: " + str(length) + ", alpha: " + str(
                alpha) + " (" + str(
                pt+1) + "/" + str(num_calc_points) + ")")

        storage_sorted_df = storage_df.sort_values(by=['RMSE'], ascending=True)

        """ TEMP: Create 'Error Array' """
        """
        This should only be thought of as a temporay fix as to not have to replace the current
        'determineTopHPs_XHP' functions. To do so, an array is created that holds double the maximum
        error from the 'storage_df' error values, and then the values that are found in PART III are put
        in their proper place in this array. This is done to ensure that the top values (the ones that would
        be picked up by the 'determineTopHPs_XHP') are still the best ones in this new array
        """
        rmse_list = list(storage_df['RMSE'])
        max_rmse = np.max(rmse_list)
        rmse_bad_input = 2 * max_rmse

        # Create list of good HP-pairs
        tup_list = []
        num_rows_df = len(storage_df['RMSE'])
        for i in range(num_rows_df):
            noise_temp = storage_df['Noise'][i]
            sigF_temp = storage_df['Sigma_F'][i]
            length_temp = storage_df['Length'][i]
            alpha_temp = storage_df['Alpha'][i]
            tup_list.append(tuple((noise_temp, sigF_temp, length_temp, alpha_temp)))

        error_array = np.zeros((numN, numS, numL, numA))

        for n_idx in range(numN):
            noise = noise_range[n_idx]
            for s_idx in range(numS):
                sigF = sigF_range[s_idx]
                for l_idx in range(numL):
                    length = length_range[l_idx]
                    for a_idx in range(numA):
                        alpha = alpha_range[a_idx]

                        if tuple((noise, sigF, length)) not in tup_list:
                            error_temp = rmse_bad_input
                        else:
                            error_temp = float(
                                storage_df.loc[(storage_df['Noise'] == noise) & (storage_df['Sigma_F'] == sigF) & (
                                        storage_df['Length'] == length) & (storage_df['Alpha'] == alpha)]['RMSE'])

                        error_array[n_idx, s_idx, l_idx, a_idx] = error_temp

        return error_array, storage_sorted_df

    def runFullGridSearch_AL_GPR_Noise_SigF_Length_Alpha(self, noise_input_data, sigF_input_data, length_input_data,
                                                      alpha_input_data):

        HP_data = [[noise_input_data, sigF_input_data, length_input_data, alpha_input_data]]

        fig_idx_list = self.figureNumbers(self.numLayers, self.numZooms)

        final_storage_df_list = []

        fig_counter = 0
        for layer in range(self.numLayers):
            HP_layer_data = []
            for position_idx in range(len(HP_data)):
                print("Figure: ", fig_idx_list[fig_counter])
                print("Current Layer: ", layer + 1)
                print("Current Position: ", position_idx + 1)

                HP_data_current = HP_data[position_idx]

                # Sets up C and Epsilon ranges
                noise_data = HP_data_current[0]
                sigF_data = HP_data_current[1]
                length_data = HP_data_current[2]
                alpha_data = HP_data_current[3]
                noise_range = np.linspace(noise_data[0], noise_data[1], self.gridLength)
                sigF_range = np.linspace(sigF_data[0], sigF_data[1], self.gridLength)
                length_range = np.linspace(length_data[0], length_data[1], self.gridLength)
                alpha_range = np.linspace(alpha_data[0], alpha_data[1], self.gridLength)

                # Runs the heatmap with the current Noise, Sigma_F, & Scale-Length ranges
                error_matrix, storage_df_temp = self.runSingleGridSearch_AL_GPR_Noise_SigF_Length_Alpha(noise_range,
                                                                                                     sigF_range,
                                                                                                     length_range,
                                                                                                     alpha_range,
                                                                                                     fig_idx_list[
                                                                                                         fig_counter])

                # Gets the new ranges from the error-matrix
                HP_new_list, Index_Values = self.determineTopHPs_4HP(error_matrix, noise_range, sigF_range,
                                                                     length_range, alpha_range, self.numZooms)

                if layer == self.numLayers - 1:
                    final_storage_df_list.append(storage_df_temp)

                for i in HP_new_list:
                    HP_layer_data.append(i)

                fig_counter += 1

            HP_data = HP_layer_data

        # Initalizes the final storage of the data on the final layer
        storage_df_current = pd.DataFrame(
            columns=['Figure', 'RMSE', 'R^2', 'Cor', 'Noise', 'Sigma_F', 'Length', 'Alpha',
                     'Average Training-RMSE', 'Average Training-R^2',
                     'Average Training-Cor',
                     'avgTR to avgTS', 'avgTR to Final Error'])

        for i in range(len(final_storage_df_list)):
            storage_df_new = pd.concat([storage_df_current, final_storage_df_list[i]])
            storage_df_current = storage_df_new

        storage_df_unsorted = storage_df_current
        storage_df_sorted = storage_df_unsorted.sort_values(by=["RMSE"], ascending=True)
        if self.save_csv_files == True:
            storage_df_unsorted.to_csv('0-Data_' + str(self.mdl_name) + '_Unsorted.csv')
            storage_df_sorted.to_csv('0-Data_' + str(self.mdl_name) + '_Sorted.csv')

        return storage_df_unsorted, storage_df_sorted




