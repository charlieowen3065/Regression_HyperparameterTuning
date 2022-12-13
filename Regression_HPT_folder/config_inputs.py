import numpy as np

run_findTopInputs = True
parm_var = True
locData_var = False
test_train_split_var = False
# When true, the data is first split into a test/train, then the kF-CV is preformed on the train-data
# When False, the data is all run into the kF-CV, and there is no test/train split
N = 1
Nk = 5
# Which properties to run
props_to_run = [0, 1, 2, 3, 4, 5, 6]
#props_to_run = [3, 5]
prop_keys = ['RD', 'M', 'YS', 'WH', 'EF', 'UE', 'TS']
# Which models to run
models_to_run = [0, 1, 2, 3, 4, 5, 6, 7]
#models_to_run = [3]
# Which Features to input
feature_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#feature_num = [0, 1, 12]
# Which cases to use
#case_use = ['org', 'diff', 'ratio']
case_use = ['org']
# Combo Array
maxNumCombos = 2
combo_array = np.arange(start=2, stop=maxNumCombos+1, step=1)
num_fts_per_sublist = 100

numTopFeatures = 5

prop_names = ["Relative_Density", "Modulus", "Yield_Strength", "Work_Hardening_Exponent", "Elongation_to_Fracture",
              "Uniform_Elongation_from_MAX", "Tensile_Strength_from_MAX"]
model_names = ['SVM_Linear', 'SVM_Poly2', "SVM_Poly3", 'SVM_RBF',
               "GPR_RationalQuadratic", "GPR_RBF", "GPR_Matern32", "GPR_Matern52"]
model_types = ['SVM', 'SVM', 'SVM', 'SVM', 'GPR', 'GPR', 'GPR', 'GPR']

# For Heatmaps
heatmap_inputs = dict()
heatmap_inputs['gridLength'] = dict()
heatmap_inputs['numZooms'] = dict()
heatmap_inputs['numLayers'] = dict()
# SVM_Linear
mdl = 0
heatmap_inputs['gridLength'][model_names[mdl]] = 20
heatmap_inputs['numZooms'][model_names[mdl]] = 2
heatmap_inputs['numLayers'][model_names[mdl]] = 3
# SVM_Poly2
mdl = 1
heatmap_inputs['gridLength'][model_names[mdl]] = 10
heatmap_inputs['numZooms'][model_names[mdl]] = 2
heatmap_inputs['numLayers'][model_names[mdl]] = 4
# SVM_Poly3
mdl = 2
heatmap_inputs['gridLength'][model_names[mdl]] = 10
heatmap_inputs['numZooms'][model_names[mdl]] = 2
heatmap_inputs['numLayers'][model_names[mdl]] = 4
# SVM_RBF
mdl = 3
heatmap_inputs['gridLength'][model_names[mdl]] = 20
heatmap_inputs['numZooms'][model_names[mdl]] = 2
heatmap_inputs['numLayers'][model_names[mdl]] = 3
# GPR_RatQuad
mdl = 4
heatmap_inputs['gridLength'][model_names[mdl]] = 10
heatmap_inputs['numZooms'][model_names[mdl]] = 2
heatmap_inputs['numLayers'][model_names[mdl]] = 4
# GPR_RBF
mdl = 5
heatmap_inputs['gridLength'][model_names[mdl]] = 20
heatmap_inputs['numZooms'][model_names[mdl]] = 2
heatmap_inputs['numLayers'][model_names[mdl]] = 3
# GPR_Matern32
mdl = 6
heatmap_inputs['gridLength'][model_names[mdl]] = 20
heatmap_inputs['numZooms'][model_names[mdl]] = 2
heatmap_inputs['numLayers'][model_names[mdl]] = 3
# GPR_Matern52
mdl = 7
heatmap_inputs['gridLength'][model_names[mdl]] = 20
heatmap_inputs['numZooms'][model_names[mdl]] = 2
heatmap_inputs['numLayers'][model_names[mdl]] = 3


#gridLength = 20
#numZooms = 2
#numLayers = 3

# Test/Train Split
split_decimal = 0.3  # the amount of data used for testing

# SEEDS ****************************************************************************************************************
seeds = [1939855286, 407978056, 1280179023, 1518676132, 916062339, 1178283666, 382075401]
tr_ts_seeds = [0, 0, 0, 0, 0, 0, 0]

# Hyperparamter Ranges *************************************************************************************************


full_HP_list = dict()
# Relative Density -----------------------------------------------------------------------------------------------------
prop = 0
full_HP_list[str(prop_names[prop])] = dict()
# * SVM Linear
mdl = 0
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
# * SVM Poly2
mdl = 1
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM Poly3
mdl = 2
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM RBF
mdl = 3
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
# * GPR RatQuad
mdl = 4
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["alpha"] = (0.001, 10)
# * GPR RBF
mdl = 5
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# * GPR Matern 3/2
mdl = 6
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# * GPR Matern 5/2
mdl = 7
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# Modulus --------------------------------------------------------------------------------------------------------------
prop = 1
full_HP_list[str(prop_names[prop])] = dict()
# * SVM Linear
mdl = 0
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
# * SVM Poly2
mdl = 1
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM Poly3
mdl = 2
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM RBF
mdl = 3
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
# * GPR RatQuad
mdl = 4
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["alpha"] = (0.001, 10)
# * GPR RBF
mdl = 5
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# * GPR Matern 3/2
mdl = 6
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# * GPR Matern 5/2
mdl = 7
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# Yield Strength -------------------------------------------------------------------------------------------------------
prop = 2
full_HP_list[str(prop_names[prop])] = dict()
# * SVM Linear
mdl = 0
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
# * SVM Poly2
mdl = 1
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM Poly3
mdl = 2
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM RBF
mdl = 3
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
# * GPR RatQuad
mdl = 4
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["alpha"] = (0.001, 10)
# * GPR RBF
mdl = 5
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# * GPR Matern 3/2
mdl = 6
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# * GPR Matern 5/2
mdl = 7
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# Work_Hardening_Exponent ----------------------------------------------------------------------------------------------
prop = 3
full_HP_list[str(prop_names[prop])] = dict()
# * SVM Linear
mdl = 0
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
# * SVM Poly2
mdl = 1
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM Poly3
mdl = 2
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM RBF
mdl = 3
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
# * GPR RatQuad
mdl = 4
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["alpha"] = (0.001, 10)
# * GPR RBF
mdl = 5
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# * GPR Matern 3/2
mdl = 6
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# * GPR Matern 5/2
mdl = 7
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# Elongation to Fracture -----------------------------------------------------------------------------------------------
prop = 4
full_HP_list[str(prop_names[prop])] = dict()
# * SVM Linear
mdl = 0
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
# * SVM Poly2
mdl = 1
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM Poly3
mdl = 2
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM RBF
mdl = 3
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
# * GPR RatQuad
mdl = 4
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["alpha"] = (0.001, 10)
# * GPR RBF
mdl = 5
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# * GPR Matern 3/2
mdl = 6
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# * GPR Matern 5/2
mdl = 7
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# Uniform Elongation from MAX ------------------------------------------------------------------------------------------
prop = 5
full_HP_list[str(prop_names[prop])] = dict()
# * SVM Linear
mdl = 0
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
# * SVM Poly2
mdl = 1
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM Poly3
mdl = 2
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM RBF
mdl = 3
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
# * GPR RatQuad
mdl = 4
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["alpha"] = (0.001, 10)
# * GPR RBF
mdl = 5
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# * GPR Matern 3/2
mdl = 6
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# * GPR Matern 5/2
mdl = 7
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# Tensile Strength from MAX --------------------------------------------------------------------------------------------
prop = 6
full_HP_list[str(prop_names[prop])] = dict()
# * SVM Linear
mdl = 0
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
# * SVM Poly2
mdl = 1
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM Poly3
mdl = 2
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM RBF
mdl = 3
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
# * GPR RatQuad
mdl = 4
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["alpha"] = (0.001, 10)
# * GPR RBF
mdl = 5
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# * GPR Matern 3/2
mdl = 6
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# * GPR Matern 5/2
mdl = 7
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
