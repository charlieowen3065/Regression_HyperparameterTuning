import numpy as np

run_findTopInputs = True
parm_var = False
locData_var = False
test_train_split_var = False
# When true, the data is first split into a test/train, then the kF-CV is preformed on the train-data
# When False, the data is all run into the kF-CV, and there is no test/train split
N = 1
Nk = 5
# Which properties to run
#props_to_run = [0, 1, 2, 3, 4, 5, 6]
props_to_run = [0, 3, 4, 5]
prop_keys = ['RD', 'M', 'YS', 'WH', 'EF', 'UE', 'TS']
# Which models to run
#models_to_run = [0, 1, 2, 3, 4, 5, 6, 7]
models_to_run = [0, 3, 5, 6, 7]
#models_to_run = [3]
# Which Features to input
feature_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#feature_num = [7, 8, 12, 13]
"""
Feature Data Information
0.  img_avg
1.  img_max
2.  img_min
3.  img_std
4.  optical_img_avg
5.  optical_img_max
6.  optical_img_min
7.  optical_img_std
8.  optical_stat_avg
9.  optical_stat_max
10. optical_stat_min
11. optical_stat_std
12. stat_avg
13. stat_max
14. stat_min
15. stat_std  
"""
"""
lin_ft = [1,5,8,9,14]
p2_ft = [1,5,9,11,13]
p3_ft = [0,7,8,12,13]
rbf_ft = [1,5]
g_rq_ft = [0]
g_rbf_ft = [0,1,2,5,13]
g_m32_ft = [0,3,5,9,12]
g_m52_ft_ft = [0,1,2,5,8]
temp_ft_list = [lin_ft, p2_ft, p3_ft, rbf_ft, g_rq_ft, g_rbf_ft, g_m32_ft, g_m52_ft_ft]

lin_hp = [3.474942119, 0.321831647,1,1]
p2_hp = [0.070275648, 0.24829284, 0.368576132, 1.605777778]
p3_hp = [38.74939971, 0.419330129, 1.600182368, 5.254246914]
rbf_hp = [5.833236872, 0.818634267, 0.065418574, 1]
g_rq_hp = [1,1,1,1]
g_rbf_hp = [125.9079372, 0.599654098, 0.805512952, 1]
g_m32_hp = [19.44995466, 0.668468106, 1.021848106,1]
g_m52_hp = [496.5528519, 2.023380774, 0.756235125,1]
temp_hp_list = [lin_hp, p2_hp, p3_hp, rbf_hp, g_rq_hp, g_rbf_hp, g_m32_hp, g_m52_hp]
"""
# Which cases to use
#case_use = ['org', 'diff', 'ratio']
case_use = ['org']
# Combo Array
maxNumCombos = 4
combo_array = np.arange(start=2, stop=maxNumCombos+1, step=1)
num_fts_per_sublist = 100

numTopFeatures = 1

prop_names = ["Relative_Density", "Modulus", "Yield_Strength", "Work_Hardening_Exponent", "Elongation_to_Fracture",
              "Uniform_Elongation_from_MAX", "Tensile_Strength_from_MAX"]
model_names = ['SVM_Linear', 'SVM_Poly2', "SVM_Poly3", 'SVM_RBF',
               "GPR_RationalQuadratic", "GPR_RBF", "GPR_Matern32", "GPR_Matern52"]
model_types = ['SVM', 'SVM', 'SVM', 'SVM', 'GPR', 'GPR', 'GPR', 'GPR']

# For Heatmaps
heatmap_inputs = dict()
heatmap_inputs['gridLength_GS'] = dict()
heatmap_inputs['numZooms_GS'] = dict()
heatmap_inputs['numLayers_GS'] = dict()
heatmap_inputs['decimal_point_GS'] = dict()
heatmap_inputs['gridLength_AL'] = dict()
heatmap_inputs['num_HP_zones_AL'] = dict()
heatmap_inputs['num_runs_AL'] = dict()
heatmap_inputs['decimal_points_int'] = dict()
heatmap_inputs['decimal_points_top'] = dict()
# SVM_Linear
mdl = 0
heatmap_inputs['gridLength_GS'][model_names[mdl]] = 15
heatmap_inputs['numZooms_GS'][model_names[mdl]] = 2
heatmap_inputs['numLayers_GS'][model_names[mdl]] = 3
heatmap_inputs['decimal_point_GS'][model_names[mdl]] = 0.1
heatmap_inputs['gridLength_AL'][model_names[mdl]] = 30
heatmap_inputs['num_HP_zones_AL'][model_names[mdl]] = 3
heatmap_inputs['num_runs_AL'][model_names[mdl]] = 3
heatmap_inputs['decimal_points_int'][model_names[mdl]] = 0.10
heatmap_inputs['decimal_points_top'][model_names[mdl]] = 0.20
# SVM_Poly2
mdl = 1
heatmap_inputs['gridLength_GS'][model_names[mdl]] = 15
heatmap_inputs['numZooms_GS'][model_names[mdl]] = 2
heatmap_inputs['numLayers_GS'][model_names[mdl]] = 3
heatmap_inputs['decimal_point_GS'][model_names[mdl]] = 0.1
heatmap_inputs['gridLength_AL'][model_names[mdl]] = 30
heatmap_inputs['num_HP_zones_AL'][model_names[mdl]] = 3
heatmap_inputs['num_runs_AL'][model_names[mdl]] = 3
heatmap_inputs['decimal_points_int'][model_names[mdl]] = 0.10
heatmap_inputs['decimal_points_top'][model_names[mdl]] = 0.20
# SVM_Poly3
mdl = 2
heatmap_inputs['gridLength_GS'][model_names[mdl]] = 15
heatmap_inputs['numZooms_GS'][model_names[mdl]] = 2
heatmap_inputs['numLayers_GS'][model_names[mdl]] = 3
heatmap_inputs['decimal_point_GS'][model_names[mdl]] = 0.1
heatmap_inputs['gridLength_AL'][model_names[mdl]] = 30
heatmap_inputs['num_HP_zones_AL'][model_names[mdl]] = 3
heatmap_inputs['num_runs_AL'][model_names[mdl]] = 3
heatmap_inputs['decimal_points_int'][model_names[mdl]] = 0.10
heatmap_inputs['decimal_points_top'][model_names[mdl]] = 0.20
# SVM_RBF
mdl = 3
heatmap_inputs['gridLength_GS'][model_names[mdl]] = 15
heatmap_inputs['numZooms_GS'][model_names[mdl]] = 2
heatmap_inputs['numLayers_GS'][model_names[mdl]] = 3
heatmap_inputs['decimal_point_GS'][model_names[mdl]] = 0.1
heatmap_inputs['gridLength_AL'][model_names[mdl]] = 30
heatmap_inputs['num_HP_zones_AL'][model_names[mdl]] = 3
heatmap_inputs['num_runs_AL'][model_names[mdl]] = 3
heatmap_inputs['decimal_points_int'][model_names[mdl]] = 0.10
heatmap_inputs['decimal_points_top'][model_names[mdl]] = 0.20
# GPR_RatQuad
mdl = 4
heatmap_inputs['gridLength_GS'][model_names[mdl]] = 15
heatmap_inputs['numZooms_GS'][model_names[mdl]] = 2
heatmap_inputs['numLayers_GS'][model_names[mdl]] = 3
heatmap_inputs['decimal_point_GS'][model_names[mdl]] = 0.1
heatmap_inputs['gridLength_AL'][model_names[mdl]] = 30
heatmap_inputs['num_HP_zones_AL'][model_names[mdl]] = 3
heatmap_inputs['num_runs_AL'][model_names[mdl]] = 3
heatmap_inputs['decimal_points_int'][model_names[mdl]] = 0.10
heatmap_inputs['decimal_points_top'][model_names[mdl]] = 0.20
# GPR_RBF
mdl = 5
heatmap_inputs['gridLength_GS'][model_names[mdl]] = 15
heatmap_inputs['numZooms_GS'][model_names[mdl]] = 2
heatmap_inputs['numLayers_GS'][model_names[mdl]] = 3
heatmap_inputs['decimal_point_GS'][model_names[mdl]] = 0.1
heatmap_inputs['gridLength_AL'][model_names[mdl]] = 30
heatmap_inputs['num_HP_zones_AL'][model_names[mdl]] = 3
heatmap_inputs['num_runs_AL'][model_names[mdl]] = 3
heatmap_inputs['decimal_points_int'][model_names[mdl]] = 0.10
heatmap_inputs['decimal_points_top'][model_names[mdl]] = 0.20
# GPR_Matern32
mdl = 6
heatmap_inputs['gridLength_GS'][model_names[mdl]] = 15
heatmap_inputs['numZooms_GS'][model_names[mdl]] = 2
heatmap_inputs['numLayers_GS'][model_names[mdl]] = 3
heatmap_inputs['decimal_point_GS'][model_names[mdl]] = 0.1
heatmap_inputs['gridLength_AL'][model_names[mdl]] = 30
heatmap_inputs['num_HP_zones_AL'][model_names[mdl]] = 3
heatmap_inputs['num_runs_AL'][model_names[mdl]] = 3
heatmap_inputs['decimal_points_int'][model_names[mdl]] = 0.10
heatmap_inputs['decimal_points_top'][model_names[mdl]] = 0.20
# GPR_Matern52
mdl = 7
heatmap_inputs['gridLength_GS'][model_names[mdl]] = 15
heatmap_inputs['numZooms_GS'][model_names[mdl]] = 2
heatmap_inputs['numLayers_GS'][model_names[mdl]] = 3
heatmap_inputs['decimal_point_GS'][model_names[mdl]] = 0.1
heatmap_inputs['gridLength_AL'][model_names[mdl]] = 30
heatmap_inputs['num_HP_zones_AL'][model_names[mdl]] = 3
heatmap_inputs['num_runs_AL'][model_names[mdl]] = 3
heatmap_inputs['decimal_points_int'][model_names[mdl]] = 0.10
heatmap_inputs['decimal_points_top'][model_names[mdl]] = 0.20


#gridLength = 20
#numZooms = 2
#numLayers = 3

# Test/Train Split
split_decimal = 0.3  # the amount of data used for testing

# SEEDS ****************************************************************************************************************
seeds = [1939855286, 407978056, 1280179023, 1518676132, 916062339, 1178283666, 382075401]
tr_ts_seeds = ['random', 'random', 'random', 'random', 'random', 'random', 'random']

# Hyperparamter Ranges *************************************************************************************************


full_HP_list = dict()
# Relative Density -----------------------------------------------------------------------------------------------------
prop = 0
full_HP_list[str(prop_names[prop])] = dict()
# * SVM Linear
mdl = 0
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 0.8)
# * SVM Poly2
mdl = 1
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 0.75)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM Poly3
mdl = 2
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 0.75)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM RBF
mdl = 3
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 0.6)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 0.8)
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
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 0.2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
# * SVM Poly2
mdl = 1
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 0.2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM Poly3
mdl = 2
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 0.2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM RBF
mdl = 3
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 0.2)
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
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 50)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM RBF
mdl = 3
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 0.4)
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
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 50)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# * GPR Matern 3/2
mdl = 6
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 50)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["sigmaF"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["scale_length"] = (0.00001, 1)
# * GPR Matern 5/2
mdl = 7
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["noise"] = (0.001, 50)
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
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 0.5)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM Poly3
mdl = 2
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 50)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 2)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["c0"] = (0.001, 10)
# * SVM RBF
mdl = 3
full_HP_list[str(prop_names[prop])][str(model_names[mdl])] = dict()
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["C"] = (0.001, 200)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["e"] = (0.00001, 1)
full_HP_list[str(prop_names[prop])][str(model_names[mdl])]["g"] = (0.001, 0.8)
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
