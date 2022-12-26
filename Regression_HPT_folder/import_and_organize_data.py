import os
from scipy.io import loadmat
from config_inputs import *

main_path = os.getcwd()

# IMPORT DATA **********************************************************************************************************
os.chdir(main_path + os.path.sep + 'feature_data')  # goes into the folder containing the 'feature data'
feature_data = loadmat('feature_data')  # loads in the feature data
feature_data.pop('__header__')
feature_data.pop('__version__')
feature_data.pop('__globals__')
nonfeature_data = loadmat('property_data')
os.chdir('..')  # goes back to main path

ftNames = list(feature_data.keys())
feature_set_list = dict()
feature_set_list['org'] = dict()
feature_set_list['org']['RD'] = []
feature_set_list['org']['M'] = []
feature_set_list['org']['YS'] = []
feature_set_list['org']['WH'] = []
feature_set_list['org']['EF'] = []
feature_set_list['org']['UE'] = []
feature_set_list['org']['TS'] = []
feature_set_list['diff'] = dict()
feature_set_list['diff']['RD'] = []
feature_set_list['diff']['M'] = []
feature_set_list['diff']['YS'] = []
feature_set_list['diff']['WH'] = []
feature_set_list['diff']['EF'] = []
feature_set_list['diff']['UE'] = []
feature_set_list['diff']['TS'] = []
feature_set_list['ratio'] = dict()
feature_set_list['ratio']['RD'] = []
feature_set_list['ratio']['M'] = []
feature_set_list['ratio']['YS'] = []
feature_set_list['ratio']['WH'] = []
feature_set_list['ratio']['EF'] = []
feature_set_list['ratio']['UE'] = []
feature_set_list['ratio']['TS'] = []

feature_name_list = dict()
feature_name_list['org'] = dict()
feature_name_list['org']['RD'] = []
feature_name_list['org']['M'] = []
feature_name_list['org']['YS'] = []
feature_name_list['org']['WH'] = []
feature_name_list['org']['EF'] = []
feature_name_list['org']['UE'] = []
feature_name_list['org']['TS'] = []
feature_name_list['diff'] = dict()
feature_name_list['diff']['RD'] = []
feature_name_list['diff']['M'] = []
feature_name_list['diff']['YS'] = []
feature_name_list['diff']['WH'] = []
feature_name_list['diff']['EF'] = []
feature_name_list['diff']['UE'] = []
feature_name_list['diff']['TS'] = []
feature_name_list['ratio'] = dict()
feature_name_list['ratio']['RD'] = []
feature_name_list['ratio']['M'] = []
feature_name_list['ratio']['YS'] = []
feature_name_list['ratio']['WH'] = []
feature_name_list['ratio']['EF'] = []
feature_name_list['ratio']['UE'] = []
feature_name_list['ratio']['TS'] = []

feat_num = feature_num
#feat_num = temp_ft_list[models_to_run[0]]
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

- original data =  3n
- residual-diff =  3n+1
- residual-ratio = 3n+2
"""

case_names = ['diff', 'org', 'ratio']
prop_temp_names = ['EF', 'M', 'RD', 'YS', 'UE', 'WH', 'TS']

for int_i in feat_num:
    for c in range(3):
        for p in range(7):
            ft_num_current = (int_i*21) + (c*7) + p
            ft_current = feature_data[ftNames[ft_num_current]]
            feature_set_list[case_names[c]][prop_temp_names[p]].append(ft_current)
            feature_name_list[case_names[c]][prop_temp_names[p]].append(ftNames[ft_num_current])



# IMPORT DATA **********************************************************************************************************

property_data = nonfeature_data['propData']
property_Av = nonfeature_data['propAv']
property_diff = nonfeature_data['propRes_diff']
property_ratio = nonfeature_data['propRes_ratio']
goodId_org = nonfeature_data['goodId_org']
goodId_res = nonfeature_data['goodId_res']
parmData = nonfeature_data['parm']
locData = nonfeature_data['locDat']
sId = nonfeature_data['sId']
sTpInt = nonfeature_data['sTpInt']

# ******************************************************************************************************************** #
