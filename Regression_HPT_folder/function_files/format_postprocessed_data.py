import os
import shutil
import pandas as pd
os.chdir('..')
#from config_inputs import *
exec(open("./config_inputs.py").read())


os.chdir('data_postprocessing_folder')


# 

s = os.path.sep
os.chdir('postprocess')
for case in case_use:
    case_folder = 'Case_'+case 
    os.chdir(case_folder)
    if 'formatted_data' in os.listdir():
        shutil.rmtree('formatted_data')
    os.mkdir('formatted_data')
    format_dest = os.getcwd() + s + 'formatted_data'
    prop = 0
    for prop in props_to_run:
        prop_folder = str(prop)+"-"+prop_names[prop]
        os.chdir(prop_folder)
        data_list = os.listdir()
        data_prop = pd.DataFrame(columns=['MODEL', 'INPUT FEATURES', 'RMSE', 'R^2', 'HP1', 'HP2', 'HP3', 'HP4', 'RATIO'])
        # SVM ---------------------------------------------------------------------------------------------------------------------------------------------------------------
        try:
            i = 0
            data_test_del = data_list[i]
            data_prop.loc[0] = ['Models', 'Input Features', 'RMSE', 'R^2', 'C', 'Epsilon', 'Gamma', 'Coef0', 'Training-to-Final Error']
            i = 0
            # SVM_Linear
            filename_temp = data_list[i]
            print("Prop: ", prop_names[prop])
            if filename_temp[-len('SVM_Linear_Sorted.csv'):] == 'SVM_Linear_Sorted.csv':
                print('SVM_Linear')
                print('i: ', i)
                data_temp_full = pd.read_csv(filename_temp)
                data_temp = data_temp_full.iloc[0,:]
                mdl = 'SVM Linear'
                inpFt = data_temp['input_features']
                rmse = data_temp['RMSE']
                r2 = data_temp['R^2']
                C = data_temp['C']
                e = data_temp['Epsilon']
                g = data_temp['Gamma']
                c0 = 'N/A'
                ratio = data_temp['avgTR to Final Error']
                data_prop.loc[i+1] = [mdl, inpFt, rmse, r2, C, e, g, c0, ratio]
                i += 1
            # SVM_Poly2
            filename_temp = data_list[i]
            if filename_temp[-len('SVM_Poly2_Sorted.csv'):] == 'SVM_Poly2_Sorted.csv':
                print('SVM_Poly2')
                print('i: ', i)
                data_temp_full = pd.read_csv(filename_temp)
                data_temp = data_temp_full.iloc[0,:]
                mdl = 'SVM Poly2'
                inpFt = data_temp['input_features']
                rmse = data_temp['RMSE']
                r2 = data_temp['R^2']
                C = data_temp['C']
                e = data_temp['Epsilon']
                g = data_temp['Gamma']
                c0 = data_temp['Coef0']
                ratio = data_temp['avgTR to Final Error']
                data_prop.loc[i+1] = [mdl, inpFt, rmse, r2, C, e, g, c0, ratio]
                i += 1
            # SVM_Poly3
            filename_temp = data_list[i]
            if filename_temp[-len('SVM_Poly3_Sorted.csv'):] == 'SVM_Poly3_Sorted.csv':
                print('SVM_Poly3')
                print('i: ', i)
                data_temp_full = pd.read_csv(filename_temp)
                data_temp = data_temp_full.iloc[0,:]
                mdl = 'SVM Poly3'
                inpFt = data_temp['input_features']
                rmse = data_temp['RMSE']
                r2 = data_temp['R^2']
                C = data_temp['C']
                e = data_temp['Epsilon']
                g = data_temp['Gamma']
                c0 = data_temp['Coef0']
                ratio = data_temp['avgTR to Final Error']
                data_prop.loc[i+1] = [mdl, inpFt, rmse, r2, C, e, g, c0, ratio]
                i += 1
            # SVM_RBF
            filename_temp = data_list[i]
            if filename_temp[-len('SVM_RBF_Sorted.csv'):] == 'SVM_RBF_Sorted.csv':
                print('SVM_RBF')
                print('i: ', i)
                data_temp_full = pd.read_csv(filename_temp)
                data_temp = data_temp_full.iloc[0,:]
                mdl = 'SVM RBF'
                inpFt = data_temp['input_features']
                rmse = data_temp['RMSE']
                r2 = data_temp['R^2']
                C = data_temp['C']
                e = data_temp['Epsilon']
                g = data_temp['Gamma']
                c0 = 'N/A'
                ratio = data_temp['avgTR to Final Error']
                data_prop.loc[i+1] = [mdl, inpFt, rmse, r2, C, e, g, c0, ratio]
                i += 1
        except:
            pass
        # GPR ---------------------------------------------------------------------------------------------------------------------------------------------------------------
        try:
            test_del = data_list[i-1]
            data_prop.loc[i+1] = ['Models', 'Input Features', 'RMSE', 'R^2', 'Noise', 'SigmaF', 'Length', 'Alpha', 'Training-to-Final Error']
            i += 1 
            # GPR_RatQuad
            filename_temp = data_list[i-1]
            if filename_temp[-len('GPR_RationalQuadratic_Sorted.csv'):] == 'GPR_RationalQuadratic_Sorted.csv':
                print('GPR_RatQuad')
                print('i: ', i)
                data_temp_full = pd.read_csv(filename_temp)
                data_temp = data_temp_full.iloc[0,:]
                mdl = 'GPR Rational Quadratic'
                inpFt = data_temp['input_features']
                rmse = data_temp['RMSE']
                r2 = data_temp['R^2']
                noise = data_temp['Noise']
                sigF = data_temp['Sigma_F']
                l = data_temp['Length']
                alpha = data_temp['Alpha']
                ratio = data_temp['avgTR to Final Error']
                data_prop.loc[i+1] = [mdl, inpFt, rmse, r2, noise, sigF, l, alpha, ratio]
                i += 1
            # GPR_RBF
            filename_temp = data_list[i-1]
            if filename_temp[-len('GPR_RBF_Sorted.csv'):] == 'GPR_RBF_Sorted.csv':
                print('GPR_RBF')
                print('i: ', i)
                data_temp_full = pd.read_csv(filename_temp)
                data_temp = data_temp_full.iloc[0,:]
                mdl = 'GPR RBF'
                inpFt = data_temp['input_features']
                rmse = data_temp['RMSE']
                r2 = data_temp['R^2']
                noise = data_temp['Noise']
                sigF = data_temp['Sigma_F']
                l = data_temp['Length']
                alpha = 'N/A'
                ratio = data_temp['avgTR to Final Error']
                data_prop.loc[i+1] = [mdl, inpFt, rmse, r2, noise, sigF, l, alpha, ratio]
                i += 1
            # GPR_Matern3/2
            filename_temp = data_list[i-1]
            if filename_temp[-len('GPR_Matern32_Sorted.csv'):] == 'GPR_Matern32_Sorted.csv':
                print('GPR_Matern32')
                print('i: ', i)
                data_temp_full = pd.read_csv(filename_temp)
                data_temp = data_temp_full.iloc[0,:]
                mdl = 'GPR Matern 3/2'
                inpFt = data_temp['input_features']
                rmse = data_temp['RMSE']
                r2 = data_temp['R^2']
                noise = data_temp['Noise']
                sigF = data_temp['Sigma_F']
                l = data_temp['Length']
                alpha = 'N/A'
                ratio = data_temp['avgTR to Final Error']
                data_prop.loc[i+1] = [mdl, inpFt, rmse, r2, noise, sigF, l, alpha, ratio]
                i += 1
            # GPR_Matern5/2
            filename_temp = data_list[i-1]
            if filename_temp[-len('GPR_Matern52_Sorted.csv'):] == 'GPR_Matern52_Sorted.csv':
                print('GPR_Matern52')
                print('i: ', i)
                data_temp_full = pd.read_csv(filename_temp)
                data_temp = data_temp_full.iloc[0,:]
                mdl = 'GPR Matern 5/2'
                inpFt = data_temp['input_features']
                rmse = data_temp['RMSE']
                r2 = data_temp['R^2']
                noise = data_temp['Noise']
                sigF = data_temp['Sigma_F']
                l = data_temp['Length']
                alpha = 'N/A'
                ratio = data_temp['avgTR to Final Error']
                data_prop.loc[i+1] = [mdl, inpFt, rmse, r2, noise, sigF, l, alpha, ratio]
                i += 1
        except:
            pass
        
        filename_temp = str(prop_folder)+'_formatted_data.csv'
        data_prop.to_csv(filename_temp)
        shutil.copyfile(filename_temp, format_dest + s + filename_temp)
        prop+=1
        os.chdir('..')
    os.chdir('..')




