import os
import shutil
import pandas as pd
os.chdir('..')
#from config_inputs import *
exec(open("./config_inputs.py").read())

if 'data_postprocessing_folder' in os.listdir():
    shutil.rmtree('data_postprocessing_folder')
os.mkdir('data_postprocessing_folder')
os.chdir('data_postprocessing_folder')
os.mkdir('raw_data')
os.chdir('raw_data')
# CREATING SKELETON FOLDERS
for case in case_use:
    os.mkdir('Case_'+case)
    os.chdir('Case_'+case)
    i = 0
    for prop in props_to_run:
        prop_folder = str(i)+"-"+prop_names[prop]
        os.mkdir(prop_folder)
        os.chdir(prop_folder)
        j = 0
        for mdl in model_names:
            mdl_folder = str(j)+"-"+str(mdl)
            os.mkdir(mdl_folder)
            j+=1
        i += 1
        os.chdir('..')
    os.chdir('..')
os.chdir('../..')

m_path = os.getcwd()
s = os.path.sep
os.chdir('data_processing_folder')

# PULLS DATA FROM 'PROCESSING FOLDER'

for case in os.listdir():
    os.chdir(case)
    for prop in os.listdir():
        os.chdir(prop)
        for mdl in os.listdir():
            os.chdir(mdl)
            i = 0
            for ft in filter(os.path.isdir, os.listdir()):
                if ft != '__pycache__':
                    os.chdir(ft)
                    
                    ft_use = ft
                    ft_use = ft_use.replace("-('", '')
                    ft_use = ft_use.replace("')", "")
                    ft_use = ft_use.replace("', '", '_and_')
                    ft_final = ft_use[1:]
                    
                    if 'Heatmaps' in os.listdir():
                        os.chdir('Heatmaps')
        
                        filename_sorted = '0-Data_'+str(mdl[2:])+'_Sorted.csv'
                        #print("FILE: ", filename_sorted)
                        if filename_sorted in os.listdir():
                            dest_path = m_path + s + 'data_postprocessing_folder' + s + 'raw_data' + s + case + s + prop + s + mdl
                            i_use = str("%01d" % (i))
                            filename_new = str(i_use)+"_"+str(mdl)+"_"+str(ft_final)+".csv"
                            shutil.copyfile(filename_sorted, dest_path+s+filename_new)
                        else:
                            print(str(mdl)+" - "+str(ft_use)+" is not completed")
        
                        i += 1
                        os.chdir('..')
                    else:
                        print(str(mdl)+" - "+str(ft_use)+" not started")
                    os.chdir('..')
            os.chdir('..')

        os.chdir('..')
    os.chdir('..')
os.chdir('..')

# RUNS POSTPROCESSING IN DATA IN 'RAW DATA' FOLDER
os.chdir('data_postprocessing_folder')
post_path = os.getcwd()


if 'postprocess' in os.listdir():
    shutil.rmtree('postprocess')
os.mkdir('postprocess')
os.chdir('postprocess')
for case in case_use:
    case_folder = 'Case_'+case 
    os.mkdir(case_folder)
    os.chdir(case_folder)
    os.mkdir('all_props')
    i = 0
    for prop in props_to_run:
        prop_folder = str(i)+"-"+prop_names[prop]
        os.mkdir(prop_folder)
        i += 1
    os.chdir('..')
os.chdir('..')

os.chdir('raw_data')
for case in case_use:
    case_folder = 'Case_'+case 
    os.chdir(case_folder)
    i = 0
    counter = 1
    for prop in props_to_run:
        prop_folder = str(i)+"-"+prop_names[prop]
        os.chdir(prop_folder)
        i+=1
        j = 0
        for mdl in model_names:
            print("MODEL: ", mdl)
            mdl_folder = str(j)+"-"+str(mdl)
            j+=1
            os.chdir(mdl_folder)
            if len(os.listdir()) > 0:
                file_for_column_names = pd.read_csv(os.listdir()[0])
                col_names_temp = ['input_features']
                for col_i in file_for_column_names.columns[1:]:
                    col_names_temp.append(col_i)
                temp_df = pd.DataFrame(columns=col_names_temp)
                csv_file_list = os.listdir()
                for f in csv_file_list:
                    ft_name = f[2:]
                    ft_name = ft_name[len(mdl_folder)+1:]
                    ft_name = ft_name[:-4]
                    temp_data = pd.read_csv(f)
                    ftNames_data = [ft_name] * len(temp_data)
                    ftNames_np_data = np.array(ftNames_data)
                    ftNames_np_data.shape = (len(temp_data),1)
                    ftNames_col = pd.DataFrame(data=ftNames_np_data, columns=['input_features'])
                    temp_data = pd.concat([ftNames_col, temp_data.iloc[:, 1:]], axis=1)
                    temp_df = pd.concat([temp_df, temp_data], axis=0)
                temp_df.to_csv('unsorted_data.csv')
                sorted_temp_df = temp_df.sort_values(by=['RMSE'], ascending=True)
                sorted_final_df = sorted_temp_df.iloc[0:50, :]
                sorted_filename = str("%02d" % (counter))+'-'+str(prop_names[prop])+"_"+str(mdl)+"_Sorted.csv"
                counter += 1
                sorted_final_df.to_csv(sorted_filename)
                
                dest_path_singleProp = post_path + s + 'postprocess' + s + case_folder + s + prop_folder + s + sorted_filename
                dest_path_multiProp = post_path + s + 'postprocess' + s + case_folder + s + 'all_props' + s + sorted_filename
                shutil.copyfile(sorted_filename, dest_path_singleProp)
                shutil.copyfile(sorted_filename, dest_path_multiProp)
                
            os.chdir('..')
        os.chdir('..')
    os.chdir('..')
os.chdir('..')

