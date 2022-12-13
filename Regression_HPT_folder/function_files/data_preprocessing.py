import os
import shutil


class data_preprocessing():
    def __init__(self, input_dict='empty'):
        self.input_dict = input_dict

    def copy_input_files(self):

        input_dict = self.input_dict

        # Copy Files
        import pickle
        file_of_input_data = open('input_file', 'wb')
        pickle.dump(input_dict, file_of_input_data)
        file_of_input_data.close

        main_path = input_dict['main_path']
        s = os.path.sep
        #os.system('cp '+main_path+s+'function_files'+s+'findTopFeatures.py findTopFeatures.py')
        shutil.copyfile(main_path + s +'function_files' + s +'findTopInputs.py', 'findTopInputs.py')
        shutil.copyfile(main_path + s + 'function_files' + s + 'Heatmaps.py', 'Heatmaps.py')
        shutil.copyfile(main_path + s + 'function_files' + s + 'data_processing.py', 'data_processing.py')
        shutil.copyfile(main_path + s + 'slurm_scripts' + s + 'data_processing.sub', 'data_processing.sub')

        shutil.copyfile(main_path + s + 'function_files' + s + 'format_input_data.py', 'format_input_data.py')
        shutil.copyfile(main_path + s + 'function_files' + s + 'run_single_inp_subset.py', 'run_single_inp_subset.py')
        shutil.copyfile(main_path + s + 'slurm_scripts' + s + 'run_single_inp_subset.sub', 'run_single_inp_subset.sub')
        shutil.copyfile(main_path + s + 'function_files' + s + 'combine_and_determine_top_inp_fts.py', 'combine_and_determine_top_inp_fts.py')

        Nk = input_dict['Nk']
        N = input_dict['N']
        numLayers = input_dict['numLayers']
        numZooms = input_dict['numZooms']
        combo_array = input_dict['combo_array']
        numTopFeatures = input_dict['numTopFeatures']
        feature_names = input_dict['feature_names']
        hyperparmeters = input_dict['hyperparameters']
        models_use = input_dict['models_use']
        Y_inp = input_dict['Y_inp']
        X_inp = input_dict['X_list']

        with open('input.txt', 'w') as f:
            f.write('Nk: '+str(Nk)+", N: "+str(N)+"\n")
            f.write("number-top-input-features: " + str(numTopFeatures)+"\n")
            f.write("number of layers: " + str(numLayers)+"\n")
            f.write("number of zooms: " + str(numZooms)+"\n")
            f.write('\n')
            f.write('Feature_set_names: \n')
            f.write(str(feature_names)+"\n")

            f.close()
