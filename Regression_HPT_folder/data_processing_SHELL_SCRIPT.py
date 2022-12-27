import os
import shutil
import sys

from function_files.data_processing import process_test

s = os.path.sep

os.chdir('data_processing_folder')
for case in os.listdir():
    os.chdir(case)
    for prop in os.listdir():
        os.chdir(prop)
        for model in os.listdir():
            os.chdir(model)
            model_dir = os.getcwd()
            exec(open("./combine_and_determine_top_inp_fts.py").read())
            for feature in filter(os.path.isdir, os.listdir()):
                if feature != 'input_feature_set_folder':
                    os.chdir(feature)

                    feature_dir = os.getcwd()
                    shutil.copyfile(model_dir+s+'Heatmaps_old.py', feature_dir+s+'Heatmaps_old.py')
                    shutil.copyfile(model_dir + s + 'input_file', feature_dir + s + 'input_file')
                    shutil.copyfile(model_dir + s + 'data_processing.py', feature_dir + s + 'data_processing.py')

                    #exec(open("./data_processing.py").read())
                    process_test()


                    os.chdir('..')
            os.chdir('..')
        os.chdir('..')
    os.chdir('..')