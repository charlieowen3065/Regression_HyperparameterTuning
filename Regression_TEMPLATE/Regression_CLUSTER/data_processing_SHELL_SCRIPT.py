import os
import shutil

s = os.path.sep

os.chdir('data_processing_folder')
for case in os.listdir():
    os.chdir(case)
    for prop in os.listdir():
        os.chdir(prop)
        for model in os.listdir():
            os.chdir(model)
            model_dir = os.getcwd()
            for feature in filter(os.path.isdir, os.listdir()):
                os.chdir(feature)

                feature_dir = os.getcwd()
                shutil.copyfile(model_dir+s+'Heatmaps.py', feature_dir+s+'Heatmaps.py')
                shutil.copyfile(model_dir + s + 'input_file', feature_dir + s + 'input_file')
                shutil.copyfile(model_dir + s + 'data_processing.py', feature_dir + s + 'data_processing.py')

                exec(open("./data_processing.py").read())

                os.chdir('..')
            os.chdir('..')
        os.chdir('..')
    os.chdir('..')