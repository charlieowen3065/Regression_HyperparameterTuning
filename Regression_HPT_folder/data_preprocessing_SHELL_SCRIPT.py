import os
import numpy as np

exec(open("./create_initial_inputs.py").read())
os.chdir('..')
os.chdir('data_processing_folder')

findTopInpts_var = True

if findTopInpts_var == True:
    case_list = os.listdir()
    for case in case_list:
        os.chdir(case)

        prop_list = os.listdir()
        for prop in prop_list:
            os.chdir(prop)

            model_list = os.listdir()
            for model in model_list:
                os.chdir(model)
                exec(open("./findTopInputs.py").read())
                os.chdir('..')

            os.chdir('..')
        os.chdir('..')
