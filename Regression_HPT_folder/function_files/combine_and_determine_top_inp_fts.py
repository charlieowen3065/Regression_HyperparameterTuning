import os
import pandas as pd
import pickle
import csv

pickle_file_decoded = []
with (open('input_file', 'rb')) as file:
    while True:
        try:
            pickle_file_decoded.append(pickle.load(file))
        except EOFError:
            break
input_dict = pickle_file_decoded[0]
file.close()

numTopFeatures = input_dict['numTopFeatures']

data_list = []
X_full_list = []
X_names_list = []
os.chdir('input_feature_set_folder')
for f in os.listdir():
    os.chdir(f)
    data_temp = pd.read_csv('data_file.csv', index_col=0)
    data_list.append(data_temp)

    pickle_file_decoded = []
    with (open('final_X_data', 'rb')) as f_x:
        try:
            pickle_file_decoded.append(pickle.load(f_x))
        except EOFError:
            pass
    final_X_data = pickle_file_decoded[0]

    X_data = final_X_data['features']
    X_names = final_X_data['names']
    for i in range(len(X_data)):
        X_full_list.append(X_data[i])
        X_names_list.append(X_names[i])

    os.chdir('..')
os.chdir('..')

final_data_df = pd.DataFrame()
for df in data_list:
    final_data_df = pd.concat([final_data_df, df], axis=1)

unsorted_filename = 'unsorted_input_data.csv'
final_data_df.to_csv(unsorted_filename)

sorted_data = final_data_df.sort_values(by=["RMSE"], axis=1, ascending=True)
sorted_filename = 'sorted_input_data.csv'
sorted_data.to_csv(sorted_filename)

X_ft_placement_values = sorted_data.loc['sorting_var']

for i in range(numTopFeatures):
    X_idx = int(X_ft_placement_values[i])
    X_data_temp = X_full_list[X_idx]
    X_name_temp = X_names_list[X_idx]

    folder_name = str(i)+'-'+str(X_name_temp)
    os.mkdir(folder_name)
    os.chdir(folder_name)

    with open('feature_data.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(X_data_temp)
    csvfile.close()

    os.chdir('..')


