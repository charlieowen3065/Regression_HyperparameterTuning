import os
os.chdir('function_files')

exec(open("./data_postprocessing.py").read())
os.chdir('../function_files')
exec(open("./format_postprocessed_data.py").read())
