module load anaconda

dos2unix data_preprocessing
chmod +x data_preprocessing
dos2unix data_processing
chmod +x data_processing
dos2unix data_postprocessing
chmod +x data_postprocessing
dos2unix clear_queue
chmod +x clear_queue

cd slurm_scripts

dos2unix data_processing.sub
dos2unix findTopInputs.sub
dos2unix run_single_inp_subset.sub

cd ..
