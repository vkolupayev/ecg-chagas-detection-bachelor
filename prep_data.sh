python -m prepare_samitrop_data \
     -i data/raw/samitrop/exams.hdf5 \
     -d data/raw/samitrop/exams.csv \
     -o data/training_data/samitrop_wfdb
     
python -m prepare_ptbxl_data \
     -i data/raw/ptbxl/records500/ \
     -d data/raw/ptbxl/ptbxl_database.csv \
     -o data/training_data/ptbxl_500_wfdb

python -m prepare_code15_data \
	-i data/raw/code15/exams_part0.hdf5 data/raw/code15/exams_part1.hdf5 \
	data/raw/code15/exams_part2.hdf5 data/raw/code15/exams_part3.hdf5 \
	data/raw/code15/exams_part4.hdf5 data/raw/code15/exams_part5.hdf5 \
	data/raw/code15/exams_part6.hdf5 data/raw/code15/exams_part7.hdf5 \
	data/raw/code15/exams_part8.hdf5 data/raw/code15/exams_part9.hdf5 \
	data/raw/code15/exams_part10.hdf5 data/raw/code15/exams_part11.hdf5 \
	data/raw/code15/exams_part12.hdf5 data/raw/code15/exams_part13.hdf5 \
	data/raw/code15/exams_part14.hdf5 data/raw/code15/exams_part15.hdf5 \
	data/raw/code15/exams_part16.hdf5 data/raw/code15/exams_part17.hdf5 \
	-d data/raw/code15/exams.csv \
	-l data/raw/code15/code15_chagas_labels.csv \
	-o data/training_data/code15_wfdb/exams_part0 data/training_data/code15_wfdb/exams_part1 \
	data/training_data/code15_wfdb/exams_part2 data/training_data/code15_wfdb/exams_part3 \
	data/training_data/code15_wfdb/exams_part4 data/training_data/code15_wfdb/exams_part5 \
	data/training_data/code15_wfdb/exams_part6 data/training_data/code15_wfdb/exams_part7 \
	data/training_data/code15_wfdb/exams_part8 data/training_data/code15_wfdb/exams_part9 \
	data/training_data/code15_wfdb/exams_part10 data/training_data/code15_wfdb/exams_part11 \
	data/training_data/code15_wfdb/exams_part12 data/training_data/code15_wfdb/exams_part13 \
	data/training_data/code15_wfdb/exams_part14 data/training_data/code15_wfdb/exams_part15 \
	data/training_data/code15_wfdb/exams_part16 data/training_data/code15_wfdb/exams_part17 \

python -m prep_meta_data -d data/raw