#PTB-XL
wget -r -N -c -np -nH --cut-dirs=3 --reject "index.html*" "https://physionet.org/files/ptb-xl/1.0.3/" -P data/raw/ptbxl/

#CODE-15%
mkdir -p data/raw/code15

# download CSV
wget -c --content-disposition "https://zenodo.org/records/4916206/files/exams.csv?download=1" -P data/raw/code15/
wget -c --content-disposition "https://moody-challenge.physionet.org/2025/data/code15_chagas_labels.zip" -P data/raw/code15/
unzip "data/raw/code15/code15_chagas_labels.zip" -d data/raw/code15/
rm data/code15/code15_chagas_labels.zip

# download all parts
for i in {0..17}; do
  wget -c --content-disposition "https://zenodo.org/records/4916206/files/exams_part${i}.zip?download=1" -P data/raw/code15/
done

#extract data and clean up folder
for i in {0..17}; do
  unzip "data/raw/code15/exams_part${i}.zip" -d data/raw/code15/
  rm "data/raw/code15/exams_part${i}.zip"
done

# SaMi-Trop
wget -c --content-disposition "https://zenodo.org/records/4905618/files/exams.zip?download=1" -P data/raw/samitrop/
wget -c --content-disposition "https://zenodo.org/records/4905618/files/exams.csv?download=1" -P data/raw/samitrop/
unzip data/raw/samitrop/exams.zip -d data/raw/samitrop/
rm data/raw/samitrop/exams.zip