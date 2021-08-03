export PYTHONPATH='.'
mkdir src/pretrain
mkdir src/pretrain/model
mkdir src/output
mkdir src/output/model_parameters
mkdir src/output/predictions
python3 src/model_setup/download_default_files.py
tar -xvf src/dataset.tar.gz
rm src/dataset.tar.gz