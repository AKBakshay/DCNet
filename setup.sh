export PYTHONPATH='.'
mkdir src/pretrain
mkdir src/pretrain/model
python3 src/model_setup/download_default_files.py
tar -xvf dataset.tar.gz
rm dataset.tar.gz