!/bin/bash

env_name="llm-train"
python_version="3.10"
install_pkgs="git vim zip tmux"
parent_dir="/workspace"
conda_dir=${parent_dir}/miniconda3
user_id="aayushgargiitr@gmail.com"
user_name="Aayush Garg"


############################################################################################
# install essential packages
############################################################################################
apt-get update && apt-get install -y $install_pkgs


############################################################################################
# Setup conda
############################################################################################
eccho "Setting up conda"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${parent_dir}/miniconda.sh
bash ${parent_dir}/miniconda.sh -b -p ${conda_dir}
rm -rf ${parent_dir}/miniconda.sh
$conda_dir/bin/conda init bash


############################################################################################
# setup HF_HOME
############################################################################################
echo "Setting up HF_HOME"
mkdir -p /workspace/.cache
echo "export HF_HOME=/workspace/.cache" >> ~/.bashrc
source ~/.bashrc


#############################################################################################
# create env
#############################################################################################
echo "Creating conda env"
conda create -n $env_name python=$python_version -y
conda activate $env_name
pip install --no-cache-dir -r requirements.py
MAX_JOBS=4 pip install --no-cache-dir flash-attn==2.3.6 --no-build-isolation # MAX_JOBS=4 if RAM<96GB
pip install --no-cache-dir sentence-transformers faiss-cpu && \
    pip install --no-cache-dir jupyter wandb && \
    pip install --no-cache-dir openai seaborn matplotlib 


#############################################################################################
# set git user
#############################################################################################
git config --global user.email $user_id
git config --global user.name $user_name