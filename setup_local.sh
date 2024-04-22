#!/bin/bash
# Assumes you have conda installed

env_name="llm-train"
python_version="3.10"

#############################################################################################
# load cuda 11.8
#############################################################################################
# check if module cuda/11.8 is loaded
if ! module list 2>&1 | grep -q "cuda/11.8"; then
    module load cuda/11.8
else
    echo "cuda/11.8 is already loaded"
fi

#############################################################################################
# create env
#############################################################################################
# delete env first if already exists
if conda env list | grep -q $env_name; then
    # get current env name, remember * is special character in grep
    current_env=`echo $CONDA_DEFAULT_ENV`
    # deactivate env if already active
    if [ "$current_env" == "$env_name" ]; then
        echo "Deactivating $env_name"
        conda deactivate
    fi
    # remove env
    conda env remove -n $env_name -y
fi

echo "Creating conda env"
conda create -n $env_name python=$python_version -y
conda activate $env_name

pip install --no-cache-dir -r requirements.py
MAX_JOBS=4 pip install --no-cache-dir flash-attn==2.3.6 --no-build-isolation # MAX_JOBS=4 if RAM<96GB
pip install --no-cache-dir jupyter openai seaborn matplotlib wandb