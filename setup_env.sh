#!/bin/bash

conda_install_path=$1
conda_env_name=$2

source $conda_install_path/etc/profile.d/conda.sh

echo "Creating Env"
conda create -y -n $conda_env_name python=3.7
conda activate $conda_env_name
which python

echo "Install Pytorch and TorchVision"
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2
pip install ninja

echo "Build Nvidia Apex"
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../
rm -rf apex/

echo "Install Tensorboard"
pip install tensorboard

echo "Build Deformable Convolution"
cd code_for_lit_s_m_b/mm_modules/DCN
python setup.py build install

cd ../../

pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
echo "Complete!"
