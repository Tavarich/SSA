

## Installation

An example conda environment setup

```shell
conda create --name ssa python=3.11 -y
conda activate ssa
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -U opencv-python

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e . --no-build-isolation

cd ..
git clone https://github.com/Tavarich/SSA.git
cd SSA
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh

pip install open_clip_torch==2.24
```

Other Pytorch version (with correspond CUDA and python package version) might be also OK.

