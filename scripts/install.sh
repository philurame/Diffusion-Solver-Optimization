conda create --prefix ../env python=3.10 -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ../env

pip install -U diffusers transformers accelerate safetensors sentencepiece pillow
pip install torchmetrics mlflow omegaconf setproctitle matplotlib hpsv2x==1.2.0

pip install ipykernel
python -m ipykernel install --user --name philurame_sd15 --display-name "Python (philurame_sd15)"