<div align="center">

## Optical Flow Estimation

</div>

Get W&B API key from https://wandb.ai/authorize

```bash
pip install -r requirements.txt
wandb login
```

# Dataset

MPI-Sintel dataset on http://sintel.is.tue.mpg.de/

```bash
# [project]
wget http://sintel.cs.washington.edu/MPI-Sintel-complete.zip
mkdir -p data/sintel
unzip MPI-Sintel-complete.zip -d data/sintel
```


# Data preprocessing

MPI-Sintel dataset path: [project] / data / sintel
Converted pt tensor path: [project] / data / sintel_pt

```bash
# [project] / data
mkdir -p sintel_pt/train
mkdir sintel_pt/valid
cd ../helper

# [project] / helper
python convert_sintel_pt.py
```

# Install dependencies

```bash
pip install -r requirements.txt
sudo apt update
sudo apt install -y libgl1-mesa-glx libglib2.0-0
```

# Train

```bash
python main.py
```