# EEG Transformer

## Installation
1. clone this repository and checkout the `newarch` branch
```shell
git clone git@github.com:PhilippThoelke/eeg-transformer.git
cd eeg-transformer
git checkout newarch
```

2. install PyTorch according to https://pytorch.org/get-started/locally/, e.g. for CUDA 12.4 do
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

3. install the EEG Transformer package
```shell
pip install -e .
```
