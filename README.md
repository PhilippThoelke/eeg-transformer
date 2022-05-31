# EEG Transformer
A Transformer architecture for classification of raw EEG signals, including several visualizations of attention weights.

## Installation
Clone this repository and install the dependencies in your Python environment.

For example:
```bash
git clone git@github.com:PhilippThoelke/eeg-transformer.git
cd eeg-transformer
pip install requirements.txt
```

## Dataset
We provide a download script ([`src/download_data.py`](https://github.com/PhilippThoelke/eeg-transformer/blob/main/src/download_data.py)) for the [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/), which combines the specified runs into a single memory mapped NumPy file and a CSV file containing labels and subject information. By default the script will save the dataset files in a directory called `data` but you can change this by editing the `result_dir` variable at the top of the script. You can additionally restrict the script to only download parts of the dataset, choose normalization and epoch duration by editing the `target_type`, `normalize_epochs` and `epoch_duration` fields respectively. By default, all tasks from the dataset are combined but it is possible to select individual tasks using the training script.

When you are done editing, simply run the script via
```bash
python src/download_data.py
```

## Training
Training the model is started by running [`src/train.py`](https://github.com/PhilippThoelke/eeg-transformer/blob/main/src/train.py). There is a wide range of different hyperparameters you can choose from. Run `python src/train.py --help` to get a list of possible arguments and their descriptions. There are 4 required arguments, namely `--data-path`, `--label-path`, `--epoch-length` and `--num-channels`, which correspond to the paths to the memory mapped dataset and CSV label files, the number of steps per epoch and number of channels in the raw EEG respectively.

To train on the dataset described in the dataset section, it would be enough to specify only the 4 required arguments but be recommend excluding the three reference channels and low-pass filtering the data. To train on the eyes open vs eyes closed condition for example, run this command:
```bash
python src/train.py --data-path path/to/raw-dataset.dat --label-path path/to/label-dataset.csv --epoch-length 320 --num-channels 64 --conditions eyes-open eyes-closed --ignore-channels 42 43 63 --sample-rate 160 --low-pass 30
```