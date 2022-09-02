# Analysis of Transformer attention in EEG signal classification
A Transformer architecture for classification of raw EEG signals, including several visualizations of attention weights.

This work was published at the Conference on Cognitive Computational Neuroscience 2022: [https://doi.org/10.32470/CCN.2022.1240-0](https://doi.org/10.32470/CCN.2022.1240-0)

## Installation
Clone this repository and install the dependencies in your Python environment of choice.

For example:
```bash
git clone git@github.com:PhilippThoelke/eeg-transformer.git
cd eeg-transformer
pip install -r requirements.txt
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
Training progress is logged in a directory called `lightning_logs`, which contains subdirectories for individual training runs. Each run contains an `hparams.yaml` file with a list of hyperparameters, a `splits.pt` file containing indices of the training and validation set, an `events.out.tfevens.*` file with Tensorboard compatible training metrics and a `checkpoints` directory with model checkpoints. You can view the training progress visually by running `tensorboard --logdir lightning_logs/`. After running this you can access a graphical view of the training progress by accessing `localhost:6006` in a webbrowser.

## Analysis
To load a model checkpoint for analysis, you can use the Lightning Module's `load_from_checkpoint` function.
```python
from module import TransformerModule
model = TransformerModule.load_from_checkpoint("path/to/model.ckpt")
```
The model's `forward` function takes expects a tensor containing the raw EEG with shape batch-size x time-steps x channels (the batch dimension is optional). It will return class probabilities with shape batch-size x num-classes. Example with random data:
```python
import torch
# batch-size=8, epoch-length=320, num-channels=64
x = torch.rand(8, 320, 64)
prediction = model(x)
# prediction.shape == (8, 2) for binary classification
```
We provide a context-manager for extracting attention weights during a forward call.
```python
from attention import Attention
# record attention weights
with Attention(model) as a:
    prediction = model(x)
attn = a.get()
```
The resulting `attn` tensor has shape batch-size x num-layers x num-heads x num-tokens x num-tokens. You can combine the attention weights of all layers and heads using attention rollout:
```python
from attention import rollout
token_attn = rollout(attn)
class_attn = rollout(attn, only_class=True)
```
The resulting `token_attn` tensor will have the shape batch-size x num-tokens x num-tokens and represents the full attention matrix between all pairs of tokens. The condensed `class_attn` tensor (shape batch-size x num-tokens) contains normalized attention weights from all input tokens towards the class token and can be thought of as a feature importance metric.

The [`src/attention.py`](https://github.com/PhilippThoelke/eeg-transformer/blob/main/src/attention.py) script can also be used from the command line. For a list of arguments run `python src/attention.py --help`. When run from the command line, the script will extract the attention matrices from the validation set of the specified dataset and store attention, as well as some other useful metrics, in a file called `attention.pt` in the model's log directory. You can load this file using
```python
attention = torch.load("path/to/attention.pt")
attn, confidence, pred, labels, stages, subjects, hparams, condition_mapping, stage_mapping, subject_mapping = attention
```
The attention weights can be visualized in the form of topomaps and interactive 3D plots using the [`notebooks/AttentionFigures.ipynb`](https://github.com/PhilippThoelke/eeg-transformer/blob/main/notebooks/AttentionFigures.ipynb) notebook. Simply adjust the `model_dir` variable at the top to point towards your model's log dir or a directory containing the log directories of multiple models.

## Examples figures
### Visualization of attention weights in the form of topomaps (attention directed at the class token):
![attn-bands-1](https://user-images.githubusercontent.com/36135990/171483279-130337c0-a7dd-4723-a70c-fafd9d2132d0.png)
### Three dimensional visualization of attention weights between EEG channels (eyes closed condition):
![eyes-closed-3d-1](https://user-images.githubusercontent.com/36135990/171483302-6fed84e8-8974-4552-9316-366249cfd575.png)
