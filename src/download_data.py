from os.path import join
import numpy as np
import pandas as pd
import mne
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci


# path to directory where the resulting dataset files will be stored
result_dir = "data/"
# which tasks should be extracted from the dataset, can be "all" or a list of multiple:
# baseline-eyes, fist-motion, fist-imagination, fist_feet-motion, fist_feet-imagination
target_type = "all"
# if true, compute z-scores for each epoch individually
normalize_epochs = False
# length of one epoch in seconds
epoch_duration = 2


def extract_baseline_eyes(subjects, runs, epoch_duration):
    epochs = []
    subject_labels = []
    labels = []
    run2label = {1: "eyes-open", 2: "eyes-closed"}
    for subject in subjects:
        for run in runs:
            raw_fnames = eegbci.load_data(subject, run, update_path=False)
            raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
            raw = concatenate_raws(raws)
            data = raw.get_data()
            epoch_steps = int(epoch_duration * raw.info["sfreq"])

            offset = 0
            for _ in range(data.shape[1] // epoch_steps):
                epoch = data[:, offset : offset + epoch_steps].astype(np.float32)
                if normalize_epochs:
                    mean = epoch.mean(axis=1, keepdims=True)
                    std = epoch.std(axis=1, keepdims=True)
                    epoch = (epoch - mean) / std
                epochs.append(epoch)
                subject_labels.append(subject)
                labels.append(run2label[run])
                offset += epoch_steps
    return epochs, subject_labels, labels


def extract_task(subjects, runs, epoch_duration, label_names):
    epochs = []
    subject_labels = []
    labels = []
    for subject in subjects:
        raw_fnames = eegbci.load_data(subject, runs, update_path=False)
        raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
        raw = concatenate_raws(raws)
        epoch_steps = int(epoch_duration * raw.info["sfreq"])

        curr_epochs = mne.Epochs(
            raw,
            *mne.events_from_annotations(raw),
            tmin=0,
            tmax=epoch_duration,
            baseline=None,
            preload=True,
        )
        if curr_epochs.get_data().shape[-1] != (epoch_duration * 160 + 1):
            # skip runs which don't have a sampling frequency of 160
            continue

        def process(xs, normalize=False, n_steps=-1):
            xs = xs[:, :, :n_steps]
            if not normalize:
                return list(xs)
            return [
                (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
                for x in xs
            ]

        # condition 1
        epochs.extend(
            process(
                curr_epochs["T1"].get_data(),
                normalize=normalize_epochs,
                n_steps=epoch_steps,
            )
        )
        subject_labels.extend([subject] * len(curr_epochs["T1"]))
        labels.extend([label_names[0]] * len(curr_epochs["T1"]))
        # condition 2
        epochs.extend(
            process(
                curr_epochs["T2"].get_data(),
                normalize=normalize_epochs,
                n_steps=epoch_steps,
            )
        )
        subject_labels.extend([subject] * len(curr_epochs["T2"]))
        labels.extend([label_names[1]] * len(curr_epochs["T2"]))
    return epochs, subject_labels, labels


def extract_epochs(targets, subjects=range(1, 110), epoch_duration=5):
    if targets == "all":
        targets = [
            "baseline-eyes",
            "fist-motion",
            "fist-imagination",
            "fist_feet-motion",
            "fist_feet-imagination",
        ]

    if isinstance(targets, str):
        targets = [targets]

    epochs, subject_labels, labels = [], [], []
    for target in targets:
        if target == "baseline-eyes":
            ep, subj, lab = extract_baseline_eyes(
                subjects=subjects, runs=[1, 2], epoch_duration=epoch_duration
            )
        elif target == "fist-motion":
            ep, subj, lab = extract_task(
                subjects=subjects,
                runs=[3, 7, 11],
                epoch_duration=epoch_duration,
                label_names=["left-fist-move", "right-fist-move"],
            )
        elif target == "fist-imagination":
            ep, subj, lab = extract_task(
                subjects=subjects,
                runs=[4, 8, 12],
                epoch_duration=epoch_duration,
                label_names=["left-fist-imag", "right-fist-imag"],
            )
        elif target == "fist_feet-motion":
            ep, subj, lab = extract_task(
                subjects=subjects,
                runs=[5, 9, 13],
                epoch_duration=epoch_duration,
                label_names=["fists-move", "feet-move"],
            )
        elif target == "fist_feet-imagination":
            ep, subj, lab = extract_task(
                subjects=subjects,
                runs=[6, 10, 14],
                epoch_duration=epoch_duration,
                label_names=["fists-imag", "feet-imag"],
            )
        else:
            raise ValueError(f"Unrecognized target {target}")

        epochs.extend(ep)
        subject_labels.extend(subj)
        labels.extend(lab)
    return epochs, subject_labels, labels


epochs, subject_labels, labels = extract_epochs(
    target_type, epoch_duration=epoch_duration
)

shape = len(epochs), epochs[0].shape[1], epochs[0].shape[0]
fname = (
    f"nsamp_{shape[0]}-"
    f"eplen_{shape[1]}"
    f"{'-norm' if normalize_epochs else ''}-"
    f"example_{'-'.join(target_type) if isinstance(target_type, list) else target_type}"
)

print("\nSaving raw data...", end="")
file = np.memmap(
    join(result_dir, "raw-" + fname + ".dat"), mode="w+", dtype=np.float32, shape=shape
)
meta_info = pd.DataFrame(
    index=np.arange(shape[0], dtype=int),
    columns=["subject", "stage", "condition"],
    dtype=str,
)

for i in range(shape[0]):
    file[i] = epochs[i].T
    file.flush()
    meta_info.iloc[i] = [subject_labels[i], -1, labels[i]]
print("done")

print("Saving metadata...", end="")
meta_info.to_csv(join(result_dir, "label-" + fname + ".csv"))
print("done")
