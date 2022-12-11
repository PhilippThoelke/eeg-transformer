import argparse
from os.path import join, basename
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import humanize


def get_hash(path):
    return path.split(".")[-2].split("_")[-1]


def main(result_name, base_dir, arg_hash):
    # collect dataset files
    label_files = sorted(glob(join(base_dir, "label-*.csv")))
    raw_files = sorted(glob(join(base_dir, "raw-*.dat")))

    assert len(label_files) == len(
        raw_files
    ), f"found {len(label_files)} label files but {len(raw_files)} raw files"

    if arg_hash is not None:
        # filter files based on their argument hash
        indices = [
            i for i in range(len(label_files)) if get_hash(label_files[i]) == arg_hash
        ]
    else:
        arg_hash = get_hash(label_files[indices[0]])
        indices = range(len(label_files))

    # get resulting dataset paths
    label_path = join(base_dir, f"label-{result_name}_{arg_hash}.csv")
    raw_path = join(base_dir, f"raw-{result_name}_{arg_hash}.dat")

    # ignore result files if present
    indices = [i for i in indices if basename(label_files[i]) != basename(label_path)]

    # check if label and raw file names match
    for i in indices:
        l = basename(label_files[i])[6:-4]
        r = basename(raw_files[i])[4:-4]
        assert l == r, f"label and raw file have different names (index {i})"
        assert get_hash(label_files[i]) == arg_hash, "found multiple argument hashes"

    # combine datasets
    frame_offset, sample_offset = 0, 0
    pbar = tqdm(indices, desc="combining datasets")
    for i in pbar:
        df = pd.read_csv(label_files[i], index_col=0)
        n_frames = df["stop_idx"].max()

        df["start_idx"] += frame_offset
        df["stop_idx"] += frame_offset
        df.index += sample_offset

        # append current metadata
        df.to_csv(
            label_path, mode="w" if frame_offset == 0 else "a", header=frame_offset == 0
        )

        # memory map current data file
        current = np.memmap(raw_files[i], mode="r", dtype=np.float32)
        if current.shape[0] < n_frames:
            raise RuntimeError(
                f"{basename(raw_files[i])} is corrupted, data file ({current.shape[0]}) "
                f"has less frames than specified by the metadata file ({n_frames})"
            )
        elif current.shape[0] > n_frames:
            print(
                f"{basename(raw_files[i])} is larger than the metadata file specifies, "
                "truncating the data file"
            )

        # append current data to the result file
        result = np.memmap(
            raw_path,
            mode="w+" if frame_offset == 0 else "r+",
            dtype=np.float32,
            offset=frame_offset * np.float32().nbytes,
            shape=n_frames,
        )
        result[:] = current

        # update offsets
        frame_offset += n_frames
        sample_offset += df.shape[0]

        # update progress bar
        pbar.set_postfix(
            dict(
                samples=sample_offset,
                size=humanize.naturalsize(frame_offset * 4, binary=True),
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-name",
        required=True,
        type=str,
        help="name of the resulting dataset",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        type=str,
        help="directory to search for datasets",
    )
    parser.add_argument(
        "--arg-hash",
        default=None,
        type=str,
        help="filter datasets by argument hash",
    )
    args = parser.parse_args()

    main(args.result_name, args.base_dir, args.arg_hash)
