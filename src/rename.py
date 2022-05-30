from os import rename
from os.path import basename, dirname, join
import yaml
import glob

model_dir = "transformer/bands"

band_mapping = {
    (None, 4): "delta",
    (0.5, 4): "delta",
    (4, 8): "theta",
    (8, 12): "alpha",
    (12, 30): "beta",
    (None, 30): "combined",
    (0.5, 30): "combined",
}

for directory in map(
    dirname, glob.glob(join(dirname(__file__), model_dir, "*", "attention.pt"))
):
    with open(join(directory, "hparams.yaml"), "r") as f:
        args = yaml.load(f, yaml.FullLoader)
    low_pass = args["low_pass"] if args["low_pass"] is not None else None
    high_pass = args["high_pass"] if args["high_pass"] is not None else None

    try:
        name = band_mapping[high_pass, low_pass]
        current_indices = list(
            map(lambda x: int(x[-1]), glob.glob(join(dirname(directory), f"{name}*")))
        )
        if len(current_indices) == 0:
            new_name = name + "0"
        else:
            new_name = name + str(max(current_indices) + 1)

        print(f"Renaming {basename(directory)} to {new_name}")
        rename(directory, join(dirname(directory), new_name))
    except KeyError:
        print("no band name found for {high_pass} - {low_pass} Hz")
        continue
