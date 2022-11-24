import setuptools
import subprocess

try:
    version = (
        subprocess.check_output(["git", "describe", "--abbrev=0", "--tags"])
        .strip()
        .decode("utf-8")
    )
except:
    print("Failed to retrieve the current version, defaulting to 0")
    version = "0"

setuptools.setup(
    name="eegt",
    version=version,
    packages=setuptools.find_packages(),
    install_requires=[r.strip() for r in open("requirements.txt").readlines()],
)
