import os
import shutil
from distutils.command.clean import clean as Clean
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from README.rst
with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]

cmdclass = {}


# Custom clean command to remove build artifacts
class CleanCommand(Clean):
    description = "Remove build artifacts from the source tree"

    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, "PKG-INFO"))
        if remove_c_files:
            print("Will remove generated .c files")
        if os.path.exists("build"):
            shutil.rmtree("build")
        for dirpath, dirnames, filenames in os.walk("sklearn"):
            for filename in filenames:
                if any(
                    filename.endswith(suffix)
                    for suffix in (".so", ".pyd", ".dll", ".pyc")
                ):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in [".c", ".cpp"]:
                    pyx_file = str.replace(filename, extension, ".pyx")
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == "__pycache__":
                    shutil.rmtree(os.path.join(dirpath, dirname))


cmdclass.update({"clean": CleanCommand})


setup(
    name="torchensemble",
    maintainer="Yi-Xuan Xu",
    maintainer_email="xuyx@lamda.nju.edu.cn",
    description=(
        "A unified ensemble framework for PyTorch to improve the performance"
        " and robustness of your deep learning model"
    ),
    license="BSD 3-Clause",
    url="https://github.com/TorchEnsemble-Community/Ensemble-Pytorch",
    project_urls={
        "Bug Tracker": "https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/issues",
        "Documentation": "https://ensemble-pytorch.readthedocs.io",
        "Source Code": "https://github.com/TorchEnsemble-Community/Ensemble-Pytorch",
    },
    version="0.1.9",
    long_description=long_description,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords=["Deep Learning", "PyTorch", "Ensemble Learning"],
    packages=find_packages(),
    cmdclass=cmdclass,
    python_requires=">=3.6",
    install_requires=install_requires,
)
