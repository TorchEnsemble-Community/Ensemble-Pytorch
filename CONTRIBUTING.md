Welcome to the contributing guidelines of Ensemble-Pytorch!

Ensemble-Pytorch is a community-driven project and your contributions are highly welcome. Feel free to [raise an issue](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/issues/new/choose) if you have any problem. Below is the table of contents in this contributing guidelines.

- [Where to contribute](#where-to-contribute)
    - [Areas of contribution](#areas-of-contribution)
  - [Roadmap](#roadmap)
- [Acknowledging contributions](#acknowledging-contributions)
- [Installation](#installation)
- [Reporting bugs](#reporting-bugs)
- [Continuous integration](#continuous-integration)
  - [Unit testing](#unit-testing)
  - [Test coverage](#test-coverage)
- [Coding style](#coding-style)
- [API design](#api-design)
- [Documentation](#documentation)
- [Acknowledgement](#acknowledgement)

Where to contribute
-------------------

#### Areas of contribution

We value all kinds of contributions - not just code. The following table gives an overview of key contribution areas.

| Area          | Description                                                                                                  |
|---------------|--------------------------------------------------------------------------------------------------------------|
| algorithm     | collect and report novel algorithms relevant to torchensemble, mainly from top-tier conferences and journals |
| code          | implement algorithms, improve or add functionality, fix bugs                                                 |
| documentation | improve or add docstrings, user guide, introduction, and experiments                                         |
| testing       | report bugs, improve or add unit tests, improve the coverage of unit tests                                   |
| maintenance   | improve the development pipeline (continuous integration, Github bots), manage and view issues/pull-requests |
| api design    | design interfaces for estimators and other functionality                                                     |

### Roadmap

For a more detailed overview of current and future work, check out our [development roadmap](https://ensemble-pytorch.readthedocs.io/en/stable/roadmap.html).

Acknowledging contributions
---------------------------

We follow the [all-contributors specification](https://allcontributors.org) and recognise various types of contributions. Take a look at our past and current [contributors](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/blob/master/CONTRIBUTORS.md)!

If you are a new contributor, please make sure we add you to our list of contributors. All contributions are recorded in [.all-contributorsrc](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/blob/master/.all-contributorsrc).

If we have missed anything, please [raise an issue](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/issues/new/choose) or create a pull request!

Installation
------------

Please visit our [installation instructions](https://ensemble-pytorch.readthedocs.io/en/stable/quick_start.html#installation) to resolve any package issues and dependency errors. Feel free to [raise an issue](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/issues/new/choose) if the problem still exists.

Reporting bugs
--------------

We use GitHub issues to track all bugs and feature requests; feel free to open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the following rules before submitting:

- Verify that your issue is not being currently addressed by other [issues](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/issues) or [pull requests](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/pulls).
- Please ensure all code snippets and error messages are formatted in appropriate code blocks. See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).
- Please be specific about what estimators and/or functions are involved and the shape of the data, as appropriate; please include a [reproducible](https://stackoverflow.com/help/mcve) code snippet. If an exception is raised, please provide the traceback.

Continuous integration
----------------------

We use continuous integration services on GitHub to automatically check if new pull requests do not break anything and meet code quality standards. Please visit our [config files on continuous integration](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/tree/master/.github/workflows).

### Unit testing

We use [pytest](https://docs.pytest.org/en/latest/) for unit testing. To check if your code passes all tests locally, you need to install the development version of torchensemble and all extra dependencies.

1. Install all extra requirements from the root directory of torchensemble:

    ```bash
    pip install -r build_tools/requirements.txt
    ```

2. Install the development version of torchensemble:

    ```bash
    pip install -e .
    ```

3. To run all unit tests, run the following commend from the root directory:

    ```bash
    pytest ./
    ```

### Test coverage

We use [coverage](https://coverage.readthedocs.io/en/coverage-5.3/) via the [pytest-cov](https://github.com/pytest-dev/pytest-cov) plugin and [codecov](https://codecov.io) to measure and compare test coverage of our code.

Coding style
------------

We follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) coding guidelines. A good example can be found [here](https://gist.github.com/nateGeorge/5455d2c57fb33c1ae04706f2dc4fee01).

We use the [pre-commit](#Code-quality-checks) workflow together with [black](https://black.readthedocs.io/en/stable/) and [flake8](https://flake8.pycqa.org/en/latest/) to automatically apply consistent formatting and check whether your contribution complies with the PEP8 style. Besides, if you are using GitHub desktop on Windows, the following code snippet allows you to format and check the coding style manually.

``` bash
black --skip-string-normalization --config pyproject.toml ./
flake8 --filename=*.py torchensemble/
```

API design
----------

The general API design we use in torchensemble is similar to [scikit-learn](https://scikit-learn.org/) and [skorch](https://skorch.readthedocs.io/en/latest/?badge=latest).

For docstrings, we use the [numpy docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html\#docstring-standard).

Documentation
-------------

We use [sphinx](https://www.sphinx-doc.org/en/master/) and [readthedocs](https://readthedocs.org/projects/ensemble-pytorch/) to build and deploy our online documentation. You can find our online documentation [here](https://ensemble-pytorch.readthedocs.io).

The source files used to generate the online documentation can be found in [docs/](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/tree/master/docs). For example, the main configuration file for sphinx is [conf.py](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/blob/master/docs/conf.py) and the main page is [index.rst](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/blob/master/docs/index.rst). To add new pages, you need to add a new `.rst` file and include it in the `index.rst` file.

To build the documentation locally, you need to install a few extra dependencies listed in [docs/requirements.txt](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/blob/master/docs/requirements.txt).

1. Install extra requirements from the root directory, run:

    ```bash
    pip install -r docs/requirements.txt
    ```

2. To build the website locally, run:

    ```bash
    cd docs
    make html
    ```

You can find the generated files in the `Ensemble-Pytorch/docs/_build/` folder. To view the website, open `Ensemble-Pytorch/docs/_build/html/index.html` with your preferred web browser.

Acknowledgement
---------------

This CONTRIBUTING file is adapted from the [PyTorch](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md) and [Sktime](https://github.com/alan-turing-institute/sktime/blob/main/CONTRIBUTING.md).