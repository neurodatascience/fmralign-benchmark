fmralign-benchmark-mockup
=========================

This repository contains code for reproducing results from the recent paper:

| *An empirical evaluation of functional alignment using inter-subject decoding.*
| Thomas Bazeille*, Elizabeth DuPre*, Jean-Baptiste Poline, & Bertrand Thirion.
| doi: 10.1101/2020.XX.XX.XXXXXX


Dependencies
------------

* fmralign
* nibabel
* numpy
* matplotlib
* pandas
* scipy
* scikit-learn

Installation
------------

First, make sure you have installed all the dependencies listed above.
Then you can install fmralign-benchmark by running the following commands::

    git clone https://github.com/thomasbazeille/fmralign-benchmark-mockup
    cd fmralign-benchmark-mockup
    pip install .

To reproduce results from the Searchlight Hyperalignment method, you'll also
need to install ``PyMVPA``.
You can do so with the following commands, *assuming you are still in the
``fmralign-benchmark-mockup`` directory*::

    cd ..
    git clone https://github.com/PyMPVA/PyMVPA
    cd PyMVPA
    pip install -e .

You can confirm that both packages have installed correctly by opening a Python
terminal and running the following commands::

    import fmralignbench
    import mvpa2


Getting started
---------------