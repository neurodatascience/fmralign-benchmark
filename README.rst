    .. -*- mode: rst -*-

.. image:: https://img.shields.io/badge/License-BSD%202--Clause-orange.svg
   :target: https://opensource.org/licenses/BSD-2-Clause
   :alt: BSD-2-Clause License


fmralign-benchmark-mockup
=========================

This repository contains code for reproducing results from the recent paper:

| *An empirical evaluation of functional alignment using inter-subject decoding.*
| Thomas Bazeille*, Elizabeth DuPre*, Jean-Baptiste Poline, & Bertrand Thirion.
| doi: 10.1101/2020.XX.XX.XXXXXX

It makes code available to replicate all experiments on a subset of the data :
the IBC dataset comprising 2 of the 5 decoding tasks studied.

Requirements
-------------

* `fmralign <https://parietal-inria.github.io/fmralign-docs/>`_
* `nibabel>=3.1 <http://nipy.org/nibabel/>`_
* `numpy>=1.18 <http://www.numpy.org/>`_
* `matplotlib <https://matplotlib.org/>`_
* `pandas <https://pandas.pydata.org/>`_
* `scipy <https://www.scipy.org/>`_
* `scikit-learn <http://scikit-learn.org/stable/>`_

Replication of the main benchmarking results already require ** significant
computational power (around 150+CPU hours) and RAM (50+Go)**. Parallelism is provided in
this repository to ease replication on clusters, if available.

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

** Modify the configuration file ``fmralignbench.py``** to provide a root folder
where data will be downloaded and the number of CPU cores usable for replication.

The folder ``experiments`` contains code to re-execute all of the main and
supplemental experiments included in the manuscript.

* ``experiment_1-2.py`` replicates the whole-brain and ROI-based level of analysis (170 CPU hours, 30+Go RAM)

* ``experiment_3.py`` replicates the qualitative comparison of alignment methods on IBC data (a few CPU hours, few hours, 30+Go RAM)
* ``supplementary_2-3.py`` replicates the supplemental experiments investigating the impact of parcellation and smoothing
* ``supplementary_4.py`` replicates the supplemental experiment comparing surface- and volume-based results for piecewise Procrustes (13Go download(high-resolution data), 45 CPU hours, 60Go RAM)



Experiment 1
---------------

.. image:: figures/experiment1.png
   :width: 400


Experiment 2
---------------

.. image:: figures/experiment2.png
   :width: 400

Experiment 3
---------------

.. image:: figures/experiment3_qualitative.png
   :width: 400

Supplementary results
---------------

|pic1| any text |pic2|

.. |pic1| image:: figures/experiment_1_within_decoding.png
   :width: 45%

.. |pic2| image:: figures/supplementary_1_roi_minus_fullbrain.png
   :width: 45%
