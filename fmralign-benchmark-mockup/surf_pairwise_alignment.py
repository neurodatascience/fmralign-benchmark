# -*- coding: utf-8 -*-
""" Module for surface pairwise functional alignment
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import (delayed, Parallel)
import nibabel as nib
from nilearn import signal
from fmralign._utils import piecewise_transform
from fmralign.pairwise_alignment import fit_one_piece


def generate_Xi_Yi(labels, X, Y, verbose):
    """ Generate source and target data X_i and Y_i for each piece i.

    Parameters
    ----------
    labels : list of ints (len n_features)
        Parcellation of features in clusters
    X: Niimg-like object
        Source data
    Y: Niimg-like object
        Target data
    masker: instance of NiftiMasker or MultiNiftiMasker
        Masker to be used on the data. For more information see:
        http://nilearn.github.io/manipulating_images/masker_objects.html
    verbose: integer, optional.
        Indicate the level of verbosity.

    Yields
    -------
    X_i: ndarray
        Source data for piece i (shape : n_samples, n_features)
    Y_i: ndarray
        Target data for piece i (shape : n_samples, n_features)

    """

    unique_labels = np.unique(labels)

    for k in range(len(unique_labels)):
        label = unique_labels[k]
        i = label == labels
        if (k + 1) % 25 == 0 and verbose > 0:
            print("Fitting parcel: " + str(k + 1) +
                  "/" + str(len(unique_labels)))
        # should return X_i Y_i
        yield X[:, i], Y[:, i]


def fit_parcellation(X_, Y_, alignment_method, clustering, n_jobs, verbose):
    """ Create one parcellation of n_pieces and align each source and target
    data in one piece i, X_i and Y_i, using alignment method
    and learn transformation to map X to Y.

    Parameters
    ----------
    X_: Niimg-like object
        Source data
    Y_: Niimg-like object
        Target data
    alignment_method: string
        algorithm used to perform alignment between each region of X_ and Y_
    clustering: string or GiftiImage
        Surf atlas with parcels
    n_jobs: integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.
    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed

    Returns
    -------
    alignment_algo
        Instance of alignment estimator class fitted for X_i, Y_i
    """
    # choose indexes maybe with index_img to not
    labels = clustering

    fit = Parallel(n_jobs, prefer="threads", verbose=verbose)(
        delayed(fit_one_piece)(
            X_i, Y_i, alignment_method
        ) for X_i, Y_i in generate_Xi_Yi(labels, X_, Y_, verbose)
    )

    return labels, fit


class SurfacePairwiseAlignment(BaseEstimator, TransformerMixin):
    """
    Decompose the source and target images into regions and align corresponding \
    regions independently.
    """

    def __init__(self, alignment_method, clustering, standardize=False, detrend=False, low_pass=None, high_pass=None, t_r=None, n_jobs=1, verbose=0):
        """
        If n_pieces > 1, decomposes the images into regions \
        and align each source/target region independantly.
        If n_bags > 1, this parcellation process is applied multiple time \
        and the resulting models are bagged.

        Parameters
        ----------
        alignment_method: string
            Algorithm used to perform alignment between X_i and Y_i :
            * either 'identity', 'scaled_orthogonal', 'optimal_transport',
            'ridge_cv', 'permutation', 'diagonal'
            * or an instance of one of alignment classes
            (imported from functional_alignment.alignment_methods)
        clustering : path or Gifti
            image used as predefined clustering
        standardize: boolean, optional (default = False)
            If standardize is True, the time-series are centered and normed:
            their variance is put to 1 in the time dimension.
        detrend: boolean, optional (default = None)
            This parameter is passed to nilearn.signal.clean.
            Please see the related documentation for details
        low_pass: None or float, optional (default = None)
            This parameter is passed to nilearn.signal.clean.
            Please see the related documentation for details.
        high_pass: None or float, optional (default = None)
            This parameter is passed to nilearn.signal.clean.
            Please see the related documentation for details.
        t_r: float, optional (default = None)
            This parameter is passed to nilearn.signal.clean.
            Please see the related documentation for details.
        n_jobs: integer, optional (default = 1)
            The number of CPUs to use to do the computation. -1 means
            'all CPUs', -2 'all CPUs but one', and so on.
        verbose: integer, optional (default = 0)
            Indicate the level of verbosity. By default, nothing is printed.
        """
        self.alignment_method = alignment_method
        self.clustering = clustering
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _load_clean_one(self, X):
        LEN_FSAV = 163842
        X_data = nib.load(X).agg_data()
        X_ = signal.clean(np.asarray(X_data), detrend=self.detrend, high_pass=self.high_pass,
                          low_pass=self.low_pass, standardize=self.standardize, t_r=self.t_r, ensure_finite=True)
        if X_.shape[0] == LEN_FSAV:
            return X_.T
        else:
            return X_

    def load_clean(self, X):
        if isinstance(X, (list, np.ndarray)):
            return np.vstack([self._load_clean_one(xi)for xi in X])
        else:
            return self._load_clean_one(X)

    def fit(self, X, Y):
        """Fit data X and Y and learn transformation to map X to Y

        Parameters
        ----------
        X: Niimg-like object
            Source data.

        Y: Niimg-like object
            Target data

        Returns
        -------
        self
        """

        #
        #self.masker_ = check_embedded_nifti_masker(self)

        X_ = self.load_clean(X)
        Y_ = self.load_clean(Y)

        self.labels_, self.fit_ = fit_parcellation(
            X_, Y_, self.alignment_method, self.clustering, self.n_jobs, self.verbose)
        # not list here unlike pairwise

        return self

    def transform(self, X):
        """Predict data from X

        Parameters
        ----------
        X: Niimg-like object
            Source data

        Returns
        -------
        X_transform: Niimg-like object
            Predicted data
        """
        X_ = self.load_clean(X)

        X_transform = piecewise_transform(
            self.labels_, self.fit_, X_)

        return X_transform

    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here. Will raise AttributeError if called.
        """
        raise AttributeError(
            "type object 'PairwiseAlignment' has no attribute 'fit_transform'")
