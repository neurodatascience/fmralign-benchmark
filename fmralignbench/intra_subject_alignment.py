# *- encoding: utf-8 -*-
"""IntraSubjectAlignment class is a hack of fmralign pairwise_alignment to replicate Tavor 2016

Care is needed, few changes but tricky ones :
* X and Y have different meanings in fit (different sets of contrasts of same subject)
* X_i and Y_i are transposed before fitting Ridge compared to alignment to predict
    Y contrasts values from X ones linearly and uniformly over voxels in a region
* To avoid changes in the code, multisubject case is done through ensembling IntraSubjectAlignment
 fitted individually.

Author : T. Bazeille, B. Thirion
"""
import warnings
import os
import nibabel as nib
import numpy as np
from fmralign.pairwise_alignment import PairwiseAlignment, generate_Xi_Yi
from joblib import Parallel, delayed, Memory
from nilearn.input_data.masker_validation import check_embedded_nifti_masker
from sklearn.model_selection import ShuffleSplit
from sklearn.base import clone
from fmralign._utils import _make_parcellation,_intersect_clustering_mask
from fmralign.template_alignment import _rescaled_euclidean_mean
from fmralign.alignment_methods import RidgeAlignment, Identity, Hungarian, \
    ScaledOrthogonalAlignment, OptimalTransportAlignment, DiagonalAlignment, Alignment
from sklearn.linear_model import Ridge


class RidgeAl(Alignment):
    """ Compute a scikit-estimator R using a mixing matrix M s.t Frobenius \
    norm || XM - Y ||^2 + alpha * ||M||^2 is minimized with cross-validation

    Parameters
    ----------
    R : scikit-estimator from sklearn.linear_model.Ridge
        with methods fit, predict

    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, Y):
        """ Fit R s.t. || XR - Y ||^2 + alpha ||R||^2 is minimized with cv

        Parameters
        -----------
        X: (n_samples, n_features) nd array
            source data
        Y: (n_samples, n_features) nd array
            target data
        """
        self.R = Ridge(alpha=self.alpha, fit_intercept=True,
                       normalize=False)
        self.R.fit(X, Y)
        return self

    def transform(self, X):
        """Transform X using optimal transform computed during fit.
        """
        return self.R.predict(X)


def fit_one_piece_intra(X_i, Y_i, alignment_method):
    """ Align source and target data in one piece i, X_i and Y_i, using
    alignment method and learn transformation to map X to Y.

    Parameters
    ----------
    X_i: ndarray
        Source data for piece i (shape : n_samples, n_features)
    Y_i: ndarray
        Target data for piece i (shape : n_samples, n_features)
    alignment_method: string
        Algorithm used to perform alignment between X_i and Y_i :
        - either 'identity', 'scaled_orthogonal', 'optimal_transport',
        'ridge_cv', 'permutation', 'diagonal'
        - or an instance of one of alignment classes
            (imported from functional_alignment.alignment_methods)
    Returns
    -------
    alignment_algo
        Instance of alignment estimator class fitted for X_i, Y_i
    """

    if alignment_method == 'identity':
        alignment_algo = Identity()
    elif alignment_method == 'scaled_orthogonal':
        alignment_algo = ScaledOrthogonalAlignment()
    elif alignment_method == 'ridge_cv':
        alignment_algo = RidgeAlignment()
    elif alignment_method == 'permutation':
        alignment_algo = Hungarian()
    elif alignment_method == 'optimal_transport':
        alignment_algo = OptimalTransportAlignment()
    elif alignment_method == 'diagonal':
        alignment_algo = DiagonalAlignment()
    elif isinstance(alignment_method, (Identity, ScaledOrthogonalAlignment,
                                       RidgeAlignment, Hungarian,
                                       OptimalTransportAlignment,
                                       DiagonalAlignment)):
        alignment_algo = clone(alignment_method)

    if isinstance(alignment_algo, RidgeAlignment):
        if isinstance(alignment_algo.cv, int):
            if alignment_algo.cv > X_i.shape[0]:
                warnings.warn(
                    "Too few samples for RidgeCV, Ridge(alpha=1) fitted instead")
                alignment_algo = RidgeAl()
    try:
        alignment_algo.fit(X_i, Y_i)
    except UnboundLocalError:
        warn_msg = ("{} is an unrecognized ".format(alignment_method) +
                    "alignment method. Please provide a recognized " +
                    "alignment method.")
        raise NotImplementedError(warn_msg)
    return alignment_algo


def piecewise_transform_intra(labels, estimators, X, Y_shape):
    """ Apply a piecewise transform to X:
    Parameters
    ----------
    labels: list of ints (len n_features)
        Parcellation of features in clusters
    estimators: list of estimators with transform() method
        I-th estimator will be applied on the i-th cluster of features
    X: nd array (n_features, n_samples)
        Data to transform

    Returns
    -------
    X_transform: nd array (n_features, n_samples)
        Transformed data
    """

    unique_labels = np.unique(labels)

    X_transform = np.zeros(Y_shape)
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        X_transform[:, labels == label] = estimators[i].transform(
            X[:, labels == label].T).T
    return X_transform


def fit_one_parcellation_intra(X_, Y_, alignment_method, masker, n_pieces,
                               clustering, clustering_index, mem,
                               n_jobs, verbose):
    """ Copy from fmralign.pairwise_alignment except for transposition of fit_one_piece input
    """

    labels = _make_parcellation(
        X_, clustering_index, clustering, n_pieces, masker)

    fit = Parallel(n_jobs, prefer="threads", verbose=verbose)(
        delayed(fit_one_piece_intra)(
            X_i.T, Y_i.T, alignment_method
        ) for X_i, Y_i in generate_Xi_Yi(labels, X_, Y_, masker, verbose)
    )

    return labels, fit


class IntraSubjectAlignment(PairwiseAlignment):
    """ This class replicates Tavor 2016 implementation reusing as much as possible
    code from fmralign that was designed to do hyperalignment for this exact purposes.

    Instead of aligning a pair of subject, we search regularities inside single
    subject fixed contrasts given, and some that we search.
    """

    def __init__(self, alignment_method="ridge_cv", n_pieces=1,
                 clustering='kmeans', n_bags=1, mask=None,
                 smoothing_fwhm=None, standardize=None, detrend=False,
                 target_affine=None, target_shape=None, low_pass=None,
                 high_pass=None, t_r=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0):
        super().__init__(
            alignment_method=alignment_method, n_pieces=n_pieces,
            clustering=clustering, n_bags=n_bags, mask=mask,
            smoothing_fwhm=smoothing_fwhm, standardize=standardize, detrend=detrend,
            target_affine=target_affine, target_shape=target_shape, low_pass=low_pass,
            high_pass=high_pass, t_r=t_r, memory=memory, memory_level=memory_level,
            n_jobs=n_jobs, verbose=verbose)

    def fit(self, X, Y):
        """Fit data X and Y and learn transformation to map X to Y

        Almost the same as pairwise align except for commented line
        Parameters
        ----------
        X: Niimg-like object
           See http://nilearn.github.io/manipulating_images/input_output.html
           source data
        Y: Niimg-like object
           See http://nilearn.github.io/manipulating_images/input_output.html
           target data

        Returns
        -------
        self
        """
        self.masker_ = check_embedded_nifti_masker(self)
        self.masker_.n_jobs = 1  # self.n_jobs
        # Avoid warning with imgs != None
        # if masker_ has been provided a mask_img
        if self.masker_.mask_img is None:
            self.masker_.fit([X])
        else:
            self.masker_.fit()

        # miss concatenation, transpose
        X_ = self.masker_.transform(X)
        Y_ = self.masker_.transform(Y)

        self.Y_shape = Y_.shape

        self.fit_, self.labels_ = [], []
        rs = ShuffleSplit(n_splits=self.n_bags,
                          test_size=.8, random_state=0)

        outputs = Parallel(n_jobs=self.n_jobs, prefer="threads",
                           verbose=self.verbose)(
            delayed(fit_one_parcellation_intra)(
                self.masker_.inverse_transform(X_), self.masker_.inverse_transform(
                    Y_), self.alignment_method, self.masker_,
                self.n_pieces, self.clustering, clustering_index,
                self.memory, self.n_jobs, self.verbose)
            for clustering_index, _ in rs.split(X_))
        # split on X_.T        X (n_features, n_samples)
        # call a different fit_one_parcellation function
        self.labels_ = [output[0] for output in outputs]
        self.fit_ = [output[1] for output in outputs]
        return self

    def transform(self, X):
        """Predict data from X
        Almost the same as pairwise align except for commented line
        Parameters
        ----------
        X: Niimg-like object
           See http://nilearn.github.io/manipulating_images/input_output.html
           source data

        Returns
        -------
        X_transform: Niimg-like object
           See http://nilearn.github.io/manipulating_images/input_output.html
           predicted data
        """
        X_ = self.masker_.transform(X)

        X_transform = np.zeros(self.Y_shape)
        for i in range(self.n_bags):
            # give transposed data to ridge compared to pairwise
            X_transform += piecewise_transform_intra(
                self.labels_[i], self.fit_[i], X_, self.Y_shape)

        X_transform /= self.n_bags
        # And remove final transposition
        return self.masker_.inverse_transform(X_transform)


class EnsembledSubjectsIntraAlignment(PairwiseAlignment):
    def __init__(self, alignment_method="ridge_cv", n_pieces=1,
                 clustering='kmeans', n_bags=1, mask=None,
                 smoothing_fwhm=None, standardize=None, detrend=False,
                 target_affine=None, target_shape=None, low_pass=None,
                 high_pass=None, t_r=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0):
        super().__init__(
            alignment_method=alignment_method, n_pieces=n_pieces,
            clustering=clustering, n_bags=n_bags, mask=mask,
            smoothing_fwhm=smoothing_fwhm, standardize=standardize, detrend=detrend,
            target_affine=target_affine, target_shape=target_shape, low_pass=low_pass,
            high_pass=high_pass, t_r=t_r, memory=memory, memory_level=memory_level,
            n_jobs=n_jobs, verbose=verbose)

    def fit(self, X, Y):
        ''' X and Y must be lists of equal length (number of subjects)
        Inside each element may lists or a Niimgs (all of the same len / shape)
        '''
        self.masker_ = check_embedded_nifti_masker(self)
        self.masker_.n_jobs = self.n_jobs

        if type(self.clustering) == nib.nifti1.Nifti1Image or os.path.isfile(self.clustering):
            # check that clustering provided fills the mask, if not, reduce the mask
            if 0 in self.masker_.transform(self.clustering):
                reduced_mask = _intersect_clustering_mask(
                    self.clustering, self.masker_.mask_img)
                self.mask = reduced_mask
                self.masker_ = check_embedded_nifti_masker(self)
                self.masker_.n_jobs = self.n_jobs
                self.masker_.fit()
                warnings.warn(
                    "Mask used was bigger than clustering provided. Its intersection with the clustering was used instead.")
        # Avoid warning with imgs != None
        # if masker_ has been provided a mask_img
        if self.masker_.mask_img is None:
            self.masker_.fit([X])
        else:
            self.masker_.fit()

        self.fitted_intra = []
        for X_sub, Y_sub in zip(X, Y):
            intra_align = IntraSubjectAlignment(alignment_method=self.alignment_method, n_pieces=self.n_pieces,
                                                clustering=self.clustering, n_bags=self.n_bags, mask=self.masker_,
                                                smoothing_fwhm=self.smoothing_fwhm, standardize=self.standardize, detrend=self.detrend,
                                                target_affine=self.target_affine, target_shape=self.target_shape, low_pass=self.low_pass,
                                                high_pass=self.high_pass, t_r=self.t_r, memory=self.memory, memory_level=self.memory_level,
                                                n_jobs=self.n_jobs, verbose=self.verbose)
            intra_align.fit(X_sub, Y_sub)
            self.fitted_intra.append(intra_align)
        self.Y_shape = intra_align.Y_shape
        return self

    def transform(self, X):
        ''' X is a list. Each elemen of it is same shape as one element in fit(X) list
        Return a list Y of the same form as one element for each test subject.
        '''
        Y = []
        # if we fitted n subjects we have n models in self.fitted_intra
        # for each new subject each sub, we ensemble the prediction of each model
        for X_sub in X:
            Y.append(_rescaled_euclidean_mean([intra_align.transform(
                X_sub) for intra_align in self.fitted_intra], self.masker_))
        return Y
