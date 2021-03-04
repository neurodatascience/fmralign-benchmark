import os
import csv
import time
import numpy as np
import pandas as pd
from os.path import join as opj
from collections import namedtuple

from sklearn.base import clone
from sklearn.utils import Bunch

import nibabel as nib
from nilearn import signal
from nilearn.input_data import NiftiMasker
from nilearn.image import load_img
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from fmralign.pairwise_alignment import PairwiseAlignment
from fmralign.alignment_methods import OptimalTransportAlignment
from fmralignbench.surf_pairwise_alignment import SurfacePairwiseAlignment
from fmralignbench.intra_subject_alignment import IntraSubjectAlignment
from fmralignbench.fastsrm import FastSRM
from fmralignbench.conf import ROOT_FOLDER, N_JOBS

mask_gm = os.path.join(
    ROOT_FOLDER,
    'tpl-MNI152NLin2009cAsym_res-3mm_label-GM_desc-thr02_probseg.nii.gz')
mask_audio_3mm = os.path.join(
    ROOT_FOLDER, 'masks', 'audio_mask_resampled_3mm.nii.gz')
language_mask_3mm = os.path.join(
    ROOT_FOLDER, 'masks', 'left_language_mask_3mm.nii.gz')


WHOLEBRAIN_DATASETS = [{"decoding_task": "cneuromod",
                        "alignment_data_label": None,
                        "roi_code": "fullbrain", "mask": mask_gm}]

ROI_DATASETS = [{"decoding_task": "ibc_rsvp", "alignment_data_label": "53_tasks",
                 "roi_code": "language_3mm", "mask": language_mask_3mm},
                {"decoding_task": "ibc_tonotopy_cond", "alignment_data_label": "53_tasks",
                 "roi_code": "audio_3mm", "mask": mask_audio_3mm}]


def _check_srm_params(srm_components, srm_atlas, trains_align, trains_decode):
    """
    * Limit number of components depending on data size
    * Reindex srm_atlas from 1 when masked atlas is not fullsize and some
        labels are not present.
    """
    if srm_atlas is not None:
        n_atlas = len(np.unique(srm_atlas))
        if not n_atlas - 1 == max(srm_atlas):
            i = 1
            for lab in np.unique(srm_atlas):
                srm_atlas[srm_atlas == lab] = i
                i += 1
    else:
        n_atlas = srm_components

    srm_components_ = np.min([srm_components, load_img(
        trains_align[0][0]).shape[-1], load_img(trains_decode[0][0]).shape[-1], n_atlas - 1])

    return srm_components_, srm_atlas


def fetch_resample_basc(mask, scale="444"):
    from nilearn.datasets import fetch_atlas_basc_multiscale_2015
    from nilearn.image import resample_to_img
    basc = fetch_atlas_basc_multiscale_2015(
        data_dir='/home/emdupre/scratch'
    )['scale{}'.format(scale)]
    resampled_basc = resample_to_img(basc, mask, interpolation='nearest')
    return resampled_basc


def make_coordinates_grid(input_shape):
    ''' Make a grid of coordinates
    '''
    x_ = range(0, input_shape[0])
    y_ = range(0, input_shape[1])
    z_ = range(0, input_shape[2])
    coordinates_points = np.vstack(np.meshgrid(
        x_, y_, z_, indexing='ij')).reshape(3, -1).T
    coordinates_matrice = coordinates_points.reshape(
        (input_shape[0], input_shape[1], input_shape[2], 3), order='C')
    return coordinates_matrice


def make_coordinates_image(input_shape):
    coordinates_matrice = make_coordinates_grid(input_shape)
    affine = np.eye(4)
    coordinates_image = nib.nifti1.Nifti1Image(
        coordinates_matrice, mask.affine)
    plain_mask = nib.nifti1.Nifti1Image(np.ones(input_shape), mask.affine)
    return coordinates_image, plain_mask


def make_affine_coordinates(mask):
    coordinates_image, plain_mask = make_coordinates_image(mask.shape)

    coordinates_vector = masker.transform(coordinates_image)
    affine_coordinates = np.round(image.coord_transform(
        coordinates_vector[1], coordinates_vector[0], coordinates_vector[2], np.linalg.inv(mask.affine)))
    return affine_coordinates


def _load_clean_one(X, masker):
    LEN_FSAV = 163842
    X_data = nib.load(X).agg_data()
    X_ = signal.clean(np.asarray(X_data), detrend=masker.detrend, high_pass=masker.high_pass,
                      low_pass=masker.low_pass, standardize=masker.standardize, t_r=masker.t_r, ensure_finite=True)
    if X_.shape[0] == LEN_FSAV:
        return X_.T
    else:
        return X_


def load_clean(X, masker):
    if isinstance(X, (list, np.ndarray)):
        return np.vstack([_load_clean_one(xi, masker)for xi in X])
    else:
        return _load_clean_one(X, masker)


def align_one_target(sources_train, sources_test, target_train, target_test, method, masker,  pairwise_method, clustering, n_pieces,
                     n_jobs, decoding_dir=None, srm_atlas=None,
                     srm_components=40, ha_radius=5, ha_sparse_radius=3,
                     smoothing_fwhm=6, surface=False):
    overhead_time = 0
    aligned_sources_test = []

    if surface == "rh":

        clustering = clustering.replace("lh", "rh")
        # clustering = load_surf_data(
        #    "/storage/store2/tbazeill/schaeffer/FreeSurfer5.3/fsaverage/label/rh.Schaefer2018_700Parcels_17Networks_order.annot")
        sources_train = np.asarray([t.replace("lh", "rh")
                                    for t in sources_train])
        sources_test = np.asarray([t.replace("lh", "rh")
                                   for t in sources_test])
        target_train.replace("lh", "rh")
        target_test.replace("lh", "rh")
    if surface in ["rh", "lh"]:
        from nilearn.surface import load_surf_data
        clustering = load_surf_data(clustering)
    if method == "anat_inter_subject":
        fit_start = time.process_time()
        if surface in ["lh", "rh"]:
            aligned_sources_test = load_clean(sources_test, masker)
            aligned_target_test = load_clean(target_test, masker)
        else:
            aligned_sources_test = np.vstack(
                [masker.transform(s) for s in sources_test])
            aligned_target_test = masker.transform(target_test)

    elif method == "smoothing":
        fit_start = time.process_time()
        smoothing_masker = NiftiMasker(
            mask_img=masker.mask_img_, smoothing_fwhm=smoothing_fwhm).fit()
        aligned_sources_test = np.vstack(
            [smoothing_masker.transform(s) for s in sources_test])
        aligned_target_test = smoothing_masker.transform(target_test)
    elif method in ["pairwise", "intra_subject"]:
        fit_start = time.process_time()
        for source_train, source_test in zip(sources_train, sources_test):
            if method == "pairwise":
                if surface in ["lh", "rh"]:
                    source_align = SurfacePairwiseAlignment(
                        alignment_method=pairwise_method, clustering=clustering,
                        n_jobs=n_jobs)
                else:
                    source_align = PairwiseAlignment(
                        alignment_method=pairwise_method, clustering=clustering,
                        n_pieces=n_pieces, mask=masker, n_jobs=n_jobs)
                source_align.fit(source_train, target_train)
                aligned_sources_test.append(
                    source_align.transform(source_test))
            elif method == "intra_subject":
                source_align = IntraSubjectAlignment(
                    alignment_method="ridge_cv", clustering=clustering,
                    n_pieces=n_pieces, mask=masker, n_jobs=n_jobs)
                source_align.fit(source_train, source_test)
                aligned_sources_test.append(
                    source_align.transform(target_train))

        if surface in ["lh", "rh"]:
            aligned_sources_test = np.vstack(aligned_sources_test)
            aligned_target_test = load_clean(target_test, masker)
        else:
            aligned_target_test = masker.transform(target_test)
            aligned_sources_test = np.vstack(
                [masker.transform(t) for t in aligned_sources_test])
    elif method == "srm":
        common_time = time.process_time()
        fastsrm = FastSRM(atlas=srm_atlas, n_components=srm_components, n_iter=1000,
                          n_jobs=n_jobs, aggregate="mean", temp_dir=decoding_dir)

        reduced_SR = fastsrm.fit_transform(
            [masker.transform(t).T for t in sources_train])
        overhead_time = time.process_time() - common_time

        fit_start = time.process_time()
        fastsrm.aggregate = None

        fastsrm.add_subjects(
            [masker.transform(t).T for t in [target_train]], reduced_SR)
        aligned_test = fastsrm.transform(
            [masker.transform(t).T for t in np.hstack([sources_test, [target_test]])])
        aligned_sources_test = np.hstack(aligned_test[:-1]).T
        aligned_target_test = aligned_test[-1].T
    elif method == "HA":
        overhead_time = 0
        fit_start = time.process_time()

        from mvpa2.algorithms.searchlight_hyperalignment import SearchlightHyperalignment
        from mvpa2.datasets.base import Dataset
        pymvpa_datasets = []

        flat_mask = load_img(
            masker.mask_img_).get_fdata().flatten()
        n_voxels = flat_mask.sum()
        flat_coord_grid = make_coordinates_grid(
            masker.mask_img_.shape).reshape((-1, 3))
        masked_coord_grid = flat_coord_grid[flat_mask != 0]
        for sub, sub_data in enumerate(np.hstack([[target_train], sources_train])):
            d = Dataset(masker.transform(sub_data))
            d.fa['voxel_indices'] = masked_coord_grid
            pymvpa_datasets.append(d)
        ha = SearchlightHyperalignment(
            radius=ha_radius, nproc=1, sparse_radius=ha_sparse_radius)
        ha.__call__(pymvpa_datasets)
        aligned_sources_test = []
        for j, source_test in enumerate(sources_test):
            if surface in ["lh", "rh"]:
                array_source = load_clean(source_test, masker)
            else:
                array_source = masker.transform(source_test)
            aligned_sources_test.append(
                array_source.dot(ha.projections[j + 1].proj.toarray()))
        aligned_sources_test = np.vstack(
            aligned_sources_test)
        aligned_target_test = masker.transform(
            target_test).dot(ha.projections[0].proj.toarray())

    fit_time = time.process_time() - fit_start

    return aligned_sources_test, aligned_target_test, fit_time, overhead_time


def try_methods_decoding(method, subjects, train, test, pairwise_method,
                         mask, clustering, pipeline, method_path,
                         n_pieces=None, srm_components=50, ha_radius=5,
                         ha_sparse_radius=3, srm_atlas=None, atlas_name="",
                         smoothing_fwhm=6, n_jobs=1, surface=False):
    """
    Parameters
    ----------
    method: str
        Desired alignment method. Must be a string in ["anat_inter_subject",
        "smoothing", "pairwise", "template", "intra_subject", "srm", "HA"]
    local_align_method: str
        An alignment method recognized by fmralign.
    srm_components: int
        The requested number of components for the Shared Response Model (SRM).
        This corresponds to the hyperparameter _k_ in the original formulation.
    ha_radius: int
        Radius of a searchlight sphere in number of voxels to be used.
        Default of 5 voxels.
    ha_sparse_radius: int
        Radius supplied to scatter_neighborhoods in units of voxels.
        This is effectively the distance between the centers of mass where
        hyperalignment is performed in searchlights.
        Default of 3 voxels.
    smoothing_fwhm: int
        Smoothing kernel to apply in the case of the 'smoothing'
        alignment baseline.
    srm_atlas: ndarray, optional
        Probabilistic or deterministic atlas on which to project the data
        Deterministic atlas is an ndarray of shape [n_voxels,] where values
        range from 1 to n_supervoxels. Voxels labelled 0 will be ignored.
    atlas_name: str, optional
        An optional name for the requested SRM atlas.
    """
    if not surface:
        if n_pieces is None:
            n_pieces = int(np.round(load_img(mask).get_fdata().sum() / 200))
        masker = NiftiMasker(mask_img=mask).fit()
    else:
        masker = NiftiMasker()

    path_to_score = method_path + ".csv"
    path_to_fit_timings = method_path + "_fit_timings.csv"
    path_to_overhead_timings = method_path + "_overhead_timings.csv"

    if not os.path.exists(path_to_score):
        scores = []
        fit_timings = []
        overhead_timings = []
        i = 0
        for (train_align, train_decode, train_decode_labels,
             LO_align, LO_decode, LO_decode_labels) in zip(
                 train.alignment, train.x, train.y,
                 test.alignment, test.x, test.y):

            print(" Fold {} : method {} : ".format(i, method_path))

            fold_scores, fold_fit_timings, fold_overhead_timings = [], 0, 0
            for (single_target, single_decode, single_labels) in zip(LO_align, LO_decode, LO_decode_labels):
                if surface:
                    (align_decoding_train_lh, aligned_target_lh,
                     fit_time_lh, overhead_time_lh) = align_one_target(
                        train_align, list(train_decode), single_target,
                        single_decode, method, masker, pairwise_method,
                        clustering, n_pieces, n_jobs, decoding_dir=None,
                        srm_atlas=srm_atlas, srm_components=srm_components,
                        ha_radius=ha_radius, ha_sparse_radius=ha_sparse_radius,
                        smoothing_fwhm=smoothing_fwhm, surface="lh")

                    (align_decoding_train_rh, aligned_target_rh,
                     fit_time_rh, overhead_time_rh) = align_one_target(
                        train_align, list(train_decode), single_target,
                        single_decode, method, masker, pairwise_method,
                        clustering, n_pieces, n_jobs, decoding_dir=None,
                        srm_atlas=srm_atlas, srm_components=srm_components,
                        ha_radius=ha_radius, ha_sparse_radius=ha_sparse_radius,
                        smoothing_fwhm=smoothing_fwhm, surface="rh")

                    align_decoding_train = [np.hstack([s, t]) for s, t in zip(
                        align_decoding_train_lh, align_decoding_train_rh)]
                    aligned_target = np.hstack(
                        [aligned_target_lh, aligned_target_rh])
                    fit_time = fit_time_lh + fit_time_rh
                    overhead_time = overhead_time_lh + overhead_time_rh
                else:
                    (align_decoding_train, aligned_target,
                     fit_time, overhead_time) = align_one_target(
                        train_align, list(train_decode), single_target,
                        single_decode, method, masker, pairwise_method,
                        clustering, n_pieces, n_jobs, decoding_dir=None,
                        srm_atlas=srm_atlas, srm_components=srm_components,
                        ha_radius=ha_radius, ha_sparse_radius=ha_sparse_radius,
                        smoothing_fwhm=smoothing_fwhm)

                fold_scores.append(decode(
                    pipeline, align_decoding_train,
                    np.hstack(train_decode_labels),
                    aligned_target, single_labels))
                fold_fit_timings += fit_time
                fold_overhead_timings += overhead_time

            scores.append(fold_scores)
            fit_timings.append(fold_fit_timings)
            overhead_timings.append(fold_overhead_timings)
            i += 1
            print("Fold fit timing : {} s".format(fold_fit_timings))

        with open(path_to_score, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows([scores])
        with open(path_to_fit_timings, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows([fit_timings])
        with open(path_to_overhead_timings, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows([overhead_timings])
    pass


def find_method_label(method, local_align_method=None, srm_components=0,
                      srm_atlas=None, atlas_name="", ha_radius=5,
                      ha_sparse_radius=3, smoothing_fwhm=6):
    """
    Creates a 'method_label' string, used in naming output files with
    derived results. Ensures that all possible alignments have their own unique
    file path to avoid naming clashes.

    Parameters
    ----------
    method: str
        Desired alignment method. Must be a string in ["anat_inter_subject",
        "smoothing", "pairwise", "template", "intra_subject", "srm", "HA"]
    local_align_method: str
        An alignment method recognized by fmralign.
    srm_components: int
        The requested number of components for the Shared Response Model (SRM).
        This corresponds to the hyperparameter _k_ in the original formulation.
    ha_radius: int
        Radius of a searchlight sphere in number of voxels to be used.
        Default of 5 voxels.
    ha_sparse_radius: int
        Radius supplied to scatter_neighborhoods in units of voxels.
        This is effectively the distance between the centers of mass where
        hyperalignment is performed in searchlights.
        Default of 3 voxels.
    smoothing_fwhm: int
        Smoothing kernel to apply in the case of the 'smoothing'
        alignment baseline.
    srm_atlas: ndarray, optional
        Probabilistic or deterministic atlas on which to project the data
        Deterministic atlas is an ndarray of shape [n_voxels,] where values
        range from 1 to n_supervoxels. Voxels labelled 0 will be ignored.
    atlas_name: str, optional
        An optional name for the requested SRM atlas.

    Returns
    -------
    method_label: str
    """
    method_label = method
    if method in ["pairwise", "template"]:
        if local_align_method is not None:
            method_label += "_{}".format(local_align_method)
        else:
            err_msg = ("Requested {} method ".format(method) +
                       "but local_align_method is undefined")
            raise ValueError(err_msg)
    if method == "intra_subject":
        method_label += "_ridge_cv"
    if method == "smoothing":
        method_label += "_{:0>2d}".format(smoothing_fwhm)
    if method in ["srm", "piecewise_srm", "mvica"]:
        if srm_components:
            method_label += "_{}".format(srm_components)
            if srm_atlas is not None:
                method_label += "_{}".format(atlas_name)
        else:
            err_msg = ("Requested SRM but srm_components is zero. Please " +
                       "request a non-zero number of components.")
            raise ValueError(err_msg)
    if method == "HA":
        method_label += "rad_{}_sparse_{}".format(ha_radius, ha_sparse_radius)
    return method_label


def experiments_variables(task, surface="", root_dir='/'):
    """
    Defines a Bunch object with experimentally relevant variables.

    Paramters
    ---------
    task : str
        The decoding task of interest
    root_dir : str
        The filepath to pre-downloaded, preprocessed IBC data

    Returns
    -------
    data : Bunch
    """
    if task in ['ibc_rsvp', 'ibc_tonotopy_cond', ]:
        mask = opj(root_dir, 'ibc', 'gm_mask_3mm.nii.gz')

    if task == 'ibc_rsvp':
        subjects = ['sub-01', 'sub-04', 'sub-05', 'sub-06', 'sub-07',
                    'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14']
        task_dir = opj(root_dir, 'rsvp_trial', '3mm')
        out_dir = opj(root_dir, 'rsvp_trial', 'decoding')
        mask_cache = opj(root_dir, 'rsvp_trial', 'mask_cache')
        # language_mask_3mm = opj(root_dir,"ibc/left_language_mask_3mm.nii.gz")

    elif task == 'ibc_tonotopy_cond':
        subjects = ['sub-04', 'sub-05', 'sub-06', 'sub-07',
                    'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14']
        task_dir = opj(root_dir, 'ibc_tonotopy_cond', '3mm')
        out_dir = opj(root_dir, 'ibc_tonotopy_cond', 'decoding')
        mask_cache = opj(root_dir, 'ibc_tonotopy_cond', 'mask_cache')
        # audio_mask_3mm = opj(root_dir,"ibc/audio_mask_resampled_3mm.nii.gz")

    elif task == "cneuromod":
        subjects = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06"]
        mask = opj(
            root_dir,
            "tpl-MNI152NLin2009cAsym_res-3mm_label-GM_desc-thr02_probseg.nii.gz")
        task_dir = opj(root_dir, "cneuro_wm", "3mm")
        out_dir = opj(root_dir, "cneuro_wm", "decoding_fmralignbench")
        mask_cache = opj(root_dir, "cneuro_wm", "mask_cache")

    else:
        err_msg = ("Unrecognized decoding task. Please provide a " +
                   "recognized decoding task and try again.")
        raise ValueError(err_msg)

    if surface in ["lh", "rh"]:
        task_dir = task_dir.replace("3mm", "surf_derivatives")

    return Bunch(subjects=subjects, mask=mask, task_dir=task_dir,
                 out_dir=out_dir, mask_cache=mask_cache)


def fetch_decoding_data(subjects, task, data_dir, surface=""):
    if surface in ["lh", "rh", "lh_fullres", "rh_fullres"]:
        decoding_subjects = [opj(data_dir, "{}_{}.gii".format(subject, surface))
                             for subject in subjects]
        decoding_conditions = [np.hstack(pd.read_csv(
            opj(data_dir, "{}_{}_labels.csv".format(subject, surface[:2])), header=None).to_numpy()) for subject in subjects]
        decoding_runs = [np.hstack(pd.read_csv(
            opj(data_dir, "{}_{}_runs.csv".format(subject, surface[:2])), header=None).to_numpy()) for subject in subjects]
    else:
        decoding_subjects = [opj(
            data_dir, "{}.nii.gz".format(subject)) for subject in subjects]
        decoding_conditions = [np.hstack(pd.read_csv(
            opj(data_dir, "{}_labels.csv".format(subject)), header=None).to_numpy()) for subject in subjects]
        decoding_runs = [np.hstack(pd.read_csv(
            opj(data_dir, "{}_runs.csv".format(subject)), header=None).to_numpy()) for subject in subjects]
    return np.asarray(decoding_conditions), np.asarray(decoding_subjects), np.asarray(decoding_runs)


def decode(pipeline, decoding_train, train_labels, decoding_test, test_labels):
    """
    Decode data with a pipeline, decoding train and train_labels
    should be arrays of same len
    Decoding_test and test_labels can be arrays or list of arrays.
    If list of arrays, will return list of scores.
    Return scores (int or list depending on decoding test type)
    """
    clf = clone(pipeline)
    clf.fit(decoding_train, train_labels)

    # if decoding test is 3 dimensions iterable and test_labels is 2
    # dimensions iterable, then iterate on both and return a list of scores.
    if (isinstance(decoding_test, (list, np.ndarray)) and len(np.shape(decoding_test)) == 3) and (isinstance(test_labels, (list, np.ndarray)) and len(np.shape(test_labels)) == 2):
        return [clf.score(single_test, single_label) for single_test, single_label in zip(decoding_test, test_labels)]
    # else just return one score
    return clf.score(decoding_test, test_labels)


def fetch_align_decode_data(task, subjects, data_dir,
                            ibc_dataset_label="53_tasks", random_state=0,
                            permutated=False, surface="", root_dir='/'):
    '''
    Parameters
    ----------
    task: str
        The decoding task over which to calculate alignment accuracy.
    subjects: list
        A list of included subjects. Must match those used for the given task.
        See `experiment_variables` for the appropriate subject-subset for
        each task.
    data_dir: str
        Filepath to the local directory where a given task's preprocessed
        data is stored. See `experiment_variables` to define this filepath.
    ibc_dataset_label: str, Optional
        If `task` is from the IBC dataset, alignments may be learnt on
        either the movie-watching dataset or over 53 task-contrasts.
        This argument specifies which to learn over, and therefore must be
        a string in ['53_tasks', 'MV']
    random_state: int, Optional
        The random state passed to np.random. Defaults to 0.
    permutated: bool, Optional
        Whether to permute the indices of images used in alignment,
        thereby destroying their spatial structure. Used as a baseline.
    root_dir: str
        The filepath to a pre-downloaded, pre-processed subset of the
        IBC dataset.

    Returns
    -------
    train: namedtuple
        Data split used for training. Contains the following fields:
        'x': decoding data
        'y': decoding labels
        'alignment': data used to derive functional alignment
    test: namedtuple
        Left-out (i.e., testing) data split. Contains the following fields:
        'x': decoding data
        'y': decoding labels
        'alignment': data used to derive functional alignment
    '''

    DataSplit = namedtuple('DataSplit', ['x', 'y', 'alignment'])
    ibc_53_tasks = opj(root_dir, 'alignment',
                       '{}_53_contrasts.nii.gz')

    decoding_conditions, decoding_subjects, _ = fetch_decoding_data(
        subjects, task, data_dir, surface=surface)
    train_align, train_decode, train_decode_labels = [], [], []
    LOs_align, LOs_decode, LOs_decode_labels = [], [], []

    if task.startswith('ibc_'):
        if ibc_dataset_label == '53_tasks':
            if surface in ["lh", "rh", "lh_fullres", "rh_fullres"]:
                paths_align = np.asarray([os.path.join(root_dir, "surf_ibc",
                                                       "{}_53_contrasts_{}.gii".format(sub, surface))for sub in subjects])
            else:
                paths_align = np.asarray([ibc_53_tasks.format(sub)
                                          for sub in subjects])
        n_subj = np.arange(len(subjects))
        leave_outs = n_subj

        for lo in leave_outs:
            # training alignment data
            train_align.append(
                paths_align[np.isin(n_subj, (lo), invert=True)])
            # training decoding data splits, labels
            train_decode.append(
                decoding_subjects[np.isin(n_subj, (lo), invert=True)])
            train_decode_labels.append(
                decoding_conditions[np.isin(n_subj, (lo), invert=True)])
            # testing decoding data splits, labels
            LOs_decode.append([decoding_subjects[lo]])
            LOs_decode_labels.append([decoding_conditions[lo]])
            LOs_align.append([paths_align[lo]])

    elif task == "cneuromod":
        life = opj(
            root_dir, 'alignment_data',
            '{}_task-life_run-1_space-MNI152NLin2009cAsym_desc-postproc_bold.nii.gz')
        paths_align = np.asarray([life.format(sub)
                                  for sub in subjects])
        n_subj = np.arange(len(subjects))
        leave_outs = n_subj  # use LOO

        for lo in leave_outs:
            # training alignment data
            train_align.append(
                paths_align[np.isin(n_subj, (lo), invert=True)])
            # training decoding data splits, labels
            train_decode.append(
                decoding_subjects[np.isin(n_subj, (lo), invert=True)])
            train_decode_labels.append(
                decoding_conditions[np.isin(n_subj, (lo), invert=True)])
            # testing decoding data splits, labels
            LOs_align.append([paths_align[lo]])
            LOs_decode.append([decoding_subjects[lo]])
            LOs_decode_labels.append([decoding_conditions[lo]])
            LOs_align.append([paths_align[lo]])

    train = DataSplit(
        x=train_decode, y=train_decode_labels, alignment=train_align)
    test = DataSplit(x=LOs_decode, y=LOs_decode_labels, alignment=LOs_align)

    return train, test


# USABLE for exp1, exp2, s3

def fetch_resample_schaeffer(mask, scale="444"):
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn.image import resample_to_img
    atlas = fetch_atlas_schaefer_2018(
        data_dir='/home/emdupre/scratch',
        n_rois=scale, resolution_mm=2)["maps"]
    resampled_atlas = resample_to_img(
        atlas, mask, interpolation='nearest')
    return resampled_atlas


def check_input_method(input_method):
    if input_method == "pairwise_scaled_orthogonal":
        method = "pairwise"
        pairwise_method = "scaled_orthogonal"
        local_align_method = "scaled_orthogonal"
    elif input_method == "pairwise_ot_e-1":
        method = "pairwise"
        pairwise_method = OptimalTransportAlignment(reg=.1)
        local_align_method = "ot_e-1"
    else:
        method = input_method
        pairwise_method = ""
        local_align_method = ""
    return method, pairwise_method, local_align_method


def inter_subject_align_decode(input_method, dataset_params, clustering, root_folder, smoothing_fwhm=0, n_pieces=None, n_jobs=15):
    """
    all methods
    """
    # LOAD DATASET AND ENVIRONMENT INFORMATIONS
    data = experiments_variables(
        dataset_params["decoding_task"], root_dir=ROOT_FOLDER)
    # check paths and create folders if needed
    if not os.path.isdir(data.out_dir):
        os.mkdir(data.out_dir)
    if not os.path.isdir(data.mask_cache):
        os.mkdir(data.mask_cache)

    # LOAD DATA
    train, test = fetch_align_decode_data(
        dataset_params["decoding_task"], data.subjects, data.task_dir, ibc_dataset_label=dataset_params["alignment_data_label"], root_dir=ROOT_FOLDER)

    # PARSE, DEFINE AND CHECK PIPELINE PARAMETERS
    method, pairwise_method, local_align_method = check_input_method(
        input_method)
    # define masker
    mask = dataset_params["mask"]
    masker = NiftiMasker(mask_img=mask, memory=data.mask_cache).fit()
    # default parameters
    ha_radius, ha_sparse_radius = 5, 3
    atlas_name, srm_components = "basc_444", 50
    basc_444 = fetch_resample_basc(mask_gm, scale='444')
    srm_atlas = masker.transform(basc_444)[0]
    pipeline = make_pipeline(LinearSVC(max_iter=10000))

    srm_components, srm_atlas = _check_srm_params(srm_components, srm_atlas,
                                                  train.alignment, train.x)

    # MAKE RESULTS NAMING (including clustering if relevant)
    method_label = find_method_label(
        method, local_align_method, srm_components=srm_components,
        srm_atlas=srm_atlas, atlas_name=atlas_name, ha_radius=ha_radius,
        ha_sparse_radius=ha_sparse_radius, smoothing_fwhm=smoothing_fwhm)
    if clustering == "schaefer" and method == "intra_subject":
        n_pieces = 1000
    if method in ["pairwise", "intra_subject"]:
        clustering_name = '{}_{}'.format(clustering, n_pieces)
        method_path = os.path.join(data.out_dir, "{}_{}_{}_{}_on_{}".format(
            dataset_params["decoding_task"], dataset_params["roi_code"], method_label, clustering_name, dataset_params["alignment_data_label"]))
    else:
        method_path = os.path.join(data.out_dir, "{}_{}_{}_on_{}".format(
            dataset_params["decoding_task"], dataset_params["roi_code"], method_label, dataset_params["alignment_data_label"]))
    if clustering == "schaefer":
        clustering = fetch_resample_schaeffer(mask, scale=n_pieces)

    # RUN THE EXPERIMENT FOR ONE SET OF PARAMETERS (if not already cached)
    try_methods_decoding(method=method, subjects=data.subjects,
                         train=train, test=test,
                         pairwise_method=pairwise_method, mask=mask,
                         clustering=clustering, pipeline=pipeline,
                         method_path=method_path,
                         srm_components=srm_components,
                         srm_atlas=srm_atlas,
                         ha_radius=ha_radius,
                         ha_sparse_radius=ha_sparse_radius,
                         atlas_name=atlas_name,
                         smoothing_fwhm=smoothing_fwhm, n_pieces=n_pieces,
                         n_jobs=N_JOBS)

    pass


def within_subject_decoding(dataset_params, root_folder, n_jobs=1):
    data = experiments_variables(
        dataset_params["decoding_task"], root_dir=root_folder)

    path_to_score = os.path.join(data.out_dir, "{}_{}_within_subject_decoding_on_{}.csv".format(
        dataset_params["decoding_task"], dataset_params["roi_code"], dataset_params["alignment_data_label"]))
    if not os.path.exists(path_to_score):
        from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
        mask = dataset_params["mask"]
        masker = NiftiMasker(mask_img=mask).fit()
        decoding_conditions, decoding_subjects, decoding_sessions = fetch_decoding_data(
            data.subjects, dataset_params["decoding_task"], data.task_dir)

        scores = []
        for ims, labs, runs in zip(decoding_subjects, decoding_conditions, decoding_sessions):
            images = np.array(masker.transform(ims))
            decoder = LinearSVC(max_iter=10000)
            cv = LeaveOneGroupOut()
            score = cross_val_score(
                decoder, images, np.array(labs), groups=np.array(runs), cv=cv, n_jobs=n_jobs)
            scores.append([np.mean(score)])
        with open(path_to_score, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows([scores])
