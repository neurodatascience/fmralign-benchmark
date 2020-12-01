import itertools
import warnings
import os
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
from nilearn.input_data import NiftiMasker
from fmralign.alignment_methods import OptimalTransportAlignment
from fmralign-benchmark-mockup.utils import try_methods_decoding, find_method_label, experiments_variables, fetch_align_decode_data, _check_srm_params, fetch_resample_basc, WHOLEBRAIN_DATASETS, ROI_DATASETS, inter_subject_align_decode
from fmralign-benchmark-mockup.conf import ROOT_FOLDER, N_JOBS
warnings.filterwarnings(action='once')

input_methods = ["anat_inter_subject", "pairwise_scaled_orthogonal",
                 "pairwise_ot_e-1",  "srm", "intra_subject", "HA"]

###### EXPERIMENT 1 #######
# Inter-subject results
experiment_parameters = list(itertools.product(
    WHOLEBRAIN_DATASETS, input_methods))

Parallel(n_jobs=1)(delayed(inter_subject_align_decode)(input_method, dataset_params, clustering,
                                                       ROOT_FOLDER, n_pieces=n_pieces, n_jobs=N_JOBS) for dataset_params, input_method in experiment_parameters)
# Within-subject results
Parallel(n_jobs=1)(delayed(within_subject_decoding)(dataset_params, root_folder)
                   for dataset_params in WHOLEBRAIN_DATASETS)

###### EXPERIMENT 2 #######

experiment_parameters = list(itertools.product(ROI_DATASETS, input_methods))
Parallel(n_jobs=1)(delayed(inter_subject_align_decode)(input_method, dataset_params, clustering,
                                                       ROOT_FOLDER, n_pieces=n_pieces, n_jobs=N_JOBS) for dataset_params, input_method in experiment_parameters)
