import itertools
import warnings
from joblib import Parallel, delayed
from fmralignbench.utils import (WHOLEBRAIN_DATASETS, ROI_DATASETS, inter_subject_align_decode,within_subject_decoding)
from fmralignbench.conf import ROOT_FOLDER, N_JOBS
warnings.filterwarnings(action='once')

input_methods = ["anat_inter_subject", "pairwise_scaled_orthogonal",
                 "pairwise_ot_e-1",  "srm", "intra_subject", "HA"]

###### EXPERIMENT 1 #######

# Inter-subject results
experiment_parameters = list(itertools.product(
    WHOLEBRAIN_DATASETS, input_methods))

Parallel(n_jobs=1)(delayed(inter_subject_align_decode)(input_method, dataset_params, "schaefer",
                                                       ROOT_FOLDER, n_pieces=300, n_jobs=N_JOBS) for dataset_params, input_method in experiment_parameters)
# Within-subject results
Parallel(n_jobs=1)(delayed(within_subject_decoding)(dataset_params, ROOT_FOLDER)
                   for dataset_params in WHOLEBRAIN_DATASETS)

###### EXPERIMENT 2 #######

experiment_parameters = list(itertools.product(ROI_DATASETS, input_methods))
Parallel(n_jobs=1)(delayed(inter_subject_align_decode)(input_method, dataset_params, "schaefer",
                                                       ROOT_FOLDER, n_pieces=300, n_jobs=N_JOBS) for dataset_params, input_method in experiment_parameters)
