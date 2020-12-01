
import itertools
from joblib import Parallel, delayed
from fmralign-benchmark-mockup.utils import WHOLEBRAIN_DATASETS, inter_subject_align_decode
from fmralign-benchmark-mockup.conf import ROOT_FOLDER, N_JOBS
import warnings
warnings.filterwarnings(action='once')


# EXPERIMENT S2
clusterings = ["schaefer", "hierarchical_kmeans", "kmeans"]
scales = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
shuffled_parameters = list(itertools.product(
    WHOLEBRAIN_DATASETS, clusterings, scales))

Parallel(n_jobs=1)(delayed(inter_subject_align_decode)("pairwise_scaled_orthogonal", dataset_params, clustering,
                                                       ROOT_FOLDER, n_pieces=n_pieces, n_jobs=N_JOBS) for dataset_params, clustering, scale in shuffled_parameters)


# EXPERIMENT S3
smoothing_fwhms = [5, 10, 15, 20, 25, 30]
shuffled_parameters = list(itertools.product(
    WHOLEBRAIN_DATASETS, smoothing_fwhms))

Parallel(n_jobs=1)(delayed(inter_subject_align_decode)("smoothing", dataset_params, None,
                                                       ROOT_FOLDER, n_pieces=None, smoothing_fwhm=smoothing_fwhm, n_jobs=N_JOBS) for dataset_params, smoothing_fwhm in shuffled_parameters)
