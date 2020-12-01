""" File to replicate 3mm volumic results with fsaverage7 surfacic representation
of the data. Only tested (and tractable) for "pairwise_scaled_orthogonal" method and on IBC RSVP decoding task
"""
import os
import warnings
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from nilearn.surface import load_surf_data
from fmralignbench.utils import (WHOLEBRAIN_DATASETS, inter_subject_align_decode, check_input_method, try_methods_decoding, find_method_label, experiments_variables,
                                 fetch_align_decode_data)
from fmralignbench.conf import ROOT_FOLDER, N_JOBS
# !!!! Missing data downloader (should include alignment and decoding derivatives + clustering in suitable positions)
warnings.filterwarnings(action='once')
input_method = "pairwise_scaled_orthogonal"
param_set = WHOLEBRAIN_DATASETS[0]
clustering = load_surf_data(os.path.join(
    ROOT_FOLDER, "masks", "lh.Schaefer2018_700Parcels_17Networks_order.annot"))

data = experiments_variables(
    param_set["decoding_task"], root_dir=ROOT_FOLDER, surface='lh')


method, pairwise_method, local_align_method = check_input_method(input_method)
if not os.path.isdir(data.out_dir):
    os.mkdir(data.out_dir)

train, test = fetch_align_decode_data(
    param_set["decoding_task"], data.subjects, data.task_dir, ibc_dataset_label=param_set["alignment_data_label"], root_dir=ROOT_FOLDER, surface='lh_fullres')
pipeline = make_pipeline(LinearSVC(max_iter=10000))

method_label = find_method_label(
    method, local_align_method)
method_path = os.path.join(data.out_dir, "surf_{}_fullres_fullbrain_{}_on_{}".format(
    param_set["decoding_task"], method_label, param_set["alignment_data_label"]))

try_methods_decoding(method=method, subjects=data.subjects,
                     train=train, test=test,
                     pairwise_method=pairwise_method, mask=None,
                     clustering=clustering, pipeline=pipeline,
                     method_path=method_path,
                     n_jobs=15, surface=True)
