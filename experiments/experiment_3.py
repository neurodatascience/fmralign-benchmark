from os.path import join as opj
import warnings
from nilearn.input_data import NiftiMasker
from nilearn.image import load_img
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
import csv
import pandas as pd
import numpy as np
import glob
import os
import time
from experiments.fastsrm import FastSRM
from experiments.intra_subject_alignment import IntraSubjectAlignment
from fmralign.pairwise_alignment import PairwiseAlignment
from fmralign.alignment_methods import OptimalTransportAlignment
import pickle
import joblib
import shutil
from nilearn.image import mean_img
from nilearn import plotting, surface
from nilearn.datasets import fetch_surf_fsaverage, load_mni152_template
import ibc_data
from nilearn.plotting import plot_anat, plot_stat_map
from ibc_public.utils_data import get_subject_session
from matplotlib import pyplot as plt
from mvpa2.datasets.base import Dataset
from experiments.utils import experiments_variables, fetch_align_decode_data, fetch_resample_basc, _check_srm_params, find_method_label, fetch_decoding_data, make_coordinates_grid, check_exist, parse_score, fetch_scores, try_methods_decoding, decode,
conjunction_inference_from_z_images
warnings.filterwarnings(action='once')


def save_contrast(align, source_test):
    dir_ = os.path.dirname(align)
    model = align.split("/")[-1].split(".")[0]
    sub_test = source_test.split("/")[6]
    contrast = source_test.split("/")[-1].split("_")[0]
    return os.path.join(dir_, "{}_{}_with_{}.nii.gz".format(
        sub_test, contrast, model))


def make_path(source_train, target_train, method):

    if not os.path.isfile(source_train):
        source = source_train
    else:
        dir_ = os.path.dirname(source_train)
        source = source_train.split('/')[-1].split('_')[0]
        alignment_dat = source_train.split('/')[-1].split('_')[1]
    if not os.path.isfile(target_train):
        target = target_train
    else:
        dir_ = os.path.dirname(target_train)
        target = target_train.split('/')[-1].split('_')[0]
        alignment_dat = target_train.split('/')[-1].split('_')[1]
    path = os.path.join(dir_, "{}_{}_to_{}_on_{}.pkl".format(
        method, source, target, alignment_dat))

    return path


def _save_align(inst, path):
    if isinstance(inst, OptimalTransportAlignment):
        inst.ot = None

    if isinstance(inst, PairwiseAlignment) and hasattr(inst, "fit_"):
        if isinstance(inst.alignment_method, OptimalTransportAlignment):
            inst.alignment_method.ot = None
        for t in inst.fit_[0]:
            if isinstance(t, OptimalTransportAlignment):
                t.ot = None
    with open(path, "wb") as f:
        pickle.dump(inst, f)


def alignment_save(method, pairwise_method, local_align_method,
                   sources_train, sources_test, target_train, mask):
    masker = NiftiMasker(mask_img=mask).fit()
    n_pieces = None
    roi_code, clustering = "fullbrain", fetch_resample_schaeffer(
        mask, scale=300)
    n_jobs = 10
    smoothing_fwhm = 5
    atlas_name, srm_components = "basc_444", 50
    ha_radius, ha_sparse_radius = 5, 3
    # fetch right mask with roi_code
    srm_atlas = masker.transform(fetch_resample_basc(mask, scale='444'))[0]
    srm_components, srm_atlas = _check_srm_params(srm_components, srm_atlas, [sources_train],
                                                  [sources_train])
    method_label = find_method_label(
        method, local_align_method, srm_components=srm_components, srm_atlas=srm_atlas,
        atlas_name=atlas_name, ha_radius=ha_radius, ha_sparse_radius=ha_sparse_radius, smoothing_fwhm=smoothing_fwhm)

    if n_pieces is None:
        n_pieces = int(np.round(load_img(mask).get_data().sum() / 200))
    if method == "pairwise":
        for source_train, source_test in zip(sources_train, sources_test):
            path = make_path(
                source_train, target_train, method_label)
            if not os.path.exists(path):
                source_align = PairwiseAlignment(
                    alignment_method=pairwise_method, clustering=clustering,
                    n_pieces=n_pieces, mask=masker, n_jobs=n_jobs)
                source_align.fit(source_train, target_train)

                _save_align(source_align, path)
            with open(path, "rb") as f:
                source_align = pickle.load(f)
            aligned_test = source_align.transform(source_test)
            obj_path = save_contrast(path, source_test)
            aligned_test.to_filename(obj_path)

    elif method == "srm":

        path = make_path("all", target_train, method_label)
        srm = FastSRM(atlas=srm_atlas, n_components=srm_components, n_iter=1000,
                      n_jobs=n_jobs, aggregate="mean")
        reduced_SR = srm.fit_transform(
            [masker.transform(t).T for t in sources_train])
        srm.aggregate = None
        srm.add_subjects(
            [masker.transform(t).T for t in [target_train]], reduced_SR)
        _save_align(srm, path)
        with open(path, "rb") as f:
            srm = pickle.load(f)
        aligned_test = srm.transform(
            [masker.transform(t).T for t in sources_test])

        for aligned_z, sub_basis, source_test in zip(aligned_test, srm.basis_list[:-1], sources_test):
            z_in_target_space = sub_basis.T.dot(aligned_z).T
            z_img = masker.inverse_transform(z_in_target_space)
            obj_path = save_contrast(path, source_test)
            z_img.to_filename(obj_path)
    elif method == "HA":
        from mvpa2.algorithms.searchlight_hyperalignment import SearchlightHyperalignment
        from mvpa2.datasets.base import Dataset
        path = make_path("all", target_train, method_label)
        if not os.path.exists(path):
            flat_mask = load_img(
                masker.mask_img_).get_data().flatten()
            n_voxels = flat_mask.sum()
            flat_coord_grid = make_coordinates_grid(
                masker.mask_img_.shape).reshape((-1, 3))
            masked_coord_grid = flat_coord_grid[flat_mask != 0]
            pymvpa_datasets = []
            for sub, sub_data in enumerate(np.hstack([[target_train], sources_train])):
                d = Dataset(masker.transform(sub_data))
                d.fa['voxel_indices'] = masked_coord_grid
                pymvpa_datasets.append(d)
            ha = SearchlightHyperalignment(
                radius=ha_radius, nproc=1, sparse_radius=ha_sparse_radius)
            ha.__call__(pymvpa_datasets)
            _save_align(ha, path)
        with open(path, "rb") as f:
            ha = pickle.load(f)
        j = 1
        for source_test in sources_test:
            obj_path = save_contrast(path, source_test)
            align_test = masker.inverse_transform(masker.transform(
                source_test).dot(ha.projections[j].proj.toarray()))
            align_test.to_filename(obj_path)
            j += 1
    pass


def run_save_align_for_tasks_and_contrasts(subjects, method, pairwise_method, local_align_method, root_folder):

    paths_align = np.asarray([os.path.join(
        root_folder, "ibc/cache_rest_movie_task/{}_53_contrasts.nii.gz".format(sub)) for sub in subjects])
    paths_contrasts = np.asarray([np.asarray([os.path.join(root_folder, "neuroimage", "contrasts",
                                                           "{}-{}.nii.gz".format(sub, contrast)) for contrast in contrasts]) for sub in subjects]).T
    mask = opj(root_folder, 'ibc/gm_mask_3mm.nii.gz')
    subject_LO = 0
    sources_train = paths_align[np.arange(len(subjects)) != subject_LO]

    sources_test = paths_contrast[np.arange(len(subjects)) != subject_LO]
    target_train = paths_align[subject_LO]
    target_contrast = paths_contrast[subject_LO]

    alignment_save(method, task, pairwise_method, local_align_method,
                   sources_train, sources_test, target_train, mask)
    pass


# %% First part of the pipeline : Create and save align estimators and aligned contrasts


# CHANGE clustering / CHANGE imports

n_cores = 20
n_jobs = 2
os.environ['OMP_NUM_THREADS'] = '{}'.format(n_cores)
os.environ['NUMEXPR_NUM_THREADS'] = '{}'.format(n_cores)
os.environ['MKL_NUM_THREADS'] = '{}'.format(n_cores)


root_folder = '/storage/store2/work/tbazeill'
mask_gm = opj(root_folder, 'ibc/gm_mask_3mm.nii.gz')
contrasts = ["sentence-word", "word-consonant_string",
             "speech-silence", "voice-silence"]
subjects = ['sub-04', 'sub-05', 'sub-06', 'sub-07',
            'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14']
alignment_data = "53"
target_ind = 0
target = subjects[target_ind]

methods = ["pairwise_scaled_orthogonal", "pairwise_ot_e-1", "srm", "HA"]

for input_method in methods:
    method, pairwise_method, local_align_method = check_input_method(
        input_method)
    run_save_align_for_tasks_and_contrasts(
        subjects, method, pairwise_method, local_align_method)


masker = NiftiMasker(mask_img=mask_gm).fit()
task = "ibc_rsvp"
cached_methods = ["anat", "pairwise_ot_e-1",
                  "pairwise_scaled_orthogonal", "HArad_5_sparse_3", "srm_50_basc_444"]

paths_contrasts = np.asarray([np.asarray([os.path.join(root_folder, "neuroimage", "contrasts",
                                                       "{}-{}.nii.gz".format(sub, contrast)) for contrast in contrasts]) for sub in subjects]).T
paths_contrasts

contrast_dir = opj(root_folder, "ibc/cache_rest_movie_task/")
alignment_data = "53"

u = 0.25

contrasts_original = np.asarray([np.asarray([os.path.join(root_folder, "neuroimage", "contrasts",
                                                          "{}-{}.nii.gz".format(sub, contrast)) for contrast in contrasts]) for sub in subjects]).T

contrast = contrasts[0]
i = 0
all_aligned = contrasts_original[i][np.arange(
    len(subjects)) != target_ind]

for i, contrast in enumerate(contrasts):
    for method in cached_methods:
        all_aligned = []
        if method == "anat":
            all_aligned = contrasts_original[i][np.arange(
                len(subjects)) != target_ind]
            target_space = "MNI"
        else:
            target_space = target
            for source in subjects:
                if source != target:
                    if any(x in method for x in ["HA", "srm"]):
                        aligned_path = os.path.join(contrast_dir, '{}_{}_with_{}_all_to_{}_on_{}.nii.gz'.format(
                            source, contrast, method, target_space, alignment_data))
                    else:
                        aligned_path = os.path.join(contrast_dir, '{}_{}_with_{}_{}_to_{}_on_{}.nii.gz'.format(
                            source, contrast, method, source, target_space, alignment_data))
                    all_aligned.append(aligned_path)

        Z = masker.transform(z_images).T
        p = int((1 - u) * Z.shape[1])
        Z_ = np.sort(Z, 1)
        conj = np.sum(Z_[:, :p], 1) / np.sqrt(p)
        path = os.path.join(contrast_dir, '{}_group_u_{}_with_{}_to_{}_on_{}.nii.gz'.format(
            contrast, u, method, target_space, alignment_data))
        conj_img = masker.inverse_transform(conj)
        conj_img.to_filename(path)


# %%
# fix u, fix contrast, fetch_all_methods_ find how to plot
u = .25
method
"all", target_train
z_images
# 
plot_stat_map(conj_img)
method = "anat"
path = os.path.join(contrast_dir, '{}_group_u_{}_with_{}_to_{}_on_{}.nii.gz'.format(
    contrast, u, method, target_space, alignment_data))

# %%
contrast_dir = opj(root_folder, "ibc/cache_rest_movie_task/")
u = .2


alignment_data = "53"
target = "sub-04"

language_mask = [8, 10]
sounds_mask = [8, 12]
visu_mask = [2, 3, 5]
if language:
    contrasts_original = language_contrasts_original
    mask = language_mask
if sounds:
    contrasts_original = sounds_contrasts_original
    mask = sounds_mask
# contrasts = np.asarray(list(contrasts_original.keys()))[mask]
contrasts = np.hstack([np.asarray(list(sounds_contrasts_original.keys()))[
                      sounds_mask], np.asarray(list(language_contrasts_original.keys()))[language_mask]])


def plot_surf_im(path, ax, fsaverage=fetch_surf_fsaverage(), colorbar=False, threshold=0, vmax=8, hemi="left", view="lateral"):
    texture = surface.vol_to_surf(path, fsaverage.pial_left)
    display = plotting.plot_surf_stat_map(fsaverage.pial_left, texture,  hemi=hemi, colorbar=colorbar,
                                          threshold=threshold, vmax=vmax, bg_map=fsaverage.sulc_left, axes=axes[i, j], view=view)
    pass


def resize_surf_im(ax, zoom, offset):
    x_full, y_full, z_full = (-104, 78), (-104, 78), (-48, 78)
    xl = (x_full[0] / zoom + offset[0], x_full[1] / zoom + offset[0])
    yl = (y_full[0] / zoom + offset[1], y_full[1] / zoom + offset[1])
    zl = (z_full[0] / zoom + offset[2], z_full[1] / zoom + offset[2])
    ax.set_xlim3d(xl[0], xl[1])
    ax.set_ylim3d(yl[0], yl[1])
    ax.set_zlim3d(zl[0], zl[1])
    pass


def find_gt(task, contrast, target):
    if task == "sounds":
        task_ = "audio1"
        ffx_ = "audio"
    elif task == "rsvp-language":
        task_ = "rsvp-language"
        ffx_ = "language"
    else:
        task_ = "lyon2"
        ffx_ = "lyon_visu"
    gt_ = glob.glob(os.path.join("/storage", "store", "data", "ibc", "3mm",
                                 target, dict(get_subject_session([task_]))[target], "res_stats_{}_ffx".format(ffx_), "stat_maps", "{}*.nii.gz".format(contrast_lab)))[0]
    return gt_


fsaverage = fetch_surf_fsaverage("fsaverage")
fig, axes = plt.subplots(nrows=len(cached_methods) + 1, ncols=len(
    contrasts), subplot_kw={'projection': '3d'}, figsize=(6 * len(
        contrasts), 4 * (len(cached_methods) + 1)), constrained_layout=True)

for j, contrast in enumerate(contrasts):

    if contrast in sounds_contrasts_original.keys():
        task_name = "sounds"
    elif contrast in language_contrasts_original.keys():
        task_name = "rsvp-language"
    else:
        task_name = "lyon"
    path = os.path.join(contrast_dir, '{}_group_u_{}_with_{}_to_{}_on_{}.nii.gz'.format(
        contrast, u, method, target, alignment_data))
    cut_coords = None
    if task_name == "sounds":
        zoom = 4.5
        offset = (0, -15, +6)
        hemi = "left"
        view = "lateral"
    elif task_name == "rsvp-language":
        zoom = 3.5
        offset = (0, -8, -6)
        hemi = "left"
        view = "lateral"
    else:
        zoom = 2
        offset = (-30, -0, -0)
        hemi = "right"
        view = "lateral"  # ‘ventral’, ‘anterior’, ‘posterior’

    contrast_lab = contrast.replace(".nii.gz", "")

    for i, method in enumerate(cached_methods):
        # for contrast in ["belief-photo"]:
        #    for method in ["anat", "smoothing", "pairwise_ot_e-1", "pairwise_scaled_orthogonal"]:
        if method == "anat":
            target_space = "MNI"
        else:
            target_space = target

        if "HA" in method:
            if task_name == "sounds":
                vmax = 20
            elif task_name == "rsvp-language":
                vmax = 70
            else:
                vmax = 40

        elif "srm" in method:
            if task_name == "sounds":
                vmax = 4
            elif task_name == "rsvp-language":
                vmax = 6
            else:
                vmax = 4

        else:
            if task_name == "sounds":
                vmax = 5
            elif task_name == "rsvp-language":
                vmax = 8
            else:
                vmax = 6

        colorbar = False
        path = os.path.join(contrast_dir, '{}_group_u_{}_with_{}_to_{}_on_{}.nii.gz'.format(
            contrast, u, method, target_space, alignment_data))
        threshold = vmax / 4
        plot_surf_im(path, axes[i, j], fsaverage=fsaverage,
                     colorbar=colorbar, threshold=threshold, vmax=vmax, hemi=hemi, view=view)
        resize_surf_im(axes[i, j], zoom, offset)

    if task_name == "sounds":
        vmax = 7
        threshold = vmax / 4
    elif task_name == "rsvp-language":
        vmax = 10
        vmax = 2
    else:
        vmax = 6
    gt_ = find_gt(task_name, contrast_lab, target)
    i = len(cached_methods)
    plot_surf_im(gt_, axes[i, j], fsaverage=fsaverage,
                 colorbar=colorbar, threshold=threshold, vmax=vmax, hemi=hemi, view=view)
    resize_surf_im(axes[i, j], zoom, offset)

fig.savefig(os.path.join("/home/parietal/tbazeill",
                         "quali_audio_language_v2.png"), bbox_inches='tight')


# %% Mean visu
sounds = True
language = True
alignment_data = "53"
target = "sub-04"
language_mask = [8, 10]
sounds_mask = [8, 12]
visu_mask = [2, 3, 5]
if language:
    contrasts_original = language_contrasts_original
    mask = language_mask
if sounds:
    contrasts_original = sounds_contrasts_original
    mask = sounds_mask

# contrasts = np.asarray(list(contrasts_original.keys()))[mask]
contrasts = np.hstack([np.asarray(list(sounds_contrasts_original.keys()))[
                      sounds_mask], np.asarray(list(language_contrasts_original.keys()))[language_mask]])


fsaverage = fetch_surf_fsaverage("fsaverage5")
fig, axes = plt.subplots(nrows=len(cached_methods) + 1, ncols=len(
    contrasts), subplot_kw={'projection': '3d'}, figsize=(6 * len(
        contrasts), 4 * (len(cached_methods) + 1)), constrained_layout=True)

for j, contrast in enumerate(contrasts):

    if contrast in sounds_contrasts_original.keys():
        task_name = "sounds"
    elif contrast in language_contrasts_original.keys():
        task_name = "rsvp-language"
    else:
        task_name = "lyon"

    cut_coords = None
    if task_name == "sounds":
        zoom = 4.5
        offset = (0, -15, +6)
        hemi = "left"
        view = "lateral"
    elif task_name == "rsvp-language":
        zoom = 3.5
        offset = (0, -8, -6)
        hemi = "left"
        view = "lateral"
    else:
        zoom = 2
        offset = (-30, -0, -0)
        hemi = "right"
        view = "lateral"  # ‘ventral’, ‘anterior’, ‘posterior’

    contrast_lab = contrast.replace(".nii.gz", "")

    for i, method in enumerate(cached_methods):
        # for contrast in ["belief-photo"]:
        #    for method in ["anat", "smoothing", "pairwise_ot_e-1", "pairwise_scaled_orthogonal"]:

        if method == "anat":
            target_space = "MNI"
        else:
            target_space = target

        if "HA" in method:
            if task_name == "sounds":
                vmax = 10
            elif task_name == "rsvp-language":
                vmax = 30

        elif "srm" in method:
            if task_name == "sounds":
                vmax = 2
            elif task_name == "rsvp-language":
                vmax = 4
        elif "anat" in method:
            if task_name == "sounds":
                vmax = 3
            elif task_name == "rsvp-language":
                vmax = 6
        else:
            if task_name == "sounds":
                vmax = 2.5
            elif task_name == "rsvp-language":
                vmax = 5

        colorbar = False
        path = os.path.join(contrast_dir, '{}_group_mean_with_0.25_to_{}_on_{}.nii.gz'.format(
            contrast, method, target_space, alignment_data))
        threshold = vmax / 3
        plot_surf_im(path, axes[i, j], fsaverage=fsaverage,
                     colorbar=colorbar, threshold=threshold, vmax=vmax, hemi=hemi, view=view)
        resize_surf_im(axes[i, j], zoom, offset)

    if task_name == "sounds":
        vmax = 6

    elif task_name == "rsvp-language":
        vmax = 10
    threshold = vmax / 3
    gt_ = find_gt(task_name, contrast_lab, target)
    i = len(cached_methods)
    plot_surf_im(gt_, axes[i, j], fsaverage=fsaverage,
                 colorbar=colorbar, threshold=threshold, vmax=vmax, hemi=hemi, view=view)
    resize_surf_im(axes[i, j], zoom, offset)


# %%
display = plotting.plot_surf_stat_map(fsaverage.pial_left, surface.vol_to_surf(path, fsaverage.pial_left),  hemi='left', colorbar=colorbar,
                                      threshold=100, bg_map=fsaverage.sulc_left)
display.savefig("/home/parietal/tbazeill/fsaverage.png")

display = plotting.plot_surf_stat_map(fsaverage.pial_left, surface.vol_to_surf(path, fsaverage.pial_left),  hemi='left', colorbar=False,
                                      threshold=100, bg_map=fsaverage.sulc_left)
resize_surf_im(display.get_axes()[0], 4.5, (0, -15, +6))
display.savefig("/home/parietal/tbazeill/fsaverage_sounds.png")

display = plotting.plot_surf_stat_map(fsaverage.pial_left, surface.vol_to_surf(path, fsaverage.pial_left),  hemi='left', colorbar=colorbar,
                                      threshold=100, bg_map=fsaverage.sulc_left)
resize_surf_im(display.get_axes()[0], 3.5, (0, -8, -6))
display.savefig("/home/parietal/tbazeill/fsaverage_language.png")
# %%

# %%


# display = plot_stat_map(path, cut_coords=cut_coords,
#                        title=method, vmax=7, threshold=2)
# cut_coords = display.cut_coords
display = plotting.plot_surf_stat_map(fsaverage.pial_left, texture,    hemi='left',
                                      colorbar=False, threshold=threshold, vmax=vmax, bg_map=fsaverage.sulc_left)
# %%

masker = NiftiMasker(mask_img=mask).fit()
mask_im = load_img(mask)
plot_anat()
mni = load_mni152_template()
plot_anat(mni)
x1, x2 = 0, 100
y1, y2 = 0, 120
z = 50

mni.get_fdata().shape
extract = mni.get_fdata()[x1:x2, y1:y2, z]
plt.imshow(extract, origin='lower',
           cmap="gray", interpolation='nearest')
contrast_dir = "/storage/tompouce/tbazeill/ibc/cache_rest_movie_task/"

affine.dot((i, j, k, 1) ^ T) = (x, y, z, 1) ^ T
np.transpose([0, -60, 25, 1])
x, y, z =  # %%


conj_img = conjunction_inference_from_z_images(all_aligned, masker, u=.5)
