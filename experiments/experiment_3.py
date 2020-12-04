from os.path import join as opj
import os
import warnings
import numpy as np
from matplotlib import pyplot as plt
import pickle
from nilearn import plotting, surface
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.input_data import NiftiMasker
from nilearn.image import load_img
from fmralign.pairwise_alignment import PairwiseAlignment
from fmralign.alignment_methods import OptimalTransportAlignment
from fmralignbench.fastsrm import FastSRM
from fmralignbench.utils import (fetch_resample_basc, _check_srm_params,
                                 find_method_label, make_coordinates_grid,
                                 check_input_method, fetch_resample_schaeffer,
                                 mask_gm)
from fmralignbench.conf import ROOT_FOLDER, N_JOBS
from fmralignbench.fetchers import fetch_ibc

warnings.filterwarnings(action='once')


def save_contrast(align, source_test):
    print(source_test)
    print(align)
    dir_ = os.path.dirname(align)
    model = align.split("/")[-1].split(".")[0]
    sub_test, contrast = source_test[:-7].split("/")[-1].split("_")
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
        root_folder, "alignment", "{}_53_contrasts.nii.gz".format(sub)) for sub in subjects])
    paths_contrasts = np.asarray([np.asarray([os.path.join(root_folder, "contrasts",
                                                           "{}_{}.nii.gz".format(sub, contrast)) for contrast in contrasts]) for sub in subjects]).T
    for paths_contrast in paths_contrasts:
        mask = opj(root_folder, 'masks', 'gm_mask_3mm.nii.gz')
        subject_LO = 0
        sources_train = paths_align[np.arange(len(subjects)) != subject_LO]

        sources_test = paths_contrast[np.arange(len(subjects)) != subject_LO]
        target_train = paths_align[subject_LO]
        target_contrast = paths_contrast[subject_LO]

        alignment_save(method, pairwise_method, local_align_method,
                       sources_train, sources_test, target_train, mask)
    pass


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


data = fetch_ibc(data_dir=ROOT_FOLDER)
contrasts = ["speech-silence", "voice-silence",
             "sentence-word", "word-consonant-string"
             ]
subjects = ['sub-04', 'sub-05', 'sub-06', 'sub-07',
            'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14']
contrasts_original = np.asarray([np.asarray([os.path.join(ROOT_FOLDER, "contrasts",
                                                          "{}_{}.nii.gz".format(sub, contrast)) for contrast in contrasts]) for sub in subjects]).T
alignment_data = "53"
target_ind = 0
target = subjects[target_ind]
methods = ["pairwise_scaled_orthogonal", "pairwise_ot_e-1", "srm", "HA"]
cached_methods = ["anat", "pairwise_ot_e-1",
                  "pairwise_scaled_orthogonal", "HArad_5_sparse_3", "srm_50_basc_444"]
contrast_dir = opj(ROOT_FOLDER, "alignment")
u = 0.25

# First part of the pipeline : Create and save align estimators and aligned contrasts


for input_method in methods:
    method, pairwise_method, local_align_method = check_input_method(
        input_method)
    run_save_align_for_tasks_and_contrasts(
        subjects, method, pairwise_method, local_align_method, root_folder=ROOT_FOLDER)

masker = NiftiMasker(mask_img=mask_gm).fit()


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

        Z = masker.transform(all_aligned).T
        p = int((1 - u) * Z.shape[1])
        Z_ = np.sort(Z, 1)
        conj = np.sum(Z_[:, :p], 1) / np.sqrt(p)
        path = os.path.join(contrast_dir, '{}_group_u_{}_with_{}_to_{}_on_{}.nii.gz'.format(
            contrast, u, method, target_space, alignment_data))
        conj_img = masker.inverse_transform(conj)
        conj_img.to_filename(path)
# %%
fsaverage = fetch_surf_fsaverage("fsaverage5")
fig, axes = plt.subplots(nrows=len(cached_methods) + 1, ncols=len(
    contrasts), subplot_kw={'projection': '3d'}, figsize=(6 * len(
        contrasts), 4 * (len(cached_methods) + 1)), constrained_layout=True)

for j, contrast in enumerate(contrasts):

    cut_coords = None
    if j < 2:
        zoom = 4.5
        offset = (0, -15, +6)
        hemi = "left"
        view = "lateral"
    elif j >= 2:
        zoom = 3.5
        offset = (0, -8, -6)
        hemi = "left"
        view = "lateral"

    for i, method in enumerate(cached_methods):
        if method == "anat":
            target_space = "MNI"
        else:
            target_space = target

        if "HA" in method:
            if j == 1:
                vmax = 30
            elif j == 0:
                vmax = 70
            elif j >= 2:
                vmax = 70
            else:
                vmax = 40

        elif "srm" in method:
            if j == 1:
                vmax = 3
            elif j == 0:
                vmax = 6
            elif j >= 2:
                vmax = 6
            else:
                vmax = 4
        else:
            if j == 1:
                vmax = 4
            elif j == 0:
                vmax = 7
            elif j >= 2:
                vmax = 8

        colorbar = False
        path = os.path.join(contrast_dir, '{}_group_u_{}_with_{}_to_{}_on_{}.nii.gz'.format(
            contrast, u, method, target_space, alignment_data))
        threshold = vmax / 4
        plot_surf_im(path, axes[i, j], fsaverage=fsaverage,
                     colorbar=colorbar, threshold=threshold, vmax=vmax, hemi=hemi, view=view)
        resize_surf_im(axes[i, j], zoom, offset)

    if j == 1:
        vmax = 7
    elif j == 0:
        vmax = 13
    elif j >= 2:
        vmax = 10
    threshold = vmax / 4
    gt_ = contrasts_original[j][target_ind]
    i = len(cached_methods)
    plot_surf_im(gt_, axes[i, j], fsaverage=fsaverage,
                 colorbar=colorbar, threshold=threshold, vmax=vmax, hemi=hemi, view=view)
    resize_surf_im(axes[i, j], zoom, offset)

    """ No labelling for now, hard to do in 3D
    vertical_labs = ["Anatomical", "Piecewise\nOptimal\nTransport",
                     "Piecewise\nProcrustes", "Searchlight\nHyperalignment", "Shared\nResponse\nModel", "Target"]

    for i, lab in enumerate(vertical_labs):
        # plt.text(-.25, 1,.5 "bla", "bla", "bla", fontsize=12)
        plt.text(-.25, (1 - i / len(vertical_labs)), .5, lab, fontsize=40)

    for i, cont in enumerate(contrasts):
        formatted_lab = cont.replace(" ", "\n")
        print(formatted_lab)
        plt.text(1.1, (i / len(vertical_labs)), .5, formatted_lab, fontsize=40)
    """

    plt.tight_layout()
if not os.path.isdir(os.path.join(ROOT_FOLDER, "figures")):
    os.mkdir(os.path.join(ROOT_FOLDER, "figures"))
fig.savefig(os.path.join(ROOT_FOLDER, "figures",
                         "experiment3_qualitative.png"), bbox_inches='tight')
