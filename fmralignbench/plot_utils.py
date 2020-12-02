import copy
from matplotlib.lines import Line2D
from matplotlib import gridspec
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import numpy as np
import csv
from fmralignbench.utils import experiments_variables
from fmralignbench.conf import ROOT_FOLDER

DATASET_LIST = ["ibc_rsvp", "ibc_tonotopy_cond"]
DATASETS_LABELS = {"ibc_rsvp": "IBC RSVP", "ibc_tonotopy_cond": "IBC Sounds"}
METHOD_WIDTH = .5
METHODS_LABELS_DICT = {"anat_inter_subject": "Anatomical",
                       "intra_subject_ridge_cv": "Intra Subject",
                       "intra_subject_ridge_cv_schaefer_1000": "Intra Subject",
                       "srm_50_basc_444": "Shared\nResponse\nModel",
                       "srm_21_basc_444": "Shared\nResponse\nModel",
                       "HArad_5_sparse_3": "Searchlight\nHyperalignment",
                       "pairwise_scaled_orthogonal_schaefer_300": "Piecewise\nProcrustes",
                       "pairwise_ot_e-1_schaefer_300": "Piecewise\nOptimal\nTransport",
                       "smoothing_05": "Smoothing 05mm",
                       "smoothing_10": "Smoothing 10mm",
                       "smoothing_15": "Smoothing 15mm",
                       "smoothing_20": "Smoothing 20mm",
                       "smoothing_25": "Smoothing 25mm",
                       "smoothing_30": "Smoothing 30mm"}


def parse_score(decoding_task, alignment_data_label, roi_code, decoding_dir, method):
    path_to_score = os.path.join(decoding_dir, "{}_{}_{}_on_{}.csv".format(
        decoding_task, roi_code, method, alignment_data_label))
    path_to_fit_timings = os.path.join(decoding_dir, "{}_{}_{}_on_{}_fit_timings.csv".format(
        decoding_task, roi_code, method, alignment_data_label))
    path_to_overhead_timings = os.path.join(decoding_dir, "{}_{}_{}_on_{}_overhead_timings.csv".format(
        decoding_task, roi_code, method, alignment_data_label))

    with open(path_to_score, 'r') as f:
        reader = csv.reader(f)
        score = list(reader)[0]
    try:
        with open(path_to_fit_timings, 'r') as f:
            reader = csv.reader(f)
    except FileNotFoundError:
        fit_time = []
    else:
        with open(path_to_fit_timings, 'r') as f:
            reader = csv.reader(f)
            fit_time = list(reader)[0]

    try:
        with open(path_to_overhead_timings, 'r') as f:
            reader = csv.reader(f)
    except FileNotFoundError:
        overhead_time = []
    else:
        with open(path_to_overhead_timings, 'r') as f:
            reader = csv.reader(f)
            overhead_time = list(reader)[0]

    return score, fit_time, overhead_time


def fetch_scores(decoding_tasks, alignment_data_label, roi_codes, decoding_dir, methods, axis="method", return_type="score"):
    """fetch score for one axis either "method", "task", "roi_code"
    returns 2D ndarray shape (not sure of order) CV dim X len(axis)
    """
    scores, fit_timings, overhead_timings = [], [], []

    if axis == "method":
        decoding_task, roi_code = decoding_tasks, roi_codes
        for method in methods:
            score, fit_time, overhead_time = parse_score(
                decoding_task, alignment_data_label, roi_code, decoding_dir, method)
            scores.append(score)
            if fit_time:
                fit_timings.append(fit_time)
            if overhead_time:
                overhead_timings.append(overhead_time)
    elif axis == "task":
        method, roi_code = methods, roi_codes
        for decoding_task in decoding_tasks:
            score, fit_time, overhead_time = parse_score(
                decoding_task, alignment_data_label, roi_code, decoding_dir, method)
            scores.append(score)
            if fit_time:
                fit_timings.append(fit_time)
            if overhead_time:
                overhead_timings.append(overhead_time)
    elif axis == "roi_code":
        method, decoding_task = methods, decoding_tasks
        for roi_code in roi_codes:
            score, fit_time, overhead_time = parse_score(
                decoding_task, alignment_data_label, roi_code, decoding_dir, method)
            scores.append(score)
            if fit_time:
                fit_timings.append(fit_time)
            if overhead_time:
                overhead_timings.append(overhead_time)

    parsed_scores = [np.hstack([[float(j) for j in i.replace(
        '[', '').replace(']', '').split(", ")] for i in score]) for score in scores]

    if return_type == "score":
        return np.asarray(parsed_scores, dtype=float).T
    elif return_type == "fit_time":
        parsed_times = [np.hstack([[float(j) for j in i.replace(
            '[', '').replace(']', '').split(", ")] for i in timing]) for timing in fit_timings]
        return np.asarray(parsed_times, dtype=float).T


def draw_plot(ax, data, positions, labels, edge_color, fill_color):
    bp = ax.boxplot(data, positions=positions, meanline=True, showmeans=True, patch_artist=True, meanprops=dict(
        linestyle='-', linewidth=1), medianprops=dict(linestyle='', linewidth=0, color='blue'), vert=False, labels=labels)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)


def plot_one(scores, methods_labels, title="", save=None, figsize=(12, 12)):
    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(1, len(methods_labels) + 1)
    draw_plot(ax, scores, positions, None, "red", "pink")
    plt.axvline(scores[:, 0].mean(), c="red", alpha=0.5)
    ax.tick_params(axis='x', labelsize=16)
    plt.xlabel('Accuracy improvement(%)', fontsize=16)
    plt.legend(fontsize=16)
    plt.title(title, fontsize=20)
    # plt.yticks(positions, ["", "", "", "", "", "", ""], fontsize=18)
    plt.yticks(positions, methods_labels, fontsize=18)
    plt.legend(fontsize=14)
    if save:
        plt.savefig(save, bbox_inches='tight')
    pass


def make_diff(decoding_task, methods, roi_code, alignment_data_label, decoding_dir):
    scores = fetch_scores(
        decoding_task, alignment_data_label, roi_code, decoding_dir, methods)
    scores_diff = np.vstack([scores[:, i] - scores[:, 0]
                             for i in range(1, np.shape(scores)[1])])
    return scores_diff


def swap_axis_time(time):
    swapped = []
    i = 0
    while i < len(time[0][0]):
        swapped.append([b_j.T[i] for b_j in time])
        i += 1
    return swapped


def swap_two_first_axis(data):
    swapped = []
    i = 0
    while i < len(data[0]):
        swapped.append([b_j[i] for b_j in data])
        i += 1
    return swapped


def background_zebra(ax, positions, method_width):
    i = 0
    for position in positions:
        if i % 2 == 0:
            plt.axhspan(position - method_width, position +
                        method_width, facecolor='0.2', alpha=0.1)
        i += 1
    pass


def plot_one_method(ax, data, position, method_width, cmap, median=False):
    n_datasets = len(data)
    offsets = np.linspace(-method_width / 2 + .1,
                          method_width / 2 - .1, n_datasets)
    from matplotlib.colors import LinearSegmentedColormap

    if not isinstance(cmap, LinearSegmentedColormap):
        colors = cmap
    else:
        colors = cmap(np.linspace(0, 1, n_datasets))
    if median:
        meanprops = dict(linestyle='', linewidth=0)
        medianprops = dict(linestyle='-', linewidth=3, color="g")
    else:
        meanprops = dict(linestyle='-', linewidth=3, color="g")
        medianprops = dict(linestyle='', linewidth=0)

    bp = ax.boxplot(np.hstack(data), positions=[position], meanline=True, showmeans=True, patch_artist=True,
                    meanprops=meanprops, medianprops=medianprops, vert=False, widths=method_width, boxprops=dict(facecolor=(0, 0, 0, 0)))
    for dataset, offset, c in zip(data, offsets, colors):
        n = len(dataset)
        ax.plot(dataset, position * np.ones(n) +
                offset, 'o', color=c, ms=4)
    pass


def make_perf_sub(ax, methods_data, positions, method_width, y_lims, cmap, x_ticks=[-.05, -.02, 0, .02, .05, .08], x_lims=(-.08, .10), title="Decoding accuracy improvement"):
    ax.axvline(0, y_lims[0], y_lims[1], ls='--', c='grey', lw=2)
    ax.grid("x")
    for meth_data, position in zip(methods_data, positions):
        plot_one_method(ax, meth_data, position, method_width, cmap)
    background_zebra(ax, positions, method_width)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlim(left=x_lims[0], right=x_lims[1])
    ax.tick_params(axis="x", labelsize=14)
    ax.set_title(title, fontsize=16)
    pass


def make_time_sub(ax, methods_times, positions, method_width, y_lims, cmap):
    ax.axvline(0, y_lims[0], y_lims[1], ls='--', c='grey', lw=2)
    ax.grid("x")
    for meth_data, position in zip(methods_times, positions):
        plot_one_method(ax, meth_data, position,
                        method_width, cmap, median=True)
    background_zebra(ax, positions, method_width)
    ax.semilogx()

    ax.set_yticks([])
    ax.set_xticks([1, 2, 10, 30])

    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("x%.f"))
    ax.set_xlim(left=.1, right=40)
    ax.tick_params(axis="x", labelsize=14,)
    ax.set_title("Relative computation time",
                 fontsize=16, fontname="Helvetica")
    pass


def fetch_dataset_roi(methods, ROI, task, root_dir, add_srm=False, surf=False):
    if not ROI:
        roi_code = "fullbrain"
    decoding_dir = experiments_variables(
        task, root_dir=root_dir).out_dir
    methods_ = copy.deepcopy(methods)
    alignment_data_label = "53_tasks"
    if task == "ibc_rsvp" and ROI:
        roi_code = "language_3mm"
        srm_ = "srm_26_basc_444"
    elif task == "ibc_tonotopy_cond" and ROI:
        roi_code = "audio_3mm"
        srm_ = "srm_21_basc_444"
    if not ROI:
        srm_ = "srm_50_basc_444"
    if add_srm:
        methods_.append(srm_)
    if surf:
        alignment_data_label = "53_tasks"
        roi_code = "fullres_fullbrain"
        task = "surf_" + task
    score = fetch_scores(
        task, alignment_data_label, roi_code, decoding_dir, methods_)
    return score


def make_score_diffs(datasets_list, methods, ROI, add_srm, root_dir, fit_times=True):
    score_diffs, times = [], []
    for task in datasets_list:
        decoding_dir = experiments_variables(task, root_dir=root_dir).out_dir
        methods_ = copy.deepcopy(methods)
        alignment_data_label = "53_tasks"
        if task == "ibc_rsvp" and ROI:
            roi_code = "language_3mm"
            srm_ = "srm_26_basc_444"
        elif task == "ibc_tonotopy_cond" and ROI:
            roi_code = "audio_3mm"
            srm_ = "srm_21_basc_444"
        if not ROI:
            srm_ = "srm_50_basc_444"
            roi_code = "fullbrain"
        if add_srm:
            methods_.append(srm_)
        score_diffs.append(make_diff(task, methods_, roi_code,
                                     alignment_data_label, decoding_dir))

        if fit_times:
            times.append(fetch_scores(
                task, alignment_data_label, roi_code, decoding_dir, methods_[1:], return_type="fit_time"))
    return score_diffs, times, methods_


def plot_props(n_methods, method_width=METHOD_WIDTH):
    cmap = cm.rainbow
    colors = cmap(np.linspace(0, 1, 5))
    positions = np.arange(0.5, 2 * n_methods * METHOD_WIDTH, 2 * METHOD_WIDTH)
    y_lims = (positions[0] - METHOD_WIDTH, positions[-1] + METHOD_WIDTH)
    return cmap, colors, positions, y_lims


def make_smoothing_figure():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rc('font', family='Helvetica')
    smoothing_methods = ["anat_inter_subject", "smoothing_30", "smoothing_25", "smoothing_20",
                         "smoothing_15", "smoothing_10", "smoothing_05", "pairwise_scaled_orthogonal_schaefer_300"]
    score_diffs, _, methods_ = make_score_diffs(
        DATASET_LIST, smoothing_methods, False, False, ROOT_FOLDER)
    swapped = swap_two_first_axis(score_diffs)
    cmap, colors, positions, y_lims = plot_props(len(methods_) - 1)
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(1, 1)
    ax0 = plt.subplot(gs[0])
    labels = [METHODS_LABELS_DICT[method]for method in methods_[1:]]
    ax0.set_ylim(y_lims[0], y_lims[1])
    make_perf_sub(ax0, swapped, positions, METHOD_WIDTH, y_lims, colors)
    plt.yticks(positions, labels, fontsize=14, fontname="sans-serif")
    legends_points = []
    for dataset, c in zip(DATASET_LIST, colors):
        legends_points.append(Line2D([0], [0], marker='o', ls="", color=c, label=DATASETS_LABELS[dataset],
                                     markerfacecolor=c, markersize=10))
    legends_points.reverse()
    plt.rcParams['legend.title_fontsize'] = 16
    plt.legend(handles=legends_points, loc=(1.2, .5), fancybox=True,
               fontsize=14, title="Decoding tasks")
    if not os.path.isdir(os.path.join(ROOT_FOLDER, "figures")):
        os.mkdir(os.path.join(ROOT_FOLDER, "figures"))
    plt.savefig(os.path.join(ROOT_FOLDER, "figures",
                             "supplementary_3.png"), bbox_inches='tight')
    pass


def make_bench_figure(ROI):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rc('font', family='Helvetica')
    methods = ["anat_inter_subject", "intra_subject_ridge_cv_schaefer_1000", "HArad_5_sparse_3",
               "pairwise_scaled_orthogonal_schaefer_300", "pairwise_ot_e-1_schaefer_300"]
    score_diffs, times, methods_ = make_score_diffs(
        DATASET_LIST, methods, ROI, True, ROOT_FOLDER)
    swapped = swap_two_first_axis(score_diffs)
    ref_index = methods_.index("pairwise_scaled_orthogonal_schaefer_300") - 1
    swapped_time = swap_axis_time(times)
    normalized_times = [[s[l] / np.mean(swapped_time[ref_index][l])
                         for l in range(len(s))] for s in swapped_time]
    cmap, colors, positions, y_lims = plot_props(len(methods_) - 1)
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    ax0 = plt.subplot(gs[0])
    labels = [METHODS_LABELS_DICT[method]for method in methods_[1:]]
    ax0.set_ylim(y_lims[0], y_lims[1])
    if ROI:
        roi_ = "ROI "
    else:
        roi_ = ""
    make_perf_sub(ax0, swapped, positions, METHOD_WIDTH, y_lims,
                  colors, title="{}Decoding accuracy improvement".format(roi_))
    plt.yticks(positions, labels, fontsize=14, fontname="sans-serif")
    ax1 = plt.subplot(gs[1])
    make_time_sub(ax1, normalized_times, positions,
                  METHOD_WIDTH, y_lims, colors)
    legends_points = []
    for dataset, c in zip(DATASET_LIST, colors):
        legends_points.append(Line2D([0], [0], marker='o', ls="", color=c, label=DATASETS_LABELS[dataset],
                                     markerfacecolor=c, markersize=10))
    legends_points.reverse()
    plt.rcParams['legend.title_fontsize'] = 16
    plt.legend(handles=legends_points, loc=(1.2, .5), fancybox=True,
               fontsize=14, title="Decoding tasks")
    if not os.path.isdir(os.path.join(ROOT_FOLDER, "figures")):
        os.mkdir(os.path.join(ROOT_FOLDER, "figures"))
    if ROI:
        plt.savefig(os.path.join(ROOT_FOLDER, "figures",
                                 "experiment2.png"), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(ROOT_FOLDER, "figures",
                                 "experiment1.png"), bbox_inches='tight')
    pass


def make_within_subject_decoding_figure():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rc('font', family='Helvetica')
    methods = ["anat_inter_subject", "within_subject_decoding"]
    score_diffs, _, methods_ = make_score_diffs(
        DATASET_LIST, methods, False, False, ROOT_FOLDER)
    swapped = swap_two_first_axis(score_diffs)
    cmap, colors, positions, y_lims = plot_props(len(methods_) - 1)

    fig = plt.figure(figsize=(3, 1))
    gs = gridspec.GridSpec(1, 1)
    ax0 = plt.subplot(gs[0])
    ax0.set_ylim(y_lims[0], y_lims[1])
    make_perf_sub(ax0, swapped, positions, METHOD_WIDTH, y_lims, colors,
                  x_ticks=[0, 0.10, 0.20], x_lims=(-0.10, 0.25), title="Within - Inter-subject")
    plt.yticks(positions, [""for i in positions],
               fontsize=14, fontname="sans-serif")
    legends_points = []
    for dataset, c in zip(DATASET_LIST, colors):
        legends_points.append(Line2D([0], [0], marker='o', ls="", color=c, label=DATASETS_LABELS[dataset],
                                     markerfacecolor=c, markersize=10))
    legends_points.reverse()
    plt.rcParams['legend.title_fontsize'] = 16
    plt.legend(handles=legends_points, loc=(1.1, -.4), fancybox=True,
               fontsize=14, title="Decoding tasks")
    if not os.path.isdir(os.path.join(ROOT_FOLDER, "figures")):
        os.mkdir(os.path.join(ROOT_FOLDER, "figures"))
    plt.savefig(os.path.join(ROOT_FOLDER, "figures",
                             "experiment_1_within_decoding.png"), bbox_inches='tight')
    pass


def make_supplementary1_roi_minus_fullbrain_figure():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rc('font', family='Helvetica')
    methods_data = [[np.hstack(fetch_dataset_roi(
        methods, True, task, ROOT_FOLDER
    ) - fetch_dataset_roi(methods, False, task, ROOT_FOLDER
                          )) for task in DATASET_LIST] for methods in [["pairwise_scaled_orthogonal_schaefer_300"]]]
    cmap, colors, positions, y_lims = plot_props(1)

    fig = plt.figure(figsize=(3, 1))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    ax.set_ylim(y_lims[0], y_lims[1])
    ax.axvline(0, y_lims[0], y_lims[1], ls='--', c='grey', lw=2)
    ax.grid("x")

    for meth_data, position in zip(methods_data, positions):
        plot_one_method(ax, np.asarray(meth_data),
                        position, METHOD_WIDTH, colors)
        ax.set_xticks([-.1, -.05, 0, .05])
        ax.xaxis.set_major_formatter(
            mtick.PercentFormatter(xmax=1, decimals=0))
        ax.set_xlim(left=-.15, right=.08)
        ax.tick_params(axis="x", labelsize=14)
        ax.set_title("ROI - Fullbrain accuracy", fontsize=16)

    labels = ["Piecewise\nProcrustes", "Anatomical"]
    plt.yticks(positions, labels, fontsize=14, fontname="sans-serif")

    legends_points = []
    for dataset, c in zip(DATASET_LIST, colors):
        legends_points.append(Line2D([0], [0], marker='o', ls="", color=c, label=DATASETS_LABELS[dataset],
                                     markerfacecolor=c, markersize=10))
    legends_points.reverse()
    plt.rcParams['legend.title_fontsize'] = 16
    plt.legend(handles=legends_points, loc=(1.1, -1), fancybox=True,
               fontsize=14, title="Decoding tasks")
    if not os.path.isdir(os.path.join(ROOT_FOLDER, "figures")):
        os.mkdir(os.path.join(ROOT_FOLDER, "figures"))
    plt.savefig(os.path.join(ROOT_FOLDER, "figures",
                             "supplementary_1_roi_minus_fullbrain.png"), bbox_inches='tight')
    pass


def make_supplementary4_surface_volumic_figure():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rc('font', family='Helvetica')

    task = "ibc_rsvp"
    fullbrain_roi_diff = [np.hstack(fetch_dataset_roi(
        ["pairwise_scaled_orthogonal_schaefer_300"], False, task, ROOT_FOLDER
    ) - fetch_dataset_roi(["anat_inter_subject"], False, task, ROOT_FOLDER
                          ))]
    surf_fullbrain_roi_diff = [np.hstack(fetch_dataset_roi(
        ["pairwise_scaled_orthogonal"], False, task, ROOT_FOLDER, surf=True) - fetch_dataset_roi(["anat_inter_subject"], False, task, ROOT_FOLDER, surf=True))]
    cmap, colors, positions, y_lims = plot_props(2)

    fig, ax = plt.subplots(figsize=(3, 1))
    ax.set_ylim(y_lims[0], y_lims[1])
    ax.axvline(0, y_lims[0], y_lims[1], ls='--', c='grey', lw=2)
    ax.grid("x")
    methods_data = [surf_fullbrain_roi_diff, fullbrain_roi_diff]
    for meth_data, position in zip(methods_data, positions):
        plot_one_method(ax, np.asarray(meth_data),
                        position, METHOD_WIDTH, colors)
        ax.set_xticks([-.05, 0, .05, .10])
        ax.xaxis.set_major_formatter(
            mtick.PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(axis="x", labelsize=14)
        ax.set_title(
            "Piecewise Procrustes gains across representations", fontsize=16)
    labels = ["Cortical Surface (fsaverage7)", "Volumetric (3mm)"]
    plt.yticks(positions, labels, fontsize=14, fontname="sans-serif")
    legends_points = []
    for dataset, c in zip(DATASET_LIST, colors):
        legends_points.append(Line2D([0], [0], marker='o', ls="", color=c, label=DATASETS_LABELS[dataset],
                                     markerfacecolor=c, markersize=10))
    legends_points.reverse()
    plt.rcParams['legend.title_fontsize'] = 16
    plt.legend(handles=legends_points, loc=(1.1, 0), fancybox=True,
               fontsize=14, title="Decoding tasks")
    if not os.path.isdir(os.path.join(ROOT_FOLDER, "figures")):
        os.mkdir(os.path.join(ROOT_FOLDER, "figures"))
    plt.savefig(os.path.join(ROOT_FOLDER, "figures",
                             "supplementary_4_surfacic_vs_volumetric.png"), bbox_inches='tight')
    pass
