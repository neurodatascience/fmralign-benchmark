import os
from pathlib import Path
from sklearn.utils import Bunch
from nilearn.datasets.utils import _get_dataset_dir, _fetch_files
from nilearn._utils.numpy_conversions import csv_to_array


def _fetch_ibc_surf_alignment(participants, data_dir, url, resume, verbose):
    """Helper function to fetch_ibc.

    This function helps in downloading functional MRI data in Nifti format
    and its corresponding CSVs each subject for functional alignment and
    decoding.

    The files are downloaded from Open Science Framework (OSF).

    Parameters
    ----------
    participants : numpy.ndarray
        Should contain column participant_id which represents subjects id. The
        number of files are fetched based on ids in this column.

    data_dir: str
        Path of the data directory. Used to force data storage in a specified
        location. If None is given, data are stored in home directory.

    url: str, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data). Default: None

    resume: bool, optional (default True)
        Whether to resume download of a partly-downloaded file.

    verbose: int
        Defines the level of verbosity of the output.

    Returns
    -------
    func: list of str (Nifti files)
        Paths to functional MRI data (4D) for each subject.
    """
    if url is None:
        # Download from the relevant OSF project, using hashes generated
        # from the OSF API. Note the trailing slash. For more info, see:
        # https://gist.github.com/emdupre/3cb4d564511d495ea6bf89c6a577da74
        url = 'https://osf.io/download/{}/'

    alignment = '{0}_53_contrasts_{1}_fullres.gii'

    # The gzip contains unique download keys per Nifti file and CSV
    # pre-extracted from OSF. Required for downloading files.
    package_directory = os.path.dirname(os.path.abspath(__file__))
    dtype = [('sid', 'U12'), ('hemi', 'U12'), ('highres_surf', 'U24')]
    names = ['sid', 'hemi', 'highres_surf']
    # csv file contains download information
    osf_data = csv_to_array(os.path.join(package_directory, "ibc_surf.csv"),
                            skip_header=True, dtype=dtype, names=names)

    derivatives_dir = Path(data_dir, 'surf_ibc')
    align = []

    for sid in participants['sid']:
        this_osf_id = osf_data[osf_data['sid'] == sid]

        # Download alignment
        alignment_url = url.format(this_osf_id['highres_surf'][0])
        alignment_target = Path(derivatives_dir, alignment.format(sid, hemi))
        alignment_file = [(alignment_target,
                           alignment_url,
                           {'move': alignment_target})]
        path_to_alignment = _fetch_files(data_dir, alignment_file,
                                         verbose=verbose)[0]
        align.append(path_to_alignment)

    return derivatives_dir


def _fetch_rsvp_trial_surf(participants, data_dir, url, resume, verbose):
    """Helper function to fetch_ibc.

    This function helps in downloading functional MRI data in Nifti format
    and its corresponding CSVs each subject for functional alignment and
    decoding.

    The files are downloaded from Open Science Framework (OSF).

    Parameters
    ----------
    participants : numpy.ndarray
        Should contain column participant_id which represents subjects id. The
        number of files are fetched based on ids in this column.

    data_dir: str
        Path of the data directory. Used to force data storage in a specified
        location. If None is given, data are stored in home directory.

    url: str, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data). Default: None

    resume: bool, optional (default True)
        Whether to resume download of a partly-downloaded file.

    verbose: int
        Defines the level of verbosity of the output.

    Returns
    -------
    func: list of str (Nifti files)
        Paths to functional MRI data (4D) for each subject.
    """
    if url is None:
        # Download from the relevant OSF project, using hashes generated
        # from the OSF API. Note the trailing slash. For more info, see:
        # https://gist.github.com/emdupre/3cb4d564511d495ea6bf89c6a577da74
        url = 'https://osf.io/download/{}/'

    surf = '{0}_{1}.gii'
    highres_surf = '{0}_{1}_fullres.gii'
    conditions = '{0}_{1}_labels.csv'
    runs = '{0}_{1}_runs.csv'

    # The gzip contains unique download keys per Nifti file and CSV
    # pre-extracted from OSF. Required for downloading files.
    package_directory = os.path.dirname(os.path.abspath(__file__))
    dtype = [('sid', 'U12'), ('hemi', 'U12'), ('surf', 'U24'),
             ('highres_surf', 'U24'), ('condition', 'U24'), ('run', 'U24')]
    names = ['sid', 'hemi', 'surf', 'highres_surf', 'condition', 'run']
    # csv file contains download information
    osf_data = csv_to_array(
        os.path.join(package_directory, "rsvp_trial_surf.csv"),
        skip_header=True, dtype=dtype, names=names)

    derivatives_dir = Path(data_dir, 'rsvp_trial', 'surf_derivatives')
    surf_files, highres_surf_files, labels, sessions = [], [], [], []

    for sid, hemi in participants['sid']:
        this_osf_id = osf_data[osf_data['sid'] == sid]

        # Download surface
        surf_url = url.format(this_osf_id['surf'][0])
        surf_target = Path(derivatives_dir, surf.format(sid, hemi))
        surf_file = [(surf_target,
                      surf_url,
                      {'move': surf_target})]
        path_to_surf = _fetch_files(data_dir, surf_file,
                                    verbose=verbose)[0]
        surf_files.append(path_to_surf)

        # Download surface
        highres_surf_url = url.format(this_osf_id['highres_surf'][0])
        highres_surf_target = Path(derivatives_dir, highres_surf.format(sid, hemi))
        highres_surf_file = [(highres_surf_target,
                              highres_surf_url,
                              {'move': highres_surf_target})]
        path_to_highres_surf = _fetch_files(data_dir, highres_surf_file,
                                            verbose=verbose)[0]
        highres_surf_files.append(path_to_highres_surf)

        # Download condition labels
        label_url = url.format(this_osf_id['condition'][0])
        label_target = Path(derivatives_dir, conditions.format(sid, hemi))
        label_file = [(label_target,
                       label_url,
                       {'move': label_target})]
        path_to_labels = _fetch_files(data_dir, label_file,
                                      verbose=verbose)[0]
        labels.append(path_to_labels)

        # Download session run numbers
        session_url = url.format(this_osf_id['run'][0])
        session_target = Path(derivatives_dir, runs.format(sid, hemi))
        session_file = [(session_target,
                         session_url,
                         {'move': session_target})]
        path_to_sessions = _fetch_files(data_dir, session_file,
                                        verbose=verbose)[0]
        sessions.append(path_to_sessions)

    # create out_dir
    Path(data_dir, "rsvp_trial", "decoding").mkdir(
        parents=True, exist_ok=True)

    # create mask_cache
    Path(data_dir, "rsvp_trial", "mask_cache").mkdir(
        parents=True, exist_ok=True)

    return derivatives_dir


def fetch_ibc_surf(participants='all', data_dir=None, resume=True, verbose=1):
    """Fetch the Individual Brain Charting (IBC) data.

    The data is downsampled to 3mm isotropic resolution. Please see
    Notes below for more information on the dataset as well as full
    pre- and post-processing details.


    Parameters
    ----------
    participants: str or list, optional (default 'all')
        Which participants to fetch. By default all are fetched.
    data_dir: str, optional (default None)
        Path of the data directory. Used to force data storage in a specified
        location. If None, data are stored in home directory.
    resume: bool, optional (default True)
        Whether to resume download of a partly-downloaded file.
    verbose: int, optional (default 1)
        Defines the level of verbosity of the output.

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :

        - 'func': list of str (Nifti files)
            Paths to downsampled functional MRI data (4D) for each subject.

    Notes
    -----
    The original data is downloaded from the OpenNeuro data portal:
    https://openneuro.org/datasets/ds002685/versions/1.3.0

    This fetcher downloads preprocessed and downsampled data that are available
    on Open Science Framework (OSF): https://osf.io/6ysra/files/

    References
    ----------
    Please cite this paper if you are using this dataset:

    Pinho, A. L., Amadon, A., Gauthier, B., Clairis, N., Knops, A., Genon, S.,
    ... & Thirion, B. (2020).
    Individual Brain Charting dataset extension, second release of
    high-resolution fMRI data for cognitive mapping. Scientific Data, 7(1), 1-16.
    https://www.nature.com/articles/s41597-020-00670-4
    """

    dataset_name = "ibc"
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=1)

    if participants == 'all':
        rsvp_participants = [
            'sub-01.nii.gz', 'sub-04.nii.gz', 'sub-05.nii.gz', 'sub-06.nii.gz',
            'sub-07.nii.gz', 'sub-09.nii.gz', 'sub-11.nii.gz', 'sub-12.nii.gz',
            'sub-13.nii.gz', 'sub-14.nii.gz'
        ]
    else:
        rsvp_participants = participants

    # Participant-level data
    align_dir = _fetch_ibc_alignment(
        rsvp_participants, data_dir=data_dir, url=None,
        resume=resume, verbose=verbose)
    rsvp_dir = _fetch_rsvp_trial(
        rsvp_participants, data_dir=data_dir, url=None,
        resume=resume, verbose=verbose)

    return Bunch(mask=mask,
                 contrasts=contrast, align_dir=align_dir,
                 rsvp_dir=rsvp_dir, tonotopy_dir=tonotopy_dir)
