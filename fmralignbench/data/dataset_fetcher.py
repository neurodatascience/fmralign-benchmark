import os
from pathlib import Path
from sklearn.utils import Bunch
from nilearn.datasets.utils import _get_dataset_dir, _fetch_files
from nilearn._utils.numpy_conversions import csv_to_array


def _fetch_ibc_masks(participants, data_dir, url, resume, verbose):
    """Helper function to fetch_ibc_fmri.

    This function helps in downloading masks for use with IBC
    functional alignment and inter-subject decoding.

    The files are downloaded from Open Science Framework (OSF).

    Parameters
    ----------
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

    # The gzip contains unique download keys per Nifti file and CSV
    # pre-extracted from OSF. Required for downloading files.
    package_directory = os.path.dirname(os.path.abspath(__file__))
    dtype = [('filename', 'U12'), ('uid', 'U24')]
    names = ['filename', 'uid']
    # csv file contains download information
    osf_data = csv_to_array(os.path.join(package_directory, "ibc_masks.csv"),
                            skip_header=True, dtype=dtype, names=names)

    derivatives_dir = Path(data_dir, 'ibc')

    for this_osf_id in osf_data:

        # Download mask
        mask_url = url.format(this_osf_id['uid'][0])
        mask_target = Path(derivatives_dir, this_osf_id['filename'][0])
        mask_file = [(mask_target,
                      mask_url,
                      {'move': mask_target})]
        path_to_mask = _fetch_files(data_dir, mask_file,
                                    verbose=verbose)[0]
        align.append(path_to_mask)

    return derivatives_dir


def _fetch_ibc_alignment(participants, data_dir, url, resume, verbose):
    """Helper function to fetch_ibc_fmri.

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

    alignment = '{0}_53_contrasts.nii.gz'

    # The gzip contains unique download keys per Nifti file and CSV
    # pre-extracted from OSF. Required for downloading files.
    package_directory = os.path.dirname(os.path.abspath(__file__))
    dtype = [('sid', 'U12'), ('alignment', 'U24')]
    names = ['sid', 'alignment']
    # csv file contains download information
    osf_data = csv_to_array(os.path.join(package_directory, "ibc_alignment.csv"),
                            skip_header=True, dtype=dtype, names=names)

    derivatives_dir = Path(data_dir, 'ibc')
    align, decode, labels, sessions = [], [], [], []

    for sid in participants['sid']:
        this_osf_id = osf_data[osf_data['sid'] == sid]

        # Download alignment
        alignment_url = url.format(this_osf_id['alignment'][0])
        alignment_target = Path(derivatives_dir, alignment.format(sid))
        alignment_file = [(alignment_target,
                           alignment_url,
                           {'move': alignment_target})]
        path_to_alignment = _fetch_files(data_dir, alignment_file,
                                         verbose=verbose)[0]
        align.append(path_to_alignment)

    return derivatives_dir


def _fetch_rsvp_trial(participants, data_dir, url, resume, verbose):
    """Helper function to fetch_ibc_fmri.

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

    betas = '{0}.nii.gz'
    conditions = '{0}_labels.csv'
    runs = '{0}_runs.csv'

    # The gzip contains unique download keys per Nifti file and CSV
    # pre-extracted from OSF. Required for downloading files.
    package_directory = os.path.dirname(os.path.abspath(__file__))
    dtype = [('sid', 'U12'), ('betas', 'U24'),
             ('condition', 'U24'), ('run', 'U24')]
    names = ['sid', 'betas', 'condition', 'run']
    # csv file contains download information
    osf_data = csv_to_array(os.path.join(package_directory, "rsvp_trial.csv"),
                            skip_header=True, dtype=dtype, names=names)

    derivatives_dir = Path(data_dir, 'rsvp_trial', '3mm')
    align, decode, labels, sessions = [], [], [], []

    for sid in participants['sid']:
        this_osf_id = osf_data[osf_data['sid'] == sid]

        # Download flanker
        betas_url = url.format(this_osf_id['betas'][0])
        betas_target = Path(derivatives_dir, betas.format(sid))
        betas_file = [(betas_target,
                       betas_url,
                       {'move': betas_target})]
        path_to_betas = _fetch_files(data_dir, betas_file,
                                     verbose=verbose)[0]
        decode.append(path_to_betas)

        # Download condition labels
        label_url = url.format(this_osf_id['condition'][0])
        label_target = Path(derivatives_dir, conditions.format(sid))
        label_file = [(label_target,
                       label_url,
                       {'move': label_target})]
        path_to_labels = _fetch_files(data_dir, label_file,
                                      verbose=verbose)[0]
        labels.append(path_to_labels)

        # Download session run numbers
        session_url = url.format(this_osf_id['run'][0])
        session_target = Path(derivatives_dir, runs.format(sid))
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


def _fetch_ibc_tonotopy(participants, data_dir, url, resume, verbose):
    """Helper function to fetch_ibc_fmri.

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

    betas = '{0}.nii.gz'
    conditions = '{0}_labels.csv'
    runs = '{0}_runs.csv'

    # The gzip contains unique download keys per Nifti file and CSV
    # pre-extracted from OSF. Required for downloading files.
    package_directory = os.path.dirname(os.path.abspath(__file__))
    dtype = [('sid', 'U12'), ('betas', 'U24'),
             ('condition', 'U24'), ('run', 'U24')]
    names = ['sid', 'betas', 'condition', 'run']
    # csv file contains download information
    osf_data = csv_to_array(os.path.join(package_directory, "ibc_tonotopy.csv"),
                            skip_header=True, dtype=dtype, names=names)

    derivatives_dir = Path(data_dir, 'ibc_tonotopy', '3mm')
    align, decode, labels, sessions = [], [], [], []

    for sid in participants['sid']:
        this_osf_id = osf_data[osf_data['sid'] == sid]

        # Download flanker
        betas_url = url.format(this_osf_id['betas'][0])
        betas_target = Path(derivatives_dir, betas.format(sid))
        betas_file = [(betas_target,
                       betas_url,
                       {'move': betas_target})]
        path_to_betas = _fetch_files(data_dir, betas_file,
                                     verbose=verbose)[0]
        decode.append(path_to_betas)

        # Download condition labels
        label_url = url.format(this_osf_id['condition'][0])
        label_target = Path(derivatives_dir, conditions.format(sid))
        label_file = [(label_target,
                       label_url,
                       {'move': label_target})]
        path_to_labels = _fetch_files(data_dir, label_file,
                                      verbose=verbose)[0]
        labels.append(path_to_labels)

        # Download session run numbers
        session_url = url.format(this_osf_id['run'][0])
        session_target = Path(derivatives_dir, runs.format(sid))
        session_file = [(session_target,
                         session_url,
                         {'move': session_target})]
        path_to_sessions = _fetch_files(data_dir, session_file,
                                        verbose=verbose)[0]
        sessions.append(path_to_sessions)

    # create out_dir
    Path(data_dir, "ibc_tonotopy", "decoding").mkdir(
        parents=True, exist_ok=True)

    # create mask_cache
    Path(data_dir, "ibc_tonotopy", "mask_cache").mkdir(
        parents=True, exist_ok=True)

    return derivatives_dir


def fetch_ibc(data_dir=None, resume=True, verbose=1):
    """Fetch the Individual Brain Charting (IBC) data.

    The data is downsampled to 3mm isotropic resolution. Please see
    Notes below for more information on the dataset as well as full
    pre- and post-processing details.


    Parameters
    ----------
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
    http://fcon_1000.projects.nitrc.org/indi/hbn_ssi/

    This fetcher downloads downsampled data that are available on Open
    Science Framework (OSF): https://osf.io/6ysra/files/

    Pre- and post-processing details for this dataset are made available
    here: https://osf.io/28qwv/wiki/

    References
    ----------
    Please cite this paper if you are using this dataset:

    O'Connor D, Potler NV, Kovacs M, Xu T, Ai L, Pellman J, Vanderwal T,
    Parra LC, Cohen S, Ghosh S, Escalera J, Grant-Villegas N, Osman Y, Bui A,
    Craddock RC, Milham MP (2017). The Healthy Brain Network Serial Scanning
    Initiative: a resource for evaluating inter-individual differences and
    their reliabilities across scan conditions and sessions.
    GigaScience, 6(2): 1-14
    https://academic.oup.com/gigascience/article/6/2/giw011/2865212
    """

    dataset_name = "ibc"
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=1)

    # Participants data
    brain_mask = _fetch_hbnssi_brain_mask(data_dir=data_dir, url=None,
                                          verbose=verbose)

    derivatives_dir = _fetch_hbnssi_functional(
        participants, data_dir=data_dir, url=None,
        resume=resume, verbose=verbose)

    return Bunch(subjects=participants['sid'], mask=brain_mask,
                 task_dir=derivatives_dir, out_dir=out_dir,
                 mask_cache=mask_cache)
