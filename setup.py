
from setuptools import setup, find_packages
import os
import sys
descr = """Benchmarking pipeline for functional alignment methods using decoding"""


def load_version():
    """Executes fmralignbench/version.py in a globals dictionary and return it.

    Note: importing fmralignbench is not an option because there may be
    dependencies like nibabel which are not installed and
    setup.py is supposed to install them.
    """
    # load all vars into globals, otherwise
    #   the later function call using global vars doesn't work.
    globals_dict = {}
    with open(os.path.join('fmralignbench', 'version.py')) as fp:
        exec(fp.read(), globals_dict)

    return globals_dict


def is_installing():
    # Allow command-lines such as "python setup.py build install"
    install_commands = set(['install', 'develop'])
    return install_commands.intersection(set(sys.argv))


# Make sources available using relative paths from this file's directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

_VERSION_GLOBALS = load_version()
DISTNAME = 'fmralignbench'
DESCRIPTION = 'Benchmarking pipeline for functional alignment methods using decoding'
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'Thomas Bazeille, Elizabeth Dupre, Bertrand Thirion'
MAINTAINER_EMAIL = 'thomas.bazeille@inria.fr'
URL = ''
LICENSE = 'new BSD'
DOWNLOAD_URL = ''
VERSION = _VERSION_GLOBALS['__version__']


if __name__ == "__main__":

    if is_installing():
        module_check_fn = _VERSION_GLOBALS['_check_module_dependencies']
        module_check_fn(is_benchmark_installing=True)

    install_requires = \
        ['%s>=%s' % (meta.get('pypi_name', mod), meta['min_version'])
            for mod, meta in _VERSION_GLOBALS['REQUIRED_MODULE_METADATA']]
    print(install_requires)
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
              'Programming Language :: Python :: 2',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3.3',
              'Programming Language :: Python :: 3.4',
              'Programming Language :: Python :: 3.5',
              'Programming Language :: Python :: 3.6',
          ],
          packages=find_packages(),
          package_data={
          },
          install_requires=install_requires,)
