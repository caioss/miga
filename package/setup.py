import os
from distutils.version import LooseVersion
from setuptools import setup, find_packages, Extension

REQUIRED_CYTHON = "0.22"

# Cython detection
def get_cython():
    if "USE_CYTHON" in os.environ:
        try:
            value = int(os.environ["USE_CYTHON"])
        except ValueError:
            return False

        if value == 1:
            try:
                import Cython
            except ImportError:
                return False

            if LooseVersion(Cython.__version__) < LooseVersion(REQUIRED_CYTHON):
                print("Cython version mismatch. Version {} is installed but pyvmd needed version {}. Cython won't be used.".format(Cython.__version__, REQUIRED_CYTHON))
                return False

            return True

    return False

# Get readme content
with open("README.rst") as fd:
    LONG_DESCRIPTION = fd.read()

use_cython = get_cython()
cython_c_suffix = ".pyx" if use_cython else ".c"
cython_cpp_suffix = ".pyx" if use_cython else ".cpp"

extensions = [
    Extension(
        "miga.miga",
        ["miga/miga" + cython_cpp_suffix],
        language = "c++",
        include_dirs = [],
        libraries = [],
        extra_compile_args = ["-O3", "-Wall", "-pedantic", "-std=c++11", "-fopenmp", "-march=native"],
        extra_link_args=["-std=c++11", "-fopenmp"]
    ),
]

if use_cython:
    from Cython.Build import cythonize
    extensions = cythonize(extensions, gdb_debug=False)

setup(
    name = "miga",
    version = "0.0.1",

    description = "Python package to solve protein networks based on sequence coevolution.",
    long_description = LONG_DESCRIPTION,
    url = "no page yet",
    author = "Caio S. Souza",
    author_email = "caiobiounb@gmail.com",
    license = "LGPLv3+",
    classifiers=[
        "Development Status :: 3 - Alpha",

        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",

        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
    ],
    keywords = "theory information alignment",
    packages = find_packages(),
    package_dir = {"miga": "miga"},
    ext_package = "miga",
    ext_modules = extensions,
    install_requires = ["numpy"],
)
