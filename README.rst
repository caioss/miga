========
  MIGA
========
**MIGA** is a Python package that provides a MSA (Multiple Sequence Alignment) mutual information genetic algorithm optimizer. It sorts two MSAs in a way that maximize or minimize their mutual information. The genetic algorithm solvers may run on both CPU and Nvidia GPUs.

This code is available under the GNU Lesser General Public License, version 3 (see LICENSE_ file).

Requirements
============
* Python version 3+
* GCC and G++
* Numpy

Optional requirements
=====================
* CUDA capable GPU with compute capability >= 3.0
* CUDA Toolkit version 9+
* Cython 0.22+

Instalation
===========
CUDA builds
-----------
For CUDA enabled installation, make sure the :code:`CUDA_HOME` is set and pointing to a valid CUDA 9+ installation root.

Pip
---
Run :code:`pip install miga`

Distributed packages
--------------------
1. Download the latest release_.
2. Run :code:`pip install miga.version.tar.gz`

From source
-----------
1. Make sure Cython version 0.22+ is installed
2. Clone this repository
3. Run :code:`git submodule update --init --recursive` to update submodules
4. Optionally set the environment variable :code:`CUDA_HOME` to point to your CUDA Toolkit installation
5. Run :code:`pip install miga/package`

Usage
=====
Plese refer to the examples_ folder and to `online documentation`_ to learn how to use this package.

Bugs and feature requests
=========================
Please report bugs and feature requests through the `Issues page`_.

Benchmark
=========

.. Footnotes
.. _LICENSE: https://github.com/caioss/miga/blob/master/LICENSE
.. _release: https://github.com/caioss/miga/releases
.. _examples: https://github.com/caioss/miga/tree/master/examples
.. _online documentation: https://miga.readthedocs.io
.. _Issues page: https://github.com/caioss/miga/issues
