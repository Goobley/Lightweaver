.. Lightweaver documentation master file, created by
   sphinx-quickstart on Fri Sep 25 16:56:45 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Lightweaver's documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   auto_examples/index
   lightweaver-api

Introduction
------------

Lightweaver is a Non-LTE (NLTE) radiative transfer framework for constructing
simulations of optically thick spectral lines (primarily solar). It is
available under the `MIT license`_. The flexible nature of its design allows
it to be simply used in direct line synthesis or as a component of more
involved frameworks (such as radiation hydrodynamics or inversion packages).
Currently, one dimensional plane-parallel, and two-dimensional Cartesian
atmospheres are supported, but three-dimensional setups could be added simply
with a specific formal solver. It is well tested against RH_ and SNAPI_.

.. _MIT license: https://opensource.org/licenses/MIT
.. _RH: https://github.com/ITA-Solar/rh
.. _SNAPI: https://github.com/ivanzmilic/snapi

The framework is managed from python (3.8+) whilst calling into an optimised
C++ backend. Whilst following a similar methods, Lightweaver is often
considerably faster than RH on non-trivial problems, whilst being slightly
slower on trivial single-threaded problems due to the increased
initialisation time the python layer brings with it.
The backend can parallelise directly over as many threads as desired on
single address space systems. MPI parallelisation across nodes can be
accomplished manually on top of the framework using the packages available in
python.

Whilst the core numerics are implented in C++, as much of the non-performance
critical code as possible is implemented in Python, and the code currently
only has a Python interface (provided through a Cython binding module). These
bindings could be rewritten in another language, so long as the same
information can be provided to the C++ core.

The aim of Lightweaver is to provide an NLTE Framework, rather than a "code".
That is to say, it should be more maleable, and provide easier access to
experimentation, with most forms of experimentation (unless one wants to play
with formal solvers or iteration schemes), being available directly from
python. Formal solvers that comply with the interface defined in Lightweaver
can be compiled into separate shared libraries and then loaded at runtime.
The preceding concepts are inspired by the well-recieved machine learning
frameworks such as PyTorch and Tensorflow.

Installation
------------

The most recent release of Lightweaver will always be available on PyPI and
is pre-compiled for Linux, Windows, and macOS (intel).

Lightweaver requires python 3.8+, and it is recommended to be run inside a virtual environment using ``conda``.

In this case a new virtual environment can be created with:

.. code-block:: bash

   conda create -n Lightweaver python=3.8

then, activate the environment:

.. code-block:: bash

   conda activate Lightweaver

and Lightweaver can then be installed with

.. code-block:: bash

   python -m pip install lightweaver

On other platforms you will need to compile the library, which will require a
C++17 compiler on your path, at which point setuptools should handle
everything. After downloading the release, or mainline version that you wish
to install, you should be able to

.. code-block:: bash

   python -m pip install .

from the Lightweaver directory created. You may also wish to do this to
create a more optimised version for modern machines; the PyPI versions
support Sandy Bridge CPUs, so newer machines may have wider instruction sets
available.
For doing a "development" installation for working inplace on the project,
you may with to use ``python -m pip install -e .``.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
