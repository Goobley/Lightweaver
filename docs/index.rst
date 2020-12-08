.. Lightweaver documentation master file, created by
   sphinx-quickstart on Fri Sep 25 16:56:45 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Lightweaver's documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   lightweaver-api

Lightweaver is a Non-LTE (NLTE) radiative transfer framework for constructing
simulations of optically thick spectral lines (primarily solar).
The flexible nature of its design allows it to be simply used in direct line
synthesis or as a component of more involved frameworks (such as radiation
hydrodynamics or inversion packages).

The framework is managed from python (3.8+) whilst calling into an optimised
C++ backend. Whilst following a similar procedure, Lightweaver is often
considerably faster than RH on non-trivial problems, whilst being slightly
slower on trivial single-threaded problems due to the increased
initialisation time the python layer brings with it.
The backend can parallelise directly over as many threads as desired on
single address space systems. MPI parallelisation across nodes can be
accomplished manually on top of the framework using the packages available in
python.

Installation
============

The most recent release of Lightweaver will always be available on PyPI and
is pre-compiled for Linux, Windows, and macOS (intel). In an environment with
python 3.8+ it can be installed with

``python -m pip install lightweaver``

On other platforms you will need to compile the library, which will require a
C++17 compiler on your path, at which point setuptools should handle
everything. After downloading the release, or mainline version that you wish
to install, you should be able to

``python -m pip install .``

from the Lightweaver directory created. You may also wish to do this to
create a more optimised version for modern machines; the PyPI versions
support Sandy Bridge CPUs, so newer machines may have wider instruction sets
available.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
