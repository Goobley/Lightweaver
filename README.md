# Lightweaver

**C. Osborne (University of Glasgow) & I. Milić (NSO/CU Boulder), 2019**

**MIT License**

Lightweaver is an NLTE radiative transfer code in the style of [RH](https://github.com/ITA-Solar/rh). It is well validated against RH and also [SNAPI](https://github.com/ivanzmilic/snapi). The code is currently designed for plane parallel atmospheres, either 1D single columns (wavelength parallelisation in progress) or 1.5D parallel columns with (currently) `ProcessPool` parallelisation.

Whilst the core numerics are implented in C++, as much of the non-performance critical code as possible is implemented in Python, and the code currently only has a Python interface (provided through a Cython binding module). These bindings could be rewritten in another language, so long as the same information can be provided to the C++ core.

The aim of Lightweaver is to provide an NLTE Framework, rather than a "code". That is to say, it should be more maleable, and provide easier access to experimentation, with most forms of experimentation (unless one wants to play with formal solvers or iteration schemes), being available directly from python.

### Installation

Requirements:

    - C++17 Compiler
    - Python 3.7+

The code isn't currently packaged as a module, (this will happen when we hit beta), so code is designed to be run in the project directory for now.
Due to the use of C++ and the Cython build system it appears to be necessary when clang isn't the default compiler to set the local shell variable `CC` to the name of your C++ compiler (e.g. `g++-9`) before invoking the build script.

The build is then run with `python3 setup.py build_ext --inplace`. The libraries currently produce a few warnings, but should not produce any errors.
At this point one of the test scripts (such as `Test12.py`) can be run. I suggest using interactive mode for this and then plotting the results.

Some of these test scripts show examples of how to do 1.5D parallel column-by-column synthesis, and parallel numeric NLTE response functions.

### Documentation

Documentation is currently on the non-existent end. This will also be written properly as we approach beta. I suggest looking through the `Test*.py` files and the `Judge*.py` files (for time-dependent populations).


### Acknowledgements

The [python implementation](https://github.com/jaimedelacruz/witt) of the Wittmann equation of state kindly provided J. de la Cruz Rodriguez.