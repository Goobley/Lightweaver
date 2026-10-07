# Lightweaver

**C. Osborne (University of Glasgow) & I. Milić (NSO/CU Boulder), 2019-2021**

**MIT License**

Lightweaver is an NLTE radiative transfer code in the style of [RH](https://github.com/ITA-Solar/rh).
It is well validated against RH and also [SNAPI](https://github.com/ivanzmilic/snapi).
The code is currently designed for plane parallel atmospheres, either 1D single columns (which can be parallelised over wavelength) or 1.5D parallel columns with `ProcessPool` or MPI parallelisation.
There is also support for unpolarised radiative transfer in 2D atmospheres.

Lightweaver is described in a [paper (including examples!)](https://arxiv.org/abs/2107.00475), and has [API documentation](https://goobley.github.io/Lightweaver/).

Whilst the core numerics are implemented in C++, as much of the non-performance critical code as possible is implemented in Python, and the code currently only has a Python interface (provided through a Cython binding module).
Other languages with a C/C++ interface could interact directly with this core, hopefully allowing it to be reused as needed in different projects.

The aim of Lightweaver is to provide an NLTE Framework, rather than a "code".
That is to say, it should be more malleable, and provide easier access to experimentation, with most forms of experimentation (unless one wants to play with formal solvers or iteration schemes), being available directly from Python.
Formal solvers that comply with the interface defined in Lightweaver can be compiled into separate shared libraries and then loaded at runtime.
The preceding concepts are inspired by the popular Python machine learning frameworks such as PyTorch and TensorFlow.

## Installation

For most users precompiled Python wheels (supporting modern Linux, Mac, and Windows 10 systems) can be installed from `pip` and are the easiest way to get started with Lightweaver.
Lightweaver requires Python 3.10+, and it is recommended to be run inside a virtual environment, e.g. using `conda`, but other venv providers exist!
In this case a new virtual environment can be created with:
```
conda create -n Lightweaver python=3.12
```
activate the environment:
```
conda activate Lightweaver
```
and Lightweaver can then be installed with
```
python -m pip install lightweaver
```

### Installation from source

Whilst the above should work for most people, if you wish to work on the Lightweaver backend it is beneficial to have a source installation.
This requires a compiler supporting C++17.
The build is then run with `python3 -m pip install -vvv -e . --config-settings editable_mode=strict` (the strict mode is optional, but advised).
The libraries currently produce a few warnings, but should not produce any errors. This setup then enables the compilation of extension modules to shared libraries that can be loaded at runtime. This is advanced functionality -- ask me if you need help!

## Documentation

- [Paper](https://arxiv.org/abs/2107.00475).
- [API documentation](https://goobley.github.io/Lightweaver/).
- [Examples Gallery](https://goobley.github.io/Lightweaver/auto_examples/index.html).
- The `tests` directory can be a good place to check simple setup/runs on known-good cases.

Please contact me through this repository if difficulties are encountered. We also have a community Discord for discussions and knowledge-pooling.

## Acknowledgements

The [python implementation](https://github.com/jaimedelacruz/witt) of the Wittmann equation of state has been kindly provided J. de la Cruz Rodriguez.