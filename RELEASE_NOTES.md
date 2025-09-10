# Release notes

## [Ideas]

- Further optimisation of backward step
    * Shared memory and tiling

## [0.2.2]

### Fixes

- Overload vmap in custom autograd function `QuarticBSplineFunction`
- Introduced tests w.r.t. vmap functionality

## [0.2.1]

### Fixes

- Fix build process such that imports work properly when installing the package via pip

## [0.2.0]

### Features

- First stable version of CUDA extension for forward- and backward step of quartic
    bspline potentials
- Tests and profiling scripts
- README.md
- Makefile for installing and building
