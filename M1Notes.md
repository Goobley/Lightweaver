### Notes from experimenting with builds for M1 (cmo)

Currently conda/miniconda do not support Apple M1. By default, their versions of Python run through Rosetta2, which does not support AVX extensions. For this reason I have enabled universal2 builds on Darwin, and removed the specialised flags that expected Sandy Bridge+ processors (i.e. AVX) on this platform.
Whilst I have tested the generated Universal build through Rosetta2 on M1, I haven't tested the arm64 component, as I haven't had enough time to play with a true arm64 Python environnment, and assistance along these lines would be appreciated.

The effect of removing these flags may somewhat slow down Lightweaver on some intel macs, ARM based macs are becoming more prominent, and this hopefully allows them to use the off the shelf binaries compiled on the CD system.

If there is a more elegant way of doing this, I would love to hear about it.
