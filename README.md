# FMPI

The repository contains Funnel-MPI (FMPI) which is a communication library to
obtain partial aggregation semantics for MPI collective communication. To obtain
a better understanding of the underlying algorithmus please refer to my
dissertation [1].

The library relies on modern C++ to maximize usability and performance.

[1]: Roger Kowalewski,  Partial aggregation for collective communication in
distributed memory machines, Dissertation,
[2021](https://doi.org/10.5282/edoc.28610)

## Building

The underlying build system relies on cmake and requires the following
dependencides on your local system:

- Any MPI library following the MPI-3 or MPI-4 standard.
- A compiler supporting at least C++17.

Before building please initialize required submodules:

```
git submodule update --init --recursive
```

## Usage

FMPI can be linked either statically or as a shared library into you scientific
application.
