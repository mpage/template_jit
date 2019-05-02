# template_jit

This is a (very) bare bones template JIT compiler for Python. It was written to
accompany a PyCon 2019 talk aboout how to the same topic. It's intended to serve
as a starting point for playing around with constructing JIT compilers in Python.

## Motivation

Compilers are fun! But production JIT compilers (e.g. v8, HotSpot) are large,
complicated pieces of software that can seem inscrutable. This provides a simpler
starting point for anyone wanting to get their feet wet.

## Requirements

This requires at least Python 3.6.

## Getting started

We use `pipenv` for managing dependencies. After cloning the repo, run

```
> pipenv install
> pipenv shell
> env PYTHONPATH=./ pytest
```