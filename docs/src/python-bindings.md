# Python Bindings
This page aims to clarify, document, and de-mystify the `tt-mlir` python bindings. It will do so by first highlighting the mechanism with which these bindings are generated and exposed to users. It will then document the nuances of `nanobind`, and the different parts of these bindings that must be written in by hand. Finally, it will go through a hands-on example of how to add your own functionality to the `tt-mlir` python bindings.

## Generating Bindings
This section will outline the mechanism with which bindings function,

### `nanobind`
Nanobind is the successor of the ubiquitous `pybind` project. In almost the same syntactical form, it provides a framework to define InterOp between C++ and Python. For more information about `nanobind` specifically, I'd recommend reading through the [documentation](https://nanobind.readthedocs.io/en/latest/index.html). `MLIR` (and by extension: `tt-mlir`) leverages `nanobind` to create
