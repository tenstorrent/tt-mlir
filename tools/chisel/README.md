# ttmlir-chisel

Chisel runtime callbacks, validators, and reporting for TT-MLIR. Hooks into
the tt-mlir runtime to compare op outputs against golden references during
execution.

## Install

`ttmlir-chisel` depends on `ttmlir-golden` (not on PyPI). Install both from
the same tt-mlir branch, **golden first**:

```bash
pip install "ttmlir-golden @ git+https://github.com/tenstorrent/tt-mlir.git@<branch>#subdirectory=tools/golden"
pip install "ttmlir-chisel @ git+https://github.com/tenstorrent/tt-mlir.git@<branch>#subdirectory=tools/chisel"
```

## Runtime requirements

In addition to `ttmlir-golden`, this package imports:

- `ttmlir.ir` from the tt-mlir Python bindings
- `_ttmlir_runtime` (a compiled extension module)

Neither is on PyPI and neither is listed in this package's `dependencies` —
both ship with the tt-mlir build and are bundled into tt-xla wheels. To use
`ttmlir-chisel` you must have a Python environment with `ttmlir` and
`_ttmlir_runtime` importable, e.g. a tt-xla venv.

## Use

```python
from chisel import bind, configure, get_report, session, ChiselContext
```
