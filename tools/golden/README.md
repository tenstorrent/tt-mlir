# ttmlir-golden

Golden reference implementations for TT-MLIR ops. Provides PyTorch-based
golden functions used to validate compiler output against expected results.

## Install

From a published branch of tt-mlir:

```bash
pip install "ttmlir-golden @ git+https://github.com/tenstorrent/tt-mlir.git@<branch>#subdirectory=tools/golden"
```

## Runtime requirement

This package imports `ttmlir.dialects`, `ttmlir.ir`, and `ttmlir.passes` from
the tt-mlir Python bindings. Those bindings are **not** on PyPI and are not
listed in this package's `dependencies` — they ship with the tt-mlir build (and
are bundled into tt-xla wheels). To use `ttmlir-golden` you must already have a
Python environment with `ttmlir` importable, e.g. a tt-xla venv.

## Use

```python
from golden import (
    GoldenMapTensor,
    get_golden_function,
    GOLDEN_MAPPINGS,
    get_pcc,
    get_atol_rtol_pcc,
)
```
