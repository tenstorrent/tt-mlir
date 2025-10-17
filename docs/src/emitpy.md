# EmitPy

`EmitPy` is part of the tt-mlir compiler project. Its primary function is to translate MLIR IR from various dialects into human-readable, executable Python source code.

By representing Python laguage constructs within a dedicated `EmitPy` dialect, the project provides a structured pathway for lowering high-level computational graphs (e.g., from machine learning frameworks) into a familiar and flexible Python language, enabling rapid prototyping, debugging, and integration with Tenstorrent's TTNN open source library.

Current implementation enables support for MNIST and ResNet models.

## Prerequisites

* [Built ttmlir](./getting-started.md)

* Activated virtual environment:

  ```bash
  source env/activate
  ```

## Usage

### ttmlir-opt
```bash
# 1. Convert a model from TTIR dialect to EmitPy dialect using ttmlir-opt
# 2. Translate the resulting EmitPy dialect to Python code using ttmlir-translate
# 3. Pipe the generated Python code to a .py file
ttmlir-opt --ttir-to-emitpy-pipeline test/ttmlir/Dialect/EmitPy/ttir_to_emitpy_pipeline_sanity.mlir | \
ttmlir-translate --mlir-to-python > example.py
```
### builder

[Builder](./builder/ttir-builder.md) offers support for building EmitPy modules.
[ttrt](./ttrt.md) offers support for running EmitPy modules.
