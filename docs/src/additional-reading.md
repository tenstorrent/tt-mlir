# Additional Reading

This section contains pointers to reading material that may be useful for
understanding the project.

### MLIR

- https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html
- https://mlir.llvm.org/docs/Tutorials/Toy/
- https://www.jeremykun.com/2023/08/10/mlir-getting-started/
- https://arxiv.org/pdf/2002.11054
- https://ieeexplore.ieee.org/abstract/document/9370308

#### Dialects

- [affine dialect](https://mlir.llvm.org/docs/Dialects/Affine/)
  - Affine map is a really powerful primitive that can be used to describe most data movement patterns.
  - It can also be used to describe memory layouts.
- [linalg dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [tosa dialect](https://mlir.llvm.org/docs/Dialects/TOSA/)
- [tosa spec](https://www.mlplatform.org/tosa/tosa_spec.html)
- [memref dialect](https://mlir.llvm.org/docs/Dialects/MemRef/)
- [torch-mlir](https://github.com/llvm/torch-mlir)
- [onnx-mlir](https://github.com/onnx/onnx-mlir)
- [triton-mlir](https://triton-lang.org/main/dialects/dialects.html)

### Tablegen

- [docs](https://llvm.org/docs/TableGen/)
- [Passes in TableGen](https://mlir.llvm.org/docs/DeclarativeRewrites/)
- [Ops in TableGen (we use this)](https://mlir.llvm.org/docs/DefiningDialects/Operations/)

### LLVM Testing Framework Tools

- [Lit](https://llvm.org/docs/CommandGuide/lit.html)
- [Filecheck](https://llvm.org/docs/CommandGuide/FileCheck.html)


[Jax](https://github.com/google/jax) \
[Flatbuffer](https://github.com/google/flatbuffers) \
[Openxla Website](https://openxla.org/) \
[openxla](https://github.com/openxla/xla) \
[StableHLO](https://github.com/openxla/stablehlo)
