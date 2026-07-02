# TT-MLIR Documentation

TT-MLIR is Tenstorrent's MLIR-based compiler infrastructure for AI hardware.

```{toctree}
:caption: Introduction
:maxdepth: 1

overview
```

```{toctree}
:caption: User Guide
:maxdepth: 2

getting-started
docker-notes
macos-ubuntu-vm
testing
lit-testing
emitc-testing
```

```{toctree}
:caption: Tools
:maxdepth: 2

tools
ttmlir-opt
ttmlir-translate
ttrt
emitpy
tt-alchemist
ttnn-standalone
tt-explorer/tt-explorer
tt-explorer/ui
tt-explorer/cli
tt-explorer/usage-api
tt-explorer/architecture
builder/ttir-builder
builder/adding-a-ttir-op
builder/stablehlo-builder
builder/testing
```

```{toctree}
:caption: Internals
:maxdepth: 2

optimizer
pykernel
ttnn-jit
ttnn-bug-repros
python-bindings
flatbuffers
llvm_dependency_update
```

```{toctree}
:caption: Code Documentation
:maxdepth: 2

project-structure
dialects-overview
guidelines
coding-guidelines
ttnn-dialect-guidelines
adding-an-op
ttnn-op-constraints
decomposing-an-op-in-ttir
docs
specs/specs
specs/runtime-stitching
specs/tensor-layout
specs/device
specs/ttnn-optimizer
```

```{toctree}
:caption: Project
:maxdepth: 1

ci
additional-reading
code-of-conduct
```
