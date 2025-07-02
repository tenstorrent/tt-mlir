# Optimizer

Optimizer is the main component responsible for performance. It is a collection of passes with the two most important purposes being optimizing tensor memory layouts and selecting optimal operation configurations.

## Prerequisites

To use the optimizer:
- A physical Tenstorrent device must be present on the machine
- Build of `tt-mlir` must be with OpModel support enabled:
```bash
cmake -G Ninja -B build -DTTMLIR_ENABLE_OPMODEL=ON
```

## Basic Usage

Optimizer is disabled by default. To enable it, use the `enable-optimizer` option:
```bash
ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true" input.mlir
```

## Optimizer Options

The optimizer provides additional configuration options:

- **`enable-optimizer`** (default: `false`)
  - Enables the optimizer pass
  - Must be set to `true` to use any other optimizer options

- **`memory-layout-analysis-enabled`** (default: `true`)
  - Enables memory layout optimization
  - Shards tensors to maximize usage of fast L1 memory instead of DRAM

- **`max-legal-layouts`** (default: `64`)
  - Maximum number of different layouts to generate for each operation during analysis
  - Higher values may provide better results but increase compile time

### Example

```bash
# Enable optimizer with default settings
ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true max-legal-layouts=8" input.mlir
```
