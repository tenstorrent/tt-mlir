# `ttnn.jit`

`ttnn.jit` is a tool that allows TTNN model developers to leverage the Direct-To-Metal (D2M) compiler Just-In-Time (JIT) compile select portions of their model.

## Table of Contents

- [Getting Started](#getting-started)
- [How to use ttnn.jit](#how-to-use-ttnnjit)
  - [JIT Flags](#jit-flags)
- [How It Works](#how-it-works)
  - [Level 1: Python Decorator](#level-1-python-decorator)
  - [Level 2: D2M Compilation Pipeline](#level-2-d2m-compilation-pipeline)
  - [Level 3: Runtime Execution](#level-3-runtime-execution)
  - [JIT Caching](#jit-caching)
- [Limitations & Constraints](#limitations--constraints)
- [Debugging FAQ](#debugging-faq)
  - [AssertionError: Function ___ not supported](#assertionerror-function-___-not-supported)
  - [Failed to run pass manager](#failed-to-run-pass-manager)

## Getting Started

### Quickstart
Wheel install coming soon!

### Building From Source
Build [tt-mlir](./getting-started.md) with the following flags:

```bash
-DTTMLIR_ENABLE_RUNTIME=ON -DTTMLIR_ENABLE_PYKERNEL=ON
```

For profiling purposes, add the following flag to enable Tracy:
```bash
-DTT_RUNTIME_ENABLE_PERF_TRACE=ON
```

After building, make sure to generate a system descriptor using [ttrt](./ttrt.md).
```bash
ttrt query --save-artifacts
export SYSTEM_DESC_PATH=`pwd`/ttrt-artifacts/system_desc.ttsys
```

## How to use ttnn.jit
Take any Python TTNN subgraph such as the cosh composite operation:
```Python
def cosh(input_tensor):
  e_pos_x = ttnn.exp(input_tensor)
  e_neg_x = ttnn.exp(ttnn.neg(input_tensor))
  nr_term = ttnn.add(e_pos_x, e_neg_x)
  return ttnn.multiply(nr_term, 0.5)

def model():
    # Create tensors as usual
    input_tensor = ...
    x = ttnn.square(input_tensor)
    y = cosh(x)     # will run the 5 TTNN ops in the cosh subgraph
    z = ttnn.log(y)
    return z
```

Simply decorate with `@ttnn_jit.jit()` to JIT compile through D2M. In this example, `cosh` will be compiled into a single fused kernel.
```Python
@ttnn_jit.jit()
def cosh(input_tensor):
  e_pos_x = ttnn.exp(input_tensor)
  e_neg_x = ttnn.exp(ttnn.neg(input_tensor))
  nr_term = ttnn.add(e_pos_x, e_neg_x)
  return ttnn.multiply(nr_term, 0.5)

def model():
    # Create tensors as usual
    input_tensor = ...
    x = ttnn.square(input_tensor)
    y = cosh(x)     # invoke the JIT'ed subgraph: will run one fused cosh kernel.
    z = ttnn.log(y)
    return z
```

### JIT Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `max_grid` | `tuple` | `(7,7)` | Maximum **sharding** grid size used for JIT compilation |
| `debug` | `bool` | `False` | Enable debug prints during compilation and execution |
| `compile_only` | `bool` | `False` | Only compile runtime without execution. The resulting flatbuffer will be dumped to `generated/jit` |

## Current Limitations
- Only select eltwise unary and binary operations.
- Only L1 block sharded and DRAM interleaved tensors.
- No control flow allowed.

See the current [test suite](../../test/ttnn-jit/) for what is guaranteed to be working.

## How It Works

The `ttnn.jit` decorator implements a three-step compilation and execution pipeline that transforms Python TTNN operations into optimized hardware kernels:

### Step 1: Python Decorator

When you decorate a function with `@ttnn_jit.jit()`, the decorator wraps it in a `JitFunction` object. On the first call:

- The Python source code is extracted and parsed into MLIR using Python's AST module.
- Each TTNN operation (eg: `ttnn.exp`, `ttnn.add`) is converted to its corresponding MLIR operation in the TTNN dialect.
- All operations are tagged with the `ttnn.hoist_generic_via_d2m` attirbute, marking them for D2M compilation

The output of the AST should be a valid MLIR module in the TTNN dialect. The previous `cosh` [example](#how-to-use-ttnnjit) will be parsed into:

```mlir
module {
  func.func @cosh(%arg0: tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout> {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.exp"(%arg0) {ttnn.hoist_generic_via_d2m} : (tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
    %2 = "ttnn.neg"(%arg0) {ttnn.hoist_generic_via_d2m} : (tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
    %3 = "ttnn.exp"(%2) {ttnn.hoist_generic_via_d2m} : (tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
    %4 = "ttnn.add"(%1, %3) <{dtype = #ttcore.supportedDataTypes<bf16>}> {ttnn.hoist_generic_via_d2m} : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
    %5 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, fill_value = 5.000000e-01 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout>
    %6 = "ttnn.multiply"(%4, %5) <{dtype = #ttcore.supportedDataTypes<bf16>}> {ttnn.hoist_generic_via_d2m} : (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>) -> tensor<32x32xbf16, #ttnn_layout>
    return %6 : tensor<32x32xbf16, #ttnn_layout>
  }
}
```

### Step 2: D2M Compilation Pipeline

The resulting MLIR module is then passed to the compiler:

- **TTNN → TTIR Conversion**: The `createConvertTTNNToTTIRPass()` lowers TTNN dialect ops to the TTIR.
- **D2M Compilation**: The `ttir-to-ttmetal-pipeline` runs with `ttnn-mode`
  - **Generates** custom kernels with techniques such as destination fusion and loop tiling.
  - Wraps the generated kernels in **`ttnn.generic`** operations that contains the necessary host side program setup.
- **Flatbuffer Serialization**: The compiled IR is serialized to a FlatBuffer binary format via `ttnnToFlatbuffer()`
  - This flatbuffer is returned to the decorator and cached as a `Binary` object.

Each `ttnn.generic` op requires a `ProgramDescriptor` that contains everything needed to construct a TT-Metal `Program`.
- Circular buffer and Semaphore config.
- Kernel source code and runtime + compile-time argument setup.

### Step 3: Runtime Execution

The decorator leverages with same MLIR runtime as [ttrt](./ttrt.md). For our purposes, it is essentially just TTNN with additional machinery to execute the serialized `ttnn.generic` operations that wrap the custom D2M-generated kernels.

As shown in [previously](#how-to-use-ttnnjit), interop with TTNN is seamless, allowing users to switch freely between JIT'ed and non-JIT'ed subgraphs of ttnn ops.

### JIT Caching

The first invocation of a JIT'ed subgraph will compile and cache the resulting flatbuffer in a `JitCache`. The cache uses tensor **metadata** (shape, dtype, memory config, etc.) as the key. The compiled flatbuffer wrapped in an MLIR runtime `Binary` object is the cache entry.

Each `JitFunction` maintains its own `JitCache`, so different JIT [configurations](#jit-flags) will have independent cache entries.

Constructing a `ProgramDescriptor` from a flatbuffer at runtime is expensive. To mitigate this, `ProgramDescriptor` instances are cached in a `ProgramDescCache` owned by the flatbuffer `Binary` object. The same cache key is also stored in the `ProgramDescriptor` as a `custom_program_hash` and passed to the TTNN runtime, allowing the `ttnn.generic` to reuse for its `ProgramCache`.

See [test_program_cache.py](../../test/ttnn-jit/test_program_cache.py) for a detailed example demonstrating cache hit/miss behavior.

## Debugging FAQ
For debugging purposes, always build with `-DCMAKE_BUILD_TYPE=Debug` and decorate with `debug=True` to see IR outputs after each step.

### `AssertionError: Function ___ not supported`
This indicates the decorated TTNN op is not supported yet in the TTNN dialect. Or you spelt it wrong, eg: `ttnn.mul` is not supported but `ttnn.multiply` is.

To start, check whether the desired TTNN op is supported in the [tablegen](../../include/ttmlir/Dialect/TTNN/IR/TTNNOps.td). If not, please file an issue.

Note: as mentioned in [Current Limitations](#current-limitations), only *select* unary and binary operations are supported.

### `Failed to run pass manager`
This means the [compilation pipeline](#step-2-d2m-compilation-pipeline) failed at a certain stage. The easiest way to debug is to copy the IR output from the AST traversal, and manaully run each individual pipeline:

```bash
ttmlir-opt --convert-ttnn-to-ttir *.mlir

ttmlir-opt --mlir-print-ir-after-all --ttir-to-ttmetal-pipeline="system-desc-path=${SYSTEM_DESC_PATH} ttnn-mode=true" *.mlir

ttmlir-translate --ttnn-to-flatbuffer *.mlir
```

For MLIR runtime and debug output:
```bash
export TTMLIR_RUNTIME_LOGGER_LEVEL=Trace
export TTRT_LOGGER_LEVEL=Debug
```
