# PyKernel Guide

PyKernel is a Python interface for developing custom TTNN operations for Tenstorrent's AI accelerators. This guide explains how to use the PyKernel interface to implement your own TTNN operations.

## Introduction to PyKernel

PyKernel provides a Python-based framework to define hardware-specific kernels that can be used with the TTNN (Tenstorrent Neural Network) framework. It allows developers to implement custom operations by defining compute kernels, reader/writer kernels, and control logic in a high-level Python interface.

The PyKernel framework consists of:
- **PyKernelOp**: Base class that manages kernel selection, compilation, and execution
- **AST Module**: Decorators and utilities for defining kernels
- **Types Module**: Type definitions for PyKernel operations

## Prerequisites

Before using PyKernel, ensure your environment is set up with:

- TT-MLIR built and installed
- Python 3.10 or newer
- Required Python packages

## Creating a Custom PyKernel Operation

To create a custom PyKernel operation, you need to:

1. Create a class that inherits from `PyKernelOp`
2. Define kernels using the `@ttkernel_*` decorators
3. Implement kernel selection logic
4. Define necessary circular buffers and semaphores
5. Implement methods to generate compile-time and runtime arguments

### Basic Structure

```python
from pykernel.ast import *
from pykernel.op import PyKernelOp
from pykernel.types import *

class MyCustomOp(PyKernelOp):
    # Define kernels with appropriate decorators
    @staticmethod
    @ttkernel_tensix_compile()
    def my_compute_kernel(cb_in: CircularBuffer, cb_out: CircularBuffer, ct_args=[]):
        # Kernel code here
        return

    # Define methods for compile-time and runtime args
    def my_compute_kernel_CT_ARGS(self, tensors, options):
        return [...]

    def my_compute_kernel_RT_ARGS(self, tensors, options):
        return [...]

    # Optional defines for the kernel
    def my_compute_kernel_DEFINES(self, tensors, options):
        return [...]

    # Select kernels to use
    def select_kernels(self, tensors, options):
        return [
            (MyCustomOp.my_compute_kernel, ttnn.ComputeConfigDescriptor),
            # Additional kernels...
        ]

    # Define circular buffers
    def define_cbs(self, tensors, options):
        # CB definitions
        return [...]

    # Define semaphores if needed
    def define_semaphores(self, tensors, options):
        return [...]
```

## Kernel Types

PyKernel supports different types of kernels:

1. **Compute Kernels**: Process data on the compute units (e.g., SFPU - Scalar Floating-Point Unit)
2. **Reader Kernels**: Transfer data from memory to circular buffers
3. **Writer Kernels**: Transfer data from circular buffers to memory

Each kernel type has a specific decorator:
- `@ttkernel_tensix_compile()` - For compute kernels
- `@ttkernel_noc_compile()` - For reader/writer kernels

## Circular Buffers

Circular buffers are used to transfer data between kernels and memory. They are defined using:

```python
# Format descriptors
cb_format = ttnn.CBFormatDescriptor(
    buffer_index=buffer_id,
    data_format=data_format,
    page_size=page_size
)

# Buffer descriptor
cb_descriptor = ttnn.CBDescriptor(
    total_size=total_size,
    core_ranges=core_ranges,
    format_descriptors=[cb_format]
)
```

## Example: EltwiseSFPU Operation

The EltwiseSFPU operation applies an exponential function element-wise to an input tensor. Let's break down how it works:

### 1. Define the Operation Class

```python
class EltwiseSFPUPyKernelOp(PyKernelOp):
    # Kernel implementations will go here
```

### 2. Define the Compute Kernel

```python
@staticmethod
@ttkernel_tensix_compile()
def eltwise_sfpu(cb_in: CircularBuffer, cb_out: CircularBuffer, ct_args=[]):
    per_core_block_cnt = ct_args[0]
    per_core_block_dim = ct_args[1]

    # Initialize the operation
    unary_op_init_common(cb_in, cb_out)

    # Process tiles
    for i in range(0, per_core_block_cnt, 1):
        cb_reserve_back(cb_out, per_core_block_dim)
        for j in range(0, per_core_block_dim, 1):
            tile_regs_acquire()
            cb_wait_front(cb_in, 1)

            # Copy input tile to register
            copy_tile(cb_in, 0, 0)

            # Apply exponential function
            exp_tile_init()
            exp_tile(0)

            # Commit results
            tile_regs_commit()
            tile_regs_wait()
            pack_tile(0, cb_out, 0)

            cb_pop_front(cb_in, 1)
            tile_regs_release()

        cb_push_back(cb_out, per_core_block_dim)
    return
```

### 3. Define Reader and Writer Kernels

Reader kernels transfer data from memory to circular buffers:

```python
@staticmethod
@ttkernel_noc_compile()
def reader_unary_interleaved(cb_in: CircularBuffer, cb_out: CircularBuffer, rt_args, ct_args=[]):
    src_addr: int = rt_args[0]
    num_tiles = rt_args[1]
    start_id = rt_args[2]

    # Read data from memory to circular buffer
    # ...
```

Writer kernels transfer data from circular buffers to memory:

```python
@staticmethod
@ttkernel_noc_compile()
def writer_unary_interleaved(cb_in: CircularBuffer, cb_out: CircularBuffer, rt_args, ct_args=[]):
    dst_addr: int = rt_args[0]
    num_tiles = rt_args[1]
    start_id = rt_args[2]

    # Write data from circular buffer to memory
    # ...
```

### 4. Define Arguments and Configuration

```python
# Compile-time arguments
def eltwise_sfpu_CT_ARGS(self, tensors, options):
    return [
        options["num_tiles"],  # per_core_block_cnt
        1,  # per_core_block_dim
    ]

# Runtime arguments
def reader_unary_interleaved_RT_ARGS(self, tensors, options):
    return [
        tensors[0].buffer_address(),
        options["num_tiles"],
        0,  # start_id
    ]

# Defines for the kernel
def eltwise_sfpu_DEFINES(self, tensors, options):
    return [
        ("SFPU_OP_EXP_INCLUDE", "1"),
        ("SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"),
    ]
```

### 5. Select Kernels

```python
def select_kernels(self, tensors, options):
    return [
        (EltwiseSFPUPyKernelOp.eltwise_sfpu, ttnn.ComputeConfigDescriptor),
        (EltwiseSFPUPyKernelOp.writer_unary_interleaved, ttnn.WriterConfigDescriptor),
        (EltwiseSFPUPyKernelOp.reader_unary_interleaved, ttnn.ReaderConfigDescriptor),
    ]
```

### 6. Define Circular Buffers

```python
def define_cbs(self, tensors, options):
    # Parse options
    input_cb_data_format = options["cb_data_format"]
    cb_total_size = options["cb_total_size"]
    cb_page_size = options["cb_page_size"]
    in_cb = options["in_cb"]
    out_cb = options["out_cb"]
    core_ranges = options["core_ranges"]

    # Create format descriptors
    in_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=in_cb,
        data_format=input_cb_data_format,
        page_size=cb_page_size,
    )
    out_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=out_cb,
        data_format=input_cb_data_format,
        page_size=cb_page_size,
    )

    # Store formats for later use
    self._cb_formats = [in_cb_format, out_cb_format]

    # Create buffer descriptors
    in_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_ranges,
        format_descriptors=[in_cb_format],
    )
    out_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_ranges,
        format_descriptors=[out_cb_format],
    )

    return [in_cb_descriptor, out_cb_descriptor]
```

## Running the EltwiseSFPU Demo

The EltwiseSFPU demo demonstrates applying an exponential function element-wise to a tensor. This can be run using the pykernel-demo target to install the required packages:

```bash
source env/activate
# Ensure the TTMLIR_ENABLE_RUNTIME and TTMLIR_ENABLE_PYKERNEL flags are set, otherwise the function will fail.
cmake --build build -- pykernel-demo
```

### Demo Breakdown

The demo:
1. Opens a device
2. Creates input and output tensors
3. Instantiates the `EltwiseSFPUPyKernelOp` with appropriate options
4. Executes the operation on the tensors
5. Compares the result with the golden reference implementation

## Comparison with Native TTNN Operations

PyKernel operations integrate seamlessly with native TTNN operations. In the demo, we compare the results with `ttnn.exp()`:

```python
output = eltwise_sfpu_op(*io_tensors, **eltwise_exp_op_options)
golden = ttnn.exp(input_tensor)

torch_golden = ttnn.to_torch(golden)
torch_output = ttnn.to_torch(output)

matching = torch.allclose(torch_golden, torch_output)
print(f"Tensors are matching: {matching}")
assert matching
```

## Building and Testing

To build and test PyKernel:
```bash
source env/activate
# Minimum flags needed to configure
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-17 -DCMAKE_CXX_COMPILER=clang++-17 -DTTMLIR_ENABLE_RUNTIME=ON -DTTMLIR_ENABLE_PYKERNEL=ON
cmake --build build
```

## Best Practices

When developing with PyKernel:

1. **Separate concerns**: Keep compute, reader, and writer kernels separate
2. **Reuse kernels**: Create reusable kernels for common operations
3. **Cache results**: PyKernelOp caches compiled kernels for performance
4. **Document dependencies**: Ensure all required defines and arguments are well-documented
5. **Test thoroughly**: Compare results with reference implementations

## Summary

PyKernel provides a flexible and powerful way to implement custom operations for Tenstorrent hardware. By following the pattern outlined in this guide, you can create your own operations that integrate seamlessly with the TTNN framework.

The PyKernelOp base class handles the complex tasks of kernel selection, compilation, and caching, while you focus on implementing the operation-specific logic.
