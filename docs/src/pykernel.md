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
- TTMLIR_ENABLE_RUNTIME and TTMLIR_ENABLE_PYKERNEL flags set during build

## Creating a Custom PyKernel Operation

To create a custom PyKernel operation, you need to:

1. Create a class that inherits from `PyKernelOp`
2. Define kernels using the `@compute_thread()`, `@reader_thread()`, or `@writer_thread()` decorators
3. Implement the `invoke` method to create and connect kernels
4. Define necessary circular buffers
5. Create a program descriptor that combines kernels and circular buffers

### Basic Structure

```python
from pykernel.ast import *
from pykernel.op import PyKernelOp
from pykernel.types import *

import ttnn
import torch

class MyCustomOp(PyKernelOp):
    # Define compute kernel with appropriate decorator
    @compute_thread()
    def my_compute_kernel(cb_in: CircularBuffer, cb_out: CircularBuffer,
                         per_core_block_cnt: CompiledValue,
                         per_core_block_dim: CompiledValue):
        # Initialize the operation
        unary_op_init_common(cb_in, cb_out)

        # Process data in blocks
        for i in range(0, per_core_block_cnt, 1):
            cb_reserve_back(cb_out, per_core_block_dim)
            for j in range(0, per_core_block_dim, 1):
                # Kernel processing code here
                tile_regs_acquire()
                cb_wait_front(cb_in, 1)

                # Your custom processing logic
                # ...

                cb_pop_front(cb_in, 1)
                tile_regs_release()

            cb_push_back(cb_out, per_core_block_dim)
        return

    # Define reader kernel
    @reader_thread()
    def reader_kernel(cb_in: CircularBuffer, cb_out: CircularBuffer,
                     src_addr, num_tiles, start_id,
                     src_is_dram: CompiledValue):
        # Reader kernel code here
        return

    # Define writer kernel
    @writer_thread()
    def writer_kernel(cb_in: CircularBuffer, cb_out: CircularBuffer,
                     dst_addr, num_tiles, start_id,
                     dst_is_dram: CompiledValue):
        # Writer kernel code here
        return

    # The invoke method is the main entry point for kernel execution
    def invoke(self, *tensors, **options):
        # Create circular buffers for input and output tensors
        in_tensor, out_tensor = tensors
        cb_in = self.create_cb(in_tensor, 0)
        cb_out = self.create_cb(out_tensor, 1)

        # Prepare parameters for kernels
        start_id = 0
        is_dram = in_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
        num_tiles = options["num_tiles"]

        # Create kernels with appropriate parameters
        kernels = [
            self.create_kernel(
                MyCustomOp.my_compute_kernel,
                cb_in, cb_out,
                per_core_block_cnt=num_tiles,
                per_core_block_dim=1
            ),
            self.create_kernel(
                MyCustomOp.writer_kernel,
                cb_in, cb_out,
                out_tensor.buffer_address(),
                num_tiles, start_id,
                dst_is_dram=is_dram
            ),
            self.create_kernel(
                MyCustomOp.reader_kernel,
                cb_in, cb_out,
                in_tensor.buffer_address(),
                num_tiles, start_id,
                src_is_dram=is_dram
            )
        ]

        # Create and return the program descriptor
        return self.create_program(kernels, [cb_in, cb_out])
```

## Kernel Types

PyKernel supports different types of kernels:

1. **Compute Kernels**: Process data on the compute units (e.g., SFPU - Scalar Floating-Point Unit)
2. **Reader Kernels**: Transfer data from memory to circular buffers
3. **Writer Kernels**: Transfer data from circular buffers to memory

Each kernel type has a specific decorator:
- `@compute_thread()` - For compute kernels that run on TenSix cores
- `@reader_thread()` - For reader kernels that transfer data from memory to circular buffers
- `@writer_thread()` - For writer kernels that transfer data from circular buffers to memory

These decorators handle the compilation of Python code into hardware-specific kernels. You can also use the older style decorators if needed:
- `@ttkernel_tensix_compile()` - Equivalent to `@compute_thread()`
- `@ttkernel_noc_compile()` - For both reader and writer kernels

## Circular Buffers

Circular buffers are used to transfer data between kernels and memory. In the PyKernel framework, there are two aspects of circular buffers:

1. **CircularBuffer class**: Used in kernel definitions to represent a circular buffer
2. **CB Descriptors**: Used at runtime to configure the actual hardware circular buffers

### CircularBuffer Class

The `CircularBuffer` class is defined in `pykernel.types` and is used in kernel definitions:

```python
class CircularBuffer:
    def __init__(self, cb_id, tensor_shape=(8, 128, 128), dtype="Float32"):
        self.cb_id = cb_id
        self.tensor_shape = tensor_shape
        self.tile_shape = 32  # default to 32x32 tile shape
        self.tilized_shape = self.get_tilized_memref_shape()
        self.dtype = dtype
```

### Creating Circular Buffers in the Invoke Method

In your custom operation's `invoke` method, you can create circular buffers using the `create_cb` helper method from the `PyKernelOp` base class:

```python
def invoke(self, *tensors, **options):
    in_tensor, out_tensor = tensors
    cb_in = self.create_cb(in_tensor, 0)  # buffer_index=0
    cb_out = self.create_cb(out_tensor, 1)  # buffer_index=1

    # Use cb_in and cb_out in kernel creation
    # ...

    return self.create_program(kernels, [cb_in, cb_out])
```

The `create_cb` method handles the creation of the necessary format descriptors and buffer descriptors based on the tensor properties:

## Example: EltwiseSFPU Operation

The EltwiseSFPU operation applies an exponential function element-wise to an input tensor. Let's examine a complete implementation based on the demo in `test/pykernel/demo/eltwise_sfpu_demo.py`:

### 1. Define the Operation Class

```python
from pykernel.ast import *
from pykernel.op import PyKernelOp
from pykernel.types import *

import ttnn
import torch

class EltwiseSFPUPyKernelOp(PyKernelOp):
    # Kernel implementations will go here
```

### 2. Define the Compute Kernel

```python
@compute_thread()
def eltwise_sfpu(
    cb_in: CircularBuffer,
    cb_out: CircularBuffer,
    per_core_block_cnt: CompiledValue,
    per_core_block_dim: CompiledValue,
):
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

### 3. Define Writer Kernel

```python
@writer_thread()
def writer_unary_interleaved(
    cb_in: CircularBuffer,
    cb_out: CircularBuffer,
    dst_addr,
    num_tiles,
    start_id,
    dst_is_dram: CompiledValue,
):
    onetile = 1
    tile_bytes = get_tile_size(cb_out)
    dataformat = get_dataformat(cb_out)

    s0 = get_interleaved_addr_gen_fast(
        dst_is_dram, dst_addr, tile_bytes, dataformat
    )

    end_id = start_id + num_tiles
    ii: int = start_id
    for i in range(start_id, end_id, onetile):
        cb_wait_front(cb_out, onetile)
        l1_read_addr = get_read_ptr(cb_out)
        noc_async_write_tile(ii, s0, l1_read_addr)
        noc_async_write_barrier()
        cb_pop_front(cb_out, onetile)
        ii += onetile
    return
```

### 4. Define Reader Kernel

```python
@reader_thread()
def reader_unary_interleaved(
    cb_in: CircularBuffer,
    cb_out: CircularBuffer,
    src_addr,
    num_tiles,
    start_id,
    src_is_dram: CompiledValue,
):
    onetile = 1
    tile_bytes = get_tile_size(cb_in)
    dataformat = get_dataformat(cb_in)

    s0 = get_interleaved_addr_gen_fast(
        src_is_dram, src_addr, tile_bytes, dataformat
    )

    end_id = start_id + num_tiles
    ii: int = start_id
    for i in range(start_id, end_id, onetile):
        cb_reserve_back(cb_in, onetile)
        l1_write_addr = get_write_ptr(cb_in)
        noc_async_read_tile(ii, s0, l1_write_addr)
        noc_async_read_barrier()
        cb_push_back(cb_in, onetile)
        ii += onetile
    return
```

### 5. Implement the Invoke Method

The `invoke` method is the critical part that connects the kernels together and creates the program descriptor:

```python
def invoke(self, *tensors, **options):
    # Extract input and output tensors
    in_tensor, out_tensor = tensors

    # Create circular buffers
    cb_in = self.create_cb(in_tensor, 0)
    cb_out = self.create_cb(out_tensor, 1)

    # Set up parameters
    start_id = 0
    is_dram_input = in_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
    num_tiles = options["num_tiles"]

    # Create kernels with appropriate parameters
    kernels = [
        self.create_kernel(
            EltwiseSFPUPyKernelOp.eltwise_sfpu,
            cb_in,
            cb_out,
            per_core_block_cnt=num_tiles,
            per_core_block_dim=1,
        ),
        self.create_kernel(
            EltwiseSFPUPyKernelOp.writer_unary_interleaved,
            cb_in,
            cb_out,
            out_tensor.buffer_address(),
            num_tiles,
            start_id,
            dst_is_dram=is_dram_input,
        ),
        self.create_kernel(
            EltwiseSFPUPyKernelOp.reader_unary_interleaved,
            cb_in,
            cb_out,
            in_tensor.buffer_address(),
            num_tiles,
            start_id,
            src_is_dram=is_dram_input,
        ),
    ]

    # Create and return the program descriptor
    return self.create_program(kernels, [cb_in, cb_out])
```

## Running the EltwiseSFPU Demo

The EltwiseSFPU demo demonstrates applying an exponential function element-wise to a tensor. This can be run using the pykernel-demo target:

```bash
source env/activate
# Ensure the TTMLIR_ENABLE_RUNTIME and TTMLIR_ENABLE_PYKERNEL flags are set during build
cmake --build build -- pykernel-demo
```

### Demo Breakdown

Let's examine how to use the PyKernel operation in practice:

```python
# Open a device
device = ttnn.open_device(device_id=0)

# Define tensor shapes and data
num_tiles = 4
shape = [1, num_tiles, 32, 32]
data = torch.rand(shape).to(torch.bfloat16)

# Configure memory
dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

# Create input tensor
input_tensor = ttnn.from_torch(
    data,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=dram_memory_config,
)

# Create output tensor
output_tensor = ttnn.allocate_tensor_on_device(
    ttnn.Shape(shape),
    ttnn.bfloat16,
    ttnn.TILE_LAYOUT,
    device,
    dram_memory_config,
)

# Prepare tensors for the operation
io_tensors = [input_tensor, output_tensor]

# Create the custom operation
eltwise_exp_op = EltwiseSFPUPyKernelOp()

# Execute the operation with the tensors and options
output = eltwise_exp_op(*io_tensors, num_tiles=num_tiles)

# Compare with the built-in exponential operation
golden = ttnn.exp(input_tensor)

# Convert to torch tensors for comparison
torch_golden = ttnn.to_torch(golden)
torch_output = ttnn.to_torch(output)

# Verify results
matching = torch.allclose(torch_golden, torch_output)
print(f"Tensors are matching: {matching}")
assert matching
```

This demo shows the complete workflow:
1. Opens a device
2. Creates input and output tensors with appropriate memory configuration
3. Instantiates the `EltwiseSFPUPyKernelOp` class
4. Executes the operation by calling the op with tensors and options
5. Compares the result with the built-in TTNN implementation

## Comparison with Native TTNN Operations

PyKernel operations integrate seamlessly with native TTNN operations. As shown in the demo, you can compare your custom PyKernel operation with built-in TTNN operations:

```python
# Execute your custom PyKernel operation
output = eltwise_exp_op(*io_tensors, num_tiles=num_tiles)

# Execute the equivalent built-in TTNN operation
golden = ttnn.exp(input_tensor)

# Convert both to torch tensors for comparison
torch_golden = ttnn.to_torch(golden)
torch_output = ttnn.to_torch(output)

# Verify the results match
matching = torch.allclose(torch_golden, torch_output)
print(f"Tensors are matching: {matching}")
assert matching
```

This approach allows you to:
1. Validate your custom operation against known implementations
2. Benchmark performance differences between custom and built-in operations
3. Extend the TTNN framework with operations not available in the standard library

## Building and Testing

To build and test PyKernel, you need to enable both the runtime and PyKernel components:

```bash
source env/activate

# Configure with PyKernel enabled
cmake -G Ninja -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-17 \
    -DCMAKE_CXX_COMPILER=clang++-17 \
    -DTTMLIR_ENABLE_RUNTIME=ON \
    -DTTMLIR_ENABLE_PYKERNEL=ON

# Build the project
cmake --build build

# Run the PyKernel demo
cmake --build build -- pykernel-demo
```

The `TTMLIR_ENABLE_RUNTIME` and `TTMLIR_ENABLE_PYKERNEL` flags are essential for PyKernel functionality. Without these flags, the PyKernel components will not be built.

## Best Practices

When developing with PyKernel, follow these best practices:

1. **Separate concerns**: Keep compute, reader, and writer kernels separate for better maintainability and reusability

2. **Use appropriate decorators**: Apply the correct decorator for each kernel type:
   - `@compute_thread()` for compute kernels
   - `@reader_thread()` for reader kernels
   - `@writer_thread()` for writer kernels

3. **Implement the invoke method properly**: The `invoke` method is critical as it connects all components:
   - Create circular buffers with appropriate parameters
   - Set up kernel parameters correctly
   - Create kernels with the right arguments
   - Return a program descriptor that includes all kernels and circular buffers

4. **Handle memory configurations**: Be aware of memory types (DRAM vs L1) when creating kernels

5. **Reuse kernels**: Create reusable kernels for common operations to avoid code duplication

6. **Leverage caching**: PyKernelOp automatically caches compiled kernels for performance

7. **Test thoroughly**: Always compare results with reference implementations or built-in TTNN operations

8. **Document parameters**: Clearly document the expected parameters for your PyKernel operation

## Summary

PyKernel provides a flexible and powerful way to implement custom operations for Tenstorrent hardware. By following the pattern outlined in this guide, you can create your own operations that integrate seamlessly with the TTNN framework.

Key components of the PyKernel framework:

1. **PyKernelOp base class**: Handles kernel management, compilation, and caching
2. **Kernel decorators**: `@compute_thread()`, `@reader_thread()`, and `@writer_thread()`
3. **CircularBuffer class**: Represents circular buffers in kernel definitions
4. **invoke method**: The critical implementation that connects kernels and creates the program

The workflow for creating a custom PyKernel operation is:

1. Create a class that inherits from `PyKernelOp`
2. Define compute, reader, and writer kernels with appropriate decorators
3. Implement the `invoke` method to create circular buffers and connect kernels
4. Use the operation by instantiating your class and calling it with tensors and options

With PyKernel, you can extend the TTNN framework with custom operations that leverage the full power of Tenstorrent hardware while maintaining a clean, high-level Python interface.
