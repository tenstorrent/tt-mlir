# PyKernel Guide

PyKernel is a Python interface for developing custom TTNN operations for Tenstorrent's AI accelerators. This guide explains how to use the PyKernel interface to implement your own TTNN operations.

## Introduction to PyKernel

PyKernel provides a Python-based framework to define hardware-specific kernels that can be used with the TTNN framework. It allows developers to implement custom operations by defining compute kernels, reader/writer kernels, and control logic in a high-level Python interface.

The PyKernel framework consists of:
- **PyKernelOp**: Base class that manages kernel selection, compilation, and execution
- **AST module**: Decorators and utilities for defining kernels
- **Types module**: Type definitions for PyKernel operations

### PyKernel Architecture
Foundationally, PyKernel is a compiler built on top of 3 core components, described below.

#### Python `ast` Frontend
The frontend of PyKernel is made to parse Python code and is enabled through using the `ast` (Abstract Syntax Tree) parser builtin to Python. By walking through the AST produced by this module, a MLIR module is created with the `ttkernel` dialect (among others such as `arith`, `memref`, `scf`). This MLIR module is then piped into the next step of the PyKernel compiler. For more information about the type of kernel code that can be parsed by the Frontend, refer to the [`ttkernel` Op spec](https://docs.tenstorrent.com/tt-mlir/autogen/md/Dialect/TTKernelOp.html).

#### Direct To Metal (D2M) Kernel Code Generation
Another component of the `tt-mlir` project that PyKernel is built on is the D2M compiler infrastructure. This infrastructure enables dynamic generation of kernels to performantly execute ML models and is leveraged by providing the custom MLIR module created by the PyKernel frontend. The compilation flow runs a series of transformations on the MLIR module and lowers to the `emitc` dialect to translate the module into `C++` code. This C++ code is the artifact that is consumed by the runtime to execute on Tenstorrent Hardware.

#### TTNN Generic Op
TTNN consists of Python bindings to precompiled kernels and operator factories that maintain API parity with PyTorch. The Generic Op extends this by operating directly on TTNN tensors and primitives but does not define its own factory or kernels. Instead, these must be supplied to the Generic Op to enable execution. PyKernel leverages this flexibility by injecting dynamically compiled C++ kernels into the Generic Op, allowing them to interface with TTNN data as if they were native “custom” ops. This mechanism serves as the integration layer that connects the compiler to TTNN.


## Prerequisites

Before using PyKernel, ensure your environment is set up with:

- TT-MLIR built and installed
- Python 3.11 or newer
- Required Python packages
- TTMLIR_ENABLE_RUNTIME and TTMLIR_ENABLE_PYKERNEL flags set during build

## Creating a Custom PyKernel Operation

To create a custom PyKernel operation, you need to:

1. Create a class that inherits from `PyKernelOp`
2. Implement the `define_core_ranges` method to specify the grid of cores for the operation
3. Define kernels using the `@compute_thread()`, `@reader_thread()`, or `@writer_thread()` decorators
4. Implement the `invoke` method to create and connect kernels
5. Define necessary circular buffers
6. Create a program descriptor that combines kernels and circular buffers

### Basic Structure

```python
from pykernel.kernel_ast import *
from pykernel.op import PyKernelOp
from pykernel.kernel_types import *

import ttnn
import torch

class MyCustomOp(PyKernelOp):
    # Define Core Grid
    def define_core_ranges(self, tensors, options):
        # Your logic to determine the core ranges
        core_1 = ttnn.CoreCoord(0, 0)
        core_2 = ttnn.CoreCoord(1, 1)
        return ttnn.CoreRangeSet([ttnn.CoreRange(core_1, core_2)])

    # Define compute kernel with appropriate decorator
    @compute_thread()
    def my_compute_kernel(cb_in: CircularBuffer, cb_out: CircularBuffer,
                         per_core_block_cnt: CompileTimeValue,
                         per_core_block_dim: CompileTimeValue):
        # Kernel processing code here
        return

    # Define reader kernel
    @reader_thread()
    def reader_kernel(cb_in: CircularBuffer, cb_out: CircularBuffer,
                     src_addr, num_tiles, start_id,
                     src_is_dram: CompileTimeValue):
        # Reader kernel code here
        return

    # Define writer kernel
    @writer_thread()
    def writer_kernel(cb_in: CircularBuffer, cb_out: CircularBuffer,
                     dst_addr, num_tiles, start_id,
                     dst_is_dram: CompileTimeValue):
        # Writer kernel code here
        return

    # The invoke method is the main entry point for kernel execution
    def invoke(self, in_tensor, out_tensor, **options):
        # Create circular buffers for input and output tensors
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
- `@compute_thread()` - For compute kernels that run on Tensix cores
- `@reader_thread()` - For reader kernels that transfer data from memory to circular buffers
- `@writer_thread()` - For writer kernels that transfer data from circular buffers to memory

These decorators handle the compilation of Python code into hardware-specific kernels. You can also use the older style decorators if needed:
- `@ttkernel_tensix_compile()` - Equivalent to `@compute_thread()`
- `@ttkernel_noc_compile()` - For both reader and writer kernels

## Runtime Arguments

In PyKernel, you can pass runtime arguments to your kernels to control their behavior on a per-core basis. There are two types of runtime arguments:

1.  **Single-Core Arguments (Common Runtime Arguments)**: These are scalar values (integers) that are broadcast to all cores in the grid. They are passed as `common_runtime_args` to the `create_kernel` method.

2.  **Multi-Core Arguments (Runtime Arguments)**: These are lists of lists of integers, where each inner list corresponds to a core in the grid. This allows you to provide different values for each core. They are passed as `runtime_args` to the `create_kernel` method.

### Single-Core Arguments

Single-core arguments are useful when all cores need the same value for a particular parameter. For example, `num_tiles_per_core` in the `VecAdd` example is a single-core argument because each core processes the same number of tiles.

### Multi-Core Arguments

Multi-core arguments are necessary when each core requires a unique value. A common use case is distributing work across cores, where each core needs a different `start_id` to process its portion of the data. In the `VecAdd` example, `start_id_multicore` is a multi-core argument.

### Default Core Range Behavior

If you do not override the `define_core_ranges` method in your `PyKernelOp` class, it will default to a single core at `(0, 0)`. This is suitable for single-core operations like the `EltwiseSFPU` demo, where the entire operation runs on a single core.

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
def invoke(self, in_tensor, out_tensor, **options):
    cb_in = self.create_cb(in_tensor, 0)  # buffer_index=0
    cb_out = self.create_cb(out_tensor, 1)  # buffer_index=1

    # Use cb_in and cb_out in kernel creation
    # ...

    return self.create_program(kernels, [cb_in, cb_out])
```

The `create_cb` method handles the creation of the necessary format descriptors and buffer descriptors based on the tensor properties:

### Kernel Decorator Options

The kernel decorators (`@compute_thread`, `@reader_thread`, and `@writer_thread`) accept two optional boolean arguments:

-   `verbose`: When set to `True`, the PyKernel compiler will print the generated MLIR and the Python AST (Abstract Syntax Tree) during compilation. This is useful for debugging.
-   `optimize`: When set to `True`, the PyKernel compiler will run an optimization pipeline on the generated MLIR before converting it to C++. This can improve the performance of your kernel.

## Example: Vector Add Operation

The `VecAdd` operation adds two tensors element-wise. Let's examine a complete implementation based on the demo in `test/pykernel/demo/vecadd_multicore_demo.py`:

### 1. Define the Operation Class

```python
from pykernel.kernel_ast import *
from pykernel.op import PyKernelOp
from pykernel.kernel_types import *

import ttnn
import torch

class VecAddMulticorePyKernelOp(PyKernelOp):
    # Kernel implementations will go here
```

### 2. Define Core Ranges

The `define_core_ranges` method specifies the grid of cores that the operation will run on.

```python
def define_core_ranges(self, tensors, options):
    core_0 = ttnn.CoreCoord(0, 0)
    if self.max_core_ranges is None:
        core_1 = ttnn.CoreCoord(1, 1)
    else:
        core_1 = self.max_core_ranges
    return ttnn.CoreRangeSet([ttnn.CoreRange(core_0, core_1)])
```

### 3. Define the Compute Kernel

```python
@compute_thread()
def add_multicore(
    cb_in0: CircularBuffer,
    cb_in1: CircularBuffer,
    cb_out: CircularBuffer,
    num_tiles,
    start_tile_id,
):
    binary_op_init_common(cb_in0, cb_in1, cb_out)
    add_tiles_init(cb_in0, cb_in1)

    end_tile_id = start_tile_id + num_tiles
    dst_reg = 0

    for i in range(start_tile_id, end_tile_id, 1):
        cb_wait_front(cb_in0, 1)
        cb_wait_front(cb_in1, 1)
        tile_regs_acquire()
        add_tiles(cb_in0, cb_in1, 0, 0, dst_reg)
        tile_regs_commit()

        cb_reserve_back(cb_out, 1)
        tile_regs_wait()
        pack_tile(dst_reg, cb_out, 0)
        tile_regs_release()

        cb_push_back(cb_out, 1)
        cb_pop_front(cb_in0, 1)
        cb_pop_front(cb_in1, 1)
        tile_regs_release()
    return
```

### 4. Define Writer Kernel

```python
@writer_thread()
def writer_multicore(
    cb_out: CircularBuffer,
    dst_addr,
    num_tiles,
    start_id,
    dst_is_dram: CompileTimeValue,
):
    onetile = 1
    tile_bytes = get_tile_size(cb_out)
    dataformat = get_dataformat(cb_out)

    s0 = get_interleaved_addr_gen_fast(
        dst_is_dram, dst_addr, tile_bytes, dataformat
    )

    end_id = start_id + num_tiles
    for i in range(start_id, end_id, onetile):
        cb_wait_front(cb_out, onetile)
        l1_read_addr = get_read_ptr(cb_out)
        noc_async_write_tile(i, s0, l1_read_addr)
        noc_async_write_barrier()
        cb_pop_front(cb_out, onetile)
    return
```

### 5. Define Reader Kernel

```python
@reader_thread()
def reader_binary_interleaved(
    cb_in0: CircularBuffer,
    cb_in1: CircularBuffer,
    src_addr0,
    src_addr1,
    num_tiles,
    start_id,
    src0_is_dram: CompileTimeValue,
    src1_is_dram: CompileTimeValue,
):
    onetile = 1
    tile_bytes0 = get_tile_size(cb_in0)
    dataformat0 = get_dataformat(cb_in0)

    s0 = get_interleaved_addr_gen_fast(
        src0_is_dram, src_addr0, tile_bytes0, dataformat0
    )

    tile_bytes1 = get_tile_size(cb_in1)
    dataformat1 = get_dataformat(cb_in1)

    s1 = get_interleaved_addr_gen_fast(
        src1_is_dram, src_addr1, tile_bytes1, dataformat1
    )

    end_id = start_id + num_tiles
    for i in range(start_id, end_id, onetile):
        cb_reserve_back(cb_in0, onetile)
        cb_reserve_back(cb_in1, onetile)

        src0_write_addr = get_write_ptr(cb_in0)
        src1_write_addr = get_write_ptr(cb_in1)

        noc_async_read_tile(i, s0, src0_write_addr)
        noc_async_read_tile(i, s1, src1_write_addr)

        noc_async_read_barrier()
        cb_push_back(cb_in0, onetile)
        cb_push_back(cb_in1, onetile)
    return
```

### 6. Implement the Invoke Method

The `invoke` method is the critical part that connects the kernels together and creates the program descriptor:

```python
def invoke(self, a_tensor, b_tensor, out_tensor):
    # Create circular buffers
    cb_in0 = self.create_cb(a_tensor, 0)
    cb_in1 = self.create_cb(b_tensor, 1)
    cb_out = self.create_cb(out_tensor, 2)

    # Set up parameters
    is_a_dram = a_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
    is_b_dram = b_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
    is_out_dram = out_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM

    num_tiles = ceil(max(map(lambda t: t.volume(), [a_tensor, b_tensor, out_tensor])) / 1024)
    num_cores = self.get_core_ranges().num_cores()
    num_tiles_per_core = int(num_tiles / num_cores)

    # Define the multicore runtime arguments
    start_id = 0
    start_id_multicore = []
    bb = self.get_core_ranges().bounding_box()
    for i in range(bb.start.x, bb.end.x + 1):
        start_id_multicore.append([])
        for j in range(bb.start.y, bb.end.y + 1):
            start_id_multicore[-1].append([start_id])
            start_id += 1

    # Create kernels with appropriate parameters
    kernels = [
        self.create_kernel(
            VecAddMulticorePyKernelOp.add_multicore,
            cb_in0,
            cb_in1,
            cb_out,
            num_tiles_per_core,
            start_id_multicore,
        ),
        self.create_kernel(
            VecAddMulticorePyKernelOp.writer_multicore,
            cb_out,
            out_tensor.buffer_address(),
            num_tiles_per_core,
            start_id_multicore,
            dst_is_dram=is_out_dram,
        ),
        self.create_kernel(
            VecAddMulticorePyKernelOp.reader_binary_interleaved,
            cb_in0,
            cb_in1,
            a_tensor.buffer_address(),
            b_tensor.buffer_address(),
            num_tiles_per_core,
            start_id_multicore,
            src0_is_dram=is_a_dram,
            src1_is_dram=is_b_dram,
        ),
    ]

    # Create and return the program descriptor
    return self.create_program(kernels, [cb_in0, cb_in1, cb_out])
```

## Running the VecAdd Demo

The `VecAdd` demo demonstrates adding two tensors element-wise. This can be run using the `pykernel-demo` target:

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
data2 = torch.rand(shape).to(torch.bfloat16)


# Configure memory
dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

# Create input tensors
a_tensor = ttnn.from_torch(
    data,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=dram_memory_config,
)

b_tensor = ttnn.from_torch(
    data2,
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

# Create the custom operation
vecadd_op = VecAddMulticorePyKernelOp()

# Execute the operation with the tensors and options
output = vecadd_op(a_tensor, b_tensor, output_tensor)

# Compare with the built-in add operation
golden = ttnn.add(a_tensor, b_tensor)

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
3. Instantiates the `VecAddMulticorePyKernelOp` class
4. Executes the operation by calling the op with tensors
5. Compares the result with the built-in TTNN implementation

## Comparison with Native TTNN Operations

PyKernel operations integrate seamlessly with native TTNN operations. As shown in the demo, you can compare your custom PyKernel operation with built-in TTNN operations:

```python
# Execute your custom PyKernel operation
output = vecadd_op(a_tensor, b_tensor, output_tensor)

# Execute the equivalent built-in TTNN operation
golden = ttnn.add(a_tensor, b_tensor)

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
