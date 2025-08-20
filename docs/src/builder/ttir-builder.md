# `ttir-builder`

`ttir-builder` is a tool for creating TTIR operations. It provides support for MLIR modules to be generated from user-constructed ops, lowered into TTNN or TTMetal backends, and finally translated into executable flatbuffers. Or you can do all three at once!

## Building

1. Build [tt-mlir](../getting-started.md)
2. Build [`ttrt`](../ttrt.md#building)
3. Generate ttsys file from the system you want to compile for using `ttrt`. This will create a `ttrt-artifacts` folder containing a `system_desc.ttsys` file.

```bash
ttrt query --save-artifacts
```

4. Export this file in your environment. `ttir_builder.utils` uses the `system_desc.ttsys` file as it runs a pass over an MLIR module to the TTNN or TTMetal backend.
```bash
export SYSTEM_DESC_PATH=/path/to/system_desc.ttsys
```

## Getting started

`TTIRBuilder` is a builder class providing the API for creating TTIR ops. The python package `builder` contains everything needed to create ops through a `TTIRBuilder` object. `builder.base.builder_utils` contains the APIs for wrapping op-creating-functions into MLIR modules and flatbuffers files.

```python
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer
```

## Creating a TTIR module

`build_ttir_module` defines an MLIR module specified as a python function. It wraps `fn` in a MLIR FuncOp then wraps that in an MLIR module, and finally ties arguments of that FuncOp to test function inputs. It will instantiate and pass a `TTIRBuilder` object as the last argument of `fn`. Each op returns an `OpView` type which is a type of `Operand` that can be passed into another builder op as an input.

```python
def build_ttir_module(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    module_dump: bool = False,
    base: Optional[str] = None,
    output_root: str = ".",
)
```

### Example

```python
from builder.base.builder import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import build_ttir_module

shapes = [(32, 32), (32, 32), (32, 32)]

def model(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
    add_0 = builder.add(in0, in1)
    multiply_1 = builder.multiply(in1, add_0)
    return builder.multiply(multiply_1, in2)

module, builder = build_ttir_module(model, shapes)
```

#### Returns

An MLIR module containing an MLIR op graph defined by `fn` and the `TTIRBuilder` object used to create it

```mlir
module {
  func.func @model(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = ttir.empty() : tensor<32x32xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %2 = ttir.empty() : tensor<32x32xf32>
    %3 = "ttir.multiply"(%arg1, %1, %2) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %4 = ttir.empty() : tensor<32x32xf32>
    %5 = "ttir.multiply"(%3, %arg2, %4) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %5 : tensor<32x32xf32>
  }
}
```

## Running a pipeline

`run_ttir_pipeline` runs a pass on the TTIR module to lower it into a backend, using `pipeline_fn`. You can pass `pipeline_fn` in as one of the following: `ttir_to_ttnn_backend_pipeline`, `ttir_to_ttmetal_backend_pipeline` (both found in `ttmlir.passes`), or a custom pipeline built with `create_custom_pipeline_fn`. The default if none is provided is the TTNN pipeline.

```python
def run_ttir_pipeline(
    module,
    pipeline_fn: Callable = ttir_to_ttnn_backend_pipeline,
    pipeline_options: List[str] = None,
    dump_to_file: bool = True,
    output_file_name: str = "test.mlir",
    system_desc_path: Optional[str] = None,
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    argument_types_string: Optional[str] = None,
)
```

### TTNN example

Let's expand on our previous example

```python
from ttmlir.passes import ttir_to_ttnn_backend_pipeline
from builder.base.builder import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import build_ttir_module, run_ttir_pipeline

shapes = [(32, 32), (32, 32), (32, 32)]

def model(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
    add_0 = builder.add(in0, in1)
    multiply_1 = builder.multiply(in1, add_0)
    return builder.multiply(multiply_1, in2)

module, builder = build_ttir_module(model, shapes)
ttnn_module = run_ttir_pipeline(module, ttir_to_ttnn_backend_pipeline)
```

#### Returns

An MLIR module lowered into TTNN

<details>

```mlir
#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 97248, erisc_l1_unreserved_base = 69632, dram_unreserved_base = 32, dram_unreserved_end = 1073158336, physical_helper_cores = {dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth_inactive = [ 16x18,  16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  16x25,  17x19,  17x20,  17x22,  17x23,  17x24]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [3 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
      func.func @model(%arg0: tensor<32x32xf32, #ttnn_layout>, %arg1: tensor<32x32xf32, #ttnn_layout>, %arg2: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
        %0 = "ttnn.abs"(%arg0) : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout>
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<32x32xf32, #ttnn_layout>) -> ()
        %1 = "ttnn.multiply"(%arg1, %0) : (tensor<32x32xf32, #ttnn_layout>, tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout>
        "ttnn.deallocate"(%0) <{force = false}> : (tensor<32x32xf32, #ttnn_layout>) -> ()
        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<32x32xf32, #ttnn_layout>) -> ()
        %2 = "ttnn.multiply"(%1, %arg2) : (tensor<32x32xf32, #ttnn_layout>, tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<32x32xf32, #ttnn_layout>) -> ()
        "ttnn.deallocate"(%arg2) <{force = false}> : (tensor<32x32xf32, #ttnn_layout>) -> ()
        return %2 : tensor<32x32xf32, #ttnn_layout>
      }
    }
  }
}
```

</details>

### TTMetal example

Let's use the same code for TTMetal that was used in the TTNN example but change the `pipeline_fn` to `ttir_to_ttmetal_backend_pipeline`. Only one or the other can be run on a module since `run_ttir_pipeline` modifies the module in place. Note that while all TTIR ops supported by builder can be lowered to TTNN, not all can be lowered to TTMetal yet. Adding documentation to specify what ops can be lowered to TTMetal is in the works.

```python
from ttmlir.passes import ttir_to_ttmetal_backend_pipeline
ttmetal_module = run_ttir_pipeline(module, ttir_to_ttmetal_backend_pipeline)
```

#### Returns

An MLIR module lowered into TTMetal

<details>

```mlir
#l1 = #ttcore.memory_space<l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, physical_helper_cores = {dram = [ 8x0,  9x0,  10x0,  8x1,  9x1,  10x1,  8x2,  9x2,  10x2,  8x3,  9x3,  10x3]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [3 : i32], [ 0x0x0x0]>
module {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
      func.func @model(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) -> memref<32x32xf32> {
        %0 = "ttmetal.create_buffer"() <{address = 9216 : i64}> : () -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
        %1 = "ttmetal.create_buffer"() <{address = 1024 : i64}> : () -> memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>
        "ttmetal.enqueue_write_buffer"(%arg0, %1) : (memref<32x32xf32>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        "ttmetal.enqueue_program"(%1, %0, %1, %0) <{cb_ports = array<i64: 0, 1>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel0, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc0>, #ttmetal.compute_config<@compute_kernel1, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, hifi4, false, false, [default]>], operandSegmentSizes = array<i32: 2, 2>}> : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%1) : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        %2 = "ttmetal.create_buffer"() <{address = 1024 : i64}> : () -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
        %3 = "ttmetal.create_buffer"() <{address = 5120 : i64}> : () -> memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>
        "ttmetal.enqueue_write_buffer"(%arg1, %3) : (memref<32x32xf32>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        "ttmetal.enqueue_program"(%3, %2, %3, %2) <{cb_ports = array<i64: 0, 1>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel2, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc0>, #ttmetal.compute_config<@compute_kernel3, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, hifi4, false, false, [default]>], operandSegmentSizes = array<i32: 2, 2>}> : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%3) : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        %4 = "ttmetal.create_buffer"() <{address = 13312 : i64}> : () -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
        "ttmetal.enqueue_program"(%0, %2, %4, %0, %2, %4) <{cb_ports = array<i64: 0, 1, 2>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel4, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>]>, noc0>, #ttmetal.noc_config<@datamovement_kernel5, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>]>, noc1>, #ttmetal.compute_config<@compute_kernel6, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>]>, hifi4, false, false, [default]>], operandSegmentSizes = array<i32: 3, 3>}> : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%0) : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%2) : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        %5 = "ttmetal.create_buffer"() <{address = 1024 : i64}> : () -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
        %6 = "ttmetal.create_buffer"() <{address = 5120 : i64}> : () -> memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>
        "ttmetal.enqueue_write_buffer"(%arg1, %6) : (memref<32x32xf32>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        "ttmetal.enqueue_program"(%6, %5, %6, %5) <{cb_ports = array<i64: 0, 1>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel7, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc0>, #ttmetal.compute_config<@compute_kernel8, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, hifi4, false, false, [default]>], operandSegmentSizes = array<i32: 2, 2>}> : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%6) : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        %7 = "ttmetal.create_buffer"() <{address = 17408 : i64}> : () -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
        "ttmetal.enqueue_program"(%5, %4, %7, %5, %4, %7) <{cb_ports = array<i64: 0, 1, 2>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel9, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>]>, noc0>, #ttmetal.noc_config<@datamovement_kernel10, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>]>, noc1>, #ttmetal.compute_config<@compute_kernel11, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>]>, hifi4, false, false, [default]>], operandSegmentSizes = array<i32: 3, 3>}> : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%5) : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%4) : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        %8 = "ttmetal.create_buffer"() <{address = 9216 : i64}> : () -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
        %9 = "ttmetal.create_buffer"() <{address = 1024 : i64}> : () -> memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>
        "ttmetal.enqueue_write_buffer"(%arg2, %9) : (memref<32x32xf32>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        "ttmetal.enqueue_program"(%9, %8, %9, %8) <{cb_ports = array<i64: 0, 1>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel12, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc0>, #ttmetal.compute_config<@compute_kernel13, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, hifi4, false, false, [default]>], operandSegmentSizes = array<i32: 2, 2>}> : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%9) : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        %10 = "ttmetal.create_buffer"() <{address = 5120 : i64}> : () -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>
        "ttmetal.enqueue_program"(%7, %8, %10, %7, %8, %10) <{cb_ports = array<i64: 0, 1, 2>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel14, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>]>, noc0>, #ttmetal.noc_config<@datamovement_kernel15, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>]>, noc1>, #ttmetal.compute_config<@compute_kernel16, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>]>, hifi4, false, false, [default]>], operandSegmentSizes = array<i32: 3, 3>}> : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%8) : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%7) : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        %alloc = memref.alloc() : memref<32x32xf32>
        %11 = "ttmetal.create_buffer"() <{address = 1024 : i64}> : () -> memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>
        "ttmetal.enqueue_program"(%10, %11, %10, %11) <{cb_ports = array<i64: 0, 1>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel17, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, noc0>, #ttmetal.compute_config<@compute_kernel18, #ttmetal.core_range<0x0, 1x1>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>]>, hifi4, false, false, [default]>], operandSegmentSizes = array<i32: 2, 2>}> : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>, memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        "ttmetal.deallocate_buffer"(%10) : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096>, #l1>) -> ()
        "ttmetal.enqueue_read_buffer"(%11, %alloc) : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>, memref<32x32xf32>) -> ()
        "ttmetal.finish"() : () -> ()
        "ttmetal.deallocate_buffer"(%11) : (memref<1x1x32x32xf32, #ttcore.shard<128x4>, #l1>) -> ()
        return %alloc : memref<32x32xf32>
      }
      func.func private @datamovement_kernel0() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @compute_kernel1() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
        %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        %2 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "tilize_init"(%1, %0, %2) : (!emitc.opaque<"::tt::CB">, i32, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "experimental::tilize_block"(%1, %2, %0, %0) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, i32, i32) -> ()
        emitc.call_opaque "cb_push_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel2() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @compute_kernel3() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
        %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        %2 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "tilize_init"(%1, %0, %2) : (!emitc.opaque<"::tt::CB">, i32, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "experimental::tilize_block"(%1, %2, %0, %0) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, i32, i32) -> ()
        emitc.call_opaque "cb_push_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel4() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel5() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        %1 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @compute_kernel6() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
        %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
        %1 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        emitc.call_opaque "tile_regs_acquire"() : () -> ()
        %2 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        %3 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        %4 = emitc.literal "get_compile_time_arg_val(2)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%4, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%2, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%3, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "binary_op_init_common"(%2, %3, %4) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "add_tiles_init"(%2, %3) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "add_tiles"(%2, %3, %0, %0, %0) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, !emitc.size_t, !emitc.size_t, !emitc.size_t) -> ()
        emitc.call_opaque "tile_regs_commit"() : () -> ()
        emitc.call_opaque "tile_regs_wait"() : () -> ()
        emitc.call_opaque "pack_tile"(%0, %4, %0) {template_args = [true]} : (!emitc.size_t, !emitc.opaque<"::tt::CB">, !emitc.size_t) -> ()
        emitc.call_opaque "tile_regs_release"() : () -> ()
        emitc.call_opaque "cb_push_back"(%4, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%4, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%2, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%3, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%4, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel7() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @compute_kernel8() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
        %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        %2 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "tilize_init"(%1, %0, %2) : (!emitc.opaque<"::tt::CB">, i32, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "experimental::tilize_block"(%1, %2, %0, %0) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, i32, i32) -> ()
        emitc.call_opaque "cb_push_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel9() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel10() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        %1 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @compute_kernel11() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
        %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
        %1 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        emitc.call_opaque "tile_regs_acquire"() : () -> ()
        %2 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        %3 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        %4 = emitc.literal "get_compile_time_arg_val(2)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%4, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%2, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%3, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "binary_op_init_common"(%2, %3, %4) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "mul_tiles_init"(%2, %3) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "mul_tiles"(%2, %3, %0, %0, %0) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, !emitc.size_t, !emitc.size_t, !emitc.size_t) -> ()
        emitc.call_opaque "tile_regs_commit"() : () -> ()
        emitc.call_opaque "tile_regs_wait"() : () -> ()
        emitc.call_opaque "pack_tile"(%0, %4, %0) {template_args = [true]} : (!emitc.size_t, !emitc.opaque<"::tt::CB">, !emitc.size_t) -> ()
        emitc.call_opaque "tile_regs_release"() : () -> ()
        emitc.call_opaque "cb_push_back"(%4, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%4, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%2, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%3, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%4, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel12() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @compute_kernel13() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
        %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        %2 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "tilize_init"(%1, %0, %2) : (!emitc.opaque<"::tt::CB">, i32, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "experimental::tilize_block"(%1, %2, %0, %0) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, i32, i32) -> ()
        emitc.call_opaque "cb_push_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel14() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel15() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        %1 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @compute_kernel16() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
        %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
        %1 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        emitc.call_opaque "tile_regs_acquire"() : () -> ()
        %2 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        %3 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        %4 = emitc.literal "get_compile_time_arg_val(2)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%4, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%2, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%3, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "binary_op_init_common"(%2, %3, %4) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "mul_tiles_init"(%2, %3) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "mul_tiles"(%2, %3, %0, %0, %0) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, !emitc.size_t, !emitc.size_t, !emitc.size_t) -> ()
        emitc.call_opaque "tile_regs_commit"() : () -> ()
        emitc.call_opaque "tile_regs_wait"() : () -> ()
        emitc.call_opaque "pack_tile"(%0, %4, %0) {template_args = [true]} : (!emitc.size_t, !emitc.opaque<"::tt::CB">, !emitc.size_t) -> ()
        emitc.call_opaque "tile_regs_release"() : () -> ()
        emitc.call_opaque "cb_push_back"(%4, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%4, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%2, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%3, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%4, %1) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @datamovement_kernel17() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
        %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_push_back"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
      func.func private @compute_kernel18() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
        %0 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
        %1 = emitc.literal "get_compile_time_arg_val(0)" : !emitc.opaque<"::tt::CB">
        %2 = emitc.literal "get_compile_time_arg_val(1)" : !emitc.opaque<"::tt::CB">
        emitc.call_opaque "cb_reserve_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "untilize_init"(%1) : (!emitc.opaque<"::tt::CB">) -> ()
        emitc.call_opaque "experimental::untilize_block"(%1, %2, %0, %0) : (!emitc.opaque<"::tt::CB">, !emitc.opaque<"::tt::CB">, i32, i32) -> ()
        emitc.call_opaque "cb_push_back"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_wait_front"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%1, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        emitc.call_opaque "cb_pop_front"(%2, %0) : (!emitc.opaque<"::tt::CB">, i32) -> ()
        return
      }
    }
  }
}
```

</details>

## Compiling into flatbuffer

`compile_ttir_to_flatbuffer` compiles a TTIRBuilder function `fn` straight to flatbuffer. This decorator is mainly a wrapper around the following functions, with each next function called on the output of the last: `build_ttir_module`, `run_ttir_pipeline`, and `ttnn_to_flatbuffer_file` or `ttmetal_to_flatbuffer_file` as dictated by the `target` parameter.

```python
def compile_ttir_to_flatbuffer(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    test_base: str = "test",
    output_root: str = ".",
    target: Literal["ttnn", "ttmetal"] = "ttnn",
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    module_dump: bool = True,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Union[Callable, str] = None,
    pipeline_options: List[str] = None,
)
```

No flatbuffer is printed or returned. It's only written to a file because it is created as an unsupported text encoding.

### TTNN example

Let's use our previous model function.

```python
from builder.base.builder import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer

shapes = [(32, 32), (32, 32), (32, 32)]

def model(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
    add_0 = builder.add(in0, in1)
    multiply_1 = builder.multiply(in1, add_0)
    return builder.multiply(multiply_1, in2)

compile_ttir_to_flatbuffer(
    model,
    shapes,
    target="ttnn",
)
```

### TTMetal example

Let's once again use the same code for TTMetal that was used in the TTNN example but change the `target` to `"ttmetal"`. Just as with `run_ttir_pipeline`, only one or the other can be run on a module since `compile_ttir_to_flatbuffer` modifies the module in place.

```python
compile_ttir_to_flatbuffer(
    model,
    shapes,
    target="ttmetal",
)
```

## Integrating with other tt-mlir tools

### Alternatives for file creation

1. The [`ttmlir-opt`](../ttmlir-opt.md) tool runs a compiler pass on an `.mlir` file.
2. The [`ttmlir-translate`](../ttmlir-translate.md) can generate a flatbuffer from an `.mlir` file.
3. [`llvm-lit`](https://github.com/tenstorrent/tt-mlir/blob/2064844f8140de7d38ba55f8acac107a016f32ab/docs/src/ttrt.md#generate-flatbuffer-files-using-llvm-lit) can also be used to generate a flatbuffer from an existing `.mlir` file.

### Running models

#### ttrt

[`ttrt`](../ttrt.md) is intended to be a swiss army knife for working with flatbuffers.

#### tt-explorer

[`tt-explorer`](../tt-explorer/tt-explorer.md) is a visualizer tool for `ttmlir`-powered compiler results.

#### ttnn-standalone

[`ttnn-standalone`](../ttnn-standalone.md) is a post-compile tuning/debugging tool.

#### llvm-lit

[`llvm-lit`](../lit-testing.md) can also be used for MLIR testing.

## Golden mode

### Golden dataclass

`TTIRBuilder` provides support to code golden tensors into flatbuffers which will be used for comparison with TT device output in `ttrt` runtime. `Golden` is the dataclass used to store information about a golden tensor. Each TTIR op should have a matching PyTorch op (or golden function built from PyTorch ops) which should perform exactly the same operation, generating the same outputs given the same inputs. You can use `TTIRBuilder` helper functions to store input, intermediate, and output tensors within the flatbuffer. Input and output goldens are mapped with keys "input_" and "output_" followed by a tensor index: `input_0`. Intermediate output tensors are mapped to the location of the respective op creation.

### GoldenCheckLevel Enum

`TTIRBuilder` stores an instance of the class `GoldenCheckLevel(Enum)` that dictates golden handling. It defaults to `GoldenCheckLevel.OP_LEVEL`. The exception is that `TTIRBuilder` CCL ops force the golden level to be set to `GRAPH_LEVEL`.

```
DISABLED : do not store goldens
OP_LEVEL : check every single op level goldens
GRAPH_LEVEL : check graph level goldens only
```

Check and set `GoldenCheckLevel` with `TTIRBuilder` APIs.

```python
from builder.base.builder import Operand, GoldenCheckLevel
from builder.ttir.ttir_builder import TTIRBuilder

def model(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
    builder.golden_check_level = GoldenCheckLevel.GRAPH_LEVEL
    add_0 = builder.add(in0, in1)
    multiply_1 = builder.multiply(in1, add_0)
    return builder.multiply(multiply_1, in2)
```

### Getting golden data

Unless otherwise specified in the `GoldenCheckLevel`, all input and output tensors will generate and store a golden in `TTIRBuilder` as a `Golden` type.

The `TTIRBuilder` API `get_golden_map(self)` is used to export golden data for flatbuffer construction. It returns a dictionary of golden tensor names and `GoldenTensor` objects.

To get info from a `GoldenTensor` object, use the attributes supported by `ttmlir.passes`: `name`, `shape`, `strides`, `dtype`, `data`.

```python
from ttmlir.passes import GoldenTensor
from builder.ttir.ttir_builder import TTIRBuilder

shapes = [(32, 32), (32, 32), (32, 32)]

def model(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
    add_0 = builder.add(in0, in1)
    builder.print_goldens()
    print(builder.get_golden_map())
    return add0
```

<details>

```
Golden tensor:
tensor([[ 4.0450e+00,  1.4274e+00,  5.9156e-01,  ..., -5.9834e-01,
         -1.1830e-01,  1.2837e-01],
        [ 2.3788e+00,  2.9242e-03, -5.2838e-02,  ...,  1.8294e+00,
          5.0348e+00,  9.7179e-01],
        [ 1.5168e-02,  1.0577e-01, -3.0682e-01,  ...,  6.7212e-01,
          9.4523e-02,  5.3765e+00],
        ...,
        [ 1.4241e-01,  1.1838e+00, -1.0601e+00,  ...,  4.9099e-01,
          4.2267e+00,  4.0610e-01],
        [ 5.6630e-01, -1.3068e-01, -1.7771e-01,  ...,  2.3862e+00,
          3.9376e-01,  7.3140e-01],
        [ 4.2420e+00,  1.7006e-01, -3.4861e-01,  ...,  1.1471e-01,
          1.6189e+00, -6.9106e-01]])
{'input_0': <ttmlir._mlir_libs._ttmlir.passes.GoldenTensor object at 0x7f77c70fa0d0>, 'output_0': <ttmlir._mlir_libs._ttmlir.passes.GoldenTensor object at 0x7f77c6fc9590>}
```

</details>

### Setting golden data

Use `TTIRBuilder` API `set_graph_input_output` to set your own input and output golden tensors using PyTorch tensors. Keep in mind that this also sets graph inputs and outputs. There are some functions for which setting custom input tensors is required to pass PCC accuracy checks: `ttir.tan`, `ttir.log`, `ttir.log1p`. See example implementation and explanation in `test/python/golden/test_ttir_ops.py`.

```python
set_graph_input_output(
        self,
        inputs: List[torch.Tensor],
        outputs: Optional[List[torch.Tensor]] = None,
        override: bool = False,
    )
```

```python
import torch

input_0 = torch.ones((32, 32))
output_0 = torch.zeros((32, 32))
builder.set_graph_input_output([input_0], [output_0], override=True)
```

### Running flatbuffer with golden data in ttrt

Running flatbuffers in `ttrt` requires additional building and setting up the environment. Run these commands before creating MLIR modules or flatbuffers so the system description in the flatbuffers match your device.

```bash
cmake --build build -- ttrt
ttrt query --save-artifacts
export SYSTEM_DESC_PATH=/path/to/system_desc.ttsys
```

Set environment variable `TTRT_LOGGER_LEVEL` to `DEBUG` so `ttrt` logs golden comparison results and prints graph level golden tensors.

```bash
export TTRT_LOGGER_LEVEL=DEBUG
```

Finally run ttrt. Our example flatbuffer file (since we didn't specify otherwise) defaulted to file path `./ttnn/test_ttnn.mlir.ttnn`. `--log-file ttrt.log` and `--save-golden-tensors` are both optional flags. They ensure that all golden data produced by the `ttrt` run gets written to files.

```bash
ttrt run ttnn/test_ttnn.mlir.ttnn --log-file ttrt.log --save-golden-tensors
```

#### Golden callbacks

The `ttrt` documentation contains a [section](https://github.com/tenstorrent/tt-mlir/blob/main/docs/src/ttrt.md#bonus-section-extending-runtime-to-other-fes) on the callback function feature. Callback functions run between each op execution during runtime and contain op level golden analysis. They are also customizable and provide the flexibility for you to get creative with your golden usage.

## Optimizer Overrides

The optimizer is the main component of tt-mlir responsible for performance. Documentation can be found [here](../optimizer.md) and more detail on many of the attributes implemented in these overrides can be found in the dropdown descriptions of the [ttnn ops](https://docs.tenstorrent.com/tt-mlir/autogen/md/Dialect/TTNNOp.html). There are three types of overrides the optimizer exposes as pipeline options - optimization policy, output layout overrides, and conv2d config overrides - all of which are designed only to be used in ttnn. `ttir-builder` supports all three, providing APIs to configure the respective pipeline options and add them to the pipeline run. To use overrides, tt-mlir must be built with `-DTTMLIR_ENABLE_OPMODEL=ON`; without it, optimizer overrides will not be applied to the pipeline.

### Optimization policy

Optimization policies instruct the optimizer how to shard tensors and/or allocate memory. The only supported policies at the moment are `DF Sharding` and `BF Interleaved`. Each can be passed into `compile_ttir_to_flatbuffer()` using the `optimization_policy` argument.

```bash
from ttmlir import optimizer_overrides

optimization_policy=optimizer_overrides.MemoryLayoutAnalysisPolicyType.BFInterleaved
```

For example, `"BF Interleaved"` will produce the following pipeline options:

```bash
system-desc-path=ttrt-artifacts/system_desc.ttsys enable-optimizer=true memreconfig-enabled=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=BFInterleaved
```

Example of ttnn module using the optimization policy `"BF Interleaved"`:

<details>

```bash
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 99904, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 2560032, dram_unreserved_end = 1073142400, physical_helper_cores = {dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth_inactive = [ 16x18,  16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  16x25,  17x18,  17x19,  17x20,  17x21,  17x22,  17x23,  17x24,  17x25]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_register_size_tiles = 8, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [3 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <interleaved>>
module {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
      func.func @matmul(%arg0: tensor<128x128xf32, #ttnn_layout>, %arg1: tensor<128x128xf32, #ttnn_layout>) -> tensor<128x128xf32, #ttnn_layout1> {
        %0 = "ttnn.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<128x128xf32, #ttnn_layout>, tensor<128x128xf32, #ttnn_layout>) -> tensor<128x128xf32, #ttnn_layout1>
        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<128x128xf32, #ttnn_layout>) -> ()
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<128x128xf32, #ttnn_layout>) -> ()
        return %0 : tensor<128x128xf32, #ttnn_layout1>
      }
    }
  }
}
```

</details>

Example of ttnn module without enabling optimization policy for comparison:

<details>

```bash
#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 99904, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 2560032, dram_unreserved_end = 1073142400, physical_helper_cores = {dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth_inactive = [ 16x18,  16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  16x25,  17x18,  17x19,  17x20,  17x21,  17x22,  17x23,  17x24,  17x25]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_register_size_tiles = 8, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [3 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
      func.func @matmul(%arg0: tensor<128x128xf32, #ttnn_layout>, %arg1: tensor<128x128xf32, #ttnn_layout>) -> tensor<128x128xf32, #ttnn_layout> {
        %0 = "ttnn.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<128x128xf32, #ttnn_layout>, tensor<128x128xf32, #ttnn_layout>) -> tensor<128x128xf32, #ttnn_layout>
        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<128x128xf32, #ttnn_layout>) -> ()
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<128x128xf32, #ttnn_layout>) -> ()
        return %0 : tensor<128x128xf32, #ttnn_layout>
      }
    }
  }
}
```

</details>

### Output layout overrides
The API [`set_output_layout_override`](https://docs.tenstorrent.com/tt-mlir/autogen/md/Module/ttir-builder/apis.html) can be used in the function `fn` passed into `compile_ttir_to_flatbuffer()`. These are op-level overrides and as such, the op to be overridden is passed in to `set_output_layout_override` as an argument. This an example of the full set of potential overrides and their implementation, any subset of the following can be used, whatever isn't will be set to default.

```bash
data_type : dictates the output tensor type
memory_layout : represents tensor memory layout ("row_major", "tile", "invalid")
buffer_type : specifies which memory type to use ("l1", "dram", "system_memory", "l1_small", "trace")
tensor_memory_layout : defines how the tensor is laid out in memory ("interleaved", "block_sharded", "width_sharded", "height_sharded")
grid_shape : shape of grid of cores which are used to store tensor in memory ([N, M])
```

```bash
output_layout_overrides = {
    "buffer_type": "l1",
}

def matmul_overrides(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
):
    matmul_0 = builder.matmul(in0, in1)
    builder.set_output_layout_override(output_layout_overrides, matmul_0)
    return matmul_0

compile_to_flatbuffer(
    matmul_overrides,
    [(128, 128)] * 2,
    [torch.float32] * 2,
)
```

This example will produce the following pipeline options:

```bash
system-desc-path=ttrt-artifacts/system_desc.ttsys enable-optimizer=true memreconfig-enabled=true override-output-layout=/home/$USER/tt-mlir/build/python_packages/ttir_builder/ops.py:3513:id(0)=1x1:l1:interleaved:tile:f32
```

Example of ttnn module using the output layout overrides detailed above:

<details>

```bash
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 99904, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 2560032, dram_unreserved_end = 1073142400, physical_helper_cores = {dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth_inactive = [ 16x18,  16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  16x25,  17x18,  17x19,  17x20,  17x21,  17x22,  17x23,  17x24,  17x25]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_register_size_tiles = 8, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [3 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x16x!ttcore.tile<32x32, f32>, #l1>, <interleaved>>
module {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
      func.func @matmul_overrides(%arg0: tensor<128x128xf32, #ttnn_layout>, %arg1: tensor<128x128xf32, #ttnn_layout>) -> tensor<128x128xf32, #ttnn_layout1> {
        %0 = "ttnn.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<128x128xf32, #ttnn_layout>, tensor<128x128xf32, #ttnn_layout>) -> tensor<128x128xf32, #ttnn_layout1>
        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<128x128xf32, #ttnn_layout>) -> ()
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<128x128xf32, #ttnn_layout>) -> ()
        return %0 : tensor<128x128xf32, #ttnn_layout1>
      }
    }
  }
}
```

</details>

Example of ttnn module without output layout overrides for comparison:

<details>

```bash
#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 99904, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 2560032, dram_unreserved_end = 1073142400, physical_helper_cores = {dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth_inactive = [ 16x18,  16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  16x25,  17x18,  17x19,  17x20,  17x21,  17x22,  17x23,  17x24,  17x25]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_register_size_tiles = 8, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [3 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
      func.func @matmul_overrides(%arg0: tensor<128x128xf32, #ttnn_layout>, %arg1: tensor<128x128xf32, #ttnn_layout>) -> tensor<128x128xf32, #ttnn_layout> {
        %0 = "ttnn.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<128x128xf32, #ttnn_layout>, tensor<128x128xf32, #ttnn_layout>) -> tensor<128x128xf32, #ttnn_layout>
        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<128x128xf32, #ttnn_layout>) -> ()
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<128x128xf32, #ttnn_layout>) -> ()
        return %0 : tensor<128x128xf32, #ttnn_layout>
      }
    }
  }
}
```

</details>

### Conv2d config overrides

The [ttnn documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.Conv2dConfig.html) provides the most detailed information on Conv2d config override options and functionality. This section will provide instructions on implementing them in `ttir-builder`.
The API [`set_conv2d_config_override(configs, op)`](https://docs.tenstorrent.com/tt-mlir/autogen/md/Module/ttir-builder/apis.html) can be used in the function `fn` passed into `compile_ttir_to_flatbuffer()`. These are op-level overrides and as such, the op to be overridden is passed in to `set_conv2d_config_override` as an argument. This API is only intended for ops `ttir.conv2d` and `ttir.conv_transpose2d`. This an example of the full set of potential overrides and their implementation, any subset of the following can be used, whatever isn't will be set to default.

```bash
conv2d_config = {
    "dtype": "f32",
    "weights_dtype": "f32",
    "activation": "relu",
}

def conv2d(
    in0: Operand,
    weight: Operand,
    bias: Operand,
    in1: Operand,
    builder: TTIRBuilder,
):
    conv2d_0 = builder.conv2d(
        in0,
        weight,
        bias,
        in1,
        stride=[2, 1],
        padding=[2, 1],
        dilation=[2, 1],
        groups=2,
    )
    builder.set_conv2d_config_override(conv2d_config, conv2d_0)
    return conv2d_0

compile_to_flatbuffer(
    conv2d,
    [(1, 32, 32, 64),
    (64, 32, 3, 3),
    (1, 1, 1, 64),
    (1, 16, 28, 64)],
    [torch.float32] * 4,
)
```

This will produce the following pipeline options:

```bash
system-desc-path=ttrt-artifacts/system_desc.ttsys enable-optimizer=true memreconfig-enabled=true override-conv2d-config=/home/$USER/tt-mlir/build/python_packages/ttir_builder/ops.py:2650:id(0)=dtype#f32:weights_dtype#f32:activation#relu:deallocate_activation#false:reallocate_halo_output#false:act_block_h_override#0:act_block_w_div#1:reshard_if_not_optimal#false:override_sharding_config#false:shard_layout#height_sharded:transpose_shards#false:output_layout#tile:enable_act_double_buffer#false:enable_weights_double_buffer#false:enable_split_reader#false:enable_subblock_padding#false
```

Example of ttnn module using the conv2d config overrides detailed above:

<details>

```bash
#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 99904, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 2560032, dram_unreserved_end = 1073142400, physical_helper_cores = {dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth_inactive = [ 16x18,  16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  16x25,  17x18,  17x19,  17x20,  17x21,  17x22,  17x23,  17x24,  17x25]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_register_size_tiles = 8, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [3 : i32], [ 0x0x0x0]>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 96 + d1 * 3 + d2, d3), <1x1>, memref<6144x3xf32, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <1x1>, memref<16x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 1024 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 576 + d1 * 576 + d2, d3), <1x1>, memref<18x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <1x1>, memref<16x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
      func.func @conv2d(%arg0: tensor<1x32x32x64xf32, #ttnn_layout>, %arg1: tensor<64x32x3x3xf32, #ttnn_layout1>, %arg2: tensor<1x1x1x64xf32, #ttnn_layout2>, %arg3: tensor<1x16x28x64xf32, #ttnn_layout3>) -> tensor<1x16x32x64xf32, #ttnn_layout3> {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        "ttnn.deallocate"(%arg3) <{force = false}> : (tensor<1x16x28x64xf32, #ttnn_layout3>) -> ()
        "ttnn.deallocate"(%arg2) <{force = false}> : (tensor<1x1x1x64xf32, #ttnn_layout2>) -> ()
        %1 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x32x32x64xf32, #ttnn_layout>) -> tensor<1x1x1024x64xf32, #ttnn_layout4>
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x32x32x64xf32, #ttnn_layout>) -> ()
        %2 = "ttnn.prepare_conv2d_weights"(%arg1, %0) <{batch_size = 1 : i32, conv2d_config = #ttnn.conv2d_config<dtype = f32, weights_dtype = f32, activation = "relu", deallocate_activation = false, reallocate_halo_output = false, act_block_h_override = 0, act_block_w_div = 1, reshard_if_not_optimal = false, override_sharding_config = false, shard_layout = height_sharded, transpose_shards = false, output_layout = tile, enable_act_double_buffer = false, enable_weights_double_buffer = false, enable_split_reader = false, enable_subblock_padding = false>, dilation = array<i32: 2, 1>, groups = 2 : i32, has_bias = false, in_channels = 64 : i32, input_dtype = #ttcore.supportedDataTypes<f32>, input_height = 32 : i32, input_memory_config = #ttnn.memory_config<#dram, <interleaved>>, input_tensor_layout = #ttnn.layout<tile>, input_width = 32 : i32, kernel_size = array<i32: 3, 3>, out_channels = 64 : i32, output_dtype = #ttcore.supportedDataTypes<f32>, padding = array<i32: 2, 1>, stride = array<i32: 2, 1>, weights_format = "OIHW"}> : (tensor<64x32x3x3xf32, #ttnn_layout1>, !ttnn.device) -> tensor<1x1x576x64xf32, #ttnn_layout5>
        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<64x32x3x3xf32, #ttnn_layout1>) -> ()
        %3 = "ttnn.conv2d"(%1, %2, %0) <{batch_size = 1 : i32, conv2d_config = #ttnn.conv2d_config<dtype = f32, weights_dtype = f32, activation = "relu", deallocate_activation = false, reallocate_halo_output = false, act_block_h_override = 0, act_block_w_div = 1, reshard_if_not_optimal = false, override_sharding_config = false, shard_layout = height_sharded, transpose_shards = false, output_layout = tile, enable_act_double_buffer = false, enable_weights_double_buffer = false, enable_split_reader = false, enable_subblock_padding = false>, dilation = array<i32: 2, 1>, groups = 2 : i32, in_channels = 64 : i32, input_height = 32 : i32, input_width = 32 : i32, kernel_size = array<i32: 3, 3>, out_channels = 64 : i32, output_dtype = #ttcore.supportedDataTypes<f32>, padding = array<i32: 2, 1>, stride = array<i32: 2, 1>}> : (tensor<1x1x1024x64xf32, #ttnn_layout4>, tensor<1x1x576x64xf32, #ttnn_layout5>, !ttnn.device) -> tensor<1x1x512x64xf32, #ttnn_layout6>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x1x576x64xf32, #ttnn_layout5>) -> ()
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x1x1024x64xf32, #ttnn_layout4>) -> ()
        %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 16 : i32, 32 : i32, 64 : i32]}> : (tensor<1x1x512x64xf32, #ttnn_layout6>) -> tensor<1x16x32x64xf32, #ttnn_layout3>
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x1x512x64xf32, #ttnn_layout6>) -> ()
        return %4 : tensor<1x16x32x64xf32, #ttnn_layout3>
      }
    }
  }
}
```

</details>

Example of ttnn module without conv2d config overrides for comparison:

<details>

```bash
#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 99904, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 2560032, dram_unreserved_end = 1073142400, physical_helper_cores = {dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth_inactive = [ 16x18,  16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  16x25,  17x18,  17x19,  17x20,  17x21,  17x22,  17x23,  17x24,  17x25]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_register_size_tiles = 8, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [3 : i32], [ 0x0x0x0]>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 96 + d1 * 3 + d2, d3), <1x1>, memref<6144x3xf32, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <1x1>, memref<16x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 1024 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 1024 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, f32>, #system_memory>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 1024 + d2, d3), <1x1>, memref<1024x64xf32, #system_memory>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 1024 + d2, d3), <1x1>, memref<1024x64xf32, #dram>, <interleaved>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <1x1>, memref<16x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
      func.func @conv2d(%arg0: tensor<1x32x32x64xf32, #ttnn_layout>, %arg1: tensor<64x32x3x3xf32, #ttnn_layout1>, %arg2: tensor<1x1x1x64xf32, #ttnn_layout2>, %arg3: tensor<1x16x28x64xf32, #ttnn_layout3>) -> tensor<1x16x32x64xf32, #ttnn_layout3> {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        "ttnn.deallocate"(%arg3) <{force = false}> : (tensor<1x16x28x64xf32, #ttnn_layout3>) -> ()
        "ttnn.deallocate"(%arg2) <{force = false}> : (tensor<1x1x1x64xf32, #ttnn_layout2>) -> ()
        %1 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x32x32x64xf32, #ttnn_layout>) -> tensor<1x1x1024x64xf32, #ttnn_layout4>
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x32x32x64xf32, #ttnn_layout>) -> ()
        %2 = "ttnn.from_device"(%1) : (tensor<1x1x1024x64xf32, #ttnn_layout4>) -> tensor<1x1x1024x64xf32, #ttnn_layout5>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x1x1024x64xf32, #ttnn_layout4>) -> ()
        %3 = "ttnn.to_layout"(%2) <{layout = #ttnn.layout<row_major>}> : (tensor<1x1x1024x64xf32, #ttnn_layout5>) -> tensor<1x1x1024x64xf32, #ttnn_layout6>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x1x1024x64xf32, #ttnn_layout5>) -> ()
        %4 = "ttnn.to_device"(%3, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x1x1024x64xf32, #ttnn_layout6>, !ttnn.device) -> tensor<1x1x1024x64xf32, #ttnn_layout7>
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x1x1024x64xf32, #ttnn_layout6>) -> ()
        %5 = "ttnn.conv2d"(%4, %arg1, %0) <{batch_size = 1 : i32, dilation = array<i32: 2, 1>, groups = 2 : i32, in_channels = 64 : i32, input_height = 32 : i32, input_width = 32 : i32, kernel_size = array<i32: 3, 3>, out_channels = 64 : i32, output_dtype = #ttcore.supportedDataTypes<f32>, padding = array<i32: 2, 1>, stride = array<i32: 2, 1>}> : (tensor<1x1x1024x64xf32, #ttnn_layout7>, tensor<64x32x3x3xf32, #ttnn_layout1>, !ttnn.device) -> tensor<1x1x512x64xf32, #ttnn_layout8>
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1x1024x64xf32, #ttnn_layout7>) -> ()
        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<64x32x3x3xf32, #ttnn_layout1>) -> ()
        %6 = "ttnn.reshape"(%5) <{shape = [1 : i32, 16 : i32, 32 : i32, 64 : i32]}> : (tensor<1x1x512x64xf32, #ttnn_layout8>) -> tensor<1x16x32x64xf32, #ttnn_layout3>
        "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x1x512x64xf32, #ttnn_layout8>) -> ()
        return %6 : tensor<1x16x32x64xf32, #ttnn_layout3>
      }
    }
  }
}
```

</details>
