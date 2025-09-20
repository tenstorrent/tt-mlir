# `ttir-builder`

`ttir-builder` is a tool for creating TTIR operations. It provides support for MLIR modules to be generated from user-constructed ops, lowered into TTNN or TTMetal backends, and finally translated into executable flatbuffers. Or you can do all three at once!

## Building

1. Build [tt-mlir](./getting-started.md)
2. Build [`ttrt`](./ttrt.md#building)
3. Generate ttsys file from the system you want to compile for using `ttrt`. This will create a `ttrt-artifacts` folder containing a `system_desc.ttsys` file.

```bash
ttrt query --save-artifacts
```

4. Export this file in your environment using `export SYSTEM_DESC_PATH=/path/to/system_desc.ttsys`. `builder.base.builder_utils` uses the `system_desc.ttsys` file as it runs a pass over an MLIR module to the TTNN or TTMetal backend.

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
) -> Tuple[Module, TTIRBuilder]:
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
    pipeline_fn: Callable,
    pipeline_options: List[str] = [],
    dump_to_file: bool = True,
    output_file_name: str = "test.mlir",
    system_desc_path: Optional[str] = None,
    mesh_dict: OrderedDict[str, int] = None,
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

`compile_ttir_to_flatbuffer` compiles a TTIRBuilder function `fn` straight to flatbuffer. This decorator is mainly a wrapper around the following functions, with each next function called on the output of the last: `build_ttir_module`, `run_ttir_pipeline`, and `ttnn_to_flatbuffer_file`, `ttmetal_to_flatbuffer_file`, or `ttir_to_ttnn_emitc_pipeline` as dictated by the `target` parameter.

```python
def compile_ttir_to_flatbuffer(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    system_desc_path: Optional[str] = None,
    test_base: str = "test",
    output_root: str = ".",
    target: Literal["ttnn", "ttmetal", "ttnn-standalone"] = "ttnn",
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    module_dump: bool = True,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    pipeline_options: List[str] = [],
    print_ir: Union[bool, str] = False,
) -> str:
```

The executable flatbuffer is written to a file, `compile_ttir_to_flatbuffer` returns the file address of that flatbuffer.

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

1. The [`ttmlir-opt`](./ttmlir-opt.md) tool runs a compiler pass on an `.mlir` file.
2. The [`ttmlir-translate`](./ttmlir-translate.md) can generate a flatbuffer from an `.mlir` file.
3. [`llvm-lit`](https://github.com/tenstorrent/tt-mlir/blob/2064844f8140de7d38ba55f8acac107a016f32ab/docs/src/ttrt.md#generate-flatbuffer-files-using-llvm-lit) can also be used to generate a flatbuffer from an existing `.mlir` file.

### Running models

#### ttrt

[`ttrt`](./ttrt.md) is intended to be a swiss army knife for working with flatbuffers.

#### tt-explorer

[`tt-explorer`](./tt-explorer/tt-explorer.md) is a visualizer tool for `ttmlir`-powered compiler results.

#### ttnn-standalone

[`ttnn-standalone`](./ttnn-standalone.md) is a post-compile tuning/debugging tool.

#### llvm-lit

[`llvm-lit`](./lit-testing.md) can also be used for MLIR testing.

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

Running flatbuffers in `ttrt` requires building and setting up the environment. Run these commands before creating MLIR modules or flatbuffers so the system description in the flatbuffers match your device.

```bash
cmake --build build
ttrt query --save-artifacts
export SYSTEM_DESC_PATH=/path/to/system_desc.ttsys
```

Set environment variable `TTRT_LOGGER_LEVEL` to `DEBUG` so `ttrt` logs golden comparison results and prints graph level golden tensors.

```bash
export TTRT_LOGGER_LEVEL=DEBUG
```

Finally run ttrt. Our example flatbuffer file (since we didn't specify otherwise) defaulted to file path `./builder-artifacts/ttir-builder/test_ttnn/test_ttnn.mlir.ttnn`. `--log-file ttrt.log` and `--save-golden-tensors` are both optional flags. They ensure that all golden data produced by the `ttrt` run gets written to files.

```bash
ttrt run builder-artifacts/ttir-builder/test_ttnn/test_ttnn.mlir.ttnn --log-file ttrt.log --save-golden-tensors
```

#### Golden callbacks

The `ttrt` documentation contains a [section](https://github.com/tenstorrent/tt-mlir/blob/main/docs/src/ttrt.md#bonus-section-extending-runtime-to-other-fes) on the callback function feature. Callback functions run between each op execution during runtime and contain op level golden analysis. They are also customizable and provide the flexibility for you to get creative with your golden usage.
