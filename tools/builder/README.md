# `builder`

`builder` contains `ttir-builder`, `stablehlo-builder` and `ttnn-builder` to create mlir graphs.

## import relevant packages

```python
from ttmlir.passes import ttir_to_ttnn_backend_pipeline
from builder.ttir.ttir_builder import TTIRBuilder
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_apis import *
from builder.base.builder_runtime import *
from builder.base.builder_enums import *

import torch
```

## build ttir module

Check each op's definition to see what parameters you can override. They are kept as close as possible to their MLIR definition. For example: tools/builder/ttir/ttir_builder.py see @tag(ttir.SigmoidOp). Under the hood, random inputs are getting generated for in0, all intermediates are evaluated and output is calculated.

```python
def module0(builder: TTIRBuilder):

  @builder.func([(32, 32)], [torch.float32])
  def modela(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0)
      return sigmoid0

new_module, builder = build_module(module0, "ttir")
print(new_module.operation.get_asm(enable_debug_info=True))
```

```mlir
#loc = loc("/code/jan-2/tt-mlir/test.py":12:0)
module {
  func.func @modela(%arg0: tensor<32x32xf32> loc("/code/jan-2/tt-mlir/test.py":12:0)) -> tensor<32x32xf32> {
    %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32> loc(#loc1)
    return %0 : tensor<32x32xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/code/jan-2/tt-mlir/test.py:16")
```

## override location

```python
def module0(builder: TTIRBuilder):

  @builder.func([(32, 32)], [torch.float32])
  def modela(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0, loc="tenstorrent://custom_location/sigmoid_op")
      return sigmoid0

new_module, builder = build_module(module0, "ttir")
print(new_module.operation.get_asm(enable_debug_info=True))
```

```mlir
#loc = loc("/code/jan-2/tt-mlir/test.py":12:0)
module {
  func.func @modela(%arg0: tensor<32x32xf32> loc("/code/jan-2/tt-mlir/test.py":12:0)) -> tensor<32x32xf32> {
    %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32> loc(#loc1)
    return %0 : tensor<32x32xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("tenstorrent://custom_location/sigmoid_op")
```

## add op attributes

```python
def module0(builder: TTIRBuilder):

  @builder.func([(32, 32)], [torch.float32])
  def modela(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0, unit_attrs=["ttir.should_hoist"])
      return sigmoid0

new_module, builder = build_module(module0, "ttir")
print(new_module)
```

```mlir
module {
  func.func @modela(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.sigmoid"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
```

## multiple returns

```python
shapes = [(2, 16, 32, 32), (16,), (16,), (16,), (16,)]
dtypes = [torch.float32, torch.float32, torch.float32, torch.float32, torch.float32]

def module(builder: TTIRBuilder):
    @builder.func(shapes, dtypes)
    def batch_norm_training(
        in0: Operand,
        scale: Operand,
        offset: Operand,
        running_mean: Operand,
        running_variance: Operand,
        builder,
        unit_attrs: Optional[List[str]] = None,
    ):

        result, batch_mean, batch_variance = builder.batch_norm_training(
            in0,
            scale,
            offset,
            running_mean,
            running_variance,
            epsilon=1e-5,
            dimension=1,
            momentum=0.1,
        )

        return result, batch_mean, batch_variance

new_module, builder = build_module(module, "ttir")
print(new_module)
```

```mlir
module {
  func.func @batch_norm_training(%arg0: tensor<2x16x32x32xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> (tensor<2x16x32x32xf32>, tensor<16xf32>, tensor<16xf32>) {
    %result, %batch_mean, %batch_variance = "ttir.batch_norm_training"(%arg0, %arg1, %arg2, %arg3, %arg4) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32, momentum = 1.000000e-01 : f32}> : (tensor<2x16x32x32xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> (tensor<2x16x32x32xf32>, tensor<16xf32>, tensor<16xf32>)
    return %result, %batch_mean, %batch_variance : tensor<2x16x32x32xf32>, tensor<16xf32>, tensor<16xf32>
  }
}
```

## set inputs/outputs

You can use builder APIs to set the inputs which will be used for all subsequent intermediate + output golden evaluation.

```python
def module(builder: TTIRBuilder):
    @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
    def test_with_mixed_init(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_goldens({in0: torch.zeros, in1: torch.ones})
        add_result = builder.add(in0, in1)
        return add_result

module, builder = build_module(module, "ttir")
print(new_module)
```

## set inputs/outputs with torch tensors

```python
def module(builder: TTIRBuilder):
    @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
    def test_with_mixed_init(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        input0 = torch.randn(32, 32)
        input1 = torch.randn(32, 32)
        builder.set_goldens({in0: input0, in1: input1})
        add_result = builder.add(in0, in1)
        return add_result

module, builder = build_module(module, "ttir")
print(new_module)
```

## set inputs/outputs with custom goldens

```python
def module(builder: TTIRBuilder):
    @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
    def test_with_mixed_init(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        input0 = torch.randn(32, 32)
        input1 = torch.randn(32, 32)
        builder.set_goldens({in0: input0, in1: input1})
        add_result = builder.add(in0, in1)
        builder.set_operand_goldens({add_result: torch.add(input0, input1)})
        return add_result

module, builder = build_module(module, "ttir")
print(new_module)
```

## change output dtype for mixed precision

```python
def module0(builder: TTIRBuilder):

  @builder.func([(32, 32)], [torch.float32])
  def modela(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0, output_type=torch.bfloat16)
      return sigmoid0

new_module, builder = build_module(module0, "ttir")
print(new_module)
```

```mlir
module {
  func.func @modela(%arg0: tensor<32x32xf32>) -> tensor<32x32xbf16> {
    %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }
}
```

## build stablehlo module

```python
def module0(builder: StableHLOBuilder):

  @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
  def modela(in0: Operand, in1: Operand, builder: StableHLOBuilder):
      add0 = builder.add(in0, in1)
      return add0

new_module, builder = build_module(module0, "stablehlo")
print(new_module)
```

```mlir
module {
  sdy.mesh @mesh = <["x"=1, "y"=1]>
  func.func @modela(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
```

## build ttnn module

```python
def module0(builder: TTNNBuilder):

  @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
  def modela(in0: Operand, in1: Operand, builder: TTNNBuilder):
      add0 = builder.add(in0, in1)
      return add0

new_module, builder = build_module(module0, "ttnn")
print(new_module)
```

```mlir
#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @modela(%arg0: tensor<32x32xf32, #ttnn_layout>, %arg1: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<32x32xf32, #ttnn_layout>, tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout>
    return %0 : tensor<32x32xf32, #ttnn_layout>
  }
}
```

## golden library

See `golden/mapping.py`.

## build multiple functions

All goldens are calculated for each function. Each root function will be executed as a separate program on device.

```python
def module0(builder: TTIRBuilder):

  @builder.func([(32, 32)], [torch.float32])
  def modela(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0)
      return sigmoid0

  @builder.func([(32, 32)], [torch.float32])
  def modelb(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0)
      return sigmoid0

new_module, builder = build_module(module0, "ttir")
print(new_module)
```

```mlir
module {
  func.func @modela(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
  func.func @modelb(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
```

## build nested functions

```python
def module0(builder: TTIRBuilder):
    @builder.func([(32, 32)], [torch.float32])
    def my_modela(in0: Operand, builder: TTIRBuilder):
        def nested_func(in0: Operand, builder: TTIRBuilder):
            sigmoid0 = builder.sigmoid(in0)
            return sigmoid0

        sigmoid0 = builder.sigmoid(in0)
        nested_func0 = builder.call(nested_func, [sigmoid0])
        return nested_func0

    @builder.func([(32, 32)], [torch.float32])
    def my_modelb(in0: Operand, builder: TTIRBuilder):
        sigmoid0 = builder.sigmoid(in0)
        return sigmoid0

new_module, builder = build_module(module0, "ttir")
print(new_module)
```

```mlir
module {
  func.func @my_modela(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    %1 = call @nested_func(%0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
  func.func private @nested_func(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
  func.func @my_modelb(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
```

## build device module graph

```python
def module0(builder: TTIRBuilder):
    @builder.device_module
    def my_device_module(builder: TTIRBuilder):
        @builder.func([(32, 32)], [torch.float32])
        def my_modela(in0: Operand, builder: TTIRBuilder):
            def nested_func(in0: Operand, builder: TTIRBuilder):
                sigmoid0 = builder.sigmoid(in0)
                return sigmoid0

            sigmoid0 = builder.sigmoid(in0)
            nested_func0 = builder.call(nested_func, [sigmoid0])
            return nested_func0

        @builder.func([(32, 32)], [torch.float32])
        def my_modelb(in0: Operand, builder: TTIRBuilder):
            sigmoid0 = builder.sigmoid(in0)
            return sigmoid0

new_module, builder = build_module(module0, "ttir")
print(new_module)
```

```mlir
module {
  ttcore.device_module {
    builtin.module {
      func.func @my_modela(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
        %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
        %1 = call @nested_func(%0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
        return %1 : tensor<32x32xf32>
      }
      func.func private @nested_func(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
        %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
        return %0 : tensor<32x32xf32>
      }
      func.func @my_modelb(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
        %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
        return %0 : tensor<32x32xf32>
      }
    }
  }
}
```

## run any pass on any module

See full list of passes supported at `python/Passes.cpp`.

```python
from ttmlir.passes import stablehlo_to_ttir_pipeline

def module0(builder: StableHLOBuilder):

  @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
  def modela(in0: Operand, in1: Operand, builder: StableHLOBuilder):
      add0 = builder.add(in0, in1)
      return add0

new_module, builder = build_module(module0, "stablehlo")
stablehlo_to_ttir_pipeline(new_module) # module is modified in-place!
print(new_module)
```

```mlir
module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
      func.func @modela(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
        %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
        return %0 : tensor<32x32xf32>
      }
    }
  }
}
```

## compile a module to a flatbuffer

```python
def module0(builder: TTIRBuilder):

  @builder.func([(32, 32)], [torch.float32])
  def modela(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0)
      return sigmoid0

  @builder.func([(32, 32)], [torch.float32])
  def modelb(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0)
      return sigmoid0

builder, module_file_path, input_output_goldens, intermediate_goldens = compile_ttir_to_flatbuffer(module0)
```

## compile a module to ttmetal

```python
def module0(builder: TTIRBuilder):

  @builder.func([(32, 32)], [torch.float32])
  def modela(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0)
      return sigmoid0

builder, module_file_path, input_output_goldens, intermediate_goldens = compile_ttir_to_flatbuffer(module0, target="ttmetal")
```

## compile a module to a emitc

```python
def module0(builder: TTIRBuilder):

  @builder.func([(32, 32)], [torch.float32])
  def modela(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0)
      return sigmoid0

builder, module_file_path, input_output_goldens, intermediate_goldens = compile_ttir_to_flatbuffer(module0, target="emitc")
```

## compile a module to a emitpy

```python
def module0(builder: TTIRBuilder):

  @builder.func([(32, 32)], [torch.float32])
  def modela(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0)
      return sigmoid0

builder, module_file_path, input_output_goldens, intermediate_goldens = compile_ttir_to_flatbuffer(module0, target="emitpy")
```

## interface with mlir runtime

All mlir runtime C++ APIs are nanobinded. See full list here: `runtime/python/runtime/runtime.cpp`. You can also pass in `enable_intermediate_verification` in `compile_and_execute_ttir` to give you back a report of all intermediate golden results.

```python
import _ttmlir_runtime as tt_runtime

tt_runtime.runtime.set_current_device_runtime(tt_runtime.runtime.DeviceRuntime.TTNN)
mesh_options = tt_runtime.runtime.MeshDeviceOptions()
mesh_options.dispatch_core_type = tt_runtime.runtime.DispatchCoreType.ETH
mesh_options.mesh_shape = (1, 1)
device = tt_runtime.runtime.open_mesh_device(mesh_options)
tt_runtime.runtime.close_mesh_device(device)
```

## nanobinds

runtime: `runtime/python/runtime/runtime.cpp`
ttcore, ttir, ttnn, d2m: `python/`
shardy: https://github.com/openxla/shardy/tree/main/shardy/integrations/python/ir
stablehlo: https://github.com/openxla/stablehlo/tree/main/stablehlo/integrations/python/mlir/dialects

## execute flatbuffer

Each root level function will execute and evaluate its pcc.

```python
import _ttmlir_runtime as tt_runtime

tt_runtime.runtime.set_current_device_runtime(tt_runtime.runtime.DeviceRuntime.TTNN)
mesh_options = tt_runtime.runtime.MeshDeviceOptions()
mesh_options.dispatch_core_type = tt_runtime.runtime.DispatchCoreType.ETH
mesh_options.mesh_shape = (1, 1)
device = tt_runtime.runtime.open_mesh_device(mesh_options)

def module0(builder: TTIRBuilder):

  @builder.func([(32, 32)], [torch.float32])
  def modela(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0)
      return sigmoid0

  @builder.func([(32, 32)], [torch.float32])
  def modelb(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0)
      return sigmoid0

compile_and_execute_ttir(
    module0,
    device=device,
)

tt_runtime.runtime.close_mesh_device(device)
```

## execute emitc

```python
import _ttmlir_runtime as tt_runtime

tt_runtime.runtime.set_current_device_runtime(tt_runtime.runtime.DeviceRuntime.TTNN)
mesh_options = tt_runtime.runtime.MeshDeviceOptions()
mesh_options.dispatch_core_type = tt_runtime.runtime.DispatchCoreType.ETH
mesh_options.mesh_shape = (1, 1)
device = tt_runtime.runtime.open_mesh_device(mesh_options)

def module0(builder: TTIRBuilder):

  @builder.func([(32, 32)], [torch.float32])
  def modela(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0)
      return sigmoid0

builder, module_file_path, input_output_goldens, intermediate_goldens = compile_ttir_to_flatbuffer(module0, target="emitc")
emitted_cpp_file =  "./ttir-builder-artifacts/emitc/test_ttnn.mlir.cpp"

execute_cpp(emitted_cpp_file, input_output_goldens, pcc=0.98, device=device)

tt_runtime.runtime.close_mesh_device(device)
```

## execute emitpy

```python
def module0(builder: TTIRBuilder):

  @builder.func([(32, 32)], [torch.float32])
  def modela(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0)
      return sigmoid0

builder, module_file_path, input_output_goldens, intermediate_goldens = compile_ttir_to_flatbuffer(module0, target="emitpy")
emitted_py_file =  "./ttir-builder-artifacts/emitpy/test_ttnn.mlir.py"

execute_py(emitted_py_file, input_output_goldens, pcc=0.98)
```

## execute flatbuffer manually

```python
import _ttmlir_runtime as tt_runtime

tt_runtime.runtime.set_current_device_runtime(tt_runtime.runtime.DeviceRuntime.TTNN)
mesh_options = tt_runtime.runtime.MeshDeviceOptions()
mesh_options.dispatch_core_type = tt_runtime.runtime.DispatchCoreType.ETH
mesh_options.mesh_shape = (1, 1)
device = tt_runtime.runtime.open_mesh_device(mesh_options)

def module0(builder: TTIRBuilder):

  @builder.func([(32, 32)], [torch.float32])
  def modela(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0)
      return sigmoid0

  @builder.func([(32, 32)], [torch.float32])
  def modelb(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0)
      return sigmoid0

module, builder = build_module(module0, "ttir")

compiled_bin, input_output_goldens, intermediate_goldens = compile_ttir_module_to_flatbuffer(
    module,
    builder,
)

execute_fb(compiled_bin, input_output_goldens, intermediate_goldens, device=device)

tt_runtime.runtime.close_mesh_device(device)
```

## bypass op

```python
import _ttmlir_runtime as tt_runtime

tt_runtime.runtime.set_current_device_runtime(tt_runtime.runtime.DeviceRuntime.TTNN)
mesh_options = tt_runtime.runtime.MeshDeviceOptions()
mesh_options.dispatch_core_type = tt_runtime.runtime.DispatchCoreType.ETH
mesh_options.mesh_shape = (1, 1)
device = tt_runtime.runtime.open_mesh_device(mesh_options)

def module0(builder: TTIRBuilder):

  @builder.func([(32, 32)], [torch.float32])
  def modela(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0)
      builder.bypass(sigmoid0)
      return sigmoid0

compile_and_execute_ttir(
    module0,
    device=device,
)

tt_runtime.runtime.close_mesh_device(device)
```

## load mlir file (useful for checking compiler pass goldens)

All goldens are evaluated under the hood with random inputs.

```python
mlir_file_path = "test.mlir"
with open(mlir_file_path, 'r') as f:
    mlir_ir_string = f.read()

module, builder = load_mlir_file(mlir_ir_string, target="ttir")
print(module)
```

## load mlir file with custom inputs

User can provide inputs per root level function to use in golden evaluation. All intermediate + output goldens are evaluated.

```python
mlir_file_path = "test.mlir"
with open(mlir_file_path, 'r') as f:
    mlir_ir_string = f.read()

module, builder = load_mlir_file(mlir_ir_string, target="ttir", golden_inputs={"modela": [torch.randn(32, 32), torch.randn(32, 32)], "modelb": [torch.randn(32, 32), torch.randn(32, 32)]})
print(module)
```

## execute loaded mlir file

```python
import _ttmlir_runtime as tt_runtime

tt_runtime.runtime.set_current_device_runtime(tt_runtime.runtime.DeviceRuntime.TTNN)
mesh_options = tt_runtime.runtime.MeshDeviceOptions()
mesh_options.dispatch_core_type = tt_runtime.runtime.DispatchCoreType.ETH
mesh_options.mesh_shape = (1, 1)
device = tt_runtime.runtime.open_mesh_device(mesh_options)

mlir_file_path = "test.mlir"
with open(mlir_file_path, 'r') as f:
    mlir_ir_string = f.read()

module, builder = load_mlir_file(mlir_ir_string, target="ttir")

compiled_bin, input_output_goldens, intermediate_goldens = compile_ttir_module_to_flatbuffer(
    module,
    builder,
)

execute_fb(compiled_bin, input_output_goldens, intermediate_goldens, device=device)

tt_runtime.runtime.close_mesh_device(device)
```

## split mlir file

Splitting a mlir module will convert each op into a standalone module + op. The goldens are reused. Each intermediate op will take it's original golden inputs and set that as the inputs to its new module. This applied to the output as well.

```python
module, builder = load_mlir_file(mlir_ir_string, target="ttir")
builder_module_list = split_mlir_file(module, builder)

for module, builder in builder_module_list:
    print(module)
```

## load+split+execute mlir file

```python
module, builder = load_mlir_file(mlir_ir_string, target="ttir")
builder_module_list = split_mlir_file(module, builder)

for split_module, split_builder in builder_module_list:
    print("-------------- Running test for split module: --------------")
    print(split_module)
    import _ttmlir_runtime as tt_runtime

    tt_runtime.runtime.set_current_device_runtime(tt_runtime.runtime.DeviceRuntime.TTNN)
    mesh_options = tt_runtime.runtime.MeshDeviceOptions()
    mesh_options.dispatch_core_type = tt_runtime.runtime.DispatchCoreType.ETH
    mesh_options.mesh_shape = (1, 1)
    device = tt_runtime.runtime.open_mesh_device(mesh_options)

    compiled_bin, input_output_goldens, intermediate_goldens = compile_ttir_module_to_flatbuffer(
        split_module,
        split_builder,
    )

    execute_fb(compiled_bin, input_output_goldens, intermediate_goldens, device=device)

    tt_runtime.runtime.close_mesh_device(device)
```

## profiler

```python
from profiler import *

with trace("/code/jan-2/tt-mlir/profiler", 8086):
    def module3(builder: TTIRBuilder):
        @builder.device_module
        def module(builder: TTIRBuilder):
            @builder.func([(32, 32), (32, 32), (32, 32)], [torch.float32, torch.float32, torch.float32])
            def model(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
                add = builder.add(in0, in1)
                exp = builder.exp(in2)
                return builder.multiply(add, exp)


    new_module, builder = build_module(module3, "ttir")
    print(new_module)

    import _ttmlir_runtime as tt_runtime

    tt_runtime.runtime.set_current_device_runtime(tt_runtime.runtime.DeviceRuntime.TTNN)
    mesh_options = tt_runtime.runtime.MeshDeviceOptions()
    mesh_options.dispatch_core_type = tt_runtime.runtime.DispatchCoreType.ETH
    mesh_options.mesh_shape = (1, 1)
    device = tt_runtime.runtime.open_mesh_device(mesh_options)

    compiled_bin, input_output_goldens, intermediate_goldens = compile_ttir_module_to_flatbuffer(
        new_module,
        builder,
    )

    golden_report = execute_fb(compiled_bin, input_output_goldens, intermediate_goldens, device=device)

    tt_runtime.runtime.close_mesh_device(device)
```

## build multi-device graphs

```python
def module(builder: TTIRBuilder):
    @builder.func([(1, 1, 256, 512)], [torch.float32])
    def all_gather(in0: Operand, builder: TTIRBuilder):
        in_shard = builder.mesh_shard(
            in0,
            shard_direction=MeshShardDirection.FullToShard.value,
            shard_type=MeshShardType.Devices.value,
            shard_shape=(1, 1, 8, 4),
            shard_dims=(2, 3),
        )

        all_gather0 = builder.all_gather(
            in_shard,
            all_gather_dim=3,
            cluster_axis=1,
        )

        return builder.mesh_shard(
            all_gather0,
            shard_direction=MeshShardDirection.ShardToFull.value,
            shard_type=MeshShardType.Devices.value,
            shard_shape=(1, 1, 8, 1),
            shard_dims=(2, -1),
        )

module, builder = build_module(module, "ttir", mesh_dict=OrderedDict([("x", 8), ("y", 4)]))
print(module)
```

```mlir
module {
  func.func @all_gather(%arg0: tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32> {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 8, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x512xf32>) -> tensor<1x1x32x128xf32>
    %1 = "ttir.all_gather"(%0) <{all_gather_dim = 3 : si32, cluster_axis = 1 : ui32}> : (tensor<1x1x32x128xf32>) -> tensor<1x1x32x512xf32>
    %2 = "ttir.mesh_shard"(%1) <{shard_dims = array<i64: 2, -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 8, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x512xf32>) -> tensor<1x1x256x512xf32>
    return %2 : tensor<1x1x256x512xf32>
  }
}
```

## build multi-device graphs with presharded args

```python
def module(builder: TTIRBuilder):
    @builder.func([(1, 1, 256, 512)], [torch.float32])
    def model(in0: Operand, builder: TTIRBuilder):
        builder.preshard_arg(in0, shard_dims=(-1, 3))
        in_shard = builder.mesh_shard(
            in0,
            shard_direction=MeshShardDirection.FullToShard.value,
            shard_type=MeshShardType.Identity.value,
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )
        exp = builder.exp(in_shard)
        out_shard = builder.mesh_shard(
            exp,
            shard_direction=MeshShardDirection.ShardToFull.value,
            shard_type=MeshShardType.Devices.value,
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )
        return out_shard

module, builder = build_module(module, "ttir", mesh_dict=OrderedDict([("x", 1), ("y", 2)]))
print(module)
```

```mlir
module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  func.func @model(%arg0: tensor<1x1x256x512xf32> {ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <presharded>, local_shape = tensor<1x1x256x256xf32>>}) -> tensor<1x1x256x512xf32> {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x256xf32>
    %1 = "ttir.exp"(%0) : (tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32>
    %2 = "ttir.mesh_shard"(%1) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x256xf32>) -> tensor<1x1x256x512xf32>
    return %2 : tensor<1x1x256x512xf32>
  }
}
```

## build shardy annotated graph

```python
def module(builder: StableHLOBuilder):
    @builder.func([(2, 4, 8, 16), (2, 4, 8, 16)], [torch.float32, torch.float32])
    def op_sharding_annotation(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        sharding_attr = builder.tensor_sharding_attr(
            mesh_name="mesh",
            dimension_shardings=[
                builder.dimension_sharding_attr(
                    axes=[builder.axis_ref_attr(name="x")],
                    is_closed=True,
                    priority=10,
                ),
                builder.dimension_sharding_attr(
                    axes=[builder.axis_ref_attr(name="y")],
                    is_closed=False,
                ),
                builder.dimension_sharding_attr(
                    axes=[],
                    is_closed=False,
                ),
                builder.dimension_sharding_attr(
                    axes=[],
                    is_closed=False,
                ),
            ],
        )

        return builder.add(
            in0,
            in1,
            sharding_attr=builder.tensor_sharding_per_value_attr([sharding_attr]),
        )

module, builder = build_module(module, "stablehlo", mesh_dict=OrderedDict([("x", 2), ("y", 4)]))
print(module)
```

```mlir
module {
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func @op_sharding_annotation(%arg0: tensor<2x4x8x16xf32>, %arg1: tensor<2x4x8x16xf32>) -> tensor<2x4x8x16xf32> {
    %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}p10, {"y", ?}, {?}, {?}]>]>} : tensor<2x4x8x16xf32>
    return %0 : tensor<2x4x8x16xf32>
  }
}
```

## build shardy annotated input graph

```python
def module(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def input_annotation(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        tensor_sharding_attr = builder.tensor_sharding_attr(
            mesh_name="mesh",
            dimension_shardings=[
                builder.dimension_sharding_attr(
                    axes=[builder.axis_ref_attr(name="x")],
                    is_closed=True,
                ),
                builder.dimension_sharding_attr(
                    axes=[builder.axis_ref_attr(name="y")],
                    is_closed=True,
                ),
            ],
        )
        builder.set_arg_attribute(in0, "sdy.sharding", tensor_sharding_attr)
        return builder.add(in0, in1)

module, builder = build_module(module, "stablehlo", mesh_dict=OrderedDict([("x", 2), ("y", 4)]))
print(module)
```

```mlir
module {
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func @input_annotation(%arg0: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
```

## run pipelines on shardy annotated graph

```python
module, builder = build_module(module, "stablehlo", mesh_dict=OrderedDict([("x", 2), ("y", 4)]))
stablehlo_pipeline(module)
print(module)
```

## build manual computation op graph

```python
def module(builder: StableHLOBuilder):
    @builder.func([(1, 1, 32, 32), (1, 1, 32, 32)], [torch.float32, torch.float32])
    def my_modela(in0: Operand, in1: Operand, builder: StableHLOBuilder):
        def single_device_func(
            inner0: Operand, inner1: Operand, builder: StableHLOBuilder
        ):
            add0 = builder.add(inner0, inner1)
            add1 = builder.add(add0, inner1)
            cosine0 = builder.cosine(add1)
            sin0 = builder.sine(add1)
            return cosine0, sin0

        tensor_sharding_attr = builder.tensor_sharding_attr(
            mesh_name="mesh",
            dimension_shardings=[
                builder.dimension_sharding_attr(
                    axes=[],
                    is_closed=True,
                ),
                builder.dimension_sharding_attr(
                    axes=[],
                    is_closed=True,
                ),
                builder.dimension_sharding_attr(
                    axes=[builder.axis_ref_attr(name="x")],
                    is_closed=True,
                ),
                builder.dimension_sharding_attr(
                    axes=[builder.axis_ref_attr(name="y")],
                    is_closed=True,
                ),
            ],
        )

        manual_computation_op0, manual_computation_op1 = builder.manual_computation(
            single_device_func,
            [in0, in1],
            in_shardings=[tensor_sharding_attr, tensor_sharding_attr],
            out_shardings=[tensor_sharding_attr, tensor_sharding_attr],
            manual_axes=["x", "y"],
        )
        return manual_computation_op0, manual_computation_op1

module, builder = build_module(module, "stablehlo", mesh_dict=OrderedDict([("x", 2), ("y", 4)]))
stablehlo_pipeline(module)
print(module)
```

```mlir
module {
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func @my_modela(%arg0: tensor<1x1x32x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, %arg1: tensor<1x1x32x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<1x1x32x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<1x1x32x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0:2 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{}, {}, {"x"}, {"y"}]>, <@mesh, [{}, {}, {"x"}, {"y"}]>] out_shardings=[<@mesh, [{}, {}, {"x"}, {"y"}]>, <@mesh, [{}, {}, {"x"}, {"y"}]>] manual_axes={"x", "y"} (%arg2: tensor<1x1x16x8xf32>, %arg3: tensor<1x1x16x8xf32>) {
      %1 = stablehlo.add %arg2, %arg3 : tensor<1x1x16x8xf32>
      %2 = stablehlo.add %1, %arg3 : tensor<1x1x16x8xf32>
      %3 = stablehlo.cosine %2 : tensor<1x1x16x8xf32>
      %4 = stablehlo.sine %2 : tensor<1x1x16x8xf32>
      sdy.return %3, %4 : tensor<1x1x16x8xf32>, tensor<1x1x16x8xf32>
    } : (tensor<1x1x32x32xf32>, tensor<1x1x32x32xf32>) -> (tensor<1x1x32x32xf32>, tensor<1x1x32x32xf32>)
    return %0#0, %0#1 : tensor<1x1x32x32xf32>, tensor<1x1x32x32xf32>
  }
}
```

## build manual computation op with ccl

```python
def module_all_gather(builder: StableHLOBuilder):
    @builder.func([(1, 1, 32, 32)], [torch.float32])
    def my_modela(in0: Operand, builder: StableHLOBuilder):
        def single_device_func(in0: Operand, builder: StableHLOBuilder):
            all_gather0 = builder.all_gather(in0, 3, [[0]])
            return all_gather0

        tensor_sharding_attr = builder.tensor_sharding_attr(
            mesh_name="mesh",
            dimension_shardings=[
                builder.dimension_sharding_attr(
                    axes=[],
                    is_closed=True,
                ),
                builder.dimension_sharding_attr(
                    axes=[],
                    is_closed=True,
                ),
                builder.dimension_sharding_attr(
                    axes=[builder.axis_ref_attr(name="x")],
                    is_closed=True,
                ),
                builder.dimension_sharding_attr(
                    axes=[builder.axis_ref_attr(name="y")],
                    is_closed=True,
                ),
            ],
        )

        manual_computation_op0 = builder.manual_computation(
            single_device_func,
            [in0],
            in_shardings=[tensor_sharding_attr],
            out_shardings=[tensor_sharding_attr],
            manual_axes=["x", "y"],
        )
        return manual_computation_op0

module, builder = build_module(module_all_gather, "stablehlo", mesh_dict=OrderedDict([("x", 1), ("y", 1)]))
print(module)
```

```mlir
module {
  sdy.mesh @mesh = <["x"=1, "y"=1]>
  func.func @my_modela(%arg0: tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32> {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}, {"x"}, {"y"}]>] out_shardings=[<@mesh, [{}, {}, {"x"}, {"y"}]>] manual_axes={"x", "y"} (%arg1: tensor<1x1x32x32xf32>) {
      %1 = "stablehlo.all_gather"(%arg1) <{all_gather_dim = 3 : i64, replica_groups = dense<0> : tensor<1x1xi64>}> : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
      sdy.return %1 : tensor<1x1x32x32xf32>
    } : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    return %0 : tensor<1x1x32x32xf32>
  }
}
```

## build sdy all gather op graph

```python
def module(builder: StableHLOBuilder):
    @builder.func([(1, 1, 32, 32), (1, 1, 32, 32)], [torch.float32, torch.float32])
    def my_modela(in0: Operand, in1: Operand, builder: StableHLOBuilder):
        tensor_sharding_attr0 = builder.tensor_sharding_attr(
            mesh_name="mesh",
            dimension_shardings=[
                builder.dimension_sharding_attr(
                    axes=[],
                    is_closed=True,
                ),
                builder.dimension_sharding_attr(
                    axes=[],
                    is_closed=True,
                ),
                builder.dimension_sharding_attr(
                    axes=[builder.axis_ref_attr(name="x")],
                    is_closed=True,
                ),
                builder.dimension_sharding_attr(
                    axes=[builder.axis_ref_attr(name="y")],
                    is_closed=True,
                ),
            ],
        )
        add0 = builder.add(
            in0,
            in1,
            sharding_attr=builder.tensor_sharding_per_value_attr(
                [tensor_sharding_attr0]
            ),
        )

        tensor_sharding_attr1 = builder.tensor_sharding_attr(
            mesh_name="mesh",
            dimension_shardings=[
                builder.dimension_sharding_attr(
                    axes=[],
                    is_closed=True,
                ),
                builder.dimension_sharding_attr(
                    axes=[],
                    is_closed=True,
                ),
                builder.dimension_sharding_attr(
                    axes=[],
                    is_closed=True,
                ),
                builder.dimension_sharding_attr(
                    axes=[],
                    is_closed=True,
                ),
            ],
        )
        axes_ref_list0 = builder.axes_ref_list_attr(axis_ref_list=[])
        axes_ref_list1 = builder.axes_ref_list_attr(axis_ref_list=[])
        axes_ref_list2 = builder.axes_ref_list_attr(
            axis_ref_list=[builder.axis_ref_attr(name="x")]
        )
        axes_ref_list3 = builder.axes_ref_list_attr(
            axis_ref_list=[builder.axis_ref_attr(name="y")]
        )
        gathering_axes = builder.list_of_axis_ref_lists_attr(
            [axes_ref_list0, axes_ref_list1, axes_ref_list2, axes_ref_list3]
        )
        sdy_all_gather0 = builder.sdy_all_gather(
            add0, gathering_axes, tensor_sharding_attr1
        )
        return sdy_all_gather0

module, builder = build_module(module, "stablehlo", mesh_dict=OrderedDict([("x", 2), ("y", 4)]))
print(module)
```

```mlir
module {
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func @my_modela(%arg0: tensor<1x1x32x32xf32>, %arg1: tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32> {
    %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"x"}, {"y"}]>]>} : tensor<1x1x32x32xf32>
    %1 = sdy.all_gather [{}, {}, {"x"}, {"y"}] %0 out_sharding=<@mesh, [{}, {}, {}, {}]> : tensor<1x1x32x32xf32>
    return %1 : tensor<1x1x32x32xf32>
  }
}
```

## load manual computation op graph

```python
mlir_file_path = "multidevice.mlir"
with open(mlir_file_path, 'r') as f:
    mlir_ir_string = f.read()

module, builder = load_mlir_file(mlir_ir_string, target="stablehlo")
print(module)
```

## automatic parallelization

```python
file_path = "permutation.mlir"
with open(file_path, "r", encoding="utf-8") as f:
    mlir_text = f.read()

module_permutations = generate_all_module_permutations(mlir_text, num_devices = 2)

for module_permutation in module_permutations:
    print("----- Module Permutation -----")
    print(module_permutation)

optimal_modules = get_optimal_module_least_num_collectives(module_permutations)

for optimal_module in optimal_modules:
    print("----- Optimal Module -----")
    print(optimal_module)
```

## debug dialect breakpoint

```python
def module0(builder: TTIRBuilder):

  @builder.func([(32, 32)], [torch.float32])
  def modela(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0)
      breakpoint0 = builder.breakpoint(sigmoid0)
      return breakpoint0

import _ttmlir_runtime as tt_runtime

tt_runtime.runtime.set_current_device_runtime(tt_runtime.runtime.DeviceRuntime.TTNN)
mesh_options = tt_runtime.runtime.MeshDeviceOptions()
mesh_options.dispatch_core_type = tt_runtime.runtime.DispatchCoreType.ETH
mesh_options.mesh_shape = (1, 2)
device = tt_runtime.runtime.open_mesh_device(mesh_options)

compile_and_execute_ttir(
    module0,
    device=device,
)

tt_runtime.runtime.close_mesh_device(device)
```

## debug dialect memory

```python
def module0(builder: TTIRBuilder):

  @builder.func([(32, 32)], [torch.float32])
  def modela(in0: Operand, builder: TTIRBuilder):
      sigmoid0 = builder.sigmoid(in0)
      memory0 = builder.memory_snapshot(sigmoid0, "sigmoid_memory_snapshot.json")
      return memory0

import _ttmlir_runtime as tt_runtime

tt_runtime.runtime.set_current_device_runtime(tt_runtime.runtime.DeviceRuntime.TTNN)
mesh_options = tt_runtime.runtime.MeshDeviceOptions()
mesh_options.dispatch_core_type = tt_runtime.runtime.DispatchCoreType.ETH
mesh_options.mesh_shape = (1, 2)
device = tt_runtime.runtime.open_mesh_device(mesh_options)

compile_and_execute_ttir(
    module0,
    device=device,
)

tt_runtime.runtime.close_mesh_device(device)
```
