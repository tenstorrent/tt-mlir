# `builder`

`builder` contains `ttir-builder`, `stablehlo-builder` and `ttnn-builder` to create mlir graphs.

## Important relevant packages
```python
from builder.base.builder_apis import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_apis import *
from builder.base.builder_runtime import *
```

## Build a ttir module
Check each op's definition to see what parameters you can override. They are kept as close as possible to their MLIR definition.
For example: tools/builder/ttir/ttir_builder.py see `@tag(ttir.SigmoidOp)`.
Under the hood, random inputs are getting generated for in0, all intermediates are evaluated and output is calculated.

```python
def module0(builder: TTIRBuilder):

  @builder.func([(32, 32)], [torch.float32])
  def modela(in0: Operand, builder: TTIRBuilder):
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
}
```

## Set inputs/outputs when building module
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
print(module)
```

## Change output dtype for mixed precision
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

## Build a stablehlo/ttnn module
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

## Build multiple functions
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

## Build nested functions
```python
def module0(builder: TTIRBuilder):
    @builder.func([(32, 32)], [torch.float32])
    def my_modela(in0: Operand, builder: TTIRBuilder):
        def nested_func(in0: Operand, builder: TTIRBuilder):
            sigmoid0 = builder.sigmoid(in0)
            return sigmoid0

        sigmoid0 = builder.sigmoid(in0)
        ttir_builder0 = TTIRBuilder(builder.context, builder.location)
        nested_func0 = builder.call(nested_func, [sigmoid0], ttir_builder0)
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

## Build device module graphs
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
            ttir_builder0 = TTIRBuilder(builder.context, builder.location)
            nested_func0 = builder.call(nested_func, [sigmoid0], ttir_builder0)
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

## Build cpu hoisted graphs
```python
def module0(builder: TTIRBuilder):
    @builder.func([(32, 32)], [torch.float32])
    def softmax(in0: Operand, builder: TTIRBuilder):
        return builder.softmax(
            in0,
            dimension=-1,
            numeric_stable=False,
            unit_attrs=["ttir.should_hoist"],
        )

new_module, builder = build_module(module0, "ttir")
print(new_module)
```

```mlir
module {
  func.func @softmax(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.softmax"(%arg0) <{dimension = -1 : si32, numericStable = false}> {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
```

## You can run any mlir pass on any generated module
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

## Compile a module to a flatbuffer
This will return the builder instance, where the .mlir file was dumped, the input/output golden dictionary per root level function and the intermediate golden dictionary.
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

## Interface with mlir runtime
All mlir runtime C++ APIs are nanobinded. See full list here: `runtime/python/runtime/runtime.cpp`.
You can also pass in `enable_intermediate_verification` in `compile_and_execute_ttir` to give you back a report of all intermediate golden results.
```python
import _ttmlir_runtime as tt_runtime

tt_runtime.runtime.set_current_device_runtime(tt_runtime.runtime.DeviceRuntime.TTNN)
mesh_options = tt_runtime.runtime.MeshDeviceOptions()
mesh_options.dispatch_core_type = tt_runtime.runtime.DispatchCoreType.ETH
mesh_options.mesh_shape = (1, 1)
device = tt_runtime.runtime.open_mesh_device(mesh_options)
tt_runtime.runtime.close_mesh_device(device)
```

## Execute flatbuffer
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

```
Program level golden for output_0 matched. pcc=0.9999998360475333
Program level golden for output_0 matched. pcc=0.9999998372226931
```

## Execute flatbuffer manually
You can optionally call the APIs manually. You can also pass in `enable_intermediate_verification` in `execute_fb` to give you back a report of all intermediate golden results.
```python
module, builder = build_module(module0, "ttir")

mlir_path, input_output_goldens, intermediate_goldens = compile_ttir_module_to_flatbuffer(
    module,
    builder,
    test_base="sample_test",
)

flatbuffer_path = os.path.join("ttir-builder-artifacts", "ttnn", f"sample_test_ttnn.mlir.ttnn")
execute_fb(flatbuffer_path, input_output_goldens, intermediate_goldens, device=device)
```

## Load mlir file
All goldens are evaluated under the hood with random inputs.
```python3
mlir_file_path = "test.mlir"
with open(mlir_file_path, 'r') as f:
    mlir_ir_string = f.read()

module, builder = load_mlir_file(mlir_ir_string, target="ttir")
print(module)
```

```mlir
module {
  func.func @my_modela(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
```

## Load mlir file with custom inputs
User can provide inputs per root level function to use in golden evaluation. All intermediate + output goldens are evaluated.
```python3
module, builder = load_mlir_file(mlir_ir_string, target="ttir", golden_inputs={"my_modela": [torch.randn(32, 32), torch.randn(32, 32)]})
```

## Execute loaded mlir file
```python3
module, builder = load_mlir_file(mlir_ir_string, target="ttir")

mlir_path, input_output_goldens, intermediate_goldens = compile_ttir_module_to_flatbuffer(
    module,
    builder,
    test_base="sample_test",
)

flatbuffer_path = os.path.join("ttir-builder-artifacts", "ttnn", f"sample_test_ttnn.mlir.ttnn")
execute_fb(flatbuffer_path, input_output_goldens, intermediate_goldens, device=device)
```

```
Program level golden for output_0 matched. pcc=0.9999998360475333
```

## Split mlir file
Splitting a mlir module will convert each op into a standalone module + op. The goldens are reused. Each intermediate op will take it's original golden inputs and set that as the inputs to its new module. This applied to the output as well.
```python3
mlir_ir_string = '''module {
  func.func @my_modela(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %1 = "ttir.sigmoid"(%0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
}'''

module, builder = load_mlir_file(mlir_ir_string, target="ttir")
builder_module_list = split_mlir_file(module, builder)

for module, builder in builder_module_list:
    print(module)
```

```mlir
module {
  func.func @add_module(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
```

```mlir
module {
  func.func @sigmoid_module(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.sigmoid"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
```

## Build multi-device graphs
```python
def module(builder: TTIRBuilder):
    @builder.func([(1, 1, 256, 512)], [torch.float32])
    def all_gather(in0: Operand, builder: TTIRBuilder):
        in_shard = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
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
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
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

## Build shardy annotated graph
```python
def module(builder: StableHLOBuilder):
    @builder.func([(2, 4, 8, 16), (2, 4, 8, 16)], [torch.float32, torch.float32])
    def op_sharding_annotation(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
    ):
        builder.set_graph_level_check(True)
        sharding_attr = builder.create_sharding_attr_from_tuples("mesh", [("x", True), ("y", False), ("", False), ("", False)])
        return builder.add(in0, in1, sharding_attr=sharding_attr)

module, builder = build_module(module, "stablehlo", mesh_dict=OrderedDict([("x", 2), ("y", 4)]))
print(module)
```

```mlir
module {
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func @op_sharding_annotation(%arg0: tensor<2x4x8x16xf32>, %arg1: tensor<2x4x8x16xf32>) -> tensor<2x4x8x16xf32> {
    %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y", ?}, {?}, {?}]>]>} : tensor<2x4x8x16xf32>
    return %0 : tensor<2x4x8x16xf32>
  }
}
```

## Run pipelines on shardy annotated graph
```python3
module, builder = build_module(module, "stablehlo", mesh_dict=OrderedDict([("x", 2), ("y", 4)]))
stablehlo_pipeline(module)
print(module)
```

```mlir
module {
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func @op_sharding_annotation(%arg0: tensor<2x4x8x16xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, %arg1: tensor<2x4x8x16xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<2x4x8x16xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"x"}, {"y"}, {}, {}]>, <@mesh, [{"x"}, {"y"}, {}, {}]>] out_shardings=[<@mesh, [{"x"}, {"y"}, {}, {}]>] manual_axes={"x", "y"} (%arg2: tensor<1x1x8x16xf32>, %arg3: tensor<1x1x8x16xf32>) {
      %1 = stablehlo.add %arg2, %arg3 : tensor<1x1x8x16xf32>
      sdy.return %1 : tensor<1x1x8x16xf32>
    } : (tensor<2x4x8x16xf32>, tensor<2x4x8x16xf32>) -> tensor<2x4x8x16xf32>
    return %0 : tensor<2x4x8x16xf32>
  }
}
```
