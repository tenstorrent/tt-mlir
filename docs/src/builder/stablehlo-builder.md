# `stablehlo-builder`

`stablehlo-builder` is a tool for creating stableHLO operations. It provides support for MLIR modules to be generated from user-constructed ops.

## Getting started

`StableHLOBuilder` is a builder class providing the API for creating stableHLO ops. The python package `builder` contains everything needed to create ops through a `StableHLOBuilder` object. `builder.base.builder_utils` contains the APIs for wrapping op-creating-functions into MLIR modules and flatbuffers files.

```python
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.base.builder_utils import build_stablehlo_module, compile_stablehlo_to_flatbuffer
```

## Creating a StableHLO module

`build_stablehlo_module` defines an MLIR module specified as a python function. It wraps `fn` in a MLIR FuncOp then wraps that in an MLIR module, and finally ties arguments of that FuncOp to test function inputs. It will instantiate and pass a `StableHLOBuilder` object as the last argument of `fn`. Each op returns an `OpView` type which is a type of `Operand` that can be passed into another builder op as an input.

```python
def build_stablehlo_module(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    module_dump: bool = False,
    base: Optional[str] = None,
    output_root: str = ".",
) -> Tuple[Module, StableHLOBuilder]:
```

### Example

```python
from builder.base.builder import Operand
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.base.builder_utils import build_stablehlo_module

shapes = [(32, 32), (32, 32), (32, 32)]

def model(in0: Operand, in1: Operand, in2: Operand, builder: StableHLOBuilder):
    return builder.add(in0, in1)

module, builder = build_stablehlo_module(model, shapes)
```

#### Returns

An MLIR module containing an MLIR op graph defined by `fn` and the `StableHLOBuilder` object used to create it

```mlir
module {
  func.func @model(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
```

## Creating a StableHLO module with Shardy annotations

`StableHLOBuilder` allows you to attach shardy annotations to the generated mlir graph.

### Example

```python
from builder.base.builder import Operand
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.base.builder_utils import build_stablehlo_module

shapes = [(32, 32), (32, 32)]

def model(in0: Operand, in1: Operand, shlo_builder: StableHLOBuilder):
    tensor_sharding_attr = shlo_builder.tensor_sharding_attr(
        mesh_name="mesh",
        dimension_shardings=[
            shlo_builder.dimension_sharding_attr(
                axes=[shlo_builder.axis_ref_attr(name="x")],
                is_closed=True,
            ),
            shlo_builder.dimension_sharding_attr(
                axes=[shlo_builder.axis_ref_attr(name="y")],
                is_closed=False,
            )
        ]
    )

    shlo_builder.sharding_constraint(in0, tensor_sharding_attr=tensor_sharding_attr)
    return shlo_builder.add(in0, in1)

module, shlo_builder = build_stablehlo_module(model, shapes, mesh_name="mesh", mesh_dict=OrderedDict([("x", 1), ("y", 8)]))
```

#### Returns

An MLIR module containing shardy annotations.

```mlir
module {
  sdy.mesh @mesh = <["x"=1, "y"=8]>
  func.func @model(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}, {"y", ?}]> : tensor<32x32xf32>
    %1 = stablehlo.add %arg0, %arg1 : tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
}
```

## Compiling into flatbuffer

`compile_stablehlo_to_flatbuffer` compiles a StableHLOBuilder function `fn` straight to flatbuffer. This decorator is mainly a wrapper around the following functions, with each next function called on the output of the last: `build_stablehlo_module`, `_run_ttir_pipeline`, and `ttnn_to_flatbuffer_file`, `ttmetal_to_flatbuffer_file`, or `ttir_to_ttnn_emitc_pipeline` as dictated by the `target` parameter.

```python
def compile_stablehlo_to_flatbuffer(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    system_desc_path: Optional[str] = None,
    test_base: str = "test",
    output_root: str = ".",
    target: Literal["ttnn", "ttmetal", "emitc"] = "ttnn",
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    module_dump: bool = True,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    ttir_pipeline_options: List[str] = [],
    shlo_pipeline_options: List[str] = [],
    shlo_to_ttir_pipeline_options: List[str] = [],
    print_ir: Union[bool, str] = False,
) -> str:
```

The executable flatbuffer is written to a file, `compile_stablehlo_to_flatbuffer` returns the file address of that flatbuffer.

### TTNN example

Let's use our previous model function.

```python
from builder.base.builder import Operand
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.base.builder_utils import compile_stablehlo_to_flatbuffer

shapes = [(32, 32), (32, 32)]

def model(in0: Operand, in1: Operand, shlo_builder: StableHLOBuilder):
    tensor_sharding_attr = shlo_builder.tensor_sharding_attr(
        mesh_name="mesh",
        dimension_shardings=[
            shlo_builder.dimension_sharding_attr(
                axes=[shlo_builder.axis_ref_attr(name="x")],
                is_closed=True,
            ),
            shlo_builder.dimension_sharding_attr(
                axes=[shlo_builder.axis_ref_attr(name="y")],
                is_closed=False,
            )
        ]
    )

    shlo_builder.sharding_constraint(in0, tensor_sharding_attr=tensor_sharding_attr)
    return shlo_builder.add(in0, in1)

compile_stablehlo_to_flatbuffer(
    model,
    shapes,
    mesh_name="mesh",
    mesh_dict=OrderedDict([("x", 1), ("y", 8)]),
    target="ttnn",
)
```

### TTMetal example

Let's once again use the same code for TTMetal that was used in the TTNN example but change the `target` to `"ttmetal"`. Just as with `_run_ttir_pipeline`, only one or the other can be run on a module since `compile_stablehlo_to_flatbuffer` modifies the module in place.

```python
compile_stablehlo_to_flatbuffer(
    model,
    shapes,
    mesh_name="mesh",
    mesh_dict=OrderedDict([("x", 1), ("y", 8)]),
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

`StableHLOBuilder` provides support to code golden tensors into flatbuffers which will be used for comparison with TT device output in `ttrt` runtime. `Golden` is the dataclass used to store information about a golden tensor. Each StableHLOBuilder op should have a matching PyTorch op (or golden function built from PyTorch ops) which should perform exactly the same operation, generating the same outputs given the same inputs. You can use `StableHLOBuilder` helper functions to store input, intermediate, and output tensors within the flatbuffer. Input and output goldens are mapped with keys "input_" and "output_" followed by a tensor index: `input_0`. Intermediate output tensors are mapped to the location of the respective op creation.

### GoldenCheckLevel Enum

`StableHLOBuilder` stores an instance of the class `GoldenCheckLevel(Enum)` that dictates golden handling. It defaults to `GoldenCheckLevel.OP_LEVEL`.

```
DISABLED : do not store goldens
OP_LEVEL : check every single op level goldens
GRAPH_LEVEL : check graph level goldens only
```

Check and set `GoldenCheckLevel` with `StableHLOBuilder` APIs.

```python
from builder.base.builder import Operand, GoldenCheckLevel
from builder.stablehlo.stablehlo_builder import StableHLOBuilder

def model(in0: Operand, in1: Operand, in2: Operand, builder: StableHLOBuilder):
    builder.golden_check_level = GoldenCheckLevel.GRAPH_LEVEL
    add_0 = builder.add(in0, in1)
    multiply_1 = builder.multiply(in1, add_0)
    return builder.multiply(multiply_1, in2)
```

### Getting golden data

Unless otherwise specified in the `GoldenCheckLevel`, all input and output tensors will generate and store a golden in `StableHLOBuilder` as a `Golden` type.

The `StableHLOBuilder` API `get_golden_map(self)` is used to export golden data for flatbuffer construction. It returns a dictionary of golden tensor names and `GoldenTensor` objects.

To get info from a `GoldenTensor` object, use the attributes supported by `ttmlir.passes`: `name`, `shape`, `strides`, `dtype`, `data`.

```python
from ttmlir.passes import GoldenTensor
from builder.stablehlo.stablehlo_builder import StableHLOBuilder

shapes = [(32, 32), (32, 32), (32, 32)]

def model(in0: Operand, in1: Operand, in2: Operand, builder: StableHLOBuilder):
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

Use `StableHLOBuilder` API `set_graph_input_output` to set your own input and output golden tensors using PyTorch tensors. Keep in mind that this also sets graph inputs and outputs.

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

Finally run ttrt. Our example flatbuffer file (since we didn't specify otherwise) defaulted to file path `./builder-artifacts/stablehlo-builder/test_ttnn/test_ttnn.mlir.ttnn`. `--log-file ttrt.log` and `--save-golden-tensors` are both optional flags. They ensure that all golden data produced by the `ttrt` run gets written to files.

```bash
ttrt run builder-artifacts/stablehlo-builder/test_ttnn/test_ttnn.mlir.ttnn --log-file ttrt.log --save-golden-tensors
```

#### Golden callbacks

The `ttrt` documentation contains a [section](https://github.com/tenstorrent/tt-mlir/blob/main/docs/src/ttrt.md#bonus-section-extending-runtime-to-other-fes) on the callback function feature. Callback functions run between each op execution during runtime and contain op level golden analysis. They are also customizable and provide the flexibility for you to get creative with your golden usage.
