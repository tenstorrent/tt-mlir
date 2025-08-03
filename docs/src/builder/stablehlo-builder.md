# `stablehlo-builder`

`stablehlo-builder` is a tool for creating stableHLO operations. It provides support for MLIR modules to be generated from user-constructed ops.

## Getting started

`StableHLOBuilder` is a builder class providing the API for creating stableHLO ops. The python package `builder` contains everything needed to create ops through a `StableHLOBuilder` object. `builder.stablehlo.stablehlo_utils` contains the APIs for wrapping op-creating-functions into MLIR modules and flatbuffers files.

```python
from builder import StableHLOBuilder
from builder.stablehlo.stablehlo_utils import build_stablehlo_module
```

## Creating a StableHLO module

`build_stablehlo_module` defines an MLIR module specified as a python function. It wraps `fn` in a MLIR FuncOp then wraps that in an MLIR module, and finally ties arguments of that FuncOp to test function inputs. It will instantiate and pass a `StableHLOBuilder` object as the last argument of `fn`. Each op returns an `OpView` type which is a type of `Operand` that can be passed into another builder op as an input.

```python
def build_stablehlo_module(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    mesh_shape: Optional[Tuple[int, int]] = None,
    module_dump: bool = False,
    base: Optional[str] = None,
    output_root: str = ".",
)
```

### Example

```python
from builder.stablehlo.stablehlo_utils import build_stablehlo_module
from builder import Operand, StableHLOBuilder

shapes = [(32, 32), (32, 32), (32, 32)]

def model(in0: Operand, in1: Operand, in2: Operand, builder: StableHLOBuilder):
    return builder.add(in0, in1)

module, builder = build_stablehlo_module(model, shapes)
```

#### Returns

An MLIR module containing an MLIR op graph defined by `fn` and the `TTIRBuilder` object used to create it

```mlir
module {
  func.func @model(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
```
