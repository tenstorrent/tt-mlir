# `ttir-builder`

ttir-builder is a tool for creating ttir operations. It provides support for ops or a series of ops to be transformed into MLIR modules, then into `ttnn` or `ttmetal` `.mlir` files, and finally into executable flatbuffers. Or you can do all three steps at once!

For a full list of supported ops, see [`ttir-builder-ops`] (https://github.com/tenstorrent/tt-mlir/blob/main/docs/src/ttir-builder-ops.md)

## Building
Build [ttmlir](./build.md).

## Getting started

### Import ttir-builder as a python package
The package `ttir_builder` contains everything needed to create and store ops in the TTIRBuilder object. `ttir_builder.utils` contains functions for wrapping op-creating-functions into MLIR modules, straight into flatbuffers, and everything in between.
```bash
from ttir_builder import TTIRBuilder, Operand, Shape
from ttir_builder.utils import compile_to_flatbuffer
```

For more information on tt-mlir python bindings, see [python-bindings](./python-bindings.md).

## Creating an op
There are essentially two ways to go about using ttir-builder. We recommend the second if the eventual goal is to convert `TTIRBuilder` objects to other files  (it's more streamlined), but we will lay out both for you.

### Creating standalone TTIRBuilder objects
It's entirely doable to use this method to transform ops into modules, `mlir` files, and flatbuffers, but there currently isn't support in `ttir_builder.utils` to do so. It requires the `ttmlir.ir`, `ttmlir.dialects`, and `ttmlir.passes` packages and a little more elbow grease.

#### Instantiate TTIRBuilder object
```bash
from ttir_builder import TTIRBuilder, Operand, Shape
from ttmlir.ir import Context, Location
ctx = Context()
loc = Location.file(file_name, line_number, line_item_number, ctx)
builder = TTIRBuilder(ctx, loc)
```

#### Creating ops
Op creation follows the structure `op_name = builder.ttir_op_type(arguments)`. TTIR ops take input operands as `RankedTensorType` objects. These can be constructed through builder using a tuple for tensor shape and MLIR Type object.
```bash
mlir_data_type = builder.get_type_from_torch_dtype(torch.float32)
in0 = builder.ranked_tensor_type((32, 32)), mlir_data_type)
in1 = builder.ranked_tensor_type((32, 32)), mlir_data_type)
add_0 = builder.add(in0, in1)
```

TTIR ops created through builder can also be passed as arguments for new ops, setting the former op's output tensor as the new op's input.
```bash
multiply_1 = builder.multiply(add_0, in0)
```

Those builder functions create the following ttir ops.
```bash
%1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
%3 = "ttir.multiply"(%1, %arg0, %2) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
```

### Using `ttir-builder.utils`
The `ttir-builder.utils` package provides the most-user friendly way to transform ops. Since `TTIRBuilder` instantiation and translation requires object types defined in a few other dialects, it's easiest to pass op-creating functions as arguments into `ttir-builder.utils` APIs and let the APIs do the work for you.

We will use a basic implementation of the API `compile_to_flatbuffer` as an example, and detail the other APIs below.
```bash
from ttir_builder.utils import compile_to_flatbuffer,
from ttir_builder import Operand, TTIRBuilder

def model(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
    shapes = [(32, 32), (32, 32), (32, 32)]
    abs = builder.abs(in0)
    multiply_1 = builder.multiply(in0, abs)
    return builder.multiply(add, exp)

compile_to_flatbuffer(
    model,
    shapes,
)
```

## `ttir_builder` APIs
get_loc_of_extra_file_callee
class Golden
class TypeInfo
class GoldenCheckLevel
class TTIRBuilder

### Golden functions



## `ttir_builder.utils` APIs

```bash
shape_str(shape) : return a shape as a string of integers separated by "x"
set_output_path(path) : set a global output path
get_target_path(output_path, filename, target) : get a path in the form of "output_path/target/filename"
create_custom_pipeline_fn(pipeline: str, verify: bool = True) -> Callable : returns a function to serve as a pipeline to be run over a module, can be used as a replacement for `ttir_to_ttnn_backend_pipeline` or `ttir_to_ttmetal_backend_pipeline`
```


### Define a MLIR module specified as a python function.

#### Definition
It will wrap `test_fn` in a MLIR FuncOp and then wrap that in a MLIR
module, and finally tie arguments of that FuncOp to test function inputs. It will
also pass a `TTIRBuilder` object as the last argument of test function.
```bash
def build_mlir_module(
    test_fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    mesh_shape: Optional[Tuple[int, int]] = None,
    module_dump: bool = False,
    base: Optional[str] = None,
    output_root: str = ".",
):
```
#### Arguments:
`test_fn: Callable` : Python function to be converted to MLIR
`inputs_shapes: List[Shape]` Shapes of the respective ranked tensor inputs of the test function.
`module_dump: bool` : Set to True to print out generated MLIR module.
`golden_dump: bool` : Set to True to dump golden info to flatbuffer file.

#### Returns
MLIR module containing MLIR op graph defined by `test_fn`

#### Example
```bash
def test_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.add(in0, in1)

build_mlir_module(test_add, ((32, 32), (32, 32)))
```

which returns

```bash
module {
    func.func @test_add(
        %arg0: tensor<32x32xf32>,
        %arg1: tensor<32x32xf32>
    ) -> tensor<32x32xf32> {
        %0 = ttir.empty() : tensor<32x32xf32>
        %1 = "ttir.add"(%arg0, %arg1, %0) ...
        return %1 : tensor<32x32xf32>
    }
}
```


```bash
run_pipeline
compile_to_flatbuffer
```


## Bonus

### Other file creation methods
1. The [`ttmlir-opt`](./ttmlir-opt.md) tool runs a compiler pass on an `mlir` file.
2. The [`ttmlir-translate`](./ttmlir-translate.md) allows for flattbuffer generation from MLIR.

#### llvm-lit
Flatbuffers can be generated from existing `.mlir` files using [llvm-lit](
https://github.com/tenstorrent/tt-mlir/blob/2064844f8140de7d38ba55f8acac107a016f32ab/docs/src/ttrt.md#generate-flatbuffer-files-using-llvm-lit)

### Running models

#### ttrt
[`ttrt`](./ttrt.md) is intended to be a swiss army knife for working with flatbuffers.

#### tt-explorer
[`tt-explorer`](./tt-explorer.md) is a visualizer tool for `ttmlir`-powered compiler results.

#### ttnn-standalone
[`ttnn-standalone`](./ttnn-standalone.md) is a post-compile tuning/debugging tool.

#### llvm-lit
[`llvm-lit`](./lit-testing.md) is a tool that can be used for MLIR testing.


Optional To Do:

Add to overview.md
Golden functions

See `python/Passes.cpp` - specifically `ttnn_to_flatbuffer_file` function for an example. This is used by `tools/ttir-builder/builder.py` to construct flatbuffers with embedded golden data. You can store input/output/intermediate data within the flatbuffer. The choice of the map `key` for inputs/outputs is left to the golden implementor. The intermediate tensor key is derived from loc data for ttrt. External users can implement their own key/value logic.
