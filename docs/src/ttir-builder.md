# `ttir-builder`

`ttir-builder` is a tool for creating TTIR operations. It provides support for MLIR modules to be generated from user-constructed ops, lowered into TTNN or TTMetal backends, and finally translated into executable flatbuffers. Or you can do all three at once!

For a full list of supported ops, see `tools/ttir-builder/builder.py`.

## Getting started

### Building
Build [ttmlir](./build.md).

### Import ttir-builder as a python package
The package `ttir_builder` contains everything needed to create ops for a TTIRBuilder object. `ttir_builder.utils` contains the APIs for wrapping op-creating-functions into MLIR modules and flatbuffers files.
```bash
from ttir_builder import TTIRBuilder, Operand, Shape
from ttir_builder.utils import compile_to_flatbuffer
```

## Creating an op
There are essentially two ways to go about using ttir-builder. `TTIRBuilder` objects can be instantiated and used directly, however we recommend writing functions that take a `TTIRBuilder` object as an argument to use for op creation. These functions can be passed into `ttir_builder.utils` APIs that will wrap your function into MLIR FuncOps then into MLIR modules. The second provides a more streamlined, user-friendly way to build modules and flatbuffers, but we will lay out both for you.

### Creating standalone TTIRBuilder objects

#### Instantiate TTIRBuilder object
```bash
from ttir_builder import TTIRBuilder, Operand
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

Those builder functions create the following TTIR ops.
```bash
%1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
%3 = "ttir.multiply"(%1, %arg0, %2) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
```

`TTIRBuilder` types can be transformed into modules, but there currently isn't support for this method. Doing so requires the `ttmlir.ir`, `ttmlir.dialects`, and `ttmlir.passes` packages and a little more elbow grease. See `tools/ttir-builder/utils.py` for guidance on how to use those packages.

### Using `ttir_builder.utils` to wrap `ttir-builder` use
For our `ttir_builder.utils` API example, we will use a basic implementation of the API `compile_to_flatbuffer` that writes a TTIRBuilder wrapper function (the example function `model`) to a TTNN flatbuffer file.
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
    target="ttnn",
)
```

## `ttir_builder` APIs

### Golden
`TTIRBuilder` provides support to code golden tensors into flatbuffers which will be used for comparison with TT device output in `ttrt` runtime. `Golden` is the dataclass used to store information about a golden tensor. Each TTIR op should have a matching PyTorch op (or golden function built from PyTorch ops) which should perform exactly the same operation, generating the same outputs given the same inputs. You can use `TTIRBuilder` helper functions to store input and output tensors within the flatbuffer. Goldens are mapped with names "input_" and "output_" followed by a tensor index: `input_0`.

### TypeInfo
`TypeInfo` is a dataclass that encapsulates required type information for quantized tensors: `dtype`, `scale` and `zero_point`. For non-quantized types, a plain torch.dtype can be used.

### GoldenCheckLevel
`TTIRBuilder` stores an instance of `GoldenCheckLevel` that dictates golden handling.
```bash
DISABLED : do not store goldens
OP_LEVEL : check every single op level goldens
GRAPH_LEVEL : check graph level goldens only
```

### TTIRBuilder
`TTIRBuilder` is a builder class providing the API for creating TTIR ops and exposes the following functions.

#### Helper functions
```bash
goldens(self) -> Dict : gets dictionary mapping each `Golden` to its respective `Operand`
golden_check_level(self) -> GoldenCheckLevel : gets the `GoldenCheckLevel`
golden_check_level(self, level: GoldenCheckLevel) : sets the `GoldenCheckLevel`
get_context(self) -> Context : gets the `Context`
get_next_global_id(self) -> int : increments and gets the `_global_id` for the next op
print_goldens(self) : prints saved operands and their respective goldens in descriptive form which follows SSA ordering from MLIR graph
get_shape(self, input: Operand) -> Shape : retrieves shape of operand as a `Shape` type
generate_and_store_random_golden(self, operand: Operand, dtype: Union[torch.dtype, TypeInfo] = torch.float32) -> Golden : generates and returns random tensor following the shape of `operand`, assigns it to a golden, and maps `operand` to that golden
generate_input_golden(self, operand: Operand, dtype: Union[torch.dtype, TypeInfo], index: int, override: bool = False) : generates random tensor following the shape of `operand`, assigns it to a `Golden`, and maps it accordingly
get_golden_map(self) -> Dict : gets a dictionary of `GoldenTensor` types mapped to tensor names
set_mesh_shape(self, mesh_shape: Tuple[int, int]) : set mesh_shape for a multi-device environment
set_graph_input_output(self, inputs: List[torch.Tensor], outputs: Optional[List[torch.Tensor]] = None, override: bool = False) : records the input and output tensors for the graph.
```

#### Utility functions
```bash
get_type_from_torch_dtype(self, dtype: Union[torch.dtype, TypeInfo]) -> Type : converts PyTorch dtype or TypeInfo to corresponding MLIR Type
ranked_tensor_type(self, shape: Shape, data_type: Optional[Type] = None, encoding: Optional[Attribute] = None) -> RankedTensorType : convenience wrapper constructing RankedTensorType
metal_tensor_layout(self, shape: Shape, grid) -> RankedTensorType : convenience wrapper constructing RankedTensorType with layout as a MetalLayoutAttr type
```

## `ttir_builder.utils` APIs

### Helper functions
```bash
shape_str(shape) : returns a shape as a string of integers separated by "x"
set_output_path(path) : sets a global output path
get_target_path(output_path, filename, target) : returns a path in the form of "output_path/target/filename"
create_custom_pipeline_fn(pipeline: str, verify: bool = True) -> Callable : returns a function to serve as a pipeline to be run over a module, an alternative to `ttir_to_ttnn_backend_pipeline` or `ttir_to_ttmetal_backend_pipeline`
```

### Define a MLIR module specified as a python function.
`build_mlir_module` will wrap `test_fn` in a MLIR FuncOp and then wrap that in an MLIR module, and finally tie arguments of that FuncOp to test function inputs. It will instantiate and pass a `TTIRBuilder` object as the last argument of `test_fn`.
```bash
def build_mlir_module(
    test_fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    mesh_shape: Optional[Tuple[int, int]] = None,
    module_dump: bool = False,
    base: Optional[str] = None,
    output_root: str = ".",
)
```

#### Example
```bash
def test_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.add(in0, in1)

build_mlir_module(test_add, ((32, 32), (32, 32)))
```

#### Returns
An MLIR module containing MLIR op graph defined by `test_fn`

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

### Run a pipeline over a module
`run_pipeline` runs a pass on the TTIR module to lower it into a backend, using `pipeline_fn`. The pipeline can be one of the following: `ttir_to_ttnn_backend_pipeline`, `ttir_to_ttmetal_backend_pipeline` (both found in `ttmlir.passes`), or a custom pipeline built with `create_custom_pipeline_fn`. The TTNN backend is the default.
```bash
def run_pipeline(
    module,
    target: Literal["ttnn", "ttmetal"],
    pipeline_fn: Callable,
    pipeline_options: List[str] = None,
    dump_to_file: bool = True,
    output_file_name: str = "test.mlir",
    system_desc_path: Optional[str] = None,
    mesh_shape: Optional[Tuple[int, int]] = None,
    argument_types_string: Optional[str] = None,
)
```

#### Returns
MLIR module containing MLIR op graph defined by `module` and `pipeline_fn`.

### Put it all together and compile to flatbuffer
`compile_to_flatbuffer` combines `build_mlir_module`, `run_pipeline`, and `ttnn_to_flatbuffer_file` or `ttmetal_to_flatbuffer_file`. The choice of TTNN or TTMetal is controlled by the `target` parameter.

```bash
def compile_to_flatbuffer(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    test_base: str = "test",
    output_root: str = ".",
    target: Literal["ttnn", "ttmetal"] = "ttnn",
    mesh_shape: Optional[Tuple[int, int]] = None,
    module_dump: bool = True,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Union[Callable, str] = None,
    pipeline_options: List[str] = None,
)
```

#### Note
Translating a TTNN or TTMetal module to flatbuffer is a mechanism provided in `ttmlir.passes` rather than `ttir_builder.utils`.
```bash
from ttmlir.passes import ttnn_to_flatbuffer_file, ttmetal_to_flatbuffer_file
ttnn_to_flatbuffer_file(module: MlirModule, filepath: str = "", goldenMap: dict = {}, moduleCache: List[(str, str)] = [])
ttmetal_to_flatbuffer_file(module: MlirModule, filepath: str = "", goldenMap: dict = {}, moduleCache: List[(str, str)] = [])
```

## Bonus Section: Integrating with other tools

### Alternatives for file creation
1. The [`ttmlir-opt`](./ttmlir-opt.md) tool runs a compiler pass on an `.mlir` file.
2. The [`ttmlir-translate`](./ttmlir-translate.md) can generate a flatbuffer from an `.mlir` file.
3. [`llvm-lit`](
https://github.com/tenstorrent/tt-mlir/blob/2064844f8140de7d38ba55f8acac107a016f32ab/docs/src/ttrt.md#generate-flatbuffer-files-using-llvm-lit) can also be used to generate a flatbuffer from an existing `.mlir` file.

### Running models

#### ttrt
[`ttrt`](./ttrt.md) is intended to be a swiss army knife for working with flatbuffers.

#### tt-explorer
[`tt-explorer`](./tt-explorer.md) is a visualizer tool for `ttmlir`-powered compiler results.

#### ttnn-standalone
[`ttnn-standalone`](./ttnn-standalone.md) is a post-compile tuning/debugging tool.

#### llvm-lit
[`llvm-lit`](./lit-testing.md) can also be used for MLIR testing.

## Bonus Section: Add a new op type
`ttir-builder` is designed to only create ops supported in TTIR. At the moment, most but not all ops are supported, and new ops are still occasionally added to TTIR. Creating `ttir-builder` support for an op entails writing a function in `tools/ttir-builder/builder.py` that will create the op and its golden counterpart.

### TTIR op factories
All ops are created when their relevant information is run through the `op_proxy` function which provides a general interface for proxy-ing and creating ops.
```bash
def op_proxy(
    self,
    op_golden_function: Callable,
    op_ttir_function: Callable,
    inputs: List[Operand],
    unit_attrs: List[str] = None,
    organize_ttir_args: Optional[Callable] = None,
    organize_golden_args: Optional[Callable] = None,
    output_shape: Optional[Shape] = None,
    output_type: Optional[Type] = None,
    output_create_fn: Optional[Callable] = None,
    golden_kwargs: dict = {},
    ttir_kwargs: dict = {},
)
```

Eltwise ops require less specialized handling and call `op_proxy` through `eltwise_proxy`.
```bash
def eltwise_proxy(
    self,
    op_golden_function: Callable,
    op_ttir_function: Callable,
    inputs: List[Operand],
    unit_attrs: List[str] = None,
)
```

CCL ops require `GoldenCheckLevel` to be set to `GRAPH_LEVEL` and integrate that into their own proxy function.
```bash
def ccl_proxy(
    self,
    op_golden_function: Callable,
    op_ttir_function: Callable,
    inputs: List[Operand],
    kwargs: dict = {},
)
```

### Golden functions
Setting the various inputs, outputs, arguments, shapes, and types are all fairly straightforward. Find the TTIR op in `include/ttmlir/Dialect/TTIR/IR/TTIROps.td` and replicate the pertinents. If there is necessary information that is not included, you may have to take it upon yourself to do some detective work and trial and error. The tricky part can be the finding or writing a golden function. It must perform exactly the same operation as the TTIR op and be written using PyTorch operations.
