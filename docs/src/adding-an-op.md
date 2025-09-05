# Adding an Op

This guide will walk you through the process of adding a new Op end to end in
tt-mlir, in this case we will be adding a `matmul` operation. Note that the matmul
op was added as part of the same changeset as this guide, it could be useful to
reference the diff alongside this guide to see the changes in full.

This guide will cover the following steps:

- [Adding an Op](#adding-an-op)
  - [1. Define the Op in the TTIR frontend dialect](#1-define-the-op-in-the-ttir-frontend-dialect)
  - [2. Define the Op in the TTNN backend dialect](#2-define-the-op-in-the-ttnn-backend-dialect)
      - [`TTNNOps.td`](#ttnnopstd)
      - [`TTNNOps.cpp`](#ttnnopscpp)
      - [Adding constraint/runtime APIs](#adding-constraintruntime-apis)
  - [3. Convert / Implement the Op in the TTNN passes](#3-convert--implement-the-op-in-the-ttnn-passes)
  - [4. Add a compiler unit test for the Op](#4-add-a-compiler-unit-test-for-the-op)
      - [`test/ttmlir/Dialect/TTNN/matmul/simple_matmul.mlir`](#testttmlirdialectttnnmatmulsimple_matmulmlir)
  - [5. Define flatbuffer schema for the Op](#5-define-flatbuffer-schema-for-the-op)
      - [`include/ttmlir/Target/TTNN/CMakeLists.txt`](#includettmlirtargetttnncmakeliststxt)
      - [`include/ttmlir/Target/TTNN/operations/matmul.fbs`](#includettmlirtargetttnnoperationsmatmulfbs)
      - [`include/ttmlir/Target/TTNN/program.fbs`](#includettmlirtargetttnnprogramfbs)
  - [6. Serialize the Op in the flatbuffer format](#6-serialize-the-op-in-the-flatbuffer-format)
  - [7. Add runtime support for the Op](#7-add-runtime-support-for-the-op)
    - [`runtime/lib/ttnn/operations/matmul/matmul.cpp`](#runtimelibttnnoperationsmatmulmatmulcpp)
    - [`runtime/lib/ttnn/operations/CMakeLists.txt`](#runtimelibttnnoperationscmakeliststxt)
    - [`runtime/lib/ttnn/program_executor.cpp`](#runtimelibttnnprogramcpp)
  - [8. Add a silicon unit test for the Op](#8-add-a-silicon-unit-test-for-the-op)
    - [`test/ttmlir/Silicon/TTNN/matmul/simple_matmul.mlir`](#testttmlirsiliconttnnmatmulsimple_matmulmlir)
  - [9. Add an EmitC test for the Op](#9-add-an-emitc-test-for-the-op)
    - [`test/ttmlir/EmitC/TTNN/matmul/matmul.mlir`](#testttmliremitcttnnmatmulmatmulmlir)

## 1. Define the Op in the TTIR frontend dialect

We will start by defining the Op in the TTIR dialect. The TTIR Ops are defined
in a tablegen file located at `include/ttmlir/Dialect/TTIR/IR/TTIROps.td`.

> Tablegen is a domain-specific language for defining ops/types/attributes in MLIR and LLVM,
> these definitions constitute the dialect's *Operation Definition Specification*
> ([ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/)).

Here is an example of defining `matmul` in the TTIR dialect:

```td
{{#include ../../../include/ttmlir/Dialect/TTIR/IR/TTIROps.td:adding_an_op_matmul_ttir}}
```

There are many things to break down here, starting from the top:

- `def` in tablegen is used to define a concrete type, this will have a 1-1
  mapping to a C++ generated class, and for this particular case the build
  will end up generating file `build/include/ttmlir/Dialect/TTIR/IR/TTIROps.h.inc`.
- It inherits from `class TTIR_DPSOp`, classes in tablegen don't define a
  concrete type, but rather an interface that augment or constrain inherited `def`s.
  `TTIR_DPSOp` is a class that defines the common attributes for all TTIR Ops
  that implement *Destination Passing Style* (DPS) semantics.  DPS just means
  that the result tensor is passed as an argument to the operation which will
  be critical for modeling buffer allocation / lifetimes. Note the 3rd argument
  `AnyRankedTensor:$output`.
- Next we have a list of `arguments`.  These arguments consist of a mixture of
  `Type`s (i.e. `AnyRankedTensor`) and `Attribute`s.
  [Read more about Types & Attributes
  here](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/#attributes).
    - `AnyRankedTensor` is part of a tablegen standard library which type
      aliases to MLIR's builtin Tensor type, with the added constraint that the
      tensor has a static rank.  As much as possible we want to use the builtin
      types and infrastructure provided by MLIR.
- Next we have a list of `results` in this case just 1, which aliases the
  `output` tensor.  One drawback of DPS is that the result tensor and the
  output tensor will appear to have different SSA names in the IR, but they
  really alias the same object. This can make writing some passes more
  cumbersome.
- Next we have `extraClassDeclaration`, which enables us to inject member
  functions, written directly in C++, into the generated class.  We are doing
  this for this particular case in order to satisfy the DPS interface which
  requires an implementation for getting the mutated output tensor.
- Finally, we have `hasVerifier = 1`, this tells MLIR that we have a verifier
  function that will be called to validate the operation.  This is a good
  practice to ensure that the IR is well formed.

We can now try building and opening the `TTIROps.h.inc` file to see the generated C++ code.
We will actually get a linker error because we have `hasVerifier = 1` which
automatically declared a verifier function, but we need to go implement.

Let's head over to `lib/Dialect/TTIR/IR/TTIROps.cpp` and implement the verifier.

```cpp
{{#include ../../../lib/Dialect/TTIR/IR/TTIROps.cpp:adding_an_op_matmul_ttir_verify}}
```

## 2. Define the Op in the TTNN backend dialect

Next we will define the Op in the TTNN dialect.  TTNN Ops are defined in the
same way, but in their respective set of dialect files.  Refer to the previous
section for details, the process is the same.

#### `TTNNOps.td`
```
{{#include ../../../include/ttmlir/Dialect/TTNN/IR/TTNNOps.td:adding_an_op_matmul_ttnn}}
```

#### `TTNNOps.cpp`
```cpp
{{#include ../../../lib/Dialect/TTNN/IR/TTNNOps.cpp:adding_an_op_matmul_ttnn_verify}}
```

For more details on adding ops to the TTNN dialect, refer to [TTNN Dialect Contribution Guidelines](./ttnn-dialect-guidelines.md).

#### Adding constraint/runtime APIs
We need to implement two APIs when adding a TTNN Op, namely `getOpConstraints` and `getOpRuntime`.
More details about this can be found [here](./ttnn-op-constraints.md).


## 3. Convert / Implement the Op in the TTNN passes

### TTIR to TTNN

Next we will implement the conversion from the TTIR `matmul` Op to the TTNN `matmul` Op.
This is a trivial conversion, as the Ops are identical in their semantics, so
the changeset isn't going to be very instructive, but will at least point to the
files involved. The conversion is implemented in the `ConvertTTIRToTTNNPass` pass in
file `lib/Conversion/TTIRToTTNN/TTIRToTTNNPass.cpp`.

Zooming into `class ConvertTTIRToTTNNPass` we can see we implement the pass interface
via member function `void runOnOperation() final`.  This function will be called
for every operation matching the type specified in the pass tablegen file. A
quick look at `include/ttmlir/Conversion/Passes.td` we can see:

```
def ConvertTTIRToTTNN: Pass<"convert-ttir-to-ttnn", "::mlir::ModuleOp"> {
```

This means that `runOnOperation` will be called for every `ModuleOp` in the
graph, usually there is only one `ModuleOp` which serves as the root of the
graph.

Inside `runOnOperation` is usually where we define a rewrite pattern set that
can match much more complicated patterns (nested inside of the `ModuleOp`'s
[regions](https://mlir.llvm.org/docs/LangRef/#regions))
than just a single operation. In `runOperation` method you will see the call to
method `populateTTIRToTTNNPatterns(...)` that actually generates rewrite patterns.
Method `populateTTIRToTTNNPatterns(...)` is defined
in `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`.

```cpp
{{#include ../../../lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp:op_rewriter_pattern_set}}
```

> More information on rewrite patterns and their capabilities can be found in the MLIR documentation [here](https://mlir.llvm.org/docs/PatternRewriter/) and [here](https://mlir.llvm.org/docs/DialectConversion/).

For matmul, we defined a new conversion pattern that's generic to all binary ops
with arguments named `a` and `b`:

```cpp
{{#include ../../../lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp:adding_an_op_matmul_op_rewriter}}
```

Invoked as part of the rewrite set:
```cpp
MatmulOpConversionPattern
```

### TTNN to EmitC

Similarly, we also need to add a pattern to convert from TTNN dialect to EmitC dialect.

Method to populate rewrite patterns can be found in `lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp`:

```cpp
{{#include ../../../lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp:op_rewriter_pattern_set_emitc}}
```

Writing conversion patterns to EmitC is a little tricky at first. In general case, we will be converting an op that has operands (SSAs) and attributes (e.g. data type) as arguments. We want to flatten these arguments at call site.

We'll use EmitC's `CallOpaqueOp` as the target op. Let's take a look at our matmul IR within TTNN dialect:
```
"ttnn.matmul"(%2, %4, %5) : (tensor<64x128xbf16, #ttnn_layout4>, tensor<128x96xbf16, #ttnn_layout6>, tensor<64x96xbf16, #ttnn_layout7>) -> tensor<64x96xbf16, #ttnn_layout7>
```

Now let's look at matmul's call signature in TTNN lib:
```cpp
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const bool transpose_a = false,
        const bool transpose_b = false,
        const std::optional<const MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const MatmulProgramConfig>& program_config = std::nullopt,
        const std::optional<const std::string>& activation = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<const CoreGrid> core_grid = std::nullopt,
        const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        const std::optional<const DeviceGlobalCircularBuffer>& global_cb = std::nullopt);
```

If we look closely, we'll notice that the IR has way less arguments than can be seen in the actual signature of the op - as we're lowering to EmitC, which gets translated into actual C++ code, we need to correct for this (ideally the op would be perfectly modelled with all the arguments, but that is not the case today).

We do this by filling in the gaps. EmitC's `CallOpaqueOp` takes in an array of attributes, and an array of operands, which need to be combined. The combining is done by extending the array of attributes with "pointers" into operands, like so:
```cpp
{{#include ../../../lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp:adding_an_op_matmul_ttnn_to_emitc_array_attrs}}
```

Pointers are denoted with `IndexType`s, wrapped into `IntegerAttr`s. Attributes are converted into EmitC's `OpaqueAttr` which can, for practical purposes, be treated as strings: a `BoolAttr` carrying "false" as value needs to be converted into an `OpaqueAttr` whose value is a string `"false"`, which is what the `convertBoolAttr` function does.

This is our final converted EmitC `CallOpaqueOp`:

```mlir
emitc.call_opaque "ttnn::matmul"(%3, %6, %9) {args = [0 : index, 1 : index, #emitc.opaque<"false">, #emitc.opaque<"false">, #emitc.opaque<"std::nullopt">, #emitc.opaque<"std::nullopt">, #emitc.opaque<"std::nullopt">, #emitc.opaque<"std::nullopt">, #emitc.opaque<"std::nullopt">, #emitc.opaque<"std::nullopt">, #emitc.opaque<"std::nullopt">, 2 : index]} : (!emitc.opaque<"ttnn::Tensor">, !emitc.opaque<"ttnn::Tensor">, !emitc.opaque<"ttnn::Tensor">) -> !emitc.opaque<"ttnn::Tensor">
```

which, when translated to C++ code, looks like:

```cpp
ttnn::matmul(v6, v9, false, false, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, v12);
```

Full conversion pattern for matmul op:

```cpp
{{#include ../../../lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp:adding_an_op_matmul_op_rewriter_emitc}}
```

## 4. Add a compiler unit test for the Op

So far we have defined the Op in the TTIR and TTNN dialects,
implemented verifiers, and have conversion passes.  Now we need to add a unit
test to ensure that the pass is working correctly.  The compiler unit tests are located
in `test/ttmlir/Dialect` area.  In this case we'll add a test under the `TTNN`
subdirectory since we are testing the `ConvertTTIRToTTNNPass`.

#### `test/ttmlir/Dialect/TTNN/matmul/simple_matmul.mlir`

```mlir
{{#include ../../../test/ttmlir/Dialect/TTNN/matmul/simple_matmul.mlir}}
```

> Unit tests in MLIR are typically written using a tool called `FileCheck`, please refer to
> the llvm [FileCheck documentation](https://llvm.org/docs/CommandGuide/FileCheck.html)
> for a tutorial and more information about the `RUN` and `CHECK` directives.

A few things to point out specifically regarding tt-mlir dialects:

- `ttcore.system_desc`: This is a 1-1 mapping to the `SystemDesc` flatbuffer schema
  that is used to describe the system configuration.  This is a required
  attribute tagged on the top level module for all tt-mlir dialects.
- Pass `--ttnn-layout` is a prerequisite before running `convert-ttir-to-ttnn`.
  This pass is responsible for converting the input tensors to device memory
  space and tile layout before lowering to TTNN.
- This test is asserting that `ttir.matmul` converts to `ttnn.matmul`.

To run the test, you can use the following command:

```bash
cmake --build build -- check-ttmlir
```

You can also manually run `ttmlir-opt` on the test file to see the
resulting output:

```bash
./build/bin/ttmlir-opt --ttcore-register-device="system-desc-path=<PATH_TO_SYSTEM_DESC>" --ttir-to-ttnn-backend-pipeline test/ttmlir/Dialect/TTNN/matmul/simple_matmul.mlir
```

## 5. Define flatbuffer schema for the Op

Next we will define the flatbuffer schema for the Op.  The schema must capture
all tensor inputs, outputs, and attributes of the Op, i.e. everything the
runtime needs to execute the Op.

The schema can be placed in an existing `.fbs` file located in the `include/ttmlir/Target/TTNN/operations` directory.

If no suitable `.fbs` file exists for the operation category, feel free to create new `.fbs` files as needed. After creating a new `.fbs` file, remember to add a corresponding cmake target in the `include/ttmlir/Target/TTNN/CMakeLists.txt` file.

#### `include/ttmlir/Target/TTNN/CMakeLists.txt`
```cmake
{{#include ../../../include/ttmlir/Target/TTNN/CMakeLists.txt:adding_an_op_matmul_fbs_cmake}}
```

In our case, we can add our schema to `include/ttmlir/Target/TTNN/operations/matmul.fbs` directly, without needing to create a new file.

#### `include/ttmlir/Target/TTNN/operations/matmul.fbs`
```cpp
{{#include ../../../include/ttmlir/Target/TTNN/operations/matmul.fbs:adding_an_op_matmul_fbs}}
```

Type `TensorRef`, flatbuffer tables with suffix `Ref` are used to represent live values
during the runtime, decoupled from the underlying `Desc` suffixes which carry the
type and attribute information for the object.

After creating the schema for our new operation type, we need to register it in the `OpType` union
within `program.fbs`. This file serves as the main entry point for all program information,
where the `OpType` union collects and defines all supported operation types and their corresponding schemas.

#### `include/ttmlir/Target/TTNN/program.fbs`
```cpp
{{#include ../../../include/ttmlir/Target/TTNN/program.fbs:adding_an_op_matmul_fbs_op_type}}
```

If a new `.fbs` file was created, don't forget to include the new file in `include/ttmlir/Target/TTNN/program.fbs`.
```cpp
{{#include ../../../include/ttmlir/Target/TTNN/program.fbs:adding_an_op_matmul_fbs_include}}
```

> More information about writing flatbuffer schemas can be found in the
> [flatbuffers documentation](https://flatbuffers.dev/flatbuffers_guide_writing_schema.html)

## 6. Serialize the Op in the flatbuffer format

In the previous section we defined the flatbuffer schema for the `matmul`
Op, now let's put our new schema definition to use. The schema is used as input
to a program called `flatc` which generates C++ code (or any language for that
matter) for serializing and deserializing the schema. This generated code can be
found in `build/include/ttmlir/Target/TTNN/program_generated.h`.

Let's head over to `lib/Target/TTNN/TTNNToFlatbuffer.cpp` to define
a `createOp` overloaded function that does the conversion from MLIR to flatbuffer:

```cpp
{{#include ../../../lib/Target/TTNN/TTNNToFlatbuffer.cpp:adding_an_op_matmul_serialize_to_binary}}
```

Lots of things are happening here, let's break it down:
- `FlatbufferObjectCache`: This is a helper class that is used to cache
  objects in the flatbuffer that are created during the serialization process.
  This is necessary for managing value lifetimes and identifiers, at the same time
  it is an optimization to avoid having multiple copies of the same object. For example,
  a `TensorRef` with multiple uses could naively be recreated, one for each use,
  but with the cache we can ensure that the object is only created once
  and all uses point to the same flatbuffer offset. The cache is passed around to all
  serialization functions and should be used whenever creating a new object.
- `getOperandThroughDPSOps`: In section 1. we discussed DPS semantics and the
  drawback of having the result alias the output tensor. This is one of those
  cases where we need to use a helper function to trace through the output
  operands to find the original SSA name in order to associate it with the original
  `TensorRef`.
- `CreateMatmulOp`: The autogenerated function from the flatbuffer schema that
  actually serializes the data into the flatbuffer format.

We can finally generate a binary with our new Op!  We can use the following command:
```bash
./build/bin/ttmlir-opt --ttcore-register-device="system-desc-path=<PATH_TO_SYSTEM_DESC>" --ttir-to-ttnn-backend-pipeline test/ttmlir/Dialect/TTNN/matmul/simple_matmul.mlir | ./build/bin/ttmlir-translate --ttnn-to-flatbuffer -o out.ttnn
```

And we can inspect the with [`ttrt`](./ttrt.md#read):
```bash
ttrt read out.ttnn
```

Note: If the above `ttrt` command yields a segfault, a clean build of your workspace may be required: [Build Instructions](./getting-started.md#building-the-tt-mlir-project)

## 7. Add runtime support for the Op

Next, we want to add runtime support for the Op by parsing the flatbuffer and
invoking the TTNN API.

#### `runtime/lib/ttnn/operations/matmul/matmul.cpp`
```cpp
{{#include ../../../runtime/lib/ttnn/operations/matmul/matmul.cpp:adding_an_op_matmul_runtime_operations}}
```

A couple things to note from above:
- Most runtime op functions will follow a similar pattern, they will take in
  some additional datastructures for managing the program context.
  - Program context tracks the state of the current program. It stores intermediate tensors and devices.
- `tensorPool.at(op->in0()->global_id())`: `global_id` is a unique identifier
  for the tensor that was generated and managed by the `FlatbufferObjectCache`.
  This is how it's intended to be used by the runtime.
- Some operations may belong to a larger set of operations. For example, any eltwise unary operations can
  be added in `runtime/lib/ttnn/operations/eltwise/unary.cpp` directly without needing to create a new file.

If a new file is created for the op, we need to add a new source to `runtime/lib/ttnn/operations/CMakeLists.txt` and a new case to `runtime/lib/ttnn/program_executor.cpp`.

To update `runtime/lib/ttnn/operations/CMakeLists.txt`, include the path to the source file in `TTNN_OPS_SRCS`:

#### `runtime/lib/ttnn/operations/CMakeLists.txt`
```cmake
{{#include ../../../runtime/lib/ttnn/operations/CMakeLists.txt:adding_an_op_matmul_runtime_cmake}}
```

To update `runtime/lib/ttnn/program_executor.cpp`, add a new case to the `runOperation` method of `ProgramExecutor`:

#### `runtime/lib/ttnn/program_executor.cpp`
```cpp
{{#include ../../../runtime/lib/ttnn/program_executor.cpp:adding_an_op_matmul_runtime_program}}
```

We can test our changes with `ttrt` (don't forget to rebuild `ttrt`):
```bash
ttrt run out.ttnn
```

## 8. Add a silicon unit test for the Op
After adding runtime support, we're ready to test our Op on silicon. All silicon tests are located
under `test/ttmlir/Silicon`. The process is similar to [adding a compiler unit test](#4-add-a-compiler-unit-test-for-the-op).

In our specific case, we create a unit test here:

#### `test/ttmlir/Silicon/TTNN/matmul/simple_matmul.mlir`
```mlir
{{#include ../../../test/ttmlir/Silicon/TTNN/n150/matmul/simple_matmul.mlir}}
```

Couple things to point out about this process:
- Tests placed under `test/ttmlir/Dialect` will only test the compiler's capability of compiling the module.
If you want the module to run on silicon in CI, the test must be placed under `test/ttmlir/Silicon`.
- Notice the differences between the compilation headers of `test/ttmlir/Silicon/TTNN/simple_matmul.mlir` and `test/ttmlir/Dialect/TTNN/matmul/simple_matmul.mlir`
  - `--ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%"`: The `system-desc-path` option specifies the location of the system descriptor
  required for compiling the module. This is crucial for silicon tests, as modules compiled with different system descriptors may vary in silicon compatibility.
  Ensuring the system descriptor accurately reflects the target hardware is essential for running the module correctly.
  - `// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn`: This runs `ttmlir-translate` that serializes the output mlir module to a flatbuffer binary.
  We added the logic for this serialization in the [Serialize the Op in the flatbuffer format](#6-serialize-the-op-in-the-flatbuffer-format) section.

## 9. Add an EmitC test for the Op
Op should be tested in the EmitC (C++ codegen) path as well.

TTNN EmitC tests live in the `test/ttmlir/EmitC/TTNN` path. In our case, the test is in `test/ttmlir/EmitC/TTNN/matmul/matmul.mlir`.

#### `test/ttmlir/EmitC/TTNN/matmul/matmul.mlir`
```cpp
{{#include ../../../test/ttmlir/EmitC/TTNN/matmul/matmul.mlir}}
```

The first two `RUN` lines create a flatbuffer. The third and forth convert to EmitC dialect, translate to C++, then output the result to `matmul.mlir.cpp` file.

Additionally, the op's header file `operations/matmul/matmul.hpp` should be added to the list of includes in `tools/ttnn-standalone/ttnn-precompiled.hpp`:

```cpp
{{#include ../../../tools/ttnn-standalone/ttnn-precompiled.hpp:standalone_includes}}
```
