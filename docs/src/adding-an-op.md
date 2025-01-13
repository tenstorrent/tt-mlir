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
  - [3. Convert / Implement the Op in the TTNN passes](#3-convert--implement-the-op-in-the-ttnn-passes)
  - [4. Add a compiler unit test for the Op](#4-add-a-compiler-unit-test-for-the-op)
      - [`test/ttmlir/Dialect/TTNN/matmul/simple_matmul.mlir`](#testttmlirdialectttnnmatmulsimple_matmulmlir)
  - [5. Define flatbuffer schema for the Op](#5-define-flatbuffer-schema-for-the-op)
      - [`include/ttmlir/Target/TTNN/program.fbs`](#includettmlirtargetttnnprogramfbs)
  - [6. Serialize the Op in the flatbuffer format](#6-serialize-the-op-in-the-flatbuffer-format)
  - [7. Add runtime support for the Op](#7-add-runtime-support-for-the-op)
    - [`runtime/lib/ttnn/operations/matmul/matmul.cpp`](#runtimelibttnnoperationsmatmulmatmulcpp)
    - [`runtime/lib/ttnn/operations/CMakeLists.txt`](#runtimelibttnnoperationscmakeliststxt)
    - [`runtime/lib/ttnn/program.cpp`](#runtimelibttnnprogramcpp)
  - [8. Add a silicon unit test for the Op](#8-add-a-silicon-unit-test-for-the-op)
    - [`test/ttmlir/Silicon/TTNN/simple_matmul.mlir`](#testttmlirsiliconttnnsimple_matmulmlir)

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

## 3. Convert / Implement the Op in the TTNN passes

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

### Note:
We also need to add this op to the C++ emitter,
`lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp` see
`populateTTNNToEmitCPatterns(...)`.

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

- `tt.system_desc`: This is a 1-1 mapping to the `SystemDesc` flatbuffer schema
  that is used to describe the system configuration.  This is a required
  attribute tagged on the top level module for all tt-mlir dialects.
- Pass `--ttir-layout` is a prerequisite before running `convert-ttir-to-ttnn`.
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
./build/bin/ttmlir-opt --ttir-load-system-desc="path=<PATH_TO_SYSTEM_DESC>" --ttir-to-ttnn-backend-pipeline test/ttmlir/Dialect/TTNN/matmul/simple_matmul.mlir
```

## 5. Define flatbuffer schema for the Op

Next we will define the flatbuffer schema for the Op.  The schema must capture
all tensor inputs, outputs, and attributes of the Op, i.e. everything the
runtime needs to execute the Op.

#### `include/ttmlir/Target/TTNN/program.fbs`
```cpp
{{#include ../../../include/ttmlir/Target/TTNN/program.fbs:adding_an_op_matmul_fbs}}
```

Type `TensorRef`, flatbuffer tables with suffix `Ref` are used to represent live values
during the runtime, decoupled from the underlying `Desc` suffixes which carry the
type and attribute information for the object.

We also add this new op to the `union OpType`, which is the variant type for all
ops.

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
./build/bin/ttmlir-opt --ttir-load-system-desc="path=<PATH_TO_SYSTEM_DESC>" --ttir-to-ttnn-backend-pipeline test/ttmlir/Dialect/TTNN/matmul/simple_matmul.mlir | ./build/bin/ttmlir-translate --ttnn-to-flatbuffer -o out.ttnn
```

And we can inspect the with [`ttrt`](./ttrt.md):
```bash
ttrt read out.ttnn
```

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

If a new file is created for the op, we need to add a new source to `runtime/lib/ttnn/operations/CMakeLists.txt` and a new case to `runtime/lib/ttnn/program.cpp`.

To update `runtime/lib/ttnn/operations/CMakeLists.txt`, include the path to the source file in `TTNN_OPS_SRCS`:

#### `runtime/lib/ttnn/operations/CMakeLists.txt`
```cmake
{{#include ../../../runtime/lib/ttnn/operations/CMakeLists.txt:adding_an_op_matmul_runtime_cmake}}
```

To update `runtime/lib/ttnn/program.cpp`, add a new case to the `runOperation` method of `ProgramExecutor`:

#### `runtime/lib/ttnn/program.cpp`
```cpp
{{#include ../../../runtime/lib/ttnn/program.cpp:adding_an_op_matmul_runtime_program}}
```

We can test our changes with `ttrt` (don't forget to rebuild `ttrt`):
```bash
ttrt run out.ttnn
```

## 8. Add a silicon unit test for the Op
After adding runtime support, we're ready to test our Op on silicon. All silicon tests are located
under `test/ttmlir/Silicon`. The process is similar to [adding a compiler unit test](#4-add-a-compiler-unit-test-for-the-op).

In our specific case, we create a unit test here: `test/ttmlir/Silicon/TTNN/simple_matmul.mlir`:

#### `test/ttmlir/Silicon/TTNN/simple_matmul.mlir`
```mlir
{{#include ../../../test/ttmlir/Silicon/TTNN/simple_matmul.mlir}}
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
