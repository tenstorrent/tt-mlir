# Decomposing an Op in TTIR

This guide explains how to add and decompose a new operation in the TTIR dialect. We’ll focus on adding an `Index` operation, which will be decomposed into the `Slice` operation. The decomposition is implemented as a conversion pass in MLIR since it allows us to mark operations or dialects as legal or illegal, type conversion...

This guide will cover the following steps:
- [Decomposing an Op in TTIR](#decomposing-an-op-in-ttir)
  - [1. Define the Op in the TTIR frontend dialect](#1-define-the-op-in-the-ttir-frontend-dialect)
  - [2. Create a conversion pattern](#2-create-a-conversion-pattern)
      - [`C++ conversion pattern`](#c-conversion-pattern)
      - [`Tablegen conversion pattern`](#tablegen-conversion-pattern)
  - [3. Register the created conversion pattern](#3-register-the-created-conversion-pattern)

## 1. Define the Op in the TTIR frontend dialect

The more information regarding this step can be found here: [Define the Op in the TTIR frontend dialect](./adding-an-op.md#1-define-the-op-in-the-ttir-frontend-dialect)

I updated the `TTIROps.td` as following:

```td
{{#include ../../../include/ttmlir/Dialect/TTIR/IR/TTIROps.td:adding_an_op_index_ttir}}
```

The verification function has been added as well:

```cpp
{{#include ../../../lib/Dialect/TTIR/IR/TTIROps.cpp:adding_an_op_index_ttir}}
```

## 2. Create a conversion pattern

A conversion pattern defines how MLIR should rewrite the Op. It can be implemented in either C++ or TableGen. Currently, we only have the C++ implementation; TableGen format will be added in the future.

#### `C++ conversion pattern`

For the `Index` operation, we use the C++ conversion pattern because it involves changing the Op’s input types from integers to arrays, which TableGen lacks flexibility for.

```
{{#include ../../../include/ttmlir/Dialect/TTNN/IR/TTNNOps.td:adding_an_op_index_ttir}}
```

The `matchAndRewrite` method from `OpConversionPattern` is implemented to replace the matched Op with the newly created Op. Since decomposition is implemented as a conversion pass, `OpAdaptor` is used to access the attributes of the original Op in their converted types. Finally, we instantiate the new Op and call the `replaceOp` method on `ConversionPatternRewriter` to replace the original Op.

#### `Tablegen conversion pattern`
TODO

## 3. Register the created conversion pattern

To register the new pattern, go to the `populateTTIRToTTIRDecompositionPatterns` function in `TTIRToTTIRDecomposition.cpp` and add it to `RewritePatternSet` using the add method. After that is done you should mark the decomposed op as illegal in `runOnOperation` method of `TTIRToTTIRDecompositionPass` in `TTIRToTTIRDecompositionPass.cpp`.

You should also add a silicon test like described here: [Add a silicon unit test for the Op](./adding-an-op.md##8-add-a-silicon-unit-test-for-the-op). This is how the silicon test for the `Index` operation looks like:
```mlir
{{#include ../../../test/ttmlir/Silicon/TTNN/simple_index.mlir}}
```
