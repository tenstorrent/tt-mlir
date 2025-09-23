# Adding OpConstraints and OpRuntime APIs to TTNN Operations

## Overview

The TTNN Op Model Interface provides two key APIs for analyzing and optimizing operations:

- **`getOpConstraints`**: Returns constraint information including memory requirements, layout compatibility, and operation feasibility
- **`getOpRuntime`**: Returns performance metrics including execution time estimates

These APIs enable the compiler to make informed decisions about operation placement, memory allocation, and performance optimization.

This guide walks you through best practices for implementing these APIs. It will cover the following steps:

1. [Architecture](#architecture)
2. [Implementation Steps](#implementation-steps)
   - [Step 1: Implement Operation-Specific Methods](#step-1-implement-operation-specific-methods)
   - [Step 2: Add Core Model Implementation](#step-2-add-core-model-implementation)
   - [Step 3: Add Unit Tests](#step-3-add-unit-tests)
   - [Step 4: Add Integration Tests](#step-4-add-integration-tests)
3. [Key Considerations](#key-considerations)
4. [Example: Complete Implementation](#example-complete-implementation)


## Architecture

The implementation follows a layered architecture:

```
TTNNOpModelInterface.cpp (Operation-specific implementations)
    ↓
TTNNOpModel.h/.cpp (Core model implementations and helpers)
    ↓
Metal Backend (Runtime execution and constraint validation)
```

Important note: `getOpConstraints` and `getOpRuntime` API calls should be identical to regular op invocation path through runtime.
The only difference is that one call is generated from the IR while the other is from serialised FB. For example, you can compare:

The runtime code `runtime/lib/ttnn/operations/conv/conv2d.cpp`:
```cpp
void run(const ::tt::target::ttnn::Conv2dOp *op, ProgramContext &context) {
  // ...
}
```

With the constraint API implementation code `lib/OpModel/TTNN/TTNNOpModel.cpp`:

```cpp
llvm::Expected<OpConstraints> OpModel<Conv2dOp>::getOpConstraints(/* args */){
  // ...
}
// and:
llvm::Expected<size_t> OpModel<Conv2dOp>::getOpRuntime(/* args */){
  // ...
}

```
And observe the similarities. This is very important to maintain throughout the lifetime of the project to guarantee
consistency and functional correctness.

## Implementation Steps

### Step 1: Implement Operation-Specific Methods

Add your operation's implementation in `lib/Dialect/TTNN/IR/TTNNOpModelInterface.cpp`:

```cpp
//===----------------------------------------------------------------------===//
// YourOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints>
YourOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  // You can extract all input tensors' layouts from `inputs`.
  // Other configurations can also be extracted from `opConfig`.
  // All inputs/attrs can be extracted from YourOp's member functions.
  // This layer is usually a wrapper to extract the op's necessary inputs/attrs
  // and pass those information to TTNNOpModel.h.
  return opConstraintsCache().getOrCompute(
      op_model::OpModel<YourOp>::getOpConstraints, *this,
      deviceGrid, /* other parameters */);
}

llvm::Expected<size_t>
YourOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  // Similar to the previous function.
  return opRuntimeCache().getOrCompute(
      op_model::OpModel<YourOp>::getOpRuntime, *this,
      /* other parameters */);
}
```

Note: The codebase provides several template helpers for common operation patterns:

#### Unary Operations
```cpp
// For simple unary operations (like ReluOp, SqrtOp, etc.)
return detail::getUnaryOpConstraints(*this, inputs, opConfig);
return detail::getUnaryOpRuntime(*this, inputs, opConfig);
```

#### Binary Operations
```cpp
// For binary element-wise operations (like AddOp, MultiplyOp, etc.)
return detail::getBinaryOpConstraints(*this, inputs, opConfig);
return detail::getBinaryOpRuntime(*this, inputs, opConfig);
```

#### Ternary Operations
```cpp
// For ternary operations (like WhereOp)
return detail::getTernaryOpConstraints(*this, inputs, opConfig);
return detail::getTernaryOpRuntime(*this, inputs, opConfig);
```

#### Reduction Operations
```cpp
// For reduction operations (like SumOp, MeanOp, etc.)
return detail::getReductionOpConstraints(*this, inputs, opConfig);
return detail::getReductionOpRuntime(*this, inputs, opConfig);
```

### Step 2: Add Core Model Implementation

Add the core implementation in `include/ttmlir/OpModel/TTNN/TTNNOpModel.h`:

```cpp
template <>
struct OpModel<YourOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   // ... operation-specific parameters ...
                   TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(// ... operation-specific parameters  ...
               TTNNLayoutAttr outputLayout);
};
```

And the corresponding implementation in `lib/OpModel/TTNN/TTNNOpModel.cpp`:

```cpp
llvm::Expected<OpConstraints>
OpModel<YourOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid,
    // operation-specific parameters
    TTNNLayoutAttr outputLayout) {
  #ifdef TTMLIR_ENABLE_OPMODEL
  // 1. Perform necessary conversions, create Tensor objects, etc.

  // 2. Create query closure
  // Here the ultimate goal is to enable the optimizer to call the
  // invoke method of the op in tt-metal. This is achieved through
  // creating a lambda that calls `query_op_constraints` which
  // receives 3 arguments:
  //   1. An op (eg. ::ttnn::yourOp). This is the op's backend
  //      found under tt-metal/src/tt-metal/ttnn/. The op usually
  //      has an 'invoke' method.
  //   2. The device,
  //   3. A variadic number of inputs that are converted to match
  //      the metal's definitions. The order and the types of these
  //      inputs are expected to match the invoke function of the
  //      op in metal.
  auto yourOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::yourOp, device, /* other converted parameters */);
  };

  // 3. Call getOpConstraints and pass the callable.
  return operation::getOpConstraints(getContext(), deviceGrid,
                                     yourOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<YourOp>::getOpRuntime(
    // operation-specific parameters
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // Similar to the previous function.
  // Create query closure
  auto yourOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::yourOp, device, /* other converted parameters */);
  };

  return operation::getOpRuntime(yourOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}
```
Note: If the op's definition cannot be found by `gcc` you might need to `#include` the
related header file in `OpModel/TTNN/MetalHeaders.h`.

Note: The codebase provides several implementations for common operation patterns which
is done through [Explicit template instantiation](https://en.cppreference.com/w/cpp/language/class_template.html).

#### Unary Operations
```cpp
// For simple unary operations (like ReluOp, SqrtOp, etc.)
template struct UnaryEltwiseOpModel</* Op */>;
```

#### Binary Operations
```cpp
// For binary element-wise operations (like AddOp, MultiplyOp, etc.)
template struct BinaryEltwiseOpModel</* Op */>;
```

#### Ternary Operations
```cpp
// For ternary operations (like WhereOp)
template struct TernaryEltwiseOpModel</* Op */>;
```

#### Reduction Operations
```cpp
// For reduction operations (like SumOp, MeanOp, etc.)
template struct ReductionOpModel</* Op */>;
```

### Step 3: Add Unit Tests

Create tests in `test/unittests/OpModel/TTNN/Op/TestOpModelInterface.cpp`:

```cpp
TEST_F(OpModelBase, YourOpInterface) {
  // Create input tensors
  auto input = createEmptyTensor({32, 64}, ttcore::DataType::Float32);

  // Create operation
  auto yourOp = builder.create<YourOp>(
      loc, createRankedTensorType({32, 64}, ttcore::DataType::Float32),
      input, /* other parameters */);

  // Test constraints
  auto constraintsExp = getOpConstraints(yourOp.getOperation());
  if (constraintsExp) {
      auto l1 = constraintsExp.get();
      const auto &[cbSize, l1PeakSize, totalPeakSize, outputSize, outputLayout] = l1;
      EXPECT_EQ(cbSize, /* some expected value */);
      EXPECT_EQ(l1PeakSize, /* some expected value */);
      EXPECT_EQ(totalPeakSize, /* some expected value */);
      EXPECT_EQ(outputSize, /* some expected value */);
  } else {
      FAIL() << "Missing L1 constraints; Error="
          << llvm::toString(constraintsExp.takeError()) << std::endl;
  }
  auto runtimeExp = getOpRuntime(yourOp.getOperation());
  if (runtimeExp) {
      EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
      FAIL() << llvm::toString(runtimeExp.takeError());
  }
}
```

### Step 4: Add Integration Tests

Create comprehensive tests in `test/unittests/OpModel/TTNN/Lib/TestOpModelLib.cpp`.
The following is one way of doing this, not the only possible test.

Note: For operations with additional parameters (like kernel size, stride, etc.),
add them between the input and output tensors in the tuple definition and destructuring assignment.

```cpp
template <typename OpTy>
class OpModelYourOpParam : public OpModelTest,
                           public ::testing::WithParamInterface<
                               std::tuple<detail::TestTensor, // input
                                          detail::TestTensor, // output
                                          detail::ExpectedResult>> {
protected:
  void RunTest() {
    auto [inputTensor, outputTensor, expectedResult] = GetParam();

    // Create tensors with specified layouts
    TTNNLayoutAttr inputLayout = createLayout(inputTensor);
    TTNNLayoutAttr outputLayout = createLayout(outputTensor);

    auto constraintsExp = OpModel<OpTy>::getOpConstraints(
        CreateWorkerGrid(), /* pass the params according to TTNNOpModel.h interface */, outputLayout);
    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedResult.expectedLegal);
    if (expectedResult.expectedLegal) {
      const auto [cbSize, l1PeakSize, totalPeakSize, outputSize, outputLayout] =
          constraintsExp.get();
      EXPECT_EQ(cbSize, expectedResult.expectedCbSize);
      EXPECT_EQ(l1PeakSize, expectedResult.expectedL1PeakSize);
      EXPECT_EQ(totalPeakSize, expectedResult.expectedTotalPeakSize);
      EXPECT_EQ(outputSize, expectedResult.expectedOutputSize);
    } else {
      // Must clean up the error
      llvm::consumeError(constraintsExp.takeError());
    }

    auto runtimeExp =
        OpModel<OpTy>::getOpRuntime(/* pass the params according to TTNNOpModel.h interface */, outputLayout);
    EXPECT_EQ(static_cast<bool>(runtimeExp), expectedResult.expectedLegal);
    if (expectedResult.expectedLegal) {
      EXPECT_TRUE(runtimeExp.get() > 0);
    } else {
      llvm::consumeError(runtimeExp.takeError());
    }
  }
};

using OpModelYourOpParamTest = OpModelYourOpParam<YourOp>;
TEST_P(OpModelYourOpParamTest, YourOp) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(
    YourOpTests, OpModelYourOpParamTest,
    ::testing::Values(
        std::make_tuple(
            detail::TestTensor{{32, 64}, TensorMemoryLayout::INTERLEAVED, BufferType::DRAM},
            detail::TestTensor{{32, 64}, TensorMemoryLayout::INTERLEAVED, BufferType::DRAM},
            detail::ExpectedResult{true, 8192, 8192, 8192, 8192}),
        // Add more test cases...
    ));
```

## Key Considerations

### Error handling: Operations Not Supported
For operations that cannot support these APIs, use the provided error helpers in `TTNNOpModelInterface.cpp`.
We're keeping track of such ops in [this issue](https://github.com/tenstorrent/tt-mlir/issues/4392).
So please either update the issue or add comments to it.

```cpp
llvm::Expected<op_model::OpConstraints>
YourOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &opConfig) {
  return detail::issueErrorForGetOpConstraints(
      getOperation(), detail::ReasonForLackOfSupport::/*..*/);
}

llvm::Expected<size_t>
YourOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  return detail::issueErrorForGetOpRuntime(
      getOperation(), detail::ReasonForLackOfSupport::/*..*/);
}
```

Available error reasons:
- `NeedsMemoryIO`: Operation requires memory I/O during trace capture
- `MissingMetalDefinition`: Metal backend implementation is missing
- `NeedsMultiDevice`: Operation requires multi-device support
- `NoNeedForConstraintAPI`: Operation doesn't benefit from constraint analysis
- `ArchitecturalMismatch`: Mismatch in Operation's definition in metal and mlir

### Device Grid Validation

Validate the device worker grid before proceeding:

```cpp
llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
if (!check) {
  return check.takeError();
}
ttcore::GridAttr deviceGrid =
      ttcore::lookupDevice(getOperation()).getWorkerGrid();
```

### Caching

Use the provided caching mechanisms for computations:

```cpp
// For getOpConstraints:
return opConstraintsCache().getOrCompute(
    op_model::OpModel<YourOp>::getOpConstraints, *this,
    /* parameters */);
// For getOpRuntime:
return opRuntimeCache().getOrCompute(
    op_model::OpModel<YourOp>::getOpRuntime, *this,
    /* parameters */);
```

### Check Metal Backend Availability

Ensure your operation has a corresponding implementation in the tt-metal backend before implementing these APIs.
As mentioned before, the current metal header files are `#include`d in `MetalHeaders.h`. If you are adding a
TTNNOp you might want to add an `#include` statement in that file to let the c++ compiler know where/how to find
the op's definition in metal.

### Validate Input Assumptions

Always validate the number of input tensors, eg.:

```cpp
assert(inputs.size() == 2); // for a binary op
assert(inputs.size() == 3); // for a ternary op
```

## Example: Complete Implementation

Here's a complete example for a hypothetical `CustomUnaryOp`:

```cpp
// In TTNNOpModelInterface.cpp
llvm::Expected<op_model::OpConstraints>
CustomUnaryOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                                const OpConfig &opConfig) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig);
}

llvm::Expected<size_t>
CustomUnaryOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                            const OpConfig &opConfig) {
  return detail::getUnaryOpRuntime(*this, inputs, opConfig);
}

// In TTNNOpModel.h
template <>
struct OpModel<CustomUnaryOp> : UnaryEltwiseOpModel<CustomUnaryOp> {};

// In TTNNOpModel.cpp
template <typename OpTy>
auto getOpSymbol() {
  // ...
  if constexpr (std::is_same_v<OpTy, CustomUnaryOp>) {
    return ::ttnn::custom_unary_op; // metal's definition
  }
  // ...
}

// Explicit template instantiation
template struct UnaryEltwiseOpModel<CustomUnaryOp>;

// Add tests in TestOpModelInterface.cpp and TestOpModelLib.cpp
```
