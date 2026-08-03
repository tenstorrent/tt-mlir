# Adding OpConstraints and OpRuntime APIs to TTNN Operations

## Overview

The TTNN Op Model Interface provides two key APIs for analyzing and optimizing operations:

- **`getOpConstraints`**: Returns constraint information including memory requirements, layout compatibility, and operation feasibility
- **`getOpRuntime`**: Returns performance metrics including execution time estimates

`getOpConstraints` has a single entry point per layer whose trailing argument
selects one of two *flavours*: a cached, nothing-is-allocated **stateless**
query, or an allocator-backed **stateful** query that evaluates the op on top of
the allocations already live in L1. See
[The Stateful (Build-From-Records) Path](#the-stateful-build-from-records-path).

These APIs enable the compiler to make informed decisions about operation placement, memory allocation, and performance optimization.

This guide walks you through best practices for implementing these APIs. It will cover the following steps:

1. [Architecture](#architecture)
2. [Implementation Steps](#implementation-steps)
   - [Step 1: Implement Operation-Specific Methods](#step-1-implement-operation-specific-methods)
   - [Step 2: Add Core Model Implementation](#step-2-add-core-model-implementation)
   - [Step 3: Add Unit Tests](#step-3-add-unit-tests)
   - [Step 4: Add Integration Tests](#step-4-add-integration-tests)
3. [The Stateful (Build-From-Records) Path](#the-stateful-build-from-records-path)
4. [Key Considerations](#key-considerations)
5. [Example: Complete Implementation](#example-complete-implementation)


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

Add your operation's implementation in
`lib/Dialect/TTNN/Interfaces/TTNNOpModelInterface.cpp`:

```cpp
//===----------------------------------------------------------------------===//
// YourOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<op_model::OpConstraints> YourOp::getOpConstraints(
    const std::vector<TTNNLayoutAttr> &inputs, const OpConfig &opConfig,
    std::optional<llvm::ArrayRef<op_model::OpModelAllocationRecord>>
        liveRecords) {
  // You can extract all input tensors' layouts from `inputs`.
  // Other configurations can also be extracted from `opConfig`.
  // All inputs/attrs can be extracted from YourOp's member functions.
  // This layer is usually a wrapper to extract the op's necessary inputs/attrs
  // and pass those information to TTNNOpModel.h.
  //
  // `liveRecords` is threaded straight through, untouched and uninspected:
  // detail::constraintsDispatch owns the stateless-vs-stateful decision.
  return detail::constraintsDispatch(*this, liveRecords,
                                     /* op-model query parameters */);
}

llvm::Expected<size_t>
YourOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const OpConfig &opConfig) {
  // Similar to the previous function; runtime queries are always stateless.
  return opRuntimeCache().getOrCompute(
      op_model::OpModel<YourOp>::getOpRuntime, *this,
      /* other parameters */);
}
```

The signature is fixed by the single `getOpConstraints` interface method in
`include/ttmlir/Dialect/TTNN/Interfaces/TTNNOpModelInterface.td`, which ODS
declares inside every non-`OpModelExempt` op class. A new op must define exactly
that 3-argument form — a 2-argument definition will not bind. (A 2-argument
convenience overload exists, but only on the `OpModel` *interface handle*, for
callers that want a plain stateless query.)

#### Why `constraintsDispatch` and not a direct cached call

`detail::constraintsDispatch` (defined near the top of
`TTNNOpModelInterface.cpp`) is the single choke point for the cache decision:

- A **stateful** query (engaged `liveRecords`) must **bypass** the constraint
  cache. The cache key is the op plus its layout/shape arguments; the live
  allocation set is deliberately *not* part of it, so a cached stateful result
  would be a fit decision taken under a different live set — a stale-fit bug.
- The op-model entry's `initialState` parameter is *defaulted*, and a default
  argument does not survive the function-pointer conversion that
  `getOrCompute`'s `is_invocable` guard performs. The dispatcher's stateless
  branch therefore binds `/*initialState=*/nullptr` inside a lambda, which also
  keeps the hashed cache key at exactly the query arguments.

Note: The codebase provides several template helpers for common operation
patterns. Each takes and forwards `liveRecords`:

#### Unary Operations
```cpp
// For simple unary operations (like ReluOp, SqrtOp, etc.)
return detail::getUnaryOpConstraints(*this, inputs, opConfig, liveRecords);
return detail::getUnaryOpRuntime(*this, inputs, opConfig);
```

#### Binary Operations
```cpp
// For binary element-wise operations (like AddOp, MultiplyOp, etc.)
return detail::getBinaryOpConstraints(*this, inputs, opConfig, liveRecords);
return detail::getBinaryOpRuntime(*this, inputs, opConfig);
```

#### Ternary Operations
```cpp
// For ternary operations (like WhereOp)
return detail::getTernaryOpConstraints(*this, inputs, opConfig, liveRecords);
return detail::getTernaryOpRuntime(*this, inputs, opConfig);
```

#### Reduction Operations
```cpp
// For reduction operations (like SumOp, MeanOp, etc.)
return detail::getReductionOpConstraints(*this, inputs, opConfig, liveRecords);
return detail::getReductionOpRuntime(*this, inputs, opConfig);
```

### Step 2: Add Core Model Implementation

Add the core implementation in `include/ttmlir/OpModel/TTNN/TTNNOpModel.h`:

```cpp
template <>
struct OpModel<YourOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(// ... operation-specific parameters ...
                   TTNNLayoutAttr outputLayout,
                   const MockAllocatorState *initialState = nullptr);

  static llvm::Expected<size_t>
  getOpRuntime(// ... operation-specific parameters  ...
               TTNNLayoutAttr outputLayout);
};
```

The trailing `initialState` is the op-model-layer half of the selector, and it
is *defaulted*: `nullptr` means the stateless query, non-null means the stateful
one. Because it is defaulted, existing call sites that pass no state keep
compiling and get the stateless behaviour. The convention is spelled out in the
comment block above `template <typename OpTy> struct OpModel;` in
`TTNNOpModel.h`.

And the corresponding implementation in `lib/OpModel/TTNN/TTNNOpModel.cpp`:

```cpp
llvm::Expected<OpConstraints>
OpModel<YourOp>::getOpConstraints(
    // operation-specific parameters
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
  #ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // 1. Convert inputs to TensorSpecs using ASSIGN_OR_RETURN.
  //    This macro unwraps llvm::Expected and returns the error
  //    on failure, avoiding repetitive error-checking boilerplate.
  ASSIGN_OR_RETURN(::tt::tt_metal::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // 2. Convert the raw state pointer to the optional tt-metal wants.
  //    tt-mlir carries the selector as a raw pointer so the default
  //    argument works and the type can stay incomplete in the header.
  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // 3. Create query closure
  // Here the ultimate goal is to enable the optimizer to call the
  // invoke method of the op in tt-metal. This is achieved through
  // creating a lambda that calls
  // `query_op_constraints_with_optional_state` which receives:
  //   1. An op (eg. ::ttnn::yourOp). This is the op's backend
  //      found under tt-metal/src/tt-metal/ttnn/. The op usually
  //      has an 'invoke' method.
  //   2. The device,
  //   3. The optional allocator state. tt-metal dispatches on it:
  //      nullopt -> the stateless query, a value -> the
  //      build-from-records query.
  //   4. A variadic number of inputs that are converted to match
  //      the metal's definitions. The order and the types of these
  //      inputs are expected to match the invoke function of the
  //      op in metal.
  auto yourOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::yourOp, device, initialStateOpt, inputSpec,
        /* other converted parameters */);
  };

  // 4. Unwrap the response. Use getOpConstraintsWithState for the
  //    state-carrying QueryOutput shape returned by the macro above.
  //    (Stateless-only families that still call QUERY_OP_CONSTRAINTS
  //    use operation::getOpConstraints instead; the two differ only in
  //    the tt-metal response shape they unwrap.)
  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              yourOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<YourOp>::getOpRuntime(
    // operation-specific parameters
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(::tt::tt_metal::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  auto yourOpQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::yourOp, device, inputSpec,
        /* other converted parameters */);
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
      const auto &[cbSize, l1PeakSize, totalPeakSize, outputSize, outputLayouts,
                   outputAllocations] = l1;
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

    // No state argument: the defaulted nullptr makes this a stateless query.
    auto constraintsExp = OpModel<OpTy>::getOpConstraints(
        /* pass the params according to TTNNOpModel.h interface */, outputLayout);
    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedResult.expectedLegal);
    if (expectedResult.expectedLegal) {
      const auto [cbSize, l1PeakSize, totalPeakSize, outputSize, outputLayouts,
                  outputAllocations] = constraintsExp.get();
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

## The Stateful (Build-From-Records) Path

There is exactly **one** `getOpConstraints` per layer. Which behaviour you get is
decided entirely by whether the caller passed allocator state:

```
                       ReluOp::getOpConstraints(inputs, config, liveRecords)
                                        ▼
                       detail::getUnaryOpConstraints(...)
                                        ▼
                       detail::constraintsDispatch(op, liveRecords, args...)
                                        │
              ┌─────────────────────────┴─────────────────────────┐
      liveRecords == nullopt                            liveRecords engaged
              │                                                   │
   opConstraintsCache().getOrCompute                 buildInitialState(*liveRecords)
   (lambda binds initialState = nullptr)             → non-null MockAllocatorState
              │                                       (cache is BYPASSED)
              └───────────────► OpModel<ReluOp>::getOpConstraints(
                                    ..., const MockAllocatorState *initialState)
                                        ▼
                    query_op_constraints_with_optional_state(...)
```

### The selector is tri-state

`liveRecords` is a `std::optional<llvm::ArrayRef<OpModelAllocationRecord>>`, and
the dispatcher tests **the optional's engagement, never its size**:

| `liveRecords` | flavour | cached? | tt-metal run mode | `outputAllocations` | peak-byte check in validation |
|---|---|---|---|---|---|
| `std::nullopt` | stateless | **yes** | `NO_DISPATCH` only | empty | **enforced** |
| engaged, **empty** | **stateful** | no | phase-1 `NO_DISPATCH` + phase-2 `NORMAL` | populated | skipped |
| engaged, non-empty | stateful | no | phase-1 `NO_DISPATCH` + phase-2 `NORMAL` | populated | skipped |

Rows 2 and 3 are the same row. **Never key the selector on
`liveRecords.empty()`.** The L1 spill pass's record set starts out empty, and the
only producer of records is a stateful query. If an empty-but-engaged live set
were treated as stateless, the first op of every run would report no
allocations, so the second op would also see an empty set, and so on — the
feature would degrade to stateless permanently and silently, with every test
still passing. This is why `op_model::buildInitialState({})` deliberately
returns a **non-null** state, and why the same contract is restated in the
`.td` method description, in `constraintsDispatch`, in `TTNNOpModel.h`/`.cpp`,
and on the stateful `validateOperation` overload.

### Threading `liveRecords` vs. passing it through unnamed

When adding an op, pick one of two shapes:

**Stateful** — the op has a real op-model query, so give its op-model entry a
trailing `const MockAllocatorState *initialState = nullptr` and thread the
caller's records straight through. `ReluOp` is the canonical example
(`TTNNOpModelInterface.cpp`):

```cpp
llvm::Expected<op_model::OpConstraints> ReluOp::getOpConstraints(
    const std::vector<TTNNLayoutAttr> &inputs, const OpConfig &opConfig,
    std::optional<llvm::ArrayRef<op_model::OpModelAllocationRecord>>
        liveRecords) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig, liveRecords);
}
```

**Intentionally stateless** — the op is a creation/bookkeeping op with no L1
buffer worth tracking, or its op-model backend has no `initialState` parameter
yet. Take the parameter **unnamed** and call the cache directly:

```cpp
llvm::Expected<op_model::OpConstraints> EmptyOp::getOpConstraints(
    const std::vector<TTNNLayoutAttr> &inputs, const OpConfig &opConfig,
    std::optional<
        llvm::ArrayRef<op_model::OpModelAllocationRecord>> /*liveRecords*/) {
  // ...
  return opConstraintsCache().getOrCompute(
      op_model::OpModel<EmptyOp>::getOpConstraints, *this,
      /* parameters */);
}
```

The unnamed parameter is the convention that says *this op deliberately discards
the caller's state*. It is safe, not buggy: such an op is modeled exactly as it
was before this feature, just less fragmentation-aware if it appears in the
spill window. Because the signature no longer distinguishes the two, the policy
comment block at the top of `TTNNOpModelInterface.cpp` enumerates every
intentionally-stateless op and the reason for it — **add your op there if you
choose this shape**, so a later reader can tell "intentionally stateless" from
"accidentally forgot". `grep -n '/\*liveRecords\*/'` finds them in code.

Migrating a blocked op later means adding the `initialState` parameter at the
op-model layer first, then switching the body to `detail::constraintsDispatch`.
That changes nothing for stateless callers.

### How the L1 spill pass uses it

`MockAllocatorL1Tracker::validate` (`lib/Dialect/TTNN/Analysis/L1SpillManagement.cpp`,
driven by `GreedyL1SpillManagement.cpp`) flattens its per-value live-record map
into a flat `ArrayRef` and passes it to the stateful `validateOperation`
overload — which is what makes the query stateful, even on the first op where
the set is empty. tt-metal applies the state, allocates the outputs for real,
and the resulting `OpConstraints::outputAllocations` are stashed and then fed
back into the tracker's record set (`addTensor`), so the next op's query sees the
previous op's real addresses. Ops that produce no records simply contribute
nothing to the state.

On the validation side, `statefulQuery` is **derived** from the same optional
(`/*statefulQuery=*/liveRecords.has_value()` in
`lib/Dialect/TTNN/Validation/OpConstraintValidation.cpp`) rather than passed
alongside it, so the two can never disagree. That flag gates the scalar
peak-byte check: the stateless path enforces
`overallPeakL1Usage + additionalL1Usage <= getUsableL1PerCore`, because nothing
was allocated and that comparison is the only thing keeping the beam search off
illegal L1 layouts. The stateful path skips it — tt-metal's real allocator
decides fit/fragmentation/CB-clash, and `MockAllocatorL1Tracker` enforces the
optimizer's own (lower) byte ceiling on top. The two checks are complementary,
not duplicated.

## Key Considerations

### Operations Not Supported by OpModel

If an operation does not need OpModel support (e.g., it has no metal backend
implementation, requires multi-device support, or simply doesn't benefit from
constraint analysis), mark it with the `OpModelExempt` trait in its TableGen
definition:

```tablegen
def TTNN_YourOp : TTNN_Op<"your_op", [OpModelExempt]> {
  // ...
}
```

The `OpModelExempt` trait prevents the base op class from adding
`DeclareOpInterfaceMethods<TTNN_OpModelInterface>`, so no stub
implementation in `TTNNOpModelInterface.cpp` is needed.

We're keeping track of ops that lack OpModel support in
[this issue](https://github.com/tenstorrent/tt-mlir/issues/4392).
Please either update the issue or add comments to it when exempting an op.

### Device Grid

The worker grid is not threaded in from the IR: `operation::getOpConstraints`
/ `operation::getOpConstraintsWithState` source it from the open device
(`SingletonDeviceContext::getInstance().getComputeGridShape()`) when building
the output layouts. The two are equivalent — the system desc that produced the
IR's `DeviceAttr` is itself derived from that grid — so op-model entries take no
`deviceGrid` parameter.

### Caching

Use the provided caching mechanisms for computations:

```cpp
// For getOpConstraints: go through the dispatcher, which caches the
// stateless flavour and deliberately bypasses the cache for the stateful one.
return detail::constraintsDispatch(*this, liveRecords, /* parameters */);

// For an intentionally stateless op (unnamed liveRecords), call the cache
// directly:
return opConstraintsCache().getOrCompute(
    op_model::OpModel<YourOp>::getOpConstraints, *this,
    /* parameters */);

// For getOpRuntime:
return opRuntimeCache().getOrCompute(
    op_model::OpModel<YourOp>::getOpRuntime, *this,
    /* parameters */);
```

Stateful constraint queries are **deliberately uncached**. The cache key is the
op plus its layout/shape arguments and does *not* include the live allocation
set, so caching a stateful result would hand back a fit decision computed under
some other live set. Never add `liveRecords` to the key either — the records
change on essentially every query, so it would only pollute the cache.

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
llvm::Expected<op_model::OpConstraints> CustomUnaryOp::getOpConstraints(
    const std::vector<TTNNLayoutAttr> &inputs, const OpConfig &opConfig,
    std::optional<llvm::ArrayRef<op_model::OpModelAllocationRecord>>
        liveRecords) {
  return detail::getUnaryOpConstraints(*this, inputs, opConfig, liveRecords);
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
