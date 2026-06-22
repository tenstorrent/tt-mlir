# The TTNN OpInvoke Pattern

## Overview

When we add a TTNN op, there are two places that need to call into the actual
tt-metal op (`::ttnn::<op>`):

- the **runtime**, which executes the op on device, and
- the **OpModel**, which asks the op for its constraints and runtime estimates so
  the optimizer can make decisions.

In the past these were two separate code paths. Each one built the op's arguments
on its own and called `::ttnn::<op>` itself. They were supposed to stay identical,
but nothing forced that, so they could drift apart.

The `TTNNOpInvokeLib` fixes this. For every op there is now a single function,
`ttnn_op_invoke::call<Op>(...)`, that builds the argument list once and then either
executes the op or runs a constraint/runtime query, depending on a `CallType` argument
you pass in. Both the runtime and the OpModel call this same function.

## Adding an op

### 1. Files

Create `include/ttmlir/OpInvoke/TTNN/<Category>/<Op>Op.h` and `lib/OpInvoke/TTNN/<Category>/<Op>Op.cpp`.
Add your new `.cpp` to the `TTNNOpInvokeLib` target in `lib/OpInvoke/TTNN/CMakeLists.txt`.

### 2. The result type

The op's call can return one of three things: a constraint query response, a runtime
query response, or a return type of the tt-metal op. So the result is a variant
of all three:

```cpp
using <Op>Result =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;
```

### 3. The resolved-params struct

The op's attributes come from the NativeTable type. Before we can call
`::ttnn::<op>`, we need to convert them into the types tt-metal expects. We keep
those converted values in a struct:

```cpp
struct <Op>ResolvedParams {
    ::ttnn::DeviceComputeKernelConfig computeConfig;
};
```

### 4. The resolve params function

`resolve<Op>Params` reads the NativeTable struct, converts each attribute with the shared
`operations::utils::*` helpers if needed, and returns `<Op>ResolvedParams` struct.

```cpp
<Op>ResolvedParams
resolve<Op>Params(const ::tt::target::ttnn::<Op>OpT &<Op>op) {
  <Op>ResolvedParams params;

  params.computeConfig = operations::utils::createDeviceComputeKernelConfig(op.compute_config);

  return params;
}
```

Note: When you need the output memory config and data type, read them from the output
tensor (`op.out`) via `getTensorRefMemoryConfig` / `getDataType`, not from `memory_config` 
or `dtype` fields of NativeTable struct.

### 5. The tuple builder

The argument list for `::ttnn::<op>` is built once, in the order tt-metal expects, by
a template helper. The only thing that changes between executing and querying is
the tensor arguments: for a query we pass a `TensorSpec`, and for execution we pass a
real `::ttnn::Tensor`. We don't branch on that ourselves — we call
`resolveTensorArg(input, tag)` and let the compiler pick the right one:

```cpp
template <typename Tag>
auto create<Op>Tuple(Tag tag,
                     const ::tt::target::ttnn::TransposeOpT &op,
                     TensorArg input,
                     const TransposeResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), op.arg1, params.computeConfig);
}
```

### 6. Call function

Finally, `call<Op>` is the public entry point that ties everything together. It resolves the
params, wraps the tuple builder in a `makeTuple` lambda, and hands everything to `callOp` 
which is a templated function that switches on `CallType`.

```cpp
<Op>Result
call<Op>(CallType callType,
         const ::tt::target::ttnn::<Op>T &op,
         TensorArg input, ::ttnn::MeshDevice *device) {
  <Op>ResolvedParams params = resolve<Op>Params(op);

  auto makeTuple = [&](auto tag) {
    return create<Op>Tuple(tag, op, input, params);
  };

  return callOp<<Op>Result>(WRAP_OP(::ttnn::<op>), callType,
                            makeTuple, device);
}
```

Note: Some ops don't have a constraint or runtime query in tt-metal. `callOp` has two boolean 
template flags for this, so set them to false if query isn't available.

```cpp
template <typename Result, bool ConstraintsImplemented = true,
          bool RuntimeImplemented = true>
Result callOp(...);
```

### 7. Testing: OpTPathParity

Write a `TEST_P` in `test/unittests/MLIRAttrToFBNative/OpTPathParity.cpp` that 
builds the NativeTable struct both ways — path A through `buildOpTFromMLIR`,
path B by `createOp` → `UnPackTo` — then `EXPECT_EQ` them. Use a
`resetUnusedFields` helper to clear the fields that are allowed to differ, 
such as the input tensor refs.