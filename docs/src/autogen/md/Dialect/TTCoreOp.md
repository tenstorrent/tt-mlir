# TTCore Dialect Ops

Auto-generated reference of operations in the `TTCore` dialect (12 ops).

## `ttcore.composite`

A named composite operation with an inlineable decomposition.

A composite operation that carries a named fused pattern along with a
reference to a decomposition function. During lowering, the composite is
either promoted to a typed backend op (if validation succeeds) or its
decomposition body is inlined as a fallback.

Example:
```mlir
%indices, %weights = ttcore.composite "topk_router_gpt"
    (%input, %weight, %bias)
    decomposition(@topk_router_gpt_decomp)
    composite_attributes = {k = 4 : i32, num_experts = 128 : i32}
    : (tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<32x128xbf16>)
      -> (tensor<32x4xui16>, tensor<32x4xbf16>)

func.func private @topk_router_gpt_decomp(
    %arg0: tensor<32x64xbf16>,
    %arg1: tensor<64x128xbf16>,
    %arg2: tensor<32x128xbf16>)
    -> (tensor<32x4xui16>, tensor<32x4xbf16>) {
  // ... sequence of primitive ops ...
}
```

## `ttcore.cpu_module`

Module-wrapper operation for CPU ops

Custom module operation that can a single ModuleOp, which should contain all funcs which should be run on CPU.

Example:
```mlir
ttcore.cpu_module {
  module {
    func.func foo() { ... }
  }
}
```

## `ttcore.device`

Named device

## `ttcore.device_module`

Module-wrapper operation for device ops

Custom module operation that can a single ModuleOp, which should contain all funcs which should be run on device.

Example:
```mlir
ttcore.device_module {
  module {
    func.func foo() { ... }
  }
}
```

## `ttcore.get_global`

Named global

Retrieves a named global value declared with `ttcore.global`

## `ttcore.get_key_value`

Get dictionary value

Retrieves the value(s) stored under the given key from the dictionary.
The key can either be a string or an index.

Example:
```mlir
%0 = ttcore.get_key_value %dict[3 : index] : (!ttcore.dict) -> tensor<32x32xbf16>
%0, %1 = ttcore.get_key_value %dict["key"]
    : (!ttcore.dict) -> (tensor<32x32xbf16>, tensor<64x64xbf16>)
```

## `ttcore.get_tuple_element`

GetTupleElement operation

Extracts element at `index` position of the `operand` tuple and produces a `result`.

Example:
```mlir
%result = ttcore.get_tuple_element %operand[0] : (tuple<tensor<32x32xbf16>, tensor<1x32xf32>>) -> tensor<32x32xbf16>
```

## `ttcore.global`

Named global

Declares a global variable with an optional index.

## `ttcore.load_cached`

Load cached results from a previously computed function

The load_cached operation calls a precomputed function with given arguments
and returns its results. This is typically used to load constant or hoisted
computation results.

Example:
```mlir
%0, %1, %2 = "ttcore.load_cached"(@forward_const_eval_1, [%arg0, %arg2])
```

## `ttcore.optimization_barrier`

Optimization barrier operation.

The `optimization_barrier` operation prevents compiler optimizations from reordering or eliminating
the values passed through it. It acts as a barrier for optimization passes.

Inputs:
- `inputs` (Variadic): Values of tensor type.

Outputs:
- `results` (Variadic): Same values as inputs, passed through unchanged.

## `ttcore.set_key_value`

Set or update dictionary value

Sets or updates a value(s) for the given key in the dictionary.
The key can either be a string or an index.

Example:
```mlir
ttcore.set_key_value %dict[3 : index] = %0
    : !ttcore.dict, tensor<32x32xbf16>
ttcore.set_key_value %dict["key"] = %0, %1
    : !ttcore.dict, tensor<32x32xbf16>, tensor<64x64xbf16>
```

## `ttcore.tuple`

Tuple operation

Produces a `result` tuple from operands `operands`.

Example:
```mlir
%result = ttcore.tuple %operand0, %operand1 : tuple<tensor<32xbf16, tensor<1x32xf32>>
```

