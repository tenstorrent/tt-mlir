# Populating CHISEL_GOLDEN_MAPPINGS

## Overview

`CHISEL_GOLDEN_MAPPINGS` in `tools/golden/mapping.py` maps TTNN op types to
thin wrapper functions with a uniform interface:

```python
fn(op: Operation, inputs: Dict[str, GoldenMapTensor], asm_state: AsmState) -> GoldenMapTensor
```

Each wrapper extracts the arguments that the existing `ttnn_*_golden` function
expects — tensors from `inputs` by resolving SSA names from `op.operands`,
attributes from `op.attributes`, output type from `op.results` — and delegates
to it. No golden logic is duplicated.

- `op` — the full MLIR Operation
- `inputs` — dict of SSA name to `GoldenMapTensor`
- `asm_state` — AsmState for resolving operand SSA names via
  `op.operands[i].get_name(asm_state)`

## Wrapper Pattern

Every wrapper follows the same skeleton — resolve each operand by SSA name:

```python
def chisel_ttnn_<name>(op, inputs, asm_state):
    input_tensor = inputs[op.operands[0].get_name(asm_state)]
    # extract more operands by index if needed ...
    # extract attributes from op.attributes if needed ...
    output_type = op.results[0].type.element_type
    return ttnn_<name>_golden(input_tensor, ..., output_type)
```

## Op Categories

### 1. Unary — 33 ops

One input tensor, no attributes. Wrapper just unpacks and delegates.

**Ops:** `Abs`, `Cbrt`, `Ceil`, `Cos`, `Acos`, `Erf`, `Erfc`, `Exp`, `Floor`,
`Gelu`, `IsFinite`, `Neg`, `Tan`, `Atan`, `Tanh`, `Reciprocal`, `Relu`,
`Relu6`, `Rsqrt`, `Sigmoid`, `Sign`, `Silu`, `Sin`, `Asin`, `Sqrt`, `Log`,
`Log1p`, `Expm1`, `Mish`, `LogicalNot`, `BitwiseNot`, `ToDevice`, `FromDevice`

**Example:**
```python
def chisel_ttnn_abs(op, inputs, asm_state):
    input_tensor = inputs[op.operands[0].get_name(asm_state)]
    return ttnn_abs_golden(input_tensor, op.results[0].type.element_type)
```

**Effort:** Mechanical. All 33 are identical modulo the function name.

---

### 2. Binary — 23 ops

Two input tensors, no attributes.

**Ops:** `Add`, `Atan2`, `Multiply`, `Subtract`, `Divide`, `Maximum`,
`Minimum`, `Remainder`, `PowTensor`, `Equal`, `NotEqual`, `GreaterEqual`,
`GreaterThan`, `LessEqual`, `LessThan`, `LogicalAnd`, `LogicalOr`,
`LogicalXor`, `LogicalLeftShift`, `LogicalRightShift`, `BitwiseAnd`,
`BitwiseOr`, `BitwiseXor`

**Example:**
```python
def chisel_ttnn_add(op, inputs, asm_state):
    lhs = inputs[op.operands[0].get_name(asm_state)]
    rhs = inputs[op.operands[1].get_name(asm_state)]
    return ttnn_add_golden(lhs, rhs, op.results[0].type.element_type)
```

**Effort:** Mechanical, same as unary but with two tensors.

---

### 3. Ternary — 2 ops

Three input tensors, no attributes.

**Ops:** `Where`, `ClampTensor`

**Example:**
```python
def chisel_ttnn_where(op, inputs, asm_state):
    condition = inputs[op.operands[0].get_name(asm_state)]
    x = inputs[op.operands[1].get_name(asm_state)]
    y = inputs[op.operands[2].get_name(asm_state)]
    return ttnn_where_golden(condition, x, y, op.results[0].type.element_type)
```

---

### 4. Unary + Scalar Attributes — 3 ops

One tensor plus scalar attributes extracted from `op.attributes`.

| Op | Attributes |
|----|-----------|
| `LeakyRelu` | `parameter` (FloatAttr) |
| `ClampScalar` | `min`, `max` (FloatAttr) |
| `RepeatInterleave` | `repeats`, `dim` (IntegerAttr) |

**Example:**
```python
def chisel_ttnn_leaky_relu(op, inputs, asm_state):
    input_tensor = inputs[op.operands[0].get_name(asm_state)]
    return ttnn_leaky_relu_golden(
        input_tensor, op.attributes["parameter"], op.results[0].type.element_type
    )
```

---

### 5. Binary + Attributes — 1 op

**Op:** `Matmul` — `transpose_a`, `transpose_b` (BoolAttr)

**Example:**
```python
def chisel_ttnn_matmul(op, inputs, asm_state):
    a = inputs[op.operands[0].get_name(asm_state)]
    b = inputs[op.operands[1].get_name(asm_state)]
    return ttnn_matmul_golden(
        a, b,
        op.attributes["transpose_a"], op.attributes["transpose_b"],
        op.results[0].type.element_type,
    )
```

---

### 6. Multiple Tensors + Optional Tensors + Attributes — 4 ops

Optional tensors that aren't present won't appear in `inputs`.
Use `len(inputs)` to detect which are present.

| Op | Required Tensors | Optional Tensors | Attributes |
|----|-----------------|-----------------|------------|
| `Linear` | input, weight | bias | `transpose_a`, `transpose_b` (BoolAttr) |
| `LayerNorm` | input | weight, bias | `normalized_shape` (ArrayAttr), `epsilon` (FloatAttr) |
| `LayerNormPreAllGather` | input | residual_input, recip | — |
| `LayerNormPostAllGather` | input, stats | weight, bias | `epsilon` (FloatAttr) |

**Example:**
```python
def chisel_ttnn_linear(op, inputs, asm_state):
    input_tensor = inputs[op.operands[0].get_name(asm_state)]
    weight = inputs[op.operands[1].get_name(asm_state)]
    bias = inputs.get(op.operands[2].get_name(asm_state)) if len(op.operands) > 2 else None
    return ttnn_linear_golden(
        input_tensor, weight, bias,
        op.attributes["transpose_a"], op.attributes["transpose_b"],
        op.results[0].type.element_type,
    )
```

---

### 7. Variadic List + Attributes — 2 ops

All input tensors form a list passed as the first argument.

| Op | Attributes |
|----|-----------|
| `Concat` | `dim` (IntegerAttr) |
| `Repeat` | `repeat_dims` (ShapeAttr) |

**Example:**
```python
def chisel_ttnn_concat(op, inputs, asm_state):
    tensors = [inputs[operand.get_name(asm_state)] for operand in op.operands]
    return ttnn_concat_golden(
        tensors, op.attributes["dim"], op.results[0].type.element_type
    )
```

---

### 8. Typecast — 1 op

```python
def chisel_ttnn_typecast(op, inputs, asm_state):
    input_tensor = inputs[op.operands[0].get_name(asm_state)]
    return ttnn_typecast_golden(input_tensor, op.results[0].type.element_type)
```

---

### 9. Layout — 1 op

**Op:** `ToLayout`

```python
def chisel_ttnn_to_layout(op, inputs, asm_state):
    input_tensor = inputs[op.operands[0].get_name(asm_state)]
    return ttnn_to_layout_golden(input_tensor, op.attributes["layout"], op.results[0].type.element_type)
```

---

### 10. Norm (Python-typed params) — 2 ops

| Op | Notes |
|----|-------|
| `GroupNorm` | Current golden takes untyped `num_groups` and `FloatAttr` epsilon |
| `RMSNorm` | Current golden takes Python `float`/`List[int]`, not MLIR attrs |

`GroupNorm` wrapper passes MLIR attrs directly (already accepted).
`RMSNorm` wrapper must `unpack_mlir_attr` before delegating since
`rms_norm_golden` expects plain Python types.

**Example (RMSNorm):**
```python
def chisel_ttnn_rms_norm(op, inputs, asm_state):
    input_tensor = inputs[op.operands[0].get_name(asm_state)]
    weight = inputs.get(op.operands[1].get_name(asm_state)) if len(op.operands) > 1 else None
    bias = inputs.get(op.operands[2].get_name(asm_state)) if len(op.operands) > 2 else None
    return rms_norm_golden(
        input_tensor, weight, bias,
        unpack_mlir_attr(op.attributes["normalized_shape"]),
        unpack_mlir_attr(op.attributes["epsilon"]),
    )
```

---

### 11. Collective Communication — 5 ops

Same wrapper pattern — extract attrs from `op.attributes` and delegate.

| Op | Delegate | Attributes |
|----|----------|-----------|
| `DistributeTensor` | `ttnn_distribute_tensor_golden` | mapper config |
| `AggregateTensor` | `ttnn_aggregate_tensor_golden` | composer config |
| `AllGather` | `ttnn_all_gather_golden` | `all_gather_dim`, `cluster_axis` |
| `ReduceScatter` | `ttnn_reduce_scatter_golden` | `reduce_type`, `scatter_dim`, `cluster_axis` |
| `AllReduceAsync` | `ttir_all_reduce_golden` | `reduce_type`, `cluster_axis` |

---

### 12. Index-based — 1 op

**Op:** `Gather` — delegates to `ttir_gather_dim_golden(input, index, dim, output_type)`.

---

### 13. Attention — 1 op

**Op:** `PagedFlashMultiLatentAttentionDecode` — most complex.
Multiple required and optional tensors, multiple attributes.
Wrapper unpacks by position and delegates to
`ttir_paged_flash_multi_latent_attention_decode_golden`.

---

## Summary

| Category | Count | Effort |
|----------|-------|--------|
| Unary | 33 | Mechanical |
| Binary | 23 | Mechanical |
| Ternary | 2 | Low |
| Unary + Scalar Attrs | 3 | Low |
| Binary + Attrs (Matmul) | 1 | Low |
| Optional Tensors + Attrs | 4 | Medium |
| Variadic + Attrs | 2 | Low |
| Typecast | 1 | Trivial |
| Layout | 1 | Trivial |
| Norm (special params) | 2 | Medium (RMSNorm needs unpack) |
| Collective Comm | 5 | Low (just attr forwarding) |
| Index-based (Gather) | 1 | Low |
| Attention | 1 | Medium (many optional tensors) |
| **Total** | **79** | |

## Suggested Migration Order

1. **Unary + Binary + Ternary + Typecast** (59 ops) — mechanical wrappers
2. **Layout + Scalar Attrs + Matmul + Variadic** (7 ops) — straightforward attr forwarding
3. **Optional Tensors + Norms** (6 ops) — careful positional optional handling
4. **Collective Comm + Gather** (6 ops) — attr forwarding, no logic change
5. **Attention** (1 op) — many optional tensors, do last
