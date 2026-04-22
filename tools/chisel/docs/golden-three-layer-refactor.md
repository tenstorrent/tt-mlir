# Proposal: Three-Layer Golden Function Structure

## Current state (main)

Two layers exist today:

**Layer 1 — pure golden** (`matmul_golden`): takes plain Python/torch types, shared by
TTIR. Already clean.

**Layer 2 — TTNN adapter** (`ttnn_matmul_golden`): takes unpacked tensors + raw MLIR
attribute/type objects, unpacks them, then duplicates the computation from
`matmul_golden`:

```python
# main: tools/golden/mapping.py

def matmul_golden(
    a: GoldenMapTensor,
    b: GoldenMapTensor,
    transpose_a=False,
    transpose_b=False,
) -> GoldenMapTensor:
    a = torch.transpose(a, -2, -1) if transpose_a else a
    b = torch.transpose(b, -2, -1) if transpose_b else b
    return torch.matmul(a, b)


def ttnn_matmul_golden(
    input_tensor: GoldenMapTensor,
    other_tensor: GoldenMapTensor,
    transpose_a_attr: BoolAttr,       # raw MLIR type
    transpose_b_attr: BoolAttr,       # raw MLIR type
    output_type_mlir: Type,           # raw MLIR type
) -> GoldenMapTensor:
    transpose_a = unpack_mlir_attr(transpose_a_attr)
    transpose_b = unpack_mlir_attr(transpose_b_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    a = torch.transpose(input_tensor, -2, -1) if transpose_a else input_tensor
    b = torch.transpose(other_tensor, -2, -1) if transpose_b else other_tensor
    return torch.matmul(a, b).to(output_dtype)  # reimplements matmul_golden


GOLDEN_MAPPINGS = {
    ttir.MatmulOp: matmul_golden,       # caller unpacks attrs externally
    ttnn.MatmulOp: ttnn_matmul_golden,  # takes raw MLIR types
    ...
}
```

**Problems:**
- `ttnn_matmul_golden` reimplements the computation already in `matmul_golden`
- The TTNN function signature mixes tensors and raw MLIR objects — awkward to call
  and test
- Callers of TTIR mappings unpack attrs externally; TTNN callers pass MLIR objects in
  — inconsistent calling convention across the two dicts

---

## Proposed structure: 3 layers per op

```python
# Layer 1: pure golden — plain Python/torch types only, no mlir.ir imports
def matmul_golden(
    a: torch.Tensor,
    b: torch.Tensor,
    transpose_a: bool = False,
    transpose_b: bool = False,
    output_dtype: torch.dtype = None,
) -> torch.Tensor:
    a = torch.transpose(a, -2, -1) if transpose_a else a
    b = torch.transpose(b, -2, -1) if transpose_b else b
    result = torch.matmul(a, b)
    return result.to(output_dtype) if output_dtype else result


# Layer 2: TTNN adapter — owns all MLIR unpacking, calls golden
def ttnn_matmul(op: Operation, inputs: dict, asm_state) -> GoldenMapTensor:
    a = inputs[op.operands[0].get_name(asm_state)]
    b = inputs[op.operands[1].get_name(asm_state)]
    return matmul_golden(
        a, b,
        transpose_a=unpack_mlir_attr(op.attributes["transpose_a"]),
        transpose_b=unpack_mlir_attr(op.attributes["transpose_b"]),
        output_dtype=mlir_type_to_torch_dtype(op.results[0].type.element_type),
    )


# Layer 3: TTIR adapter — same uniform signature, different attrs if needed
def ttir_matmul(op: Operation, inputs: dict, asm_state) -> GoldenMapTensor:
    a = inputs[op.operands[0].get_name(asm_state)]
    b = inputs[op.operands[1].get_name(asm_state)]
    return matmul_golden(
        a, b,
        transpose_a=unpack_mlir_attr(op.attributes["transpose_a"]),
        transpose_b=unpack_mlir_attr(op.attributes["transpose_b"]),
    )


TTNN_GOLDEN_MAPPINGS = {
    ttnn.MatmulOp: ttnn_matmul,
    ...
}

TTIR_GOLDEN_MAPPINGS = {
    ttir.MatmulOp: ttir_matmul,
    ...
}
```

---

## What changes

| Before (main) | After |
|---------------|-------|
| `matmul_golden` — pure, TTIR maps here; callers unpack attrs externally | `matmul_golden` — pure, no MLIR types, accepts `output_dtype` |
| `ttnn_matmul_golden` — takes raw MLIR types + reimplements computation | deleted — replaced by `ttnn_matmul` adapter that calls `matmul_golden` |
| Single `GOLDEN_MAPPINGS` dict, mixed calling conventions | `TTNN_GOLDEN_MAPPINGS` and `TTIR_GOLDEN_MAPPINGS`, uniform `(op, inputs, asm_state)` |

**Key invariant**: `matmul_golden` never imports or touches anything from `mlir.ir`.
All MLIR unpacking lives in the adapter layer.

---

## Effect on builder code

The win is in the `matmul_parser()` path, where the builder already has an existing
`old_op`. Today it passes raw MLIR attrs directly to `ttnn_matmul_golden`, which
unpacks them. After the refactor it just calls the adapter — no unpacking in the
builder at all:

```python
# Before: ttnn_builder.matmul_parser()
golden_output = op_golden_function(
    input0, input1,
    old_op.transpose_a,    # raw BoolAttr passed in, unpacked inside
    old_op.transpose_b,
    result.element_type,
)

# After: ttnn_builder.matmul_parser()
golden_output = ttnn_matmul(old_op, inputs={
    old_op.operands[0]: input0,
    old_op.operands[1]: input1,
})
```

Same for the TTIR parser — it currently unpacks manually before calling `matmul_golden`.
After the refactor it just calls `ttir_matmul(old_op, inputs={...})`.

Note: the `matmul()` method (not parser) is unaffected — the op doesn't exist yet when
the golden runs there (golden output shape is needed to create the result tensor before
the op is constructed), so it calls `matmul_golden` directly with the plain Python bools
it already has.

---

## How Chisel uses the adapters

Chisel's `chisel_post_op_callback` already builds `_stashed_inputs` as a dict keyed
by SSA name strings — exactly what the adapter expects. The callback just looks up the
adapter from `TTNN_GOLDEN_MAPPINGS` and calls it:

```python
# tools/chisel/chisel/callbacks.py — chisel_post_op_callback (future)

ctx = ChiselContext.get_instance()
op = ctx._current_op
asm_state = ctx.ir_module.get_asm_state(ctx._current_program_name)

adapter = TTNN_GOLDEN_MAPPINGS.get(type(op))
if adapter is not None:
    golden_output = adapter(op, ctx._stashed_inputs, asm_state)
    # compare golden_output vs device_tensor (PCC, atol, rtol)
```

`ctx._stashed_inputs` is `{ssa_name: torch.Tensor}`, built in `chisel_pre_op_callback`
by iterating `op.operands` and calling `mlir_input.get_name(asm_state)`. The adapter
resolves tensors the same way — `inputs[op.operands[i].get_name(asm_state)]` — so no
translation is needed between what Chisel stashes and what the adapter consumes.

---

## Benefits summary

- **No duplication**: one computation in `matmul_golden`, adapters only unpack
- **Uniform calling convention**: every mapping entry is `(op, inputs, asm_state)` —
  no special-casing per dialect
- **Builder parser simplified**: `matmul_parser()` calls the adapter directly — no
  manual attr unpacking in the builder
- **Testable in isolation**: `matmul_golden` unit-testable with plain tensors, no MLIR
  setup required
