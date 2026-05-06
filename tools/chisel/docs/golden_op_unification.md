# Golden Op Dispatch Unification

## Problem

Golden function calls are scattered across `ttir_builder.py` (~200+ call sites)
and `ttnn_builder.py` (~200+ call sites) with manual per-op argument assembly.
Every op method independently:

1. Extracts golden tensors via `_get_golden_tensor()`
2. Looks up the golden function via `get_golden_function()`
3. Manually assembles positional args (tensors, attrs, output_type)
4. Calls the golden function
5. Stores the result via `_set_golden_tensor()`

This creates massive duplication and makes it impossible for Chisel to reuse
the same dispatch logic. The goal is a single generic `dispatch_golden()`
function that both builders and Chisel can share.

## Current State

### TTIR Builder (`tools/builder/ttir/ttir_builder.py`)

**Two paths:**

1. **`_op_proxy()` (lines 59-155)** — generic dispatcher used by ~10-15% of ops
   (softmax, matmul, transpose, embedding, etc.). Accepts `organize_golden_args`,
   `golden_kwargs`, `ttir_kwargs`, `skip_golden`.

2. **Direct manual calls (~85-90% of ops)** — each `@tag` and `@parse` method
   manually assembles the golden call. Pattern:

```python
# @tag method (e.g., log, lines 5886-5923)
input0 = self._get_golden_tensor(in0)
op_golden_function = get_golden_function(ttir_op)
golden_output = op_golden_function(input0, mlir_output_type)

# @parse method (e.g., log_parser, lines 5925-5949)
input0 = self._get_golden_tensor(in0)
op_golden_function = get_golden_function(ttir_op)
golden_output = op_golden_function(input0, result.element_type)
```

### TTNN Builder (`tools/builder/ttnn/ttnn_builder.py`)

**Same two paths:**

1. **`_op_proxy()` (lines 65-156)** — identical structure to TTIR's version.

2. **Direct manual calls** — same pattern, example:

```python
# add @tag (lines 282-326)
input0 = self._get_golden_tensor(in0)
input1 = self._get_golden_tensor(in1)
op_golden_function = get_golden_function(ttnn_op)
golden_output = op_golden_function(input0, input1, mlir_output_type)
```

### Golden Function Signature Categories

| Category | Example | Signature |
|----------|---------|-----------|
| Unary | `ttir_abs_golden` | `(input, output_type_mlir)` |
| Binary | `ttir_add_golden` | `(input0, input1, output_type_mlir)` |
| Reduction | `ttir_sum_golden` | `(input, dim_attr, keep_dim_attr, output_type_mlir)` |
| Multi-attr | `conv2d_golden` | `(input, weight, bias, stride, padding, dilation, groups, ...)` |
| Variadic | `ttir_concat_golden` | `(input_tensors: List, dim_attr, output_type_mlir)` |
| Kwargs | `softmax_golden` | `(input, **kwargs)` |
| Full result type | `ttir_to_layout_golden` | `(input, output_ranked_tensor_type)` |

### Key Constraints

- **Attribute naming mismatch**: golden param `transpose_a_attr` vs MLIR attr
  name `transpose_a`. Some params use bare names (`epsilon`).
- **Infrastructure attrs to ignore**: `memory_config`, `compute_config`, etc.
  exist on MLIR ops but golden functions never consume them.
- **TTNN ops are NOT DPS**: operands contain only real inputs.
- **TTIR ops ARE DPS**: last operand is the output tensor — must be stripped.
- **Optional tensors**: e.g., `bias` in `LinearOp` can be `None`.
- **Special cases**: `where` op transforms condition tensor before passing;
  `slice` uses keyword arguments.

## Design

### New shared utility: `tools/golden/dispatch.py`

A new module alongside `tools/golden/mapping.py` containing `dispatch_golden()`.
Both builders and Chisel import from here.

```python
# tools/golden/dispatch.py

import inspect
from typing import get_origin, List, Optional, Callable, Any

from golden.mapping import get_golden_function, GoldenMapTensor, unpack_mlir_attr

IGNORED_ATTRS = frozenset({
    "memory_config", "dtype", "matmul_program_config",
    "activation", "compute_config", "core_grid", "layout",
    "program_config", "sub_device_id", "num_links", "topology",
    "operand_constraints",
})


def _build_attr_lookup(op):
    """
    Build dict mapping both 'attr_name' and 'attr_name_attr'
    to the raw MLIR Attribute value for each op attribute.
    """
    lookup = {}
    for attr_name in op.operation.attributes:
        if attr_name in IGNORED_ATTRS:
            continue
        val = op.operation.attributes[attr_name]
        lookup[attr_name] = val
        lookup[attr_name + "_attr"] = val
    return lookup


def dispatch_golden(
    golden_fn: Callable,
    operand_tensors: List[GoldenMapTensor],
    op,  # MLIR Operation (for attributes and result type)
) -> Any:
    """
    Generically call a golden function by matching its signature params
    to tensor operands, op attributes, and result type info.

    Parameters
    ----------
    golden_fn : Callable
        The golden reference function from GOLDEN_MAPPINGS.
    operand_tensors : List[GoldenMapTensor]
        Golden tensors for the op's tensor operands, in order.
        Caller is responsible for extracting these from the pool/builder.
    op : MLIR Operation
        The MLIR op — used to read attributes and result element type.
    """
    sig = inspect.signature(golden_fn)
    attr_lookup = _build_attr_lookup(op)

    args = []
    tensor_idx = 0

    for name, param in sig.parameters.items():
        # 1. Output element type
        if name == "output_type_mlir":
            args.append(op.results[0].type.element_type)
            continue

        # 2. Full output RankedTensorType
        if name == "output_ranked_tensor_type":
            args.append(op.results[0].type)
            continue

        # 3. **kwargs — pass remaining attrs as dict
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            # Collect any unmatched non-ignored attrs
            continue

        # 4. Attribute — name matches an MLIR attribute
        if name in attr_lookup:
            args.append(attr_lookup[name])
            continue

        # 5. Variadic tensor (List[GoldenMapTensor])
        ann = param.annotation
        if ann is not inspect.Parameter.empty and get_origin(ann) is list:
            args.append(operand_tensors[tensor_idx:])
            tensor_idx = len(operand_tensors)
            continue

        # 6. Regular tensor — consume next operand
        if tensor_idx < len(operand_tensors):
            args.append(operand_tensors[tensor_idx])
            tensor_idx += 1
        else:
            args.append(None)  # Optional tensor absent

    return golden_fn(*args)
```

### Why `operand_tensors` is caller-provided (not extracted internally)

The three consumers resolve golden tensors differently:

| Consumer | How golden tensors are obtained |
|----------|--------------------------------|
| TTIR builder `@tag` | `self._get_golden_tensor(in0)` from builder's `_goldens` dict |
| TTIR builder `@parse` | Same, but via `global_dict` mapped operands |
| TTNN builder | Same as TTIR |
| Chisel `execute_golden` | `tensor_pool[ssa_name]` keyed by SSA name via `AsmState` |

By taking `operand_tensors` as input, the dispatch function stays agnostic to
the storage mechanism.

### Special case: `where` op condition transform

The `where` golden function expects a boolean condition tensor, but the builder
passes a float tensor that must be transformed first. This transform stays in
the builder — it's an operand preparation concern, not a dispatch concern:

```python
# In the builder's where() method — BEFORE calling dispatch_golden:
condition = in0_tensor.apply_shardwise(
    lambda shard: torch.where(shard > 0, torch.tensor(True), torch.tensor(False))
)
golden_output = dispatch_golden(golden_fn, [condition, input1, input2], op)
```

### Special case: `slice` with keyword args

`ttir_slice_golden` currently uses keyword args (`begins=`, `ends=`, `step=`).
The dispatch handles this because these match MLIR attribute names that appear
in `attr_lookup`. They'll be matched as attribute params and passed positionally,
which works because `ttir_slice_golden` also accepts them positionally.

If it doesn't, we add a thin wrapper with positional params.

## Implementation Plan

### Step 1: Create `tools/golden/dispatch.py`

New file containing:
- `dispatch_golden(golden_fn, operand_tensors, op)` — core dispatch
- `_build_attr_lookup(op)` — attribute name resolution helper
- `IGNORED_ATTRS` — infrastructure attributes set

Add to `tools/golden/__init__.py` exports.

### Step 2: Add `_dispatch_golden()` helper to base `Builder`

In `tools/builder/base/builder.py`, add a method that wraps `dispatch_golden`
with builder-specific tensor resolution:

```python
def _dispatch_golden(
    self,
    op,                         # The created MLIR op (OpView)
    golden_fn: Callable,
    operands: List[Operand],    # Builder operands (for golden tensor lookup)
) -> GoldenMapTensor:
    """Call golden function generically, extracting golden tensors from builder state."""
    golden_tensors = [self._get_golden_tensor(o) for o in operands]
    return dispatch_golden(golden_fn, golden_tensors, op)
```

### Step 3: Refactor `_op_proxy()` in both builders

Replace the manual golden dispatch in `_op_proxy()` with `_dispatch_golden()`.

**Before** (current `_op_proxy`, lines 143-153):
```python
if not skip_golden:
    op_golden_function = get_golden_function(op_ttir_function, **golden_kwargs)
    if op_golden_function is not None:
        if len(inputs) == 0:
            golden_output = op_golden_function(**golden_kwargs)
        else:
            golden_output = op_golden_function(
                *(organize_golden_args(inputs)), **golden_kwargs
            )
        self._set_golden_tensor(op.result, golden_output)
```

**After**:
```python
if not skip_golden:
    op_golden_function = get_golden_function(op_ttir_function)
    if op_golden_function is not None:
        golden_output = self._dispatch_golden(op, op_golden_function, inputs)
        self._set_golden_tensor(op.result, golden_output)
```

This eliminates `organize_golden_args`, `golden_kwargs`, and the
`len(inputs) == 0` branch.

### Step 4: Migrate direct-call ops incrementally

For each op that currently does manual golden dispatch, replace with
`_dispatch_golden()`. This can be done incrementally — one category at a time.

**Before** (e.g., `log`, lines 5900-5902):
```python
input0 = self._get_golden_tensor(in0)
op_golden_function = get_golden_function(ttir_op)
golden_output = op_golden_function(input0, mlir_output_type)
```

**After**:
```python
op_golden_function = get_golden_function(ttir_op)
golden_output = self._dispatch_golden(op, op_golden_function, [in0])
```

**Migration order** (by risk, lowest first):

1. **Unary ops** (~80 in TTIR, ~30 in TTNN) — simplest, one tensor + output_type
2. **Binary ops** (~35 in TTIR, ~30 in TTNN) — two tensors + output_type
3. **Ternary ops** (~5 each) — three tensors + output_type (where, clamp_tensor)
4. **Ops with scalar attrs** (~15 each) — tensors + MLIR attrs + output_type
   (reshape, broadcast, pad, clamp_scalar, leaky_relu, repeat_interleave)
5. **Reduction ops** (~10 each) — tensors + dim/keepdim attrs + output_type
6. **Complex ops** (~10 each) — conv2d, gather, scatter, matmul, linear
7. **Variadic ops** (~2 each) — concat
8. **Special cases** — where (condition transform), to_layout (full result type),
   slice (keyword args)

### Step 5: Handle `@parse` methods

`@parse` methods pass `result.element_type` instead of `mlir_output_type`.
Since `dispatch_golden` reads from `op.results[0].type.element_type`, and the
parse method creates the new op before calling golden, this works automatically
— the new op already has the correct result type.

### Step 6: Normalize outlier golden functions

A few golden functions don't follow the standard convention. Add thin wrappers
in `tools/golden/mapping.py`:

| Function | Issue | Fix |
|----------|-------|-----|
| `rms_norm_golden` | Takes unpacked Python values | Add `ttnn_rms_norm_golden` wrapper accepting MLIR attrs |
| `softmax_golden` | Uses `**kwargs` with `dim` key | Works as-is — `dim` will be in `attr_lookup` |
| `transpose_golden` | Uses `**kwargs` with `dim0`/`dim1` | Verify MLIR attr names match; add wrapper if not |

### Step 7: Update `tools/golden/__init__.py`

```python
from .dispatch import dispatch_golden

__all__ = [
    "GoldenMapTensor",
    "unpack_mlir_attr",
    "get_golden_function",
    "dispatch_golden",
    "GOLDEN_MAPPINGS",
]
```

### Step 8: Update `tools/golden/CMakeLists.txt`

Add `dispatch.py` to the declared sources.

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `tools/golden/dispatch.py` | **Create** | `dispatch_golden()`, `_build_attr_lookup()`, `IGNORED_ATTRS` |
| `tools/golden/__init__.py` | **Modify** | Export `dispatch_golden` |
| `tools/golden/CMakeLists.txt` | **Modify** | Add `dispatch.py` to sources |
| `tools/golden/mapping.py` | **Modify** | Add wrapper functions for outlier goldens |
| `tools/builder/base/builder.py` | **Modify** | Add `_dispatch_golden()` method |
| `tools/builder/ttir/ttir_builder.py` | **Modify** | Replace manual golden calls with `_dispatch_golden()` |
| `tools/builder/ttnn/ttnn_builder.py` | **Modify** | Replace manual golden calls with `_dispatch_golden()` |

## Verification

1. **Unit tests for `dispatch_golden`** in `test/python/test_golden_dispatch.py`:
   - Unary: mock op with 1 operand, verify golden called with `(tensor, element_type)`
   - Binary: 2 operands
   - With attributes: op has MLIR attrs, verify they're passed to correct params
   - Variadic: List annotation detected, all remaining tensors packed
   - Optional: fewer operands than params, verify `None` passed
   - `**kwargs`: verify no crash
   - `_attr` suffix resolution: golden param `dim_attr` matches MLIR attr `dim`

2. **Existing builder tests**: `pytest test/python` — full regression after each
   migration batch. No golden output should change since dispatch is a refactor.

3. **Incremental migration**: migrate one op category, run tests, then next.
   Never migrate all at once.

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Attribute name collision with tensor param | Tensor param names (`input_tensor`, `other_tensor`) never match MLIR attr names (`transpose_a`, `dim`). Verified across all ~160 golden functions. |
| Missing `List` type annotation on variadic param | Only `concat_golden` needs this — already annotated. |
| `inspect.signature` performance | Called once per golden invocation. Negligible vs torch computation. Can cache if needed. |
| Breaking `@parse` methods | `dispatch_golden` reads element type from `op.results[0]` — parse methods create the op first, so result type is available. |
| `_op_proxy` callers passing custom `golden_kwargs` | These are the first ops to verify manually (softmax, clamp_scalar, quantize, rms_norm). |
