# Code Review — `tools/chisel/chisel/`

## High-severity issues

### 5. `op_configs.py:81-125` — `_load_cached_post_op` duplicates ~20 lines of `_default_post_op`

The "for each output: check_mlir_vs_tensor_ref, retrieve device, check_mlir_vs_runtime_tensor, record errors" loop is reproduced verbatim. Extract a helper in `callbacks.py` or `checker.py`:

```python
def _validate_device_outputs(program_context, op_outputs, output_refs, asm_state, checker) -> list[Optional[torch.Tensor]]:
    ...
```

Then both default and `load_cached` post-op reuse it. The only difference is what they do *after*: default runs golden checks, `load_cached` records `"golden": "skipped"`.

## Minor / stylistic

### 15. Inconsistent copyright years
`utils.py`, `executor.py`, `checker.py` say 2025; other files say 2026. Normalize.

### 17. `__init__.py` over-exports
Exporting the four `chisel_*_callback` functions invites callers to bypass `bind()`. If the recommended entry point is `bind()/unbind()`, hide the callbacks:
```python
from .bind import bind, unbind
from .context import ChiselContext
__all__ = ["bind", "unbind", "ChiselContext"]
```

### 18. `bind.py:30` — `bind()` returns nothing
Callers can't mutate flags (`ctx.strict`, `ctx.isolation_check`) without a separate `ChiselContext.get_instance()` call. Return the context:
```python
def bind(*, skip_criterion=None, results_path=None, strict=False, ...) -> ChiselContext:
```

Also accept the behavior flags as kwargs rather than requiring callers to post-mutate the singleton — the post-construction mutation pattern is junior-smelling.

### 19. `context.py:125-129` — `reset_for_new_execution` is named for a thing that isn't happening
The comment says "golden_tensor_pool is intentionally NOT cleared — cross-run chaining", but `__init__` already set up `op_iter = iter(self.ops)` and `current_op = None`. The reset method is redundant with the constructor. Either have the constructor call it, or drop the duplication.

### 20. `ops.py:110-126` — `_find_function` walks the whole module
`func.FuncOp`s are top-level children of a Module. Iterate `self.module.body.operations` directly — no walker, no `WalkResult.SKIP` dance. Also, you already know `functions` at `__init__`; you could build the dict in a single pass over `module.body.operations` instead of N walks for N functions.

### 21. Local imports of `ttrt.runtime` scattered across 4 files
`callbacks.py`, `bind.py`, `utils.py`, `context.py` each do `from ttrt import runtime as tt_runtime` inside functions. If the deferred import is for import-order reasons, put the lazy accessor in one place:

```python
# utils.py
def _rt():
    from ttrt import runtime as r
    return r
```

Or (better) import at module top-level in each module and fail loudly if `ttrt` isn't available — that's the user's bug, not something to paper over.

### 22. `checker.py` — `ChiselChecker` is instantiated twice per op (once in pre, once in post)
Each construction recomputes `_op_asm`. Move it to a lazy `@cached_property` or cache on `ProgramState.current_op`. Low impact but it's churn per op.

## Summary of the biggest wins

1. **Kill or promote `TensorPool`** — pick one.
2. **Remove the implicit results-file side-effect** from `ChiselContext.__init__`.
