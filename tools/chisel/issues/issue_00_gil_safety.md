# GIL-Safety Fix for DebugHooks Callbacks

## Problem

`debug::Hooks` currently copies callback `std::function` objects by value,
which triggers Python GIL acquisition on copy/destroy in non-Python threads.

This surfaces when a Python process uses both `ttrt` and another framework that
releases the GIL for its own threading (e.g. `torch_xla`, `jax`). Even trivial
callbacks that never touch Python objects will segfault because the C++ side
copies the `std::function` wrapper, and that copy increments the Python
reference count **without holding the GIL** — a data race on the refcount that
corrupts the Python object.

### Reproduction sketch

```python
import ttrt.runtime as runtime

# Any callback, even a no-op print, is wrapped in std::function<void(...)>
def my_pre_op(op_ctx):
    print("pre-op")

def my_post_op(op_ctx):
    print("post-op")

hooks = runtime.DebugHooks()
hooks.set_pre_op(my_pre_op)
hooks.set_post_op(my_post_op)

# When the C++ runtime later copies `hooks` (e.g. into a worker thread),
# it copies the std::function values — incrementing Python refcounts
# without holding the GIL → refcount corruption → segfault.
```

> **Note:** This cannot currently be tested inside tt-mlir because the repo has
> no test harness that exercises `DebugHooks` callbacks from a non-Python
> thread. The issue was observed in downstream environments that combine `ttrt`
> with GIL-releasing frameworks.

## Proposed Fix

Return callbacks by `const&` and accept by rvalue ref + `std::move` so no
`std::function` copies occur after initial registration.

No API change — all existing callers work as-is.

## Acceptance Criteria

- Existing runtime tests pass unchanged
- No `std::function` value copies remain in `debug::Hooks` (all by ref or move)
