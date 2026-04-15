# GIL-Safety Fix for DebugHooks Callbacks

## Problem

`debug::Hooks` currently copies callback `std::function` objects by value,
which triggers Python GIL acquisition on copy/destroy in non-Python threads.

This surfaces when a Python process uses both `ttrt` and tt-xla that
releases the GIL for its own threading (e.g. `torch_xla`, `jax`). Even trivial
callbacks that never touch Python objects will segfault because the C++ side
copies the `std::function` wrapper, and that copy increments the Python
reference count **without holding the GIL** — a data race on the refcount that
corrupts the Python object.

## Proposed Fix

Return callbacks by `const&` and accept by rvalue ref + `std::move` so no
`std::function` copies occur after initial registration.

No API change — all existing callers work as-is.
