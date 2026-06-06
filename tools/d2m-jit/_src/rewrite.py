# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pattern-rewrite framework for d2m-jit.

Lets users define MLIR pattern rewrites in plain Python that match TTIR
(or any) ops and replace them with d2m-jit kernel-template subgraphs.

Surface:

    @d2m.pattern(root=ttir.MatmulOp, benefit=5)
    def lower_matmul(op, rewriter):
        a = d2m.from_value(op.operand(0))
        b = d2m.from_value(op.operand(1))
        out = d2m.empty(d2m.infer_layout(op.result, grid_shape=[4, 4]))
        my_fused_kernel(a, b, out, grid=(4, 4))
        return out  # replaces op.result

    # DAG match: predicate decides; the rewrite is unconditional.
    @d2m.pattern(
        root="ttir.all_reduce",
        benefit=10,
        match=lambda op: (
            op.operands[0].owner is not None
            and op.operands[0].owner.name == "ttir.matmul"
        ),
    )
    def fuse_mm_ar(op, rewriter):
        ...

    d2m.apply_patterns(module)  # runs the greedy driver

Under the hood each `@d2m.pattern` lowers to a per-pattern `pdl.pattern`
stub (single-op match by root op name; variadic operands/types so the
template works for any-arity ops). If a `match` predicate is provided,
its inline-`apply_native_constraint` runs first — if it returns false,
the bytecode driver cleanly aborts the match (no IR mutated). The
unconditional rewrite then runs in `pdl.apply_native_rewrite`, emitting
the replacement subgraph at the rewriter's insertion point through the
regular d2m-jit emission helpers (the framework auto-pushes a
`RewriteScope`), and returning the final SSA Value or `LazyTensor`. PDL
then issues a `pdl.replace` to wire the rest of the IR over.

DAG matching is imperative — express the producer-chain check in the
`match=` predicate (pure, no IR mutation); the decorated function gets
called only on successful matches. There is no declarative DAG-builder.
This is intentional for the POC.

The greedy pattern rewriter does *not* support recoverable failures
inside `pdl.apply_native_rewrite`; that's why the bail-out path lives
in `match=` rather than as a `return None` from the rewrite body.
"""

from __future__ import annotations

import itertools
from typing import Callable, List, Optional, Sequence, Union

from ttmlir import ir
from ttmlir.dialects import ttcore
from ttmlir.rewrite import PDLModule, apply_patterns_and_fold_greedily

from .builder import (
    LazyTensor,
    RewriteScope,
    _push_scope,
)
from .tensor_layout import Layout


# --- PatternRegistry --------------------------------------------------------


class _Pattern:
    """A registered rewrite pattern.

    Holds the user's Python function plus the synthetic PDL symbol names
    used to wire it into the PDL bytecode driver.
    """

    def __init__(
        self,
        root_op_name: str,
        benefit: int,
        fn: Callable,
        match_fn: Optional[Callable] = None,
    ):
        self.root_op_name = root_op_name
        self.benefit = benefit
        self.fn = fn
        self.match_fn = match_fn

        slug = root_op_name.replace(".", "_")
        # Use a process-wide counter so that re-registering the same root
        # multiple times yields distinct symbols.
        idx = next(_PatternRegistry._sym_counter)
        self.sym_name = f"d2m_pat_{slug}_{idx}"
        self.dispatch_name = f"d2m_dispatch_{slug}_{idx}"
        self.match_name = f"d2m_match_{slug}_{idx}"

    def build_pdl_text(self) -> str:
        """One `pdl.pattern { ... }` stub. If a `match_fn` is registered,
        an inline `apply_native_constraint` runs first — its failure
        cleanly aborts the match (the rewrite below never runs)."""
        constraint_line = (
            f'    pdl.apply_native_constraint "{self.match_name}"(%op : !pdl.operation)\n'
            if self.match_fn is not None
            else ""
        )
        return f"""\
  pdl.pattern @{self.sym_name} : benefit({self.benefit}) {{
    %operands = pdl.operands
    %types = pdl.types
    %op = pdl.operation "{self.root_op_name}"(%operands : !pdl.range<value>) -> (%types : !pdl.range<type>)
{constraint_line}    pdl.rewrite %op {{
      %v = pdl.apply_native_rewrite "{self.dispatch_name}"(%op : !pdl.operation) : !pdl.value
      pdl.replace %op with (%v : !pdl.value)
    }}
  }}"""


class _PatternRegistry:
    _sym_counter = itertools.count()

    def __init__(self):
        self._patterns: List[_Pattern] = []

    def register(
        self,
        root: Union[str, type],
        benefit: int,
        fn: Callable,
        match_fn: Optional[Callable] = None,
    ) -> _Pattern:
        root_name = (
            root if isinstance(root, str) else getattr(root, "OPERATION_NAME", None)
        )
        if not root_name:
            raise TypeError(
                f"pattern root must be an op-name string or an OpView class with "
                f"OPERATION_NAME; got {root!r}"
            )
        pat = _Pattern(
            root_op_name=root_name,
            benefit=int(benefit),
            fn=fn,
            match_fn=match_fn,
        )
        self._patterns.append(pat)
        return pat

    def all(self) -> List[_Pattern]:
        return list(self._patterns)

    def clear(self) -> None:
        """For tests — drop all registered patterns."""
        self._patterns.clear()


_registry = _PatternRegistry()


def pattern(*, root, benefit: int = 1, match: Optional[Callable] = None):
    """Decorator: register `fn` as a rewrite pattern rooted on `root`.

    `root` is either an op-name string (e.g. "ttir.matmul") or an OpView
    class with an `OPERATION_NAME` attribute (e.g. `ttir.MatmulOp`).

    The decorated function has signature `(op, rewriter) -> ret`, where:
      - `op` is the matched MLIR `ir.Operation`.
      - `rewriter` is the upstream `PatternRewriter` (its `.ip` is set to
        before the matched op).
      - `ret` is the SSA value (or a `LazyTensor` wrapping one) that
        replaces the matched op's first (currently only) result.
        The function MUST always return a value — the greedy driver
        doesn't support recoverable rewrite failures.

    For DAG-conditional matching (e.g. "fire only when this op's
    producer is X"), pass a `match=callable` predicate: `match(op) ->
    bool` runs as a PDL native constraint before the rewrite. If it
    returns falsy the bytecode aborts the match cleanly (no IR
    mutated). Predicates must be pure — do not emit IR from them.
    Root on the tail op of a DAG so the replace removes the last
    visible node.
    """

    def decorator(fn):
        _registry.register(root=root, benefit=benefit, fn=fn, match_fn=match)
        return fn

    return decorator


# --- LazyTensor wrapping ----------------------------------------------------


def from_value(value: ir.Value, layout: Optional[Layout] = None) -> LazyTensor:
    """Wrap an existing SSA `Value` as a `LazyTensor`.

    Used in pattern rewrites: the operands of the matched op already exist
    as `ir.Value`s in the surrounding module; this lifts them into the
    d2m-jit emission helpers' tensor-handle abstraction.

    If `layout` is omitted it is inferred from `value.type` via
    `infer_layout` (safe defaults; override liberally).
    """
    from .builder import _get_scope

    if not isinstance(value, ir.Value):
        raise TypeError(f"from_value expects an ir.Value, got {type(value).__name__}")
    if layout is None:
        layout = infer_layout(value)
    scope = _get_scope()
    return LazyTensor(layout, value, scope.generation)


def from_device(lt: LazyTensor) -> ir.Value:
    """Emit a `d2m.to_layout` that materializes a device-laid-out `LazyTensor`
    back to a plain host tensor type (no metal-layout encoding).

    This is the typical final step inside a pattern rewrite that consumes a
    TTIR op whose result type is a plain `tensor<MxNxf32>` — patterns build
    a device subgraph, then call `from_device(...)` to land back on the
    matched op's result type before returning the SSA Value.
    """
    from .builder import _get_scope

    if not isinstance(lt, LazyTensor):
        raise TypeError(f"from_device expects a LazyTensor, got {type(lt).__name__}")
    lt = lt._resolve()
    scope = _get_scope()
    with scope.ctx, scope.loc, scope.insert_point:
        dev_val = lt.layout.build_device_view(scope.ctx, lt.value)
        host_val = lt.layout.build_from_device(scope.ctx, dev_val)
    return host_val


# --- Layout inference -------------------------------------------------------


def _try_tile_element_type(elem):
    """Return (tile_h, tile_w, ttcore.DataType) if elem is a TileType, else None."""
    try:
        tile = ttcore.ir.TileType.maybe_downcast(elem)
    except AttributeError:
        tile = None
    if tile is None:
        return None
    # TileType has shape attrs; pull them defensively.
    try:
        return (int(tile.shape[0]), int(tile.shape[1]), tile.data_type)
    except Exception:
        return None


def _scalar_dtype_from_mlir(elem):
    """Map an MLIR float type to a ttcore.DataType, or raise."""
    ctx = elem.context
    if ir.F32Type.isinstance(elem):
        return ttcore.DataType.Float32
    if ir.F16Type.isinstance(elem):
        return ttcore.DataType.Float16
    if ir.BF16Type.isinstance(elem):
        return ttcore.DataType.BFloat16
    raise TypeError(
        f"infer_layout: unsupported element type {elem}; "
        f"only f32/f16/bf16 are mapped today. Pass an explicit Layout."
    )


def infer_layout(value_or_type, **overrides) -> Layout:
    """Build a safe-default `Layout` from an `ir.Value` or `RankedTensorType`.

    The default is logical / untiled / single-cell:
        - `shape` = tensor type shape
        - `dtype` = mapped from the element type (f32/f16/bf16)
        - `tiled` = False
        - `block_shape` = full shape (one block holds the whole tensor)
        - `grid_shape` = [1, 1, ...]
        - `mem_space` = "l1"

    Anything passed in `overrides` replaces the default field. Patterns
    typically override `tiled=True`, set a real `grid_shape`, and pick a
    smaller `block_shape`.

    Tile-typed tensors (`!ttcore.tile<32x32, ...>` element) currently
    require an explicit Layout — raise rather than silently guessing.
    """
    t = value_or_type.type if isinstance(value_or_type, ir.Value) else value_or_type
    if not ir.RankedTensorType.isinstance(t):
        raise TypeError(f"infer_layout: expected RankedTensorType, got {t}")
    rtt = ir.RankedTensorType(t)
    shape = list(rtt.shape)
    elem = rtt.element_type

    if _try_tile_element_type(elem) is not None:
        raise NotImplementedError(
            "infer_layout: tile-typed tensors are not auto-inferred yet; "
            "pass an explicit Layout."
        )

    dtype = _scalar_dtype_from_mlir(elem)
    fields = dict(
        shape=shape,
        dtype=dtype,
        block_shape=list(shape),
        grid_shape=[1] * len(shape),
        tiled=False,
        mem_space="l1",
    )
    fields.update(overrides)
    return Layout(**fields)


# --- Driver -----------------------------------------------------------------


def _make_dispatcher(pat: _Pattern):
    """Wrap `pat.fn` so PDL's native-rewrite signature dispatches to the user."""

    def dispatcher(rewriter, results, values):
        # PDL hands us [op_handle]. The rewriter's IP is "before the matched op".
        if len(values) != 1:
            raise RuntimeError(
                f"d2m-jit dispatch '{pat.dispatch_name}': expected 1 PDL value, "
                f"got {len(values)}"
            )
        op = values[0]
        if not isinstance(op, ir.Operation):
            raise RuntimeError(
                f"d2m-jit dispatch '{pat.dispatch_name}': expected an Operation, "
                f"got {type(op).__name__}"
            )

        scope = RewriteScope(rewriter, op)
        with _push_scope(scope):
            ret = pat.fn(op, rewriter)

        if ret is None:
            raise RuntimeError(
                f"pattern '{pat.fn.__name__}' returned None. The greedy driver "
                f"does not support recoverable rewrite failures; for "
                f"DAG-conditional matching, pass `match=<predicate>` to "
                f"@d2m.pattern instead."
            )

        # Coerce to a single ir.Value.
        if isinstance(ret, LazyTensor):
            replacement = ret.value
        elif isinstance(ret, ir.Value):
            replacement = ret
        else:
            raise TypeError(
                f"pattern '{pat.fn.__name__}' must return an ir.Value or a "
                f"LazyTensor; got {type(ret).__name__}"
            )

        results.append(replacement)
        # None / False are success; True is failure.
        return None

    return dispatcher


def _make_match_dispatcher(pat: _Pattern):
    """Wrap `pat.match_fn` as a PDL native constraint.

    Constraints must be pure — do not mutate IR. They return success
    (None/False) to let the match proceed, or failure (True) to abort.
    """
    assert pat.match_fn is not None

    def constraint(rewriter, results, values):
        op = values[0]
        try:
            ok = pat.match_fn(op)
        except Exception as e:
            # Treat an exception inside the predicate as "no match" so the
            # greedy driver can keep going. Log via diagnostic when possible.
            try:
                op.emit_remark(
                    f"d2m-jit match predicate raised: {e!r}; treating as no-match"
                )
            except Exception:
                pass
            return True  # failure -> skip
        return None if ok else True

    return constraint


def _build_pdl_text(patterns: Sequence[_Pattern]) -> str:
    body = "\n".join(p.build_pdl_text() for p in patterns)
    return f"module {{\n{body}\n}}\n"


def apply_patterns_text(input_text: str, pattern_paths: Sequence[str]) -> str:
    """Text-IO entrypoint for the d2m-jit pattern framework.

    Parses `input_text` (textual MLIR) in a fresh ir.Context, imports each
    file in `pattern_paths` so its `@d2m.pattern` decorators register, runs
    `apply_patterns` on the parsed module, and returns the rewritten
    module as a printed string.

    This is the helper the embedded-Python C++ pass calls — it sidesteps
    the C++/Python "two MLIRs" problem (ttmlir-opt is statically linked
    against MLIR while ttmlir's Python bindings have their own shared
    copy, so capsule pointers can't cross the boundary safely).

    Pattern registration is process-global; we snapshot and restore so
    repeated calls don't accumulate stale entries.
    """
    import importlib.util
    import sys
    import uuid

    ctx = ir.Context()
    ctx.load_all_available_dialects()
    module = ir.Module.parse(input_text, ctx)

    snapshot = list(_registry.all())
    _registry.clear()
    try:
        for path in pattern_paths:
            mod_name = f"d2m_python_rewrites_user_{uuid.uuid4().hex}"
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"could not build a module spec for {path}")
            user_mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = user_mod
            spec.loader.exec_module(user_mod)

        apply_patterns(module)
        module.operation.verify()
        return str(module)
    finally:
        _registry.clear()
        for pat in snapshot:
            _registry._patterns.append(pat)


def apply_patterns(module_or_op) -> None:
    """Run the greedy driver with all registered patterns on the given op/module.

    The module's `Context` must have the `pdl` and `pdl_interp` dialects
    registered (true for tt-mlir's default `ttmlir.ir.Context` after the
    PDL bootstrap landed). The driver runs to fixpoint; canonicalization /
    folding interleave with the user patterns.

    Patterns whose user function returns `None` are treated as no-match
    and the greedy driver moves on. Otherwise the matched op is replaced
    by the returned Value via `pdl.replace`.
    """
    patterns = _registry.all()
    if not patterns:
        return

    # PDL stub module must live in the same Context as the target so the
    # frozen pattern set is compatible with the driver.
    ctx = module_or_op.context
    pdl_text = _build_pdl_text(patterns)
    try:
        pdl_module = ir.Module.parse(pdl_text, ctx)
    except Exception as e:
        # Helpful failure: dump the generated PDL so we can debug bad input.
        raise RuntimeError(
            f"d2m-jit: failed to parse generated PDL module:\n{pdl_text}\n--- error: {e}"
        ) from e

    pdl = PDLModule(pdl_module)
    for pat in patterns:
        pdl.register_rewrite_function(pat.dispatch_name, _make_dispatcher(pat))
        if pat.match_fn is not None:
            pdl.register_constraint_function(
                pat.match_name, _make_match_dispatcher(pat)
            )
    frozen = pdl.freeze()

    apply_patterns_and_fold_greedily(module_or_op, frozen)
