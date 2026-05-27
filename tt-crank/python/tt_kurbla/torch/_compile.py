"""Phase-0 `torch.compile()` backend for tt-kurbla.

Registers a dynamo backend under the name `tt` that lowers a post-aot FX
graph into a single TTIR module, compiles it through tt-mlir, and returns a
runner closure that binds inputs and executes the compiled program on each
call.

Pipeline::

    torch.compile(model, backend="tt")
      -> dynamo trace
      -> aot_module_simplified
      -> _fw_compiler: torch.fx.Interpreter walks the post-aot FX graph,
                       dispatching each call_function to a decorator-registered
                       lowering that emits TTIR via _native.ModuleBuilder
      -> _native.ModuleBuilder.compile() produces a CompiledProgram
      -> runner(*inputs) -> _native.run_program(...)
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.fx
from torch._dynamo.backends.common import aot_module_simplified

from . import _native

_aten = torch.ops.aten


_DTYPE_TO_RUNTIME = {
    torch.float32: _native.DataType.Float32,
    torch.float64: _native.DataType.Float64,
    torch.float16: _native.DataType.Float16,
    torch.bfloat16: _native.DataType.BFloat16,
    torch.int32: _native.DataType.Int32,
    torch.int64: _native.DataType.Int64,
    torch.bool: _native.DataType.Bool,
    torch.uint8: _native.DataType.UInt8,
}


def _to_runtime_dtype(dtype: torch.dtype) -> "_native.DataType":
    try:
        return _DTYPE_TO_RUNTIME[dtype]
    except KeyError as e:
        raise NotImplementedError(f"tt-kurbla compile: unsupported torch dtype {dtype}") from e


def _spec_from_tensor(t: torch.Tensor) -> "_native.TensorTypeSpec":
    return _native.TensorTypeSpec(list(t.shape), _to_runtime_dtype(t.dtype))


# Lowering registry: FX call_function target -> function (mb, *args, **kwargs).
# Each FX op supported by the tt backend will be registered here via the @_lowering decorator.
_LOWERINGS: dict = {}


def _lowering(*targets):
    def decorator(fn):
        for t in targets:
            _LOWERINGS[t] = fn
        return fn

    return decorator


@_lowering(_aten.add.Tensor)
def _(mb, a, b, *, alpha=1):
    return mb.add(a, b, float(alpha))


def _prepare_op_args(
    mb: "_native.ModuleBuilder",
    args: tuple,
    target_dtype: "_native.DataType",
) -> tuple:
    """Normalize a lowering's positional args to `target_dtype`.

    Casts tensor args to `target_dtype` and lifts Python scalars to
    broadcastable `ttir.constant` values at the same dtype. After this every
    positional arg is a `Value` at the target element type.
    """

    def _convert(a):
        if isinstance(a, _native.Value):
            return mb.typecast(a, target_dtype)
        if isinstance(a, (int, float)):
            return mb.scalar(target_dtype, float(a))
        return a

    return tuple(_convert(a) for a in args)


class _TTIRInterpreter(torch.fx.Interpreter):
    """Walks the post-aot FX graph, dispatching each call_function to its
    registered lowering. `torch.fx.Interpreter` handles env management,
    arg/kwarg resolution, and placeholder/output plumbing.
    """

    def __init__(self, gm: torch.fx.GraphModule, mb: "_native.ModuleBuilder") -> None:
        super().__init__(gm)
        self.mb = mb
        self._current_node: torch.fx.Node | None = None

    def run_node(self, n: torch.fx.Node):
        self._current_node = n
        return super().run_node(n)

    def call_function(self, target, args, kwargs):
        fn = _LOWERINGS.get(target)
        if fn is None:
            raise NotImplementedError(f"tt-kurbla compile: unsupported FX target {target}")
        target_dtype = _to_runtime_dtype(self._current_node.meta["val"].dtype)
        args = _prepare_op_args(self.mb, args, target_dtype)
        return fn(self.mb, *args, **kwargs)


def _fw_compiler(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
    """aot_module_simplified's fw_compiler hook.

    Walks the post-aot FX graph once via _TTIRInterpreter, finalizes the
    accumulated TTIR module, compiles it to a flatbuffer, and returns a
    runner closure that binds inputs and runs the compiled program on each
    call.
    """
    specs = [_spec_from_tensor(t) for t in example_inputs]
    mb = _native.ModuleBuilder(specs)
    placeholder_values = [mb.arg(i) for i in range(len(specs))]

    # Read each output's user-facing dtype from FX meta. The program's
    # output descriptors can disagree after the tt-mlir rewriter demotes
    # wide types (e.g. i64 to i32), and we want to give the user back
    # what they asked for.
    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    fx_outputs = output_node.args[0]
    if not isinstance(fx_outputs, (tuple, list)):
        fx_outputs = (fx_outputs,)

    result = _TTIRInterpreter(gm, mb).run(*placeholder_values)
    if not isinstance(result, (tuple, list)):
        result = (result,)

    outputs: list = []
    output_dtypes: list = []
    for v, fx_node in zip(result, fx_outputs):
        if v is None:
            raise NotImplementedError("tt-kurbla compile: None output not supported")
        if isinstance(v, tuple):
            raise NotImplementedError(
                "tt-kurbla compile: tuple-valued graph output not supported (use getitem first)"
            )
        outputs.append(v)
        output_dtypes.append(_to_runtime_dtype(fx_node.meta["val"].dtype))

    program = mb.compile(outputs)

    def runner(*inputs: torch.Tensor) -> list[torch.Tensor]:
        return _native.run_program(program, list(inputs), output_dtypes)

    return runner


def tt_backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    """Top-level dynamo backend. Delegates to aot_module_simplified."""
    return aot_module_simplified(gm, example_inputs, fw_compiler=_fw_compiler)


# Self-register on import. After this, `torch.compile(model, backend="tt")` works.
torch._dynamo.register_backend(name="tt", compiler_fn=tt_backend)
