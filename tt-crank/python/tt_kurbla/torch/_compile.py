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

import operator
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


# Python operator registry: callable -> fn(*args, **kwargs).
_OPERATORS: dict = {}

def _operator(*targets):
    def decorator(fn):
        for t in targets:
            _OPERATORS[t] = fn
        return fn

    return decorator


@_operator(operator.getitem)
def _(container, idx):
    if isinstance(container, tuple):
        return container[idx]
    raise NotImplementedError(f"tt-kurbla compile: getitem on non-tuple {type(container).__name__}")


# ATen op lowering registry: OpOverload -> fn(mb, *args, **kwargs).
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


@_lowering(_aten.sub.Tensor)
def _(mb, a, b, *, alpha=1):
    return mb.sub(a, b, float(alpha))


@_lowering(_aten.mul.Tensor)
def _(mb, a, b):
    return mb.mul(a, b)


@_lowering(_aten.rsqrt.default)
def _(mb, x):
    return mb.rsqrt(x)


@_lowering(_aten.view.default, _aten.reshape.default)
def _(mb, x, size):
    return mb.reshape(x, list(size))


@_lowering(_aten.mean.dim)
def _(mb, x, dim, keepdim=False, *, dtype=None):
    return mb.mean(x, list(dim), keepdim)


@_lowering(_aten.mm.default)
def _(mb, a, b):
    return mb.mm(a, b)


@_lowering(_aten.addmm.default)
def _(mb, bias, mat1, mat2, *, beta=1, alpha=1):
    return mb.addmm(bias, mat1, mat2, float(beta), float(alpha))


@_lowering(_aten.t.default)
def _(mb, input):
    return mb.t(input)


@_lowering(_aten.relu.default)
def _(mb, input):
    return mb.relu(input)

@_lowering(_aten.convolution.default)
def _(mb, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    if transposed:
        raise NotImplementedError("tt-kurbla compile: transposed convolution not supported")
    return mb.conv2d(input, weight, bias, list(stride), list(padding), list(dilation), int(groups))


@_lowering(_aten.max_pool2d_with_indices.default)
def _(mb, x, kernel_size, stride, padding=0, dilation=1, ceil_mode=False):
    if isinstance(padding, int):
        padding = [padding, padding]
    if isinstance(dilation, int):
        dilation = [dilation, dilation]
    if not stride:
        stride = list(kernel_size)
    result = mb.max_pool2d(x, list(kernel_size), list(stride), list(padding), list(dilation), bool(ceil_mode))
    return (result, None)


@_lowering(_aten._native_batch_norm_legit_no_training.default)
def _(mb, input, weight, bias, running_mean, running_var, momentum, eps):
    if weight is None or bias is None:
        raise NotImplementedError(
            "tt-kurbla compile: batch_norm without affine parameters (affine=False) not supported"
        )
    result = mb.batch_norm_inference(input, weight, bias, running_mean, running_var, float(eps))
    return (result, None, None)


def _is_tensor_schema_arg(
    idx: int,
    schema: torch._C.FunctionSchema,
) -> bool:
    return idx < len(schema.arguments) and "Tensor" in str(schema.arguments[idx].type)


def _prepare_op_args(
    mb: "_native.ModuleBuilder",
    args: tuple,
    target_dtype: "_native.DataType",
    schema: torch._C.FunctionSchema,
) -> tuple:
    """Uses the ATen schema to distinguish tensor-typed positions from
    non-tensor attributes (keepdim, dim, eps, momentum, etc.).
    - _native.Value at a tensor position: typecast to target_dtype
    - Python scalar (int/float) at a tensor position: lifted to a [1]-shaped
      ttir.constant (handles e.g. aten.add.Tensor(x, 3.14))
    - Everything else (None, list, bool, ...): passed through unchanged
    """
    out = []
    for i, a in enumerate(args):
        if not _is_tensor_schema_arg(i, schema):
            out.append(a)
            continue

        if isinstance(a, _native.Value):
            out.append(mb.typecast(a, target_dtype))
        elif isinstance(a, (int, float)):
            out.append(mb.scalar(target_dtype, float(a)))
        else:
            out.append(a)

    return tuple(out)


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

    def _lower_op(self, target, args, kwargs):
        fn = _LOWERINGS.get(target)
        if fn is None:
            raise NotImplementedError(f"tt-kurbla compile: op {target} not implemented")

        val = self._current_node.meta["val"]
        target_dtype = _to_runtime_dtype(val[0].dtype if isinstance(val, (tuple, list)) else val.dtype)
        args = _prepare_op_args(self.mb, args, target_dtype, target._schema)
        return fn(self.mb, *args, **kwargs)

    def _call_operator(self, target, args, kwargs):
        op = _OPERATORS.get(target)
        if op is None:
            raise NotImplementedError(f"tt-kurbla compile: operator {target} not implemented")

        return op(*args, **kwargs)

    def call_function(self, target, args, kwargs):
        if isinstance(target, torch._ops.OpOverload):
            return self._lower_op(target, args, kwargs)

        return self._call_operator(target, args, kwargs)


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
