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


# Ops in this set bypass _prepare_op_args (used for ops with intentional dtype
# mismatches between tensor arguments, e.g. embedding where indices are int64).
_SKIP_PREPARE_OP_ARGS: set = set()


def _skip_prepare(*targets):
    """Mark targets as bypassing _prepare_op_args. Stack with @_lowering."""
    def decorator(fn):
        for t in targets:
            _SKIP_PREPARE_OP_ARGS.add(t)
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


@_lowering(_aten.sum.dim_IntList)
def _(mb, x, dim, keepdim=False, *, dtype=None):
    return mb.sum(x, list(dim), keepdim)


@_lowering(_aten.detach.default)
def _(mb, x):
    # Autograd bookkeeping only — no data movement. The aot joint graph emits
    # detach around saved-for-backward tensors; lower it to the identity.
    return x


@_lowering(_aten.threshold_backward.default)
def _(mb, grad_output, self, threshold):
    # relu's backward in the autograd graph: grad_output * (self > threshold).
    return mb.threshold_backward(grad_output, self, float(threshold))


@_lowering(_aten.mse_loss.default)
def _(mb, self, target, reduction=1):
    return mb.mse_loss(self, target, int(reduction))


@_lowering(_aten.mse_loss_backward.default)
def _(mb, grad_output, self, target, reduction):
    return mb.mse_loss_backward(grad_output, self, target, int(reduction))


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


@_lowering(_aten.cos.default)
def _(mb, x):
    return mb.cos(x)


@_lowering(_aten.sin.default)
def _(mb, x):
    return mb.sin(x)


@_lowering(_aten.neg.default)
def _(mb, x):
    return mb.neg(x)


@_lowering(_aten.silu.default)
def _(mb, x):
    return mb.silu(x)


@_lowering(_aten.div.Tensor)
def _(mb, lhs, rhs):
    return mb.div(lhs, rhs)


@_lowering(_aten.div.Scalar)
def _(mb, x, scalar):
    return mb.div(x, mb.scalar_like(x, float(scalar)))


@_lowering(_aten.pow.Tensor_Tensor)
def _(mb, lhs, rhs):
    return mb.pow(lhs, rhs)


@_lowering(_aten.pow.Tensor_Scalar)
def _(mb, lhs, exp):
    return mb.pow(lhs, mb.scalar_like(lhs, float(exp)))


@_lowering(_aten.matmul.default, _aten.bmm.default)
def _(mb, lhs, rhs):
    return mb.matmul(lhs, rhs)


@_lowering(_aten.add.Scalar)
def _(mb, x, scalar, alpha=1):
    return mb.add(x, mb.scalar_like(x, float(scalar)), float(alpha))


@_lowering(_aten.mul.Scalar)
def _(mb, x, scalar):
    return mb.mul(x, mb.scalar_like(x, float(scalar)))


@_lowering(_aten._softmax.default, _aten._safe_softmax.default)
def _(mb, x, dim, half_to_float=False):
    return mb.softmax(x, dim)


@_lowering(_aten.argmax.default)
@_skip_prepare(_aten.argmax.default)
def _(mb, x, dim=None, keepdim=False):
    # Skip _prepare_op_args: argmax output dtype (int64) is an index type,
    # unrelated to the input dtype. Promoting the input to int64 changes the
    # values being compared and produces wrong results.
    return mb.argmax(x, dim, keepdim)


@_lowering(_aten.unsqueeze.default)
def _(mb, x, dim):
    return mb.unsqueeze(x, dim)


@_lowering(_aten.squeeze.dim)
def _(mb, x, dim):
    return mb.squeeze(x, dim)


@_lowering(_aten.transpose.int)
def _(mb, x, dim0, dim1):
    return mb.transpose(x, dim0, dim1)


@_lowering(_aten.expand.default)
def _(mb, x, target_shape):
    return mb.broadcast(x, list(target_shape))


@_lowering(_aten.permute.default)
def _(mb, x, dims):
    return mb.permute(x, list(dims))


@_lowering(_aten.cat.default)
def _(mb, tensors, dim=0):
    items = [v for v in tensors if not (isinstance(v, torch.Tensor) and v.numel() == 0)]
    if len(items) == 1:
        return items[0]
    return mb.cat(items, int(dim))


@_lowering(_aten.slice.Tensor)
def _(mb, x, dim=0, start=None, end=None, step=1):
    return mb.slice(x, int(dim), start if start is None else int(start),
                    end if end is None else int(end), int(step))


@_lowering(_aten.arange.default, _aten.arange.start, _aten.arange.start_step)
def _(mb, *args, dtype=None, layout=None, device=None, pin_memory=None):
    if len(args) == 1:
        start, end, step = 0, int(args[0]), 1
    elif len(args) == 2:
        start, end, step = int(args[0]), int(args[1]), 1
    else:
        start, end, step = int(args[0]), int(args[1]), int(args[2])

    rt_dtype = _to_runtime_dtype(dtype if dtype is not None else torch.int64)
    return mb.arange(start, end, step, rt_dtype)


@_lowering(_aten.embedding.default)
@_skip_prepare(_aten.embedding.default)
def _(mb, weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    return mb.embedding(weight, indices)


@_lowering(_aten.le.Tensor)
@_skip_prepare(_aten.le.Tensor)
def _(mb, lhs, rhs):
    return mb.le(lhs, rhs)


@_lowering(_aten.where.self)
@_skip_prepare(_aten.where.self)
def _(mb, condition, self, other):
    return mb.where(condition, self, other)


@_lowering(_aten.tril.default)
@_skip_prepare(_aten.tril.default)
def _(mb, input, diagonal=0):
    return mb.tril(input, int(diagonal))


@_lowering(_aten._scaled_dot_product_flash_attention_for_cpu.default)
@_skip_prepare(_aten._scaled_dot_product_flash_attention_for_cpu.default)
def _(mb, query, key, value, dropout_p=0.0, is_causal=False, attn_mask=None, scale=None):
    result = mb.sdpa(query, key, value, is_causal=is_causal, scale=scale, attn_mask=attn_mask)
    # Returns a 9-tuple; downstream getitem[0] extracts the attention output.
    return (result, None, None, None, None, None, None, None, None)


@_lowering(_aten._to_copy.default)
@_skip_prepare(_aten._to_copy.default)
def _(mb, x, dtype=None, layout=None, device=None, pin_memory=None, non_blocking=False, memory_format=None):
    if dtype is not None:
        return mb.typecast(x, _to_runtime_dtype(dtype))
    return x


@_lowering(_aten._unsafe_view.default)
def _(mb, x, size):
    return mb.reshape(x, list(size))


@_lowering(_aten.alias.default, _aten.clone.default, _aten.lift_fresh_copy.default)
@_skip_prepare(_aten.alias.default, _aten.clone.default, _aten.lift_fresh_copy.default)
def _(mb, x, **kwargs):
    return x


@_lowering(_aten.scalar_tensor.default)
@_skip_prepare(_aten.scalar_tensor.default)
def _(mb, value, dtype=None, layout=None, device=None, pin_memory=None):
    rt_dtype = _to_runtime_dtype(dtype if dtype is not None else torch.float32)
    return mb.scalar(rt_dtype, float(value))


@_lowering(_aten.full.default)
@_skip_prepare(_aten.full.default)
def _(mb, size, fill_value, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    rt_dtype = _to_runtime_dtype(dtype if dtype is not None else torch.float32)
    return mb.broadcast(mb.scalar(rt_dtype, float(fill_value)), list(size))


@_lowering(_aten.zeros.default)
@_skip_prepare(_aten.zeros.default)
def _(mb, size, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    rt_dtype = _to_runtime_dtype(dtype if dtype is not None else torch.float32)
    return mb.broadcast(mb.scalar(rt_dtype, 0.0), list(size))


@_lowering(_aten.index_copy.default)
@_skip_prepare(_aten.index_copy.default)
def _(mb, input, dim, index, source):
    return mb.index_copy(input, int(dim), index, source)


@_lowering(_aten.ones.default)
@_skip_prepare(_aten.ones.default)
def _(mb, size, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    rt_dtype = _to_runtime_dtype(dtype if dtype is not None else torch.float32)
    return mb.broadcast(mb.scalar(rt_dtype, 1.0), list(size))


def _is_tensor_schema_arg(
    idx: int,
    schema: torch._C.FunctionSchema,
) -> bool:
    return idx < len(schema.arguments) and "Tensor" in str(schema.arguments[idx].type)


def _prepare_op_args(
    mb: "_native.ModuleBuilder",
    args: tuple,
    target_dtype: "_native.DataType",
    target: torch._ops.OpOverload,
) -> tuple:
    """Uses the ATen schema to distinguish tensor-typed positions from
    non-tensor attributes (keepdim, dim, eps, momentum, etc.).
    - _native.Value at a tensor position: typecast to target_dtype
    - Python scalar (int/float) at a tensor position: lifted to a [1]-shaped
      ttir.constant (handles e.g. aten.add.Tensor(x, 3.14))
    - Everything else (None, list, bool, ...): passed through unchanged
    """
    if target in _SKIP_PREPARE_OP_ARGS:
        return args

    out = []
    for i, a in enumerate(args):
        if not _is_tensor_schema_arg(i, target._schema):
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
        args = _prepare_op_args(self.mb, args, target_dtype, target)
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

    # `None` outputs are real graph outputs: aot_autograd's backward graph emits
    # one gradient slot per forward input and fills `None` where that input
    # doesn't require grad (e.g. the data/target tensors — the tensor-valued
    # grads are the model parameters). The compiled program only carries the
    # tensor-valued outputs; `none_mask` records where to splice the `None`s back
    # in so the runner returns one value per slot, as autograd expects.
    outputs: list = []
    output_dtypes: list = []
    none_mask: list[bool] = []
    for v, fx_node in zip(result, fx_outputs):
        if v is None:
            none_mask.append(True)
            continue
        if isinstance(v, tuple):
            raise NotImplementedError(
                "tt-kurbla compile: tuple-valued graph output not supported (use getitem first)"
            )
        none_mask.append(False)
        outputs.append(v)
        output_dtypes.append(_to_runtime_dtype(fx_node.meta["val"].dtype))

    program = mb.compile(outputs)

    def runner(*inputs: torch.Tensor) -> list:
        produced = iter(_native.run_program(program, list(inputs), output_dtypes))
        return [None if is_none else next(produced) for is_none in none_mask]

    return runner


def tt_backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    """Top-level dynamo backend. Delegates to aot_module_simplified."""
    return aot_module_simplified(gm, example_inputs, fw_compiler=_fw_compiler)


# Self-register on import. After this, `torch.compile(model, backend="tt")` works.
torch._dynamo.register_backend(name="tt", compiler_fn=tt_backend)
