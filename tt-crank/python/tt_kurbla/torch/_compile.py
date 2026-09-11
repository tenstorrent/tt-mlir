"""`torch.compile()` backend for tt-kurbla.

Registers a dynamo backend under the name `tt` that lowers a post-aot FX
graph into a single TTIR module, compiles it through tt-mlir, and returns a
runner closure that binds inputs and executes the compiled program on each
call.

Pipeline::

    torch.compile(model, backend="tt")
      -> dynamo trace
      -> aot_module_simplified (fw_compiler + bw_compiler)
      -> _lower_and_compile: torch.fx.Interpreter walks the post-aot FX graph,
                             dispatching each call_function to a decorator-registered
                             lowering that emits TTIR via _native.ModuleBuilder
      -> _native.ModuleBuilder.compile() produces a CompiledProgram
      -> runner(*inputs) -> _native.run_program(...)
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable
from enum import StrEnum

import torch
import torch.fx
from torch._decomp import core_aten_decompositions, get_decompositions
from torch._dynamo.backends.common import aot_module_simplified
from torch._subclasses.fake_tensor import unset_fake_temporarily

from . import _native
from ._artifacts import Artifact, is_artifacts_dumper_active, register_artifact

_aten = torch.ops.aten
_funcol = torch.ops._c10d_functional


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
        item = container[idx]
        if item is None:
            # A multi-output lowering returns None for any slot it does not
            # materialise. Reading such a slot is unsupported - raise here rather
            # than letting the None flow on (it would otherwise be silently
            # accepted as a graph output and returned to the caller).
            raise NotImplementedError(f"tt-kurbla compile: output [{idx}] of a multi-output op is not lowered")
        return item
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


@_lowering(_aten.linalg_vector_norm.default)
def _(mb, x, ord=2, dim=None, keepdim=False, *, dtype=None):
    if ord != 2:
        raise NotImplementedError(f"tt-kurbla compile: linalg_vector_norm only supports ord=2, got {ord}")
    return mb.vector_norm(x, [] if dim is None else [int(d) for d in dim], keepdim)


# any's three overloads differ only in how the reduced dims are spelled: none at
# all (reduce everything), one int, or an optional list where None again means
# everything. mb.any takes the normalized list, with [] for the full reduction.
#
# Skip _prepare_op_args: the output dtype is Bool, and typecasting the input to
# i1 would truncate instead of testing against zero. mb.any does that test.
@_lowering(_aten.any.default)
@_skip_prepare(_aten.any.default)
def _(mb, x):
    return mb.any(x, [], False)


@_lowering(_aten.any.dim)
@_skip_prepare(_aten.any.dim)
def _(mb, x, dim, keepdim=False):
    return mb.any(x, [int(dim)], keepdim)


@_lowering(_aten.any.dims)
@_skip_prepare(_aten.any.dims)
def _(mb, x, dim=None, keepdim=False):
    return mb.any(x, [] if dim is None else [int(d) for d in dim], keepdim)


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

@_lowering(_aten.div.Tensor)
def _(mb, a, b):
    return mb.div(a, b)


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
    # NCW / NCHW / NCDHW select conv1d / conv2d / conv3d respectively.
    builders = {3: mb.conv1d, 4: mb.conv2d, 5: mb.conv3d}
    rank = len(input.shape)
    if rank not in builders:
        raise NotImplementedError(f"tt-kurbla compile: convolution supports rank 3, 4, or 5 inputs, got rank {rank}")
    return builders[rank](input, weight, bias, list(stride), list(padding), list(dilation), int(groups))


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


@_lowering(_aten.native_layer_norm.default)
def _(mb, input, normalized_shape, weight, bias, eps):
    # ttir.layer_norm returns only the normalized tensor; native_layer_norm's
    # contract is (out, mean, rstd), so layer_norm_with_stats recomputes them.
    # bf16/fp16 stats come back in f32, matching torch._refs._normalize.
    return mb.layer_norm_with_stats(input, weight, bias, list(normalized_shape), float(eps))


@_lowering(_aten.cos.default)
def _(mb, x):
    return mb.cos(x)


@_lowering(_aten.sin.default)
def _(mb, x):
    return mb.sin(x)


@_lowering(_aten.neg.default)
def _(mb, x):
    return mb.neg(x)


@_lowering(_aten.log.default)
def _(mb, x):
    return mb.log(x)


@_lowering(_aten.exp.default)
def _(mb, x):
    return mb.exp(x)


@_lowering(_aten.log1p.default)
def _(mb, x):
    return mb.log1p(x)


@_lowering(_aten.sqrt.default)
def _(mb, x):
    return mb.sqrt(x)


@_lowering(_aten.tanh.default)
def _(mb, x):
    return mb.tanh(x)


@_lowering(_aten.reciprocal.default)
def _(mb, x):
    return mb.reciprocal(x)


# `dtype` (a wider accumulator) needs no handling here, and the mean/sum
# reductions above drop it for the same reason: `_lower_op` takes each node's
# target dtype from that node's own output meta, so a widening request has
# already typecast the input up by the time the lowering runs and ttir.cumsum
# accumulates at that width.
# `dim` is normalized because ttir.cumsum's verifier rejects a negative one.
@_lowering(_aten.cumsum.default)
def _(mb, x, dim, dtype=None):
    return mb.cumsum(x, int(dim) % max(len(x.shape), 1))


@_lowering(_aten.silu.default)
def _(mb, x):
    return mb.silu(x)


@_lowering(_aten.sigmoid.default)
def _(mb, x):
    return mb.sigmoid(x)


@_lowering(_aten.clamp.default)
@_skip_prepare(_aten.clamp.default)
def _(mb, x, min=None, max=None):
    return mb.clamp(x, None if min is None else float(min), None if max is None else float(max))


# clamp_min is clamp with no upper bound.
@_lowering(_aten.clamp_min.default)
@_skip_prepare(_aten.clamp_min.default)
def _(mb, x, min):
    return mb.clamp(x, float(min), None)


# clamp_max is clamp with no lower bound.
@_lowering(_aten.clamp_max.default)
@_skip_prepare(_aten.clamp_max.default)
def _(mb, x, max):
    return mb.clamp(x, None, float(max))


@_lowering(_aten.floor_divide.default)
def _(mb, a, b):
    return mb.floor_divide(a, b)


def _remainder(mb, lhs, rhs):
    """torch.remainder is a *floored* modulo: the result takes the divisor's sign,
    so `-1 % 3 == 2`. ttir.remainder is the truncated (fmod) variant instead, so
    build this from floor division: lhs - floor(lhs / rhs) * rhs.
    """
    return mb.sub(lhs, mb.mul(mb.floor_divide(lhs, rhs), rhs))


@_lowering(_aten.remainder.Scalar)
def _(mb, x, other):
    return _remainder(mb, x, mb.scalar_like(x, float(other)))


@_lowering(_aten.remainder.Tensor)
def _(mb, lhs, rhs):
    return _remainder(mb, lhs, rhs)


@_lowering(_aten.gelu.default)
def _(mb, x, approximate="none"):
    # tt-mlir lowers gelu to ttnn.gelu(fast_and_approximate_mode=false): the
    # exact/accurate variant, i.e. approximate="none". A "tanh" request is
    # served by this same accurate op - correct (well within precision), it
    # just doesn't get the faster tanh-approx kernel it asked for.
    return mb.gelu(x)


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


@_lowering(_aten.pow.Scalar)
def _(mb, base, exp):
    return mb.pow(mb.scalar_like(exp, float(base)), exp)


@_lowering(_aten.matmul.default, _aten.bmm.default)
def _(mb, lhs, rhs):
    return mb.matmul(lhs, rhs)


@torch.library.register_fake("aten::matmul_backward")
def _(grad, self, other, mask):
    grad_self = torch.empty_like(self) if mask[0] else None
    grad_other = torch.empty_like(other) if mask[1] else None
    return grad_self, grad_other


@_lowering(_aten.matmul_backward.default)
def _(mb, grad, self, other, mask):
    return mb.matmul_backward(grad, self, other, bool(mask[0]), bool(mask[1]))


@_lowering(_aten.linear.default)
def _(mb, input, weight, bias=None):
    return mb.linear(input, weight, bias)


@_lowering(_aten.linear_backward.default)
def _(mb, self, grad_output, weight, output_mask):
    return mb.linear_backward(self, grad_output, weight, bool(output_mask[0]), bool(output_mask[1]),
                              bool(output_mask[2]))


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


@_lowering(_aten.max.dim)
def _(mb, x, dim, keepdim=False):
    # Materialise only the indices (== argmax). The values slot is left None:
    # no compile-path caller reads it, and getitem raises if one ever does.
    return (None, mb.argmax(x, int(dim), bool(keepdim)))


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


@_lowering(_aten.split.Tensor, _aten.split_with_sizes.default)
def _(mb, x, split_size_or_sizes, dim=0):
    # Multi-output: emit one contiguous slice per chunk. Chunk lengths come
    # from split_with_sizes' explicit list, or for split.Tensor are derived
    # from the input's dim length (the last chunk of an uneven split is
    # shorter). x.shape reads the value's static ranked-tensor dims off the IR.
    # Downstream getitem extracts the chunks. DTensor lowers rotary's
    # rotate_half slice pair into a split, which is why this surfaces only on
    # the multi-chip path.
    shape = x.shape
    d = int(dim) if int(dim) >= 0 else int(dim) + len(shape)
    if isinstance(split_size_or_sizes, (list, tuple)):
        sizes = [int(s) for s in split_size_or_sizes]
    else:
        size, dim_len = int(split_size_or_sizes), int(shape[d])
        sizes = [size] * (dim_len // size)
        if dim_len % size:
            sizes.append(dim_len % size)
    chunks = []
    start = 0
    for length in sizes:
        chunks.append(mb.slice(x, d, start, start + length, 1))
        start += length
    return tuple(chunks)


@_lowering(_aten.select.int)
def _(mb, x, dim, index):
    # slice keeps the indexed dim at size 1; squeeze drops it. end=None
    # covers index == -1, where index + 1 would wrap to an empty slice.
    end = None if index == -1 else int(index) + 1
    return mb.squeeze(mb.slice(x, int(dim), int(index), end, 1), int(dim))


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


@_lowering(_aten.embedding_dense_backward.default)
@_skip_prepare(_aten.embedding_dense_backward.default)
def _(mb, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq=False):
    if scale_grad_by_freq:
        raise NotImplementedError(
            "tt-kurbla compile: embedding_dense_backward with scale_grad_by_freq=True is not lowered"
        )
    return mb.embedding_backward(grad_output, indices, int(num_weights), int(padding_idx))


@_lowering(_aten.le.Scalar)
@_skip_prepare(_aten.le.Scalar)
def _(mb, x, scalar):
    return mb.le(x, mb.scalar_like(x, float(scalar)))


@_lowering(_aten.le.Tensor)
@_skip_prepare(_aten.le.Tensor)
def _(mb, lhs, rhs):
    return mb.le(lhs, rhs)


@_lowering(_aten.ge.Scalar)
@_skip_prepare(_aten.ge.Scalar)
def _(mb, x, scalar):
    return mb.ge(x, mb.scalar_like(x, float(scalar)))


@_lowering(_aten.ge.Tensor)
@_skip_prepare(_aten.ge.Tensor)
def _(mb, lhs, rhs):
    return mb.ge(lhs, rhs)


@_lowering(_aten.lt.Scalar)
@_skip_prepare(_aten.lt.Scalar)
def _(mb, x, scalar):
    return mb.lt(x, mb.scalar_like(x, float(scalar)))


@_lowering(_aten.lt.Tensor)
@_skip_prepare(_aten.lt.Tensor)
def _(mb, lhs, rhs):
    return mb.lt(lhs, rhs)


@_lowering(_aten.gt.Scalar)
@_skip_prepare(_aten.gt.Scalar)
def _(mb, x, scalar):
    return mb.gt(x, mb.scalar_like(x, float(scalar)))


@_lowering(_aten.gt.Tensor)
@_skip_prepare(_aten.gt.Tensor)
def _(mb, lhs, rhs):
    return mb.gt(lhs, rhs)


@_lowering(_aten.eq.Scalar)
@_skip_prepare(_aten.eq.Scalar)
def _(mb, x, scalar):
    return mb.eq(x, mb.scalar_like(x, float(scalar)))


@_lowering(_aten.eq.Tensor)
@_skip_prepare(_aten.eq.Tensor)
def _(mb, lhs, rhs):
    return mb.eq(lhs, rhs)


@_lowering(_aten.ne.Scalar)
@_skip_prepare(_aten.ne.Scalar)
def _(mb, x, scalar):
    return mb.ne(x, mb.scalar_like(x, float(scalar)))


@_lowering(_aten.ne.Tensor)
@_skip_prepare(_aten.ne.Tensor)
def _(mb, lhs, rhs):
    return mb.ne(lhs, rhs)


@_lowering(_aten.where.self)
@_skip_prepare(_aten.where.self)
def _(mb, condition, self, other):
    return mb.where(condition, self, other)


def _numpy_broadcast_shape(shapes):
    """numpy-style broadcast of a list of shapes (right-aligned, 1 broadcasts)."""
    ndim = max(len(s) for s in shapes)
    out = [1] * ndim
    for s in shapes:
        s = [1] * (ndim - len(s)) + list(s)
        for i, d in enumerate(s):
            if d != 1:
                if out[i] != 1 and out[i] != d:
                    raise NotImplementedError(f"tt-kurbla compile: incompatible index broadcast {shapes}")
                out[i] = d
    return out


def _broadcast_to(mb, value, target):
    """Reshape (left-pad rank) then broadcast an mb value to `target` shape."""
    shape = list(value.shape)
    if shape == target:
        return value
    if len(shape) < len(target):
        shape = [1] * (len(target) - len(shape)) + shape
        value = mb.reshape(value, shape)
    return mb.broadcast(value, target) if shape != target else value


def _normalize_neg_index(mb, idx, size):
    """Resolve torch's from-the-end negative indices before a gather (which reads
    out of bounds on them): return `idx + size` where `idx < 0`, else `idx`."""
    zero = mb.scalar_like(idx, 0.0)
    from_end = mb.add(idx, mb.scalar_like(idx, float(size)))
    return mb.where(mb.lt(idx, zero), from_end, idx)


# Advanced indexing x[..., idx, ...] (aten.index.Tensor). `indices` is a per-dim
# list of index tensors or None. Two supported shapes: a single index tensor
# (gather along that dim), or index tensors covering all leading dims (flatten to
# a 1-D linear-index gather).
@_lowering(_aten.index.Tensor)
@_skip_prepare(_aten.index.Tensor)
def _(mb, x, indices):
    non_none = [(d, idx) for d, idx in enumerate(indices) if idx is not None]
    in_shape = list(x.shape)

    if len(non_none) == 1:
        dim, idx = non_none[0]
        idx_shape = list(idx.shape)
        if len(idx_shape) != 1:
            raise NotImplementedError("tt-kurbla compile: single-index aten.index.Tensor requires a 1-D index")
        k = idx_shape[0]
        out_shape = list(in_shape)
        out_shape[dim] = k
        view_shape = [1] * len(in_shape)
        view_shape[dim] = k
        idx_i32 = _normalize_neg_index(mb, mb.typecast(idx, _native.DataType.Int32), in_shape[dim])
        idx_full = mb.broadcast(mb.reshape(idx_i32, view_shape), out_shape)
        return mb.gather(x, idx_full, dim)

    dims = [d for d, _ in non_none]
    k = len(non_none)
    if dims != list(range(k)) or k != len(in_shape):
        raise NotImplementedError(
            "tt-kurbla compile: aten.index.Tensor supports a single index, or index "
            "tensors covering all leading dims with no trailing dims; "
            f"got indexed dims {dims} on a {len(in_shape)}-D tensor"
        )
    idx_tensors = [idx for _, idx in non_none]
    out_shape = _numpy_broadcast_shape([list(t.shape) for t in idx_tensors])
    numel_out = 1
    for d in out_shape:
        numel_out *= d
    # row-major linear index over the leading (== all) dims
    linear = None
    for j, t in enumerate(idx_tensors):
        t = _normalize_neg_index(mb, t, in_shape[j])
        stride = 1
        for m in range(j + 1, k):
            stride *= in_shape[m]
        tb = _broadcast_to(mb, t, out_shape)
        term = tb if stride == 1 else mb.mul(tb, mb.scalar_like(tb, float(stride)))
        linear = term if linear is None else mb.add(linear, term, 1.0)
    flat_n = 1
    for d in in_shape:
        flat_n *= d
    x_flat = mb.reshape(x, [flat_n])
    linear_1d = mb.reshape(mb.typecast(linear, _native.DataType.Int32), [numel_out])
    gathered = mb.gather(x_flat, linear_1d, 0)
    return mb.reshape(gathered, out_shape)


@_lowering(_aten.bitwise_and.Tensor)
@_skip_prepare(_aten.bitwise_and.Tensor)
def _(mb, lhs, rhs):
    return mb.bitwise_and(lhs, rhs)


@_lowering(_aten.bitwise_or.Tensor)
@_skip_prepare(_aten.bitwise_or.Tensor)
def _(mb, lhs, rhs):
    return mb.bitwise_or(lhs, rhs)


@_lowering(_aten.bitwise_not.default)
@_skip_prepare(_aten.bitwise_not.default)
def _(mb, x):
    return mb.bitwise_not(x)


@_lowering(_aten.logical_and.default)
@_skip_prepare(_aten.logical_and.default)
def _(mb, lhs, rhs):
    return mb.logical_and(lhs, rhs)


@_lowering(_aten.logical_or.default)
@_skip_prepare(_aten.logical_or.default)
def _(mb, lhs, rhs):
    return mb.logical_or(lhs, rhs)


@_lowering(_aten.logical_not.default)
@_skip_prepare(_aten.logical_not.default)
def _(mb, x):
    return mb.logical_not(x)


@_lowering(_aten.tril.default)
@_skip_prepare(_aten.tril.default)
def _(mb, input, diagonal=0):
    return mb.tril(input, int(diagonal))


@_lowering(_aten._scaled_dot_product_fused_attention_overrideable.default)
@_skip_prepare(_aten._scaled_dot_product_fused_attention_overrideable.default)
def _(mb, query, key, value, attn_bias=None, dropout_p=0.0, is_causal=False, return_debug_mask=False, scale=None):
    if dropout_p:
        raise NotImplementedError(f"tt-kurbla sdpa: dropout_p must be 0 (inference only), got {dropout_p}")
    if return_debug_mask:
        raise NotImplementedError("tt-kurbla sdpa: return_debug_mask=True is not supported")
    result = mb.sdpa(query, key, value, is_causal=is_causal, scale=scale, attn_mask=attn_bias)
    # Returns a 9-tuple; downstream getitem[0] extracts the attention output.
    return (result, None, None, None, None, None, None, None, None)


@_lowering(_aten._to_copy.default)
@_skip_prepare(_aten._to_copy.default)
def _(mb, x, dtype=None, layout=None, device=None, pin_memory=None, non_blocking=False, memory_format=None):
    if dtype is not None:
        return mb.typecast(x, _to_runtime_dtype(dtype))
    return x


@_lowering(_aten.copy.default)
def _(mb, self, src, non_blocking=False):
    # Functionalized `copy_`: the result is `src` taken to self's shape and dtype.
    # _prepare_op_args has already cast src to the output (i.e. self's) dtype, so
    # only the broadcast is left. Shows up wherever a slice assignment gets
    # functionalized into slice + copy + slice_scatter.
    return _broadcast_to(mb, src, list(self.shape))


@_lowering(_aten.constant_pad_nd.default)
def _(mb, x, pad, value=0.0):
    # aten lists the amounts from the *last* dimension backwards as (low, high)
    # pairs, covering only the trailing dims it touches; mb.pad wants one pair
    # per dim in dim order. Negative amounts crop, which mb.pad handles.
    rank = len(x.shape)
    low = [0] * rank
    high = [0] * rank
    for i in range(len(pad) // 2):
        low[rank - 1 - i] = int(pad[2 * i])
        high[rank - 1 - i] = int(pad[2 * i + 1])
    return mb.pad(x, low, high, float(value))


@_lowering(_aten._unsafe_view.default)
def _(mb, x, size):
    return mb.reshape(x, list(size))


@_lowering(_aten.alias.default, _aten.clone.default, _aten.lift_fresh_copy.default)
@_skip_prepare(_aten.alias.default, _aten.clone.default, _aten.lift_fresh_copy.default)
def _(mb, x, **kwargs):
    # clone/alias are identity; lift_fresh_copy passes a baked-in constant tensor
    if isinstance(x, torch.Tensor):
        if x.numel() > 1:
            raise NotImplementedError(
                f"tt-kurbla compile: non-scalar constant tensor (shape {tuple(x.shape)}) not supported"
            )
        # .item() under fake mode would dispatch _local_scalar_dense
        with unset_fake_temporarily():
            value = float(x.item()) if x.numel() == 1 else 0.0
        return mb.full(list(x.shape), value, _to_runtime_dtype(x.dtype))
    return x


@_lowering(_aten.scalar_tensor.default)
@_skip_prepare(_aten.scalar_tensor.default)
def _(mb, value, dtype=None, layout=None, device=None, pin_memory=None):
    rt_dtype = _to_runtime_dtype(dtype if dtype is not None else torch.float32)
    return mb.scalar(rt_dtype, float(value))


def _default_rt_dtype(dtype):
    return _to_runtime_dtype(dtype if dtype is not None else torch.float32)


@_lowering(_aten.full.default)
@_skip_prepare(_aten.full.default)
def _(mb, size, fill_value, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    return mb.full(list(size), float(fill_value), _default_rt_dtype(dtype))


@_lowering(_aten.zeros.default)
@_skip_prepare(_aten.zeros.default)
def _(mb, size, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    return mb.zeros(list(size), _default_rt_dtype(dtype))


@_lowering(_aten.index_copy.default)
@_skip_prepare(_aten.index_copy.default)
def _(mb, input, dim, index, source):
    return mb.index_copy(input, int(dim), index, source)


@_lowering(_aten.ones.default)
@_skip_prepare(_aten.ones.default)
def _(mb, size, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    return mb.ones(list(size), _default_rt_dtype(dtype))


# new_* mirror zeros/ones/full but take a reference tensor first; unlike them,
# their dtype defaults to the reference tensor's (not float32).
@_lowering(_aten.new_zeros.default)
@_skip_prepare(_aten.new_zeros.default)
def _(mb, self, size, dtype=None, layout=None, device=None, pin_memory=None):
    if dtype is None:
        return mb.zeros_like(self, list(size))
    return mb.zeros(list(size), _to_runtime_dtype(dtype))


@_lowering(_aten.new_ones.default)
@_skip_prepare(_aten.new_ones.default)
def _(mb, self, size, dtype=None, layout=None, device=None, pin_memory=None):
    if dtype is None:
        return mb.ones_like(self, list(size))
    return mb.ones(list(size), _to_runtime_dtype(dtype))


@_lowering(_aten.new_full.default)
@_skip_prepare(_aten.new_full.default)
def _(mb, self, size, fill_value, dtype=None, layout=None, device=None, pin_memory=None):
    if dtype is None:
        return mb.full_like(self, list(size), float(fill_value))
    return mb.full(list(size), float(fill_value), _to_runtime_dtype(dtype))


# full_like is new_full with the shape taken from the reference tensor instead of
# passed in. zeros_like/ones_like arrive here too: core aten decomposes them to
# full_like rather than giving them their own op.
@_lowering(_aten.full_like.default)
@_skip_prepare(_aten.full_like.default)
def _(mb, self, fill_value, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    shape = list(self.shape)
    if dtype is None:
        return mb.full_like(self, shape, float(fill_value))
    return mb.full(shape, float(fill_value), _to_runtime_dtype(dtype))


def _is_tensor_schema_arg(
    idx: int,
    schema: torch._C.FunctionSchema,
) -> bool:
    return idx < len(schema.arguments) and "Tensor" in str(schema.arguments[idx].type)


def _cluster_axis_for_group(group_name: str) -> int:
    """The runtime cluster axis of the mesh dim behind `group_name` — read off
    the eager `TTProcessGroup.cluster_axis` (derived from rank composition)."""
    from torch.distributed.distributed_c10d import _resolve_process_group

    return _resolve_process_group(group_name).cluster_axis


@_lowering(_funcol.all_reduce.default)
def _(mb, input, reduce_op, group_name):
    return mb.all_reduce(input, reduce_op, _cluster_axis_for_group(group_name))


@_lowering(_funcol.all_gather_into_tensor.default)
def _(mb, input, group_size, group_name):
    return mb.all_gather(input, group_size, _cluster_axis_for_group(group_name))


@_lowering(_funcol.reduce_scatter_tensor.default)
def _(mb, input, reduce_op, group_size, group_name):
    # The c10d functional reduce_scatter always scatters dim 0 (funcol moves any
    # other dim there itself). Sum only, matching build_reduce_scatter.
    return mb.reduce_scatter(input, group_size, _cluster_axis_for_group(group_name), 0)


@_lowering(torch.ops.tt_kurbla.reduce_scatter.default)
def _(mb, input, group_name, group_size, scatter_dim):
    # Our own reduce_scatter carries the real shard dim (used by the
    # Replicate->Shard redistribute patch); scatter it directly.
    return mb.reduce_scatter(input, group_size, _cluster_axis_for_group(group_name), scatter_dim)


@_lowering(_funcol.wait_tensor.default)
def _(mb, input):
    # TODO: Should we wait here?
    return input


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


class CompileOption(StrEnum):
    """Compile options in ``torch.compile(model, backend="tt", options={...})``."""

    OPT_LEVEL = "optimization_level" # int
    EXPERIMENTAL_WEIGHT_DTYPE = "experimental_weight_dtype" # BfpDtype
    EXPERIMENTAL_KV_CACHE_DTYPE = "experimental_kv_cache_dtype" # BfpDtype
    MATH_FIDELITY = "math_fidelity" # MathFidelity
    FP32_DEST_ACC_EN = "fp32_dest_acc_en" # bool
    EXPERIMENTAL_ENABLE_FUSING_CONV2D_WITH_MULTIPLY_PATTERN = "experimental_enable_fusing_conv2d_with_multiply_pattern" # bool
    EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION = "experimental_enable_permute_matmul_fusion" # bool
    ENABLE_TRACE = "enable_trace" # bool
    ENABLE_CONST_EVAL = "enable_const_eval" # bool
    ENABLE_CONST_EVAL_ON_CPU = "enable_const_eval_on_cpu" # bool
    ENABLE_CONST_EVAL_INPUTS_TO_SYSTEM_MEMORY = "enable_const_eval_inputs_to_system_memory" # bool
    EXPERIMENTAL_ENABLE_DRAM_SPACE_SAVING_OPTIMIZATION = "experimental_enable_dram_space_saving_optimization" # bool
    ENABLE_CREATE_D2M_SUBGRAPHS = "enable_create_d2m_subgraphs" # bool
    TTNN_PERF_METRICS_ENABLED = "ttnn_perf_metrics_enabled" # bool
    TTNN_PERF_METRICS_OUTPUT_FILE = "ttnn_perf_metrics_output_file" # str


COMPILE_OPTIONS = [opt for opt in CompileOption]

# Exported native enums so callers can set the typed options directly.
# e.g. options={
#   CompileOption.MATH_FIDELITY: MathFidelity.HiFi4,
#   CompileOptions.EXPERIMENTAL_WEIGHT_DTYPE: BfpDtype.BfpBf4
# }
BfpDtype = _native.BfpDtype          # BfpBf8, BfpBf4
MathFidelity = _native.MathFidelity  # LoFi, HiFi2, HiFi3, HiFi4

def _compile_options_dict(options: _native.CompileOptions) -> dict[str, object]:
    """JSON-ready view of the options a graph was compiled with, for a dump's
    `artifacts.json`. `CompileOption`'s values are the bound attribute names, so this
    stays in step with the native struct on its own.

    An option the user never set reads back as `None` (the underlying field is a
    `std::optional`), which is worth recording: it means "tt-mlir's default", not
    "off".
    """

    def value_of(option: CompileOption):
        value = getattr(options, option.value)
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        # BfpDtype / MathFidelity, and anything else native and enum-shaped.
        return getattr(value, "name", None) or str(value)

    return {opt.value: value_of(opt) for opt in COMPILE_OPTIONS}


def _compile_options(options: dict [CompileOption, str | int | bool] | None) -> _native.CompileOptions:
    """Converts python dict with CompileOption to _native.CompileOptions"""

    opts = _native.CompileOptions() # default options from config.hpp
    if (options is None):
        return opts

    unknown = options.keys() - COMPILE_OPTIONS
    if unknown:
        raise ValueError(f"Unknown compile option(s) {sorted(unknown)}; supported: {sorted(COMPILE_OPTIONS)}")

    if (CompileOption.OPT_LEVEL in options):
        opts.optimization_level = options[CompileOption.OPT_LEVEL]

    if (CompileOption.EXPERIMENTAL_WEIGHT_DTYPE in options):
        opts.experimental_weight_dtype = options[CompileOption.EXPERIMENTAL_WEIGHT_DTYPE]

    if (CompileOption.EXPERIMENTAL_KV_CACHE_DTYPE in options):
        opts.experimental_kv_cache_dtype = options[CompileOption.EXPERIMENTAL_KV_CACHE_DTYPE]

    if (CompileOption.MATH_FIDELITY in options):
        opts.math_fidelity = options[CompileOption.MATH_FIDELITY]

    if (CompileOption.FP32_DEST_ACC_EN in options):
        opts.fp32_dest_acc_en = options[CompileOption.FP32_DEST_ACC_EN]

    if (CompileOption.EXPERIMENTAL_ENABLE_FUSING_CONV2D_WITH_MULTIPLY_PATTERN in options):
        opts.experimental_enable_fusing_conv2d_with_multiply_pattern = options[CompileOption.EXPERIMENTAL_ENABLE_FUSING_CONV2D_WITH_MULTIPLY_PATTERN]

    if (CompileOption.EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION in options):
        opts.experimental_enable_permute_matmul_fusion = options[CompileOption.EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION]

    if (CompileOption.ENABLE_TRACE in options):
        opts.enable_trace = options[CompileOption.ENABLE_TRACE]

    if (CompileOption.ENABLE_CONST_EVAL in options):
        opts.enable_const_eval = options[CompileOption.ENABLE_CONST_EVAL]

    if (CompileOption.ENABLE_CONST_EVAL_ON_CPU in options):
        opts.enable_const_eval_on_cpu = options[CompileOption.ENABLE_CONST_EVAL_ON_CPU]

    if (CompileOption.ENABLE_CONST_EVAL_INPUTS_TO_SYSTEM_MEMORY in options):
        opts.enable_const_eval_inputs_to_system_memory = options[CompileOption.ENABLE_CONST_EVAL_INPUTS_TO_SYSTEM_MEMORY]

    if (CompileOption.EXPERIMENTAL_ENABLE_DRAM_SPACE_SAVING_OPTIMIZATION in options):
        opts.experimental_enable_dram_space_saving_optimization = options[CompileOption.EXPERIMENTAL_ENABLE_DRAM_SPACE_SAVING_OPTIMIZATION]

    if (CompileOption.ENABLE_CREATE_D2M_SUBGRAPHS in options):
        opts.enable_create_d2m_subgraphs = options[CompileOption.ENABLE_CREATE_D2M_SUBGRAPHS]

    if (CompileOption.TTNN_PERF_METRICS_ENABLED in options):
        opts.ttnn_perf_metrics_enabled = options[CompileOption.TTNN_PERF_METRICS_ENABLED]

    if (CompileOption.TTNN_PERF_METRICS_OUTPUT_FILE in options):
        opts.ttnn_perf_metrics_output_file = options[CompileOption.TTNN_PERF_METRICS_OUTPUT_FILE]

    return opts



def _fw_args_roles(num_inputs: int, fw_meta) -> list["_native.ArgumentType"]:
    """Tags the forward graph's lifted weight/buffer args ``Parameter``.

    aot_module_simplified lifts module params/buffers as leading graph args; their
    indices are in fw_metadata.static_input_indices and align 1:1 with the forward
    placeholders. Tagging them ``Parameter`` lets tt-mlir's const-eval hoist fold
    weight-only subgraphs - the compiler does its own per-function dataflow over
    these args, so only the args (not interior nodes) need marking. Graph-mutated
    args (e.g. KV caches written via index_copy_) stay ``Input``: freezing them
    would serve stale values.
    """
    roles = [_native.ArgumentType.Input] * num_inputs
    if fw_meta is None:
        return roles
    mutated = {i for i, info in enumerate(fw_meta.input_info) if info.mutates_data}
    for i in fw_meta.static_input_indices:
        if i < num_inputs and i not in mutated:
            roles[i] = _native.ArgumentType.Parameter
    return roles

def _bw_args_roles(num_inputs: int) -> list["_native.ArgumentType"]:
    """Tags the backward graph's args as ``Input``.

    aot_module_simplified runs the backward compile under the *forward* tracing context,
    so the only metadata available describes forward inputs, not the backward graph's
    own inputs (saved values + tangents).

    More fundamentally, const-eval pays off only when a weight-derived value is
    stable across calls - but in training the optimizer re-versions every weight
    each step, so weight-derived backward const-eval entries are written then
    invalidated before they are read (~zero benefit), and any mis-tag there is the
    one path to a stale-gradient wrong result. So we don't tag it.
    """
    return [_native.ArgumentType.Input] * num_inputs

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
        if isinstance(val, (tuple, list)):
            val = next(v for v in val if v is not None)
        target_dtype = _to_runtime_dtype(val.dtype)
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


# Hook which enables tests to analyze the fx graph AOTAutograd provides to us.
_post_aot_fx_hook: Callable[[torch.fx.GraphModule], None] | None = None


def _aot_graph_kind() -> str | None:
    """`forward` / `backward` / `inference` for the graph aot has us compiling, taken
    from its `<aot id>_<kind>` tag.
    """
    tag = getattr(torch._guards.TracingContext.try_get(), "aot_graph_name", None) or ()
    return "_".join(tag).rpartition("_")[2] or None


def _lower_and_compile(
    gm: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    roles: list["_native.ArgumentType"],
    *,
    options: _native.CompileOptions
) -> Callable:
    """Lower one post-aot FX graph (forward or backward) to a runnable program.

    Walks the graph once via _TTIRInterpreter, finalizes the accumulated TTIR
    module, compiles it to a flatbuffer, and returns a runner closure that binds
    inputs and runs the compiled program on each call. `roles` tags each graph
    arg for const-eval (see tt_backend / _forward_parameter_roles).
    """
    if _post_aot_fx_hook is not None:
        _post_aot_fx_hook(gm)
    specs = [_spec_from_tensor(t) for t in example_inputs]
    mb = _native.ModuleBuilder(specs, roles)
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

    # Capturing the TTIR costs a full module print, so only pay for it when
    # something is actually collecting artifacts.
    dumping_artifacts = is_artifacts_dumper_active()
    result = mb.compile(outputs, options, capture_ttir=dumping_artifacts)
    program = result.program

    if dumping_artifacts:
        register_artifact(Artifact(_compile_options_dict(options), result, _aot_graph_kind()))

    def runner(*inputs: torch.Tensor) -> list:
        produced = iter(_native.run_program(program, list(inputs), output_dtypes))
        return [None if is_none else next(produced) for is_none in none_mask]

    return runner


def _empty_like_decomp(self, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    # No uninitialized-allocation lowering; materialize as zeros ("empty" content
    # is unspecified anyway, and zeros avoids garbage if the buffer is read).
    return self.new_zeros(self.shape, dtype=dtype if dtype is not None else self.dtype)


def _fill_scalar_decomp(self, value):
    # fill.Scalar: self's shape/dtype filled with `value`, via new_full.
    return self.new_full(self.shape, value, dtype=self.dtype)


def _new_empty_strided_decomp(self, size, stride, dtype=None, layout=None, device=None, pin_memory=None):
    contiguous, acc = [], 1
    for dim in reversed(size):
        contiguous.insert(0, acc)
        acc *= dim
    if any(s != c for d, s, c in zip(size, stride, contiguous) if d > 1):
        raise NotImplementedError(
            f"tt-kurbla compile: new_empty_strided with non-contiguous strides {list(stride)} "
            f"for size {list(size)} (contiguous would be {contiguous})"
        )
    return self.new_zeros(size, dtype=dtype if dtype is not None else self.dtype)


# A few scatter decomps we rely on aren't in the core set; pull them in explicitly.
_EXTRA_DECOMP_OPS = [
    torch.ops.aten.slice_scatter,
]


def _build_decomposition_table():
    # Use default core decompositions, plus a few extra and some custom ones.
    table = dict(core_aten_decompositions())
    table.update(get_decompositions(_EXTRA_DECOMP_OPS))
    table.update({
        torch.ops.aten.empty_like.default: _empty_like_decomp,
        torch.ops.aten.fill.Scalar: _fill_scalar_decomp,
        torch.ops.aten.new_empty_strided.default: _new_empty_strided_decomp,
    })
    # Never decompose an op tt lowers directly — keep it as a leaf for its kernel.
    return {op: fn for op, fn in table.items() if op not in _LOWERINGS}


_TT_DECOMPOSITIONS = _build_decomposition_table()


def tt_backend(
    gm: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    options: dict [CompileOption, str | int | bool] | None = None,
):
    """Top-level dynamo backend. Delegates to aot_module_simplified."""
    lower_and_compile = functools.partial(_lower_and_compile, options=_compile_options(options))

    def fw_compiler(fw_gm: torch.fx.GraphModule, fw_inputs: list[torch.Tensor]) -> Callable:
        fw_meta = getattr(torch._guards.TracingContext.try_get(), "fw_metadata", None)
        roles = _fw_args_roles(len(fw_inputs), fw_meta)
        return lower_and_compile(fw_gm, fw_inputs, roles)

    def bw_compiler(bw_gm: torch.fx.GraphModule, bw_inputs: list[torch.Tensor]) -> Callable:
        roles = _bw_args_roles(len(bw_inputs))
        return lower_and_compile(bw_gm, bw_inputs, roles)

    return aot_module_simplified(
        gm, example_inputs, fw_compiler=fw_compiler, bw_compiler=bw_compiler,
        decompositions=_TT_DECOMPOSITIONS,
    )


# Self-register on import. After this, `torch.compile(model, backend="tt")` works.
torch._dynamo.register_backend(name="tt", compiler_fn=tt_backend)
