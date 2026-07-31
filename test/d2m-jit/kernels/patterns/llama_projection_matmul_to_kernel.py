# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DRAM-streamed matmul lowering for Llama prefill projection weights."""

import math
import torch

import d2m_jit as d2m
from ttmlir import ir

try:
    from runner import InputSpec, KernelBench, PatternTest
except ModuleNotFoundError as exc:
    if exc.name != "runner":
        raise

    class PatternTest:
        def __init__(self, **kwargs):
            pass

    class InputSpec:
        def __init__(self, *args, **kwargs):
            pass

    class KernelBench:
        def __init__(self, **kwargs):
            pass


_PHYSICAL_GRID = 8
_N_BLOCK_TILES = 4
_K_BLOCK_TILES = 8


@d2m.kernel
def llama_projection_matmul_transposed_weight(
    lhs,
    weight,
    out,
    out_physical,
    m_tiles,
    n_tiles,
    m_blocks,
    n_blocks,
    k_tiles,
    grid_rows,
    grid_cols,
):
    for m_block in range(m_blocks):
        m = core_index(0) + m_block * grid_rows
        for n_block in range(n_blocks):
            n = core_index(1) + n_block * grid_cols
            acc = zeros([1, 1], dtype="bf16")
            for k in range(k_tiles):
                lhs_tile = remote_load(lhs, [m, k])
                weight_tile = remote_load(weight, [n, k])
                acc = matmul(
                    lhs_tile,
                    weight_tile,
                    transpose_b=True,
                    acc=acc,
                )
            row_major = untilize_block(acc)
            remote_store(out, [m, n], row_major)


@d2m.kernel
def llama_projection_matmul_blocked_2x4_transposed_weight(
    lhs,
    weight,
    out,
    out_physical,
    m_blocks,
    n_blocks,
    k_blocks,
    grid_rows,
    grid_cols,
):
    for m_block in range(m_blocks):
        m = core_index(0) + m_block * grid_rows
        for n_block in range(n_blocks):
            n = core_index(1) + n_block * grid_cols
            acc = zeros([2, 4], dtype="bf16")
            for k in range(k_blocks):
                lhs_block = remote_load(
                    lhs,
                    [m, k],
                    mcast_start_index=[core_index(0), 0],
                    mcast_shape=[1, grid_cols],
                )
                weight_block = remote_load(
                    weight,
                    [n, k],
                    mcast_start_index=[0, core_index(1)],
                    mcast_shape=[grid_rows, 1],
                )
                acc = matmul(
                    lhs_block,
                    weight_block,
                    transpose_b=True,
                    acc=acc,
                )
            row_major = untilize_block(acc)
            remote_store(out, [m, n], row_major)


def _shape(value):
    try:
        return tuple(ir.RankedTensorType(value.type).shape)
    except (TypeError, ValueError):
        return None


def _array_attr(op, name):
    try:
        return tuple(ir.DenseI64ArrayAttr(op.attributes[name]))
    except (KeyError, TypeError, ValueError):
        return None


def _peel_transposed_weight(value):
    permute = value.owner
    if (
        permute is None
        or permute.name != "ttir.permute"
        or _array_attr(permute, "permutation") != (1, 0)
    ):
        return None

    source = permute.operands[0]
    while True:
        owner = source.owner
        if getattr(owner, "name", None) != "ttir.reshape":
            break
        source = owner.operands[0]
    return source


def _match_projection_matmul(op):
    if len(op.operands) != 2 or len(op.results) != 1:
        return None

    lhs_type = ir.RankedTensorType(op.operands[0].type)
    rhs_type = ir.RankedTensorType(op.operands[1].type)
    result_type = ir.RankedTensorType(op.results[0].type)
    if not all(
        ir.BF16Type.isinstance(t.element_type)
        for t in (lhs_type, rhs_type, result_type)
    ):
        return None

    lhs_shape = tuple(lhs_type.shape)
    rhs_shape = tuple(rhs_type.shape)
    result_shape = tuple(result_type.shape)
    if len(lhs_shape) != 2 or len(rhs_shape) != 2 or len(result_shape) != 2:
        return None
    m, k = lhs_shape
    rhs_k, n = rhs_shape
    if rhs_k != k or result_shape != (m, n):
        return None
    if min(m, n, k) <= 0 or any(dim % 32 != 0 for dim in (m, n, k)):
        return None

    if op.name == "ttir.dot_general":
        if (
            _array_attr(op, "batch_dims_lhs") != ()
            or _array_attr(op, "batch_dims_rhs") != ()
            or _array_attr(op, "contract_dims_lhs") != (1,)
            or _array_attr(op, "contract_dims_rhs") != (0,)
        ):
            return None
    elif op.name == "ttir.matmul":
        try:
            if bool(ir.BoolAttr(op.attributes["transpose_a"]).value) or bool(
                ir.BoolAttr(op.attributes["transpose_b"]).value
            ):
                return None
        except (KeyError, TypeError, ValueError):
            return None
    else:
        return None

    weight = _peel_transposed_weight(op.operands[1])
    if weight is None or _shape(weight) != (n, k):
        return None
    return op.operands[0], weight, m, n, k


def _match_terminal_projection_reshape(op):
    if getattr(op, "name", None) != "ttir.reshape" or len(op.operands) != 1:
        return None

    matmul_op = op.operands[0].owner
    match = _match_projection_matmul(matmul_op)
    if match is None:
        return None

    lhs_value, weight_value, m, n, k = match
    result_shape = _shape(op.results[0])
    if (
        result_shape is None
        or len(result_shape) != 3
        or math.prod(result_shape[:-1]) != m
        or result_shape[-1] != n
        or result_shape[0] % 32 != 0
    ):
        return None

    uses = list(op.results[0].uses)
    if len(uses) != 1 or uses[0].owner.name != "func.return":
        return None

    return lhs_value, weight_value, m, n, k, result_shape


def _matches_terminal_projection_reshape(op):
    try:
        return _match_terminal_projection_reshape(op) is not None
    except (IndexError, TypeError, ValueError):
        return False


def _matches_projection_matmul(op):
    try:
        match = _match_projection_matmul(op)
        if match is None:
            return False
        return not any(
            _match_terminal_projection_reshape(use.owner) is not None
            for use in op.results[0].uses
        )
    except (IndexError, TypeError, ValueError):
        return False


def _largest_grid_divisor(tile_count):
    for candidate in range(min(_PHYSICAL_GRID, tile_count), 0, -1):
        if tile_count % candidate == 0:
            return candidate
    raise AssertionError("positive tile count must have a grid divisor")


def _lower_projection(lhs_value, weight_value, m, n, k, output_shape):
    m_tiles = m // 32
    n_tiles = n // 32
    k_tiles = k // 32
    # Use the blocked kernel only when its 2x4 output blocks fill the 8x8 grid.
    use_blocked = (
        m_tiles % (2 * _PHYSICAL_GRID) == 0
        and n_tiles % (_N_BLOCK_TILES * _PHYSICAL_GRID) == 0
        and k_tiles % _K_BLOCK_TILES == 0
    )
    m_block_tiles = 2 if use_blocked else 1
    n_block_tiles = _N_BLOCK_TILES if use_blocked else 1
    k_block_tiles = _K_BLOCK_TILES if use_blocked else 1
    m_block_count = m_tiles // m_block_tiles
    n_block_count = n_tiles // n_block_tiles
    k_block_count = k_tiles // k_block_tiles
    k_storage_grid = _largest_grid_divisor(k_block_count)
    m_storage_grid = _largest_grid_divisor(m_block_count)
    n_storage_grid = _largest_grid_divisor(n_block_count)
    grid_rows = m_storage_grid
    grid_cols = n_storage_grid
    m_blocks = m_block_count // grid_rows
    n_blocks = n_block_count // grid_cols

    lhs_layout = d2m.Layout(
        shape=(m, k),
        dtype=d2m.bfloat16,
        block_shape=[m_block_tiles, k_block_tiles],
        grid_shape=[m_storage_grid, k_storage_grid],
        mem_space="dram",
    )
    weight_layout = d2m.Layout(
        shape=(n, k),
        dtype=d2m.bfloat16,
        block_shape=[n_block_tiles, k_block_tiles],
        grid_shape=[n_storage_grid, k_storage_grid],
        mem_space="dram",
    )
    out_layout = d2m.Layout(
        shape=(m, n),
        dtype=d2m.bfloat16,
        block_shape=[m_block_tiles * 32, n_block_tiles * 32],
        grid_shape=[m_storage_grid, n_storage_grid],
        tiled=False,
        mem_space="dram",
    )

    lhs = d2m.to_layout(d2m.from_value(lhs_value), lhs_layout)
    weight = d2m.to_layout(d2m.from_value(weight_value), weight_layout)
    out = d2m.empty(out_layout)
    out_physical = d2m.from_value(out.unblocked_value, out_layout)

    old_use_tile_matmul = d2m.config.use_tile_matmul
    d2m.config.use_tile_matmul = False
    try:
        if use_blocked:
            llama_projection_matmul_blocked_2x4_transposed_weight(
                lhs,
                weight,
                out,
                out_physical,
                m_blocks,
                n_blocks,
                k_block_count,
                grid_rows,
                grid_cols,
                grid=(grid_rows, grid_cols),
                num_outs=2,
            )
        else:
            llama_projection_matmul_transposed_weight(
                lhs,
                weight,
                out,
                out_physical,
                m_tiles,
                n_tiles,
                m_blocks,
                n_blocks,
                k_tiles,
                grid_rows,
                grid_cols,
                grid=(grid_rows, grid_cols),
                num_outs=2,
            )
    finally:
        d2m.config.use_tile_matmul = old_use_tile_matmul

    return d2m.from_device(
        out_physical,
        physical_storage=True,
        output_shape=output_shape,
    )


@d2m.pattern(
    root="ttir.reshape",
    benefit=60,
    match=_matches_terminal_projection_reshape,
)
def lower_terminal_llama_projection_reshape(op, rewriter):
    lhs_value, weight_value, m, n, k, result_shape = _match_terminal_projection_reshape(
        op
    )
    return _lower_projection(
        lhs_value,
        weight_value,
        m,
        n,
        k,
        result_shape,
    )


@d2m.pattern(
    root="ttir.matmul",
    benefit=50,
    match=_matches_projection_matmul,
)
@d2m.pattern(
    root="ttir.dot_general",
    benefit=50,
    match=_matches_projection_matmul,
)
def lower_llama_projection_matmul(op, rewriter):
    lhs_value, weight_value, m, n, k = _match_projection_matmul(op)
    return _lower_projection(lhs_value, weight_value, m, n, k, (m, n))


def _golden(lhs, weight):
    return lhs @ weight.transpose(-2, -1)


def _terminal_reshape_golden(lhs, weight):
    return (lhs @ weight.transpose(-2, -1)).reshape(32, 2, 64)


def _full_shape_golden(lhs, weight):
    return (lhs @ weight.transpose(-2, -1)).reshape(32, 18, 8, 128).permute(0, 2, 1, 3)


def _rms_projection_inputs(shape, dtype, generator):
    shape = tuple(shape)
    values = torch.rand(shape, generator=generator, dtype=torch.float32)
    if shape == (32, 18, 4096):
        row_scales = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0], dtype=torch.float32)
        scales = row_scales[
            torch.arange(32 * 18).remainder(row_scales.numel())
        ].reshape(32, 18, 1)
        values = (values * 2.0 - 1.0) * scales
    elif shape == (4096,):
        values = 0.5 + values
    else:
        values = (values * 2.0 - 1.0) / 64.0
    return values.to(dtype)


def _rms_flatten_golden(input_value, gamma):
    input_f32 = input_value.float()
    inverse_rms = torch.rsqrt(input_f32.square().mean(dim=-1, keepdim=True) + 1.0e-5)
    normalized = (input_f32 * inverse_rms).to(torch.bfloat16)
    scaled = (normalized * gamma).to(torch.bfloat16)
    return scaled.reshape(576, 4096)


def _rms_projection_golden(input_value, gamma, weight):
    return _rms_flatten_golden(input_value, gamma) @ weight.to(
        torch.bfloat16
    ).transpose(0, 1)


def _computed_flatten_golden(input_value):
    doubled = (input_value + input_value).to(torch.bfloat16)
    return doubled.reshape(576, 4096)


def _full_shape_projection_run(kernel, inputs, cfg):
    lhs_value, weight_value = inputs
    m, k = lhs_value.shape
    n = weight_value.shape[0]
    m_tiles = m // 32
    n_tiles = n // 32
    k_tiles = k // 32
    m_grid = _largest_grid_divisor(m_tiles)
    n_grid = _largest_grid_divisor(n_tiles)
    k_grid = _largest_grid_divisor(k_tiles)
    lhs_layout = d2m.Layout(
        shape=(m, k),
        dtype=d2m.bfloat16,
        block_shape=[1, 1],
        grid_shape=[m_grid, k_grid],
        mem_space="dram",
    )
    weight_layout = d2m.Layout(
        shape=(n, k),
        dtype=d2m.bfloat16,
        block_shape=[1, 1],
        grid_shape=[n_grid, k_grid],
        mem_space="dram",
    )
    out_layout = d2m.Layout(
        shape=(m, n),
        dtype=d2m.bfloat16,
        block_shape=[32, 32],
        grid_shape=[m_grid, n_grid],
        tiled=False,
        mem_space="dram",
    )
    out = d2m.empty(out_layout)
    out_physical = d2m.from_value(out.unblocked_value, out_layout)
    kernel(
        d2m.to_layout(lhs_value, lhs_layout),
        d2m.to_layout(weight_value, weight_layout),
        out,
        out_physical,
        m_tiles,
        n_tiles,
        m_tiles // m_grid,
        n_tiles // n_grid,
        k_tiles,
        m_grid,
        n_grid,
        grid=(m_grid, n_grid),
        num_outs=2,
    )
    return out_physical.to_host()


PATTERN_TESTS = [
    PatternTest(
        name="llama_projection_matmul_e2e",
        ttir="""
        module {
          func.func @forward(
              %lhs: tensor<64x96xbf16>,
              %weight: tensor<64x96xbf16>) -> tensor<64x64xbf16> {
            %weight_t = "ttir.permute"(%weight) <{
              permutation = array<i64: 1, 0>
            }> : (tensor<64x96xbf16>) -> tensor<96x64xbf16>
            %out = "ttir.dot_general"(%lhs, %weight_t) <{
              batch_dims_lhs = array<i64>,
              batch_dims_rhs = array<i64>,
              contract_dims_lhs = array<i64: 1>,
              contract_dims_rhs = array<i64: 0>
            }> : (tensor<64x96xbf16>, tensor<96x64xbf16>) -> tensor<64x64xbf16>
            return %out : tensor<64x64xbf16>
          }
        }
        """,
        check="""
        CHECK-LABEL: func.func @forward
        CHECK-NOT: ttir.dot_general
        CHECK-NOT: ttir.permute
        CHECK: grid = #ttcore.grid<2x2>
        CHECK: d2m.remote_load
        CHECK: d2m.tile_matmul
        CHECK: d2m.remote_store
        CHECK: d2m.independent_loop
        """,
        golden=_golden,
        inputs=InputSpec("uniform(-0.125,0.125)", seed=0),
        pcc=0.99,
        use_tile_matmul=False,
        e2e=True,
    ),
    PatternTest(
        name="llama_terminal_projection_reshape_e2e",
        ttir="""
        module {
          func.func @forward(
              %lhs: tensor<64x96xbf16>,
              %weight: tensor<64x96xbf16>) -> tensor<32x2x64xbf16> {
            %weight_t = "ttir.permute"(%weight) <{
              permutation = array<i64: 1, 0>
            }> : (tensor<64x96xbf16>) -> tensor<96x64xbf16>
            %out = "ttir.dot_general"(%lhs, %weight_t) <{
              batch_dims_lhs = array<i64>,
              batch_dims_rhs = array<i64>,
              contract_dims_lhs = array<i64: 1>,
              contract_dims_rhs = array<i64: 0>
            }> : (tensor<64x96xbf16>, tensor<96x64xbf16>) -> tensor<64x64xbf16>
            %reshaped = "ttir.reshape"(%out) <{
              shape = [32 : i32, 2 : i32, 64 : i32]
            }> :
              (tensor<64x64xbf16>) -> tensor<32x2x64xbf16>
            return %reshaped : tensor<32x2x64xbf16>
          }
        }
        """,
        check="""
        CHECK-LABEL: func.func @forward
        CHECK-NOT: ttir.dot_general
        CHECK-NOT: ttir.reshape
        CHECK: grid = #ttcore.grid<2x2>
        CHECK: d2m.tile_untilize_block
        CHECK: d2m.remote_store
        CHECK: d2m.to_layout
        CHECK-SAME: tensor<2x2x32x32xbf16
        CHECK-SAME: -> tensor<32x2x64xbf16>
        """,
        golden=_terminal_reshape_golden,
        inputs=InputSpec("uniform(-0.125,0.125)", seed=0),
        pcc=0.99,
        use_tile_matmul=False,
        e2e=True,
    ),
    PatternTest(
        name="llama_projection_matmul_full_shape_lowers",
        ttir="""
        module {
          func.func @forward(
              %lhs: tensor<576x4096xbf16>,
              %weight: tensor<1024x4096xbf16>) -> tensor<32x8x18x128xbf16> {
            %weight_t = "ttir.permute"(%weight) <{
              permutation = array<i64: 1, 0>
            }> : (tensor<1024x4096xbf16>) -> tensor<4096x1024xbf16>
            %out = "ttir.dot_general"(%lhs, %weight_t) <{
              batch_dims_lhs = array<i64>,
              batch_dims_rhs = array<i64>,
              contract_dims_lhs = array<i64: 1>,
              contract_dims_rhs = array<i64: 0>
            }> : (tensor<576x4096xbf16>, tensor<4096x1024xbf16>) -> tensor<576x1024xbf16>
            %reshaped = "ttir.reshape"(%out) <{
              shape = [32 : i32, 18 : i32, 8 : i32, 128 : i32]
            }> : (tensor<576x1024xbf16>) -> tensor<32x18x8x128xbf16>
            %permuted = "ttir.permute"(%reshaped) <{
              permutation = array<i64: 0, 2, 1, 3>
            }> : (tensor<32x18x8x128xbf16>) -> tensor<32x8x18x128xbf16>
            return %permuted : tensor<32x8x18x128xbf16>
          }
        }
        """,
        check="""
        CHECK-LABEL: func.func @forward
        CHECK-NOT: ttir.dot_general
        CHECK-NOT: ttir.permute
        CHECK: grid = #ttcore.grid<6x8>
        CHECK: d2m.tile_matmul
        CHECK: d2m.independent_loop
        """,
    ),
    PatternTest(
        name="llama_computed_flatten_e2e",
        ttir="""
        module {
          func.func @forward(
              %input: tensor<32x18x4096xbf16>) -> tensor<576x4096xbf16> {
            %doubled = "ttir.add"(%input, %input) :
              (tensor<32x18x4096xbf16>, tensor<32x18x4096xbf16>) ->
              tensor<32x18x4096xbf16>
            %flat = "ttir.reshape"(%doubled) <{
              shape = [576 : i32, 4096 : i32]
            }> : (tensor<32x18x4096xbf16>) -> tensor<576x4096xbf16>
            return %flat : tensor<576x4096xbf16>
          }
        }
        """,
        golden=_computed_flatten_golden,
        inputs=InputSpec(_rms_projection_inputs, seed=0),
        pcc=0.99,
        e2e=True,
    ),
    PatternTest(
        name="llama_rmsnorm_projection_composed_e2e",
        ttir="""
        module {
          func.func @forward(
              %input: tensor<32x18x4096xbf16>,
              %gamma: tensor<4096xbf16>,
              %weight: tensor<32x4096xbf16>) -> tensor<576x32xbf16> {
            %two = "ttir.full"() <{
              fill_value = 2.000000e+00 : f32,
              shape = array<i32: 32, 18, 4096>
            }> : () -> tensor<32x18x4096xf32>
            %inverse_hidden = "ttir.full"() <{
              fill_value = 2.44140625E-4 : f32,
              shape = array<i32: 32, 18>
            }> : () -> tensor<32x18xf32>
            %epsilon = "ttir.full"() <{
              fill_value = 9.99999974E-6 : f32,
              shape = array<i32: 32, 18, 1>
            }> : () -> tensor<32x18x1xf32>
            %gamma_3d = "ttir.reshape"(%gamma) <{
              shape = [1 : i32, 1 : i32, 4096 : i32]
            }> : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
            %gamma_broadcast = "ttir.broadcast"(%gamma_3d) <{
              broadcast_dimensions = array<i64: 32, 18, 1>
            }> : (tensor<1x1x4096xbf16>) -> tensor<32x18x4096xbf16>
            %input_f32 = "ttir.typecast"(%input) <{
              conservative_folding = false
            }> : (tensor<32x18x4096xbf16>) -> tensor<32x18x4096xf32>
            %squared = "ttir.pow"(%input_f32, %two) :
              (tensor<32x18x4096xf32>, tensor<32x18x4096xf32>) ->
              tensor<32x18x4096xf32>
            %sum = "ttir.sum"(%squared) <{
              dim_arg = [2 : i32], keep_dim = false
            }> : (tensor<32x18x4096xf32>) -> tensor<32x18xf32>
            %mean = "ttir.multiply"(%sum, %inverse_hidden) :
              (tensor<32x18xf32>, tensor<32x18xf32>) -> tensor<32x18xf32>
            %mean_3d = "ttir.reshape"(%mean) <{
              shape = [32 : i32, 18 : i32, 1 : i32]
            }> : (tensor<32x18xf32>) -> tensor<32x18x1xf32>
            %variance = "ttir.add"(%mean_3d, %epsilon) :
              (tensor<32x18x1xf32>, tensor<32x18x1xf32>) ->
              tensor<32x18x1xf32>
            %inverse_rms = "ttir.rsqrt"(%variance) :
              (tensor<32x18x1xf32>) -> tensor<32x18x1xf32>
            %inverse_rms_broadcast = "ttir.broadcast"(%inverse_rms) <{
              broadcast_dimensions = array<i64: 1, 1, 4096>
            }> : (tensor<32x18x1xf32>) -> tensor<32x18x4096xf32>
            %normalized_f32 = "ttir.multiply"(
                %input_f32, %inverse_rms_broadcast) :
              (tensor<32x18x4096xf32>, tensor<32x18x4096xf32>) ->
              tensor<32x18x4096xf32>
            %normalized = "ttir.typecast"(%normalized_f32) <{
              conservative_folding = false
            }> : (tensor<32x18x4096xf32>) -> tensor<32x18x4096xbf16>
            %scaled = "ttir.multiply"(%gamma_broadcast, %normalized) :
              (tensor<32x18x4096xbf16>, tensor<32x18x4096xbf16>) ->
              tensor<32x18x4096xbf16>
            %flat = "ttir.reshape"(%scaled) <{
              shape = [576 : i32, 4096 : i32]
            }> : (tensor<32x18x4096xbf16>) -> tensor<576x4096xbf16>
            %weight_t = "ttir.permute"(%weight) <{
              permutation = array<i64: 1, 0>
            }> : (tensor<32x4096xbf16>) -> tensor<4096x32xbf16>
            %out = "ttir.dot_general"(%flat, %weight_t) <{
              batch_dims_lhs = array<i64>,
              batch_dims_rhs = array<i64>,
              contract_dims_lhs = array<i64: 1>,
              contract_dims_rhs = array<i64: 0>
            }> : (tensor<576x4096xbf16>, tensor<4096x32xbf16>) ->
              tensor<576x32xbf16>
            return %out : tensor<576x32xbf16>
          }
        }
        """,
        golden=_rms_projection_golden,
        inputs=InputSpec(_rms_projection_inputs, seed=0),
        pcc=0.98,
        use_tile_matmul=False,
        e2e=True,
    ),
]


KERNEL_BENCHES = [
    KernelBench(
        name="llama_projection_matmul_full_shape",
        kernel=llama_projection_matmul_transposed_weight,
        golden=_golden,
        input_shapes=[(576, 4096), (1024, 4096)],
        run=_full_shape_projection_run,
        inputs=InputSpec("uniform(-0.125,0.125)", seed=0),
        default_cfg=dict(block_shape=[1, 1], grid_shape=[6, 8], dtype="bfloat16"),
        pcc=0.99,
    ),
]
