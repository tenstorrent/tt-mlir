# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Composed full Llama 3 8B prefill-layer validation."""

import math
from pathlib import Path

import torch

from runner import InputSpec, PatternTest

_ROOT = Path(__file__).resolve().parents[4]
_MODEL = (
    _ROOT
    / "test"
    / "ttmlir"
    / "models"
    / "single_blocks_and_layers"
    / "llama_3_8b_prefill_layer.mlir"
)
_MODEL_TEXT = _MODEL.read_text()
_WEIGHT_GROUPS = 8

_EMBEDDING_TTIR = """
module {
  func.func @main(
      %indices: tensor<576xui32>,
      %weight: tensor<512x4096xbf16>) -> tensor<576x4096xbf16> {
    %0 = "ttir.embedding"(%indices, %weight) :
      (tensor<576xui32>, tensor<512x4096xbf16>) -> tensor<576x4096xbf16>
    return %0 : tensor<576x4096xbf16>
  }
}
"""
_LARGE_EMBEDDING_TTIR = _EMBEDDING_TTIR.replace(
    "tensor<512x4096xbf16>", "tensor<128256x4096xbf16>"
)
_I64_EMBEDDING_TTIR = """
module {
  func.func @main(
      %indices: tensor<576xi64>,
      %weight: tensor<512x4096xbf16>) -> tensor<576x4096xbf16> {
    %0 = "ttir.typecast"(%indices) <{conservative_folding = false}> :
      (tensor<576xi64>) -> tensor<576xui32>
    %1 = "ttir.embedding"(%0, %weight) :
      (tensor<576xui32>, tensor<512x4096xbf16>) -> tensor<576x4096xbf16>
    return %1 : tensor<576x4096xbf16>
  }
}
"""
_RESHAPED_I64_EMBEDDING_TTIR = """
module {
  func.func @main(
      %indices: tensor<32x18xi64>,
      %weight: tensor<512x4096xbf16>) -> tensor<32x18x4096xbf16> {
    %0 = "ttir.reshape"(%indices) <{shape = [1 : i32, 32 : i32, 18 : i32]}> :
      (tensor<32x18xi64>) -> tensor<1x32x18xi64>
    %1 = "ttir.reshape"(%0) <{shape = [576 : i32]}> :
      (tensor<1x32x18xi64>) -> tensor<576xi64>
    %2 = "ttir.typecast"(%1) <{conservative_folding = false}> :
      (tensor<576xi64>) -> tensor<576xui32>
    %3 = "ttir.embedding"(%2, %weight) :
      (tensor<576xui32>, tensor<512x4096xbf16>) -> tensor<576x4096xbf16>
    %4 = "ttir.reshape"(%3) <{shape = [32 : i32, 18 : i32, 4096 : i32]}> :
      (tensor<576x4096xbf16>) -> tensor<32x18x4096xbf16>
    return %4 : tensor<32x18x4096xbf16>
  }
}
"""

_RMS_NORM_BODY = """
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
    %input_f32 = "ttir.typecast"(@INPUT@) <{
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
    %normalized_f32 = "ttir.multiply"(%input_f32, %inverse_rms_broadcast) :
      (tensor<32x18x4096xf32>, tensor<32x18x4096xf32>) ->
      tensor<32x18x4096xf32>
    %normalized = "ttir.typecast"(%normalized_f32) <{
      conservative_folding = false
    }> : (tensor<32x18x4096xf32>) -> tensor<32x18x4096xbf16>
    %scaled = "ttir.multiply"(%gamma_broadcast, %normalized) :
      (tensor<32x18x4096xbf16>, tensor<32x18x4096xbf16>) ->
      tensor<32x18x4096xbf16>
    return %scaled : tensor<32x18x4096xbf16>
"""

_RMS_NORM_TTIR = (
    """
module {
  func.func @main(
      %input: tensor<32x18x4096xbf16>,
      %gamma: tensor<4096xbf16>) -> tensor<32x18x4096xbf16> {
"""
    + _RMS_NORM_BODY.replace("@INPUT@", "%input")
    + """
  }
}
"""
)

_EMBEDDING_RMS_NORM_TTIR = (
    """
module {
  func.func @main(
      %indices: tensor<32x18xi64>,
      %weight: tensor<512x4096xbf16>,
      %gamma: tensor<4096xbf16>) -> tensor<32x18x4096xbf16> {
    %indices_3d = "ttir.reshape"(%indices) <{
      shape = [1 : i32, 32 : i32, 18 : i32]
    }> : (tensor<32x18xi64>) -> tensor<1x32x18xi64>
    %indices_flat = "ttir.reshape"(%indices_3d) <{
      shape = [576 : i32]
    }> : (tensor<1x32x18xi64>) -> tensor<576xi64>
    %indices_ui32 = "ttir.typecast"(%indices_flat) <{
      conservative_folding = false
    }> : (tensor<576xi64>) -> tensor<576xui32>
    %embedding_flat = "ttir.embedding"(%indices_ui32, %weight) :
      (tensor<576xui32>, tensor<512x4096xbf16>) -> tensor<576x4096xbf16>
    %embedding = "ttir.reshape"(%embedding_flat) <{
      shape = [32 : i32, 18 : i32, 4096 : i32]
    }> : (tensor<576x4096xbf16>) -> tensor<32x18x4096xbf16>
"""
    + _RMS_NORM_BODY.replace("@INPUT@", "%embedding")
    + """
  }
}
"""
)


def _uniform(shape, dtype, generator, radius):
    values = torch.rand(shape, dtype=dtype, generator=generator)
    return values.mul_(2.0 * radius).sub_(radius)


def _structured_weight(shape, dtype, generator):
    rows, columns = shape
    base = 1.0 / (128.0 if shape == (4096, 14336) else 64.0)
    choices = torch.tensor([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], dtype=dtype)
    row_indices = torch.randint(
        choices.numel(),
        (rows, _WEIGHT_GROUPS),
        generator=generator,
    )
    row_factors = choices[row_indices]
    row_factors[0, :] = 1.0

    column_indices = torch.randint(
        choices.numel(),
        (columns,),
        generator=generator,
    )
    column_factors = choices[column_indices].mul_(base)
    group_indices = torch.arange(columns).mul(_WEIGHT_GROUPS).floor_divide(columns)
    weight = row_factors[:, group_indices]
    return weight.mul_(column_factors)


def _layer_inputs(shape, dtype, generator):
    shape = tuple(shape)
    if not dtype.is_floating_point:
        if shape == (18,):
            return torch.arange(18, dtype=dtype)
        if shape == (32, 18):
            return torch.randint(0, 256, shape, dtype=dtype, generator=generator)
        raise ValueError(f"unsupported integer input shape {shape}")

    if shape == (64,):
        exponent = -math.log(500000.0) / 64.0
        return torch.exp(torch.arange(64, dtype=torch.float32) * exponent).to(dtype)

    if shape == (4096,):
        return 0.75 + 0.5 * torch.rand(shape, dtype=dtype, generator=generator)

    if len(shape) == 2:
        return _structured_weight(shape, dtype, generator)

    return _uniform(shape, dtype, generator, 1.0)


def _embedding_inputs(shape, dtype, generator):
    if not dtype.is_floating_point:
        return torch.randint(0, 512, shape, dtype=dtype, generator=generator)
    return _uniform(shape, dtype, generator, 1.0)


def _large_embedding_inputs(shape, dtype, generator):
    if not dtype.is_floating_point:
        return torch.randint(0, 256, shape, dtype=dtype, generator=generator)
    return _structured_weight(tuple(shape), dtype, generator)


def _rms_norm_inputs(shape, dtype, generator):
    shape = tuple(shape)
    if shape == (32, 18, 4096):
        return _structured_weight((32 * 18, 4096), dtype, generator).reshape(shape)
    if shape == (4096,):
        return 0.75 + 0.5 * torch.rand(shape, dtype=dtype, generator=generator)
    raise ValueError(f"unsupported RMSNorm input shape {shape}")


def _embedding_rms_norm_inputs(shape, dtype, generator):
    shape = tuple(shape)
    if not dtype.is_floating_point:
        return torch.randint(0, 512, shape, dtype=dtype, generator=generator)
    if shape == (512, 4096):
        return _structured_weight(shape, dtype, generator)
    if shape == (4096,):
        return 0.75 + 0.5 * torch.rand(shape, dtype=dtype, generator=generator)
    raise ValueError(f"unsupported embedding RMSNorm input shape {shape}")


def _bf16(value):
    return value.to(torch.bfloat16).float()


def _structured_linear(value, weight):
    value = value.reshape(-1, value.shape[-1]).float()
    weight = weight.float()
    output = torch.zeros(
        (value.shape[0], weight.shape[0]),
        dtype=torch.float32,
    )
    column_factors = weight[0]
    for group in range(_WEIGHT_GROUPS):
        begin = weight.shape[1] * group // _WEIGHT_GROUPS
        end = weight.shape[1] * (group + 1) // _WEIGHT_GROUPS
        row_factors = weight[:, begin] / column_factors[begin]
        reduction = torch.matmul(
            value[:, begin:end],
            column_factors[begin:end],
        )
        output.add_(reduction[:, None] * row_factors[None, :])
    return _bf16(output)


def _rms_norm(value, gamma):
    inverse_rms = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1.0e-5)
    normalized = _bf16(value * inverse_rms)
    return _bf16(normalized * gamma.float().reshape(1, 1, 4096))


def _embedding_rms_norm_golden(indices, weight, gamma):
    return _rms_norm(weight[indices.long()].float(), gamma)


def _project(hidden, weight, heads):
    projected = _structured_linear(hidden, weight)
    return projected.reshape(32, 18, heads, 128).permute(0, 2, 1, 3)


def _rotate_half(value):
    return torch.cat((-value[..., 64:], value[..., :64]), dim=-1)


def _layer_decoder_golden(
    positions,
    inv_freq,
    key_weight,
    token_ids,
    embedding,
    gamma,
    key_cache,
    value_weight,
    value_cache,
    lm_head,
    down_weight,
    up_weight,
    output_weight,
    query_weight,
    post_gamma,
    gate_weight,
    final_gamma,
):
    hidden = embedding[token_ids.long()].float()
    normalized = _rms_norm(hidden, gamma)

    angles = torch.outer(positions.float(), inv_freq.float())
    angles = torch.cat((angles, angles), dim=-1)
    cosine = _bf16(torch.cos(angles)).reshape(1, 1, 18, 128)
    sine = _bf16(torch.sin(angles)).reshape(1, 1, 18, 128)

    key_update = _project(normalized, key_weight, 8)
    key_update = _bf16(
        _bf16(key_update * cosine) + _bf16(_rotate_half(key_update) * sine)
    )
    key_cache = key_cache.float().clone()
    key_cache[:, :, :18, :] = key_update

    value_update = _project(normalized, value_weight, 8)
    value_cache = value_cache.float().clone()
    value_cache[:, :, :18, :] = value_update

    query = _project(normalized, query_weight, 32)
    query = _bf16(_bf16(query * cosine) + _bf16(_rotate_half(query) * sine))

    key = key_cache.repeat_interleave(4, dim=1)
    value = value_cache.repeat_interleave(4, dim=1)
    scale = 0.297301769
    scores = torch.matmul(query * scale, (key * scale).transpose(-2, -1))
    causal = torch.arange(128).reshape(1, 1, 1, 128) <= positions.reshape(1, 1, 18, 1)
    probabilities = torch.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1)
    attention = _bf16(torch.matmul(probabilities, value))

    attention = attention.permute(0, 2, 1, 3).reshape(32, 18, 4096)
    attention_output = _structured_linear(attention, output_weight).reshape(
        32, 18, 4096
    )
    attention_residual = _bf16(hidden + attention_output)

    normalized = _rms_norm(attention_residual, post_gamma)
    gate = _structured_linear(normalized, gate_weight)
    gate = _bf16(gate * torch.sigmoid(gate))
    up = _structured_linear(normalized, up_weight)
    gated = _bf16(gate * up)
    down = _structured_linear(gated, down_weight).reshape(32, 18, 4096)
    decoder_output = _bf16(attention_residual + down)

    return decoder_output


def _layer_embedding_golden(*args):
    token_ids = args[3]
    embedding = args[4]
    return embedding[token_ids.long()].float()


def _layer_input_norm_golden(*args):
    hidden = _layer_embedding_golden(*args)
    gamma = args[5]
    return _rms_norm(hidden, gamma)


def _layer_final_norm_golden(*args):
    decoder_output = _layer_decoder_golden(*args)
    final_gamma = args[16]
    return _rms_norm(decoder_output, final_gamma)


def _layer_logits_golden(*args):
    normalized = _layer_final_norm_golden(*args)
    lm_head = args[9]
    logits = _structured_linear(normalized, lm_head)
    return logits.reshape(32, 18, lm_head.shape[0])


_LAYER_OUTPUT_SIGNATURE = (
    "tensor<32x18x128256xbf16> "
    "{ttcore.shard_status = #ttcore.shard_status<unsharded>}) {"
)
_LAYER_RESULT_SIGNATURE = (
    "-> (tensor<32x8x128x128xbf16> "
    "{ttcore.shard_status = #ttcore.shard_status<unsharded>}, "
    "tensor<32x8x128x128xbf16> "
    "{ttcore.shard_status = #ttcore.shard_status<unsharded>}, "
    f"{_LAYER_OUTPUT_SIGNATURE}"
)
_LAYER_RETURN = (
    "return %151, %221, %355 : tensor<32x8x128x128xbf16>, "
    "tensor<32x8x128x128xbf16>, tensor<32x18x128256xbf16>"
)


def _model_with_layer_output(value, width, include_caches=True):
    output_type = f"tensor<32x18x{width}xbf16>"
    if include_caches:
        text = _MODEL_TEXT.replace(
            _LAYER_OUTPUT_SIGNATURE,
            f"{output_type} "
            "{ttcore.shard_status = #ttcore.shard_status<unsharded>}) {",
            1,
        )
        replacement_return = (
            f"        return %151, %221, %{value} : "
            f"tensor<32x8x128x128xbf16>, tensor<32x8x128x128xbf16>, "
            f"{output_type}\n"
        )
    else:
        text = _MODEL_TEXT.replace(
            _LAYER_RESULT_SIGNATURE,
            f"-> ({output_type} "
            "{ttcore.shard_status = #ttcore.shard_status<unsharded>}) {",
            1,
        )
        replacement_return = f"        return %{value} : {output_type}\n"
    next_op = text.index(f"        %{value + 1} = ")
    original_return = text.index(f"        {_LAYER_RETURN}", next_op)
    return_end = text.index("\n", original_return) + 1
    return text[:next_op] + replacement_return + text[return_end:]


_EMBEDDING_MODEL_TEXT = _model_with_layer_output(43, 4096, include_caches=False)
_INPUT_NORM_MODEL_TEXT = _model_with_layer_output(56, 4096, include_caches=False)
_DECODER_MODEL_TEXT = _model_with_layer_output(336, 4096)
_FINAL_NORM_MODEL_TEXT = _model_with_layer_output(349, 4096)


PATTERN_TESTS = [
    PatternTest(
        name="llama_prefill_embedding_like_rms_norm_direct_e2e",
        ttir=_RMS_NORM_TTIR,
        golden=_rms_norm,
        inputs=InputSpec(_rms_norm_inputs, seed=0),
        pcc=0.99,
        std_rtol=0.05,
        use_tile_matmul=False,
        num_stream_buffers=1,
        e2e=True,
        output_index=0,
        golden_inputs_as_float=False,
    ),
    PatternTest(
        name="llama_prefill_embedding_rms_norm_direct_e2e",
        ttir=_EMBEDDING_RMS_NORM_TTIR,
        golden=_embedding_rms_norm_golden,
        inputs=InputSpec(_embedding_rms_norm_inputs, seed=0),
        pcc=0.99,
        std_rtol=0.05,
        use_tile_matmul=False,
        num_stream_buffers=1,
        e2e=True,
        output_index=0,
        golden_inputs_as_float=False,
    ),
    PatternTest(
        name="llama_prefill_embedding_direct_e2e",
        ttir=_EMBEDDING_TTIR,
        golden=lambda indices, weight: weight[indices.long()].float(),
        inputs=InputSpec(_embedding_inputs, seed=0),
        pcc=0.99,
        std_rtol=0.05,
        use_tile_matmul=False,
        num_stream_buffers=1,
        e2e=True,
        output_index=0,
        golden_inputs_as_float=False,
    ),
    PatternTest(
        name="llama_prefill_large_embedding_direct_e2e",
        ttir=_LARGE_EMBEDDING_TTIR,
        golden=lambda indices, weight: weight[indices.long()].float(),
        inputs=InputSpec(_large_embedding_inputs, seed=0),
        pcc=0.99,
        std_rtol=0.05,
        use_tile_matmul=False,
        num_stream_buffers=1,
        e2e=True,
        output_index=0,
        golden_inputs_as_float=False,
    ),
    PatternTest(
        name="llama_prefill_i64_embedding_direct_e2e",
        ttir=_I64_EMBEDDING_TTIR,
        golden=lambda indices, weight: weight[indices.long()].float(),
        inputs=InputSpec(_embedding_inputs, seed=0),
        pcc=0.99,
        std_rtol=0.05,
        use_tile_matmul=False,
        num_stream_buffers=1,
        e2e=True,
        output_index=0,
        golden_inputs_as_float=False,
    ),
    PatternTest(
        name="llama_prefill_reshaped_i64_embedding_direct_e2e",
        ttir=_RESHAPED_I64_EMBEDDING_TTIR,
        golden=lambda indices, weight: weight[indices.long()].float(),
        inputs=InputSpec(_embedding_inputs, seed=0),
        pcc=0.99,
        std_rtol=0.05,
        use_tile_matmul=False,
        num_stream_buffers=1,
        e2e=True,
        output_index=0,
        golden_inputs_as_float=False,
    ),
    PatternTest(
        name="llama_prefill_layer_embedding_composed_e2e",
        ttir=_EMBEDDING_MODEL_TEXT,
        golden=_layer_embedding_golden,
        inputs=InputSpec(_layer_inputs, seed=0),
        pcc=0.99,
        use_tile_matmul=False,
        num_stream_buffers=1,
        e2e=True,
        pattern_files=(
            "llama_projection_matmul_to_kernel.py",
            "llama_prefill_sdpa_to_kernel.py",
        ),
        output_index=0,
        golden_inputs_as_float=False,
    ),
    PatternTest(
        name="llama_prefill_layer_input_norm_composed_e2e",
        ttir=_INPUT_NORM_MODEL_TEXT,
        golden=_layer_input_norm_golden,
        inputs=InputSpec(_layer_inputs, seed=0),
        pcc=0.99,
        std_rtol=0.05,
        use_tile_matmul=False,
        num_stream_buffers=1,
        e2e=True,
        pattern_files=(
            "llama_projection_matmul_to_kernel.py",
            "llama_prefill_sdpa_to_kernel.py",
        ),
        output_index=0,
        golden_inputs_as_float=False,
    ),
    PatternTest(
        name="llama_prefill_layer_decoder_composed_e2e",
        ttir=_DECODER_MODEL_TEXT,
        golden=_layer_decoder_golden,
        inputs=InputSpec(_layer_inputs, seed=0),
        pcc=0.95,
        use_tile_matmul=False,
        num_stream_buffers=1,
        e2e=True,
        pattern_files=(
            "llama_projection_matmul_to_kernel.py",
            "llama_prefill_sdpa_to_kernel.py",
        ),
        output_index=2,
        golden_inputs_as_float=False,
    ),
    PatternTest(
        name="llama_prefill_layer_final_norm_composed_e2e",
        ttir=_FINAL_NORM_MODEL_TEXT,
        golden=_layer_final_norm_golden,
        inputs=InputSpec(_layer_inputs, seed=0),
        pcc=0.95,
        use_tile_matmul=False,
        num_stream_buffers=1,
        e2e=True,
        pattern_files=(
            "llama_projection_matmul_to_kernel.py",
            "llama_prefill_sdpa_to_kernel.py",
        ),
        output_index=2,
        golden_inputs_as_float=False,
    ),
    PatternTest(
        name="llama_prefill_layer_composed_e2e",
        ttir=_MODEL_TEXT,
        check="""
        CHECK-LABEL: func.func @main
        CHECK-COUNT-13: d2m.generic
        CHECK-NOT: d2m.generic
        CHECK: return
        """,
        golden=_layer_logits_golden,
        inputs=InputSpec(_layer_inputs, seed=0),
        pcc=0.95,
        use_tile_matmul=False,
        num_stream_buffers=1,
        e2e=True,
        pattern_files=(
            "llama_projection_matmul_to_kernel.py",
            "llama_prefill_sdpa_to_kernel.py",
        ),
        output_index=2,
        golden_inputs_as_float=False,
    ),
]
