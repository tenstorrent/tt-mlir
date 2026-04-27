// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/layout/to_dtype.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <ttnn/operations/core/to_dtype/to_dtype_op.hpp>
#include <ttnn/tensor/host_buffer/functions.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>

#include <vector>

namespace tt::runtime::ttnn::operations::layout {

// `pack_as_bfp_tiles` API and `TensorLayout::fromPaddedShape` carry
// deprecation diagnostics in current tt-metal but remain the supported
// path; mirror the suppression pattern used elsewhere in the runtime.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace {

// Per-shard host packer: bf16 → blockfloat (BFP4_B / BFP8_B). Mirrors
// what tt-metal's `tensor_impl::to_dtype<bfloat16, bfp_tag>` does
// internally; we implement it directly here so the loop is explicit and
// we can reason about the multi-chip distributed buffer layout.
::tt::tt_metal::HostBuffer
packBfpShard(const ::tt::tt_metal::HostBuffer &bf16_buf,
             ::ttnn::DataType targetDtype,
             const std::optional<::tt::tt_metal::Tile> &tile) {
  auto bf16_span = ::tt::tt_metal::host_buffer::get_as<::bfloat16>(bf16_buf);
  std::vector<float> f32_data;
  f32_data.reserve(bf16_span.size());
  for (size_t i = 0; i < bf16_span.size(); ++i) {
    f32_data.push_back(static_cast<float>(bf16_span[i]));
  }
  ::tt::stl::Span<const float> f32_span(f32_data.data(), f32_data.size());
  std::vector<uint32_t> packed;
  if (targetDtype == ::ttnn::DataType::BFLOAT4_B) {
    packed = ::pack_as_bfp4_tiles<float>(f32_span, /*row_major_input=*/false,
                                         /*is_exp_a=*/false, tile);
  } else {
    packed = ::pack_as_bfp8_tiles<float>(f32_span, /*row_major_input=*/false,
                                         /*is_exp_a=*/false, tile);
  }
  return ::tt::tt_metal::HostBuffer(std::move(packed));
}

::ttnn::Tensor toDtypeBlockfloat(const ::ttnn::Tensor &hostInput,
                                 ::ttnn::DataType targetDtype) {
  // Per-shard packing: handles single-host (one shard) and multi-host
  // (Galaxy mesh: one shard per chip) cases uniformly. Byte-equivalent
  // to packing the full pre-shard tensor and re-sharding when shard
  // sizes align with bfp face boundaries (16×16) — true for every
  // production graph we've inspected (gpt-oss-120b, llama).
  auto tile = hostInput.tensor_spec().tile();
  const ::tt::tt_metal::DistributedHostBuffer &distBf16 =
      hostInput.host_tensor().buffer();

  ::tt::tt_metal::DistributedHostBuffer distOut = distBf16.transform(
      [&tile, targetDtype](const ::tt::tt_metal::HostBuffer &shard) {
        return packBfpShard(shard, targetDtype, tile);
      },
      ::tt::tt_metal::DistributedHostBuffer::ProcessShardExecutionPolicy::
          PARALLEL);

  ::tt::tt_metal::TensorSpec outSpec(
      hostInput.logical_shape(),
      ::tt::tt_metal::TensorLayout::fromPaddedShape(
          targetDtype, hostInput.tensor_spec().page_config(),
          ::tt::tt_metal::MemoryConfig{}, hostInput.logical_shape(),
          hostInput.padded_shape()));
  ::tt::tt_metal::HostTensor hostOut(std::move(distOut), std::move(outSpec),
                                     hostInput.tensor_topology());
  return ::ttnn::Tensor(std::move(hostOut));
}

} // namespace

// Host-side dtype conversion. Input must be a host tensor (typically the
// output of `ttnn.from_device`). For BFLOAT4_B / BFLOAT8_B targets with
// BFLOAT16 input dispatches to a custom per-shard host packer, since
// `::ttnn::to_dtype` (which calls `tensor_impl::to_dtype`) does not
// reliably handle multi-host `DistributedHostBuffer` inputs in our
// pipeline. For other targets we forward to `::ttnn::to_dtype`.
void run(const ::tt::target::ttnn::ToDtypeOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());
  DEBUG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->in()),
               "Calling ttnn::to_dtype on a device tensor; expected host");

  ::ttnn::DataType targetDataType =
      ::tt::runtime::ttnn::utils::toTTNNDataType(op->dtype());

  ::ttnn::Tensor out;
  if ((targetDataType == ::ttnn::DataType::BFLOAT4_B ||
       targetDataType == ::ttnn::DataType::BFLOAT8_B) &&
      inputTensor.dtype() == ::ttnn::DataType::BFLOAT16) {
    out = toDtypeBlockfloat(inputTensor, targetDataType);
  } else {
    out = ::ttnn::to_dtype(inputTensor, targetDataType);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

#pragma GCC diagnostic pop

} // namespace tt::runtime::ttnn::operations::layout
