// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/topk_sample.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt_stl/span.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace tt::runtime::ttnn::operations::reduction::topk_sample {

using ::tt::runtime::logger::LogType;

// Multi-core topk requires power-of-2 input dim < 65536.
static constexpr uint32_t kMaxMulticoreTopkSize = 32768;
static constexpr uint32_t kTopkK = 32;
// ttnn::sampling kernel requires batch dimension = 32.
static constexpr uint32_t kSamplingBatchSize = 32;

static uint32_t nextPowerOf2(uint32_t n) {
  uint32_t p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

static ::ttnn::Tensor sliceTensor(const ::ttnn::Tensor &in, uint32_t batch,
                                   uint32_t start, uint32_t end) {
  ::ttsl::SmallVector<int32_t> begins = {0, static_cast<int32_t>(start)};
  ::ttsl::SmallVector<int32_t> ends = {static_cast<int32_t>(batch),
                                        static_cast<int32_t>(end)};
  ::ttsl::SmallVector<int32_t> steps = {1, 1};
  return ::ttnn::slice(in, ::ttsl::Span<const int32_t>(begins.data(), 2),
                       ::ttsl::Span<const int32_t>(ends.data(), 2),
                       ::ttsl::Span<const int32_t>(steps.data(), 2));
}

static ::ttnn::Tensor padLastDim(const ::ttnn::Tensor &in, uint32_t padAmount,
                                  float padValue) {
  ::ttsl::SmallVector<::ttnn::operations::data_movement::PadSpecDim> padding;
  uint32_t rank = in.logical_shape().rank();
  for (uint32_t d = 0; d < rank; ++d) {
    uint32_t amt = (d == rank - 1) ? padAmount : 0;
    padding.emplace_back(0, amt);
  }
  return ::ttnn::pad(in, padding, padValue, /*use_multicore=*/true);
}

static ::ttnn::Tensor padFirstDim(const ::ttnn::Tensor &in, uint32_t padAmount,
                                   float padValue) {
  ::ttsl::SmallVector<::ttnn::operations::data_movement::PadSpecDim> padding;
  uint32_t rank = in.logical_shape().rank();
  for (uint32_t d = 0; d < rank; ++d) {
    uint32_t amt = (d == 0) ? padAmount : 0;
    padding.emplace_back(0, amt);
  }
  return ::ttnn::pad(in, padding, padValue, /*use_multicore=*/true);
}

void run(const ::tt::target::ttnn::TopKSampleOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor logits = tensorPool.getTTNNTensorAndValidate(op->logits());
  ::ttnn::Tensor temperature =
      tensorPool.getTTNNTensorAndValidate(op->temperature());

  std::optional<uint32_t> seed;
  if (op->seed().has_value()) {
    seed = op->seed().value();
  }

  auto shape = logits.logical_shape();
  uint32_t batch = shape[0];
  uint32_t vocabSize = shape[1];
  uint32_t numChunks =
      (vocabSize + kMaxMulticoreTopkSize - 1) / kMaxMulticoreTopkSize;
  uint32_t chunkSize = (vocabSize + numChunks - 1) / numChunks;
  uint32_t paddedChunkSize = nextPowerOf2(chunkSize);

  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: vocab={}, batch={}, chunks={}, chunkSize={}, "
            "paddedChunkSize={}",
            vocabSize, batch, numChunks, chunkSize, paddedChunkSize);

  // --- Phase 1: Chunked topk ---
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: Phase 1 - chunked topk");
  std::vector<::ttnn::Tensor> chunkValues;
  std::vector<::ttnn::Tensor> chunkIndices;

  for (uint32_t i = 0; i < numChunks; ++i) {
    uint32_t start = i * chunkSize;
    uint32_t end = std::min(start + chunkSize, vocabSize);
    uint32_t actualSize = end - start;

    LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: chunk {} slice [{}, {})", i, start, end);
    ::ttnn::Tensor chunk = sliceTensor(logits, batch, start, end);

    // Pad to power-of-2 for multi-core topk.
    if (actualSize < paddedChunkSize) {
      ::ttnn::Tensor unpadded = std::move(chunk);
      chunk = padLastDim(unpadded, paddedChunkSize - actualSize,
                         -std::numeric_limits<float>::infinity());
      ::ttnn::deallocate(unpadded, /*force=*/true);
    }

    // Typecast to bf16 (topk requires bf16 input).
    if (chunk.dtype() != ::ttnn::DataType::BFLOAT16) {
      ::ttnn::Tensor precast = std::move(chunk);
      chunk = ::ttnn::typecast(precast, ::ttnn::DataType::BFLOAT16);
      ::ttnn::deallocate(precast, /*force=*/true);
    }

    // Per-chunk topk.
    LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: chunk {} topk k={}", i, kTopkK);
    auto result = ::ttnn::topk(chunk, kTopkK, /*dim=*/-1, /*largest=*/true,
                               /*sorted=*/true);
    ::ttnn::deallocate(chunk, /*force=*/true);

    chunkValues.push_back(result[0]);

    // Offset indices to global vocab positions.
    ::ttnn::Tensor indices =
        ::ttnn::typecast(result[1], ::ttnn::DataType::INT32);
    ::ttnn::deallocate(result[1], /*force=*/true);

    if (i > 0) {
      OptionalMeshDeviceRef meshDevice = std::ref(context.getMeshDevice());
      ::ttnn::Tensor offset = ::ttnn::full(
          indices.logical_shape(), static_cast<float>(i * chunkSize),
          ::ttnn::DataType::INT32, ::ttnn::Layout::TILE, meshDevice);
      ::ttnn::Tensor preAdd = std::move(indices);
      indices = ::ttnn::add(preAdd, offset);
      ::ttnn::deallocate(preAdd, /*force=*/true);
      ::ttnn::deallocate(offset, /*force=*/true);
    }
    chunkIndices.push_back(indices);
  }

  // Merge chunks.
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: merging {} chunks", numChunks);
  ::ttnn::Tensor allValues = (numChunks > 1)
                                 ? ::ttnn::concat(chunkValues, 1)
                                 : chunkValues[0];
  ::ttnn::Tensor allIndices = (numChunks > 1)
                                  ? ::ttnn::concat(chunkIndices, 1)
                                  : chunkIndices[0];
  // Free chunk tensors after concat.
  if (numChunks > 1) {
    for (auto &cv : chunkValues) { ::ttnn::deallocate(cv, true); }
    for (auto &ci : chunkIndices) { ::ttnn::deallocate(ci, true); }
  }
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: merge done");

  // --- Phase 2: Final topk on merged candidates ---
  uint32_t mergedK = numChunks * kTopkK;
  if (mergedK > kTopkK) {
    LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: Phase 2 - final topk on {} candidates", mergedK);
    uint32_t paddedMergedK = nextPowerOf2(mergedK);
    if (mergedK < paddedMergedK) {
      ::ttnn::Tensor prepadV = std::move(allValues);
      ::ttnn::Tensor prepadI = std::move(allIndices);
      allValues = padLastDim(prepadV, paddedMergedK - mergedK,
                             -std::numeric_limits<float>::infinity());
      allIndices = padLastDim(prepadI, paddedMergedK - mergedK, 0.0f);
      ::ttnn::deallocate(prepadV, true);
      ::ttnn::deallocate(prepadI, true);
    }

    LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: final topk k={}", kTopkK);
    ::ttnn::Tensor preTopkV = std::move(allValues);
    auto finalResult = ::ttnn::topk(preTopkV, kTopkK, /*dim=*/-1,
                                    /*largest=*/true, /*sorted=*/true);
    ::ttnn::deallocate(preTopkV, true);
    allValues = finalResult[0];

    // Map local indices back to global via gather.
    LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: gather for index mapping");
    ::ttnn::Tensor localInds = finalResult[1];
    if (localInds.dtype() != ::ttnn::DataType::UINT32 &&
        localInds.dtype() != ::ttnn::DataType::UINT16) {
      ::ttnn::Tensor preCast = std::move(localInds);
      localInds = ::ttnn::typecast(preCast, ::ttnn::DataType::UINT32);
      ::ttnn::deallocate(preCast, true);
    }
    ::ttnn::Tensor preGatherI = std::move(allIndices);
    allIndices = ::ttnn::gather(preGatherI, /*dim=*/1, localInds,
                                /*sparse_grad=*/false, std::nullopt,
                                std::nullopt, std::nullopt);
    ::ttnn::deallocate(preGatherI, true);
    ::ttnn::deallocate(localInds, true);
    LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: gather done");
  }

  // Return top-k global indices as [32] INT32.
  // The ttnn::sampling call is done separately by the compiled sampler graph
  // (calling the existing tt::sampling custom op) to avoid a hang when
  // ttnn::sampling is called after the topk chain within the same runtime.
  // Free values — we only return indices.
  ::ttnn::deallocate(allValues, true);

  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: returning top-k indices");

  // allIndices is [1, 32] INT32. Reshape to [32].
  ::ttnn::Tensor output =
      ::ttnn::reshape(allIndices, ::ttnn::Shape({kSamplingBatchSize}));

  // Sync before returning.
  ::tt::tt_metal::distributed::Synchronize(&context.getMeshDevice(),
                                           std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: done");
  return;

  // --- Phase 3: ttnn::sampling --- (disabled, done separately)
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: Phase 3 - sampling");

  // Compute 1/temperature for the sampling kernel.
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: reciprocal temperature");
  ::ttnn::Tensor tempRecip = ::ttnn::reciprocal(temperature);
  if (tempRecip.dtype() != ::ttnn::DataType::BFLOAT16) {
    tempRecip = ::ttnn::typecast(tempRecip, ::ttnn::DataType::BFLOAT16);
  }

  // Pad batch to 32 (sampling kernel requirement).
  if (batch < kSamplingBatchSize) {
    LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: padding batch {} -> {}", batch, kSamplingBatchSize);
    uint32_t padBatch = kSamplingBatchSize - batch;
    allValues = padFirstDim(allValues, padBatch,
                            -std::numeric_limits<float>::infinity());
    allIndices = padFirstDim(allIndices, padBatch, 0.0f);

    // Pad 1D temperature.
    ::ttsl::SmallVector<::ttnn::operations::data_movement::PadSpecDim>
        tempPadding;
    tempPadding.emplace_back(0, padBatch);
    tempRecip =
        ::ttnn::pad(tempRecip, tempPadding, 1.0f, /*use_multicore=*/true);
  }
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: batch padding done");
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: DEBUG sync after padding");
  ::tt::tt_metal::distributed::Synchronize(&context.getMeshDevice(), std::nullopt);
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: DEBUG sync after padding OK");

  // Build k and p tensors (constants for the sampling kernel).
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: creating k/p tensors");
  OptionalMeshDeviceRef meshDevice = std::ref(context.getMeshDevice());
  ::ttnn::Tensor kTensor = ::ttnn::full(
      ::ttnn::Shape({kSamplingBatchSize}), static_cast<float>(kTopkK),
      ::ttnn::DataType::UINT32, ::ttnn::Layout::ROW_MAJOR, meshDevice);
  ::ttnn::Tensor pTensor = ::ttnn::full(
      ::ttnn::Shape({kSamplingBatchSize}), 1.0f, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, meshDevice);
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: k/p tensors created");

  // Ensure correct layouts.
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: to_layout adjustments");
  if (allValues.layout() != ::ttnn::Layout::TILE) {
    LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: allValues to TILE");
    allValues = ::ttnn::to_layout(allValues, ::ttnn::Layout::TILE);
  }
  if (allIndices.layout() != ::ttnn::Layout::ROW_MAJOR) {
    LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: allIndices to ROW_MAJOR");
    allIndices =
        ::ttnn::to_layout(allIndices, ::ttnn::Layout::ROW_MAJOR);
  }
  if (tempRecip.layout() != ::ttnn::Layout::ROW_MAJOR) {
    LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: tempRecip to ROW_MAJOR");
    tempRecip =
        ::ttnn::to_layout(tempRecip, ::ttnn::Layout::ROW_MAJOR);
  }
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: layouts done");
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: DEBUG sync after layouts");
  ::tt::tt_metal::distributed::Synchronize(&context.getMeshDevice(), std::nullopt);
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: DEBUG sync after layouts OK");

  // Reshape to 4D for sampling kernel: [1, 1, batch, candidates].
  uint32_t candidates = allValues.logical_shape()[1];
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: reshape to 4D, candidates={}", candidates);
  allValues = ::ttnn::reshape(
      allValues, ::ttnn::Shape({1, 1, kSamplingBatchSize, candidates}));
  allIndices = ::ttnn::reshape(
      allIndices, ::ttnn::Shape({1, 1, kSamplingBatchSize, candidates}));
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: reshape done");

  // DEBUG: skip sampling — return dummy to confirm sampling is the hang source.
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: DEBUG skipping sampling after full Phase 3 prep");
  ::tt::tt_metal::distributed::Synchronize(&context.getMeshDevice(), std::nullopt);
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: DEBUG sync OK after Phase 3 prep");
  {
    OptionalMeshDeviceRef meshDev3 = std::ref(context.getMeshDevice());
    ::ttnn::Tensor output = ::ttnn::full(
        ::ttnn::Shape({32}), 77.0f, ::ttnn::DataType::INT32,
        ::ttnn::Layout::TILE, meshDev3);
    tensorPool.insertTTNNTensorAndValidate(op->out(), output);
    LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: DEBUG dummy return after Phase 3 prep done");
    return;
  }

#if 0 // DEBUG: disabled while testing — sampling call hangs
  // Return the full [32] output (batch=32 from sampling kernel).
  // Do NOT slice to actual batch here — the physical layout of a sliced
  // tensor doesn't match the compiler's TILE allocation, causing
  // TransferFromDevice to hang. The Python caller trims to actual batch.

  // Debug: log output tensor properties for comparison with SamplingOp.
  {
    auto s = output.logical_shape();
    std::string shapeStr = "[";
    for (uint32_t i = 0; i < s.rank(); ++i) {
      shapeStr += std::to_string(s[i]);
      if (i < s.rank() - 1) shapeStr += ",";
    }
    shapeStr += "]";
    LOG_INFO(LogType::LogRuntimeTTNN,
             "TopKSample output: shape={}, dtype={}, layout={}, storage={}",
             shapeStr, static_cast<int>(output.dtype()),
             static_cast<int>(output.layout()),
             static_cast<int>(output.storage_type()));
  }

  // Sync device — the ~30 internal TTNN ops dispatched async commands that
  // the program executor doesn't know about. Without this, XLA's
  // TransferFromDevice deadlocks waiting on the unsynchronized queue.
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: syncing device");
  ::tt::tt_metal::distributed::Synchronize(&context.getMeshDevice(),
                                           std::nullopt);
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: sync done");

  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: inserting output tensor");
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
  LOG_INFO(LogType::LogRuntimeTTNN, "TopKSample: done");
#endif
}
} // namespace tt::runtime::ttnn::operations::reduction::topk_sample
