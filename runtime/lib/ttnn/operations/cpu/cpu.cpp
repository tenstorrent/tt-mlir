// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cpu.h"

#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/utils.h"

#include <dlfcn.h>
#include <link.h>

namespace tt::runtime::ttnn::operations::cpu {

namespace {
struct WrappedTensor {
  float *start;
  float *alignedStart;
  int64_t startIdx;
  int64_t *sizesAndStrides;
};
} // namespace

// generic signature to call all our funcs; args will be an array of input
// tensors + a counter to tell us how many
using WrappedFunc = void (*)(WrappedTensor *);

std::vector<WrappedTensor> packTensors(
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *ins,
    const tt::target::ttnn::TensorRef *out, const ProgramContext &context,
    std::vector<std::vector<int64_t>> &allSizesAndStrides) {
  allSizesAndStrides.reserve(ins->size() + 1);
  std::vector<WrappedTensor> packedTensors;
  packedTensors.reserve(ins->size());
  for (size_t i = 0; i < ins->size(); ++i) {
    const size_t rank = ins->Get(i)->desc()->shape()->size();
    std::vector<int64_t> sizes(rank);
    const auto &tens =
        context.getTensorPool().getTTNNTensorAndValidate(ins->Get(i));
    for (size_t j = 0; j < rank; ++j) {
      sizes[j] = ins->Get(i)->desc()->shape()->Get(j);
    }
    std::vector<uint32_t> strides = tt::runtime::utils::calculateStride(sizes);
    allSizesAndStrides.emplace_back(2 * rank);
    std::copy(sizes.begin(), sizes.end(), allSizesAndStrides.back().begin());
    std::transform(strides.begin(), strides.end(),
                   allSizesAndStrides.back().begin() + rank,
                   [](uint32_t s) -> int64_t { return s; });

    float *rawDataPtr = static_cast<float *>(
        ::tt::runtime::ttnn::utils::getRawHostDataPtr(tens));
    packedTensors.emplace_back(rawDataPtr, rawDataPtr, 0,
                               allSizesAndStrides.back().data());
  }
  return packedTensors;
}

void run(const ::tt::target::ttnn::CpuOp *op, ProgramContext &context) {
  auto *dylibHandle = context.tryGetDylibHandle(op->dylib_id());
  if (!dylibHandle) {
    LOG_FATAL("could not find dylib corresponding to id: " +
              std::to_string(op->dylib_id()));
  }

  WrappedFunc fn = reinterpret_cast<WrappedFunc>(
      dlsym(dylibHandle, op->func_name()->str().c_str()));
  if (!fn) {
    LOG_FATAL("could not find requested op: \"" + op->func_name()->str() +
              "\" in dylib with id: " + std::to_string(op->dylib_id()));
  }

  const auto *fbInputs = op->ins();

  std::vector<std::vector<int64_t>> allSizesAndStrides;
  auto dylibInputs =
      packTensors(fbInputs, op->out(), context, allSizesAndStrides);
  ::ttnn::Tensor out = context.getTensorPool().getTTNNTensorAndValidate(
      fbInputs->Get(fbInputs->size() - 1));

  context.getTensorPool().insertTTNNTensorAndValidate(op->out(), out);
  fn(dylibInputs.data());
  // We don't need to unpack any data from output, it should be written directly
  // to correct memory.
}

} // namespace tt::runtime::ttnn::operations::cpu
