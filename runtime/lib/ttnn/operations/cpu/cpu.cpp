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

std::vector<common::WrappedTensor> packTensors(
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *ins,
    const tt::target::ttnn::TensorRef *out, const ProgramContext &context,
    std::vector<std::vector<int64_t>> &allSizesAndStrides) {
  allSizesAndStrides.reserve(ins->size());
  std::vector<common::WrappedTensor> packedTensors;
  packedTensors.reserve(ins->size());

  for (size_t i = 0; i < ins->size(); ++i) {
    auto tensorRef = ins->Get(i);
    const auto &tens =
        context.getTensorPool().getTTNNTensorAndValidate(tensorRef);

    const std::vector<int64_t> sizes = tt::runtime::common::extractSizes(tensorRef);
    tt::runtime::common::prepareSizesAndStrides(sizes, allSizesAndStrides);

    float *rawDataPtr = static_cast<float *>(
        ::tt::runtime::ttnn::utils::getRawHostDataPtr(tens));

    packedTensors.push_back(common::WrappedTensor{
        rawDataPtr, rawDataPtr, 0, allSizesAndStrides[i].data()});
  }

  return packedTensors;
}

void run(const ::tt::target::ttnn::CpuOp *op, ProgramContext &context) {
  common::WrappedFunc fn = context.getDylibManager().getFunc(
      op->dylib_id(), op->func_name()->c_str());
  LOG_ASSERT(fn != nullptr);

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
