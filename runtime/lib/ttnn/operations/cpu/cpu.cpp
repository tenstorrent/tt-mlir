// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cpu.h"

#include "tt/runtime/detail/ttnn.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/strides.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

#include <dlfcn.h>
#include <link.h>

namespace tt::runtime::ttnn::operations::cpu {

float *align_to_64(float const *ptr) {
  uintptr_t ptr_val = (uintptr_t)ptr;
  uintptr_t aligned_ptr = (ptr_val + 63) & ~((uintptr_t)63);
  return (float *)aligned_ptr;
}

std::vector<wrapped_tensor> pack_tensors(
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *ins,
    const tt::target::ttnn::TensorRef *out, const ProgramContext &context) {
  std::vector<wrapped_tensor> packed_tensors;
  packed_tensors.reserve(ins->size());
  for (size_t i = 0; i < ins->size(); ++i) {
    LOG_INFO("input has id: ", ins->Get(i)->global_id());
  }
  for (size_t i = 0; i < ins->size(); ++i) {
    LOG_INFO("Trying to fetch tensor w id: ", ins->Get(i)->global_id());
    const size_t rank = ins->Get(i)->desc()->shape()->size();
    std::vector<int64_t> sizes(rank);
    const auto &tens = context.getTensorPool().at(ins->Get(i)->global_id());
    for (size_t j = 0; j < rank; ++j) {
      sizes[j] = ins->Get(i)->desc()->shape()->Get(j);
    }
    std::vector<int64_t> strides = common::calculateStride(sizes);
    int64_t *sizes_and_strides = new int64_t[2 * rank];
    std::copy(sizes.begin(), sizes.end(), sizes_and_strides);
    std::copy(strides.begin(), strides.end(), sizes_and_strides + rank);

    float *raw_data_ptr = static_cast<float *>(get_raw_host_data_ptr(tens));
    LOG_INFO("raw_ptr for ", i, " is ", raw_data_ptr);
    LOG_INFO("aligned_ptr for ", i, " is ", align_to_64(raw_data_ptr));
    packed_tensors.emplace_back(raw_data_ptr, raw_data_ptr, 0,
                                sizes_and_strides);
  }
  return packed_tensors;
}

void run(const ::tt::target::ttnn::CpuOp *op, ProgramContext &context) {
  auto *dylib_handle = context.tryGetDylibHandle(op->dylib_id());
  if (!dylib_handle) {
    throw std::runtime_error("could not find dylib corresponding to id: " +
                             std::to_string(op->dylib_id()));
  }

  WrappedFunc fn =
      (WrappedFunc)dlsym(dylib_handle, op->func_name()->str().c_str());
  if (!fn) {
    throw std::runtime_error(
        "could not find requested op: \"" + op->func_name()->str() +
        "\" in dylib with id: " + std::to_string(op->dylib_id()));
  }

  const auto *fbInputs = op->ins();

  auto dylibInputs = pack_tensors(fbInputs, op->out(), context);
  ::Tensor out = context.getTensorPool().at(
      fbInputs->Get(fbInputs->size() - 1)->global_id());

  context.getTensorPool().insert_or_assign(op->out()->global_id(), out);
  LOG_INFO("Raw ptr: ", dylibInputs.data());
  // TODO: do we really not need to use return value?
  fn(dylibInputs.data());

  LOG_INFO("ran func!");

  // We don't need to unpack any data from output, it should be written directly
  // to correct memory.

  // Clean up everything we heap alloc'ed.
  for (auto &input_tensor : dylibInputs) {
    delete[] input_tensor.sizes_and_strides;
  }
  // The ins vector itself will be cleared by going out of scope, and the output
  // underlying data won't be affected.
}

} // namespace tt::runtime::ttnn::operations::cpu
