// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cpu.h"

namespace tt::runtime::ttnn::operations::cpu {
// using

float *align_to_64(float *ptr) {
  uintptr_t ptr_val = (uintptr_t)ptr;
  uintptr_t aligned_ptr = (ptr_val + 63) & ~((uintptr_t)63);
  return (float *)aligned_ptr;
}

std::vector<wrapped_tensor> pack_tensors(
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::TensorRef>> *ins,
    const tt::target::TensorRef *out, const ProgramContext &context) {
  std::vector<wrapped_tensor> packed_tensors;
  packed_tensors.reserve(ins->size() + 1);
  for (size_t i = 0; i < packed_tensors.size(); ++i) {
    const size_t rank = ins->Get(i)->desc()->shape()->size();
    auto *sizes_and_strides = new int64_t[2 * rank];
    const auto &tens = context.getTensorPool().at(ins->Get(i)->global_id());
    for (size_t j = 0; j < rank; ++j) {
      sizes_and_strides[j] = ins->Get(i)->desc()->shape()->Get(j);
    }
    for (size_t j = 0; j < rank; ++j) {
      sizes_and_strides[rank + j] =
          ins->Get(i)->desc()->layout()->stride()->Get(j);
    }
    packed_tensors.emplace_back(tens.data, align_to_64(tens.data), rank,
                                sizes_and_strides);
  }
  const size_t rank = out->desc()->shape()->size();
  const auto &out_tens = context.getTensorPool().at(ins->Get(i)->global_id);
  auto *out_sizes_and_strides = new int64_t[2 * rank];
  for (size_t j = 0; j < rank; ++j) {
    out_sizes_and_strides[j] = out->desc()->shape()->Get(j);
  }
  for (size_t j = 0; j < rank; ++j) {
    out_sizes_and_strides[rank + j] = out->desc()->layout()->stride()->Get(j);
  }
  packed_tensors.emplace_back(out_tens.data, align_to_64(out_tens.data), rank,
                              out_sizes_and_strides);
  return packed_tensors;
}

void run(const ::tt::target::ttnn::CpuOp *op, ProgramContext &context) {
  const auto *dylib_handle = context.tryGetDylibHandle(op.dylib_id());
  if (!dylib_handle) {
    throw std::runtime_error("could not find dylib corresponding to id: " +
                             op->dylib_id())
  }

  WrappedFunc fn = (WrappedFunc)dlsym(dylib_handle, op->name());
  if (!fn) {
    throw std::runtime_error("could not find requested op: \"" + op->name() +
                             "\" in dylib with id: " + op->dylib_id());
  }

  // validate that the tensors are actually on CPU already

  const auto *fbInputs = op->ins();

  auto dylibInputs = pack_tensors(fbInputs, context);

  auto result =
      static_cast<wrapped_tensor>(fn(dylibInputs.data(), dylibInputs.size()));

  // we don't need to unpack any data from output, it should be written direclty
  // to correct memory

  // we should cleanup everything we heap alloc'ed
  for (auto &input_tensor : dylibInputs) {
    delete[] input_tensor.sizes_and_strides;
  }
  // ins itself will be cleared by going out of scope, and the output underlying
  // data won't be affected
}

} // namespace tt::runtime::ttnn::operations::cpu
