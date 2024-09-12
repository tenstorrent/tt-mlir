// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_RUNTIME_TYPES_H
#define TTNN_RUNTIME_TYPES_H

#include "tt/runtime/detail/ttnn.h"

namespace tt::runtime::ttnn {

using DeviceMap = std::unordered_map<uint32_t, ::ttnn::Device *>;
using TensorMap = std::unordered_map<uint32_t, ::ttnn::Tensor *>;
struct ProgramTensorPool {
  ProgramTensorPool(const TensorMap &liveTensors) : liveTensors(liveTensors) {}

  auto try_emplace(std::uint32_t global_id, const ::ttnn::Tensor &tensor) {
    auto it = liveTensors.find(global_id);
    if (it != liveTensors.end()) {
      return std::make_pair(it, false);
    }
    assert(!intermedTensors.contains(global_id));
    intermedTensors.try_emplace(global_id, tensor);
    return liveTensors.try_emplace(global_id, &intermedTensors.at(global_id));
  }

  auto insert_or_assign(std::uint32_t global_id, const ::ttnn::Tensor &tensor) {
    intermedTensors.insert_or_assign(global_id, tensor);
    return liveTensors.insert_or_assign(global_id,
                                        &intermedTensors.at(global_id));
  }

  ::ttnn::Tensor &at(std::uint32_t global_id) {
    assert(liveTensors.contains(global_id));
    return *liveTensors.at(global_id);
  }

  size_t erase(std::uint32_t global_id) {
    assert(liveTensors.contains(global_id) &&
           intermedTensors.contains(global_id));
    intermedTensors.erase(global_id);
    return liveTensors.erase(global_id);
  }

  bool contains(std::uint32_t global_id) const {
    return liveTensors.contains(global_id);
  }

private:
  // A superset of intermedTensors, containing pointers to all tensors created
  // by the program and the input/output tensors passed in by the user
  TensorMap liveTensors;

  // A subset of liveTensors, containing values of any intermediate tensors
  // created by the program
  std::unordered_map<std::uint32_t, ::ttnn::Tensor> intermedTensors;
};

struct ProgramContext {
  ProgramTensorPool tensorPool;
  DeviceMap allDevices;
  DeviceMap devicePool;

  ProgramContext(const TensorMap &liveTensors, const DeviceMap &allDevices)
      : tensorPool(ProgramTensorPool(liveTensors)), allDevices(allDevices) {}
};
} // namespace tt::runtime::ttnn

#endif
