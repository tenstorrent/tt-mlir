// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/concat.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include <sstream>

namespace tt::runtime::ttnn::operations::data_movement {

namespace {
std::string formatFbShape(const flatbuffers::Vector<int> *shape) {
  std::ostringstream os;
  os << "[";
  if (shape) {
    for (uint32_t i = 0; i < shape->size(); ++i) {
      if (i > 0) {
        os << ", ";
      }
      os << shape->Get(i);
    }
  }
  os << "]";
  return os.str();
}

std::string formatTtnnShape(const ::ttnn::Shape &shape) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < shape.rank(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << shape[i];
  }
  os << "]";
  return os.str();
}
} // namespace

void run(const ::tt::target::ttnn::ConcatOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  std::vector<::ttnn::Tensor> inputs;
  for (const auto &input : *op->inputs()) {
    const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(input);
    inputs.push_back(in);
  }
  int32_t dim = op->dim();

  // Pre-validate input ranks. ttnn::concat's own validator (in tt-metal
  // concat.cpp shapes_match lambda) indexes t_shape[i] without short-circuiting
  // on rank mismatch, producing the cryptic "ShapeBase[] index out of range"
  // crash when input ranks disagree at runtime. Surface a clear error here
  // that names the op and shows both MLIR-declared and actual TTNN shapes so
  // the offending compiler-emitted ConcatOp can be identified.
  if (!inputs.empty()) {
    const size_t firstRank = inputs[0].logical_shape().rank();
    bool ranksConsistent = true;
    for (size_t i = 1; i < inputs.size(); ++i) {
      if (inputs[i].logical_shape().rank() != firstRank) {
        ranksConsistent = false;
        break;
      }
    }
    if (!ranksConsistent) {
      std::ostringstream os;
      os << "ttnn.concat input rank mismatch (would crash ttnn::concat with "
            "ShapeBase[] OOB). out global_id="
         << op->out()->global_id() << ", dim=" << dim << "\n";
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto *ref = op->inputs()->Get(i);
        os << "  input[" << i << "] global_id=" << ref->global_id()
           << " mlir_shape=" << formatFbShape(ref->desc()->shape())
           << " ttnn_logical_shape="
           << formatTtnnShape(inputs[i].logical_shape())
           << " ttnn_logical_rank=" << inputs[i].logical_shape().rank() << "\n";
      }
      os << "  out  global_id=" << op->out()->global_id()
         << " mlir_shape=" << formatFbShape(op->out()->desc()->shape());
      LOG_FATAL(os.str());
    }
  }

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      op->memory_config() == 0
          ? ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()))
          : ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                op->memory_config());
  ::ttnn::Tensor out = ::ttnn::concat(inputs, dim, memoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
