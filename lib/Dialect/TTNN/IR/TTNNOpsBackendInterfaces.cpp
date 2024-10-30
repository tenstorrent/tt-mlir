
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#ifndef TT_RUNTIME_ENABLE_TTNN
#warning                                                                       \
    "TT_RUNTIME_ENABLE_TTNN should not be defined in the backend interface file."
#else // TT_RUNTIME_ENABLE_TTNN enabled

#include "ttnn/cpp/ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
// #include "common/core_coord.h" // CoreRangeSet
// #include "ttnn/tensor/types.hpp" // DataType, Layout, StorageType,
// MemoryConfig #include "tt_metal/impl/buffers/buffer.hpp" // BufferType,
// ShardSpec #include "tt_metal/impl/buffers/buffer_constants.hpp" //
// TensorMemoryLayout, ShardOrientation

namespace mlir::tt::ttnn {

static std::vector<std::tuple<uint32_t, uint32_t>>
get_cb_allocations_from_trace(const nlohmann::json &json_trace) {
  auto graph_circular_buffer_allocations =
      ::ttnn::graph::extract_circular_buffer_allocations_per_core(json_trace);

  std::vector<std::tuple<uint32_t, uint32_t>> cbs_per_core;
  for (auto cb_allocation : graph_circular_buffer_allocations) {
    cbs_per_core.emplace_back(std::make_tuple(cb_allocation, (uint32_t)64));
  }

  return cbs_per_core;
}

static std::vector<std::tuple<uint32_t, uint32_t>>
get_tensor_allocations_from_trace(const nlohmann::json &json_trace) {
  auto graph_tensor_allocations =
      ::ttnn::graph::extract_l1_buffer_allocations(json_trace);

  std::vector<std::tuple<uint32_t, uint32_t>> tensors_per_core;
  for (auto tensor_allocation : graph_tensor_allocations) {
    tensors_per_core.emplace_back(
        std::make_tuple(tensor_allocation, (uint32_t)64));
  }

  return tensors_per_core;
}

class ScopedDeviceContext {
public:
  ScopedDeviceContext() : m_device(::tt::tt_metal::CreateDevice(0)) {}
  ~ScopedDeviceContext() { ::tt::tt_metal::CloseDevice(m_device); }

  ::tt::tt_metal::Device &get_device() { return *m_device; }

private:
  ::tt::tt_metal::Device *m_device;
};

// //===----------------------------------------------------------------------===//
// // ReluOp
// //===----------------------------------------------------------------------===//

// // Relu backend interface
size_t ReluOp::getOpPerfCycles() {
  // Implement a custom estimate for relu op cycles.
  return 5;
}

size_t ReluOp::getOpL1Usage() {
  // Implement a custom estimate for relu op L1 usage.

  // const ttnn::types::Shape &shape_a;
  // tt::tt_metal::DataType data_type_a;
  // tt::tt_metal::Layout layout_a;
  // const tt::tt_metal::MemoryConfig &memory_config_a;
  // const tt::tt_metal::MemoryConfig &memory_config_o;

  ScopedDeviceContext ctx;

  // auto input_tensor = create_tensor(ctx.get_device(), shape_a, data_type_a,
  //                                   layout_a, memory_config_a);

  // auto call = [&] {
  //   const auto output_tensor = ttnn::relu(input_tensor, memory_config_o);
  //   return output_tensor;
  // };

  // auto json_trace = graph::query_trace(call);

  // uint32_t cbs_per_core = get_cb_allocations_from_trace(json_trace);
  // uint32_t tensors_per_core =
  // get_tensor_allocations_from_trace(m_json_trace);

  return 10;
}

bool ReluOp::isOpLegal() {
  // Implement a custom check for relu op legality.
  return true;
}

} // namespace mlir::tt::ttnn

#endif // TT_RUNTIME_ENABLE_TTNN enabled