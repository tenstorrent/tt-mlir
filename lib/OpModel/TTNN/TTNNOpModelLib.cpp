// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "TTNNOpModel.h"
#include <cstdint>
#include <iostream>
#include <llvm/ADT/ArrayRef.h>

#ifdef TTMLIR_ENABLE_OPMODEL
#include "MetalHeaders.h"

#include "Conversion.hpp"
#include "SingletonDeviceContext.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include <llvm/Support/Casting.h>
#include <mlir/IR/AttrTypeSubElements.h>

#include <cstddef>
#include <stdexcept>
#endif // TTMLIR_ENABLE_OPMODEL

namespace mlir::tt::op_model::ttnn {

namespace operation {

template <class Callable>
std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
getOpConstraints(const std::string_view &name, Callable &callable,
                 auto &&...args) {
  ::ttnn::graph::QueryResponse query;
  try {
    query = callable(std::make_tuple(std::forward<decltype(args)>(args)...));
  } catch (const std::exception &e) {
    query.status = ::ttnn::graph::ExecutionStatus::Error;
    query.error_message = e.what();
  }

  // check if query was successful
  if (query.status != ::ttnn::graph::ExecutionStatus::Success) {
    // TODO(mbezulj): remove this debug print
    llvm::errs() << "FAILED " << name << ": "
                 << query.error_message.value_or("<error message not set>");
    return make_tuple(false, std::nullopt,
                      query.error_message.value_or("<error message not set>"));
  }

  return std::make_tuple(
      true,
      std::make_tuple(query.resource_usage.cb_peak_size_per_core,
                      query.resource_usage.l1_buffers_peak_per_core,
                      query.resource_usage.l1_output_buffer_per_core),
      std::nullopt);
}
} // namespace operation

//===----------------------------------------------------------------------===//
// Test MLIR<->METAL Conversion functionality
//===----------------------------------------------------------------------===//
namespace conversion {
void debug(const ::llvm::ArrayRef<int64_t> shape,
           const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  layout.dump();
  auto tensorSpec = conversion::getTensorSpec(shape, layout);
  std::cout << "TensorSpec: \n";

  std::cout << "Shape: " << tensorSpec.shape() << std::endl;

  std::cout << "DataType: " << (int)tensorSpec.data_type() << "\n";
  std::cout << "Layout: "
            << ((tensorSpec.layout() == ::tt::tt_metal::Layout::ROW_MAJOR)
                    ? "ROW_MAJOR"
                    : "TILE")
            << "\n";

  const auto memoryConfig = tensorSpec.tensor_layout().get_memory_config();
  std::cout << "MemoryConfig: ";
  std::cout << "TensorMemoryLayout: "
            << ((memoryConfig.memory_layout ==
                 ::tt::tt_metal::TensorMemoryLayout::INTERLEAVED)
                    ? "INTERLEAVED"
                    : "SHARDED")
            << "\n";
  std::cout << "BufferType: "
            << ((memoryConfig.buffer_type == ::tt::tt_metal::BufferType::DRAM)
                    ? "DRAM"
                    : "L1")
            << "\n";
  std::cout << "ShardSpec: "
            << (memoryConfig.shard_spec.has_value() ? "SHARDED" : "NOT SHARDED")
            << "\n";
  if (memoryConfig.shard_spec.has_value()) {
    std::cout << "ShardSpec: " << memoryConfig.shard_spec.value() << "\n";
  }
}
}; // namespace conversion

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::ttnn::graph::QueryResponse
ReluOpQuery(const std::tuple<
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr> &key) {
  // open device device, will close it at the end of function
  auto *device = SingletonDeviceContext::getInstance().getDevice();

  // prepare io specs
  const ::ttnn::TensorSpec input_spec =
      conversion::getTensorSpec(std::get<0>(key), std::get<1>(key));
  const ::ttnn::TensorSpec output_spec =
      conversion::getTensorSpec(std::get<2>(key), std::get<3>(key));

  // run op constraint query
  return ::ttnn::graph::query_op_constraints(
      ::ttnn::relu, device, input_spec,
      output_spec.tensor_layout().get_memory_config());
};

#endif // TTMLIR_ENABLE_OPMODEL

std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
ReluOpInterface::getOpConstraints(
    const ::llvm::ArrayRef<int64_t> &inputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
    const ::llvm::ArrayRef<int64_t> &outputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return operation::getOpConstraints("ReluOpInterface", ReluOpQuery, inputShape,
                                     inputLayout, outputShape, outputLayout);
#else
  return std::make_tuple(true, make_tuple(0, 0, 0), std::nullopt);
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::ttnn::graph::QueryResponse
AddOpQuery(const std::tuple<
           ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
           ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
           ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr> &key) {
  // open device device, will close it at the end of function
  auto *device = SingletonDeviceContext::getInstance().getDevice();

  // prepare io specs
  const ::ttnn::TensorSpec input_spec_a =
      conversion::getTensorSpec(std::get<0>(key), std::get<1>(key));
  const ::ttnn::TensorSpec input_spec_b =
      conversion::getTensorSpec(std::get<2>(key), std::get<3>(key));
  const ::ttnn::TensorSpec output_spec =
      conversion::getTensorSpec(std::get<4>(key), std::get<5>(key));

  return ::ttnn::graph::query_op_constraints(
      ::ttnn::add, device, input_spec_a, input_spec_b, output_spec.data_type(),
      output_spec.tensor_layout().get_memory_config());
};
#endif // TTMLIR_ENABLE_OPMODEL

std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
AddOpInterface::getOpConstraints(
    const ::llvm::ArrayRef<int64_t> &inputShape_a,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
    const ::llvm::ArrayRef<int64_t> &inputShape_b,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
    const ::llvm::ArrayRef<int64_t> &outputShape,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return operation::getOpConstraints("AddOpInterface", AddOpQuery, inputShape_a,
                                     inputLayout_a, inputShape_b, inputLayout_b,
                                     outputShape, outputLayout);
#else
  return std::make_tuple(true, make_tuple(0, 0, 0), std::nullopt);
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::ttnn::graph::QueryResponse SoftmaxOpQuery(
    const std::tuple<
        ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr, int,
        ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr> &key) {
  // open device device, will close it at the end of function
  auto *device = SingletonDeviceContext::getInstance().getDevice();

  // prepare io specs
  const ::ttnn::TensorSpec input_spec =
      conversion::getTensorSpec(std::get<0>(key), std::get<1>(key));
  const int dim_arg = std::get<2>(key);
  const ::ttnn::TensorSpec output_spec =
      conversion::getTensorSpec(std::get<3>(key), std::get<4>(key));

  // run op constraint query
  return ::ttnn::graph::query_op_constraints(
      ::ttnn::softmax, device, input_spec, dim_arg,
      output_spec.tensor_layout().get_memory_config());
};
#endif // TTMLIR_ENABLE_OPMODEL

std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
SoftmaxOpInterface ::getOpConstraints(
    const llvm::ArrayRef<int64_t> &inputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout, const int dim_arg,
    const llvm::ArrayRef<int64_t> &outputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return operation::getOpConstraints("SoftmaxOpInterface", SoftmaxOpQuery,
                                     inputShape, inputLayout, dim_arg,
                                     outputShape, outputLayout);
#else
  return std::make_tuple(true, make_tuple(0, 0, 0), std::nullopt);
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::ttnn::graph::QueryResponse
MatmulOpQuery(const std::tuple<
              ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
              ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
              ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr, bool,
              bool> &key) {
  // open device device, will close it at the end of function
  auto *device = SingletonDeviceContext::getInstance().getDevice();

  // prepare io specs
  const ::ttnn::TensorSpec input_spec_a =
      conversion::getTensorSpec(std::get<0>(key), std::get<1>(key));
  const ::ttnn::TensorSpec input_spec_b =
      conversion::getTensorSpec(std::get<2>(key), std::get<3>(key));
  const ::ttnn::TensorSpec output_spec =
      conversion::getTensorSpec(std::get<4>(key), std::get<5>(key));
  const bool transpose_a = std::get<6>(key);
  const bool transpose_b = std::get<7>(key);

  // run op constraint query
  return ::ttnn::graph::query_op_constraints(
      ::ttnn::matmul, device, input_spec_a, input_spec_b, transpose_a,
      transpose_b, output_spec.tensor_layout().get_memory_config(),
      output_spec.data_type());
};
#endif // TTMLIR_ENABLE_OPMODEL

std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
MatmulOpInterface::getOpConstraints(
    const llvm::ArrayRef<int64_t> &inputShape_a,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
    const llvm::ArrayRef<int64_t> &inputShape_b,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
    const llvm::ArrayRef<int64_t> &outputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout, bool transpose_a,
    bool transpose_b) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return operation::getOpConstraints("MatmulOpInterface", MatmulOpQuery,
                                     inputShape_a, inputLayout_a, inputShape_b,
                                     inputLayout_b, outputShape, outputLayout,
                                     transpose_a, transpose_b);
#else
  return std::make_tuple(true, make_tuple(0, 0, 0), std::nullopt);
#endif // TTMLIR_ENABLE_OPMODEL
}

} // namespace mlir::tt::op_model::ttnn
