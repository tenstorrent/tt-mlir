// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "TTNNOpModel.h"
#include <cstdint>
#include <iostream>
#include <llvm/ADT/ArrayRef.h>

#ifdef TTMLIR_ENABLE_OPMODEL
#include "SingletonDeviceContext.h"
#include "TTNNOpModelLib_Impl.h"
#include "TupleCache.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include <llvm/Support/Casting.h>
#include <mlir/IR/AttrTypeSubElements.h>

#include <cstddef>
#include <stdexcept>
#endif // TTMLIR_ENABLE_OPMODEL

namespace mlir::tt::ttnn {

std::size_t hashMemRefType(MemRefType memref) {
  return llvm::hash_combine(llvm::hash_combine_range(memref.getShape().begin(),
                                                     memref.getShape().end()),
                            memref.getElementType(), memref.getMemorySpace());
}

std::size_t hashLayoutAttr(TTNNLayoutAttr layout) {
  return llvm::hash_combine(
      hashMemRefType(layout.getMemref()),
      llvm::hash_combine_range(layout.getGrid().getShape().begin(),
                               layout.getGrid().getShape().end()),
      layout.getMemLayout());
}

std::size_t hashArrayRef(::llvm::ArrayRef<int64_t> array) {
  return llvm::hash_combine_range(array.begin(), array.end());
}
} // namespace mlir::tt::ttnn

namespace std {

template <>
struct hash<mlir::MemRefType> {
  std::size_t operator()(const mlir::MemRefType &memref) const {
    return mlir::tt::ttnn::hashMemRefType(memref);
  }
};

template <>
struct hash<mlir::tt::ttnn::TTNNLayoutAttr> {
  std::size_t operator()(const mlir::tt::ttnn::TTNNLayoutAttr &layout) const {
    return mlir::tt::ttnn::hashLayoutAttr(layout);
  }
};

template <>
struct hash<llvm::ArrayRef<int64_t>> {
  std::size_t operator()(const llvm::ArrayRef<int64_t> &array) const {
    return mlir::tt::ttnn::hashArrayRef(array);
  }
};
} // namespace std

namespace mlir::tt::op_model::ttnn {

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/Hashing.h"

#ifdef TTMLIR_ENABLE_OPMODEL
// alias to a common tt_metal types
using DataType = ::tt::tt_metal::DataType;
using Layout = ::tt::tt_metal::Layout;
using CoreRange = ::tt::tt_metal::CoreRange;
using CoreRangeSet = ::tt::tt_metal::CoreRangeSet;
using CoreCoord = ::tt::tt_metal::CoreCoord;
using ShardSpec = ::tt::tt_metal::ShardSpec;
using ShardOrientation = ::tt::tt_metal::ShardOrientation;
using TensorMemoryLayout = ::tt::tt_metal::TensorMemoryLayout;
using MemoryConfig = ::tt::tt_metal::MemoryConfig;

namespace detail {

DataType getDataType(const mlir::tt::ttnn::TTNNLayoutAttr layout) {

  auto dataType = layout.getDataType();

  switch (dataType) {
  case tt::DataType::Float32:
    return DataType::FLOAT32;
  case tt::DataType::BFloat16:
    return DataType::BFLOAT16;
  case tt::DataType::BFP_BFloat8:
    return DataType::BFLOAT8_B;
  case tt::DataType::BFP_BFloat4:
    return DataType::BFLOAT4_B;
  case tt::DataType::UInt32:
    return DataType::UINT32;
  case tt::DataType::UInt16:
    return DataType::UINT16;
  case tt::DataType::UInt8:
    return DataType::UINT8;
  default:
    throw std::runtime_error("Invalid element type");
  }
}

::ttnn::SimpleShape getSimpleShape(const ::llvm::ArrayRef<int64_t> &shape) {
  ::tt::tt_metal::SmallVector<uint32_t> small_vector_shape;
  for (const auto &dim : shape) {
    small_vector_shape.push_back(static_cast<uint32_t>(dim));
  }

  return ::ttnn::SimpleShape(small_vector_shape);
}

const std::array<uint32_t, 2>
getShardShape(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  const auto layoutShardTile = layout.getScalarShardShape();

  if (layoutShardTile.size() != 2) {
    llvm::errs() << "ERROR: layout_shard_tile.size() != 2\n";
    return {0, 0};
  }

  std::array<uint32_t, 2> shardShape;
  shardShape[0] = layoutShardTile[0];
  shardShape[1] = layoutShardTile[1];
  return shardShape;
}

Layout getPageLayout(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  llvm::outs() << "getPageLayout" << (layout.isTiled() ? "TILE" : "ROW_MAJOR")
               << "\n";
  return layout.isTiled() ? Layout::TILE : Layout::ROW_MAJOR;
}

CoreRangeSet getCoreRangeSet(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  // TODO(mbezulj): handle more complex grid shapes
  // assuming grid shape is one rect starting at (0,0)

  const auto layoutGrid = layout.getGrid();

  const auto layoutGridShape = layoutGrid.getShape();
  if (layoutGridShape.size() != 2) {
    llvm::errs() << "ERROR: layout_grid.getShape().size() == 2\n";
    return {};
  }

  return CoreRangeSet(
      CoreRange(CoreCoord(0, 0),
                CoreCoord(layoutGridShape[0] - 1, layoutGridShape[1] - 1)));
}

std::optional<ShardSpec>
layout_get_shard_spec(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  // tt_ShardOrientation is not part of ttnn::TTNNLayoutAttr;
  // defaulting to ROW_MAJOR. TODO: figure out if we need to expose this
  return isShardedMemoryLayout(layout.getMemLayout().getValue())
             ? std::make_optional(ShardSpec(getCoreRangeSet(layout),
                                            getShardShape(layout),
                                            ShardOrientation::ROW_MAJOR, false))
             : std::nullopt;
}

::tt::tt_metal::BufferType getBufferType(const mlir::MemRefType &memref) {
  auto memorySpace =
      mlir::cast<mlir::tt::ttnn::BufferTypeAttr>(memref.getMemorySpace())
          .getValue();

  switch (memorySpace) {
  case mlir::tt::ttnn::BufferType::DRAM:
    return ::tt::tt_metal::BufferType::DRAM;
  case mlir::tt::ttnn::BufferType::L1:
    return ::tt::tt_metal::BufferType::L1;
  case mlir::tt::ttnn::BufferType::SystemMemory:
    return ::tt::tt_metal::BufferType::SYSTEM_MEMORY;
  case mlir::tt::ttnn::BufferType::L1Small:
    return ::tt::tt_metal::BufferType::L1_SMALL;
  case mlir::tt::ttnn::BufferType::Trace:
    return ::tt::tt_metal::BufferType::TRACE;
  }
}

::tt::tt_metal::TensorMemoryLayout
getTensorMemoryLayout(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  auto tensorMemoryLayout = layout.getMemLayout().getValue();

  switch (tensorMemoryLayout) {
  case mlir::tt::ttnn::TensorMemoryLayout::Interleaved:
    return ::tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
  case mlir::tt::ttnn::TensorMemoryLayout::SingleBank:
    return ::tt::tt_metal::TensorMemoryLayout::SINGLE_BANK;
  case mlir::tt::ttnn::TensorMemoryLayout::HeightSharded:
    return ::tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED;
  case mlir::tt::ttnn::TensorMemoryLayout::WidthSharded:
    return ::tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
  case mlir::tt::ttnn::TensorMemoryLayout::BlockSharded:
    return ::tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
  }
}

::tt::tt_metal::MemoryConfig
getMemoryConfig(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {

  auto tensorMemoryLayout = getTensorMemoryLayout(layout);
  auto bufferType = getBufferType(layout.getMemref());

  auto shardSpec = layout_get_shard_spec(layout);
  return ::tt::tt_metal::MemoryConfig(tensorMemoryLayout, bufferType,
                                      shardSpec);
}

::tt::tt_metal::TensorLayout
getTensorLayout(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  return ::tt::tt_metal::TensorLayout(
      getDataType(layout), getPageLayout(layout), getMemoryConfig(layout));
}

::ttnn::TensorSpec getTensorSpec(const ::llvm::ArrayRef<int64_t> shape,
                                 const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  return ::ttnn::TensorSpec(getSimpleShape(shape), getTensorLayout(layout));
}

} // namespace detail

namespace wrapper {
template <class T>
bool isLegal(T &cache, auto &&...args) {
  ::ttnn::graph::QueryResponse query;
  try {
    query = cache.getOrCreate(
        std::make_tuple(std::forward<decltype(args)>(args)...));
  } catch (const std::exception &e) {
    query.status = ::ttnn::graph::ExecutionStatus::Error;
    query.error_message = e.what();
  }

  // check if query was successful
  if (query.status != ::ttnn::graph::ExecutionStatus::Success) {
    llvm::errs() << "FAILED " << cache.getName() << ": "
                 << query.error_message.value_or("<error message not set>");
    return false;
  }

  return true;
}

template <class T>
std::tuple<size_t, size_t, size_t> getL1Usage(T &cache, auto &&...args) {
  ::ttnn::graph::QueryResponse query;
  try {
    query = cache.getOrCreate(
        std::make_tuple(std::forward<decltype(args)>(args)...));
  } catch (const std::exception &e) {
    query.status = ::ttnn::graph::ExecutionStatus::Error;
    query.error_message = e.what();
  }

  // check if query was successful
  if (query.status != ::ttnn::graph::ExecutionStatus::Success) {
    llvm::errs() << "FAILED " << cache.getName() << ": "
                 << query.error_message.value_or("<error message not set>");
    return std::make_tuple(0, 0, 0);
  }

  return std::make_tuple(query.resource_usage.cb_peak_size_per_core,
                         query.resource_usage.l1_buffers_peak_per_core,
                         query.resource_usage.l1_output_buffer_per_core);
}
} // namespace wrapper
#endif // TTMLIR_ENABLE_OPMODEL

//===----------------------------------------------------------------------===//
// Test MLIR<->METAL Conversion functionality
//===----------------------------------------------------------------------===//
namespace conversion {
void debug(const ::llvm::ArrayRef<int64_t> shape,
           const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  layout.dump();
  auto tensorSpec = detail::getTensorSpec(shape, layout);
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
TupleCache<
    std::tuple<::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
               ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr>,
    ::ttnn::graph::QueryResponse,
    std::function<::ttnn::graph::QueryResponse(
        const std::tuple<
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr> &)>>
    g_constraints_cache_relu_op(
        "ReluOpInterface",
        [](const std::tuple<
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr> &key) {
          // open device / get existing device
          Device *device = SingletonDeviceContext::getInstance().getDevice();

          // prepare io specs
          const ::ttnn::TensorSpec input_spec =
              detail::getTensorSpec(std::get<0>(key), std::get<1>(key));
          const ::ttnn::TensorSpec output_spec =
              detail::getTensorSpec(std::get<2>(key), std::get<3>(key));

          // run op constraint query
          return ::ttnn::graph::query_op_constraints(
              ::ttnn::relu, device, input_spec,
              output_spec.tensor_layout().get_memory_config());
        });
#endif // TTMLIR_ENABLE_OPMODEL

bool ReluOpInterface::isLegal(
    const ::llvm::ArrayRef<int64_t> &inputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
    const ::llvm::ArrayRef<int64_t> &outputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return wrapper::isLegal(g_constraints_cache_relu_op, inputShape, inputLayout,
                          outputShape, outputLayout);
#else
  return true;
#endif // TTMLIR_ENABLE_OPMODEL
}

std::tuple<size_t, size_t, size_t> ReluOpInterface::getOpL1Usage(
    const ::llvm::ArrayRef<int64_t> &inputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
    const ::llvm::ArrayRef<int64_t> &outputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return wrapper::getL1Usage(g_constraints_cache_relu_op, inputShape,
                             inputLayout, outputShape, outputLayout);
#else
  return std::make_tuple(0, 0, 0);
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
TupleCache<
    std::tuple<::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
               ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
               ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr>,
    ::ttnn::graph::QueryResponse,
    std::function<::ttnn::graph::QueryResponse(
        const std::tuple<
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr> &)>>
    g_constraints_cache_add_op(
        "AddOpInterface",
        [](const std::tuple<
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr> &key) {
          // open device / get existing device
          Device *device = SingletonDeviceContext::getInstance().getDevice();

          // prepare io specs
          const ::ttnn::TensorSpec input_spec_a =
              detail::getTensorSpec(std::get<0>(key), std::get<1>(key));
          const ::ttnn::TensorSpec input_spec_b =
              detail::getTensorSpec(std::get<2>(key), std::get<3>(key));
          const ::ttnn::TensorSpec output_spec =
              detail::getTensorSpec(std::get<4>(key), std::get<5>(key));

          return ::ttnn::graph::query_op_constraints(
              ::ttnn::add, device, input_spec_a, input_spec_b,
              output_spec.data_type(),
              output_spec.tensor_layout().get_memory_config());
        });
#endif // TTMLIR_ENABLE_OPMODEL

bool AddOpInterface::isLegal(
    const ::llvm::ArrayRef<int64_t> &inputShape_a,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
    const ::llvm::ArrayRef<int64_t> &inputShape_b,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
    const ::llvm::ArrayRef<int64_t> &outputShape,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return wrapper::isLegal(g_constraints_cache_add_op, inputShape_a,
                          inputLayout_a, inputShape_b, inputLayout_b,
                          outputShape, outputLayout);
#else
  return true;
#endif // TTMLIR_ENABLE_OPMODEL
}

std::tuple<size_t, size_t, size_t> AddOpInterface::getOpL1Usage(
    const ::llvm::ArrayRef<int64_t> &inputShape_a,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
    const ::llvm::ArrayRef<int64_t> &inputShape_b,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
    const ::llvm::ArrayRef<int64_t> &outputShape,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return wrapper::getL1Usage(g_constraints_cache_add_op, inputShape_a,
                             inputLayout_a, inputShape_b, inputLayout_b,
                             outputShape, outputLayout);
#else
  return std::make_tuple(0, 0, 0);
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
TupleCache<
    std::tuple<::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr, int,
               ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr>,
    ::ttnn::graph::QueryResponse,
    std::function<::ttnn::graph::QueryResponse(
        const std::tuple<
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr, int,
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr> &)>>
    g_constraints_cache_softmax_op(
        "SoftmaxOpInterface",
        [](const std::tuple<
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr, int,
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr> &key) {
          // open device / get existing device
          Device *device = SingletonDeviceContext::getInstance().getDevice();

          // prepare io specs
          const ::ttnn::TensorSpec input_spec =
              detail::getTensorSpec(std::get<0>(key), std::get<1>(key));
          const int dim_arg = std::get<2>(key);
          const ::ttnn::TensorSpec output_spec =
              detail::getTensorSpec(std::get<3>(key), std::get<4>(key));

          // run op constraint query
          return ::ttnn::graph::query_op_constraints(
              ::ttnn::softmax, device, input_spec, dim_arg,
              output_spec.tensor_layout().get_memory_config());
        });
#endif // TTMLIR_ENABLE_OPMODEL

bool SoftmaxOpInterface::isLegal(
    const llvm::ArrayRef<int64_t> &inputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout, const int dim_arg,
    const llvm::ArrayRef<int64_t> &outputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return wrapper::isLegal(g_constraints_cache_softmax_op, inputShape,
                          inputLayout, dim_arg, outputShape, outputLayout);
#else
  return true;
#endif // TTMLIR_ENABLE_OPMODEL
}

std::tuple<size_t, size_t, size_t> SoftmaxOpInterface::getOpL1Usage(
    const llvm::ArrayRef<int64_t> &inputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout, const int dim_arg,
    const llvm::ArrayRef<int64_t> &outputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return wrapper::getL1Usage(g_constraints_cache_softmax_op, inputShape,
                             inputLayout, dim_arg, outputShape, outputLayout);
#else
  return std::make_tuple(0, 0, 0);
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
TupleCache<
    std::tuple<::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
               ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
               ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
               bool, bool>,
    ::ttnn::graph::QueryResponse,
    std::function<::ttnn::graph::QueryResponse(
        const std::tuple<
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr, bool,
            bool> &)>>
    g_constraints_cache_matmul_op(
        "MatmulOpInterface",
        [](const std::tuple<
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr,
            ::llvm::ArrayRef<int64_t>, ::mlir::tt::ttnn::TTNNLayoutAttr, bool,
            bool> &key) {
          // open device / get existing device
          Device *device = SingletonDeviceContext::getInstance().getDevice();

          // prepare io specs
          const ::ttnn::TensorSpec input_spec_a =
              detail::getTensorSpec(std::get<0>(key), std::get<1>(key));
          const ::ttnn::TensorSpec input_spec_b =
              detail::getTensorSpec(std::get<2>(key), std::get<3>(key));
          const ::ttnn::TensorSpec output_spec =
              detail::getTensorSpec(std::get<4>(key), std::get<5>(key));
          const bool transpose_a = std::get<6>(key);
          const bool transpose_b = std::get<7>(key);

          // run op constraint query
          return ::ttnn::graph::query_op_constraints(
              ::ttnn::matmul, device, input_spec_a, input_spec_b, transpose_a,
              transpose_b, output_spec.tensor_layout().get_memory_config(),
              output_spec.data_type());
        });
#endif // TTMLIR_ENABLE_OPMODEL

bool MatmulOpInterface::isLegal(
    const llvm::ArrayRef<int64_t> &inputShape_a,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
    const llvm::ArrayRef<int64_t> &inputShape_b,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
    const llvm::ArrayRef<int64_t> &outputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout, bool transpose_a,
    bool transpose_b) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return wrapper::isLegal(g_constraints_cache_matmul_op, inputShape_a,
                          inputLayout_a, inputShape_a, inputLayout_b,
                          outputShape, outputLayout, transpose_a, transpose_b);
#else
  return true;
#endif // TTMLIR_ENABLE_OPMODEL
}

std::tuple<size_t, size_t, size_t> MatmulOpInterface::getOpL1Usage(
    const llvm::ArrayRef<int64_t> &inputShape_a,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
    const llvm::ArrayRef<int64_t> &inputShape_b,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
    const llvm::ArrayRef<int64_t> &outputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout, bool transpose_a,
    bool transpose_b) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return wrapper::getL1Usage(
      g_constraints_cache_matmul_op, inputShape_a, inputLayout_a, inputShape_a,
      inputLayout_b, outputShape, outputLayout, transpose_a, transpose_b);
#else
  return std::make_tuple(0, 0, 0);
#endif // TTMLIR_ENABLE_OPMODEL
}

} // namespace mlir::tt::op_model::ttnn
