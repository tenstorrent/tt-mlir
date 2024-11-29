// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "TTNNOpModel.h"

#include "TupleCache.h"

#ifdef TTMLIR_ENABLE_OPMODEL
#include "SingletonDeviceContext.h"
#include "TTNNOpModelLib_Impl.h"
#include "TupleCache.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include <llvm/Support/Casting.h>
#include <mlir/IR/AttrTypeSubElements.h>

#include <cstddef>
#include <stdexcept>
#endif // TTMLIR_ENABLE_OPMODEL

namespace mlir::tt::ttnn {

std::size_t hashMemRefType(MemRefType memref) {
  // is memref.getLayout() leveraged in TTNN dialect?
  return llvm::hash_combine(llvm::hash_combine_range(memref.getShape().begin(),
                                                     memref.getShape().end()),
                            memref.getElementType(), memref.getMemorySpace());
}

std::size_t hashLayoutAttr(TTNNLayoutAttr layout) {
  // ignoring linear/affine map, because it higher level of abstraction
  return llvm::hash_combine(
      hashMemRefType(layout.getMemref()),
      llvm::hash_combine_range(layout.getGrid().getShape().begin(),
                               layout.getGrid().getShape().end()),
      layout.getMemLayout());
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
} // namespace std

namespace mlir::tt::op_model::ttnn {

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/Hashing.h"

// // Example function to hash a MemRefType
// llvm::hash_code hashMemRefType(mlir::MemRefType memref) {
//   return
//   llvm::hash_combine(llvm::hash_combine_range(memref.getShape().begin(),
//                                                      memref.getShape().end()),
//                             memref.getElementType(), memref.getMemorySpace(),
//                             elementTypeToDataType(memref.getElementType()));
// }

// std::size_t hashLayoutAttr(mlir::tt::ttnn::TTNNLayoutAttr layout) {
//   return llvm::hash_combine(
//       hashMemRefType(layout.getMemref()),
//       llvm::hash_combine_range(layout.getGrid().getShape().begin(),
//                                layout.getGrid().getShape().end()),
//       layout.getMemLayout(), layout.isTiled());
// }

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

DataType getDataType(const mlir::MemRefType &memref) {

  auto dataType = elementTypeToDataType(memref.getElementType());

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

::ttnn::SimpleShape getTensorShape(const mlir::MemRefType &memref) {
  ::tt::tt_metal::SmallVector<uint32_t> small_vector_shape(
      memref.getShape().begin(), memref.getShape().end());
  return ::ttnn::SimpleShape(small_vector_shape);
}

const std::array<uint32_t, 2>
getShardShape(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  const auto layoutShardTile = layout.getShardShape();

  if (layoutShardTile.size() != 2) {
    llvm::errs() << "ERROR: layout_shard_tile.size() != 2\n";
    return {0, 0};
  }

  std::array<uint32_t, 2> shardShape;
  shardShape[0] = layoutShardTile[0];
  shardShape[1] = layoutShardTile[1];
  return shardShape;
}

Layout getTensorLayout(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
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

  return CoreRangeSet(CoreRange(CoreCoord(0, layoutGridShape[0]),
                                CoreCoord(0, layoutGridShape[1])));
}

std::optional<ShardSpec>
layout_get_shard_spec(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  // tt_ShardOrientation is not part of ttnn::TTNNLayoutAttr;
  // defaulting to ROW_MAJOR. TODO: figure out if we need to expose this
  return isShardedMemoryLayout(layout.getMemLayout())
             ? std::make_optional(ShardSpec(getCoreRangeSet(layout),
                                            getShardShape(layout),
                                            ShardOrientation::ROW_MAJOR, false))
             : std::nullopt;
}

::tt::tt_metal::BufferType getBufferType(const mlir::MemRefType &memref) {
  auto memorySpace =
      mlir::cast<tt::MemorySpaceAttr>(memref.getMemorySpace()).getValue();

  switch (memorySpace) {
  case tt::MemorySpace::DeviceDRAM:
    return ::tt::tt_metal::BufferType::DRAM;
  case tt::MemorySpace::DeviceL1:
    return ::tt::tt_metal::BufferType::L1;
  default: // TODO(mbezulj): handle other memory spaces
    throw std::runtime_error("Unsupported memory space");
  }
}

::tt::tt_metal::TensorMemoryLayout
getTensorMemoryLayout(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  auto tensorMemoryLayout = layout.getMemLayout();

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
  default:
    throw std::runtime_error("Unsupported tensor memory layout");
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

} // namespace detail
#endif // TTMLIR_ENABLE_OPMODEL

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

bool ReluOpInterface::isLegal(
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // open device / get existing device
  Device *device = SingletonDeviceContext::get_instance().get_device();

  // prepare io specs
  const ::ttnn::TensorSpec input_spec = detail::getTensorSpec(inputLayout);
  const ::ttnn::TensorSpec output_spec = detail::getTensorSpec(outputLayout);

  // run op constraint query
  auto query = ::ttnn::compiler_interface::unary_op_constraints<::ttnn::relu6>(
      device, input_spec, output_spec);

  // check if query was successful
  if (query.status != ::ttnn::compiler_interface::ExecutionStatus::Success) {
    llvm::outs() << "FAILED ReluOpInterface::isLegal: "
                 << query.error_message.value_or("no error message");
    return false;
  }
#else

  return true;
#endif // TTMLIR_ENABLE_OPMODEL
}

std::tuple<size_t, size_t, size_t> ReluOpInterface::getOpL1Usage(
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // open device / get existing device
  Device *device = SingletonDeviceContext::get_instance().get_device();

  // prepare io specs
  const ::ttnn::TensorSpec input_spec = detail::getTensorSpec(inputLayout);
  const ::ttnn::TensorSpec output_spec = detail::getTensorSpec(outputLayout);

  // run op constraint query
  auto query = ::ttnn::compiler_interface::unary_op_constraints<::ttnn::relu6>(
      device, input_spec, output_spec);

  // check if query was successful
  if (query.status != ::ttnn::compiler_interface::ExecutionStatus::Success) {
    llvm::outs() << "FAILED ReluOpInterface::getOpL1Usage: "
                 << query.error_message.value_or("no error message");
    return std::make_tuple(0, 0, 0);
  }
  return std::make_tuple(query.resource_usage.cb_peak_size_per_core,
                         query.resource_usage.l1_buffers_peak_per_core,
                         query.resource_usage.l1_output_buffer_per_core);
#else
  return std::make_tuple(0, 0, 0);
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

TupleCache<std::tuple<::mlir::tt::ttnn::TTNNLayoutAttr,
                      ::mlir::tt::ttnn::TTNNLayoutAttr,
                      ::mlir::tt::ttnn::TTNNLayoutAttr>,
           mlir::tt::op_model::QueryResponse,
           std::function<QueryResponse(
               const std::tuple<::mlir::tt::ttnn::TTNNLayoutAttr,
                                ::mlir::tt::ttnn::TTNNLayoutAttr,
                                ::mlir::tt::ttnn::TTNNLayoutAttr> &)>>
    g_constrainst_cache_add_op(
        "add_op", [](const std::tuple<::mlir::tt::ttnn::TTNNLayoutAttr,
                                      ::mlir::tt::ttnn::TTNNLayoutAttr,
                                      ::mlir::tt::ttnn::TTNNLayoutAttr> &key) {
          return QueryResponse{ExecutionStatus::Success, ResourceUsage{0, 0, 0},
                               std::nullopt};
        });

bool AddOpInterface::isLegal(
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // open device / get existing device
  Device *device = SingletonDeviceContext::get_instance().get_device();

  // prepare io specs
  const ::ttnn::TensorSpec input_spec_a = detail::getTensorSpec(inputLayout_a);
  const ::ttnn::TensorSpec input_spec_b = detail::getTensorSpec(inputLayout_b);
  const ::ttnn::TensorSpec output_spec = detail::getTensorSpec(outputLayout);

  // run op constraint query
  auto query = ::ttnn::compiler_interface::binary_op_constraints<::ttnn::add>(
      device, input_spec_a, input_spec_b, output_spec);

  // check if query was successful
  if (query.status != ::ttnn::compiler_interface::ExecutionStatus::Success) {
    llvm::outs() << "FAILED AddOpInterface::isLegal: "
                 << query.error_message.value_or("no error message");
    return false;
  }
  return true;
#else

  g_constrainst_cache_add_op.get_or_create(
      {inputLayout_a, inputLayout_b, outputLayout});

  return true;
#endif // TTMLIR_ENABLE_OPMODEL
}

std::tuple<size_t, size_t, size_t> AddOpInterface::getOpL1Usage(
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // open device / get existing device
  Device *device = SingletonDeviceContext::get_instance().get_device();

  // prepare io specs
  const ::ttnn::TensorSpec input_spec_a = detail::getTensorSpec(inputLayout_a);
  const ::ttnn::TensorSpec input_spec_b = detail::getTensorSpec(inputLayout_b);
  const ::ttnn::TensorSpec output_spec = detail::getTensorSpec(outputLayout);

  // run op constraint query
  auto query = ::ttnn::compiler_interface::binary_op_constraints<::ttnn::add>(
      device, input_spec_a, input_spec_b, output_spec);

  // check if query was successful
  if (query.status != ::ttnn::compiler_interface::ExecutionStatus::Success) {
    llvm::outs() << "FAILED AddOpInterface::isLegal: "
                 << query.error_message.value_or("no error message");
    return std::make_tuple(0, 0, 0);
  }
  return std::make_tuple(query.resource_usage.cb_peak_size_per_core,
                         query.resource_usage.l1_buffers_peak_per_core,
                         query.resource_usage.l1_output_buffer_per_core);
#else
  g_constrainst_cache_add_op.get_or_create(
      {inputLayout_a, inputLayout_b, outputLayout});
  return std::make_tuple(0, 0, 0);
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

bool SoftmaxOpInterface::isLegal(
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout, const int dim_arg,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // open device / get existing device
  Device *device = SingletonDeviceContext::get_instance().get_device();

  // prepare io specs
  const ::ttnn::TensorSpec input_spec = detail::getTensorSpec(inputLayout);
  const ::ttnn::TensorSpec output_spec = detail::getTensorSpec(outputLayout);

  // run op constraint query
  auto query = ::ttnn::compiler_interface::softmax_op_constraints(
      device, input_spec, dim_arg, output_spec);

  // check if query was successful
  if (query.status != ::ttnn::compiler_interface::ExecutionStatus::Success) {
    llvm::outs() << "FAILED SoftmaxOpInterface::isLegal: "
                 << query.error_message.value_or("no error message");
    return false;
  }
  return true;
#else
  return true;
#endif // TTMLIR_ENABLE_OPMODEL
}

std::tuple<size_t, size_t, size_t> SoftmaxOpInterface::getOpL1Usage(
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout, const int dim_arg,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // open device / get existing device
  Device *device = SingletonDeviceContext::get_instance().get_device();

  // prepare io specs
  const ::ttnn::TensorSpec input_spec = detail::getTensorSpec(inputLayout);
  const ::ttnn::TensorSpec output_spec = detail::getTensorSpec(outputLayout);

  // run op constraint query
  auto query = ::ttnn::compiler_interface::softmax_op_constraints(
      device, input_spec, dim_arg, output_spec);

  // check if query was successful
  if (query.status != ::ttnn::compiler_interface::ExecutionStatus::Success) {
    llvm::outs() << "FAILED SoftmaxOpInterface::getOpL1Usage: "
                 << query.error_message.value_or("no error message");
    return std::make_tuple(0, 0, 0);
  }
  return std::make_tuple(query.resource_usage.cb_peak_size_per_core,
                         query.resource_usage.l1_buffers_peak_per_core,
                         query.resource_usage.l1_output_buffer_per_core);
#else
  return std::make_tuple(0, 0, 0);
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

bool MatmulOpInterface::isLegal(
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout, bool transpose_a,
    bool transpose_b) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // open device / get existing device
  Device *device = SingletonDeviceContext::get_instance().get_device();

  // prepare io specs
  const ::ttnn::TensorSpec input_spec_a = detail::getTensorSpec(inputLayout_a);
  const ::ttnn::TensorSpec input_spec_b = detail::getTensorSpec(inputLayout_b);
  const ::ttnn::TensorSpec output_spec = detail::getTensorSpec(outputLayout);

  // run op constraint query
  auto query = ::ttnn::compiler_interface::binary_op_constraints<::ttnn::add>(
      device, input_spec_a, input_spec_b, output_spec);

  // check if query was successful
  if (query.status != ::ttnn::compiler_interface::ExecutionStatus::Success) {
    llvm::outs() << "FAILED MatmulOpInterface::isLegal: "
                 << query.error_message.value_or("no error message");
    return false;
  }
  return true;
#else
  return true;
#endif // TTMLIR_ENABLE_OPMODEL
}

std::tuple<size_t, size_t, size_t> MatmulOpInterface::getOpL1Usage(
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout, bool transpose_a,
    bool transpose_b) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // open device / get existing device
  Device *device = SingletonDeviceContext::get_instance().get_device();

  // prepare io specs
  const ::ttnn::TensorSpec input_spec_a = detail::getTensorSpec(inputLayout_a);
  const ::ttnn::TensorSpec input_spec_b = detail::getTensorSpec(inputLayout_b);
  const ::ttnn::TensorSpec output_spec = detail::getTensorSpec(outputLayout);

  // run op constraint query
  auto query = ::ttnn::compiler_interface::binary_op_constraints<::ttnn::add>(
      device, input_spec_a, input_spec_b, output_spec);

  // check if query was successful
  if (query.status != ::ttnn::compiler_interface::ExecutionStatus::Success) {
    llvm::outs() << "FAILED MatmulOpInterface::isLegal: "
                 << query.error_message.value_or("no error message");
    return std::make_tuple(0, 0, 0);
  }
  return std::make_tuple(query.resource_usage.cb_peak_size_per_core,
                         query.resource_usage.l1_buffers_peak_per_core,
                         query.resource_usage.l1_output_buffer_per_core);
#else
  return std::make_tuple(0, 0, 0);
#endif // TTMLIR_ENABLE_OPMODEL
}

} // namespace mlir::tt::op_model::ttnn
