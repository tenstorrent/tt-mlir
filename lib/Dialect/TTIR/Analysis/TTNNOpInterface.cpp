// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/TTNNOPInterface.h"

#include "mlir_interface_api.hpp"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include <cassert>
#include <mlir/IR/Operation.h>

namespace mlir::tt::ttir {

namespace ttnn_wrapper {
static std::vector<uint32_t>
memref_get_tensor_shape(const mlir::MemRefType &memref) {
  std::vector<uint32_t> shape;
  for (auto i = 0; i < memref.getRank(); i++) {
    shape.push_back(memref.getShape()[i]);
  }
  return shape;
}

static std::array<uint32_t, 2>
layout_get_shard_shape(const mlir::tt::LayoutAttr &layout) {
  const auto layout_shard_tile = layout.getShardShape(false);

  if (layout_shard_tile.size() != 2) {
    llvm::errs() << "ERROR: layout_shard_tile.size() != 2\n";
    return {0, 0};
  }

  std::array<uint32_t, 2> shard_shape;
  shard_shape[0] = layout_shard_tile[0];
  shard_shape[1] = layout_shard_tile[1];
  return shard_shape;
}

static std::string
layout_get_tensor_layout(const mlir::tt::LayoutAttr &layout) {
  return layout.isTiled() ? mlir::tt::stringifyOperandConstraint(
                                mlir::tt::OperandConstraint::Tile)
                          : mlir::tt::stringifyOperandConstraint(
                                mlir::tt::OperandConstraint::Scalar);
}

static std::vector<std::array<uint32_t, 4>>
layout_get_grid_shape(const mlir::tt::LayoutAttr &layout) {
  // todo: handle more complex grid shapes
  // assuming grid shape is one rect starting at (0,0)

  const auto layout_grid = layout.getGrid();

  if (layout_grid.getShape().size() != 2) {
    llvm::errs() << "ERROR: layout_grid.getShape().size() == 2\n";
    return {};
  }

  std::vector<std::array<uint32_t, 4>> grid_shapes;
  std::array<uint32_t, 4> grid_shape;
  grid_shape[0] = 0;
  grid_shape[1] = 0;
  grid_shape[2] = layout_grid.getShape()[0];
  grid_shape[3] = layout_grid.getShape()[1];
  grid_shapes.emplace_back(grid_shape);

  return grid_shapes;
}

static std::optional<ttnn::mlir_interface::shard_spec_tuple>
layout_get_shard_spec(const mlir::tt::LayoutAttr &layout) {
  return isShardedMemoryLayout(layout.getMemLayout())
             ? std::make_optional(std::make_tuple(
                   layout_get_grid_shape(layout),
                   layout_get_shard_shape(layout),
                   "row_major", // todo: expose parameter to LayoutAttr
                   false))
             : std::nullopt;
}

static std::string memref_get_buffer_type_str(const mlir::MemRefType &memref) {
  return ::mlir::tt::MemorySpaceEnumToString(
             mlir::cast<tt::MemorySpaceAttr>(memref.getMemorySpace())
                 .getValue())
      .str();
}

static std::string
layout_attr_get_tensor_memory_layout_str(const mlir::tt::LayoutAttr &layout) {
  return ::mlir::tt::TensorMemoryLayoutEnumToString(layout.getMemLayout())
      .str();
}

static ttnn::mlir_interface::memory_config_tuple
layout_get_memory_config(const mlir::tt::LayoutAttr &layout) {
  std::string tensor_memory_layout =
      layout_attr_get_tensor_memory_layout_str(layout);
  std::string buffer_type = memref_get_buffer_type_str(layout.getMemref());
  auto shard_spec = layout_get_shard_spec(layout);
  return std::make_tuple(tensor_memory_layout, buffer_type, shard_spec);
}

static std::string memref_get_element_type(const mlir::MemRefType &memref) {
  mlir::Type element_type = memref.getElementType();
  // what's better way to to this?
  // auto data_type = mlir::cast<DataType>(element_type);
  std::string ret_value;
  llvm::raw_string_ostream os(ret_value);
  element_type.print(os);
  return os.str();
}
}; // namespace ttnn_wrapper

bool is_op_configuration_valid(const std::vector<Operation *> &producer_ops,
                               const std::vector<LayoutAttr> &producer_layouts,
                               Operation *consumer_op,
                               const LayoutAttr &consumer_layout) {

  const std::size_t num_operands = consumer_op->getNumOperands();
  assert(producer_ops.size() == num_operands);
  assert(producer_layouts.size() == num_operands);

  // serialize mlir structures to interface structures
  std::vector<std::vector<uint32_t>> input_shapes(num_operands);
  std::vector<std::string> input_data_types(num_operands);
  std::vector<ttnn::mlir_interface::memory_config_tuple> input_memory_configs(
      num_operands);
  std::vector<std::string> input_tensor_layouts(num_operands);

  for (unsigned int input_idx = 0; input_idx < num_operands; input_idx++) {
    input_shapes[input_idx] =
        ttnn_wrapper::memref_get_tensor_shape(consumer_layout.getMemref());
    input_data_types[input_idx] =
        ttnn_wrapper::memref_get_element_type(consumer_layout.getMemref());
    input_memory_configs[input_idx] =
        ttnn_wrapper::layout_get_memory_config(consumer_layout);
    input_tensor_layouts[input_idx] =
        ttnn_wrapper::layout_get_tensor_layout(consumer_layout);
  }

  auto output_shape =
      ttnn_wrapper::memref_get_tensor_shape(consumer_layout.getMemref());
  std::string output_data_type =
      ttnn_wrapper::memref_get_element_type(consumer_layout.getMemref());
  ttnn::mlir_interface::memory_config_tuple output_memory_config =
      ttnn_wrapper::layout_get_memory_config(consumer_layout);
  std::string output_tensor_layout =
      ttnn_wrapper::layout_get_tensor_layout(consumer_layout);

  bool is_valid = false;
  if (llvm::isa<MultiplyOp>(consumer_op) || llvm::isa<AddOp>(consumer_op) ||
      llvm::isa<SubtractOp>(consumer_op)) {
    is_valid =
        ttnn::mlir_interface::does_binary_op_support_input_output_constraints(
            input_shapes[0], input_memory_configs[0], input_data_types[0],
            input_tensor_layouts[0], input_shapes[1], input_memory_configs[1],
            input_data_types[1], input_tensor_layouts[1], output_memory_config,
            output_data_type);
  } else if (llvm::isa<SoftmaxOp>(consumer_op)) {
    is_valid =
        ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(
            input_shapes[0], input_memory_configs[0], input_data_types[0],
            input_tensor_layouts[0], output_shape, output_memory_config,
            output_data_type);
  } else if (llvm::isa<ReluOp>(consumer_op)) {
    is_valid =
        ttnn::mlir_interface::does_unary_op_support_input_output_constraints(
            "RELU", // todo agree upon mapping to ttnn::mlir_interface
            input_shapes[0], input_memory_configs[0], input_data_types[0],
            input_tensor_layouts[0], output_shape, output_memory_config,
            output_data_type);
  } else {
    llvm::outs() << consumer_op->getName() << " missing ttnn interface\n";
    is_valid = false;
  }

  return is_valid;
}

bool is_op_configuration_valid(Operation *consumer_op,
                               const LayoutAttr &consumer_layout) {
  std::vector<LayoutAttr> producer_layouts(consumer_op->getNumOperands(),
                                           consumer_layout);
  std::vector<Operation *> producer_ops;
  for (auto operand : consumer_op->getOperands()) {
    producer_ops.push_back(operand.getDefiningOp());
  }

  return is_op_configuration_valid(producer_ops, producer_layouts, consumer_op,
                                   consumer_layout);
}

}; // namespace mlir::tt::ttir
