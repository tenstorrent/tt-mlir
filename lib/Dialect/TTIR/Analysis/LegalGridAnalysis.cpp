// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/LegalGridAnalysis.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir_interface_api.hpp"
#include <llvm/Support/raw_ostream.h>
#include <optional>
#include <sstream>
#include <string>

// todo: move to a common place as ttnn_mlir_interface_wrapper
namespace mlir::ttnn_wrapper {
std::vector<uint32_t> memref_get_tensor_shape(const mlir::MemRefType &memref) {
  std::vector<uint32_t> shape;
  for (auto i = 0; i < memref.getRank(); i++) {
    shape.push_back(memref.getShape()[i]);
  }
  return shape;
}

std::array<uint32_t, 2>
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

std::string layout_get_tensor_layout(const mlir::tt::LayoutAttr &layout) {
  return layout.isTiled() ? mlir::tt::stringifyOperandConstraint(
                                mlir::tt::OperandConstraint::Tile)
                          : mlir::tt::stringifyOperandConstraint(
                                mlir::tt::OperandConstraint::Scalar);
}

std::vector<std::array<uint32_t, 4>>
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

std::optional<ttnn::mlir_interface::shard_spec_tuple>
layout_get_shard_spec(const mlir::tt::LayoutAttr &layout) {
  return isShardedMemoryLayout(layout.getMemLayout())
             ? std::make_optional(std::make_tuple(
                   layout_get_grid_shape(layout),
                   layout_get_shard_shape(layout),
                   "row_major", // todo: expose parameter to LayoutAttr
                   false))
             : std::nullopt;
}

const std::string memref_get_buffer_type_str(const mlir::MemRefType &memref) {
  return ::mlir::tt::MemorySpaceEnumToString(
             mlir::cast<tt::MemorySpaceAttr>(memref.getMemorySpace())
                 .getValue())
      .str();
}

const std::string
layout_attr_get_tensor_memory_layout_str(const mlir::tt::LayoutAttr &layout) {
  return ::mlir::tt::TensorMemoryLayoutEnumToString(layout.getMemLayout())
      .str();
}

ttnn::mlir_interface::memory_config_tuple
layout_get_memory_config(const mlir::tt::LayoutAttr &layout) {
  std::string tensor_memory_layout =
      layout_attr_get_tensor_memory_layout_str(layout);
  std::string buffer_type = memref_get_buffer_type_str(layout.getMemref());
  auto shard_spec = layout_get_shard_spec(layout);
  return std::make_tuple(tensor_memory_layout, buffer_type, shard_spec);
}

std::string memref_get_element_type(const mlir::MemRefType &memref) {
  mlir::Type element_type = memref.getElementType();
  // what's better way to to this?
  // auto data_type = mlir::cast<DataType>(element_type);
  std::string ret_value;
  llvm::raw_string_ostream os(ret_value);
  element_type.print(os);
  return os.str();
}
} // namespace mlir::ttnn_wrapper

namespace mlir::tt::ttir {

bool mock_is_output_tensor_legal_for_op(Operation *op, LayoutAttr layout) {
  // Placeholder, needs to be replaced with a call the the TTNN op interface.

  // serialize mlir structures to interface structures
  auto memref = layout.getMemref();
  auto shape = ttnn_wrapper::memref_get_tensor_shape(memref);
  std::string data_type = ttnn_wrapper::memref_get_element_type(memref);
  ttnn::mlir_interface::memory_config_tuple memory_config =
      ttnn_wrapper::layout_get_memory_config(layout);
  std::string tensor_layout =
      ttnn_wrapper::layout_attr_get_tensor_memory_layout_str(layout);

  // call mlir_interface library per op name
  auto op_name_str = op->getName().getStringRef().str();
  bool is_valid = false;

  if (llvm::isa<MultiplyOp>(op) || llvm::isa<AddOp>(op) ||
      llvm::isa<SubtractOp>(op)) {
    is_valid =
        ttnn::mlir_interface::does_binary_op_support_input_output_constraints(
            shape, memory_config, data_type, tensor_layout, shape,
            memory_config, data_type, tensor_layout, memory_config, data_type);
  } else if (llvm::isa<SoftmaxOp>(op)) {
    is_valid =
        ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(
            shape, memory_config, data_type, tensor_layout, shape,
            memory_config, data_type);
  } else if (llvm::isa<ReluOp>(op)) {
    is_valid =
        ttnn::mlir_interface::does_unary_op_support_input_output_constraints(
            "RELU", // todo agree upon mapping to ttnn::mlir_interface
            shape, memory_config, data_type, tensor_layout, shape,
            memory_config, data_type);
  } else {
    llvm::outs() << op->getName() << " missing ttnn interface\n";
    return false;
  }

  return is_valid;
}

bool tensor_shape_compatible_with_shard(Operation *op, LayoutAttr layout) {
  // These constraints are implemented seperatelly in every TTNN op.
  // Almost nothing seems to be shared between EVERY op, so is hard to have any
  // logic here without the risk of discarding a valid configuraiton or modeling
  // the constraint for each op. This logic may be offloaded to the TTNN op
  // interface.

  // For now we will check if the tilised tensor dims are divisible by the grid
  // dims. This will definitly discard possible valid configurations, but is a
  // start.
  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getResult(0).getType());
  llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();

  int64_t MTiles = 1;
  if (tensorType.getRank() >= 2) {
    MTiles = (tensorShape.rbegin()[1] + 31) / 32;
  }

  int64_t KTIles = (tensorShape.back() + 31) / 32;

  int64_t gridR = layout.getGrid().getShape()[0];
  int64_t gridC = layout.getGrid().getShape()[1];

  return (MTiles % gridR == 0) && (KTIles % gridC == 0);
}

bool cantChangeOutputLayout(Operation *op) {
  // Only TTIR ops.
  if (not llvm::isa<TTIROp>(op)) {
    return true;
  }
  if (llvm::isa<ToLayoutOp>(op)) {
    return true;
  }
  return false;
}

bool LegalGridAnalysis::applyOverrides() {
  // Lookup grid size overrides based on location information for current
  // operation.
  //

  // TODO(odjuricic): Need to override all params, not just grid size.
  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getResult(0).getType());
  LayoutAttr layout = mlir::cast<LayoutAttr>(tensorType.getEncoding());
  llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();

  if (analysisInput.gridSizeOverrides && isa<NameLoc>(op->getLoc())) {
    StringRef loc_str_op_name = mlir::cast<NameLoc>(op->getLoc()).getName();
    auto gridOverride = analysisInput.gridSizeOverrides->find(loc_str_op_name);
    if (gridOverride != analysisInput.gridSizeOverrides->end()) {
      analysisResult.push_back(layout.withGrid(
          op->getContext(), tensorShape,
          GridAttr::get(op->getContext(),
                        ArrayRef<int64_t>(gridOverride->second))));
      analysisResult.push_back(layout.withGrid(
          op->getContext(), tensorShape,
          GridAttr::get(op->getContext(),
                        {gridOverride->second[0], gridOverride->second[1]})));
      return true;
    }
  }

  return false;
}

void LegalGridAnalysis::analysisImplementation() {
  // A first incomplete implementation of the LegalGridAnalysis.
  // This implementation is a placeholder and is meant to just enable testing of
  // other components.

  // Skip operations that don't have output tensors.
  if (op->getNumResults() == 0) {
    return;
  }

  // Get output tensor type.
  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(op->getResult(0).getType());
  LayoutAttr layout = mlir::cast<LayoutAttr>(tensorType.getEncoding());

  // Return existing layout if it is not possible to change it.
  if (cantChangeOutputLayout(op)) {
    analysisResult.push_back(layout);
    return;
  }

  // DRAM
  // No grid is set since the tensor is not sharded.
  // TODO(odjuricic): We need to set grid here since it will be used as the
  // compute gird. (not implemented in runtime atm)
  LayoutAttr dram =
      layout.withMemorySpace(op->getContext(), MemorySpace::DeviceDRAM)
          .withMemoryLayout(op->getContext(), TensorMemoryLayout::Interleaved)
          .withGrid(op->getContext(), tensorType,
                    GridAttr::get(op->getContext(),
                                  analysisInput.maxGrid.getShape()));
  if (mock_is_output_tensor_legal_for_op(op, dram)) {
    analysisResult.push_back(dram);
  }

  // L1 Interleaved (same as above).
  LayoutAttr l1Interleaved =
      layout.withMemorySpace(op->getContext(), MemorySpace::DeviceL1)
          .withMemoryLayout(op->getContext(), TensorMemoryLayout::Interleaved)
          .withGrid(op->getContext(), tensorType,
                    GridAttr::get(op->getContext(),
                                  analysisInput.maxGrid.getShape()));
  if (mock_is_output_tensor_legal_for_op(op, l1Interleaved)) {
    analysisResult.push_back(l1Interleaved);
  }

  // L1 Sharded
  LayoutAttr shardedBase =
      layout.withMemorySpace(op->getContext(), MemorySpace::DeviceL1);
  std::vector<LayoutAttr> shardedResults;

  // Block Sharded
  for (auto width = 1; width <= analysisInput.maxGrid.getShape()[0]; ++width) {
    for (auto height = 1; height <= analysisInput.maxGrid.getShape()[1];
         ++height) {
      shardedResults.push_back(
          shardedBase
              .withGrid(op->getContext(), tensorType,
                        GridAttr::get(op->getContext(), {width, height}))
              .withMemoryLayout(op->getContext(),
                                TensorMemoryLayout::BlockSharded));
    }
  }

  auto numCores =
      analysisInput.maxGrid.getShape()[0] * analysisInput.maxGrid.getShape()[1];
  // Height Sharded
  // TODO(odjuricic): Missing affine mapping to actual grid. Need to check with
  // runtime implementation on what to produce here.
  for (auto height = 2; height <= numCores; ++height) {
    shardedResults.push_back(
        shardedBase
            .withGrid(op->getContext(), tensorType,
                      GridAttr::get(op->getContext(), {height, 1}))
            .withMemoryLayout(op->getContext(),
                              TensorMemoryLayout::HeightSharded));
  }

  // Width Sharded
  for (auto width = 2; width <= numCores; ++width) {
    shardedResults.push_back(
        shardedBase
            .withGrid(op->getContext(), tensorType,
                      GridAttr::get(op->getContext(), {1, width}))
            .withMemoryLayout(op->getContext(),
                              TensorMemoryLayout::WidthSharded));
  }

  // Filter layouts based on output tensor legality for current op.
  shardedResults.erase(
      std::remove_if(shardedResults.begin(), shardedResults.end(),
                     [this](LayoutAttr layout) {
                       return !tensor_shape_compatible_with_shard(op, layout) ||
                              !mock_is_output_tensor_legal_for_op(op, layout);
                     }),
      shardedResults.end());

  // Pick top largest sharded grids.
  std::sort(shardedResults.begin(), shardedResults.end(),
            [](LayoutAttr a, LayoutAttr b) {
              return a.getGrid().getShape()[0] * a.getGrid().getShape()[1] >
                     b.getGrid().getShape()[0] * b.getGrid().getShape()[1];
            });

  analysisResult.insert(
      analysisResult.end(), shardedResults.begin(),
      shardedResults.begin() +
          std::min(analysisInput.maxShardedGrids,
                   static_cast<int64_t>(shardedResults.size())));
}
} // namespace mlir::tt::ttir
