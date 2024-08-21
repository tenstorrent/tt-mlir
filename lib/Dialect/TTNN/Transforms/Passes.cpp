// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNOPENDEVICE
#define GEN_PASS_DEF_TTNNGENERIC
#define GEN_PASS_DEF_CONVERTTTIRTOTTNN
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNOpenDevice : public impl::TTNNOpenDeviceBase<TTNNOpenDevice> {
public:
  using impl::TTNNOpenDeviceBase<TTNNOpenDevice>::TTNNOpenDeviceBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OpBuilder builder(module);
    auto systemDesc = llvm::cast<tt::SystemDescAttr>(
        module->getAttr(tt::SystemDescAttr::name));

    module->walk([&](func::FuncOp func) {
      // For now just push the open and close device ops to the beginning and
      // end of the function
      assert(func.getBody().hasOneBlock());
      auto *block = &func.getBody().front();
      auto opRange = block->without_terminator();

      llvm::SmallVector<Attribute, 8> chipDescIndices;
      for (size_t i = 0; i < systemDesc.getChipDescIndices().size(); i++) {
        chipDescIndices.push_back(builder.getIntegerAttr(
            builder.getIntegerType(64), systemDesc.getChipDescIndices()[i]));
      }

      builder.setInsertionPoint(block, opRange.begin());
      auto openDevice = builder.create<OpenDeviceOp>(
          func.getLoc(),
          builder.getType<tt::DeviceType>(
              builder.getAttr<tt::DeviceAttr>(systemDesc)),
          builder.getArrayAttr(chipDescIndices));

      builder.setInsertionPoint(block, opRange.end());
      builder.create<CloseDeviceOp>(func.getLoc(), openDevice.getResult());
    });
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttnn::TTNNDialect>();
  }
};

// Rewrites `ttir.add` call to `ttnn.generic`. This is just a dummy rewriter spitting out hard-coded
// values for generic op attributes, as part of R&D with generic op. 
// TODO delete later.
// template <typename TTIROpType>
// class TTNNNamedTTIROpToTTNNGenericOpRewriter
//     : public OpRewritePattern<TTIROpType> {
// public:
//   using OpRewritePattern<TTIROpType>::OpRewritePattern;

//   LogicalResult matchAndRewrite(TTIROpType op,
//                                 PatternRewriter &rewriter) const final {
//     if (!std::is_same<TTIROpType, ttir::AddOp>::value) {
//       return failure();
//     }

//     // Get the grid attribute
//     tt::GridAttr grid = get_grid(rewriter);

//     // Get the core range over the entire specified grid
//     auto all_cores = rewriter.getAttr<tt::CoreRangeAttr>(grid);

//     // Create the dummy attributes
//     auto circular_buffer_attributes =
//         create_dummy_circular_buffer_attributes(rewriter, all_cores);
//     auto data_movement_attributes =
//         create_dummy_data_movement_attributes(rewriter, all_cores);
//     auto compute_attributes =
//         create_dummy_compute_attributes(rewriter, all_cores);

//     // Create the GenericOp
//     auto generic_op = rewriter.create<ttnn::GenericOp>(
//         op.getLoc(), op.getResults().getTypes(), op.getInputs(),
//         op.getOutputs(), rewriter.getArrayAttr(circular_buffer_attributes),
//         rewriter.getArrayAttr(data_movement_attributes),
//         rewriter.getArrayAttr(compute_attributes));

//     // Replace the original op with the new GenericOp
//     rewriter.replaceOp(op, generic_op.getResults());

//     return success();
//   }

// private:
//   SmallVector<Attribute>
//   create_dummy_circular_buffer_attributes(PatternRewriter &rewriter,
//                                           tt::CoreRangeAttr &core_range) const {
//     auto data_format = tt::DataType::BFloat16;
//     auto tile = tt::TileType::get(rewriter.getContext(), {32, 32}, data_format);
//     auto page_size = tile.getSizeBytes();
//     auto total_size = 2 * page_size;

//     auto in0_circular_buffer_attributes =
//         rewriter.getAttr<tt::CircularBufferAttributesAttr>(
//             tt::CB::c_in0, core_range, total_size, page_size, data_format);

//     auto in1_circular_buffer_attributes =
//         rewriter.getAttr<tt::CircularBufferAttributesAttr>(
//             tt::CB::c_in1, core_range, total_size, page_size, data_format);

//     auto out0_circular_buffer_attributes =
//         rewriter.getAttr<tt::CircularBufferAttributesAttr>(
//             tt::CB::c_out0, core_range, total_size, page_size, data_format);

//     return {in0_circular_buffer_attributes, in1_circular_buffer_attributes,
//             out0_circular_buffer_attributes};
//   }

//   SmallVector<Attribute>
//   create_dummy_data_movement_attributes(PatternRewriter &rewriter,
//                                         tt::CoreRangeAttr &core_range) const {
//     const char *reader_kernel_path =
//         "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/"
//         "dataflow/reader_binary_interleaved_start_id.cpp";

//     // Assuming interleaved mem layout where inputs are in DRAM.
//     auto src0_is_dram = true;
//     auto src1_is_dram = true;
//     SmallVector<uint32_t> reader_compile_time_args = {src0_is_dram,
//                                                       src1_is_dram};

//     auto reader_config = rewriter.getAttr<tt::DataMovementConfigAttr>(
//         tt::DataMovementType::Reader, reader_compile_time_args);

//     auto reader_attributes = rewriter.getAttr<tt::DataMovementAttributesAttr>(
//         core_range, rewriter.getAttr<StringAttr>(reader_kernel_path),
//         reader_config);

//     const char *writer_kernel_path =
//         "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
//         "writer_unary_interleaved_start_id.cpp";

//     // Assuming interleaved mem layout where outputs are in DRAM.
//     auto dst_is_dram = true;
//     SmallVector<uint32_t> writer_compile_time_args = {
//         static_cast<uint32_t>(tt::CB::c_out0), dst_is_dram};

//     auto writer_config = rewriter.getAttr<tt::DataMovementConfigAttr>(
//         tt::DataMovementType::Writer, reader_compile_time_args);

//     auto writer_attributes = rewriter.getAttr<tt::DataMovementAttributesAttr>(
//         core_range, rewriter.getAttr<StringAttr>(writer_kernel_path),
//         writer_config);

//     return {reader_attributes, writer_attributes};
//   }

//   SmallVector<Attribute>
//   create_dummy_compute_attributes(PatternRewriter &rewriter,
//                                   tt::CoreRangeAttr &core_range) const {
//     const char *compute_kernel_path =
//         "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/"
//         "eltwise_binary_kernel.cpp";

//     const std::map<std::string, std::string> defines_eltwise_add = {
//         {"ELTWISE_OP", "add_tiles"},
//         {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD"},
//     };

//     SmallVector<NamedAttribute> namedAttributes;

//     for (const auto &[name, value] : defines_eltwise_add) {
//       namedAttributes.emplace_back(rewriter.getStringAttr(name),
//                                    rewriter.getStringAttr(value));
//     }

//     auto compute_config = tt::ComputeConfigAttr::get(
//         rewriter.getContext(), tt::MathFidelity::HiFi4,
//         rewriter.getAttr<BoolAttr>(false), rewriter.getAttr<BoolAttr>(false),
//         rewriter.getAttr<BoolAttr>(false), {1},
//         rewriter.getDictionaryAttr(namedAttributes));

//     auto compute_attributes = rewriter.getAttr<tt::ComputeAttributesAttr>(
//         core_range, rewriter.getAttr<StringAttr>(compute_kernel_path),
//         compute_config);

//     return {compute_attributes};
//   }

//   mlir::tt::GridAttr get_grid(PatternRewriter &rewriter) const {
//     return GridAttr::get(rewriter.getContext(), {6, 6});
//   }

// };

class TTNNGeneric : public impl::TTNNGenericBase<TTNNGeneric> {
public:
  using impl::TTNNGenericBase<TTNNGeneric>::TTNNGenericBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());

    // patterns.add<TTNNNamedTTIROpToTTNNGenericOpRewriter<ttir::AddOp>>(&getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));

    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttnn::TTNNDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};

} // namespace mlir::tt::ttnn
