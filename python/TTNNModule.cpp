// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/CAPI/AffineMap.h"
#include "ttmlir/Bindings/Python/TTMLIRModule.h"

namespace mlir::ttmlir::python {
void populateTTNNModule(nb::module_ &m) {

  tt_attribute_class<tt::ttnn::CoreRangeAttr>(m, "CoreRangeAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> offset,
                     std::vector<int64_t> size) {
                    return wrap(tt::ttnn::CoreRangeAttr::get(unwrap(ctx),
                                                             offset, size));
                  })
      .def_static(
          "get_with_grid",
          [](MlirContext ctx, MlirAttribute grid, std::vector<int64_t> offset) {
            llvm::SmallVector<int64_t> offsetVec{0, 0};
            if (offset.size() == 2 && not(offset[0] == 0 && offset[1] == 0)) {
              offsetVec[0] = offset[0];
              offsetVec[1] = offset[1];
            }
            return wrap(tt::ttnn::CoreRangeAttr::get(
                unwrap(ctx), mlir::cast<tt::GridAttr>(unwrap(grid)),
                offsetVec));
          },
          nb::arg("ctx"), nb::arg("grid"),
          nb::arg("offset") = std::vector<int64_t>{0, 0})
      .def_prop_ro(
          "offset",
          [](tt::ttnn::CoreRangeAttr self) { return self.getOffset().vec(); })
      .def_prop_ro("size", [](tt::ttnn::CoreRangeAttr self) {
        return self.getSize().vec();
      });

  tt_attribute_class<tt::ttnn::LayoutAttr>(m, "LayoutAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t layout) {
                    return wrap(tt::ttnn::LayoutAttr::get(
                        unwrap(ctx), static_cast<tt::ttnn::Layout>(layout)));
                  })
      .def_prop_ro("value", [](tt::ttnn::LayoutAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::ttnn::TensorMemoryLayoutAttr>(m,
                                                       "TensorMemoryLayoutAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t tensorMemoryLayout) {
                    return wrap(tt::ttnn::TensorMemoryLayoutAttr::get(
                        unwrap(ctx), static_cast<tt::ttnn::TensorMemoryLayout>(
                                         tensorMemoryLayout)));
                  })
      .def_prop_ro("value", [](tt::ttnn::TensorMemoryLayoutAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });
  tt_attribute_class<tt::ttnn::BufferTypeAttr>(m, "BufferTypeAttr")
      .def_static(
          "get",
          [](MlirContext ctx, uint32_t bufferType) {
            return wrap(tt::ttnn::BufferTypeAttr::get(
                unwrap(ctx), static_cast<tt::ttnn::BufferType>(bufferType)));
          })
      .def_prop_ro("value", [](tt::ttnn::BufferTypeAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::ttnn::ShardSpecAttr>(m, "ShardSpecAttr")
      .def_static("get",
                  [](MlirContext ctx, tt::ttnn::ShapeAttr shardShape) {
                    return wrap(
                        tt::ttnn::ShardSpecAttr::get(unwrap(ctx), shardShape));
                  })
      .def_prop_ro("shard_shape", &tt::ttnn::ShardSpecAttr::getShardShape);

  tt_attribute_class<tt::ttnn::MemoryConfigAttr>(m, "MemoryConfigAttr")
      .def_static("get",
                  [](MlirContext ctx,
                     tt::ttnn::TensorMemoryLayoutAttr tensorMemoryLayoutAttr,
                     tt::ttnn::BufferTypeAttr bufferTypeAttr,
                     tt::ttnn::ShardSpecAttr shardSpecAttr) {
                    return wrap(tt::ttnn::MemoryConfigAttr::get(
                        unwrap(ctx), bufferTypeAttr, shardSpecAttr,
                        tensorMemoryLayoutAttr));
                  })
      .def_static(
          "get_by_value",
          [](MlirContext ctx, uint32_t tensorMemoryLayout, uint32_t bufferType,
             std::vector<int64_t> shardShape) {
            tt::ttnn::TensorMemoryLayoutAttr layoutAttr =
                tt::ttnn::TensorMemoryLayoutAttr::get(
                    unwrap(ctx), static_cast<tt::ttnn::TensorMemoryLayout>(
                                     tensorMemoryLayout));

            return wrap(tt::ttnn::MemoryConfigAttr::get(
                unwrap(ctx),
                tt::ttnn::BufferTypeAttr::get(
                    unwrap(ctx), static_cast<tt::ttnn::BufferType>(bufferType)),
                tt::ttnn::ShardSpecAttr::get(
                    unwrap(ctx),
                    tt::ttnn::ShapeAttr::get(unwrap(ctx), shardShape)),
                layoutAttr));
          })
      .def_prop_ro("tensor_memory_layout",
                   &tt::ttnn::MemoryConfigAttr::getTensorMemoryLayout)
      .def_prop_ro("buffer_type", &tt::ttnn::MemoryConfigAttr::getBufferType)
      .def_prop_ro("shard_spec", &tt::ttnn::MemoryConfigAttr::getShardSpec);

  tt_attribute_class<tt::ttnn::ShapeAttr>(m, "ShapeAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> shape) {
                    return wrap(tt::ttnn::ShapeAttr::get(unwrap(ctx), shape));
                  })
      .def_prop_ro("shape", [](tt::ttnn::ShapeAttr self) {
        return std::vector<int64_t>(self.getShape().begin(),
                                    self.getShape().end());
      });

  tt_attribute_class<tt::ttnn::MeshShapeAttr>(m, "MeshShapeAttr")
      .def_static("get",
                  [](MlirContext ctx, int64_t y, int64_t x) {
                    return wrap(
                        tt::ttnn::MeshShapeAttr::get(unwrap(ctx), y, x));
                  })
      .def_prop_ro("y", &tt::ttnn::MeshShapeAttr::getY)
      .def_prop_ro("x", &tt::ttnn::MeshShapeAttr::getX);

  tt_attribute_class<tt::ttnn::TTNNLayoutAttr>(m, "TTNNLayoutAttr")
      .def_static(
          "get",
          [](MlirContext ctx, MlirAffineMap linear, MlirAttribute grid,
             MlirType memref,
             std::optional<unsigned> memLayout = std::nullopt) {
            tt::ttnn::TensorMemoryLayoutAttr memLayoutAttr;
            if (memLayout.has_value()) {
              memLayoutAttr = tt::ttnn::TensorMemoryLayoutAttr::get(
                  unwrap(ctx),
                  static_cast<tt::ttnn::TensorMemoryLayout>(memLayout.value()));
            }
            return wrap(tt::ttnn::TTNNLayoutAttr::get(
                unwrap(ctx), mlir::cast<AffineMap>(unwrap(linear)),
                mlir::cast<tt::GridAttr>(unwrap(grid)),
                mlir::cast<MemRefType>(unwrap(memref)), memLayoutAttr));
          })
      .def_prop_ro(
          "linear",
          [](tt::ttnn::TTNNLayoutAttr self) { return wrap(self.getLinear()); })
      .def_prop_ro("grid_attr", &tt::ttnn::TTNNLayoutAttr::getGrid)
      .def_prop_ro(
          "memref",
          [](tt::ttnn::TTNNLayoutAttr self) { return wrap(self.getMemref()); })
      .def_prop_ro("tensor_memory_layout_as_int",
                   [](tt::ttnn::TTNNLayoutAttr self)
                       -> std::variant<uint32_t, nb::object> {
                     if (!self.getMemLayout()) {
                       return nb::none();
                     }
                     return static_cast<uint32_t>(
                         self.getMemLayout().getValue());
                   })
      .def_prop_ro("memory_layout_as_int",
                   [](tt::ttnn::TTNNLayoutAttr self) {
                     return static_cast<uint32_t>(self.getLayout());
                   })
      .def_prop_ro("data_type_as_int", [](tt::ttnn::TTNNLayoutAttr self) {
        return static_cast<uint32_t>(self.getDataType());
      });
  tt_attribute_class<tt::ttnn::Conv2dConfigAttr>(m, "Conv2dConfigAttr")
      .def_static(
          "get",
          [](MlirContext ctx, tt::DataType dtype, tt::DataType weightsDtype,
             StringAttr activation, IntegerAttr inputChannelsAlignment,
             BoolAttr deallocateActivation, BoolAttr reallocateHaloOutput,
             IntegerAttr actBlockHOverride, IntegerAttr actBlockWDiv,
             BoolAttr reshardIfNotOptimal, BoolAttr overrideShardingConfig,
             tt::ttnn::TensorMemoryLayoutAttr shardLayout, Attribute coreGrid,
             BoolAttr transposeShards, tt::ttnn::Layout outputLayout,
             BoolAttr enableActDoubleBuffer, BoolAttr enableWeightsDoubleBuffer,
             BoolAttr enableSplitReader, BoolAttr enableSubblockPadding) {
            return wrap(tt::ttnn::Conv2dConfigAttr::get(
                unwrap(ctx), dtype, weightsDtype, activation,
                inputChannelsAlignment, deallocateActivation,
                reallocateHaloOutput, actBlockHOverride, actBlockWDiv,
                reshardIfNotOptimal, overrideShardingConfig, shardLayout,
                coreGrid, transposeShards, outputLayout, enableActDoubleBuffer,
                enableWeightsDoubleBuffer, enableSplitReader,
                enableSubblockPadding));
          })
      .def_prop_ro("dtype",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     return static_cast<uint32_t>(self.getDtype());
                   })
      .def_prop_ro("weights_dtype",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     return static_cast<uint32_t>(self.getDtype());
                   })
      .def_prop_ro("activation",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     return self.getActivation().str();
                   })
      .def_prop_ro("input_channels_alignment",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     return self.getInputChannelsAlignment().getInt();
                   })
      .def_prop_ro("deallocate_activation",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     return self.getDeallocateActivation().getValue();
                   })
      .def_prop_ro("reallocate_halo_output",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     return self.getReallocateHaloOutput().getValue();
                   })
      .def_prop_ro("act_block_h_override",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     return self.getActBlockHOverride().getInt();
                   })
      .def_prop_ro("act_block_w_div",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     return self.getActBlockWDiv().getInt();
                   })
      .def_prop_ro("reshard_if_not_optimal",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     return self.getReshardIfNotOptimal().getValue();
                   })
      .def_prop_ro("override_sharding_config",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     return self.getOverrideShardingConfig().getValue();
                   })
      .def_prop_ro("shard_layout",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     return static_cast<uint32_t>(
                         self.getShardLayout().getValue());
                   })
      // TODO(vkovacevic): parse core_grid
      .def_prop_ro("core_grid",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     assert(!self.getCoreGrid());
                     return nb::none();
                   })
      .def_prop_ro("transpose_shards",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     return self.getTransposeShards().getValue();
                   })
      .def_prop_ro("output_layout",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     return static_cast<uint32_t>(self.getOutputLayout());
                   })
      .def_prop_ro("enable_act_double_buffer",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     return self.getEnableActDoubleBuffer().getValue();
                   })
      .def_prop_ro("enable_weights_double_buffer",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     return self.getEnableWeightsDoubleBuffer().getValue();
                   })
      .def_prop_ro("enable_split_reader",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     return self.getEnableSplitReader().getValue();
                   })
      .def_prop_ro("enable_subblock_padding",
                   [](tt::ttnn::Conv2dConfigAttr self) {
                     return self.getEnableSubblockPadding().getValue();
                   });
}
} // namespace mlir::ttmlir::python
