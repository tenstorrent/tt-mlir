// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/CAPI/AffineMap.h"
#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "ttmlir-c/TTNNAttrs.h"

#include <nanobind/stl/optional.h>
namespace mlir::ttmlir::python {
void populateTTNNModule(nb::module_ &m) {

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
             MlirType memref, std::optional<unsigned> memLayout = std::nullopt,
             std::optional<tt::TensorMeshShardingAttr> tensorMeshSharding =
                 std::nullopt) {
            tt::ttnn::TensorMemoryLayoutAttr memLayoutAttr;
            if (memLayout.has_value()) {
              memLayoutAttr = tt::ttnn::TensorMemoryLayoutAttr::get(
                  unwrap(ctx),
                  static_cast<tt::ttnn::TensorMemoryLayout>(memLayout.value()));
            }
            tt::TensorMeshShardingAttr tensorMeshShardingAttr;
            if (tensorMeshSharding.has_value()) {
              tensorMeshShardingAttr = tensorMeshSharding.value();
            }
            return wrap(tt::ttnn::TTNNLayoutAttr::get(
                unwrap(ctx), mlir::cast<AffineMap>(unwrap(linear)),
                mlir::cast<tt::GridAttr>(unwrap(grid)),
                mlir::cast<MemRefType>(unwrap(memref)), memLayoutAttr,
                tensorMeshShardingAttr));
          },
          nb::arg("ctx"), nb::arg("linear"), nb::arg("grid"), nb::arg("memref"),
          nb::arg("memLayout") = nb::none(),
          nb::arg("tensorMeshSharding") = nb::none())
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
          [](MlirContext ctx, std::optional<tt::DataType> dtype,
             std::optional<tt::DataType> weightsDtype, StringAttr activation,
             std::optional<uint32_t> inputChannelsAlignment,
             BoolAttr deallocateActivation, BoolAttr reallocateHaloOutput,
             std::optional<uint32_t> actBlockHOverride,
             std::optional<uint32_t> actBlockWDiv, BoolAttr reshardIfNotOptimal,
             BoolAttr overrideShardingConfig,
             std::optional<tt::ttnn::TensorMemoryLayout> shardLayout,
             tt::ttnn::CoreRangeSetAttr coreGrid, BoolAttr transposeShards,
             std::optional<tt::ttnn::Layout> outputLayout,
             BoolAttr preprocessWeightsOnDevice,
             BoolAttr alwaysPreprocessWeights, BoolAttr enableActDoubleBuffer,
             BoolAttr enableWeightsDoubleBuffer, BoolAttr enableSplitReader,
             BoolAttr enableSubblockPadding) {
            MLIRContext *context = unwrap(ctx);

            return wrap(tt::ttnn::Conv2dConfigAttr::get(
                context, dtype, weightsDtype, activation,
                inputChannelsAlignment, deallocateActivation,
                reallocateHaloOutput, actBlockHOverride, actBlockWDiv,
                reshardIfNotOptimal, overrideShardingConfig, shardLayout,
                coreGrid, transposeShards, outputLayout,
                preprocessWeightsOnDevice, alwaysPreprocessWeights,
                enableActDoubleBuffer, enableWeightsDoubleBuffer,
                enableSplitReader, enableSubblockPadding));
          })
      .def_prop_ro("dtype_as_int",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, uint32_t> {
                     if (!self.getDtype()) {
                       return nb::none();
                     }
                     return static_cast<uint32_t>(*self.getDtype());
                   })
      .def_prop_ro("weights_dtype_as_int",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, uint32_t> {
                     if (!self.getWeightsDtype()) {
                       return nb::none();
                     }
                     return static_cast<uint32_t>(*self.getWeightsDtype());
                   })
      .def_prop_ro("activation",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, std::string> {
                     if (!self.getActivation()) {
                       return nb::none();
                     }
                     return self.getActivation().getValue().str();
                   })
      .def_prop_ro("input_channels_alignment",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, uint32_t> {
                     if (!self.getInputChannelsAlignment()) {
                       return nb::none();
                     }
                     return static_cast<uint32_t>(
                         *self.getInputChannelsAlignment());
                   })
      .def_prop_ro("deallocate_activation",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, bool> {
                     if (!self.getDeallocateActivation()) {
                       return nb::none();
                     }
                     return self.getDeallocateActivation().getValue();
                   })
      .def_prop_ro("reallocate_halo_output",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, bool> {
                     if (!self.getReallocateHaloOutput()) {
                       return nb::none();
                     }
                     return self.getReallocateHaloOutput().getValue();
                   })
      .def_prop_ro("act_block_h_override",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, uint32_t> {
                     if (!self.getActBlockHOverride()) {
                       return nb::none();
                     }
                     return *self.getActBlockHOverride();
                   })
      .def_prop_ro("act_block_w_div",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, uint32_t> {
                     if (!self.getActBlockWDiv()) {
                       return nb::none();
                     }
                     return *self.getActBlockWDiv();
                   })
      .def_prop_ro("reshard_if_not_optimal",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, bool> {
                     if (!self.getReshardIfNotOptimal()) {
                       return nb::none();
                     }
                     return self.getReshardIfNotOptimal().getValue();
                   })
      .def_prop_ro("override_sharding_config",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, bool> {
                     if (!self.getOverrideShardingConfig()) {
                       return nb::none();
                     }
                     return self.getOverrideShardingConfig().getValue();
                   })
      .def_prop_ro("shard_layout_as_int",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, uint32_t> {
                     if (!self.getShardLayout()) {
                       return nb::none();
                     }
                     return static_cast<uint32_t>(*self.getShardLayout());
                   })
      // TODO(vkovacevic): parse core_grid #2781
      .def_prop_ro("core_grid",
                   [](tt::ttnn::Conv2dConfigAttr self) { return nb::none(); })
      .def_prop_ro("transpose_shards",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, bool> {
                     if (!self.getTransposeShards()) {
                       return nb::none();
                     }
                     return self.getTransposeShards().getValue();
                   })
      .def_prop_ro("output_layout_as_int",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, uint32_t> {
                     if (!self.getOutputLayout()) {
                       return nb::none();
                     }
                     return static_cast<uint32_t>(*self.getOutputLayout());
                   })
      .def_prop_ro("preprocess_weights_on_device",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, bool> {
                     if (!self.getPreprocessWeightsOnDevice()) {
                       return nb::none();
                     }
                     return self.getPreprocessWeightsOnDevice().getValue();
                   })
      .def_prop_ro("always_preprocess_weights",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, bool> {
                     if (!self.getAlwaysPreprocessWeights()) {
                       return nb::none();
                     }
                     return self.getAlwaysPreprocessWeights().getValue();
                   })
      .def_prop_ro("enable_act_double_buffer",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, bool> {
                     if (!self.getEnableActDoubleBuffer()) {
                       return nb::none();
                     }
                     return self.getEnableActDoubleBuffer().getValue();
                   })
      .def_prop_ro("enable_weights_double_buffer",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, bool> {
                     if (!self.getEnableWeightsDoubleBuffer()) {
                       return nb::none();
                     }
                     return self.getEnableWeightsDoubleBuffer().getValue();
                   })
      .def_prop_ro("enable_split_reader",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, bool> {
                     if (!self.getEnableSplitReader()) {
                       return nb::none();
                     }
                     return self.getEnableSplitReader().getValue();
                   })
      .def_prop_ro("enable_subblock_padding",
                   [](tt::ttnn::Conv2dConfigAttr self)
                       -> std::variant<nb::object, bool> {
                     if (!self.getEnableSubblockPadding()) {
                       return nb::none();
                     }
                     return self.getEnableSubblockPadding().getValue();
                   });

  tt_attribute_class<tt::ttnn::CoreRangeAttr>(m, "CoreRangeAttr")
      .def_static("get",
                  [](MlirContext ctx, MlirAttribute start, MlirAttribute end) {
                    return ttmlirTTNNCoreRangeAttrGet(ctx, start, end);
                  })
      .def_prop_ro("start_coord", &tt::ttnn::CoreRangeAttr::getStartCoord)
      .def_prop_ro("end_coord", &tt::ttnn::CoreRangeAttr::getEndCoord);

  tt_attribute_class<tt::ttnn::CoreCoordAttr>(m, "CoreCoordAttr")
      .def_static("get",
                  [](MlirContext ctx, uint64_t y, uint64_t x) {
                    return ttmlirTTNNCoreCoordAttrGet(ctx, y, x);
                  })
      .def_prop_ro("y", &tt::ttnn::CoreCoordAttr::getY)
      .def_prop_ro("x", &tt::ttnn::CoreCoordAttr::getX);

  tt_attribute_class<tt::ttnn::CoreRangeSetAttr>(m, "CoreRangeSetAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<MlirAttribute> coreRanges) {
                    return ttmlirTTNNCoreRangeSetAttrGet(ctx, coreRanges.data(),
                                                         coreRanges.size());
                  })
      .def_prop_ro("core_ranges", [](tt::ttnn::CoreRangeSetAttr self) {
        return self.getCoreRanges().vec();
      });

  tt_attribute_class<tt::ttnn::UnaryWithParamAttr>(m, "UnaryWithParamAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t opTypeEnum,
                     std::vector<MlirAttribute> params) {
                    return ttmlirTTNNUnaryWithParamAttr(
                        ctx, opTypeEnum, params.data(), params.size());
                  })
      .def_prop_ro("op_type_as_int",
                   [](tt::ttnn::UnaryWithParamAttr self) {
                     return static_cast<uint32_t>(self.getOpType());
                   })
      .def_prop_ro("params", [](tt::ttnn::UnaryWithParamAttr self) {
        return self.getParams().vec();
      });

  tt_attribute_class<tt::ttnn::MatmulMultiCoreReuseProgramConfigAttr>(
      m, "MatmulMultiCoreReuseProgramConfigAttr")
      .def_static("get",
                  [](MlirContext ctx, MlirAttribute computeWithStorageGridSize,
                     uint64_t in0BlockW, uint64_t outSubblockH,
                     uint64_t outSubblockW, uint64_t perCoreM,
                     uint64_t perCoreN) {
                    return ttmlirTTNNMatmulMultiCoreReuseProgramConfigAttr(
                        ctx, computeWithStorageGridSize, in0BlockW,
                        outSubblockH, outSubblockW, perCoreM, perCoreN);
                  })
      .def_prop_ro("compute_with_storage_grid_size",
                   [](tt::ttnn::MatmulMultiCoreReuseProgramConfigAttr self) {
                     return wrap(self.getComputeWithStorageGridSize());
                   })
      .def_prop_ro(
          "in0_block_w",
          &tt::ttnn::MatmulMultiCoreReuseProgramConfigAttr::getIn0BlockW)
      .def_prop_ro(
          "out_subblock_h",
          &tt::ttnn::MatmulMultiCoreReuseProgramConfigAttr::getOutSubblockH)
      .def_prop_ro(
          "out_subblock_w",
          &tt::ttnn::MatmulMultiCoreReuseProgramConfigAttr::getOutSubblockW)
      .def_prop_ro(
          "per_core_m",
          &tt::ttnn::MatmulMultiCoreReuseProgramConfigAttr::getPerCoreM)
      .def_prop_ro(
          "per_core_n",
          &tt::ttnn::MatmulMultiCoreReuseProgramConfigAttr::getPerCoreN);

  tt_attribute_class<tt::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr>(
      m, "MatmulMultiCoreReuseMultiCastProgramConfigAttr")
      .def_static(
          "get",
          [](MlirContext ctx, MlirAttribute computeWithStorageGridSize,
             uint64_t in0BlockW, uint64_t outSubblockH, uint64_t outSubblockW,
             uint64_t outBlockH, uint64_t outBlockW, uint64_t perCoreM,
             uint64_t perCoreN, bool transposeMcast,
             MlirAttribute fusedActivation, bool fuseBatch) {
            return ttmlirTTNNMatmulMultiCoreReuseMultiCastProgramConfigAttr(
                ctx, computeWithStorageGridSize, in0BlockW, outSubblockH,
                outSubblockW, outBlockH, outBlockW, perCoreM, perCoreN,
                transposeMcast, fusedActivation, fuseBatch);
          })
      .def_prop_ro(
          "compute_with_storage_grid_size",
          [](tt::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr self) {
            return wrap(self.getComputeWithStorageGridSize());
          })
      .def_prop_ro("in0_block_w",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr::
                       getIn0BlockW)
      .def_prop_ro("out_subblock_h",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr::
                       getOutSubblockH)
      .def_prop_ro("out_subblock_w",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr::
                       getOutSubblockW)
      .def_prop_ro("out_block_h",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr::
                       getOutBlockH)
      .def_prop_ro("out_block_w",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr::
                       getOutBlockW)
      .def_prop_ro("per_core_m",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr::
                       getPerCoreM)
      .def_prop_ro("per_core_n",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr::
                       getPerCoreN)
      .def_prop_ro("transpose_mcast",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr::
                       getTransposeMcast)
      .def_prop_ro(
          "fused_activation",
          [](tt::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr self) {
            return wrap(self.getFusedActivation());
          })
      .def_prop_ro("fuse_batch",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCastProgramConfigAttr::
                       getFuseBatch);

  tt_attribute_class<
      tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr>(
      m, "MatmulMultiCoreReuseMultiCast1DProgramConfigAttr")
      .def_static(
          "get",
          [](MlirContext ctx, MlirAttribute computeWithStorageGridSize,
             uint64_t in0BlockW, uint64_t outSubblockH, uint64_t outSubblockW,
             uint64_t outBlockH, uint64_t outBlockW, uint64_t perCoreM,
             uint64_t perCoreN, bool fuseBatch, MlirAttribute fusedActivation,
             bool mcastIn0, bool gatherIn0, MlirAttribute hopCores,
             uint64_t numGlobalCbReceivers) {
            return ttmlirTTNNMatmulMultiCoreReuseMultiCast1DProgramConfigAttrGet(
                ctx, computeWithStorageGridSize, in0BlockW, outSubblockH,
                outSubblockW, outBlockH, outBlockW, perCoreM, perCoreN,
                fuseBatch, fusedActivation, mcastIn0, gatherIn0, hopCores,
                numGlobalCbReceivers);
          })
      .def_prop_ro(
          "compute_with_storage_grid_size",
          [](tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr self) {
            return wrap(self.getComputeWithStorageGridSize());
          })
      .def_prop_ro("in0_block_w",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::
                       getIn0BlockW)
      .def_prop_ro("out_subblock_h",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::
                       getOutSubblockH)
      .def_prop_ro("out_subblock_w",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::
                       getOutSubblockW)
      .def_prop_ro("out_block_h",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::
                       getOutBlockH)
      .def_prop_ro("out_block_w",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::
                       getOutBlockW)
      .def_prop_ro("per_core_m",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::
                       getPerCoreM)
      .def_prop_ro("per_core_n",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::
                       getPerCoreN)
      .def_prop_ro("fuse_batch",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::
                       getFuseBatch)
      .def_prop_ro(
          "fused_activation",
          [](tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr self) {
            return wrap(self.getFusedActivation());
          })
      .def_prop_ro("mcast_in0",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::
                       getMcastIn0)
      .def_prop_ro("gather_in0",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::
                       getGatherIn0)
      .def_prop_ro(
          "hop_cores",
          [](tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr self) {
            return wrap(self.getHopCores());
          })
      .def_prop_ro("num_global_cb_receivers",
                   &tt::ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::
                       getNumGlobalCbReceivers);

  tt_attribute_class<
      tt::ttnn::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
      m, "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr")
      .def_static(
          "get",
          [](MlirContext ctx, uint64_t in0BlockW, uint64_t perCoreM,
             uint64_t perCoreN, MlirAttribute fusedActivation) {
            return ttmlirTTNNMatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttrGet(
                ctx, in0BlockW, perCoreM, perCoreN, fusedActivation);
          })
      .def_prop_ro(
          "in0_block_w",
          &tt::ttnn::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr::
              getIn0BlockW)
      .def_prop_ro(
          "per_core_m",
          &tt::ttnn::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr::
              getPerCoreM)
      .def_prop_ro(
          "per_core_n",
          &tt::ttnn::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr::
              getPerCoreN)
      .def_prop_ro(
          "fused_activation",
          [](tt::ttnn::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr
                 self) { return wrap(self.getFusedActivation()); });
}
} // namespace mlir::ttmlir::python
