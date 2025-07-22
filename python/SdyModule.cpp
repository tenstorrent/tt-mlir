// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h" // IWYU pragma: keep
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h" // IWYU pragma: keep
#include "nanobind/stl/string.h"   // IWYU pragma: keep
#include "nanobind/stl/variant.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"   // IWYU pragma: keep
#include "shardy/integrations/c/attributes.h"
#include "shardy/integrations/c/dialect.h"

namespace mlir::ttmlir::python {

namespace nb = nanobind;

void populateSdyModule(nb::module_ &m) {

  // Returns a vector containing elements with type T extracted from an
  // attribute using the two provided callbacks.
  template <typename T>
  std::vector<T> propertyVector(
      MlirAttribute attr, llvm::function_ref<intptr_t(MlirAttribute)> sizeFn,
      llvm::function_ref<T(MlirAttribute, intptr_t)> getFn) {
    std::vector<T> result;
    intptr_t size = sizeFn(attr);
    result.reserve(size);
    for (intptr_t i = 0; i < size; ++i) {
      result.push_back(getFn(attr, i));
    }
    return result;
  }

  nb::str toPyString(MlirStringRef mlirStringRef) {
    return nb::str(mlirStringRef.data, mlirStringRef.length);
  }

  MlirStringRef toStringRef(const std::string &s) {
    return mlirStringRefCreate(s.c_str(), s.size());
  }

  MlirAttribute toMeshOrRefAttr(
      MlirContext ctx,
      const std::variant<std::string, MlirAttribute> &meshOrRef) {
    if (auto *meshName = std::get_if<std::string>(&meshOrRef)) {
      return mlirFlatSymbolRefAttrGet(ctx, toStringRef(*meshName));
    }
    return std::get<MlirAttribute>(meshOrRef);
  }

  m.doc() = "SDY main Python extension";

  //
  // Dialects.
  //

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__sdy__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  //
  // Attributes.
  //

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "MeshAxisAttr", sdyAttributeIsAMeshAxisAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::string &name, int64_t size,
             MlirContext ctx) {
            return cls(sdyMeshAxisAttrGet(ctx, toStringRef(name), size));
          },
          nb::arg("cls"), nb::arg("name"), nb::arg("size"),
          nb::arg("context").none() = nb::none(),
          "Creates a MeshAxisAttr with the given axis name and size.")
      .def_property_readonly("name",
                             [](MlirAttribute self) {
                               return toPyString(sdyMeshAxisAttrGetName(self));
                             })
      .def_property_readonly("size", [](MlirAttribute self) {
        return sdyMeshAxisAttrGetSize(self);
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "MeshAttr", sdyAttributeIsAMeshAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::vector<MlirAttribute> &meshAxes,
             const std::vector<int64_t> &deviceIds, MlirContext ctx) {
            return cls(sdyMeshAttrGet(ctx, meshAxes.size(), meshAxes.data(),
                                      deviceIds.size(), deviceIds.data()));
          },
          nb::arg("cls"), nb::arg("mesh_axes"),
          nb::arg("device_ids") = std::vector<int64_t>(),
          nb::arg("context").none() = nb::none(),
          "Creates a MeshAttr with the given mesh axes.")
      .def_property_readonly("device_ids",
                             [](MlirAttribute self) {
                               return propertyVector<int64_t>(
                                   self, sdyMeshAttrGetDeviceIdsSize,
                                   sdyMeshAttrGetDeviceIdsElem);
                             })
      .def_property_readonly("axes", [](MlirAttribute self) {
        return propertyVector<MlirAttribute>(self, sdyMeshAttrGetAxesSize,
                                             sdyMeshAttrGetAxesElem);
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "SubAxisInfoAttr", sdyAttributeIsASubAxisInfoAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, int64_t preSize, int64_t size, MlirContext ctx) {
            return cls(sdySubAxisInfoAttrGet(ctx, preSize, size));
          },
          nb::arg("cls"), nb::arg("pre_size"), nb::arg("size"),
          nb::arg("context").none() = nb::none(),
          "Creates a SubAxisInfoAttr with the given pre-size and size.")
      .def_property_readonly(
          "pre_size",
          [](MlirAttribute self) { return sdySubAxisInfoAttrGetPreSize(self); })
      .def_property_readonly("size", [](MlirAttribute self) {
        return sdySubAxisInfoAttrGetSize(self);
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "AxisRefAttr", sdyAttributeIsAnAxisRefAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::string &name,
             std::optional<MlirAttribute> subAxisInfoAttr, MlirContext ctx) {
            return cls(sdyAxisRefAttrGet(ctx, toStringRef(name),
                                         subAxisInfoAttr.has_value()
                                             ? *subAxisInfoAttr
                                             : MlirAttribute()));
          },
          nb::arg("cls"), nb::arg("name"),
          nb::arg("sub_axis_info").none() = nb::none(),
          nb::arg("context").none() = nb::none(),
          "Creates an AxisRefAttr with the given name and SubAxisInfoAttr.")
      .def_property_readonly("name",
                             [](MlirAttribute self) {
                               return toPyString(sdyAxisRefAttrGetName(self));
                             })
      .def_property_readonly("sub_axis_info", [](MlirAttribute self) {
        MlirAttribute subAxisInfo = sdyAxisRefAttrGetSubAxisInfo(self);
        return subAxisInfo.ptr == nullptr ? std::nullopt
                                          : std::optional(subAxisInfo);
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "DimensionShardingAttr", sdyAttributeIsADimensionShardingAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::vector<MlirAttribute> &axes,
             bool isClosed, std::optional<int64_t> priority, MlirContext ctx) {
            return cls(sdyDimensionShardingAttrGet(
                ctx, axes.size(), axes.data(), isClosed,
                priority.has_value() ? *priority : -1));
          },
          nb::arg("cls"), nb::arg("axes"), nb::arg("is_closed"),
          nb::arg("priority").none() = nb::none(),
          nb::arg("context").none() = nb::none(),
          "Creates a DimensionShardingAttr with the given axes, whether it's "
          "closed, and priority.")
      .def_property_readonly("axes",
                             [](MlirAttribute self) {
                               return propertyVector<MlirAttribute>(
                                   self, sdyDimensionShardingAttrGetAxesSize,
                                   sdyDimensionShardingAttrGetAxesElem);
                             })
      .def_property_readonly("is_closed",
                             [](MlirAttribute self) {
                               return sdyDimensionShardingAttrGetIsClosed(self);
                             })
      .def_property_readonly("priority", [](MlirAttribute self) {
        int64_t priority = sdyDimensionShardingAttrGetPriority(self);
        return priority == -1 ? std::nullopt : std::optional(priority);
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "TensorShardingAttr", sdyAttributeIsATensorShardingAttr)
      .def_classmethod(
          "get",
          [](nb::object cls,
             const std::variant<std::string, MlirAttribute> &meshOrRef,
             const std::vector<MlirAttribute> &dimensionShardings,
             const std::vector<MlirAttribute> &replicatedAxes,
             const std::vector<MlirAttribute> &unreducedAxes, MlirContext ctx) {
            return cls(sdyTensorShardingAttrGet(
                ctx, toMeshOrRefAttr(ctx, meshOrRef), dimensionShardings.size(),
                dimensionShardings.data(), replicatedAxes.size(),
                replicatedAxes.data(), unreducedAxes.size(),
                unreducedAxes.data()));
          },
          nb::arg("cls"), nb::arg("mesh_or_ref"),
          nb::arg("dimension_shardings"),
          nb::arg("replicated_axes") = std::vector<MlirAttribute>(),
          nb::arg("unreduced_axes") = std::vector<MlirAttribute>(),
          nb::arg("context").none() = nb::none(),
          "Creates a TensorShardingAttr with either an inlined mesh or mesh "
          "name, dimension shardings, and replicated axes.")
      .def_property_readonly("mesh_or_ref",
                             [](MlirAttribute self) {
                               return sdyTensorShardingAttrGetMeshOrRef(self);
                             })
      .def_property_readonly("dimension_shardings",
                             [](MlirAttribute self) {
                               return propertyVector<MlirAttribute>(
                                   self,
                                   sdyTensorShardingAttrGetDimShardingsSize,
                                   sdyTensorShardingAttrGetDimShardingsElem);
                             })
      .def_property_readonly("replicated_axes",
                             [](MlirAttribute self) {
                               return propertyVector<MlirAttribute>(
                                   self,
                                   sdyTensorShardingAttrGetReplicatedAxesSize,
                                   sdyTensorShardingAttrGetReplicatedAxesElem);
                             })
      .def_property_readonly("unreduced_axes", [](MlirAttribute self) {
        return propertyVector<MlirAttribute>(
            self, sdyTensorShardingAttrGetUnreducedAxesSize,
            sdyTensorShardingAttrGetUnreducedAxesElem);
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "TensorShardingPerValueAttr",
      sdyAttributeIsATensorShardingPerValueAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::vector<MlirAttribute> &shardings,
             MlirContext ctx) {
            return cls(sdyTensorShardingPerValueAttrGet(ctx, shardings.size(),
                                                        shardings.data()));
          },
          nb::arg("cls"), nb::arg("shardings"),
          nb::arg("context").none() = nb::none(),
          "Creates a TensorShardingPerValueAttr with the tensor shardings.")
      .def_property_readonly("shardings", [](MlirAttribute self) {
        return propertyVector<MlirAttribute>(
            self, sdyTensorShardingPerValueAttrGetShardingsSize,
            sdyTensorShardingPerValueAttrGetShardingsElem);
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "DimMappingAttr", sdyAttributeIsADimMappingAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::vector<int64_t> &factorIndices,
             MlirContext ctx) {
            return cls(sdyDimMappingAttrGet(ctx, factorIndices.size(),
                                            factorIndices.data()));
          },
          nb::arg("cls"), nb::arg("factor_indices"),
          nb::arg("context").none() = nb::none(),
          "Creates a DimMappingAttr with the factor indices.")
      .def_property_readonly("factor_indices", [](MlirAttribute self) {
        return propertyVector<intptr_t>(self,
                                        sdyDimMappingAttrGetFactorIndicesSize,
                                        sdyDimMappingAttrGetFactorIndicesElem);
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "TensorMappingAttr", sdyAttributeIsATensorMappingAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::vector<MlirAttribute> &mappings,
             MlirContext ctx) {
            return cls(
                sdyTensorMappingAttrGet(ctx, mappings.size(), mappings.data()));
          },
          nb::arg("cls"), nb::arg("dim_mappings"),
          nb::arg("context").none() = nb::none(),
          "Creates a TensorMappingAttr with the dim mappings.")
      .def_property_readonly("dim_mappings",
                             [](MlirAttribute self) {
                               return propertyVector<MlirAttribute>(
                                   self, sdyTensorMappingAttrGetDimMappingsSize,
                                   sdyTensorMappingAttrGetDimMappingsElem);
                             })
      .def_property_readonly("rank", [](MlirAttribute self) {
        return sdyTensorMappingAttrGetRank(self);
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "OpShardingRuleAttr", sdyAttributeIsAOpShardingRuleAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::vector<int64_t> &factorSizes,
             const std::vector<MlirAttribute> &operandMappings,
             const std::vector<MlirAttribute> &resultMappings,
             const std::vector<int64_t> &reductionFactors,
             const std::vector<int64_t> &needReplicationFactors,
             const std::vector<int64_t> &permutationFactors,
             const std::vector<int64_t> &blockedPropagationFactors,
             bool isCustom, MlirContext ctx) {
            return cls(sdyOpShardingRuleAttrGet(
                ctx, factorSizes.size(), factorSizes.data(),
                operandMappings.size(), operandMappings.data(),
                resultMappings.size(), resultMappings.data(),
                reductionFactors.size(), reductionFactors.data(),
                needReplicationFactors.size(), needReplicationFactors.data(),
                permutationFactors.size(), permutationFactors.data(),
                blockedPropagationFactors.size(),
                blockedPropagationFactors.data(), isCustom));
          },
          nb::arg("cls"), nb::arg("factor_sizes"), nb::arg("operand_mappings"),
          nb::arg("result_mappings"),
          nb::arg("reduction_factors") = std::vector<int64_t>(),
          nb::arg("need_replication_factors") = std::vector<int64_t>(),
          nb::arg("permutation_factors") = std::vector<int64_t>(),
          nb::arg("blocked_propagation_factors") = std::vector<int64_t>(),
          nb::arg("is_custom") = false, nb::arg("context").none() = nb::none(),
          "Creates a OpShardingRuleAttr with the factor sizes and mappings for "
          "operands and results.")
      .def_property_readonly("is_custom",
                             [](MlirAttribute self) {
                               return sdyOpShardingRuleAttrGetIsCustom(self);
                             })
      .def_property_readonly("factor_sizes",
                             [](MlirAttribute self) {
                               return propertyVector<intptr_t>(
                                   self,
                                   sdyOpShardingRuleAttrGetFactorSizesSize,
                                   sdyOpShardingRuleAttrGetFactorSizesElem);
                             })
      .def_property_readonly("operand_mappings",
                             [](MlirAttribute self) {
                               return propertyVector<MlirAttribute>(
                                   self,
                                   sdyOpShardingRuleAttrGetOperandMappingsSize,
                                   sdyOpShardingRuleAttrGetOperandMappingsElem);
                             })
      .def_property_readonly("result_mappings",
                             [](MlirAttribute self) {
                               return propertyVector<MlirAttribute>(
                                   self,
                                   sdyOpShardingRuleAttrGetResultMappingsSize,
                                   sdyOpShardingRuleAttrGetResultMappingsElem);
                             })
      .def_property_readonly(
          "reduction_factors",
          [](MlirAttribute self) {
            return propertyVector<intptr_t>(
                self, sdyOpShardingRuleAttrGetReductionFactorsSize,
                sdyOpShardingRuleAttrGetReductionFactorsElem);
          })
      .def_property_readonly(
          "need_replication_factors",
          [](MlirAttribute self) {
            return propertyVector<intptr_t>(
                self, sdyOpShardingRuleAttrGetNeedReplicationFactorsSize,
                sdyOpShardingRuleAttrGetNeedReplicationFactorsElem);
          })
      .def_property_readonly(
          "permutation_factors",
          [](MlirAttribute self) {
            return propertyVector<intptr_t>(
                self, sdyOpShardingRuleAttrGetPermutationFactorsSize,
                sdyOpShardingRuleAttrGetPermutationFactorsElem);
          })
      .def_property_readonly(
          "blocked_propagation_factors", [](MlirAttribute self) {
            return propertyVector<intptr_t>(
                self, sdyOpShardingRuleAttrGetBlockedPropagationFactorsSize,
                sdyOpShardingRuleAttrGetBlockedPropagationFactorsElem);
          });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "ManualAxesAttr", sdyAttributeIsAManualAxesAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::vector<MlirAttribute> &meshAxes,
             MlirContext ctx) {
            return cls(
                sdyManualAxesAttrGet(ctx, meshAxes.size(), meshAxes.data()));
          },
          nb::arg("cls"), nb::arg("manual_axes"),
          nb::arg("context").none() = nb::none(),
          "Creates a ManualAxesAttr with the given manual axes.")
      .def("__getitem__",
           [](MlirAttribute &self, unsigned index) {
             if (index >= sdyManualAxesAttrGetAxesSize(self)) {
               throw nb::index_error();
             }
             return toPyString(sdyManualAxesAttrGetAxesElem(self, index));
           })
      .def("__len__", [](MlirAttribute &self) {
        return sdyManualAxesAttrGetAxesSize(self);
      });
}

NB_MODULE(_sdy, m) { populateSdyModule(m); }

} // namespace mlir::ttmlir::python
