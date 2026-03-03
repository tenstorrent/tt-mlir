// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/AffineMap.h"

#include <cstdint>
#include <vector>

namespace mlir::ttmlir::python {
void populateTTIRModule(nb::module_ &m) {
  tt_attribute_class<tt::ttir::ConvolutionLayoutAttr>(m,
                                                      "ConvolutionLayoutAttr")
      .def_static(
          "get",
          [](MlirContext ctx, int64_t input_batch, int64_t input_feature,
             std::vector<int64_t> input_spatial_dimensions,
             int64_t kernel_output_feature, int64_t kernel_input_feature,
             std::vector<int64_t> kernel_spatial_dimensions,
             int64_t output_batch, int64_t output_feature,
             std::vector<int64_t> output_spatial_dimensions) {
            return wrap(tt::ttir::ConvolutionLayoutAttr::get(
                unwrap(ctx), input_batch, input_feature,
                input_spatial_dimensions, kernel_output_feature,
                kernel_input_feature, kernel_spatial_dimensions, output_batch,
                output_feature, output_spatial_dimensions));
          })
      .def_prop_ro("input_batch",
                   &tt::ttir::ConvolutionLayoutAttr::getInputBatchDimension)
      .def_prop_ro("input_feature",
                   &tt::ttir::ConvolutionLayoutAttr::getInputFeatureDimension)
      .def_prop_ro("input_spatial_dimensions",
                   [](const tt::ttir::ConvolutionLayoutAttr &self) {
                     auto inputSpatialDimensions =
                         self.getInputSpatialDimensions();
                     return std::vector<int64_t>(inputSpatialDimensions.begin(),
                                                 inputSpatialDimensions.end());
                   })
      .def_prop_ro(
          "kernel_output_feature",
          &tt::ttir::ConvolutionLayoutAttr::getKernelOutputFeatureDimension)
      .def_prop_ro(
          "kernel_input_feature",
          &tt::ttir::ConvolutionLayoutAttr::getKernelInputFeatureDimension)
      .def_prop_ro(
          "kernel_spatial_dimensions",
          [](const tt::ttir::ConvolutionLayoutAttr &self) {
            auto kernelSpatialDimensions = self.getKernelSpatialDimensions();
            return std::vector<int64_t>(kernelSpatialDimensions.begin(),
                                        kernelSpatialDimensions.end());
          })
      .def_prop_ro("output_batch",
                   &tt::ttir::ConvolutionLayoutAttr::getOutputBatchDimension)
      .def_prop_ro("output_feature",
                   &tt::ttir::ConvolutionLayoutAttr::getOutputFeatureDimension)
      .def_prop_ro(
          "output_spatial_dimensions",
          [](const tt::ttir::ConvolutionLayoutAttr &self) {
            auto outputSpatialDimensions = self.getOutputSpatialDimensions();
            return std::vector<int64_t>(outputSpatialDimensions.begin(),
                                        outputSpatialDimensions.end());
          });
  m.def(
      "rearrange_inv_pattern_map",
      [](MlirContext context, std::string pattern, std::vector<int64_t> shape) {
        mlir::FailureOr<AffineMap> failureOrMap =
            tt::ttir::RearrangeOp::getInvPatternMap(unwrap(context), pattern,
                                                    shape);
        if (failed(failureOrMap)) {
          throw std::runtime_error(
              "rearrange_pattern_affine_map: failed to parse pattern \"" +
              pattern + "\".");
        }
        return wrap(*failureOrMap);
      });

  m.def("affine_map_compose",
        [](MlirAffineMap map, std::vector<int64_t> index) {
          auto sample = unwrap(map).compose(index);
          return std::vector<int64_t>(sample.begin(), sample.end());
        });
}
} // namespace mlir::ttmlir::python
