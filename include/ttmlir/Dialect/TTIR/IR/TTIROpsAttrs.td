// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTIR_ATTRS_TD
#define TTMLIR_TTIR_ATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "ttmlir/Dialect/TTIR/IR/TTIRBase.td"

def TTIR_ConvolutionLayoutAttr : AttrDef<TTIR_Dialect, "ConvolutionLayout", [], "::mlir::Attribute"> {
  let mnemonic = "convolution_layout";
  let summary = "Structure of dimension information for convolution op";
  let description = [{
    Holds the layout information for the input activation, weights, and output.
  }];
  let parameters = (ins
    "int64_t":$inputBatchDimension,
    "int64_t":$inputFeatureDimension,
    ArrayRefParameter<"int64_t">:$inputSpatialDimensions,

    "int64_t":$kernelOutputFeatureDimension,
    "int64_t":$kernelInputFeatureDimension,
    ArrayRefParameter<"int64_t">:$kernelSpatialDimensions,

    "int64_t":$outputBatchDimension,
    "int64_t":$outputFeatureDimension,
    ArrayRefParameter<"int64_t">:$outputSpatialDimensions
  );

  let assemblyFormat = [{
        `input_batch` `=` $inputBatchDimension `,`
        `input_feature` `=` $inputFeatureDimension`,`
        `input_spatial_dimensions` `=` custom<DimensionList>($inputSpatialDimensions) `,`
        `kernel_output_feature` `=` $kernelOutputFeatureDimension `,`
        `kernel_input_feature` `=` $kernelInputFeatureDimension `,`
        `kernel_spatial_dimensions` `=` custom<DimensionList>($kernelSpatialDimensions) `,`
        `output_batch` `=` $outputBatchDimension `,`
        `output_feature` `=` $outputFeatureDimension `,`
        `output_spatial_dimensions` `=` custom<DimensionList>($outputSpatialDimensions)
  }];
}

#endif // TTMLIR_TTIR_ATTRS_TD
