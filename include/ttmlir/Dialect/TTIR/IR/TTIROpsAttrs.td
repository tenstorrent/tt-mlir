// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTIR_ATTRS_TD
#define TTMLIR_TTIR_ATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsEnums.td"
include "ttmlir/Dialect/TTIR/IR/TTIRBase.td"

def TTIR_PoolingMethodAttr : EnumAttr<TTIR_Dialect, TTIR_PoolingMethod, "pooling_method">;

def TTIR_ReduceDimAttr : EnumAttr<TTIR_Dialect, TTIR_ReduceDim, "reduce_dim">;

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

def TTIR_FlattenedCompatInfoAttr : AttrDef<TTIR_Dialect, "FlattenedCompatInfo", [], "::mlir::Attribute"> {
  let mnemonic = "flattened_compat";
  let summary = "Information for sliding window operations with tensors flattened to (1, 1, N*H*W, C)";
  let description = [{
    This attribute marks operations that are compatible with flattened tensors.
    It is used as a marker and doesn't carry any additional data.
  }];


  let parameters = (ins
    "int64_t":$batchSize,
    "int64_t":$inputHeight,
    "int64_t":$inputWidth
  );

  let assemblyFormat = [{
        `batch_size` `=` $batchSize `,`
        `input_height` `=` $inputHeight `,`
        `input_width` `=` $inputWidth
  }];

  let constBuilderCall = "$0";
}

def TTIR_HoistedCallAttr : AttrDef<TTIR_Dialect, "HoistedCall", [], "::mlir::Attribute"> {
  let mnemonic = "hoisted_call";
  let summary = "Indicates that this call operation has been hoisted";
  let description = [{
    This attribute marks call operations that have been hoisted during optimization.
    It is used as a marker and doesn't carry any additional data.
  }];
}

// Unit attribute for should_hoist flag
def TTIR_ShouldHoistAttr : AttrDef<TTIR_Dialect, "ShouldHoist", [], "::mlir::Attribute"> {
  let mnemonic = "should_hoist";
  let summary = "Indicates that this function should be hoisted";
  let description = [{
    This attribute marks functions that should be hoisted during optimization.
    It is used as a marker and doesn't carry any additional data.
  }];
}

// Integer attribute for return mapping
def TTIR_ReturnToOutputMappingAttr : AttrDef<TTIR_Dialect, "ReturnToOutputMapping", [], "::mlir::Attribute"> {
  let mnemonic = "return_to_output_mapping";
  let summary = "Maps return values to output indices";
  let description = [{
    This attribute stores an integer mapping value that indicates how
    return values correspond to output indices.
  }];

  let parameters = (ins
    "int64_t":$value
  );

  let assemblyFormat = [{ `<` $value `>` }];

  let extraClassDeclaration = [{
    // Convenience getter that matches IntegerAttr interface
    int64_t getInt() const { return getValue(); }
  }];
}

def TTIR_ThreadAttr : AttrDef<TTIR_Dialect, "Thread", [], "::mlir::Attribute"> {
  let mnemonic = "thread";
  let summary = "Thread information for a generic op.";
  let description = [{
    Holds the thread information corresponding to a single generic op region.
  }];
  let parameters = (ins
    "ThreadType":$threadType,
    OptionalParameter<"SymbolRefAttr">:$kernelSymbol
  );

  let assemblyFormat = [{ `<` $threadType (`,` $kernelSymbol^)? `>` }];

  let extraClassDeclaration = [{
      static ThreadAttr get(::mlir::MLIRContext *context, ThreadType threadType) {
        return get(context, threadType, nullptr);
      }
  }];
}

def TTIR_ThreadArrayAttr : TypedArrayAttrBase<TTIR_ThreadAttr, "">;

#endif // TTMLIR_TTIR_ATTRS_TD
