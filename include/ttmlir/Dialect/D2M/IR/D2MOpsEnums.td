// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSENUMS_TD
#define TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSENUMS_TD

include "mlir/IR/EnumAttr.td"

// D2M ReduceDim mirrors D2M reduce dim but lives in d2m namespace
def D2M_ReduceDimRow : I32EnumAttrCase<"R", 0>;
def D2M_ReduceDimCol : I32EnumAttrCase<"C", 1>;
def D2M_ReduceDimRowCol : I32EnumAttrCase<"RC", 2>;

def D2M_ReduceDim : I32EnumAttr<"ReduceDim", "D2M ReduceDim", [
  D2M_ReduceDimRow,
  D2M_ReduceDimCol,
  D2M_ReduceDimRowCol,
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::d2m";
}

def D2M_TileBcastTypeNone   : I32EnumAttrCase<"None",   0, "none">;
def D2M_TileBcastTypeCol    : I32EnumAttrCase<"Col",    1, "col">;
def D2M_TileBcastTypeRow    : I32EnumAttrCase<"Row",    2, "row">;
def D2M_TileBcastTypeScalar : I32EnumAttrCase<"Scalar", 3, "scalar">;

def D2M_TileBcastType : I32EnumAttr<"TileBcastType", "D2M TileBcastType", [
  D2M_TileBcastTypeNone,
  D2M_TileBcastTypeCol,
  D2M_TileBcastTypeRow,
  D2M_TileBcastTypeScalar
]> {
  let summary = "D2M tile-level broadcast types.";
  let description = [{
    The type of the input shape (not the direction) of a unary tile-level
    broadcast operation.

    For type Col, the input should be a single tile with a filled 0-column and
    zeros elsewhere.

    For type Row, the input should be a single tile with a filled 0-row and
    zeros elsewhere.

    For type Scalar, the input should be a single tile with a filled single
    value at location [0,0], and zeros elsewhere.

    The output will be a full tile filled with the broadcasted input values.
  }];

  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::d2m";
}

// D2M ThreadType mirrors D2M thread type and is used by D2M_ThreadAttr
def D2M_ThreadTypeCompute : I32EnumAttrCase<"Compute", 0, "compute">;
def D2M_ThreadTypeDatamovement : I32EnumAttrCase<"Datamovement", 1, "datamovement">;
def D2M_ThreadTypeUnified : I32EnumAttrCase<"Unified", 2, "unified">;

def D2M_ThreadType : I32EnumAttr<"ThreadType", "D2M ThreadType", [
                                  D2M_ThreadTypeCompute,
                                  D2M_ThreadTypeDatamovement,
                                  D2M_ThreadTypeUnified,
                                ]> {
  let genSpecializedAttr = 1;
  let cppNamespace = "::mlir::tt::d2m";
}

#endif // TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSENUMS_TD
