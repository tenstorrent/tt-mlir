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

// D2M ThreadType mirrors D2M thread type and is used by D2M_ThreadAttr
def D2M_ThreadTypeCompute : I32EnumAttrCase<"Compute", 0, "compute">;
def D2M_ThreadTypeDatamovement : I32EnumAttrCase<"Datamovement", 1, "datamovement">;

def D2M_ThreadType : I32EnumAttr<"ThreadType", "D2M ThreadType", [
                                  D2M_ThreadTypeCompute,
                                  D2M_ThreadTypeDatamovement,
                                ]> {
  let genSpecializedAttr = 1;
  let cppNamespace = "::mlir::tt::d2m";
}

#endif // TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSENUMS_TD
