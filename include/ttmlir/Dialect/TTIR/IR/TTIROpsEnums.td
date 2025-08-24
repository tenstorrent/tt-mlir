// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTIR_ENUMS_TD
#define TTMLIR_TTIR_ENUMS_TD

include "mlir/IR/EnumAttr.td"

def TTIR_PoolingMethodAverage : I32EnumAttrCase<"Average", 0>;
def TTIR_PoolingMethodMax : I32EnumAttrCase<"Max", 1>;
def TTIR_PoolingMethodSum : I32EnumAttrCase<"Sum", 2>;

def TTIR_PoolingMethod : I32EnumAttr<"PoolingMethod", "TTIR PoolingMethod", [
                                      TTIR_PoolingMethodAverage,
                                      TTIR_PoolingMethodMax,
                                      TTIR_PoolingMethodSum
                                    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttir";
}

def TTIR_ReduceDimRow : I32EnumAttrCase<"R", 0>;
def TTIR_ReduceDimCol : I32EnumAttrCase<"C", 1>;
def TTIR_ReduceDimRowCol : I32EnumAttrCase<"RC", 2>;

def TTIR_ReduceDim : I32EnumAttr<"ReduceDim", "TTIR ReduceDim", [
                                  TTIR_ReduceDimRow,
                                  TTIR_ReduceDimCol,
                                  TTIR_ReduceDimRowCol
                                ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttir";
}

def TTIR_ThreadTypeCompute : I32EnumAttrCase<"Compute", 0, "compute">;
def TTIR_ThreadTypeDatamovement : I32EnumAttrCase<"Datamovement", 1, "datamovement">;

def TTIR_ThreadType : I32EnumAttr<"ThreadType", "TTIR ThreadType", [
                                  TTIR_ThreadTypeCompute,
                                  TTIR_ThreadTypeDatamovement,
                                ]> {
  let genSpecializedAttr = 1;
  let cppNamespace = "::mlir::tt::ttir";
}

#endif
