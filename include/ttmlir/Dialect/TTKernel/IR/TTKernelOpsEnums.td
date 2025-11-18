// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELOPSENUMS_TD
#define TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELOPSENUMS_TD

include "mlir/IR/EnumAttr.td"

def TTKernel_ThreadTypeNoc : I32EnumAttrCase<"Noc", 0, "noc">;
def TTKernel_ThreadTypeCompute : I32EnumAttrCase<"Compute", 1, "compute">;
def TTKernel_ThreadTypeEthernet : I32EnumAttrCase<"Ethernet", 2, "ethernet">;

def TTKernel_ThreadType : I32EnumAttr<"ThreadType", "TTKernel ThreadTypes",
                           [
                            TTKernel_ThreadTypeNoc,
                            TTKernel_ThreadTypeCompute,
                            TTKernel_ThreadTypeEthernet,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttkernel";
}

def TTKernel_ReduceSum : I32EnumAttrCase<"Sum", 0, "reduce_sum">;
def TTKernel_ReduceMax : I32EnumAttrCase<"Max", 1, "reduce_max">;

def TTKernel_ReduceType : I32EnumAttr<"ReduceType", "TTKernel Reduce Types",
                          [
                            TTKernel_ReduceSum,
                            TTKernel_ReduceMax
                          ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttkernel";
}

def TTKernel_ReduceDimRow : I32EnumAttrCase<"Row", 0, "reduce_dim_row">;
def TTKernel_ReduceDimCol : I32EnumAttrCase<"Col", 1, "reduce_dim_col">;
def TTKernel_ReduceDimScalar
    : I32EnumAttrCase<"Scalar", 2, "reduce_dim_scalar">;

def TTKernel_ReduceDim
    : I32EnumAttr<"ReduceDim", "TTKernel Reduce Dimensions",
                  [TTKernel_ReduceDimRow, TTKernel_ReduceDimCol,
                   TTKernel_ReduceDimScalar]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttkernel";
}

def TTKernel_BcastTypeNone :   I32EnumAttrCase<"None",   0, "none">;
def TTKernel_BcastTypeCol :    I32EnumAttrCase<"Col",    1, "col">;
def TTKernel_BcastTypeRow :    I32EnumAttrCase<"Row",    2, "row">;
def TTKernel_BcastTypeScalar : I32EnumAttrCase<"Scalar", 3, "scalar">;

def TTKernel_BcastType : I32EnumAttr<"BcastType", "TTKernel Broadcast Types",
                         [
                           TTKernel_BcastTypeNone,
                           TTKernel_BcastTypeCol,
                           TTKernel_BcastTypeRow,
                           TTKernel_BcastTypeScalar
                         ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttkernel";
}

def TTKernel_ArgTypeCBPort : I32EnumAttrCase<"CBPort", 0, "cb_port">;
def TTKernel_ArgTypeBufferAddress : I32EnumAttrCase<"BufferAddress", 1, "buffer_address">;
def TTKernel_ArgTypeSemaphore : I32EnumAttrCase<"Semaphore", 2, "semaphore">;

def TTKernel_ArgType : I32EnumAttr<"ArgType", "TTKernel Argument Types",
                         [
                           TTKernel_ArgTypeCBPort,
                           TTKernel_ArgTypeBufferAddress,
                           TTKernel_ArgTypeSemaphore
                         ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttkernel";
}

#endif
