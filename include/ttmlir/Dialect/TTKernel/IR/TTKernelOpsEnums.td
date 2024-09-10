// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELOPSENUMS_TD
#define TTMLIR_TTMLIR_DIALECT_TTKERNEL_TTKERNELOPSENUMS_TD

include "mlir/IR/EnumAttr.td"

def TTKernel_ThreadTypeNoc : I32EnumAttrCase<"Noc", 0, "noc">;
def TTKernel_ThreadTypeTensix : I32EnumAttrCase<"Tensix", 1, "tensix">;
def TTKernel_ThreadTypeEthernet : I32EnumAttrCase<"Ethernet", 2, "ethernet">;

def TTKernel_ThreadType : I32EnumAttr<"ThreadType", "TTKernel ThreadTypes",
                           [
                            TTKernel_ThreadTypeNoc,
                            TTKernel_ThreadTypeTensix,
                            TTKernel_ThreadTypeEthernet,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttkernel";
}

def TTKernel_MathFidelityLoFi : I32EnumAttrCase<"LoFi", 0, "lofi">;
def TTKernel_MathFidelityHiFi2 : I32EnumAttrCase<"HiFi2", 2, "hifi2">;
def TTKernel_MathFidelityHiFi3 : I32EnumAttrCase<"HiFi3", 3, "hifi3">;
def TTKernel_MathFidelityHiFi4 : I32EnumAttrCase<"HiFi4", 4, "hifi4">;

def TTKernel_MathFidelity : I32EnumAttr<"MathFidelity", "TT MathFidelity",
                           [
                            TTKernel_MathFidelityLoFi,
                            TTKernel_MathFidelityHiFi2,
                            TTKernel_MathFidelityHiFi3,
                            TTKernel_MathFidelityHiFi4,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttkernel";
}

def TTKernel_NocIndexNoc0 : I32EnumAttrCase<"Noc0", 0, "noc0">;
def TTKernel_NocIndexNoc1 : I32EnumAttrCase<"Noc1", 1, "noc1">;

def TTKernel_NocIndex : I32EnumAttr<"NocIndex", "TT NocIndex",
                           [
                            TTKernel_NocIndexNoc0,
                            TTKernel_NocIndexNoc1,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttkernel";
}

def TTKernel_Sender : I32EnumAttrCase<"Sender", 0, "sender">;
def TTKernel_Receiver : I32EnumAttrCase<"Receiver", 1, "receiver">;

def TTKernel_EthType : I32EnumAttr<"EthType", "TT EthType",
                           [
                            TTKernel_Sender,
                            TTKernel_Receiver,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttkernel";
}

def TTKernel_CBPortIn0       : I32EnumAttrCase<"In0",        0, "cb_in0">;
def TTKernel_CBPortIn1       : I32EnumAttrCase<"In1",        1, "cb_in1">;
def TTKernel_CBPortIn2       : I32EnumAttrCase<"In2",        2, "cb_in2">;
def TTKernel_CBPortIn3       : I32EnumAttrCase<"In3",        3, "cb_in3">;
def TTKernel_CBPortIn4       : I32EnumAttrCase<"In4",        4, "cb_in4">;
def TTKernel_CBPortIn5       : I32EnumAttrCase<"In5",        5, "cb_in5">;
def TTKernel_CBPortIn6       : I32EnumAttrCase<"In6",        6, "cb_in6">;
def TTKernel_CBPortIn7       : I32EnumAttrCase<"In7",        7, "cb_in7">;
def TTKernel_CBPortDataFlow0 : I32EnumAttrCase<"DataFlow0",  8, "cb_dfl0">;
def TTKernel_CBPortDataFlow1 : I32EnumAttrCase<"DataFlow1",  9, "cb_dfl1">;
def TTKernel_CBPortDataFlow2 : I32EnumAttrCase<"DataFlow2", 10, "cb_dfl2">;
def TTKernel_CBPortDataFlow3 : I32EnumAttrCase<"DataFlow3", 11, "cb_dfl3">;
def TTKernel_CBPortDataFlow4 : I32EnumAttrCase<"DataFlow4", 12, "cb_dfl4">;
def TTKernel_CBPortDataFlow5 : I32EnumAttrCase<"DataFlow5", 13, "cb_dfl5">;
def TTKernel_CBPortDataFlow6 : I32EnumAttrCase<"DataFlow6", 14, "cb_dfl6">;
def TTKernel_CBPortDataFlow7 : I32EnumAttrCase<"DataFlow7", 15, "cb_dfl7">;
def TTKernel_CBPortOut0      : I32EnumAttrCase<"Out0",      16, "cb_out0">;
def TTKernel_CBPortOut1      : I32EnumAttrCase<"Out1",      17, "cb_out1">;
def TTKernel_CBPortOut2      : I32EnumAttrCase<"Out2",      18, "cb_out2">;
def TTKernel_CBPortOut3      : I32EnumAttrCase<"Out3",      19, "cb_out3">;
def TTKernel_CBPortOut4      : I32EnumAttrCase<"Out4",      20, "cb_out4">;
def TTKernel_CBPortOut5      : I32EnumAttrCase<"Out5",      21, "cb_out5">;
def TTKernel_CBPortOut6      : I32EnumAttrCase<"Out6",      22, "cb_out6">;
def TTKernel_CBPortOut7      : I32EnumAttrCase<"Out7",      23, "cb_out7">;
def TTKernel_CBPortIntermed0 : I32EnumAttrCase<"Intermed0", 24, "cb_int0">;
def TTKernel_CBPortIntermed1 : I32EnumAttrCase<"Intermed1", 25, "cb_int1">;
def TTKernel_CBPortIntermed2 : I32EnumAttrCase<"Intermed2", 26, "cb_int2">;
def TTKernel_CBPortIntermed3 : I32EnumAttrCase<"Intermed3", 27, "cb_int3">;
def TTKernel_CBPortIntermed4 : I32EnumAttrCase<"Intermed4", 28, "cb_int4">;
def TTKernel_CBPortIntermed5 : I32EnumAttrCase<"Intermed5", 29, "cb_int5">;
def TTKernel_CBPortIntermed6 : I32EnumAttrCase<"Intermed6", 30, "cb_int6">;
def TTKernel_CBPortIntermed7 : I32EnumAttrCase<"Intermed7", 31, "cb_int7">;

def TTKernel_CBPort : I32EnumAttr<"CBPort", "TTKernel Circular Buffer Ports",
                           [
                            TTKernel_CBPortIn0,
                            TTKernel_CBPortIn1,
                            TTKernel_CBPortIn2,
                            TTKernel_CBPortIn3,
                            TTKernel_CBPortIn4,
                            TTKernel_CBPortIn5,
                            TTKernel_CBPortIn6,
                            TTKernel_CBPortIn7,
                            TTKernel_CBPortDataFlow0,
                            TTKernel_CBPortDataFlow1,
                            TTKernel_CBPortDataFlow2,
                            TTKernel_CBPortDataFlow3,
                            TTKernel_CBPortDataFlow4,
                            TTKernel_CBPortDataFlow5,
                            TTKernel_CBPortDataFlow6,
                            TTKernel_CBPortDataFlow7,
                            TTKernel_CBPortOut0,
                            TTKernel_CBPortOut1,
                            TTKernel_CBPortOut2,
                            TTKernel_CBPortOut3,
                            TTKernel_CBPortOut4,
                            TTKernel_CBPortOut5,
                            TTKernel_CBPortOut6,
                            TTKernel_CBPortOut7,
                            TTKernel_CBPortIntermed0,
                            TTKernel_CBPortIntermed1,
                            TTKernel_CBPortIntermed2,
                            TTKernel_CBPortIntermed3,
                            TTKernel_CBPortIntermed4,
                            TTKernel_CBPortIntermed5,
                            TTKernel_CBPortIntermed6,
                            TTKernel_CBPortIntermed7,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttkernel";
}

#endif
