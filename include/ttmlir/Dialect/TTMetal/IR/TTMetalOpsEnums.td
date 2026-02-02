// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPSENUMS_TD
#define TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALOPSENUMS_TD

include "mlir/IR/EnumAttr.td"

def TTMetal_MathFidelityLoFi : I32EnumAttrCase<"LoFi", 0, "lofi">;
def TTMetal_MathFidelityHiFi2 : I32EnumAttrCase<"HiFi2", 2, "hifi2">;
def TTMetal_MathFidelityHiFi3 : I32EnumAttrCase<"HiFi3", 3, "hifi3">;
def TTMetal_MathFidelityHiFi4 : I32EnumAttrCase<"HiFi4", 4, "hifi4">;

def TTMetal_MathFidelity : I32EnumAttr<"MathFidelity", "TT MathFidelity",
                           [
                            TTMetal_MathFidelityLoFi,
                            TTMetal_MathFidelityHiFi2,
                            TTMetal_MathFidelityHiFi3,
                            TTMetal_MathFidelityHiFi4,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttmetal";
}

def TTMetal_UnpackToDestFp32 : I32EnumAttrCase<"Fp32", 0, "fp32">;
def TTMetal_UnpackToDestDefault : I32EnumAttrCase<"Default", 1, "default">;

def TTMetal_UnpackToDestMode : I32EnumAttr<"UnpackToDestMode", "TT UnpackToDestMode",
                           [
                            TTMetal_UnpackToDestFp32,
                            TTMetal_UnpackToDestDefault,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttmetal";
}


def TTMetal_NocIndexNoc0 : I32EnumAttrCase<"Noc0", 0, "noc0">;
def TTMetal_NocIndexNoc1 : I32EnumAttrCase<"Noc1", 1, "noc1">;

def TTMetal_NocIndex : I32EnumAttr<"NocIndex", "TT NocIndex",
                           [
                            TTMetal_NocIndexNoc0,
                            TTMetal_NocIndexNoc1,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttmetal";
}

def TTMetal_Sender : I32EnumAttrCase<"Sender", 0, "sender">;
def TTMetal_Receiver : I32EnumAttrCase<"Receiver", 1, "receiver">;

def TTMetal_EthType : I32EnumAttr<"EthType", "TT EthType",
                           [
                            TTMetal_Sender,
                            TTMetal_Receiver,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttmetal";
}

// Routing mode for fabric connections. This is used to decide how we want to route
// in the given topology, as well as which links a core should connect to.
// In line/mesh topologies, we need to connect to all links to ensure reachability, and there is
// only one way to route to get to the given device.
// In ring/torus topologies, we can split the two directions along each axis and
// connect each core to only a single direction per axis which ensures reachability.
// The core will then only route in the selected direction (e.g. always routes East
// to destination on a ring) which allows for better link saturation.
// See usages of RoutingMode and RoutingDirection in the LLKs for implementation details.
// (include/ttmlir/Target/TTKernel/LLKs/experimental_fabric_1d_routing.h and
// include/ttmlir/Target/TTKernel/LLKs/experimental_fabric_2d_routing.h)
def TTMetal_RoutingModeBidirLineMesh : I32EnumAttrCase<"BidirLineMesh", 0, "bidir_line_mesh">;
def TTMetal_RoutingModeUnidirRingTorus : I32EnumAttrCase<"UnidirRingTorus", 1, "unidir_ring_torus">;

def TTMetal_RoutingMode : I32EnumAttr<"RoutingMode", "TT RoutingMode",
                           [
                            TTMetal_RoutingModeBidirLineMesh,
                            TTMetal_RoutingModeUnidirRingTorus,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttmetal";
}

#endif
