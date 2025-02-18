// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPSINTERFACES_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPSINTERFACES_TD

include "mlir/IR/OpBase.td"
include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"

def TTIROpInterface : OpInterface<"TTIROp"> {
  let cppNamespace = "::mlir::tt::ttir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Get the device of the current scope.
      }],
      /*retTy=*/"::mlir::tt::SystemDescAttr",
      /*methodName=*/"getSystemDesc",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::tt::getCurrentScopeSystemDesc($_op);"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the device of the current scope.
      }],
      /*retTy=*/"::mlir::tt::DeviceAttr",
      /*methodName=*/"getDevice",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::tt::getCurrentScopeDevice($_op);"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether op can be run on CPU.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"supportsCPUExecution",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return false;"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether op must be run on CPU.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"supportsOnlyCPUExecution",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return false;"
    >,
  ];
}

def TTIR_Broadcastable : OpInterface<"Broadcastable"> {
  let cppNamespace = "::mlir::tt::ttir";

  let dependentTraits = [AttrSizedOperandSegments];

  let verify = [{
    return detail::verifyBroadcastable($_op);
  }];
}

def TTIR_GenericRegionOpInterface : OpInterface<"GenericRegionOp"> {
  let cppNamespace = "::mlir::tt::ttir";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return a pair of indexingMaps and iteratorTypes for the given GenericRegionOp.
        Where:
        indexingMaps: a list of AffineMapAttr, one AffineMapAttr per each input and
                      output view. Such AffineMapAttr specifies the mapping between
                      the loops and the indexing within each view. It effectively
                      defines how the op can legally be parallelized.
        iteratorTypes: an ArrayAttr specifying the type of the enclosing loops. Each
                       element of the list represents an iterator of one of the
                       following types, parallel and reduction.
      }],
      /*retTy=*/"std::pair<::mlir::ArrayAttr, ::mlir::ArrayAttr>",
      /*methodName=*/"getIndexingMaps",
      /*args=*/(ins "::mlir::Builder &":$builder),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
    >,
    InterfaceMethod<
      /*desc=*/[{
        Rewrite self into the generic region block using the arith and math dialects.
      }],
      /*retTy=*/"void",
      /*methodName=*/"buildGenericRegion",
      /*args=*/(ins "::mlir::OpBuilder &":$op_builder,
                         "::mlir::Block*":$block),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
    >,
    StaticInterfaceMethod<
      /*desc=*/[{
        Return if the given operands have the same rank.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"sameRank",
      /*args=*/(ins "::mlir::OperandRange":$operands),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::all_equal(llvm::map_range(operands, [](Value operand) {
          return mlir::cast<RankedTensorType>(operand.getType()).getRank();
        }));
      }]
    >
  ];
}

def TTIR_PartiallyBroadcastable : OpInterface<"PartiallyBroadcastable"> {
  // Supports implicit broadcast for first operand only.
  let cppNamespace = "::mlir::tt::ttir";
}

def TTIR_FullyBroadcastable : OpInterface<"FullyBroadcastable"> {
  // Supports implicit broadcast for all the operands.
  let cppNamespace = "::mlir::tt::ttir";
}

def TTIR_GenericParent : OpInterface<"GenericParent"> {
  let cppNamespace = "::mlir::tt::ttir";

  let verify = [{
    return detail::verifyGenericParent($_op);
  }];
}

#endif
