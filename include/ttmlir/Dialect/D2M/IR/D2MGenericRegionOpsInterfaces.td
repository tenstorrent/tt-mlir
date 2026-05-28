// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_D2M_DIALECT_D2M_IR_D2MGENERICREGIONOPSINTERFACES_TD
#define TTMLIR_D2M_DIALECT_D2M_IR_D2MGENERICREGIONOPSINTERFACES_TD

include "mlir/IR/OpBase.td"
include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"

def D2M_SynchronizableOpInterface : OpInterface<"SynchronizableOpInterface"> {
  let description = [{
     Define a base interface for synchronizable ops in the generic region.
  }];

  let cppNamespace = "::mlir::tt::d2m";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{Returns true if the given operand is a producer (i.e. written
                 to) for synchronization purposes.}],
      /*retTy=*/"bool",
      /*methodName=*/"isProducer",
      /*args=*/(ins "mlir::OpOperand &":$operand),
      /*methodBody=*/""
    >,
    InterfaceMethod<
      /*desc=*/[{Returns true if the given operand is a consumer (i.e. read
                 from) for synchronization purposes.}],
      /*retTy=*/"bool",
      /*methodName=*/"isConsumer",
      /*args=*/(ins "mlir::OpOperand &":$operand),
      /*methodBody=*/""
    >,
  ];

}

// Mirror D2M DMA interface into d2m namespace for d2m ops

def D2M_DMAOpInterface : OpInterface<"DMAOpInterface"> {
  let cppNamespace = "::mlir::tt::d2m";
  let methods = [
    InterfaceMethod<[{}], "::mlir::TypedValue<::mlir::Type>", "getSrc">,
    InterfaceMethod<[{}], "::mlir::TypedValue<::mlir::Type>", "getDst">,
    InterfaceMethod<[{}], "::mlir::OpOperand&", "getSrcMutable">,
    InterfaceMethod<[{}], "::mlir::OpOperand&", "getDstMutable">,
    InterfaceMethod<[{}], "::mlir::Location", "getLoc">,
    InterfaceMethod<[{}], "::mlir::TypedValue<::mlir::Type>", "getResult">,
  ];
}

// Interface for CB-based DMA ops that run on the DM thread (RemoteLoadOp,
// RemoteStoreOp, LocalCopyOp).
def D2M_ShardDMAOpInterface : OpInterface<"ShardDMAOpInterface"> {
  let cppNamespace = "::mlir::tt::d2m";
  let description = [{
    Implemented by the shard-level DMA ops (RemoteLoadOp, RemoteStoreOp,
    LocalCopyOp) that run on the datamovement thread before lowering to fully
    indexed form.  Provides a uniform getCBPort() accessor so scheduling and
    thread-splitting passes do not need to enumerate individual op types.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the CB port number for this DMA operation. "
               "The op must be in explicit CB form (getCb() != nullptr). ",
      /*retTy=*/"unsigned",
      /*methodName=*/"getCBPort"
    >,
  ];
}

// Parent interface used to verify D2M region ops are nested in d2m.generic or func.func

def D2M_GenericParent : OpInterface<"GenericParent"> {
  let cppNamespace = "::mlir::tt::d2m";

  let verify = [{
    return detail::verifyGenericParent($_op);
  }];
}

// Interface denoting which operands of an op must be sourced from dst register

def D2M_OperandLoadStoreRegisterOpInterface : OpInterface<"OperandLoadStoreRegisterOpInterface"> {
  let cppNamespace = "::mlir::tt::d2m";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        This interface denotes which op operand indices require their operands to be
        sourced from the destination register. It is primarily used by the
        insert-dst-register-access pass to autogenerate copynests to and from dst.
      }],
      /*retTy=*/"::mlir::SmallVector<int64_t>",
      /*methodName=*/"getOperandsLoadFromDstRegister",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::SmallVector<int64_t>{};"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns whether this operation should use the destination register in place.
        When true, the operation can read from and write to the same register location.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"getDstRegInPlace",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return false;"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns whether this operation supports both tile+tile and tile+scalar operands.
        For such operations, the in-place behavior depends on whether rhs is a scalar.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"supportsTileOrScalarRhs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return false;"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns whether the operand at the given index is a scalar.
        A scalar is defined as NOT a tile AND is int/float.
        This is used to skip loading scalar operands from the destination register.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isScalarOperand",
      /*args=*/(ins "int64_t":$operandIdx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if ($_op.supportsTileOrScalarRhs() && operandIdx == 1) {
          auto rhsOperand = $_op->getOperand(1);
          auto rhsType = rhsOperand.getType();
          // Check if rhs is a scalar: NOT a tile AND is int/float
          return !llvm::isa<::mlir::tt::ttcore::TileType>(rhsType) &&
                 rhsType.isIntOrFloat();
        }
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the number of DST scratch slices this op needs beyond its
        operand/output slices. Only 0 or 1 are supported today: when 1, the
        insert-dst-register-access pass reserves a slice from the DST
        allocator and records its index on the op via `dst_scratch_index`
        for the lowering to consume.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getNumDstScratchSlices",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return 0;"
    >,
  ];
}

// Interface declaring per-operand tilization fusion CAPABILITY on compute ops
// in the generic region. Used by the post-LowerToLayout `D2MGenericFusion`
// pass to decide whether a tilize-only (or untilize-only) `d2m.generic` can
// be folded into a consumer/producer's body.
//
// === Responsibility split ===
//
//   * Interface (this file): per-op CAPABILITY + per-op ENVIRONMENTAL
//     CONSTRAINTS. "Given the current pipeline setting `l1AccEnabled`, can
//     this op accept untilized data on operand `i` / emit untilized data on
//     result `i`?" The op encodes its own physical limitations here (e.g.
//     matmul's `pack_untilize` requires L1-acc off because it reads DST).
//
//   * Pass (D2MGenericFusion): IR STRUCTURE. Finds pure tilize-/untilize-
//     shaped `d2m.generic` neighbours, enforces single-use / grid-match /
//     no-`d2m.mask` invariants, performs the rewrite. The pass never makes
//     op-specific decisions — it only asks the interface.
//
// Defaults return false so non-opting ops never enter the fusion path.

def D2M_TilizedOperandInterface : OpInterface<"TilizedOperandInterface"> {
  let cppNamespace = "::mlir::tt::d2m";
  let description = [{
    Compute ops opt in to tilize/untilize fusion by implementing this
    interface and overriding the relevant per-operand / per-result methods.

    The output-side method takes `l1AccEnabled` because the only env-
    dependent constraint we currently model lives on the pack-untilize
    path. The input-side method is env-free — the unpacker can tilize on
    the fly regardless of L1-acc setting. New env-dependent constraints
    on the input side would be added by extending that method's signature.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns true if operand `operandIdx` may be sourced as scalar
        (row-major) data and tilized on the fly by an inlined
        `d2m.tile_tilize_block` inside the consumer's body. When true, the
        post-LTL `D2MGenericFusion` pass is allowed to fold a pure
        tilize-`d2m.generic` producer into this consumer over that operand.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"canConsumeUntilizedOperand",
      /*args=*/(ins "int64_t":$operandIdx),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return false;"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if result `resultIdx` may be emitted as scalar
        (row-major) data, untilizing on the fly via an inlined
        `d2m.tile_untilize_block`. When true, the post-LTL `D2MGenericFusion`
        pass is allowed to fold a pure untilize-`d2m.generic` consumer into
        this producer's body.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"canProduceUntilizedResult",
      /*args=*/(ins "int64_t":$resultIdx, "bool":$l1AccEnabled),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return false;"
    >,
  ];
}

#endif // TTMLIR_D2M_DIALECT_D2M_IR_D2MGENERICREGIONOPSINTERFACES_TD
