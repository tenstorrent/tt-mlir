// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_D2M_D2MGENERICREGIONOPS_TD
#define TTMLIR_TTMLIR_DIALECT_D2M_D2MGENERICREGIONOPS_TD

include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"
include "ttmlir/Dialect/D2M/IR/D2MOpsTypes.td"
include "ttmlir/Dialect/D2M/IR/D2MBase.td"
include "ttmlir/Dialect/D2M/IR/D2MOpsAttrs.td"
include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.td"

include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Generic Region Op Traits and Classes
//===----------------------------------------------------------------------===//

def IsDeviceL1MemorySpace : CPred<"::llvm::cast<::mlir::tt::ttcore::MemorySpaceAttr>(::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace()).getValue() == ::mlir::tt::ttcore::MemorySpace::DeviceL1">;

def IsDeviceDRAMMemorySpace : CPred<"::llvm::cast<::mlir::tt::ttcore::MemorySpaceAttr>(::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace()).getValue() == ::mlir::tt::ttcore::MemorySpace::DeviceDRAM">;

def IsDeviceMemorySpace : Or<[IsDeviceL1MemorySpace, IsDeviceDRAMMemorySpace]>;

def IsRegisterDstMemorySpace : CPred<"::llvm::cast<::mlir::tt::ttcore::MemorySpaceAttr>(::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace()).getValue() == ::mlir::tt::ttcore::MemorySpace::RegisterDst">;

def DeviceL1MemRef : Type<
  And<[IsDeviceL1MemorySpace, HasRankGreaterOrEqualPred<1>]>,
  "device l1 memoryspace memref type",
  "::mlir::MemRefType">;

def DeviceMemRef : Type<
  And<[IsDeviceMemorySpace, HasRankGreaterOrEqualPred<1>]>,
  "device memoryspace memref type",
  "::mlir::MemRefType">;

def RegisterDstMemRef : Type<
  And<[IsRegisterDstMemorySpace, HasRankGreaterOrEqualPred<1>]>,
  "register memoryspace memref type",
  "::mlir::MemRefType">;

def DeviceOrRegisterMemRef : AnyTypeOf<[DeviceMemRef, RegisterDstMemRef]>;

// Allow either device/register memrefs or ranked tensors for generic region block ops.
def DeviceOrRegisterMemRefOrAnyRankedTensor : AnyTypeOf<[
  DeviceOrRegisterMemRef,
  AnyRankedTensor
]>;

class D2M_GenericRegionOp<string mnemonic, list<Trait> traits = []> :
    D2M_Op<mnemonic, [D2M_GenericParent] # traits> {}

class D2M_GenericRegionComputeOp<string mnemonic, list<Trait> traits = []> :
    D2M_GenericRegionOp<mnemonic, [D2M_GenericRegionComputeOpTrait, D2M_OperandLoadStoreRegisterOpInterface] # traits> {}

class D2M_GenericRegionComputeUnaryDstOp<string mnemonic, list<Trait> traits = []> :
    D2M_GenericRegionComputeOp<mnemonic, traits> {

  let extraClassDeclaration = [{
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      return {0};
    }
     bool getDstRegInPlace() { return true; }
  }];
}

class D2M_GenericRegionComputeBinaryDstOp<string mnemonic, list<Trait> traits = []> :
    D2M_GenericRegionComputeOp<mnemonic, traits> {

  let extraClassDeclaration = [{
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      return {0, 1};
    }
  }];
}

class D2M_GenericRegionDatamovementOp<string mnemonic, list<Trait> traits = []> :
    D2M_GenericRegionOp<mnemonic, [D2M_GenericRegionDatamovementOpTrait] # traits> {}

//===----------------------------------------------------------------------===//
// D2M Generic Region Trait Bundles
//===----------------------------------------------------------------------===//

// Shared trait bundle for TTIR tile compute ops.
defvar D2M_TileComputeOpTraitBundle = [
  Pure
];

//===----------------------------------------------------------------------===//
// D2M Generic Region Math Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def D2M_TileAddOp : D2M_GenericRegionComputeBinaryDstOp<"tile_add", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Add Op";
    let description = [{
        The `tile_add` operation adds two tiles element-wise on the SFPU.
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileSubOp : D2M_GenericRegionComputeBinaryDstOp<"tile_sub", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Sub Op";
    let description = [{
        The `tile_sub` operation subtracts two tiles element-wise on the SFPU.
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileMulOp : D2M_GenericRegionComputeBinaryDstOp<"tile_mul", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Mul Op";
    let description = [{
        The `tile_mul` operation multiplies two tiles element-wise on the SFPU.
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileDivOp : D2M_GenericRegionComputeBinaryDstOp<"tile_div", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Div Op";
    let description = [{
        The `tile_div` operation divides two tiles element-wise.
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileRecipOp : D2M_GenericRegionComputeUnaryDstOp<"tile_recip", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Recip Op";
    let description = [{
        The `tile_recip` operation computes the reciprocal of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TilePowOp : D2M_GenericRegionComputeBinaryDstOp<"tile_pow", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Pow Op";
    let description = [{
        The `tile_pow` operation raises two tiles element-wise.
    }];

    let arguments = (ins TTCore_Tile:$lhs,
                         TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileExpOp : D2M_GenericRegionComputeUnaryDstOp<"tile_exp", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Exp Op";
    let description = [{
        The `tile_exp` operation computes the exponential of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileLogOp : D2M_GenericRegionComputeUnaryDstOp<"tile_log", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Log Op";
    let description = [{
        The `tile_log` operation computes the natural logarithm of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileNegativeOp : D2M_GenericRegionComputeUnaryDstOp<"tile_negative", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Negative Op";
    let description = [{
        The `tile_negative` operation computes the negative of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileCosOp : D2M_GenericRegionComputeUnaryDstOp<"tile_cos", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Cos Op";
    let description = [{
        The `tile_cos` operation computes the cosine function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileTanOp : D2M_GenericRegionComputeUnaryDstOp<"tile_tan", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Tan Op";
    let description = [{
        The `tile_tan` operation computes the tangent function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileSqrtOp : D2M_GenericRegionComputeUnaryDstOp<"tile_sqrt", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Sqrt Op";
    let description = [{
        The `tile_sqrt` operation computes the sqrt function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileRsqrtOp : D2M_GenericRegionComputeUnaryDstOp<"tile_rsqrt", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Rsqrt Op";
    let description = [{
        The `tile_rsqrt` operation computes the reciprocal sqrt function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileSinOp : D2M_GenericRegionComputeUnaryDstOp<"tile_sin", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Sin Op";
    let description = [{
        The `tile_sin` operation computes the sine function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileSigmoidOp : D2M_GenericRegionComputeUnaryDstOp<"tile_sigmoid", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Sigmoid Op";
    let description = [{
        The `tile_sigmoid` operation computes the sigmoid of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileSiluOp : D2M_GenericRegionComputeUnaryDstOp<"tile_silu", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Sliu Op";
    let description = [{
        The `tile_silu` operation computes the silu of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileReluOp : D2M_GenericRegionComputeUnaryDstOp<"tile_relu", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Relu Op";
    let description = [{
        The `tile_relu` operation computes the relu of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileGeluOp : D2M_GenericRegionComputeUnaryDstOp<"tile_gelu", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile GELU Op";
    let description = [{
        The `tile_gelu` operation computes the GELU (Gaussian Error Linear Unit) of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileCeilOp : D2M_GenericRegionComputeUnaryDstOp<"tile_ceil", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Ceil Op";
    let description = [{
        The `tile_ceil` operation computes the ceiling function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileFloorOp : D2M_GenericRegionComputeUnaryDstOp<"tile_floor", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Floor Op";
    let description = [{
        The `tile_floor` operation computes the floor function of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileAbsOp : D2M_GenericRegionComputeUnaryDstOp<"tile_abs", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Abs Op";
    let description = [{
        The `tile_abs` operation computes the absolute value of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileBitwiseNotOp : D2M_GenericRegionComputeUnaryDstOp<"tile_bitwise_not", D2M_TileComputeOpTraitBundle>{
    let summary = "D2M Tile Bitwise Not Op";
    let description = [{
        The `tile_bitwise_not` operation computes the bitwise negation of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileLogicalNotOp : D2M_GenericRegionComputeUnaryDstOp<"tile_logical_not", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Logical Not Op";
    let description = [{
        The `tile_logical_not` operation computes the logical negation of each element in the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileEqzOp : D2M_GenericRegionComputeUnaryDstOp<"tile_eqz"> {
    let summary = "D2M Tile Eqz Op";
    let description = [{
        The `tile_eqz` operation checks if each element in the input tile == 0
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileNezOp : D2M_GenericRegionComputeUnaryDstOp<"tile_nez", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Nez Op";
    let description = [{
        The `tile_nez` operation checks if each element in the input tile != 0
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileGtzOp : D2M_GenericRegionComputeUnaryDstOp<"tile_gtz", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Gtz Op";
    let description = [{
        The `tile_gtz` operation checks if each element in the input tile > 0
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileGezOp : D2M_GenericRegionComputeUnaryDstOp<"tile_gez", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Gez Op";
    let description = [{
        The `tile_gez` operation checks if each element in the input tile >= 0
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileLtzOp : D2M_GenericRegionComputeUnaryDstOp<"tile_ltz", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Ltz Op";
    let description = [{
        The `tile_ltz` operation checks if each element in the input tile < 0
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileLezOp : D2M_GenericRegionComputeUnaryDstOp<"tile_lez", D2M_TileComputeOpTraitBundle> {
    let summary = "D2M Tile Lez Op";
    let description = [{
        The `tile_lez` operation checks if each element in the input tile <= 0
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileMaximumOp : D2M_GenericRegionComputeBinaryDstOp<"tile_maximum", D2M_TileComputeOpTraitBundle>{
    let summary = "D2M Tile Maximum Op";
    let description = [{
        The `tile_maximum` operation calculates the maximum of two tensors element-wise.
    }];

    let arguments = (ins TTCore_Tile:$lhs, TTCore_Tile:$rhs);
    let results = (outs TTCore_Tile:$result);
}

def D2M_TileReduceSumOp : D2M_GenericRegionComputeOp<"tile_reduce_sum"> {
  let summary = "D2M Tile Reduce Sum Op";
  let description = [{
        The `tile_reduce_sum` operation computes the sum of all elements in the input A element-wise multiplied by B and input C over the specified reduction dim(s): result <- sum<dims>(A * B, C)
    }];

  let arguments = (ins TTCore_Tile:$a,
                       TTCore_Tile:$b,
                       TTCore_Tile:$c,
                       D2M_ReduceDimAttr:$reduce_dim);
  let results = (outs TTCore_Tile:$result);

  let extraClassDeclaration = [{
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      return {2};
    }
    bool getDstRegInPlace() {
      return true;
    }
  }];
}

def D2M_TileReduceMaxOp : D2M_GenericRegionComputeOp<"tile_reduce_max"> {
  let summary = "D2M Tile Reduce Max Op";
  let description = [{
        The `tile_reduce_max` operation computes the max of all elements in the input A element-wise multiplied by B and input C over the specified reduction dim(s): result <- max<dims>(A * B, C)
    }];

  let arguments = (ins TTCore_Tile:$a,
                       TTCore_Tile:$b,
                       TTCore_Tile:$c,
                       D2M_ReduceDimAttr:$reduce_dim);
  let results = (outs TTCore_Tile:$result);

  let extraClassDeclaration = [{
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      return {2};
    }
    bool getDstRegInPlace() {
      return true;
    }
  }];
}

def D2M_PackerMaskResetOp : D2M_GenericRegionComputeOp<"packer_mask_reset"> {
  let summary = "D2M Packer Mask Reset Op";
  let description = [{
        The `packer_mask_reset` operation resets the packer mask. This should be removed once LLK refactors the reduce API.
        See tt-metal#22904 and tt-metal -> reduce.h.
    }];
}

def D2M_TileMatmulOp : D2M_GenericRegionComputeOp<"tile_matmul"> {
  let summary = "D2M Tile Matmul Op";
  let description = [{
        The `tile_matmul` operation computes the matrix multiplication of A and B input tiles and element-wise adds C tile: result <- a @ b + c.
    }];

  let arguments = (ins TTCore_Tile:$a, TTCore_Tile:$b, TTCore_Tile:$c);
  let results = (outs TTCore_Tile:$result);

  let extraClassDeclaration = [{
    mlir::SmallVector<int64_t> getOperandsLoadFromDstRegister() {
      return {2};
    }
    bool getDstRegInPlace() {
      return true;
    }
  }];
}

def D2M_TileMatmulBlockOp : D2M_GenericRegionComputeOp<"tile_matmul_block",
  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "D2M Tile Matmul Block Op";
    let description = [{
        The `tile_matmul_block` operation computes the matrix multiplication of two input blocks.
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$a,
                         AnyRankedTensorOrMemRef:$b,
                         AnyRankedTensorOrMemRef:$output);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def D2M_TileTilizeBlockOp : D2M_GenericRegionComputeOp<"tile_tilize_block",
  [ DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       ]>
  , D2M_SkipOpEltwiseFusionTrait
  ]> {
    let summary = "D2M Tile Tilize Block Op";
    let description = [{
        The `tile_tilize_block` operation tilizes the input row major memref block and outputs the memref containing the tilized data.
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$input,
                         AnyRankedTensorOrMemRef:$output);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def D2M_TileUntilizeBlockOp : D2M_GenericRegionComputeOp<"tile_untilize_block",
  [ DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       ]>
  , D2M_SkipOpEltwiseFusionTrait
  ]> {
    let summary = "D2M Tile Untilize Block Op";
    let description = [{
        The `tile_untilize_block` operation untilizes the input tilized memref block and outputs the memref contianing the row major data.
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$input,
                         AnyRankedTensorOrMemRef:$output);

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
    }];

    let hasVerifier = 1;
}

def D2M_TileTypecastOp : D2M_GenericRegionComputeUnaryDstOp<"tile_typecast"> {
    let summary = "D2M Tile Typecast Op";
    let description = [{
        The `tile_typecast` operation casts the input tile to the desired dataformat.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

def D2M_DstReinterpretCastOp : D2M_GenericRegionOp<"dst_reinterpret_cast"> {
    let summary = "D2M Destination Register Reinterpret Cast Op";
    let description = [{
        The `dst_reinterpret_cast` operation reinterprets a tile value as a different
        tile type for the purpose of storing/loading from the destination register.

        This models the hardware behavior that destination register memory is untyped
        and can be accessed with different type interpretations. It is inserted during
        the InsertDstRegisterAccess pass to reconcile type mismatches between compute
        operations and destination register allocations, and is removed during lowering
        to TTKernel as a no-op.

        Example:
        ```mlir
        %0 = affine.load %dst : memref<1x1x4x!ttcore.tile<32x32, f32>, #dst>
        %1 = d2m.tile_typecast %0 : !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, bf16>
        %2 = d2m.dst_reinterpret_cast %1 : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
        affine.store %2, %dst : memref<1x1x4x!ttcore.tile<32x32, f32>, #dst>
        ```
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

//===----------------------------------------------------------------------===//
// D2M Destination Control Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def D2M_AcquireDstOp : D2M_GenericRegionComputeOp<"acquire_dst",
  [ MemoryEffects<[MemRead, MemWrite]>
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
    let summary = "Acquire Destination Register op.";
    let description = [{
      This op returns a memref with memory space dest that models the destination register
      resource on tensorrent hardware. Example IR:

      ```mlir
      %dst = d2m.acquire_dst() : memref<2x4x!tt.tile<32x32, f32>, #tt.memory_space<dst>>
      ```
    }];

    let results = (outs RegisterDstMemRef:$result);

    let assemblyFormat = [{ `(` `)` attr-dict `:` type($result) }];
}

def D2M_TileTransposeOp : D2M_GenericRegionComputeOp<"tile_transpose"> {
    let summary = "D2M Tile Transpose Op";
    let description = [{
        The `tile_transpose` operation computes the transpose of the input tile.
    }];

    let arguments = (ins TTCore_Tile:$input);
    let results = (outs TTCore_Tile:$result);
}

//===----------------------------------------------------------------------===//
// D2M Generic Region Datamovement Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def D2M_DMAOp : D2M_GenericRegionDatamovementOp<"dma",
  [ AttrSizedOperandSegments
  , DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "isNotConflicting"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       ]>
  ]> {
    let summary = "D2M DMA Op";
    let description = [{
      DMA operation, moves data from src to dst where src and dst are memrefs that may or may not belong to
      different memory spaces. This op comes in a few flavors and is capable of roughly expressing everything
      that underlying Noc hardware can do. The op can be used to express a wide range of data movement operations:

      - Local to local
      ```mlir
      %tx = d2m.dma %src, %dst : (memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
      ```

      - Local to remote dram
      ```mlir
      %tx = d2m.dma %src, %dst : (memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<1x1x6x8x!ttcore.tile<32x32, f32>, $ttcore.shard<...>, #dram>) -> !d2m.mem_tx
      ```

      - Remote dram to local
      ```mlir
      %tx = d2m.dma %src, %dst : (memref<1x1x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<...>, #dram>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
      ```

      - Local to mcast, e.g. starting at offset core[1, 2] with mcast shape [4, 4] (src and dst have the same SSA value, implies NoC doesn't loopback)
      ```mlir
      %tx = d2m.dma %foo, %foo, core[%c1, %c2] mcast[%c4, %c4] : (memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
      ```

      - Local to mcast w/ loopback (same as above but src and dst are different SSA values)
      ```mlir
      %tx = d2m.dma %src, %dst, core[%c1, %c2] mcast[%c4, %c4] : (memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
      ```

      - A lower level form, is "fully indexed" DMA, where the src and dst memrefs have thier ranks fully indexed the additional numElems argument
        is used to specify the number of elements to transfer. As opposed to the forms above that are not fully indexed which implicitly means to
        transfer the leftover block.  That is given memref<2x2x3x4x!ttcore.tile<>>, if we indexed by [i, j], since we only indexed a rank of 2, the remaining
        3x4 block is transferred.  In the fully indexed form, to express the same thing we would need to specify the number of elements to transfer:
      ```mlir
      %tx = d2m.dma %src[%i, %j, %k, %l], %dst[%i, %j, %k, %l], 12 : (memref<2x2x3x4x!ttcore.tile<32x32, f32>, #dram>, memref<2x2x3x4x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
      ```

      - Conversely, a higher level form is affine form, where the src and dst indices are implied by an affine map shared with the parent generic op.
        Given some affine map, as demonstrated below, the dma op will be evaluated at each point in the affine map's iteration space.
      ```mlir
      #map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
      %tx = d2m.dma %stream<#map1>, %cb0 : (memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<rank(4)>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
      ```

      Some constraints:
      - src and dst must have the same element type.
      - src and dst cannot both be remote
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$src, OptionalAttr<AffineMapAttr>:$srcAffineMap, Variadic<Index>:$srcIndices,
                         AnyRankedTensorOrMemRef:$dst, OptionalAttr<AffineMapAttr>:$dstAffineMap, Variadic<Index>:$dstIndices,
                         OptionalAttr<I64Attr>:$optNumElems, Variadic<Index>:$mcastStartIndex, Variadic<Index>:$mcastShape);
    let results = (outs D2M_MemTx:$result);

    let builders =
    [
      // %tx = d2m.dma %src, %dst
      OpBuilder<(ins "Value": $src, "Value": $dst),
      [{
        build($_builder, $_state, src, nullptr, dst);
      }]>,
      // %tx = d2m.dma %src<#map>, %dst
      OpBuilder<(ins "Value": $src, "AffineMapAttr": $srcAffineMap, "Value": $dst),
      [{
        build($_builder, $_state, src, srcAffineMap, dst, ValueRange(), ValueRange());
      }]>,
      // %tx = d2m.dma %src<#map>, %dst core[%c0, %c1] mcast[%c2, %c3]
      OpBuilder<(ins "Value": $src, "AffineMapAttr": $srcAffineMap, "Value": $dst, "ValueRange": $mcastStartIndex, "ValueRange": $mcastShape),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>(), src, srcAffineMap, ValueRange(), dst, nullptr, ValueRange(), nullptr, mcastStartIndex, mcastShape);
      }]>,
      // %tx = d2m.dma %src, %dst<#map> core[%c0, %c1] mcast[%c2, %c3]
      OpBuilder<(ins "Value": $src, "Value": $dst, "AffineMapAttr": $dstAffineMap, "ValueRange": $mcastStartIndex, "ValueRange": $mcastShape),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>(), src, nullptr, ValueRange(), dst, dstAffineMap, ValueRange(), nullptr, mcastStartIndex, mcastShape);
      }]>,
      // %tx = d2m.dma %src, %dst<#map>
      OpBuilder<(ins "Value": $src, "Value": $dst, "AffineMapAttr": $dstAffineMap),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>(), src, nullptr, ValueRange(), dst, dstAffineMap, ValueRange(), nullptr, ValueRange(), ValueRange());
      }]>,
      // %tx = d2m.dma %src[%c0, %c1], %dst
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst),
      [{
        build($_builder, $_state, src, srcIndices, dst, ValueRange());
      }]>,
      // %tx = d2m.dma %src[%c0, %c1], %dst core[%c2, %c3] mcast[%c4, %c5]
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $mcastStartIndex, "ValueRange": $mcastShape),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>(), src, nullptr, srcIndices, dst, nullptr, ValueRange(), nullptr, mcastStartIndex, mcastShape);
      }]>,
      // %tx = d2m.dma %src, %dst[%c0, %c1]
      OpBuilder<(ins "Value": $src, "Value": $dst, "ValueRange": $dstIndices),
      [{
        build($_builder, $_state, src, ValueRange(), dst, dstIndices);
      }]>,
      // %tx = d2m.dma %src[%c0, %c1], %dst[%c2, %c3]
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $dstIndices),
      [{
        build($_builder, $_state, src, srcIndices, dst, dstIndices, ValueRange(), ValueRange());
      }]>,
      // %tx = d2m.dma %src[%c0, %c1], %dst[%c2, %c3] core[%c4, %c5] mcast[%c6, %c7]
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $dstIndices, "ValueRange": $mcastStartIndex, "ValueRange": $mcastShape),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>(), src, nullptr, srcIndices, dst, nullptr, dstIndices, nullptr, mcastStartIndex, mcastShape);
      }]>,
      // %tx = d2m.dma %src[%c0, %c1], %dst[%c2, %c3], <4>
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $dstIndices, "size_t": $numElems),
      [{
        build($_builder, $_state, src, srcIndices, dst, dstIndices, ValueRange(), ValueRange(), numElems);
      }]>,
      // %tx = d2m.dma %src[%c0, %c1], %dst[%c2, %c3] core[%c4, %c5] mcast[%c6, %c7], <4>
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $dstIndices, "ValueRange": $mcastStartIndex, "ValueRange": $mcastShape, "size_t": $numElems),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>(), src, nullptr, srcIndices, dst, nullptr, dstIndices, $_builder.getI64IntegerAttr(numElems), mcastStartIndex, mcastShape);
      }]>,
    ];

    let assemblyFormat = [{ $src (`<` $srcAffineMap^ `>`)? (`[` $srcIndices^ `]`)? `,` $dst (`<` $dstAffineMap^ `>`)? (`[` $dstIndices^ `]`)? (`core` `[` $mcastStartIndex^ `]` `mcast` `[` $mcastShape `]`)? (`,` `<` $optNumElems^ `>`)? attr-dict `:` `(` type($src) `,` type($dst) `)` `->` type($result)}];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      MemRefType getSrcMemRefType() { return cast<MemRefType>(getSrc().getType()); }
      MemRefType getDstMemRefType() { return cast<MemRefType>(getDst().getType()); }
      int64_t getNumElems();
      size_t getSizeBytes();
      bool isSrcLocal() { return !isSrcRemote(); }
      bool isSrcRemote() { return ttcore::hasDeviceLayout(getSrc()); }
      bool isDstLocal() { return !isDstRemote(); }
      bool isDstRemote() { return ttcore::hasDeviceLayout(getDst()); }
      bool isMcast() { return !getMcastShape().empty(); }
      bool isAffine() { return (getSrcAffineMap() || getDstAffineMap()) && getSrcIndices().empty() && getDstIndices().empty(); }
      ::mlir::tt::ttcore::MemorySpace getSrcMemorySpace() {
        return mlir::cast<::mlir::tt::ttcore::MemorySpaceAttr>(getSrcMemRefType().getMemorySpace()).getValue();
      }
      ::mlir::tt::ttcore::MemorySpace getDstMemorySpace() {
        return mlir::cast<::mlir::tt::ttcore::MemorySpaceAttr>(getDstMemRefType().getMemorySpace()).getValue();
      }

    }];
}


def D2M_DMAWriteOp : D2M_GenericRegionDatamovementOp<"dma_write",
  [ AttrSizedOperandSegments
  , DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  , D2M_DMAOpInterface
  ]> {

    let summary = "Fully indexed and lowered D2M DMA Op for writes";
    let description = [{
      This op is the fully lowered explicit write form of the higher level `d2m.dma` operation. It corresponds 1:1
      to a single concrete DMA write operation from src memref to dst memref, where src and dst indices are fully resolved
      (all view and memspace-specific affine maps applied).

      This operation supports multicast; providing an empty multicast shape argument _implies a unicast write_.

      ```mlir
      %tx = d2m.dma_write %src[%gridy, %gridx, %shardy, %shardx], %dst[%gridy, %gridx, %shardy, %shardx], <%size>
        : (memref<2x2x3x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x2x3x4x!ttcore.tile<32x32, f32>, #dram>) -> !d2m.mem_tx
      ```

      Constraints:
      - src MUST be a LOCAL memref!
      - src and dst must have the same element type.
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$src, Variadic<Index>:$srcIndices,
                         AnyRankedTensorOrMemRef:$dst, Variadic<Index>:$dstIndices,
                         I64Attr:$numElems,
                         Variadic<Index>:$mcastStartIndex, Variadic<Index>:$mcastShape);
    let results = (outs D2M_MemTx:$result);

    let builders =
    [
      // Unicast - multicast args are defaulted
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $dstIndices, "size_t": $numElems),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>(), src, srcIndices, dst, dstIndices, $_builder.getI64IntegerAttr(numElems), ValueRange(), ValueRange());
      }]>,

      // Multicast - multicast args must be provided
      OpBuilder<(ins "Value": $src, "ValueRange": $srcIndices, "Value": $dst, "ValueRange": $dstIndices, "ValueRange": $mcastStartIndex, "ValueRange": $mcastShape, "size_t": $numElems),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>(), src, srcIndices, dst, dstIndices, $_builder.getI64IntegerAttr(numElems), mcastStartIndex, mcastShape);
      }]>,
    ];

    let assemblyFormat = [{ $src `[` $srcIndices `]` `,` $dst `[` $dstIndices `]` (`core` `[` $mcastStartIndex^ `]` `mcast` `[` $mcastShape `]`)?  `,` `<` $numElems `>` attr-dict `:` `(` type($src) `,` type($dst) `)` `->` type($result)}];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      MemRefType getSrcMemRefType() { return cast<MemRefType>(getSrc().getType()); }
      MemRefType getDstMemRefType() { return cast<MemRefType>(getDst().getType()); }
      size_t getSizeBytes() {
        auto elementSizeBytes =
          ttcore::getElementSizeBytes(getSrcMemRefType().getElementType());
        return getNumElems() * elementSizeBytes;
      }
      bool isDstLocal() { return !isDstRemote(); }
      bool isDstRemote() { return ttcore::hasDeviceLayout(getDst()); }
      bool isMcast() { return !getMcastShape().empty(); }
      ::mlir::tt::ttcore::MemorySpace getDstMemorySpace() {
        return mlir::cast<::mlir::tt::ttcore::MemorySpaceAttr>(getDstMemRefType().getMemorySpace()).getValue();
      }
    }];
}

def D2M_DMAReadOp : D2M_GenericRegionDatamovementOp<"dma_read",
  [ AttrSizedOperandSegments
  , DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  , D2M_DMAOpInterface
  ]> {

    let summary = "Fully indexed and lowered D2M DMA Op for read";
    let description = [{
      This op is the fully lowered explicit read form of the higher level `d2m.dma` operation. It corresponds 1:1
      to a single concrete DMA read operation from src memref to dst memref, where src and dst indices are fully resolved
      (all view and memspace-specific affine maps applied). The dst memref _must_ be local!

      This operation only supports unicast (mulicast are write-only).

      ```mlir
      %tx = d2m.dma_read %src[%gridy, %gridx, %shardy, %shardx], %dst[%gridy, %gridx, %shardy, %shardx], %size
        : (memref<2x2x3x4x!ttcore.tile<32x32, f32>, #dram>, memref<2x2x3x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
      ```

      Constraints:
      - src MUST be a REMOTE memref! if both operands are local, use DMAWriteOp
      - dst MUST be a LOCAL memref!
      - src and dst must have the same element type.
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$src, Variadic<Index>:$srcIndices,
                         AnyRankedTensorOrMemRef:$dst, Variadic<Index>:$dstIndices,
                         I64Attr:$numElems);
    let results = (outs D2M_MemTx:$result);

    let assemblyFormat    = [{ $src `[` $srcIndices `]` `,` $dst `[` $dstIndices `]` `,` `<` $numElems `>` attr-dict `:` `(` type($src) `,` type($dst) `)` `->` type($result)}];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      MemRefType getSrcMemRefType() { return cast<MemRefType>(getSrc().getType()); }
      MemRefType getDstMemRefType() { return cast<MemRefType>(getDst().getType()); }
      size_t getSizeBytes() {
        auto elementSizeBytes =
          ttcore::getElementSizeBytes(getSrcMemRefType().getElementType());
        return getNumElems() * elementSizeBytes;
      }
      ::mlir::tt::ttcore::MemorySpace getSrcMemorySpace() {
        return mlir::cast<::mlir::tt::ttcore::MemorySpaceAttr>(getSrcMemRefType().getMemorySpace()).getValue();
      }
    }];
}

def D2M_NullTxOp : D2M_GenericRegionDatamovementOp<"null_tx", [Pure]> {
    let summary = "Create a null transaction.";
    let description = [{
      Utility op to create a null transaction.  This is required for creating a sentinel
      starting transaction for a DMA nested inside of a loop nest.
    }];

    let results = (outs D2M_MemTx:$result);

    let builders =
    [
      OpBuilder<(ins),
      [{
        build($_builder, $_state, $_builder.getType<MemTxType>());
      }]>
    ];

    let assemblyFormat = [{ attr-dict }];
}

def D2M_DMAWaitOp : D2M_GenericRegionDatamovementOp<"dma_wait", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "D2M DMA wait Op";
    let description = [{
      Waits for the producer DMA memory transaction to complete.
    }];

    let arguments = (ins D2M_MemTx:$mem_tx);

    let assemblyFormat = [{ $mem_tx attr-dict }];
}

//===----------------------------------------------------------------------===//
// D2M Generic Region Semaphore Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

class D2M_SemaphoreUpdateOp<string mnemonic> : D2M_GenericRegionOp<mnemonic,
  [ AttrSizedOperandSegments
  , MemoryEffects<[MemRead, MemWrite]>
  ]> {
    let summary = "D2M Semaphore Set or Inc Op";
    let description = [{
      Set or increment the semaphore value atomically. This op comes in a few flavors:

      - Set this local core's semaphore value
      ```mlir
      d2m.semaphore_set %sem0, %c1
      ```

      - Update a remote core's semaphore value
      ```mlir
      d2m.semaphore_inc %sem0, %c1 core[%c2, %c2]
      ```

      - Update a remote mcast region of cores' semaphore value
      ```mlir
      d2m.semaphore_set %sem0, %c1 core[%c2, %c2] mcast[%c4, %c4]
      ```
    }];

    let arguments = (ins D2M_Semaphore:$semaphore, Index:$value,
                         Variadic<Index>:$dstCoreIndex, Variadic<Index>:$mcastShape);

    let assemblyFormat = [{ $semaphore `,` $value (`,` `core` `[` $dstCoreIndex^ `]`)? (`mcast` `[` $mcastShape^ `]`)? attr-dict }];

    let builders =
    [
      OpBuilder<(ins "Value":$semaphore, "Value":$value),
      [{
        build($_builder, $_state, semaphore, value, ValueRange(), ValueRange());
      }]>,
      OpBuilder<(ins "Value":$semaphore, "Value":$value, "ValueRange":$dstCoreIndex),
      [{
        build($_builder, $_state, semaphore, value, dstCoreIndex, ValueRange());
      }]>,
    ];
}

def D2M_SemaphoreSetOp : D2M_SemaphoreUpdateOp<"semaphore_set"> {}

def D2M_SemaphoreIncOp : D2M_SemaphoreUpdateOp<"semaphore_inc"> {}

def D2M_SemaphoreWaitOp : D2M_GenericRegionOp<"semaphore_wait", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "D2M Semaphore Set Op.";
    let description = [{
      Wait for the semaphore value to reach the specified value. Optionall supply a reset value as a shorthand syntax.

      ```mlir
      d2m.semaphore_wait %sem1, %c1 reset %c0
      // is equivalent to
      d2m.semaphore_wait %sem1, %c1
      d2m.semaphore_set %sem1, %c0
      ```
    }];

    let arguments = (ins D2M_Semaphore:$semaphore, Index:$value, Optional<Index>:$resetValue);

    let assemblyFormat = [{ $semaphore `,` $value (`reset` $resetValue^)? attr-dict }];

    let builders =
    [
      OpBuilder<(ins "Value":$semaphore, "Value":$value),
      [{
        build($_builder, $_state, semaphore, value, nullptr);
      }]>,
    ];
}

//===----------------------------------------------------------------------===//
// D2M Generic Region Control Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def D2M_YieldOp : D2M_GenericRegionOp<"yield", [
    MemoryEffects<[MemRead, MemWrite]>,
    DeclareOpInterfaceMethods<BufferizableOpInterface, [
      "bufferizesToMemoryRead",
      "bufferizesToMemoryWrite",
      "bufferize",
      "getAliasingValues",
      "getBufferType"
    ]>
  ]> {
    let summary = "Yield op.";
    let description = [{
      Yield operation, return control flow to another thread. Corresponds to a circular buffer push.
    }];

    let arguments = (ins Variadic<AnyRankedTensor>:$values);

    let assemblyFormat = [{ $values attr-dict `:` `(` type($values) `)` }];

    let hasVerifier = 1;

    let extraClassDefinition = [{
      bool $cppClass::bufferizesToMemoryRead(
          mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
        return false;
      }
      bool $cppClass::bufferizesToMemoryWrite(
          mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
        return false;
      }
      mlir::bufferization::AliasingValueList $cppClass::getAliasingValues(
          mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
        mlir::bufferization::AliasingValueList result;
        return result;
      }
      mlir::FailureOr<mlir::BaseMemRefType> $cppClass::getBufferType(
          mlir::Value, const mlir::bufferization::BufferizationOptions &,
          const mlir::bufferization::BufferizationState &,
          ::llvm::SmallVector<mlir::Value> &) {
        return mlir::failure();
      }
      mlir::LogicalResult $cppClass::bufferize(
          mlir::RewriterBase &rewriter,
          const mlir::bufferization::BufferizationOptions &,
          mlir::bufferization::BufferizationState &state) {
        rewriter.eraseOp(*this);
        return mlir::success();
      }
    }];
}

class D2M_CBOp<string mnemonic, list<Trait> traits = []> : D2M_GenericRegionOp<mnemonic,
  traits #
  [ MemoryEffects<[MemRead, MemWrite]>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "isNotConflicting"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       , "hasTensorSemantics"
                                                       ]>
  ]> {
  let arguments = (ins D2M_CB:$cb);
  let results = (outs AnyRankedTensorOrMemRef:$result);

  let builders =
  [
    OpBuilder<(ins "Value":$cb),
    [{
      build($_builder, $_state, mlir::cast<CBType>(cb.getType()).getUnderlying(), cb);
    }]>
  ];

  let assemblyFormat = [{ $cb attr-dict `:` type($cb) `->` type($result) }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    CBType getCbType() {
      return mlir::cast<CBType>(getCb().getType());
    }

    ShapedType getUnderlyingType() {
      return getCbType().getUnderlying();
    }

    // Get the wrapped memref type from the circular buffer
    MemRefType getMemRefType() {
      return llvm::cast<MemRefType>(getUnderlyingType());
    }

    // Get the wrapped tensor type from the circular buffer
    RankedTensorType getTensorType() {
      return llvm::cast<RankedTensorType>(getUnderlyingType());
    }
  }];

  let extraClassDefinition = [{
    bool $cppClass::hasTensorSemantics() {
      return getCbType().hasTensorType();
    }

    bool $cppClass::bufferizesToMemoryRead(
        mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
      return false;
    }

    bool $cppClass::bufferizesToMemoryWrite(
        mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
      return false;
    }

    bool $cppClass::isNotConflicting(mlir::OpOperand *, mlir::OpOperand *,
                                 const mlir::bufferization::AnalysisState &) {
      // Return true to avoid forcing out of place bufferization.
      return true;
    }

    mlir::bufferization::AliasingValueList $cppClass::getAliasingValues(
        mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
      mlir::bufferization::AliasingValueList result;
      return result;
    }

    mlir::FailureOr<mlir::BaseMemRefType> $cppClass::getBufferType(
        mlir::Value value, const mlir::bufferization::BufferizationOptions &,
        const mlir::bufferization::BufferizationState &,
        ::llvm::SmallVector<mlir::Value> &) {
      llvm_unreachable("intentionally unimplemented, this op can only accept block arguments which should have already been converted");
    }

    mlir::LogicalResult $cppClass::bufferize(
        mlir::RewriterBase &rewriter,
        const mlir::bufferization::BufferizationOptions &options,
        mlir::bufferization::BufferizationState &) {
      auto cbBufferType = mlir::cast<bufferization::TensorLikeType>(getCbType()).getBufferType(options, [&] () {
        return this->emitOpError();
      });
      assert(succeeded(cbBufferType));
      auto toBuffer = rewriter.create<bufferization::ToBufferOp>(
          this->getLoc(), *cbBufferType, getCb());
      mlir::bufferization::replaceOpWithNewBufferizedOp<$cppClass>(
          rewriter, *this, toBuffer.getResult());
      return mlir::success();
    }

    mlir::LogicalResult $cppClass::verify() {
      // Verify that the result type matches the wrapped memref type
      auto cbType = llvm::cast<CBType>(getCbType());
      if (cbType.getUnderlying() != getResult().getType()) {
        return emitOpError() << "result type does not match circular buffer's "
                                "wrapped memref type";
      }

      return ::mlir::success();
    }
  }];
}

def D2M_WaitOp : D2M_CBOp<"wait"> {
  let summary = "Wait from circular buffer.";
  let description = [{
    Wait operation, extracts the enclosing memref from a circular buffer.
    This operation is used by consumer threads to access data from the circular buffer.
    It implicitly blocks until a chunk of memref sized underlying type is made
    available by a producer thread via d2m.reserve.

    Each value of !d2m.cb type can be thought of as a shared resource between threads
    akin to a single-producer, single-consumer queue. Where d2m.reserve and d2m.wait
    effectively implement push/pop queue semantics. One distinction which lends itself
    better to DPS style is that both reserve and wait guarantee acquisition of the
    underlying memory after the op has executed.

    The resource is implicitly released at the end of block scope.

    Example:
    ```mlir
    %memref = d2m.wait %cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    ```
  }];
}

def D2M_ReserveOp : D2M_CBOp<"reserve"> {
  let summary = "Reserve from circular buffer.";
  let description = [{
    Reserve operation, extracts the enclosing memref from a circular buffer.
    This operation is used by producer threads to reserve a memref sized underlying
    type's worth of space in the circular buffer. It implicitly blocks if there hasn't
    yet been a consumer of the data from another thread via d2m.wait.

    Each value of !d2m.cb type can be thought of as a shared resource between threads
    akin to a single-producer, single-consumer queue. Where d2m.reserve and d2m.wait
    effectively implement push/pop queue semantics. One distinction which lends itself
    better to DPS style is that both reserve and wait guarantee acquisition of the
    underlying memory after the op has executed.

    The resource is implicitly released at the end of block scope.

    Example:
    ```mlir
    %memref = d2m.reserve %cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    ```
  }];
}

def D2M_StoreOp : D2M_GenericRegionOp<"store",
  [ SameTypeOperands
  , MemoryEffects<[MemRead, MemWrite]>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "isNotConflicting"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       ]>
  ]> {
  let summary = "TODO.";
  let description = [{
    TODO
  }];

  let arguments = (ins AnyRankedTensor:$dst, AnyRankedTensor:$src);

  let assemblyFormat = [{ $dst `,` $src attr-dict `:` type($dst) }];

  let extraClassDefinition = [{
    bool mlir::tt::d2m::StoreOp::bufferizesToMemoryRead(
        mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
      return false;
    }

    bool mlir::tt::d2m::StoreOp::bufferizesToMemoryWrite(
        mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
      return false;
    }

    bool mlir::tt::d2m::StoreOp::isNotConflicting(mlir::OpOperand *, mlir::OpOperand *,
                                 const mlir::bufferization::AnalysisState &) {
      // Return true to avoid forcing out of place bufferization.
      return true;
    }

    mlir::bufferization::AliasingValueList mlir::tt::d2m::StoreOp::getAliasingValues(
        mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
      mlir::bufferization::AliasingValueList result;
      return result;
    }

    mlir::FailureOr<mlir::BaseMemRefType> mlir::tt::d2m::StoreOp::getBufferType(
        mlir::Value value, const mlir::bufferization::BufferizationOptions &,
        const mlir::bufferization::BufferizationState &,
        ::llvm::SmallVector<mlir::Value> &) {
      llvm_unreachable("intentionally unimplemented, this op will be erased during bufferization");
    }

    mlir::LogicalResult mlir::tt::d2m::StoreOp::bufferize(
        mlir::RewriterBase &rewriter,
        const mlir::bufferization::BufferizationOptions &options,
        mlir::bufferization::BufferizationState &state) {
      auto src = bufferization::getBuffer(rewriter, getSrc(), options, state);
      if (failed(src)) {
        return mlir::failure();
      }
      auto dst = bufferization::getBuffer(rewriter, getDst(), options, state);
      if (failed(dst)) {
        return mlir::failure();
      }
      rewriter.replaceAllUsesWith(*src, *dst);
      rewriter.eraseOp(*this);
      return mlir::success();
    }
  }];
}

//===----------------------------------------------------------------------===//
// D2M Generic Region Indexing Ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

class D2M_IndexOp<string mnemonic, list<Trait> traits = []> : D2M_GenericRegionOp<mnemonic,
  traits #
  [ Pure
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  , DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
  , DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
  ]> {
    let arguments = (ins ConfinedAttr<I64Attr, [IntMinValue<0>]>:$dim);
    let results = (outs Index:$result);
    let assemblyFormat = [{ `(` $dim `)` attr-dict `:` type($result) }];
    let hasFolder = true;

    let builders =
    [
      OpBuilder<(ins "int64_t":$dim),
      [{
        build($_builder, $_state, $_builder.getIndexType(), $_builder.getI64IntegerAttr(dim));
      }]>
    ];
}

def D2M_IterIndexOp : D2M_IndexOp<"iter_index"> {
    let summary = "Iter Index op.";
    let description = [{
      Return the index of the current element in the iteration for the given generic op dimension.
    }];
}

def D2M_CoreIndexOp : D2M_IndexOp<"core_index"> {
    let summary = "Core Index op.";
    let description = [{
      Return the index of this core's coordinate inside the generic op's grid dimension.
    }];
}

//===----------------------------------------------------------------------===//
// D2M Remote Access ops (Used in TTMetal Lowering)
//===----------------------------------------------------------------------===//

def D2M_GetGlobalOperandOp : D2M_GenericRegionOp<"get_global_operand", [Pure]> {
    let summary = "Get global operand op.";
    let description = [{
      Access the global, aka parent generic op, operand at the specified index.

      The following forms are all equivalent, but the latter forms are required
      when moving generic regions to top level func ops in the module.
      ```mlir
      d2m.generic (%arg0, %arg1, %arg2) {
        ^datamovement(%cb0, %cb1, %cb2)
          d2m.dma %arg1, %cb1 // Capture %arg1 from parent scope
      }
      ```

      And:
      ```mlir
      d2m.generic (%arg0, %arg1, %arg2) {
        ^datamovement(%cb0, %cb1, %cb2)
          %operand_arg1 = d2m.get_global_operand 1
          d2m.dma %operand_arg1, %cb0
      }
      ```

      And:
      ```mlir
      func.func @main(...) {
        d2m.generic (%arg0, %arg1, %arg2) { kernel_symbols = [@dm0] }
      }

      func.func private @dm0(%cb0, %cb1, %cb2) {
        %operand_arg1 = d2m.get_global_operand 1
        d2m.dma %operand_arg1, %cb0
      }
      ```
    }];

    let arguments = (ins ConfinedAttr<I64Attr, [IntMinValue<0>]>:$operand_index);
    let results = (outs DeviceMemRef:$result);

    let assemblyFormat = [{ `(` $operand_index `)` attr-dict `:` type($result) }];
}

#endif // TTMLIR_TTMLIR_DIALECT_D2M_D2MGENERICREGIONOPS_TD
