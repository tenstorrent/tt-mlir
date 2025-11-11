// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSTYPES_TD
#define TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.td"
include "ttmlir/Dialect/D2M/IR/D2MBase.td"

class D2M_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<D2M_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def D2M_MemTx : D2M_Type<"MemTx", "mem_tx"> {
    let summary = "D2M memory transaction type.";
    let description = "Memory transaction returned by dma op, used to wait for completion.";
}

def D2M_Semaphore : D2M_Type<"Semaphore", "semaphore"> {
    let summary = "D2M semaphore type.";
    let description = "Semaphore primitive type used with semaphore ops to synchronize cores.";
}

def D2M_CB : D2M_Type<"CB", "cb", [ShapedTypeInterface, Bufferization_TensorLikeTypeInterface, Bufferization_BufferLikeTypeInterface]> {
    let summary = "D2M circular buffer type.";
    let description = [{
        Circular buffer type that wraps a memref for non-pure semantics.
        Used in d2m.generic region arguments to represent post-bufferization
        circular buffer semantics instead of raw memref types.

        This type also supports pure tensor semantics (hence ShapedType, which
        allows RankedTensorType) to enable fully formed generic ops that come
        through the pipeline via DSL paths.
    }];
    let parameters = (ins
      "ShapedType":$underlying,
      OptionalParameter<"ttcore::MemorySpace">:$memorySpace
    );

    let assemblyFormat = "`<` $underlying `,`$memorySpace `>`";

    let skipDefaultBuilders = 1;
    let builders = [
      TypeBuilder<(ins "mlir::ShapedType":$underlying), [{
        return $_get($_ctxt, underlying, ttcore::MemorySpace::DeviceL1);
      }]>,
      TypeBuilderWithInferredContext<(ins "mlir::ShapedType":$underlying), [{
        return $_get(underlying.getContext(), underlying, ttcore::MemorySpace::DeviceL1);
      }]>,
      TypeBuilder<(ins "mlir::ShapedType":$underlying, "ttcore::MemorySpace":$memorySpace), [{
        return $_get($_ctxt, underlying, memorySpace);
      }]>, 
      TypeBuilderWithInferredContext<(ins "mlir::ShapedType":$underlying, "ttcore::MemorySpace":$memorySpace), [{
          return $_get(underlying.getContext(), underlying, memorySpace);
      }]>
    ];

    let extraClassDeclaration = [{
      mlir::ShapedType cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape,
                                 mlir::Type elementType) {
        auto clone = getUnderlying().cloneWith(shape, elementType);
        return mlir::cast<mlir::ShapedType>(get(clone));
      }

      template <typename T>
      T getUnderlyingAs() {
        return mlir::cast<T>(getUnderlying());
      }

      mlir::Type getElementType() {
        return getUnderlying().getElementType();
      }

      bool hasRank() {
        return getUnderlying().hasRank();
      }

      llvm::ArrayRef<int64_t> getShape() {
        return getUnderlying().getShape();
      }

      bool hasTensorType() {
        return mlir::isa<RankedTensorType>(getUnderlying());
      }

      bool hasMemRefType() {
        return mlir::isa<MemRefType>(getUnderlying());
      }

      mlir::FailureOr<mlir::bufferization::BufferLikeType> getBufferType(
        const mlir::bufferization::BufferizationOptions &,
        llvm::function_ref<mlir::InFlightDiagnostic()>) {
        auto underlying = MemRefType::get(getShape(), getElementType(), nullptr,
            ttcore::MemorySpaceAttr::get(getContext(), ttcore::MemorySpace::DeviceRiscvL1));
        return mlir::cast<bufferization::BufferLikeType>(get(
                mlir::cast<ShapedType>(underlying)));
      }

      mlir::LogicalResult verifyCompatibleBufferType(
        mlir::bufferization::BufferLikeType bufferType,
        llvm::function_ref<mlir::InFlightDiagnostic()>) {
        auto shapedType = mlir::dyn_cast<ShapedType>(bufferType);
        return success(shapedType &&
                       getElementType() == shapedType.getElementType() &&
                       getShape() == shapedType.getShape());
      }
    }];
}

#endif  // TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSTYPES_TD
