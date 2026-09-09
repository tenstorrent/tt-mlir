// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_INTERFACES_TTNNOPMODELINTERFACE_TD
#define TTMLIR_DIALECT_TTNN_INTERFACES_TTNNOPMODELINTERFACE_TD

include "mlir/IR/OpBase.td"

// TODO(odjuricic): support ops with multiple outputs
def TTNN_OpModelInterface : OpInterface<"OpModel"> {
    let description = [{
        Interface to access TTNN op model methods.
    }];
    let cppNamespace = "::mlir::tt::ttnn";
    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Measures and returns the runtime of the op in nanoseconds by running it on the device.
                If the op is illegal or execution fails, returns an Error with a string describing the failure.
            }],
            /*retTy=*/"llvm::Expected<size_t>",
            /*methodName=*/"getOpRuntime",
            /*args=*/(ins "const std::vector<TTNNLayoutAttr>&":$inputs, "const OpConfig&":$config),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return llvm::createStringError(\"Not Implemented\");"
        >,
        InterfaceMethod<
            /*desc=*/[{
                Checks if the op is legal for the given input/output layout. If it is, returns a tuple of three values:**
                   - The first value is the CB L1 peak allocation in bytes.
                   - The second value is the Tensor L1 peak allocation in bytes.
                   - The third value is the Output L1 buffer allocation in bytes.
                   - The fourth value is the actual TTNNLayoutAttr of the output tensor, as returned by the operation. For
                     some operations this may be different from the requested output layout
                If the op is illegal, returns an Error with a string describing the failure.

                `liveRecords` selects the query flavour and is TRI-STATE -- do not
                collapse it to "is the record set empty":
                   - std::nullopt: stateless query. Nothing is allocated
                     (NO_DISPATCH), `outputAllocations` comes back empty, and the
                     result is cached by the op-model constraint cache.
                   - engaged, EVEN IF EMPTY: stateful query. The records describe
                     the allocations already live in L1; the op is evaluated on
                     top of that state by a query that really allocates, reports
                     per-output `outputAllocations`, and bypasses the cache (the
                     live set is deliberately not part of the cache key). An
                     empty-but-engaged live set is a real state the L1 spill path
                     depends on: it is how the record set bootstraps off the
                     first op, so treating it as stateless would silently degrade
                     the whole feature.
                Use the 2-argument overload below for a plain stateless query.
            }],
            /*retTy=*/"llvm::Expected<mlir::tt::ttnn::op_model::OpConstraints>",
            /*methodName=*/"getOpConstraints",
            /*args=*/(ins "const std::vector<TTNNLayoutAttr>&":$inputs, "const OpConfig&":$config, "std::optional<llvm::ArrayRef<mlir::tt::ttnn::op_model::OpModelAllocationRecord>>":$liveRecords),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return llvm::createStringError(\"Not Implemented\");"
        >,
        ];

    // Declared on the interface class only (NOT extraSharedClassDeclaration):
    // ODS declares the 3-argument getOpConstraints inside every implementing op
    // class, which would hide a 2-argument overload inherited from the Trait.
    // Every caller of the convenience form goes through an interface handle.
    let extraClassDeclaration = [{
        // Convenience overload for a stateless constraint query; equivalent to
        // passing std::nullopt.
        llvm::Expected<mlir::tt::ttnn::op_model::OpConstraints>
        getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const OpConfig &config) {
          return getOpConstraints(inputs, config, /*liveRecords=*/std::nullopt);
        }
    }];
}

#endif // TTMLIR_DIALECT_TTNN_INTERFACES_TTNNOPMODELINTERFACE_TD
