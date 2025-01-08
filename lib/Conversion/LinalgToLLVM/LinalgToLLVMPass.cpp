// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToLinalg/TTIRToLinalg.h"
#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TT/IR/TT.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"


using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt {

#define GEN_PASS_DEF_CONVERTLINALGTOLLVM
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

struct ConvertLinalgToLLVMPass
    : public mlir::tt::impl::ConvertLinalgToLLVMBase<ConvertLinalgToLLVMPass> {
    void runOnOperation() final {
        // Creating a MLIRContext w/o multithreading is easier than pre-registering all the dialects we need.
        mlir::MLIRContext context;
        context.disableMultithreading();

        DialectRegistry registry;

        // Register required dialects.
        registry.insert<mlir::bufferization::BufferizationDialect>();
        registry.insert<mlir::func::FuncDialect>();
        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::memref::MemRefDialect>();
        registry.insert<mlir::scf::SCFDialect>();
        registry.insert<mlir::LLVM::LLVMDialect>();

        // Append the registry to the context

        arith::registerBufferizableOpInterfaceExternalModels(registry);
        linalg::registerBufferizableOpInterfaceExternalModels(registry);
        scf::registerBufferizableOpInterfaceExternalModels(registry);
        bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
            registry);
        tensor::registerBufferizableOpInterfaceExternalModels(registry);
        vector::registerBufferizableOpInterfaceExternalModels(registry);

        context.appendDialectRegistry(registry);

        llvm::outs() << "Loaded dialects in context:\n";
        for (auto *dialect : context.getLoadedDialects())
            llvm::outs() << "  " << dialect->getNamespace() << "\n";

        // Explicitly load the required dialects.
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
        context.getOrLoadDialect<mlir::scf::SCFDialect>();
        context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
        context.getOrLoadDialect<mlir::tensor::TensorDialect>();
        context.getOrLoadDialect<mlir::vector::VectorDialect>();

        llvm::outs() << "Loaded dialects in context:\n";
        for (auto *dialect : context.getLoadedDialects()) {
            llvm::outs() << "Dialect: " << dialect->getNamespace() << "\n";
            // Print registered operations if possible
        }

        llvm::outs () << "ConvertLinalgToLLVMPass::rOO() 1\n";
        tt::CPUModuleOp cpuModule = getOperation();

        // Create a new temporary ModuleOp

        OpBuilder builder(&context);
        mlir::ModuleOp tempModule = builder.create<mlir::ModuleOp>(cpuModule.getLoc());
        llvm::outs () << "ConvertLinalgToLLVMPass::rOO() 2\n";

        // Move the `CPUModuleOp` into the temporary ModuleOp
        // cpuModule.getOperation()->moveBefore(&*tempModule.getBody()->begin());

        // Create a temporary module to hold the `cpuModule` contents.
        // auto tempModule = builder.create<ModuleOp>(cpuModule.getLoc());

        // Clone operations from `cpuModule` into `tempModule`, and remove originals.
        std::vector<Operation *> opsToErase;
        for (Operation &nestedOp : cpuModule.getBody().getOps()) {
            opsToErase.push_back(&nestedOp);
            if (isa<tt::CPUModuleTerminatorOp>(nestedOp)) {
                // Skip the CPU module terminator
                continue;
            }
            llvm::outs () << "ConvertLinalgToLLVMPass::rOO() 2.1\n";
            tempModule.getBody()->push_back(nestedOp.clone());
            llvm::outs () << "ConvertLinalgToLLVMPass::rOO() 2.2\n";
            
            // nestedOp.erase();
        }

        for (auto *op : opsToErase)
        {
            op->erase();
        }

        llvm::outs () << "ConvertLinalgToLLVMPass::rOO() 3\n";

        // Create a pass manager to apply passes to the temporary module.
        PassManager tempPM(&context);

        // Add the pass pipeline.
        tempPM.addPass(createCanonicalizerPass());
        tempPM.addPass(createConvertElementwiseToLinalgPass());
        tempPM.addPass(createConvertTensorToLinalgPass());

        // Add bufferization passes.
        mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
        bufferizationOptions.bufferizeFunctionBoundaries = true;
        tempPM.addPass(
            mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
        mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
        mlir::bufferization::buildBufferDeallocationPipeline(tempPM,
                                                            deallocationOptions);
        tempPM.addPass(mlir::createBufferizationToMemRefPass());

        // Add lowering passes.
        tempPM.addPass(createConvertLinalgToLoopsPass());
        tempPM.addPass(mlir::memref::createExpandStridedMetadataPass());
        tempPM.addPass(createConvertSCFToCFPass());
        tempPM.addPass(createConvertControlFlowToLLVMPass());
        tempPM.addPass(createArithToLLVMConversionPass());
        tempPM.addPass(createConvertFuncToLLVMPass());
        tempPM.addPass(createFinalizeMemRefToLLVMConversionPass());
        tempPM.addPass(createReconcileUnrealizedCastsPass());

        llvm::outs () << "ConvertLinalgToLLVMPass::rOO() 4\n";

        // Run the pass pipeline on the temporary module.
        if (failed(tempPM.run(tempModule))) {
            signalPassFailure();
            return;
        }

        llvm::outs () << "ConvertLinalgToLLVMPass::rOO() 5\n";

        // Move transformed funcs back to original CPUModuleOp from temp ModuleOp
        for (Operation &transformedOp : tempModule.getBody()->getOperations()) {
            cpuModule.getBody().front().push_back(transformedOp.clone());
        }

        // Replace terminator in our CPUModuleOp.
        OpBuilder cpuModuleBuilder(cpuModule.getBody());
        cpuModuleBuilder.setInsertionPointToEnd(&cpuModule.getBody().front());
        cpuModuleBuilder.create<tt::CPUModuleTerminatorOp>(cpuModule.getLoc());

        // Clean up the temporary ModuleOp now that it's empty.
        tempModule.erase();
        llvm::outs () << "ConvertLinalgToLLVMPass::rOO() 6\n";
    }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<tt::CPUModuleOp>> createConvertLinalgToLLVMPass() {
  return std::make_unique<ConvertLinalgToLLVMPass>();
}

} // namespace mlir::tt
