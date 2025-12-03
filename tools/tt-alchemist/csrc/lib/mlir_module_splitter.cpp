// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir_module_splitter.hpp"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include <set>

namespace tt::alchemist {

std::vector<OpInfo> MLIRModuleSplitter::split(mlir::ModuleOp module) {
    // Reset state for new split operation
    operations_.clear();
    funcMap_.clear();
    processedOps_.clear();
    currentOpIndex_ = 0;

    // Build function map
    buildFunctionMap(module);

    // Get main function
    mlir::func::FuncOp mainFunc = getMainFunction(module);
    if (!mainFunc) {
        // If no main function, process all functions
        for (auto& [name, func] : funcMap_) {
            auto ops = extractOpsFromFunc(func);
            operations_.insert(operations_.end(), ops.begin(), ops.end());
        }
    } else {
        // Process from main function
        operations_ = extractOpsFromFunc(mainFunc);
    }

    return operations_;
}

std::vector<OpInfo> MLIRModuleSplitter::extractOpsFromFunc(mlir::func::FuncOp func) {
    std::vector<OpInfo> funcOps;
    std::string funcName = func.getName().str();

    // Walk through all operations in the function
    func.walk([&](mlir::Operation* op) {
        // Skip the function operation itself
        if (mlir::isa<mlir::func::FuncOp>(op)) {
            return;
        }

        // Skip if already processed
        if (processedOps_.count(op)) {
            return;
        }

        // Handle function calls specially
        if (mlir::isa<mlir::func::CallOp>(op)) {
            processCallOp(op, funcName);
        }
        // Process regular operations
        else if (shouldIncludeOp(op)) {
            processOperation(op, funcName);
        }
    });

    return operations_;
}

mlir::func::FuncOp MLIRModuleSplitter::getMainFunction(mlir::ModuleOp module) {
    // First try to find "main" function
    auto mainIt = funcMap_.find("main");
    if (mainIt != funcMap_.end()) {
        return mainIt->second;
    }

    // If no main, look for "forward" or similar entry points
    for (const auto& name : {"forward", "run", "execute"}) {
        auto it = funcMap_.find(name);
        if (it != funcMap_.end()) {
            return it->second;
        }
    }

    // Return nullptr if no entry point found
    return nullptr;
}

bool MLIRModuleSplitter::shouldIncludeOp(mlir::Operation* op) {
    // Include TTNN operations
    if (isTTNNOp(op)) {
        return true;
    }

    // Include TTIR operations
    if (isTTIROp(op)) {
        return true;
    }

    // Exclude func operations, returns, etc.
    if (mlir::isa<mlir::func::ReturnOp>(op) ||
        mlir::isa<mlir::func::FuncOp>(op)) {
        return false;
    }

    // Exclude constant operations for now (they're usually inputs)
    if (op->getName().getStringRef().contains("constant")) {
        return false;
    }

    return false;
}

std::vector<int64_t> MLIRModuleSplitter::extractShape(mlir::Type type) {
    std::vector<int64_t> shape;

    if (auto rankedType = type.dyn_cast<mlir::RankedTensorType>()) {
        for (int64_t dim : rankedType.getShape()) {
            shape.push_back(dim);
        }
    }
    // Handle other tensor types if needed
    else if (auto unrankedType = type.dyn_cast<mlir::UnrankedTensorType>()) {
        // Unranked tensor - shape unknown
        shape.push_back(-1);
    }

    return shape;
}

std::string MLIRModuleSplitter::getOpBaseName(mlir::Operation* op) {
    std::string fullName = op->getName().getStringRef().str();

    // Remove dialect prefix (e.g., "ttnn." or "ttir.")
    size_t dotPos = fullName.find('.');
    if (dotPos != std::string::npos) {
        return fullName.substr(dotPos + 1);
    }

    return fullName;
}

void MLIRModuleSplitter::processOperation(mlir::Operation* op, const std::string& parentFunc) {
    if (processedOps_.count(op)) {
        return;
    }

    OpInfo info = createOpInfo(op, parentFunc);
    operations_.push_back(info);
    processedOps_.insert(op);
}

void MLIRModuleSplitter::processCallOp(mlir::Operation* callOp, const std::string& parentFunc) {
    auto call = mlir::dyn_cast<mlir::func::CallOp>(callOp);
    if (!call) {
        return;
    }

    std::string calleeName = call.getCallee().str();
    auto it = funcMap_.find(calleeName);
    if (it != funcMap_.end()) {
        // Process the called function
        extractOpsFromFunc(it->second);
    }

    // Mark the call operation as processed
    processedOps_.insert(callOp);
}

void MLIRModuleSplitter::buildFunctionMap(mlir::ModuleOp module) {
    // Handle DeviceModuleOp wrapping if present
    mlir::ModuleOp actualModule = unwrapDeviceModule(module);

    // Build map of all functions
    actualModule.walk([&](mlir::func::FuncOp func) {
        funcMap_[func.getName().str()] = func;
    });
}

mlir::ModuleOp MLIRModuleSplitter::unwrapDeviceModule(mlir::ModuleOp module) {
    // Check if module contains a DeviceModuleOp
    mlir::Operation* deviceModuleOp = nullptr;

    for (auto& op : module.getOps()) {
        if (op.getName().getStringRef() == "ttcore.device_module") {
            deviceModuleOp = &op;
            break;
        }
    }

    if (deviceModuleOp) {
        // Extract the inner module from DeviceModuleOp
        // The structure is typically: DeviceModuleOp -> Region -> Block -> ModuleOp
        auto& region = deviceModuleOp->getRegion(0);
        if (!region.empty()) {
            auto& block = region.front();
            for (auto& op : block) {
                if (auto innerModule = mlir::dyn_cast<mlir::ModuleOp>(op)) {
                    return innerModule;
                }
            }
        }
    }

    // No DeviceModuleOp found, return original module
    return module;
}

OpInfo MLIRModuleSplitter::createOpInfo(mlir::Operation* op, const std::string& parentFunc) {
    OpInfo info;
    info.op = op;
    info.opName = op->getName().getStringRef().str();
    info.parentFunc = parentFunc;
    info.opIndex = currentOpIndex_++;
    info.attributes = op->getAttrDictionary();

    // Extract input types and shapes
    for (mlir::Value operand : op->getOperands()) {
        info.inputTypes.push_back(operand.getType());
        info.inputShapes.push_back(extractShape(operand.getType()));
    }

    // Extract output types and shapes
    for (mlir::Value result : op->getResults()) {
        info.outputTypes.push_back(result.getType());
        info.outputShapes.push_back(extractShape(result.getType()));
    }

    return info;
}

bool MLIRModuleSplitter::isTTNNOp(mlir::Operation* op) {
    return op->getName().getDialectNamespace() == "ttnn";
}

bool MLIRModuleSplitter::isTTIROp(mlir::Operation* op) {
    return op->getName().getDialectNamespace() == "ttir";
}

} // namespace tt::alchemist