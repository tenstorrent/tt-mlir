// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/Overrides.h"

namespace mlir::ttmlir::python {

    void populateOverridesModule(py::module &m) {
        
        m.def("get_ptr", [](void* op) {
            return reinterpret_cast<uintptr_t>(op);
        }, py::arg("op").noconvert());

        /*m.def("makeMap", [](const std::string& filepath) {
            std::unordered_map<std::string, int> numMap;
            std::unordered_map<std::string, mlir::Operation*> idMap;
            mlir::MLIRContext ctx;
            mlir::DialectRegistry registry;

            // Load all the Dialects
            registry.insert<tt::TTDialect>();
            registry.insert<ttir::TTIRDialect>();
            ctx.appendDialectRegistry(registry);
            ctx.loadAllAvailableDialects();

            mlir::ParserConfig config(ctx);
            llvm::StringRef filepathRef(filepath);
            
            auto module = mlir::parseSourceFile(filepath, config);
            if (!module)
                throw py::value_error("Invalid TTIR file: " + filepath);
            
            mlir::Operation* op = module.get();

            op -> walk([](mlir::Operation* op) {
                std::string opId = op -> getName();
                opId += std::to_string(numMap[opId]++);
                idMap[opId] = op;
            });

            return idMap;
        });  */

    }

}

