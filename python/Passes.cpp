// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/InitAllTranslations.h"
#include "ttmlir/Bindings/Python/TTMLIRModule.h"
#include "ttmlir/RegisterAll.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);

namespace mlir::ttmlir::python {

void populatePassesModule(py::module &m) {
  // When populating passes, need to first register them

  mlir::tt::registerAllPasses();
  mlir::registerAllTranslations();

  m.def("ttir_to_ttnn_backend_pipeline", [](MlirModule module) {
    mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
    mlir::PassManager pm(moduleOp->getName());

    mlir::DialectRegistry registry;
    mlir::tt::registerAllDialects(registry);
    mlir::MLIRContext *ctx = unwrap(mlirModuleGetContext(module));
    ctx->appendDialectRegistry(registry);

    const auto pipeline =
        mlir::PassPipelineInfo::lookup("ttir-to-ttnn-backend-pipeline");

    std::string options = "";

    mlir::function_ref<mlir::LogicalResult(const llvm::Twine &)> err_handler =
        [](const llvm::Twine &loc) { return mlir::failure(); };

    if (mlir::failed(pipeline->addToPipeline(pm, options, err_handler))) {
      throw std::runtime_error("Failed to add pipeline to pass manager");
    }

    if (mlir::failed(pm.run(moduleOp))) {
      throw std::runtime_error("Failed to run pass manager");
    }
  });

  py::class_<std::shared_ptr<void>>(m, "SharedVoidPtr")
      .def(py::init<>())
      .def("from_ttnn", [](std::shared_ptr<void> data, MlirModule module) {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        data = mlir::tt::ttnn::ttnnToFlatbuffer(moduleOp);
      });
}

} // namespace mlir::ttmlir::python
