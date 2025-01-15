// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/InitAllTranslations.h"
#include "ttmlir/Bindings/Python/TTMLIRModule.h"
#include "ttmlir/RegisterAll.h"
#include "ttmlir/Target/TTMetal/TTMetalToFlatbuffer.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"
#include <cstdint>

PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);

namespace mlir::tt::ttnn {
void registerTTNNToFlatbuffer();
} // namespace mlir::tt::ttnn

namespace mlir::ttmlir::python {

void populatePassesModule(py::module &m) {
  // When populating passes, need to first register them

  mlir::tt::registerAllPasses();
  mlir::tt::ttnn::registerTTNNToFlatbuffer();

  m.def(
      "ttnn_decompose_layouts",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createTTNNDecomposeLayoutsHelperFromString(pm, options);

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  m.def(
      "ttnn_deallocate",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createTTNNDeallocateHelperFromString(pm, options);

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  m.def(
      "ttir_to_ttir_decomposition_pass",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createTTIRToTTIRDecompositionPassFromString(pm, options);

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  m.def(
      "inliner_pass",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createInlinerPassFromString(pm, options);

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  m.def(
      "ttir_load_system_desc",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createTTIRLoadSystemDescFromString(pm, options);

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  m.def(
      "ttir_implicit_device",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createTTIRImplicitDeviceFromString(pm, options);

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  m.def(
      "ttnn_layout",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createTTNNLayoutFromString(pm, options);

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  m.def(
      "convert_ttir_to_ttnn_pass",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createConvertTTIRToTTNNPassFromString(pm, options);

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  m.def(
      "remove_dead_values_pass",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createRemoveDeadValuesPassFromString(pm, options);

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  m.def(
      "ttnn_pipeline_lowering_passes",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createTTNNPipelineLoweringPassesFromString(pm, options);

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  m.def(
      "ttnn_pipeline_ttir_passes",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createTTNNPipelineTTIRPassesFromString(pm, options);

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  m.def(
      "ttnn_workarounds",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createTTNNWorkaroundsFromString(pm, options);

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  m.def(
      "canonicalizer_pass",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createCanonicalizerPassFromString(pm, options);

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  m.def(
      "ttnn_pipeline_workaround_pass",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createTTNNPipelineWorkaroundPassFromString(pm, options);

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  m.def(
      "ttnn_optimizer",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createTTNNOptimizerFromString(pm, options);

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  m.def(
      "ttir_to_ttnn_backend_pipeline",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getName());

        mlir::DialectRegistry registry;
        mlir::tt::registerAllDialects(registry);
        mlir::tt::registerAllExtensions(registry);
        mlir::MLIRContext *ctx = unwrap(mlirModuleGetContext(module));
        ctx->appendDialectRegistry(registry);

        const auto *pipeline =
            mlir::PassPipelineInfo::lookup("ttir-to-ttnn-backend-pipeline");

        std::function<mlir::LogicalResult(const llvm::Twine &)> err_handler =
            [](const llvm::Twine &) { return mlir::failure(); };

        if (mlir::failed(pipeline->addToPipeline(pm, options, err_handler))) {
          throw std::runtime_error("Failed to add pipeline to pass manager");
        }

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  m.def(
      "ttir_to_ttmetal_backend_pipeline",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getName());
        mlir::DialectRegistry registry;
        mlir::tt::registerAllDialects(registry);
        mlir::tt::registerAllExtensions(registry);
        mlir::MLIRContext *ctx = unwrap(mlirModuleGetContext(module));
        ctx->appendDialectRegistry(registry);
        const auto *pipeline =
            mlir::PassPipelineInfo::lookup("ttir-to-ttmetal-backend-pipeline");
        std::function<mlir::LogicalResult(const llvm::Twine &)> err_handler =
            [](const llvm::Twine &) { return mlir::failure(); };
        if (mlir::failed(pipeline->addToPipeline(pm, options, err_handler))) {
          throw std::runtime_error("Failed to add pipeline to pass manager");
        }
        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  py::class_<std::shared_ptr<void>>(m, "SharedVoidPtr")
      .def(py::init<>())
      .def("from_ttnn", [](std::shared_ptr<void> data, MlirModule module) {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        data = mlir::tt::ttnn::ttnnToFlatbuffer(moduleOp);
      });

  m.def("ttnn_to_flatbuffer_binary", [](MlirModule module) {
    // NOLINTBEGIN
    mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
    std::shared_ptr<void> *binary = new std::shared_ptr<void>();
    *binary = mlir::tt::ttnn::ttnnToFlatbuffer(moduleOp);
    return py::capsule((void *)binary, [](void *data) {
      std::shared_ptr<void> *bin = static_cast<std::shared_ptr<void> *>(data);
      delete bin;
    });
    // NOLINTEND
  });

  m.def("ttnn_to_flatbuffer_file",
        [](MlirModule module, std::string &filepath,
           std::unordered_map<std::string, mlir::tt::GoldenTensor> goldenMap) {
          mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));

          std::error_code fileError;
          llvm::raw_fd_ostream file(filepath, fileError);

          if (fileError) {
            throw std::runtime_error("Failed to open file: " + filepath +
                                     ". Error: " + fileError.message());
          }

          if (mlir::failed(mlir::tt::ttnn::translateTTNNToFlatbuffer(
                  moduleOp, file, goldenMap))) {
            throw std::runtime_error("Failed to write flatbuffer to file: " +
                                     filepath);
          }
        });

  m.def("ttmetal_to_flatbuffer_file",
        [](MlirModule module, std::string &filepath,
           std::unordered_map<std::string, mlir::tt::GoldenTensor> goldenMap) {
          mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
          std::error_code fileError;
          llvm::raw_fd_ostream file(filepath, fileError);
          if (fileError) {
            throw std::runtime_error("Failed to open file: " + filepath +
                                     ". Error: " + fileError.message());
          }
          if (mlir::failed(mlir::tt::ttmetal::translateTTMetalToFlatbuffer(
                  moduleOp, file, goldenMap))) {
            throw std::runtime_error("Failed to write flatbuffer to file: " +
                                     filepath);
          }
        });

  py::enum_<::tt::target::DataType>(m, "DataType")
      .value("Float32", ::tt::target::DataType::Float32)
      .value("Float16", ::tt::target::DataType::Float16);

  py::class_<mlir::tt::GoldenTensor>(m, "GoldenTensor")
      .def(py::init<std::string, std::vector<int64_t>, std::vector<int64_t>,
                    ::tt::target::DataType, std::uint8_t *>())
      .def_readwrite("name", &mlir::tt::GoldenTensor::name)
      .def_readwrite("shape", &mlir::tt::GoldenTensor::shape)
      .def_readwrite("strides", &mlir::tt::GoldenTensor::strides)
      .def_readwrite("dtype", &mlir::tt::GoldenTensor::dtype)
      .def_readwrite("data", &mlir::tt::GoldenTensor::data);

  m.def("create_golden_tensor",
        [](std::string name, std::vector<int64_t> shape,
           std::vector<int64_t> strides, ::tt::target::DataType dtype,
           std::uintptr_t ptr) {
          return mlir::tt::GoldenTensor(name, shape, strides, dtype,
                                        reinterpret_cast<std::uint8_t *>(ptr));
        });
}

} // namespace mlir::ttmlir::python
