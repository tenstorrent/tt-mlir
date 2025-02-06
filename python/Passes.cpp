// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/Passes.h"
#include "mlir/InitAllTranslations.h"
#include "ttmlir/Bindings/Python/TTMLIRModule.h"
#include "ttmlir/RegisterAll.h"
#include "ttmlir/Target/TTMetal/TTMetalToFlatbuffer.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"
#include <cstdint>
#include <pybind11/stl_bind.h>

// Make Opaque so Casts & Copies don't occur
PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
PYBIND11_MAKE_OPAQUE(std::vector<std::pair<std::string, std::string>>);

namespace mlir::tt::ttnn {
void registerTTNNToFlatbuffer();
} // namespace mlir::tt::ttnn

namespace mlir::ttmlir::python {

void populatePassesModule(py::module &m) {
  // When populating passes, need to first register them

  mlir::tt::registerAllPasses();
  mlir::tt::ttnn::registerTTNNToFlatbuffer();

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
      "ttnn_pipeline_analysis_passes",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createTTNNPipelineAnalysisPassesFromString(pm, options);

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
      "ttnn_pipeline_layout_decomposition_pass",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createTTNNPipelineLayoutDecompositionPassFromString(pm,
                                                                      options);

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      py::arg("module"), py::arg("options") = "");

  m.def(
      "ttnn_pipeline_dealloc_pass",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        tt::ttnn::createTTNNPipelineDeallocPassFromString(pm, options);

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

  // This binds the vector into an interfaceable object in python and also an
  // opaquely passed one into other functions.
  py::bind_vector<std::vector<std::pair<std::string, std::string>>>(
      m, "ModuleLog");

  m.def(
      "ttnn_to_flatbuffer_file",
      [](MlirModule module, std::string &filepath,
         const std::unordered_map<std::string, mlir::tt::GoldenTensor>
             &goldenMap = {},
         const std::vector<std::pair<std::string, std::string>> &moduleCache =
             {}) {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));

        std::error_code fileError;
        llvm::raw_fd_ostream file(filepath, fileError);

        if (fileError) {
          throw std::runtime_error("Failed to open file: " + filepath +
                                   ". Error: " + fileError.message());
        }

        if (mlir::failed(mlir::tt::ttnn::translateTTNNToFlatbuffer(
                moduleOp, file, goldenMap, moduleCache))) {
          throw std::runtime_error("Failed to write flatbuffer to file: " +
                                   filepath);
        }
      },
      py::arg("module"), py::arg("filepath"), py::arg("goldenMap") = py::dict(),
      py::arg("moduleCache") =
          std::vector<std::pair<std::string, std::string>>());

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

  m.def("lookup_dtype", [](std::string enumName) {
    // Function to return the enum value based on the name.
    const uint16_t minI = static_cast<uint16_t>(::tt::target::DataType::MIN),
                   maxI = static_cast<uint16_t>(::tt::target::DataType::MAX);
    for (int i = minI; i <= maxI; i++) {
      auto dtype = static_cast<::tt::target::DataType>(i);
      std::string currEnumName = EnumNameDataType(dtype);
      if (currEnumName == enumName) {
        return dtype;
      }
    }
    // Not found so return the MIN value (Float32) by Default
    return ::tt::target::DataType::MIN;
  });

  // Preserve the Data by holding it in a SharedPtr.
  py::class_<mlir::tt::GoldenTensor, std::shared_ptr<mlir::tt::GoldenTensor>>(
      m, "GoldenTensor")
      .def(py::init([](std::string name, std::vector<int64_t> shape,
                       std::vector<int64_t> strides,
                       ::tt::target::DataType dtype, std::uintptr_t ptr,
                       std::size_t dataSize) {
        // Create Golden Tensor and move ownership to GoldenTensor
        std::vector<std::uint8_t> dataVec(dataSize);
        std::memcpy(dataVec.data(), reinterpret_cast<std::uint8_t *>(ptr),
                    dataSize);

        return std::make_shared<mlir::tt::GoldenTensor>(
            name, shape, strides, dtype, std::move(dataVec));
      }))
      .def_readwrite("name", &mlir::tt::GoldenTensor::name)
      .def_readwrite("shape", &mlir::tt::GoldenTensor::shape)
      .def_readwrite("strides", &mlir::tt::GoldenTensor::strides)
      .def_readwrite("dtype", &mlir::tt::GoldenTensor::dtype)
      .def_readwrite("data", &mlir::tt::GoldenTensor::data);

  py::class_<mlir::tt::MLIRModuleLogger,
             std::shared_ptr<mlir::tt::MLIRModuleLogger>>(m, "MLIRModuleLogger")
      .def(py::init<>())
      .def(
          "attach_context",
          [](std::shared_ptr<mlir::tt::MLIRModuleLogger> &self, MlirContext ctx,
             std::vector<std::string> &passnames_to_cache) {
            self->attachContext(unwrap(ctx), passnames_to_cache);
          },
          py::arg("ctx"), py::arg("passnames_to_cache") = py::list())
      .def_property_readonly(
          "module_log", [](std::shared_ptr<mlir::tt::MLIRModuleLogger> &self) {
            return self->moduleCache;
          });
}
} // namespace mlir::ttmlir::python
