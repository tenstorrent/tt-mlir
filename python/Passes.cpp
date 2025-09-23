// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/InitAllTranslations.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "ttmlir/Bindings/Python/TTMLIRModule.h"
#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"
#include "ttmlir/RegisterAll.h"
#include "ttmlir/Target/Python/PythonEmitter.h"
#include "ttmlir/Target/TTKernel/TTKernelToCpp.h"
#include "ttmlir/Target/TTMetal/TTMetalToFlatbuffer.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

#include <cstdint>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/shared_ptr.h>

// Make Opaque so Casts & Copies don't occur
NB_MAKE_OPAQUE(std::vector<std::pair<std::string, std::string>>);
NB_MAKE_OPAQUE(mlir::tt::GoldenTensor);
NB_MAKE_OPAQUE(std::unordered_map<std::string, mlir::tt::GoldenTensor>);

namespace mlir::tt::ttnn {
void registerTTNNToFlatbuffer();
} // namespace mlir::tt::ttnn

namespace mlir::ttmlir::python {

// Helper function to enable IR dumping for PassManagers created in Python
// bindings
void enableIRDumpingForPM(mlir::PassManager &pm) {
  mlir::tt::MLIRModuleLogger::enableGlobalIRDumping(pm.getContext());
}

void populatePassesModule(nb::module_ &m) {
  // When populating passes, need to first register them

  mlir::tt::registerAllPasses();
  mlir::tt::ttnn::registerTTNNToFlatbuffer();

  m.def(
      "tt_populate_argument_types",
      [](MlirModule module, std::string argument_types_string = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        // Enable IR dumping if requested
        enableIRDumpingForPM(pm);

        // Add the pass with options if provided
        if (!argument_types_string.empty()) {
          std::string passWithOptions =
              "tt-populate-argument-types{argument-types=" +
              argument_types_string + "}";
          if (failed(
                  mlir::parsePassPipeline(passWithOptions, pm, llvm::errs()))) {
            throw std::runtime_error("Failed to parse pass options: " +
                                     passWithOptions);
          }
        } else {
          pm.addPass(mlir::tt::ttcore::createTTPopulateArgumentTypes());
        }

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      nb::arg("module"), nb::arg("argument_types_string") = "");

  m.def(
      "stablehlo_pipeline",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        // Enable IR dumping if requested
        enableIRDumpingForPM(pm);

        const auto *pipeline =
            mlir::PassPipelineInfo::lookup("stablehlo-pipeline");

        std::function<mlir::LogicalResult(const llvm::Twine &)> err_handler =
            [](const llvm::Twine &) { return mlir::failure(); };

        if (mlir::failed(pipeline->addToPipeline(pm, options, err_handler))) {
          throw std::runtime_error("Failed to add pipeline to pass manager");
        }

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      nb::arg("module"), nb::arg("options") = "");

  m.def(
      "ttir_to_ttnn_backend_pipeline",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getName());

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
      nb::arg("module"), nb::arg("options") = "");

  m.def(
      "ttir_to_ttmetal_backend_pipeline",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getName());

        const auto *pipeline =
            mlir::PassPipelineInfo::lookup("ttir-to-ttmetal-pipeline");
        std::function<mlir::LogicalResult(const llvm::Twine &)> err_handler =
            [](const llvm::Twine &) { return mlir::failure(); };
        if (mlir::failed(pipeline->addToPipeline(pm, options, err_handler))) {
          throw std::runtime_error("Failed to add pipeline to pass manager");
        }
        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      nb::arg("module"), nb::arg("options") = "");

  m.def(
      "stablehlo_to_ttir_pipeline",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        // Implicit nesting required to call the stablehlo.composite ->
        // func.call conversion.
        mlir::PassManager pm(moduleOp->getName(),
                             mlir::PassManager::Nesting::Implicit);

        const auto *pipeline =
            mlir::PassPipelineInfo::lookup("stablehlo-to-ttir-pipeline");

        std::function<mlir::LogicalResult(const llvm::Twine &)> err_handler =
            [](const llvm::Twine &) { return mlir::failure(); };

        if (mlir::failed(pipeline->addToPipeline(pm, options, err_handler))) {
          throw std::runtime_error("Failed to add pipeline to pass manager");
        }

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      nb::arg("module"), nb::arg("options") = "");

  m.def(
      "ttir_to_emitpy_pipeline",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getContext());

        const auto *pipeline =
            mlir::PassPipelineInfo::lookup("ttir-to-emitpy-pipeline");

        std::function<mlir::LogicalResult(const llvm::Twine &)> err_handler =
            [](const llvm::Twine &) { return mlir::failure(); };

        if (mlir::failed(pipeline->addToPipeline(pm, options, err_handler))) {
          throw std::runtime_error("Failed to add pipeline to pass manager");
        }

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      nb::arg("module"), nb::arg("options") = "");

  // This binds the vector into an interfaceable object in python and also an
  // opaquely passed one into other functions.
  nb::bind_vector<std::vector<std::pair<std::string, std::string>>>(
      m, "ModuleLog");

  nb::bind_map<std::unordered_map<std::uint32_t, mlir::tt::GoldenTensor>>(
      m, "InnerGoldenMap");

  nb::bind_map<std::unordered_map<
      std::string, std::unordered_map<unsigned int, mlir::tt::GoldenTensor>>>(
      m, "GoldenMap");

  m.def("ttnn_to_flatbuffer_file",
        [](MlirModule module, std::string &filepath,
           const std::unordered_map<
               std::string,
               std::unordered_map<std::uint32_t, mlir::tt::GoldenTensor>>
               &goldenMap = {},
           const std::vector<std::pair<std::string, std::string>> &moduleCache =
               {}) {
          mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));

          // Create a dialect registry and register all necessary dialects and
          // translations
          mlir::DialectRegistry registry;

          // Register all LLVM IR translations
          registerAllToLLVMIRTranslations(registry);

          // Apply the registry to the module's context
          moduleOp->getContext()->appendDialectRegistry(registry);

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
        });

  m.def("ttmetal_to_flatbuffer_file",
        [](MlirModule module, std::string filepath,
           const std::unordered_map<
               std::string,
               std::unordered_map<std::uint32_t, mlir::tt::GoldenTensor>>
               &goldenMap = {},
           const std::vector<std::pair<std::string, std::string>> &moduleCache =
               {}) {
          mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));

          // Create a dialect registry and register all necessary dialects and
          // translations
          mlir::DialectRegistry registry;

          // Register all LLVM IR translations
          registerAllToLLVMIRTranslations(registry);

          // Apply the registry to the module's context
          moduleOp->getContext()->appendDialectRegistry(registry);

          std::error_code fileError;
          llvm::raw_fd_ostream file(filepath, fileError);
          if (fileError) {
            throw std::runtime_error("Failed to open file: " + filepath +
                                     ". Error: " + fileError.message());
          }
          if (mlir::failed(mlir::tt::ttmetal::translateTTMetalToFlatbuffer(
                  moduleOp, file, goldenMap, moduleCache))) {
            throw std::runtime_error("Failed to write flatbuffer to file: " +
                                     filepath);
          }
        });

  m.def(
      "ttkernel_to_cpp",
      [](MlirModule module) {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));

        // Convert to EmitC
        mlir::PassManager pm(moduleOp->getName());
        pm.addPass(mlir::tt::createConvertTTKernelToEmitC());
        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }

        // Translate to C++
        std::string output;
        llvm::raw_string_ostream output_stream(output);
        if (mlir::failed(mlir::tt::ttkernel::translateTopLevelKernelsToCpp(
                mlir::cast<ModuleOp>(moduleOp), output_stream))) {
          throw std::runtime_error("Failed to generate cpp");
        }
        output_stream.flush();
        return output;
      },
      nb::arg("module"));

  m.def(
      "pykernel_compile_pipeline",
      [](MlirModule module, std::string options = "") {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        mlir::PassManager pm(moduleOp->getName());

        const auto *pipeline =
            mlir::PassPipelineInfo::lookup("pykernel-compile-pipeline");

        std::function<mlir::LogicalResult(const llvm::Twine &)> err_handler =
            [](const llvm::Twine &) { return mlir::failure(); };

        if (mlir::failed(pipeline->addToPipeline(pm, options, err_handler))) {
          throw std::runtime_error("Failed to add pipeline to pass manager");
        }

        if (mlir::failed(pm.run(moduleOp))) {
          throw std::runtime_error("Failed to run pass manager");
        }
      },
      nb::arg("module"), nb::arg("options") = "");

  m.def(
      "translate_to_cpp",
      [](MlirModule module) {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        // Translate to C++
        std::string output;
        llvm::raw_string_ostream output_stream(output);
        if (mlir::failed(mlir::emitc::translateToCpp(
                mlir::cast<ModuleOp>(moduleOp), output_stream))) {
          throw std::runtime_error("Failed to generate cpp");
        }
        output_stream.flush();
        return output;
      },
      nb::arg("module"));

  m.def(
      "translate_to_python",
      [](MlirModule module) {
        mlir::Operation *moduleOp = unwrap(mlirModuleGetOperation(module));
        // Translate to Python
        std::string output;
        llvm::raw_string_ostream output_stream(output);
        if (mlir::failed(mlir::tt::emitpy::translateToPython(
                mlir::cast<ModuleOp>(moduleOp), output_stream))) {
          throw std::runtime_error("Failed to generate py");
        }
        output_stream.flush();
        return output;
      },
      nb::arg("module"));

  nb::enum_<::tt::target::DataType>(m, "DataType")
      .value("Float32", ::tt::target::DataType::Float32)
      .value("Float16", ::tt::target::DataType::Float16)
      .value("BFloat16", ::tt::target::DataType::BFloat16)
      .value("Int32", ::tt::target::DataType::Int32)
      .value("UInt8", ::tt::target::DataType::UInt8);

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

  nb::class_<mlir::tt::GoldenTensor>(m, "GoldenTensor")
      .def("__init__",
           [](mlir::tt::GoldenTensor *self, std::string name,
              std::vector<int64_t> shape, std::vector<int64_t> strides,
              ::tt::target::DataType dtype, std::uintptr_t ptr,
              std::size_t dataSize) {
             new (self) mlir::tt::GoldenTensor(
                 name, shape, strides, dtype,
                 std::vector<std::uint8_t>(
                     reinterpret_cast<std::uint8_t *>(ptr),
                     reinterpret_cast<std::uint8_t *>(ptr) + dataSize));
           })
      .def_rw("name", &mlir::tt::GoldenTensor::name)
      .def_rw("shape", &mlir::tt::GoldenTensor::shape)
      .def_rw("strides", &mlir::tt::GoldenTensor::strides)
      .def_rw("dtype", &mlir::tt::GoldenTensor::dtype)
      .def_rw("data", &mlir::tt::GoldenTensor::data);

  // Supposedly no need for shared_ptr holder types anymore, have python take
  // ownership of ModuleLog
  nb::class_<mlir::tt::MLIRModuleLogger>(m, "MLIRModuleLogger")
      .def(nb::init<>(), nb::rv_policy::take_ownership)
      .def(
          "attach_context",
          [](mlir::tt::MLIRModuleLogger *self, MlirContext ctx,
             std::vector<std::string> &passnames_to_cache) {
            self->attachContext(unwrap(ctx), passnames_to_cache);
          },
          nb::arg("ctx"), nb::arg("passnames_to_cache") = nb::list())
      .def_prop_ro("module_log", [](mlir::tt::MLIRModuleLogger *self) {
        return self->moduleCache;
      });
}
} // namespace mlir::ttmlir::python
