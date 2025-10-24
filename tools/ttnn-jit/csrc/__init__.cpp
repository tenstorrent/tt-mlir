// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_cache.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "ttmlir/RegisterAll.h"
#include "ttnn/tensor/tensor.hpp"

#include "tt/runtime/detail/python/nanobind_headers.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#pragma clang diagnostic pop

namespace nb = nanobind;
namespace py = pybind11;

namespace mlir::tt::ttnn::jit {

std::vector<::ttnn::Tensor> convert_args_to_tensors(nb::args args) {
  std::vector<::ttnn::Tensor> tensor_args;
  for (auto arg : args) {
    py::handle arg_pybind_obj(arg.ptr());
    if (py::isinstance<::ttnn::Tensor>(arg_pybind_obj)) {
      tensor_args.push_back(py::cast<::ttnn::Tensor>(arg_pybind_obj));
    } else {
      throw std::runtime_error(
          "Unsupported argument type: expected ttnn.Tensor");
    }
  }
  return tensor_args;
}

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
NB_MODULE(_ttnn_jit, m) {
  m.doc() = "TTNN JIT C++ bindings";

  mlir::tt::registerAllPasses();

  nb::class_<JitCache>(m, "JitCache")
      .def(nb::init<std::size_t>(), nb::rv_policy::take_ownership)
      .def("contains",
           [](JitCache *self, nb::args args) {
             std::vector<::ttnn::Tensor> tensor_args =
                 convert_args_to_tensors(args);
             return self->contains(tensor_args);
           })
      .def("get",
           [](JitCache *self, nb::args args) {
             // Note: Along with tensors, we should allow any other params to be
             // passed into a jit'ed function
             std::vector<::ttnn::Tensor> tensor_args =
                 convert_args_to_tensors(args);

             std::shared_ptr<::tt::runtime::Binary> binary =
                 self->get(tensor_args);

             return *binary;
           })
      .def("compile_and_insert",
           [](JitCache *self, std::string ir, std::string options,
              nb::args args) {
             // Parse IR string into MLIR module; unable to recognize MlirModule
             // from python, without using MLIR cmake macros for python
             // bindings.
             MlirContext ctx = mlirContextCreate();
             mlir::MLIRContext *ctx_ptr = unwrap(ctx);
             mlir::DialectRegistry registry;
             mlir::tt::registerAllDialects(registry);
             mlir::registerAllToLLVMIRTranslations(registry);
             ctx_ptr->appendDialectRegistry(registry);
             MlirModule module = mlirModuleCreateParse(
                 ctx, mlirStringRefCreate(ir.c_str(), ir.size()));
             if (mlirModuleIsNull(module)) {
               mlirModuleDestroy(module);
               mlirContextDestroy(ctx);
               throw std::runtime_error("Failed to parse IR string");
             }
             std::vector<::ttnn::Tensor> tensor_args =
                 convert_args_to_tensors(args);

             mlir::Operation *op = unwrap(mlirModuleGetOperation(module));
             JitCacheEntry binary =
                 self->compile_and_insert(op, tensor_args, options);

             mlirModuleDestroy(module);
             mlirContextDestroy(ctx);
             return *binary;
           })
      .def("num_entries", &JitCache::num_entries);
}
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
} // namespace mlir::tt::ttnn::jit
