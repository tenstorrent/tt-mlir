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
static nb::capsule wrapInCapsule(std::shared_ptr<void> underlying) {
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  std::shared_ptr<void> *binary = static_cast<std::shared_ptr<void> *>(
      std::malloc(sizeof(std::shared_ptr<void>)));
  assert(binary);
  *binary = underlying;
  return nb::capsule(
      static_cast<void *>(
          binary), // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      +[](void *data) noexcept { std::free(data); });
}

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
NB_MODULE(_ttnn_jit, m) {
  m.doc() = "TTNN JIT C++ bindings";

  mlir::tt::registerAllPasses();

  nb::class_<JitCache>(m, "JitCache")
      .def(nb::init<std::size_t>(), nb::rv_policy::take_ownership)
      .def("get",
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

             // Note: Along with tensors, we should allow any other params to be
             // passed into a jit'ed function
             std::vector<::ttnn::Tensor> tensor_args;
             for (auto arg : args) {
               py::handle arg_pybind_obj(arg.ptr());
               if (py::isinstance<::ttnn::Tensor>(arg_pybind_obj)) {
                 tensor_args.push_back(
                     py::cast<::ttnn::Tensor>(arg_pybind_obj));
               } else {
                 throw std::runtime_error(
                     "Unsupported argument type: expected ttnn.Tensor");
               }
             }

             mlir::Operation *op = unwrap(mlirModuleGetOperation(module));
             auto result = wrapInCapsule(self->get(op, tensor_args, options));

             mlirModuleDestroy(module);
             mlirContextDestroy(ctx);
             return result;
           })
      .def("cache_hits", &JitCache::get_cache_hits);
}
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
} // namespace mlir::tt::ttnn::jit
