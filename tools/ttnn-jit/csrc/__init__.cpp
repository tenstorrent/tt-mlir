// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_cache.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinOps.h"
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

  nb::class_<JitCache>(m, "JitCache")
      .def(nb::init<std::size_t>(), nb::rv_policy::take_ownership)
      .def("get", [](JitCache *self, std::string func_name, std::string ir,
                     std::string backend, nb::tuple max_grid_,
                     nb::object tensor_arg_, std::string options = "") {
        std::tuple<uint32_t, uint32_t> max_grid = {
            nb::cast<uint32_t>(max_grid_[0]), nb::cast<uint32_t>(max_grid_[1])};

        py::handle tensor_arg_pybind_obj(tensor_arg_.ptr());
        ::ttnn::Tensor tensor_arg =
            py::cast<::ttnn::Tensor>(tensor_arg_pybind_obj);

        // Parse IR string into MLIR module, since we don't have access to MLIR
        // types. Can remove this once we have tracer.
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

        mlir::Operation *op = unwrap(mlirModuleGetOperation(module));
        auto result = wrapInCapsule(
            self->get(op, JitCacheKey{func_name, backend, max_grid}, tensor_arg,
                      options));

        mlirModuleDestroy(module);
        mlirContextDestroy(ctx);
        return result;
      });
}
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

} // namespace mlir::tt::ttnn::jit
