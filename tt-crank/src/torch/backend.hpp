#pragma once

#include <cstddef>
#include <vector>

#include <ATen/core/ScalarType.h>
#include <ATen/core/Tensor.h>
#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <tt/runtime/types.h>
#include <ttmlir/Target/Common/types_generated.h>

namespace tt::kurbla::torch_backend {

// Compile a finalized TTIR module through the engine pipeline, execute it with
// `inputs`, and return the raw runtime output tensors. Callers wrap the result
// into at::Tensor via `wrap_tt_tensor` — they know the user-facing shape/dtype
// (e.g. the pre-demotion logical type that the runtime descriptor doesn't
// preserve), no parameter plumbing needed here.
//
// On a cache miss (currently: every call) this is O(compile). Each ATen kernel
// finalizes its own module and calls this once.
std::vector<::tt::runtime::Tensor> compile_and_run(mlir::OwningOpRef<mlir::ModuleOp> module_op,
                                                   llvm::ArrayRef<at::Tensor> inputs);

// Installs the PrivateUse1 allocator with c10. Idempotent; safe to call from
// the nanobind module init. Most allocations come through aten::empty
// (overridden in ops/tensor.cpp), which constructs TensorStorages directly;
// the allocator below is a defensive fallback for code paths that bypass that
// override (resize_, manual Storage construction, etc.).
void register_allocator();

// Translate dtypes between torch's `c10::ScalarType` and tt-runtime's
// `tt::target::DataType`.
::tt::target::DataType to_runtime_dtype(c10::ScalarType torch_dtype);
c10::ScalarType to_torch_dtype(::tt::target::DataType runtime_dtype);

// Byte width of a runtime dtype's element. Mirrors c10::elementSize for the
// torch side; we keep both APIs symmetrical so allocation/copy code doesn't
// have to translate to torch and back.
std::size_t element_size(::tt::target::DataType runtime_dtype);

} // namespace tt::kurbla::torch_backend
