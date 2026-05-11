#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <tt/runtime/types.h>

#include "engine/compile.hpp"
#include "tt_kurbla_export.hpp"

namespace tt::kurbla {

// Binds inputs to a shared CompiledProgram and runs it. Holds the program via
// shared_ptr so multiple payloads can target the same compiled binary with
// different inputs. Input layout conversion happens at bind_tensor time, not
// at run() time, so a "bind once, run many" loop only re-toLayouts slots that
// get rebound between runs. Not thread-safe in v1 — the runtime device
// singleton has no internal locking.
class TT_KURBLA_API ExecutionPayload {
public:
    explicit ExecutionPayload(std::shared_ptr<CompiledProgram> program, std::uint32_t program_index = 0);

    ~ExecutionPayload();
    ExecutionPayload(const ExecutionPayload &) = delete;
    ExecutionPayload &operator=(const ExecutionPayload &) = delete;
    ExecutionPayload(ExecutionPayload &&) noexcept;
    ExecutionPayload &operator=(ExecutionPayload &&) noexcept;

    const std::shared_ptr<CompiledProgram> &compiled_program() const;
    std::uint32_t program_index() const;

    // Calls tt::runtime::toLayout immediately and stores the device tensor in
    // slot `index`. Rebinding overwrites. Throws InputBindingError on
    // out-of-range index or shape/dtype mismatch against
    // `compiled_program()->input_descs(program_index())[index]`.
    void bind_tensor(const tt::runtime::Tensor &tensor, std::uint32_t index);

    // Returns device-resident outputs; caller does toHost. Throws
    // InputBindingError if any slot is unbound, DeviceError on submit failure.
    std::vector<tt::runtime::Tensor> run();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// TT_KURBLA_API on the exception types so their typeinfo crosses the .so
// boundary — see CompileError in compile.hpp for the same rationale.
class TT_KURBLA_API ExecuteError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class TT_KURBLA_API InputBindingError : public ExecuteError {
public:
    using ExecuteError::ExecuteError;
};

class TT_KURBLA_API DeviceError : public ExecuteError {
public:
    using ExecuteError::ExecuteError;
};

} // namespace tt::kurbla
