// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <tt/runtime/types.h>

#include "engine/compile.hpp"
#include "tt_kurbla_export.hpp"

namespace tt::kurbla {

// Binds inputs to a CompiledProgram and runs it. Holds a non-owning reference to
// the program (owned by the compile cache) so multiple payloads can target the
// same compiled binary with different inputs. Input layout conversion happens at
// bind_tensor time, not at run() time, so a "bind once, run many" loop only
// re-toLayouts slots that get rebound between runs. Not thread-safe in v1 — the
// runtime device singleton has no internal locking.
class TT_KURBLA_API ExecutionPayload {
public:
    explicit ExecutionPayload(CompiledProgram &program);

    ~ExecutionPayload();
    ExecutionPayload(const ExecutionPayload &) = delete;
    ExecutionPayload &operator=(const ExecutionPayload &) = delete;
    ExecutionPayload(ExecutionPayload &&) noexcept;
    ExecutionPayload &operator=(ExecutionPayload &&) noexcept;

    CompiledProgram &compiled_program() const;

    // Calls tt::runtime::toLayout if tensor is in wrong layout and stores the device tensor in
    // slot `index`. Returns bound tensor.
    tt::runtime::Tensor bind_tensor(tt::runtime::Tensor tensor, std::uint32_t index);

    // Returns device-resident outputs; caller does toHost.
    std::vector<tt::runtime::Tensor> run();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tt::kurbla
