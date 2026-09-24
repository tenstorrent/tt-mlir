// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

#include "tt_crank_export.hpp"

namespace tt::crank {

TT_CRANK_API std::string_view version() noexcept;

// Commit hash of the tt-mlir checkout this library was built against.
TT_CRANK_API std::string_view ttmlir_git_hash() noexcept;

// Hash of tracked tt-mlir files under lib/ and include/, including uncommitted
// changes.
TT_CRANK_API std::string_view ttmlir_git_worktree_hash() noexcept;

// Hash of tracked tt-metal files, including uncommitted changes.
TT_CRANK_API std::string_view ttmetal_git_worktree_hash() noexcept;

} // namespace tt::crank
