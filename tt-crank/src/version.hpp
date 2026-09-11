#pragma once

#include <string_view>

#include "tt_kurbla_export.hpp"

namespace tt::kurbla {

TT_KURBLA_API std::string_view version() noexcept;

// Commit hash of the tt-mlir submodule this library was built against.
TT_KURBLA_API std::string_view ttmlir_git_hash() noexcept;

// tt-mlir source identity combining the commit hash with any uncommitted
// changes: equals ttmlir_git_hash() when the submodule is clean, and a SHA-1 of
// (commit + local diff) when there are local edits. Use it to key a compile
// cache so stale artifacts aren't reused after editing tt-mlir.
TT_KURBLA_API std::string_view ttmlir_git_worktree_hash() noexcept;

} // namespace tt::kurbla
