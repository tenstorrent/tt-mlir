#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Align with coding style guidelines outlined here: docs/src/coding-guidelines.md. Namely:
# Only the [standard lib header includes](https://en.cppreference.com/w/cpp/header) should use <> whereas all the others should use quotes "".

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root" || exit 1

# Get list of staged files that are C++ source or header files. Store each filename as a separate array element to preserve the filename.
mapfile -t staged_files < <(git diff --cached --name-only --diff-filter=ACMR | grep -E '\.(cpp|h)$')

if [ ${#staged_files[@]} -eq 0 ]; then
  echo "No C++ files staged for commit."
  exit 0
fi

# Check each staged file for incorrect include style. Prevent word splitting and process the entire filename.
files_with_issues=()
for rel_path in "${staged_files[@]}"; do
  abs_path="$repo_root/$rel_path"
  if [ -f "$abs_path" ] && grep -q "#include <\(llvm\|mlir\)/" "$abs_path"; then
    files_with_issues+=("$rel_path")
  fi
done

# If no issues found, exit successfully.
if [ ${#files_with_issues[@]} -eq 0 ]; then
  echo "No include style issues found."
  exit 0
fi

echo "Found ${#files_with_issues[@]} files with incorrect include style:"
for file in "${files_with_issues[@]}"; do
  echo "  $file"
done

for rel_path in "${files_with_issues[@]}"; do
  abs_path="$repo_root/$rel_path"
  echo "Fixing $rel_path"
    # Replace <llvm/...> or <mlir/...> with "llvm/..." or "mlir/..." using a regex.
  sed -i 's/#include <\(llvm\|mlir\)\/\([^>]*\)>/#include "\1\/\2"/g' "$abs_path"
  git add "$rel_path"
done

echo "Fixed include style issues and staged the changes. Refer to docs/src/coding-guidelines.md for more details."
exit 0
