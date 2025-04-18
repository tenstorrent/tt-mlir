#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Align with coding style guidelines outlined here: docs/src/coding-guidelines.md. Namely:
# Only the [standard lib header includes](https://en.cppreference.com/w/cpp/header) should use <> whereas all the others should use quotes "".

# Get list of staged files that are C++ source or header files.
staged_files=$(git diff --cached --name-only --diff-filter=ACMR | grep -E '\.(cpp|h)$')

if [ -z "$staged_files" ]; then
  echo "No C++ files staged for commit."
  exit 0
fi

# Check each staged file for incorrect include style.
files_with_issues=()
for file in $staged_files; do
  if grep -q "#include <\(llvm\|mlir\)/" "$file"; then
    files_with_issues+=("$file")
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

for file in "${files_with_issues[@]}"; do
  echo "Fixing $file"
  # Replace <llvm/ with "llvm/
  sed -i 's/#include <llvm\//#include "llvm\//g' "$file"
  # Replace <mlir/ with "mlir/
  sed -i 's/#include <mlir\//#include "mlir\//g' "$file"
  # Replace > at the end of LLVM/MLIR includes with "
  sed -i 's/\.h>/\.h"/g' "$file"
  # Stage the fixed file.
  git add "$file"
done

echo "Fixed include style issues and staged the changes. Refer to docs/src/coding-guidelines.md for more details."
exit 0
