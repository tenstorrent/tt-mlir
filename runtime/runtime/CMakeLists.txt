# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Copy the contents of build/runtime/python into build/runtime/runtime and copy golden.py
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_ttmlir_runtime_copied
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_BINARY_DIR}/runtime/python
    ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/golden.py
    ${CMAKE_CURRENT_BINARY_DIR}/golden.py
  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/_ttmlir_runtime_copied
  DEPENDS _ttmlir_runtime
  COMMENT "Copying Python runtime contents to runtime/runtime and copying golden.py"
)

# Create a custom target to ensure the copy operation is executed
add_custom_target(copy_runtime_files ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/_ttmlir_runtime_copied
)

# Add dependency on the runtime/python CMake build
add_dependencies(copy_runtime_files _ttmlir_runtime)
