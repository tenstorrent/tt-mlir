add_custom_target(copy-docs-dir
  COMMAND
    cp -r ${CMAKE_CURRENT_SOURCE_DIR}/* ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(autogen-summary
  COMMAND
    python ${CMAKE_CURRENT_SOURCE_DIR}/autogen-summary.py
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS
    copy-docs-dir mlir-doc
)

add_custom_command(
  OUTPUT /opt/ttmlir-toolchain/src/shardy/shardy/dialect/sdy/transforms/export/Passes.td
  COMMAND ${CMAKE_COMMAND} -E create_symlink passes.td Passes.td
  WORKING_DIRECTORY /opt/ttmlir-toolchain/src/shardy/shardy/dialect/sdy/transforms/export
  DEPENDS /opt/ttmlir-toolchain/src/shardy/shardy/dialect/sdy/transforms/export/passes.td
  COMMENT "Creating symlink Passes.td -> passes.td in shardy export"
)

add_custom_target(SdyExportPassesSymlink ALL
  DEPENDS /opt/ttmlir-toolchain/src/shardy/shardy/dialect/sdy/transforms/export/Passes.td
)

add_custom_command(
  OUTPUT /opt/ttmlir-toolchain/src/shardy/shardy/dialect/sdy/transforms/import/Passes.td
  COMMAND ${CMAKE_COMMAND} -E create_symlink passes.td Passes.td
  WORKING_DIRECTORY /opt/ttmlir-toolchain/src/shardy/shardy/dialect/sdy/transforms/import
  DEPENDS /opt/ttmlir-toolchain/src/shardy/shardy/dialect/sdy/transforms/import/passes.td
  COMMENT "Creating symlink Passes.td -> passes.td in shardy import"
)

add_custom_target(SdyImportPassesSymlink ALL
  DEPENDS /opt/ttmlir-toolchain/src/shardy/shardy/dialect/sdy/transforms/import/Passes.td
)

add_custom_command(
  OUTPUT /opt/ttmlir-toolchain/src/shardy/shardy/dialect/sdy/transforms/propagation/Passes.td
  COMMAND ${CMAKE_COMMAND} -E create_symlink passes.td Passes.td
  WORKING_DIRECTORY /opt/ttmlir-toolchain/src/shardy/shardy/dialect/sdy/transforms/propagation
  DEPENDS /opt/ttmlir-toolchain/src/shardy/shardy/dialect/sdy/transforms/propagation/passes.td
  COMMENT "Creating symlink Passes.td -> passes.td in shardy propagation"
)

add_custom_target(SdyPropagationPassesSymlink ALL
  DEPENDS /opt/ttmlir-toolchain/src/shardy/shardy/dialect/sdy/transforms/propagation/Passes.td
)

add_custom_target(docs
  COMMAND
    mdbook build
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS
    autogen-summary mlir-doc copy-docs-dir SdyExportPassesSymlink SdyImportPassesSymlink SdyPropagationPassesSymlink
)

find_package(Doxygen)

if (DOXYGEN_FOUND)

  if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/book/doxygen)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/book/doxygen)
  endif()

  set(DOXYGEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/book/doxygen)
  set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

  add_custom_target(doxygen
    COMMAND
      ${CMAKE_COMMAND} -E make_directory ${DOXYGEN_DIR}
    COMMAND
      ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules/ConfigureDoxygen.cmake
    COMMAND
      doxygen ${DOXYGEN_DIR}/doxygen.cfg
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS
      mlir-headers mlir-doc docs copy-docs-dir
  )
else()
  message("Doxygen is required to build documentation")
endif()
