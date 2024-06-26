add_custom_target(copy-docs-dir
  COMMAND
    cp -r ${CMAKE_CURRENT_SOURCE_DIR}/ ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(autogen-summary
  COMMAND
    python ${CMAKE_CURRENT_SOURCE_DIR}/autogen-summary.py
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS
    mlir-doc copy-docs-dir
)

add_custom_target(docs
  COMMAND
    mdbook build
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS
    autogen-summary mlir-doc copy-docs-dir
)
