add_custom_target(explorer
  COMMENT "Building tt-explorer..."
  COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/* ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND pip install tt_adapter
  COMMENT "Done. Use 'model-explorer --extensions=tt_adapter' to start the explorer."
  # DEPENDS ttrt
)
# add_dependencies(ttrt TTRuntime)
