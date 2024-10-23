set(TT_EXPLORER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run.sh)
set(TTMLIR_BUILD_BIN_DIR ${TTMLIR_BINARY_DIR}/bin)

add_custom_target(explorer
  COMMENT "Building tt-explorer... ${TTMLIR_BIN_DIR}"
  COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/* ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND pip install ${CMAKE_CURRENT_BINARY_DIR}/tt_adapter
  # DEPENDS ttrt
)

add_custom_command(TARGET explorer POST_BUILD
  COMMENT "Installing tt-explorer command..."
  COMMAND ${CMAKE_COMMAND} -E copy ${TT_EXPLORER_SCRIPT} ${TTMLIR_BUILD_BIN_DIR}/tt-explorer
  COMMAND ${CMAKE_COMMAND} -E echo "Done. Run: tt-explorer to start the server."
)
