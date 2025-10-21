include(ExternalProject)

set(TT_EXPLORER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run.py)
set(TTMLIR_BUILD_BIN_DIR ${TTMLIR_BINARY_DIR}/bin)

set(MODEL_EXPLORER_VERSION "a8c2343ea437b045c7a4a59cd8c43d94940e484c")

ExternalProject_Add(
  model-explorer
  PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/model-explorer
  GIT_REPOSITORY https://github.com/tenstorrent/model-explorer.git
  GIT_TAG ${MODEL_EXPLORER_VERSION}
  GIT_PROGRESS OFF
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)

set(PIP_EDITABLE_FLAG "")

if (TT_EXPLORER_EDITABLE)
  set(PIP_EDITABLE_FLAG "-e")
endif()

set(EXPLORER_REQ_FLAGS
  TT_RUNTIME_DEBUG
  TT_RUNTIME_ENABLE_PERF_TRACE
  TTMLIR_ENABLE_RUNTIME
)

foreach(RUNTIME_OPT ${EXPLORER_REQ_FLAGS})
  if(NOT ${RUNTIME_OPT})
    message(WARNING "${RUNTIME_OPT} must be enabled to execute model as intended in tt-explorer.")
  endif()
endforeach()

add_custom_target(explorer
  COMMENT "Building tt-explorer... ${TTMLIR_BIN_DIR}"
  COMMAND pip install ${PIP_EDITABLE_FLAG} ${CMAKE_CURRENT_SOURCE_DIR}/tt_adapter
  COMMAND pip uninstall -y ai-edge-model-explorer
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/model-explorer/src/model-explorer/src/server/package/build
  COMMAND pip install --upgrade ${CMAKE_CURRENT_SOURCE_DIR}/model-explorer/src/model-explorer/src/server/package
  COMMAND python -m pip wheel ${CMAKE_CURRENT_SOURCE_DIR}/tt_adapter --wheel-dir ${TTMLIR_BUILD_BIN_DIR}/wheels
  COMMAND python -m pip wheel ${CMAKE_CURRENT_SOURCE_DIR}/model-explorer/src/model-explorer/src/server/package --wheel-dir ${TTMLIR_BUILD_BIN_DIR}/wheels

  DEPENDS TTMLIRPythonModules model-explorer ttmlir-opt ttmlir-translate
)

add_custom_command(TARGET explorer POST_BUILD
  COMMENT "Installing tt-explorer command..."
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${TT_EXPLORER_SCRIPT} ${TTMLIR_BUILD_BIN_DIR}/tt-explorer
  COMMAND ${CMAKE_COMMAND} -E echo "Done. Run: tt-explorer to start the server."
)
