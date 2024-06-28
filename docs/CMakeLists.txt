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
      ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules/configure_doxygen.cmake
    COMMAND
      doxygen ${DOXYGEN_DIR}/doxygen.cfg
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS
      docs mlir-doc copy-docs-dir
  )


else()
  message(FATAL_ERROR "Doxygen is required to build documentation")
endif()
