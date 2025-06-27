include(${CMAKE_SOURCE_DIR}/cmake/modules/ConfigureSphinx.cmake)

add_custom_target(copy-docs-dir
  COMMAND
    cp -r ${CMAKE_CURRENT_SOURCE_DIR}/* ${CMAKE_CURRENT_BINARY_DIR}
)

# Add Sphinx documentation target
add_custom_target(sphinx-docs
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${SPHINX_BUILD_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SPHINX_BUILD_DIR}
  COMMAND ${CMAKE_COMMAND} -E env LC_ALL=C ${SPHINX_APIDOC_COMMAND}
  COMMAND ${CMAKE_COMMAND} -E env LC_ALL=C ${SPHINX_BUILD_COMMAND}
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS
    TTIRBuilderModules
    copy-docs-dir mlir-doc
  COMMENT "Generating Sphinx documentation"
)

add_custom_target(autogen-summary
  COMMAND
    python ${CMAKE_CURRENT_SOURCE_DIR}/autogen-summary.py
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS
    mlir-doc copy-docs-dir sphinx-docs
)

add_custom_target(remove-index
    COMMAND ${CMAKE_COMMAND} -E remove ${SPHINX_MD_DIR}/index.md
    DEPENDS sphinx-docs
)

add_custom_target(docs
  COMMAND
    mdbook build
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS
    mlir-doc
    copy-docs-dir
    sphinx-docs
    autogen-summary
    remove-index
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
