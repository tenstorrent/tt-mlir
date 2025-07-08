include(${CMAKE_SOURCE_DIR}/cmake/modules/ConfigureSphinx.cmake)

add_custom_target(copy-docs-dir
  COMMAND
    cp -r ${CMAKE_CURRENT_SOURCE_DIR}/* ${CMAKE_CURRENT_BINARY_DIR}
)

execute_process(
    COMMAND ${Python_EXECUTABLE} -m pip install sphinx sphinx-markdown-builder
    # Optional arguments for pip install, like --upgrade, -e for editable install, etc.
    # ARGS ${_pip_args}
    # Specify the working directory if needed
    # WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)

# Find Sphinx executables
find_program(SPHINX_EXECUTABLE
    NAMES sphinx-build
    DOC "Sphinx documentation generator"
)

find_program(SPHINX_APIDOC_EXECUTABLE
    NAMES sphinx-apidoc
    DOC "Sphinx API doc generator"
)

if(NOT SPHINX_EXECUTABLE)
    message(FATAL_ERROR "sphinx-build not found - required to build documentation. Make sure sphinx and sphinx-markdown-builder are installed.")
endif()

if(NOT SPHINX_APIDOC_EXECUTABLE)
    message(FATAL_ERROR "sphinx-apidoc not found - required to build API documentation. Make sure sphinx and sphinx-markdown-builder are installed.")
endif()

# Configure sphinx-apidoc command
set(SPHINX_APIDOC_COMMAND ${SPHINX_APIDOC_EXECUTABLE}
    -o ${SPHINX_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/python_packages/ttir_builder
)

set(SPHINX_BUILD_COMMAND ${SPHINX_EXECUTABLE}
    -b markdown
    ${SPHINX_SOURCE_DIR}
    ${SPHINX_MD_DIR}
)

# Add Sphinx documentation target
add_custom_target(sphinx
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${SPHINX_BUILD_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SPHINX_BUILD_DIR}
  COMMAND ${CMAKE_COMMAND} -E env LC_ALL=C ${SPHINX_APIDOC_COMMAND}
  COMMAND ${CMAKE_COMMAND} -E env LC_ALL=C ${SPHINX_BUILD_COMMAND}
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS
    TTIRBuilderModules
    TTMLIRPythonModules
    copy-docs-dir mlir-doc
  COMMENT "Generating Sphinx documentation"
)

add_custom_target(autogen-summary
  COMMAND
    python ${CMAKE_CURRENT_SOURCE_DIR}/autogen-summary.py
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS
    mlir-doc copy-docs-dir sphinx
)

add_custom_target(remove-index
    COMMAND ${CMAKE_COMMAND} -E remove ${SPHINX_MD_DIR}/index.md
    DEPENDS sphinx
)

add_custom_target(docs
  COMMAND
    mdbook build
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS
    mlir-doc
    copy-docs-dir
    sphinx
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
