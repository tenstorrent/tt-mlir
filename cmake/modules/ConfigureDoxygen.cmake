set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(DOXYGEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/book/doxygen)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/book/doxygen/doxygen.cfg @ONLY)
