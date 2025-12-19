add_custom_target(chisel
  COMMENT "python chisel package")

add_custom_command(
  TARGET chisel
  POST_BUILD
  COMMAND python -m pip install -e .
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
