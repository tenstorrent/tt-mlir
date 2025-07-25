add_custom_target(chisel
  COMMENT "python chisel package")

add_custom_command(
  COMMAND python -m pip install -e .
  TARGET chisel
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
