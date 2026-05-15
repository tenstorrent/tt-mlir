# Cross-compile the X280 firmware harness for RISC-V bare metal.
#
# Produces x280_firmware.o — a relocatable object containing the startup code
# (fw_start.S) and the firmware body (fw_step5.c). At model compile time the
# CPU-hoisting pipeline links this with a per-model cpu.o and fw.ld to produce
# the final firmware binary.

if (TTMLIR_ENABLE_RUNTIME)
  find_program(X280_CC clang-20 clang REQUIRED)
  find_program(X280_LD ld.lld-20 ld.lld REQUIRED)

  set(X280_CFLAGS
    --target=riscv64-unknown-elf
    -mcpu=sifive-x280 -mabi=lp64d -mcmodel=medany
    -O2 -ffreestanding -nostdlib -nostartfiles -static
    -Wall -Wextra -fno-common
  )

  set(X280_FW_DIR ${CMAKE_CURRENT_BINARY_DIR})
  set(X280_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  add_custom_command(
    OUTPUT ${X280_FW_DIR}/fw_start.o
    COMMAND ${X280_CC} ${X280_CFLAGS} -c
      ${X280_SRC_DIR}/fw_start.S -o ${X280_FW_DIR}/fw_start.o
    DEPENDS ${X280_SRC_DIR}/fw_start.S
    COMMENT "Cross-compiling fw_start.S for X280"
  )

  add_custom_command(
    OUTPUT ${X280_FW_DIR}/fw_step5.o
    COMMAND ${X280_CC} ${X280_CFLAGS} -c
      ${X280_SRC_DIR}/fw_step5.c -o ${X280_FW_DIR}/fw_step5.o
    DEPENDS ${X280_SRC_DIR}/fw_step5.c
    COMMENT "Cross-compiling fw_step5.c for X280"
  )

  add_custom_command(
    OUTPUT ${X280_FW_DIR}/x280_firmware.o
    COMMAND ${X280_LD} -r
      ${X280_FW_DIR}/fw_start.o ${X280_FW_DIR}/fw_step5.o
      -o ${X280_FW_DIR}/x280_firmware.o
    DEPENDS ${X280_FW_DIR}/fw_start.o ${X280_FW_DIR}/fw_step5.o
    COMMENT "Linking x280_firmware.o (partial link)"
  )

  add_custom_target(x280_firmware ALL
    DEPENDS ${X280_FW_DIR}/x280_firmware.o
  )

  install(
    FILES ${X280_FW_DIR}/x280_firmware.o ${X280_SRC_DIR}/fw.ld
    DESTINATION lib/ttmlir/firmware/x280
  )
endif()
