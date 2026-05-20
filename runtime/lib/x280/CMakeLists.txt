# Cross-compile X280 firmware and runtime support for RISC-V bare metal.
#
# Produces:
#   x280_firmware.o — relocatable object containing the startup code
#     (fw_start.S) and the firmware body (fw_step6.c), used to build the
#     firmware binary that runs the persistent task loop on the X280.
#   x280_rt.o — freestanding runtime support (memcpy) for kernel code blobs.
#   x280_blob.ld — linker script for code blobs (places dispatch at
#     CODE_LOAD_ADDR).

if (TTMLIR_ENABLE_RUNTIME)
  find_program(X280_CC clang-20 clang REQUIRED)
  find_program(X280_LD ld.lld-20 ld.lld REQUIRED)
  find_program(X280_OBJCOPY llvm-objcopy-20 llvm-objcopy REQUIRED)

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
    OUTPUT ${X280_FW_DIR}/fw_step6.o
    COMMAND ${X280_CC} ${X280_CFLAGS} -c
      ${X280_SRC_DIR}/fw_step6.c -o ${X280_FW_DIR}/fw_step6.o
    DEPENDS ${X280_SRC_DIR}/fw_step6.c
    COMMENT "Cross-compiling fw_step6.c for X280"
  )

  add_custom_command(
    OUTPUT ${X280_FW_DIR}/x280_firmware.o
    COMMAND ${X280_LD} -r
      ${X280_FW_DIR}/fw_start.o ${X280_FW_DIR}/fw_step6.o
      -o ${X280_FW_DIR}/x280_firmware.o
    DEPENDS ${X280_FW_DIR}/fw_start.o ${X280_FW_DIR}/fw_step6.o
    COMMENT "Linking x280_firmware.o (partial link)"
  )

  add_custom_command(
    OUTPUT ${X280_FW_DIR}/x280_rt.o
    COMMAND ${X280_CC} ${X280_CFLAGS} -c
      ${X280_SRC_DIR}/x280_rt.c -o ${X280_FW_DIR}/x280_rt.o
    DEPENDS ${X280_SRC_DIR}/x280_rt.c
    COMMENT "Cross-compiling x280_rt.c for X280"
  )

  configure_file(${X280_SRC_DIR}/fw.ld ${X280_FW_DIR}/fw.ld COPYONLY)
  configure_file(${X280_SRC_DIR}/x280_blob.ld ${X280_FW_DIR}/x280_blob.ld COPYONLY)

  # Final-link the firmware into a standalone ELF and convert to flat binary.
  # This binary is loaded onto the X280 at device init time.
  add_custom_command(
    OUTPUT ${X280_FW_DIR}/x280_firmware.elf
    COMMAND ${X280_LD}
      -T ${X280_FW_DIR}/fw.ld --build-id=none
      ${X280_FW_DIR}/x280_firmware.o
      -o ${X280_FW_DIR}/x280_firmware.elf
    DEPENDS ${X280_FW_DIR}/x280_firmware.o ${X280_FW_DIR}/fw.ld
    COMMENT "Linking x280_firmware.elf"
  )

  add_custom_command(
    OUTPUT ${X280_FW_DIR}/x280_firmware.bin
    COMMAND ${X280_OBJCOPY} -O binary
      ${X280_FW_DIR}/x280_firmware.elf ${X280_FW_DIR}/x280_firmware.bin
    DEPENDS ${X280_FW_DIR}/x280_firmware.elf
    COMMENT "Creating x280_firmware.bin"
  )

  add_custom_target(x280_firmware ALL
    DEPENDS
      ${X280_FW_DIR}/x280_firmware.bin
      ${X280_FW_DIR}/x280_rt.o
  )

  install(
    FILES
      ${X280_FW_DIR}/x280_firmware.bin
      ${X280_FW_DIR}/x280_rt.o
      ${X280_SRC_DIR}/x280_blob.ld
    DESTINATION lib/ttmlir/firmware/x280
  )
endif()
