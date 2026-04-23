# Cross-build for X280 bare-metal firmware.
#
# Defaults to clang-20 + ld.lld + llvm-objcopy because that's what's on this
# host. To use the riscv64-linux-gnu GCC toolchain instead, build with:
#   make CC=riscv64-linux-gnu-gcc OBJCOPY=riscv64-linux-gnu-objcopy LDFLAGS="-T fw.ld -Wl,--build-id=none"
# (drop the `-fuse-ld=lld` and `--target=` from CFLAGS in that case).
#
# -nostdlib/-nostartfiles/-static keeps us free of any libc; -mcmodel=medany
# gives us the full 64-bit address range needed for the 0x400030000000 link
# address.

# `?=` doesn't help: make's implicit rules pre-set CC to `cc`. Command-line
# overrides (`make CC=riscv64-linux-gnu-gcc ...`) still beat `:=`.
CC      := clang-20
OBJCOPY := llvm-objcopy-20
OBJDUMP := llvm-objdump-20

CFLAGS  := --target=riscv64-unknown-elf \
           -march=rv64imafdc -mabi=lp64d -mcmodel=medany \
           -O2 -ffreestanding -nostdlib -nostartfiles -static \
           -Wall -Wextra -fno-common
LDFLAGS := -fuse-ld=lld -T fw.ld -Wl,--build-id=none

BINS := fw_step1.bin fw_step2.bin fw_step4.bin

.PHONY: all clean
all: $(BINS)

fw_step1.elf: fw_start.S fw_step1.c fw.ld
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ fw_start.S fw_step1.c

fw_step2.elf: fw_start.S fw_step2.c fw.ld
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ fw_start.S fw_step2.c

fw_step4.elf: fw_start.S fw_step4.c fw.ld
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ fw_start.S fw_step4.c

%.bin: %.elf
	$(OBJCOPY) -O binary $< $@

%.lst: %.elf
	$(OBJDUMP) -d $< > $@

clean:
	$(RM) *.elf *.bin *.lst
