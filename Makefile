OUT ?= build
ENV ?= $(shell realpath .)/.local
CONFIG ?= Release
BUILDER ?= Ninja
OS ?= $(shell uname)

all: build

build: $(OUT)/bin/ttmlir-opt

.PHONY: $(OUT)/bin/ttmlir-opt
$(OUT)/bin/ttmlir-opt: $(OUT)/build.ninja
	bash -c "source env/activate && cmake --build $(OUT)"

clean:
	rm -rf $(OUT)

spotless:
	rm -rf $(OUT)
	rm -rf $(ENV)

.PRECIOUS: $(OUT)/build.ninja
$(OUT)/build.ninja: env CMakeLists.txt cmake.sh
	OUT=$(OUT) ENV=$(ENV) CONFIG=$(CONFIG) BUILDER=$(BUILDER) ./cmake.sh

include env/module.mk
