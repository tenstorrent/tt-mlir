OUT ?= build
ENV ?= $(shell realpath .)/.local
CONFIG ?= Release
BUILDER ?= Ninja
OS ?= $(shell uname)

all: build

build: $(OUT)/bin/ttmlir-opt

.PHONY: $(OUT)/bin/ttmlir-opt
$(OUT)/bin/ttmlir-opt: $(OUT)/CMakeCache.txt
	bash -c "source env/activate && cmake --build $(OUT)"

clean:
	bash -c "source env/activate && cmake --build $(OUT) clean"

spotless:
	rm -rf $(OUT)

$(OUT)/CMakeCache.txt: env CMakeLists.txt cmake.sh
	OUT=$(OUT) ENV=$(ENV) CONFIG=$(CONFIG) BUILDER=$(BUILDER) ./cmake.sh

include env/module.mk
