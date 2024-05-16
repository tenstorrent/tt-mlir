TOOLCHAIN_ENV=$(ENV)/toolchain
TTMLIR_VENV=$(ENV)/ttmlir_venv
MLIR_PYTHON_REQUIREMENTS=third_party/llvm-project/mlir/python/requirements.txt
TTMLIR_PYTHON_EXAMPLES_REQUIREMENTS=python_torch_examples/requirements$(OS).txt
CMAKE=$(TOOLCHAIN_ENV)/bin/cmake
NINJA=$(TOOLCHAIN_ENV)/bin/ninja
LLVM_BUILD_DIR=$(TOOLCHAIN_ENV)/llvm_build
LLVM_INSTALL=$(LLVM_BUILD_DIR)/.installed
FLATBUFFERS_INSTALL=$(TOOLCHAIN_ENV)/bin/flatc

env: $(TTMLIR_VENV) $(LLVM_INSTALL) $(FLATBUFFERS_INSTALL) ;
toolchain: $(LLVM_INSTALL) ;

$(TTMLIR_VENV): $(TTMLIR_VENV)/.dep ;

git_submodule_update:
	git submodule update --init --recursive

$(MLIR_PYTHON_REQUIREMENTS): git_submodule_update ;
third_party/llvm-project/mlir/CMakeLists.txt: git_submodule_update ;

$(TTMLIR_VENV)/bin/activate:
	bash -c "python3 -m venv $(TTMLIR_VENV)"
	bash -c "source env/activate && python -m pip install --upgrade pip"

$(TTMLIR_VENV)/.dep: $(TTMLIR_VENV)/bin/activate $(MLIR_PYTHON_REQUIREMENTS) $(TTMLIR_PYTHON_EXAMPLES_REQUIREMENTS)
	bash -c "source env/activate && pip install -r third_party/llvm-project/mlir/python/requirements.txt"
	bash -c "source env/activate && pip install --pre -f https://llvm.github.io/torch-mlir/package-index/ --extra-index-url https://download.pytorch.org/whl/nightly/cpu -r python_torch_examples/requirements$(OS).txt"
	touch $@

$(LLVM_INSTALL): $(TTMLIR_VENV) $(CMAKE) $(NINJA) third_party/llvm-project/mlir/CMakeLists.txt ./env/build_llvm.sh
	PATH=$(TOOLCHAIN_ENV)/bin:$(PATH) INSTALL_PREFIX=$(TOOLCHAIN_ENV) ./env/build_llvm.sh
	touch $@

$(CMAKE):
	mkdir -p $(TOOLCHAIN_ENV)/bin
ifeq ($(OS),Linux)
	bash -c "curl -LO https://github.com/Kitware/CMake/releases/download/v3.29.0-rc2/cmake-3.29.0-rc2-linux-x86_64.sh && bash cmake-3.29.0-rc2-linux-x86_64.sh --prefix=$(TOOLCHAIN_ENV) --skip-license && rm cmake-3.29.0-rc2-linux-x86_64.sh"
else ifeq ($(OS),Darwin)
	brew install cmake
	ln -s $(shell command -v cmake) $@
else
error "Unsupported cmake OS $(OS)"
endif

$(NINJA):
	mkdir -p $(TOOLCHAIN_ENV)/bin
ifeq ($(OS),Linux)
	bash -c "cd $(TOOLCHAIN_ENV)/bin && curl -LO https://github.com/ninja-build/ninja/releases/download/v1.11.1/ninja-linux.zip && unzip ninja-linux.zip && rm ninja-linux.zip"
else ifeq ($(OS),Darwin)
	brew install ninja
	ln -s $(shell command -v ninja) $@
else
error "Unsupported ninja OS $(OS)"
endif

$(FLATBUFFERS_INSTALL): $(CMAKE) $(NINJA)
	git submodule update --init --recursive
	mkdir -p $(TOOLCHAIN_ENV)/bin
	PATH=$(TOOLCHAIN_ENV)/bin:$(PATH) cmake -S third_party/flatbuffers -B $(TOOLCHAIN_ENV)/flatbuffers_build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(TOOLCHAIN_ENV)
	PATH=$(TOOLCHAIN_ENV)/bin:$(PATH) cmake --build $(TOOLCHAIN_ENV)/flatbuffers_build
	PATH=$(TOOLCHAIN_ENV)/bin:$(PATH) cmake --install $(TOOLCHAIN_ENV)/flatbuffers_build
