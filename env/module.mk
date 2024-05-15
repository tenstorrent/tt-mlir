TOOLCHAIN_ENV=$(ENV)/toolchain
CONDA_ENV=$(ENV)/conda
CONDA_ENV_INSTALLER=$(CONDA_ENV)/scripts/Miniconda3-latest-Linux-x86_64.sh
TTMLIR_ENV=$(CONDA_ENV)/envs/ttmlir
CMAKE=$(TOOLCHAIN_ENV)/bin/cmake
NINJA=$(TOOLCHAIN_ENV)/bin/ninja
LLVM_BUILD_DIR=$(TOOLCHAIN_ENV)/llvm_build
LLVM_INSTALL=$(LLVM_BUILD_DIR)/.installed
FLATBUFFERS_INSTALL=$(TOOLCHAIN_ENV)/bin/flatc
ifeq ($(OS),Linux)
MINICONDA_OS=Linux
else ifeq ($(OS),Darwin)
MINICONDA_OS=MacOSX
else
error "Unsupported miniconda OS $(OS)"
endif

env: $(TTMLIR_ENV) $(LLVM_INSTALL) $(FLATBUFFERS_INSTALL) ;
conda_env: $(TTMLIR_ENV) ;
toolchain: $(LLVM_INSTALL) ;

$(CONDA_ENV_INSTALLER):
	mkdir -p $(@D)
	curl -L --create-dirs --output $@ https://repo.anaconda.com/miniconda/Miniconda3-latest-$(MINICONDA_OS)-x86_64.sh

$(CONDA_ENV)/bin/activate: $(CONDA_ENV_INSTALLER)
	bash $(CONDA_ENV_INSTALLER) -u -b -s -p $(CONDA_ENV)

$(TTMLIR_ENV): $(TTMLIR_ENV)/.dep ;

$(TTMLIR_ENV)/.dep: $(CONDA_ENV)/bin/activate env/environment.yml python/requirements.txt
	bash -c "source $(CONDA_ENV)/bin/activate && conda env create -f env/environment.yml || true"
	bash -c "source env/activate && pip install --pre -f https://llvm.github.io/torch-mlir/package-index/ --extra-index-url https://download.pytorch.org/whl/nightly/cpu -r python/requirements$(OS).txt"
	touch $@

$(LLVM_INSTALL): $(CMAKE) $(NINJA) ./env/build_llvm.sh
	git submodule update --init --recursive
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
