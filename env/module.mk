TOOLCHAIN_ENV=$(ENV)/toolchain
CONDA_ENV=$(ENV)/conda
CONDA_ENV_INSTALLER=$(CONDA_ENV)/scripts/Miniconda3-latest-Linux-x86_64.sh
TTMLC_ENV=$(CONDA_ENV)/envs/ttmlc
CMAKE=$(TOOLCHAIN_ENV)/bin/cmake
NINJA=$(TOOLCHAIN_ENV)/bin/ninja
LLVM_BUILD_DIR=$(TOOLCHAIN_ENV)/llvm_build
LLVM_INSTALL=$(LLVM_BUILD_DIR)/.installed

env: $(TTMLC_ENV) $(LLVM_INSTALL) ;
conda_env: $(TTMLC_ENV) ;
toolchain: $(LLVM_INSTALL) ;

$(CONDA_ENV_INSTALLER):
	curl -L --create-dirs --output $@ https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

$(CONDA_ENV)/bin/activate: $(CONDA_ENV_INSTALLER)
	bash $(CONDA_ENV_INSTALLER) -u -b -s -p $(CONDA_ENV)

$(TTMLC_ENV): $(TTMLC_ENV)/.dep ;

$(TTMLC_ENV)/.dep: $(CONDA_ENV)/bin/activate env/environment.yml python/requirements.txt
	bash -c "source $(CONDA_ENV)/bin/activate && conda env create -f env/environment.yml"
	bash -c "source env/activate && pip install --pre -f https://llvm.github.io/torch-mlir/package-index/ --extra-index-url https://download.pytorch.org/whl/nightly/cpu -r python/requirements.txt"
	touch $@

$(LLVM_INSTALL): $(CMAKE) $(NINJA) ./env/build_llvm.sh
	git submodule update --init --recursive
	PATH=$(TOOLCHAIN_ENV)/bin:$(PATH) INSTALL_PREFIX=$(TOOLCHAIN_ENV) ./env/build_llvm.sh
	touch $@

$(CMAKE):
	mkdir -p $(TOOLCHAIN_ENV)
	bash -c "curl -LO https://github.com/Kitware/CMake/releases/download/v3.29.0-rc2/cmake-3.29.0-rc2-linux-x86_64.sh && bash cmake-3.29.0-rc2-linux-x86_64.sh --prefix=$(TOOLCHAIN_ENV) --skip-license && rm cmake-3.29.0-rc2-linux-x86_64.sh"

$(NINJA):
	mkdir -p $(TOOLCHAIN_ENV)/bin
	bash -c "cd $(TOOLCHAIN_ENV)/bin && curl -LO https://github.com/ninja-build/ninja/releases/download/v1.11.1/ninja-linux.zip && unzip ninja-linux.zip && rm ninja-linux.zip"
