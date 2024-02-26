CONDA_ENV=conda_env
CONDA_ENV_INSTALLER=$(CONDA_ENV)/Miniconda3-latest-Linux-x86_64.sh
TTMLC_ENV=$(CONDA_ENV)/envs/ttmlc

$(TTMLC_ENV): $(TTMLC_ENV)/.dep

$(CONDA_ENV_INSTALLER):
	curl -L --create-dirs --output $@ https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

$(CONDA_ENV)/bin/activate: $(CONDA_ENV_INSTALLER)
	bash $(CONDA_ENV_INSTALLER) -u -b -s -p $(PWD)/$(CONDA_ENV)

$(TTMLC_ENV)/.dep: $(CONDA_ENV)/bin/activate python/requirements.txt
	bash -c "source $(CONDA_ENV)/bin/activate && conda create -n ttmlc -y python=3.11 && conda activate ttmlc && python -m pip install --upgrade pip"
	bash -c "source $(CONDA_ENV)/bin/activate && pip install --pre -f https://llvm.github.io/torch-mlir/package-index/ --extra-index-url https://download.pytorch.org/whl/nightly/cpu -r python/requirements.txt"
	touch $@
