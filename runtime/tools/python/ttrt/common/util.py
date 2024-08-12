# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttrt.binary
import os
import json
import importlib.machinery
import sys
import signal
import os
import io
import subprocess
import time
import socket
from pkg_resources import get_distribution
import sys
import shutil
import logging

if "LOGGER_LEVEL" not in os.environ:
    os.environ["LOGGER_LEVEL"] = "FATAL"
if "TT_METAL_LOGGER_LEVEL" not in os.environ:
    os.environ["TT_METAL_LOGGER_LEVEL"] = "FATAL"

class FileManager:
    def __init__(self):
        pass

    def file_directory_constraint_check(self, path):
        if not isinstance(path, str):
            raise TypeError(f"the path '{path}' must be a string")

        if not path:
            raise ValueError(f"the path '{path}' cannot be an empty string")

    @staticmethod
    def create_file(file_path):
        file_directory_constraint_check(file_path)

        try:
            if not FileManager.check_directory_exists(os.path.dirname(file_path)):
                FileManager.create_directory(os.path.dirname(file_path))

            with open(file_path, 'w') as file:
                file.write('')
        except OSError as e:
            raise OSError(f"error creating file: {e}")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    @staticmethod
    def create_directory(directory_path):
        file_directory_constraint_check(directory_path)

        try:
            os.makedirs(directory_path)
        except OSError as e:
            raise OSError(f"error creating directory: {e}")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    @staticmethod
    def remove_file(file_path):
        file_directory_constraint_check(file_path)

        try:
            os.remove(file_path)
        except FileNotFoundError:
            logging.warning(f"file '{file_path}' not found")
        except PermissionError:
            raise PermissionError(f"insufficient permissions to remove file '{file_path}'")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    @staticmethod
    def remove_directory(directory_path):
        file_directory_constraint_check(directory_path)

        try:
            shutil.rmtree(directory_path)
        except FileNotFoundError:
            logging.warning(f"directory '{directory_path}' not found")
        except PermissionError:
            raise PermissionError(f"insufficient permissions to remove directory '{directory_path}'")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    @staticmethod
    def copy_file(dest_file_path, src_file_path):
        file_directory_constraint_check(dest_file_path)
        file_directory_constraint_check(src_file_path)

        try:
            shutil.copy2(src_file_path, dest_file_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"the source file does not exist: '{src_file_path}'")
        except PermissionError as e:
            raise PermissionError(f"permission denied: '{src_file_path}' or '{dest_file_path}'")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    @staticmethod
    def copy_directory(dest_directory_path, src_directory_path):
        file_directory_constraint_check(dest_directory_path)
        file_directory_constraint_check(src_directory_path)

        try:
            shutil.copytree(src_directory_path, dest_directory_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"source directory does not exist: '{src_directory_path}")'
        except FileExistsError as e:
            raise FileExistsError(f"destination directory already exists: '{dest_directory_path}'")
        except PermissionError as e:
            raise PermissionError(f"permission denied: '{src_directory_path}' or '{dest_directory_path}'") from e
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")
    
    @staticmethod
    def check_file_exists(file_path):
        file_directory_constraint_check(file_path)
        exists = False

        try:
            if os.path.exists(file_path):
                exists = True
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return exists

    @staticmethod
    def check_directory_exists(directory_path):
        file_directory_constraint_check(directory_path)
        exists = False

        try:
            if os.path.isdir(directory_path):
                exists = True
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return exists

    @staticmethod
    def is_file(file_path):
        file_directory_constraint_check(file_path)
        is_file = False

        try:
            if os.path.exists(file_path):
                is_file = True
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return is_file

    @staticmethod
    def is_directory(directory_path):
        file_directory_constraint_check(directory_path)
        is_directory = False

        try:
            if os.path.isdir(directory_path):
                is_directory = True
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return is_directory

    @staticmethod
    def get_file_name(file_path):
        file_directory_constraint_check(file_path)
        file_name = ""

        try:
            file_name = os.path.basename(file_path)
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return file_name

    @staticmethod
    def get_file_extension(file_path):
        file_directory_constraint_check(file_path)
        file_extension = ""

        try:
            _, file_extension = os.path.splitext(filepath)
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return file_extension

class Artifacts:

    def __init__(self, ttmlir_home_path, ttmlir_venv_path, ttmetal_home_path, artifacts_folder_path, log_name):
      self.ttmlir_home_path = ttmlir_home_path
      self.ttmlir_venv_path = ttmlir_venv_path
      self.ttmetal_home_path = ttmetal_home_path
      self.artifacts_folder_path = artifacts_folder_path
      self.log_name = log_name

    def get_ttmlir_home_path(self):
        return self.ttmlir_home_path

    def get_ttmlir_venv_path(self):
        return self.ttmlir_venv_path

    def get_ttmetal_home_path(self):
        return self.ttmetal_home_path

    def get_artifacts_folder_path(self):
        return self.artifacts_folder_path

    def get_log_name(self):
        return self.log_name

    def get_binary_folder_path(self, binary):
        return f"{self.get_artifacts_folder_path()}/{binary.name}_{binary.extension}"

    def create_artifacts(self):
        FileManager.create_directory(self.get_artifacts_folder_path())

    def clean_artifacts(self):
        FileManager.remove_directory(self.get_artifacts_folder_path())

    def clean_binary_artifacts(self, binary):
        FileManager.remove_directory(self.get_binary_folder_path(binary))

    def save_binary(self, binary):
        FileManager.create_directory(self.get_binary_folder_path(binary))
        FileManager.create_directory(f"{self.get_binary_folder_path(binary)}/perf")
        FileManager.copy_file(f"{self.get_binary_folder_path(binary)}", binary.file_path)

    def __str__(self):
        return f"ttmlir_home_path={self.ttmlir_home_path}\nttmlir_venv_path={self.ttmlir_venv_path}\nttmetal_home_path={self.ttmetal_home_path}\nartifacts_folder_path={self.artifacts_folder_path}\nlog_name={self.log_name}\n"

    @staticmethod
    def prepare_artifact(ttmlir_home_path="", ttmlir_venv_path="", ttmetal_home_path="", artifacts_folder_path="", log_name="ttrt.log"):
        ttmlir_home_path = os.environ.get("TT_MLIR_HOME", f"{os.getcwd()}") if ttmlir_home_path == "" else ttmlir_home_path
        ttmlir_venv_path = os.environ.get("TTMLIR_VENV_DIR", "/opt/ttmlir-toolchain/venv") if ttmlir_venv_path == "" else ttmlir_venv_path
        ttmetal_home_path = os.environ.get("TT_METAL_HOME", "third_party/tt-metal/src/tt-metal") if ttmetal_home_path == "" else ttmetal_home_path
        artifacts_folder_path = f"{ttmlir_home_path}/ttrt-artifacts" if artifacts_folder_path == "" else artifacts_folder_path

        return Artifacts(ttmlir_home_path, ttmlir_venv_path, ttmetal_home_path, artifacts_folder_path, log_name)

    @staticmethod
    def save_torch_tensor(binary_name, torch_tensor, torch_tensor_name):
        import torch

        try:
            torch.save(torch_tensor, f"{self.get_artifacts_folder_path()}/{binary_name}/{torch_tensor_name}")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    @staticmethod
    def save_system_desc(system_desc, system_desc_name="system_desc.ttsys"):
        try:
            system_desc.store(f"{self.get_artifacts_folder_path()}/{system_desc_name}")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

class Binary:

    def __init__(self, file_path):
        import ttrt.binary

        self.file_path = file_path
        self.name = FileManager.get_file_name(file_path)
        self.extension = FileManager.get_file_extension(file_path)
        self.fbb = ttrt.binary.load_from_path(file_path)

    def check_version(self):
        package_name = "ttrt"

        try:
            package_version = get_distribution(package_name).version
        except Exception as e:
            print(f"error retrieving version: {e} for {package_name}")

        if package_version != self.fbb.version:
            logging.info(f"{package_name}: v{package_version} does not match flatbuffer: v{self.fbb.version} for flatbuffer: {self.file_path} - skipping this test")
            return False
        
        return True

class BinaryTTNN(Binary):
    file_extension = ".ttnn"

    def __int__(self, file_path):
        super().__init__(file_path)

    @staticmethod
    def get_file_extension():
        return BinaryTTNN.file_extension

    @staticmethod
    def prepare_binary(file_path):
        return BinaryTTNN(file_path)

    @staticmethod
    def find_ttnn_binary_paths(path):
        files = []

        if FileManager.is_file(path):
            if FileManager.check_file_exists(path):
                files.append(path)
            else:
                logging.info(f"file '{path}' not found - skipping")
        else:
            FileManager.check_directory_exists(path)
            try: 
                for root, _, files in os.walk(path):
                    for file in files:
                        if FileManager.get_file_extension(file) == BinaryTTNN.get_file_extension()
                            files.append(os.path.join(root, file))
            except Exception as e:
                raise Exception(f"an unexpected error occurred: {e}")

        return files

class BinaryTTMetal(Binary):
    file_extension = ".ttb"

    def __int__(self, file_path):
        super().__init__(file_path)

    @staticmethod
    def get_file_extension():
        return BinaryTTMetal.file_extension

    @staticmethod
    def prepare_binary(file_path):
        return BinaryTTMetal(file_path)

    @staticmethod
    def find_ttmetal_binary_paths(directory_path):
        files = []

        if FileManager.is_file(path):
            if FileManager.check_file_exists(path):
                files.append(path)
            else:
                logging.info(f"file '{path}' not found - skipping")
        else:
            FileManager.check_directory_exists(path)
            try: 
                for root, _, files in os.walk(path):
                    for file in files:
                        if FileManager.get_file_extension(file) == BinaryTTMetal.get_file_extension()
                            files.append(os.path.join(root, file))
            except Exception as e:
                raise Exception(f"an unexpected error occurred: {e}")

        return files
