# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Configuration handling for tt-alchemist
"""

import os
import json
from pathlib import Path


class Config:
    """Configuration class for tt-alchemist"""

    _instance = None

    @classmethod
    def get_instance(cls):
        """Get the singleton instance"""
        if cls._instance is None:
            cls._instance = Config()
        return cls._instance

    def __init__(self):
        """Initialize the configuration"""
        self.config_data = {
            "tools_path": "",
            "templates_path": "",
            "available_targets": ["grayskull", "wormhole", "blackhole"],
            "available_flavors": ["release", "debug", "profile"],
            "default_target": "grayskull",
            "default_flavor": "release",
            "default_opt_level": "normal",
        }

        # Try to load the configuration from the environment
        self._load_from_environment()

        # Try to load the configuration from the default config file
        config_file = os.environ.get("TTALCHEMIST_CONFIG", "~/.ttalchemist/config.json")
        self.load_from_file(os.path.expanduser(config_file))

    def _load_from_environment(self):
        """Load configuration from environment variables"""
        if "TTALCHEMIST_TOOLS_PATH" in os.environ:
            self.config_data["tools_path"] = os.environ["TTALCHEMIST_TOOLS_PATH"]

        if "TTALCHEMIST_TEMPLATES_PATH" in os.environ:
            self.config_data["templates_path"] = os.environ[
                "TTALCHEMIST_TEMPLATES_PATH"
            ]

        if "TTALCHEMIST_DEFAULT_TARGET" in os.environ:
            self.config_data["default_target"] = os.environ[
                "TTALCHEMIST_DEFAULT_TARGET"
            ]

        if "TTALCHEMIST_DEFAULT_FLAVOR" in os.environ:
            self.config_data["default_flavor"] = os.environ[
                "TTALCHEMIST_DEFAULT_FLAVOR"
            ]

        if "TTALCHEMIST_DEFAULT_OPT_LEVEL" in os.environ:
            self.config_data["default_opt_level"] = os.environ[
                "TTALCHEMIST_DEFAULT_OPT_LEVEL"
            ]

    def load_from_file(self, config_file):
        """Load configuration from a file"""
        try:
            with open(config_file, "r") as f:
                data = json.load(f)
                self.config_data.update(data)
            return True
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    def save_to_file(self, config_file):
        """Save configuration to a file"""
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(config_file), exist_ok=True)

            with open(config_file, "w") as f:
                json.dump(self.config_data, f, indent=2)
            return True
        except Exception:
            return False

    def get(self, key, default=None):
        """Get a configuration value"""
        return self.config_data.get(key, default)

    def set(self, key, value):
        """Set a configuration value"""
        self.config_data[key] = value

    def get_tools_path(self):
        """Get the path to the tt-mlir tools"""
        return self.get("tools_path", "")

    def get_templates_path(self):
        """Get the path to the templates"""
        return self.get("templates_path", "")

    def get_available_targets(self):
        """Get the available hardware targets"""
        return self.get("available_targets", [])

    def get_available_flavors(self):
        """Get the available build flavors"""
        return self.get("available_flavors", [])

    def get_default_target(self):
        """Get the default hardware target"""
        return self.get("default_target", "grayskull")

    def get_default_flavor(self):
        """Get the default build flavor"""
        return self.get("default_flavor", "release")

    def get_default_opt_level(self):
        """Get the default optimization level"""
        return self.get("default_opt_level", "normal")
