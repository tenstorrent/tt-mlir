# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
``tt-crank-install-sfpi`` console script.

The wheel does NOT bundle the SFPI RISC-V toolchain (GPLv3 GCC, ~436 MB) — the
payload rules drop it, so tt-metal's kernel JIT falls through to
``/opt/tenstorrent/sfpi`` at runtime. This script provisions that: it reads the
version pinned in ``tt_metal/sfpi-version`` of the resolved tt-metal tree (the
wheel's copy, or the dev submodule), downloads the matching distro package from
the sfpi GitHub releases, verifies its sha256, and installs it with the system
package manager (``apt``/``dnf``/``yum``/``zypper``).

Run once after ``pip install tt-crank`` (needs sudo unless already root):

    tt-crank-install-sfpi

URL, filename and distro naming follow tt-metal's ``tt_metal/sfpi-info.sh``,
which is the upstream source of truth for them.
"""

import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


def _find_value(content: str, name: str) -> str | None:
    """Extract ``name='value'`` (or ``name=value``) from the sfpi-version file.

    Returns None for both a missing key and an empty value (``sfpi_base=''``),
    so callers can use ``or`` to fall back.
    """
    match = re.search(
        rf"^{re.escape(name)}\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|(.*?))\s*$",
        content,
        re.MULTILINE,
    )
    if match is None:
        return None
    value = next(group for group in match.groups() if group is not None)
    return value or None


def _sfpi_version_path() -> Path:
    """Locate the pinned sfpi-version file in the resolved tt-metal tree."""
    from ._runtime_env import tt_metal_home

    return tt_metal_home() / "tt_metal" / "sfpi-version"


def _detect_arch() -> str:
    """Map the host machine to the arch token used in sfpi package names."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    raise RuntimeError(f"unsupported CPU architecture for SFPI: {machine!r}")


def _detect_distro() -> tuple[str, str, list[str]]:
    """Return (dist_token, pkg_ext, install_cmd), following tt-metal's sfpi-info.sh.

    Matches on ID/ID_LIKE only. Just fedora rpms are published, so every rpm
    distro downloads the fedora build.
    """
    try:
        os_release = Path("/etc/os-release").read_text().lower()
    except FileNotFoundError:
        os_release = ""

    if re.search(r"^id(_like)?=.*(fedora|rhel|centos|suse)", os_release, re.MULTILINE):
        for mgr in ("dnf", "yum", "zypper"):
            if shutil.which(mgr):
                return "fedora", "rpm", [mgr, "install", "-y"]
        raise RuntimeError("no dnf/yum/zypper on PATH.")
    if not re.search(r"^id(_like)?=.*(debian|ubuntu)", os_release, re.MULTILINE):
        print("Warning: unknown distribution; defaulting to debian.", file=sys.stderr)
    return "debian", "deb", ["apt-get", "install", "-y", "--allow-downgrades"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    try:
        version_path = _sfpi_version_path()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not version_path.exists():
        print(f"Error: sfpi-version file not found at {version_path}.", file=sys.stderr)
        return 1

    content = version_path.read_text()
    repo = _find_value(content, "sfpi_repo")
    version = _find_value(content, "sfpi_version")
    hashtype = _find_value(content, "sfpi_hashtype") or "sha256"
    # sfpi-info.sh: when sfpi_base is set it replaces the version in both the
    # release tag and the package filename.
    release = _find_value(content, "sfpi_base") or version
    if not repo or not version:
        print(
            f"Error: could not parse sfpi_repo/sfpi_version from {version_path}.",
            file=sys.stderr,
        )
        return 1

    arch = _detect_arch()
    dist, pkg, install_cmd = _detect_distro()

    filename = f"sfpi_{release}_{arch}_{dist}.{pkg}"
    url = f"{repo}/releases/download/{release}/{filename}"
    expected_hash = _find_value(content, f"sfpi_{arch}_{dist}_{pkg}_hash")

    # Both of these mean we cannot verify what we are about to install as root,
    # which is the whole point of reading sfpi-version. Fail like tt-metal's own
    # installer rather than falling back to an unverified install.
    if hashtype != "sha256":
        print(
            f"Error: unsupported hash type {hashtype!r} in {version_path}.",
            file=sys.stderr,
        )
        return 1
    if not expected_hash:
        print(
            f"Error: SFPI {version} {pkg} package for {arch}/{dist} is not "
            f"available (no hash in {version_path}).",
            file=sys.stderr,
        )
        return 1

    print(f"SFPI {version} ({arch}/{dist}) from {url}")
    with tempfile.TemporaryDirectory() as tmp:
        pkg_path = Path(tmp) / filename
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                with pkg_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
        except (urllib.error.URLError, OSError) as exc:
            print(f"Error downloading {url}: {exc}", file=sys.stderr)
            return 1

        actual = _sha256(pkg_path)
        if actual != expected_hash:
            print(
                f"Error: sha256 mismatch for {filename}\n"
                f"  expected {expected_hash}\n  got      {actual}",
                file=sys.stderr,
            )
            return 1
        print("sha256 OK")

        cmd = [*install_cmd, str(pkg_path)]
        if os.geteuid() != 0:
            if not shutil.which("sudo"):
                print(
                    f"Error: installing {filename} needs root, but sudo is not on "
                    "PATH. Re-run as root.",
                    file=sys.stderr,
                )
                return 1
            cmd.insert(0, "sudo")
        print(f"Installing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Error installing SFPI package: {exc}", file=sys.stderr)
            return 1

    print("SFPI installed to /opt/tenstorrent/sfpi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
