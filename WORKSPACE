workspace(name = "tt-mlir")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

LLVM_COMMIT = "9ddfe62f5c11e3f65f444209f514029ded2d58b9"

LLVM_SHA256 = "cb59f31fd0060e9d6f1142c702cad742b52a294ef9dbed87a864213fdcc007cd"

http_archive(
    name = "llvm-raw",
    build_file_content = "# empty",
    sha256 = LLVM_SHA256,
    strip_prefix = "llvm-project-" + LLVM_COMMIT,
    urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
)

# This is needed since https://reviews.llvm.org/D143344.
# Not sure if it's a bug or a feature, but it doesn't hurt to keep an additional
# dependency here.
http_archive(
    name = "llvm_zstd",
    build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
    sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
    strip_prefix = "zstd-1.5.2",
    urls = [
        "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz",
    ],
)

# This is needed since https://reviews.llvm.org/D143320
# Not sure if it's a bug or a feature, but it doesn't hurt to keep an additional
# dependency here.
http_archive(
    name = "llvm_zlib",
    build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
    sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
    strip_prefix = "zlib-ng-2.0.7",
    urls = [
        "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
    ],
)


load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")

llvm_configure(name = "llvm-project")

#load flatbuffer
FLATBUFFERS_COMMIT = "fb9afbafc7dfe226b9db54d4923bfb8839635274"
http_archive(
    name = "flatbuffers",
    urls = ["https://github.com/google/flatbuffers/archive/{commit}.zip".format(commit = FLATBUFFERS_COMMIT)],
    strip_prefix = "flatbuffers-" + FLATBUFFERS_COMMIT,
    # build_file_content = "# empty",
)

# Add rules_foreign_cc for cmake_external
RULES_FOREIGN_CC_TAG = "0.11.1"
http_archive(
    name = "rules_foreign_cc",
    strip_prefix = "rules_foreign_cc-" + RULES_FOREIGN_CC_TAG,
    urls = ["https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/{tag}.zip".format(tag = RULES_FOREIGN_CC_TAG)],
)
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
rules_foreign_cc_dependencies()


_ALL_CONTENT = """\
filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
"""

TT_METAL_TAG = "v0.49.0"
git_repository(
    name = "tt-metal",
    tag = TT_METAL_TAG,
    build_file_content = _ALL_CONTENT,
    recursive_init_submodules = True,
    remote = "https://github.com/tenstorrent/tt-metal.git",
    patch_cmds = [
        "rm tt_metal/hw/inc/wormhole/BUILD.bazel",
        "rm tt_metal/hw/toolchain/BUILD",
    ]
)
