#!/usr/bin/env bash

REPO_ROOT="$(git rev-parse --show-toplevel)"

LLVM_TARGETS=${LLVM_TARGETS:-Native;NVPTX;AMDGPU}
LLVM_PROJECTS=${LLVM_PROJECTS:-mlir;llvm;lld;clang}
LLVM_DISTRIBUTION_COMPONENTS=${LLVM_DISTRIBUTION_COMPONENTS:-"llvm-headers;llvm-libraries;cmake-exports;mlir-headers;mlir-libraries;mlir-cmake-exports;lld-headers;lld-libraries;lld-cmake-exports;clang;clang-resource-headers;FileCheck;not;split-file;llc;opt;llvm-config;mlir-tblgen;mlir-translate"}
LLVM_BUILD_TYPE=${LLVM_BUILD_TYPE:-RelWithDebInfo}
LLVM_COMMIT_HASH=${LLVM_COMMIT_HASH:-$(jq -r '.llvm_hash' "$REPO_ROOT/cmake/llvm-info.json")}
LLVM_PROJECT_PATH=${LLVM_PROJECT_PATH:-"$REPO_ROOT/llvm-project"}
LLVM_BUILD_PATH=${LLVM_BUILD_PATH:-"$LLVM_PROJECT_PATH/build"}
LLVM_INSTALL_PATH=${LLVM_INSTALL_PATH:-"$LLVM_PROJECT_PATH/install"}
LLVM_PROJECT_URL=${LLVM_PROJECT_URL:-"https://github.com/llvm/llvm-project"}

# Build+link with clang+lld only when explicitly requested (the macOS toolchain
# does not ship lld). Otherwise fall back to the default compiler and linker.
CLANG_LLD_ARGS=()
case "$(printf '%s' "$TRITON_BUILD_WITH_CLANG_LLD" | tr '[:upper:]' '[:lower:]')" in
    1 | on | true) TRITON_BUILD_WITH_CLANG_LLD=1 ;;
esac
if [ "$TRITON_BUILD_WITH_CLANG_LLD" = "1" ]; then
    CLANG_LLD_ARGS=(
        -DCMAKE_C_COMPILER=clang
        -DCMAKE_CXX_COMPILER=clang++
        -DLLVM_ENABLE_LLD=ON
    )
fi

if [ -z "$CMAKE_ARGS" ]; then
    if [ "$#" -eq 0 ]; then
        CMAKE_ARGS=(
            -G Ninja
              -DCMAKE_BUILD_TYPE="$LLVM_BUILD_TYPE"
              -DLLVM_CCACHE_BUILD=OFF
              -DLLVM_ENABLE_ASSERTIONS=ON
              "${CLANG_LLD_ARGS[@]}"
              -DLLVM_OPTIMIZED_TABLEGEN=ON
              -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
              -DLLVM_ENABLE_ZSTD=OFF
              -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS"
              -DCMAKE_EXPORT_COMPILE_COMMANDS=1
              -DLLVM_ENABLE_PROJECTS="$LLVM_PROJECTS"
              -DLLVM_BUILD_UTILS=ON
              -DLLVM_INSTALL_UTILS=ON
              -DLLVM_DISTRIBUTION_COMPONENTS="$LLVM_DISTRIBUTION_COMPONENTS"
              -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_PATH"
              -B"$LLVM_BUILD_PATH" "$LLVM_PROJECT_PATH/llvm"
        )
    else
        CMAKE_ARGS=("$@")
    fi
fi

if [ -n "$LLVM_CLEAN" ] && [ -e "$LLVM_PROJECT_PATH" ]; then
    rm -rf "$LLVM_PROJECT_PATH"
fi

if [ ! -e "$LLVM_PROJECT_PATH" ]; then
    echo "Cloning from $LLVM_PROJECT_URL"
    git clone "$LLVM_PROJECT_URL" "$LLVM_PROJECT_PATH"
fi
echo "Resetting to $LLVM_COMMIT_HASH"
git -C "$LLVM_PROJECT_PATH" fetch origin "$LLVM_COMMIT_HASH"
git -C "$LLVM_PROJECT_PATH" reset --hard "$LLVM_COMMIT_HASH"
# `git reset --hard` reverts tracked edits but leaves untracked files behind
# (e.g. SMTRealOps.td created by the patch on a previous run), which would make
# `git apply` fail the second time. Drop them so patch application is idempotent.
git -C "$LLVM_PROJECT_PATH" clean -fdq

# Apply Triton-local MLIR patches on top of the pinned revision. These carry
# changes that are not (yet) upstream in the pinned LLVM but are required to
# build Triton -- e.g. the SMT dialect `Real` sort used by the TritonToSMT
# translation-validation pass (scripts/patches/mlir-smt-real.patch). Because the
# reset above wipes any local edits, the patch must be (re)applied on every build.
for patch in "$REPO_ROOT"/scripts/patches/*.patch; do
    [ -e "$patch" ] || continue
    echo "Applying MLIR patch $(basename "$patch")"
    git -C "$LLVM_PROJECT_PATH" apply "$patch" || {
        echo "ERROR: failed to apply $patch" >&2
        exit 1
    }
done

echo "Configuring with ${CMAKE_ARGS[@]}"
cmake "${CMAKE_ARGS[@]}"
echo "Building LLVM"
ninja -C "$LLVM_BUILD_PATH"
