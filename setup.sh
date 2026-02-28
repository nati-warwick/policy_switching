#!/usr/bin/env bash
set -euo pipefail

CONDA_DIR="${HOME}/miniconda3"
CONDA_BIN=""
ENV_NAME="rlenv"
MUJOCO_DIR="${HOME}/.mujoco"
MUJOCO_VERSION="mujoco210"
MUJOCO_TARBALL="${MUJOCO_DIR}/${MUJOCO_VERSION}.tar.gz"
D4RL_SRC="git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl"

download_file() {
    local url="$1"
    local out="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$out"
    elif command -v wget >/dev/null 2>&1; then
        wget "$url" -O "$out"
    else
        echo "Error: neither curl nor wget is available."
        exit 1
    fi
}

resolve_conda_bin() {
    if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
        CONDA_BIN="${CONDA_EXE}"
        return
    fi

    if command -v conda >/dev/null 2>&1; then
        CONDA_BIN="$(command -v conda)"
        return
    fi

    if [ -x "${CONDA_DIR}/bin/conda" ]; then
        CONDA_BIN="${CONDA_DIR}/bin/conda"
        return
    fi
}

install_miniconda_if_needed() {
    if [ -n "$CONDA_BIN" ] && [ -x "$CONDA_BIN" ]; then
        return
    fi

    mkdir -p "$CONDA_DIR"

    local os arch installer_url installer
    os="$(uname -s)"
    arch="$(uname -m)"
    installer="${CONDA_DIR}/miniconda.sh"

    case "$os-$arch" in
        Linux-x86_64)
            installer_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            ;;
        Darwin-arm64)
            installer_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
            ;;
        Darwin-x86_64)
            installer_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
            ;;
        *)
            echo "Unsupported platform for automatic Miniconda install: ${os}-${arch}"
            exit 1
            ;;
    esac

    download_file "$installer_url" "$installer"
    bash "$installer" -b -u -p "$CONDA_DIR"
    rm -f "$installer"
    CONDA_BIN="${CONDA_DIR}/bin/conda"
}

create_or_update_env() {
    if "$CONDA_BIN" run -n "$ENV_NAME" python -V >/dev/null 2>&1; then
        "$CONDA_BIN" env update -n "$ENV_NAME" -f environment.yml --prune
    else
        "$CONDA_BIN" env create -f environment.yml
    fi
}

install_d4rl_without_solver_conflicts() {
    "$CONDA_BIN" run -n "$ENV_NAME" python -m pip install --no-deps "$D4RL_SRC"
}

install_d4rl_required_extras() {
    "$CONDA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade --no-cache-dir \
        "mjrl @ git+https://github.com/aravindr93/mjrl@master#egg=mjrl"
}

enforce_numpy_pin() {
    "$CONDA_BIN" run -n "$ENV_NAME" python -m pip install --upgrade --force-reinstall "numpy==1.26.4"
}

configure_env_compiler_vars() {
    local os
    os="$(uname -s)"
    if [ "$os" != "Darwin" ]; then
        return
    fi

    local sdkroot
    sdkroot="$(xcrun --sdk macosx --show-sdk-path)"
    local mujoco_path="${MUJOCO_DIR}/${MUJOCO_VERSION}"

    "$CONDA_BIN" env config vars set -n "$ENV_NAME" \
        CC=/usr/bin/clang \
        CXX=/usr/bin/clang++ \
        SDKROOT="$sdkroot" \
        CONDA_BUILD_SYSROOT="$sdkroot" \
        MUJOCO_PY_MUJOCO_PATH="$mujoco_path" >/dev/null
}

install_d4rl_runtime_deps() {
    local mujoco_path="${MUJOCO_DIR}/${MUJOCO_VERSION}"
    local os
    os="$(uname -s)"

    # Remove cached mujoco-py builds so compiler/toolchain changes take effect.
    rm -rf "${MUJOCO_DIR}/mujoco_py"

    if [ "$os" = "Darwin" ]; then
        if ! xcode-select -p >/dev/null 2>&1; then
            echo "Xcode Command Line Tools are required on macOS."
            echo "Run: xcode-select --install"
            exit 1
        fi

        local sdkroot
        sdkroot="$(xcrun --sdk macosx --show-sdk-path)"
        local mac_target
        mac_target="$(sw_vers -productVersion | awk -F. '{print $1"."$2}')"
        local arch
        arch="$(uname -m)"
        local archflags
        if [ "$arch" = "arm64" ]; then
            archflags="-arch arm64"
        else
            archflags="-arch x86_64"
        fi

        # Avoid user/global distutils configs forcing old gcc toolchains.
        local tmp_home
        tmp_home="$(mktemp -d)"

        "$CONDA_BIN" run -n "$ENV_NAME" env \
            -u CC -u CXX -u LDSHARED -u CPP -u CFLAGS -u CXXFLAGS -u LDFLAGS \
            HOME="$tmp_home" \
            MUJOCO_PY_MUJOCO_PATH="$mujoco_path" \
            CC=/usr/bin/clang \
            CXX=/usr/bin/clang++ \
            SDKROOT="$sdkroot" \
            CONDA_BUILD_SYSROOT="$sdkroot" \
            DISTUTILS_USE_SDK=1 \
            MACOSX_DEPLOYMENT_TARGET="$mac_target" \
            ARCHFLAGS="$archflags" \
            CFLAGS="$archflags -isysroot $sdkroot" \
            CXXFLAGS="$archflags -isysroot $sdkroot" \
            CPPFLAGS="$archflags -isysroot $sdkroot" \
            python -m pip install --upgrade --no-cache-dir --force-reinstall "cython==0.29.36" wheel "mujoco-py==2.1.2.14" tqdm

        rm -rf "$tmp_home"
    else
        "$CONDA_BIN" run -n "$ENV_NAME" env \
            MUJOCO_PY_MUJOCO_PATH="$mujoco_path" \
            python -m pip install --upgrade "cython==0.29.36" wheel "mujoco-py==2.1.2.14" tqdm
    fi
}

patch_mujoco_py_builder_for_macos() {
    if [ "$(uname -s)" != "Darwin" ]; then
        return
    fi

    "$CONDA_BIN" run -n "$ENV_NAME" python - <<'PY'
import os
import re
import site

builder_path = None
for p in site.getsitepackages():
    cand = os.path.join(p, "mujoco_py", "builder.py")
    if os.path.isfile(cand):
        builder_path = cand
        break

if builder_path is None:
    raise SystemExit(0)

with open(builder_path, "r", encoding="utf-8") as f:
    text = f.read()

patched = re.sub(
    r"\n\s*'-fopenmp',\s*# needed for OpenMP",
    "",
    text,
    count=1,
)
patched = patched.replace("extra_link_args=['-fopenmp'],", "extra_link_args=[],")

if patched != text:
    with open(builder_path, "w", encoding="utf-8") as f:
        f.write(patched)
    print("Patched mujoco_py builder for macOS clang compatibility.")
PY
}

install_mujoco_if_needed() {
    if [ -d "${MUJOCO_DIR}/${MUJOCO_VERSION}" ]; then
        return
    fi

    local os arch mujoco_url
    os="$(uname -s)"
    arch="$(uname -m)"
    case "$os-$arch" in
        Linux-x86_64)
            mujoco_url="https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz"
            ;;
        Darwin-x86_64)
            mujoco_url="https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-macos-x86_64.tar.gz"
            ;;
        Darwin-arm64)
            mujoco_url="https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-macos-arm64.tar.gz"
            ;;
        *)
            echo "Unsupported platform for MuJoCo download: ${os}-${arch}"
            exit 1
            ;;
    esac

    mkdir -p "$MUJOCO_DIR"
    download_file "$mujoco_url" "$MUJOCO_TARBALL"
    tar -xzf "$MUJOCO_TARBALL" -C "$MUJOCO_DIR"
    rm -f "$MUJOCO_TARBALL"
}

install_system_deps_if_available() {
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get install -y libglew-dev
    else
        echo "Skipping libglew install (apt-get not available on this platform)."
    fi
}

unpack_models_if_present() {
    if [ -f "models.zip" ]; then
        unzip -o models.zip
    else
        echo "Skipping models.zip extraction (file not found)."
    fi
}

main() {
    resolve_conda_bin
    install_miniconda_if_needed
    create_or_update_env
    install_mujoco_if_needed
    install_system_deps_if_available
    configure_env_compiler_vars
    install_d4rl_runtime_deps
    patch_mujoco_py_builder_for_macos
    install_d4rl_without_solver_conflicts
    install_d4rl_required_extras
    enforce_numpy_pin
    unpack_models_if_present

    echo "Setup complete."
    echo "Run: ${CONDA_BIN} run -n ${ENV_NAME} python smoke_test_compat.py --env-id halfcheetah-medium-v2"
}

main "$@"
