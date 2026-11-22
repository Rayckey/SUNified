#!/usr/bin/env bash
# Separate, inference-focused SAM 3D runtime. Does not download gated model weights.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
sam_root="$("${KUAFU_PYTHON:-python}" -c 'import os; from kuafu.paths import resolve_artifact_path; print(resolve_artifact_path(os.environ.get("SAM3D_ROOT") or "assets/sam3d", "SAM 3D runtime"))')"
sam_conda="${SAM3D_CONDA:-$(command -v conda || true)}"
if [[ -z "${sam_conda}" ]]; then
    printf '%s\n' 'Set SAM3D_CONDA to the conda executable or add conda to PATH.' >&2
    exit 1
fi
mkdir -p "$sam_root"/{cache/{conda,pip,torch,huggingface,extensions},tmp,logs}
export CONDA_PKGS_DIRS="$sam_root/cache/conda"
export PIP_CACHE_DIR="$sam_root/cache/pip"
export HF_HUB_CACHE="$sam_root/cache/huggingface"
export TORCH_HOME="$sam_root/cache/torch"
export TORCH_EXTENSIONS_DIR="$sam_root/cache/extensions"
export TMPDIR="$sam_root/tmp"
if [[ ! -d "$sam_root/repo/.git" ]]; then
    git -c core.hooksPath=/dev/null clone --depth 1 \
        https://github.com/facebookresearch/sam-3d-objects.git "$sam_root/repo"
fi
sam_revision="${SAM3D_REVISION:-f91db411c50efee93d8db7aeb323885650f6f722}"
if [[ "$(git -C "$sam_root/repo" rev-parse HEAD)" != "$sam_revision" ]]; then
    printf '%s\n' "Checkout revision differs from tested SAM3D_REVISION=$sam_revision; review it before continuing." >&2
    exit 1
fi
if [[ ! -x "$sam_root/env/bin/python" ]]; then
    "$sam_conda" create --yes --prefix "$sam_root/env" --override-channels -c conda-forge \
        python=3.11 pip 'setuptools<76' 'cuda-toolkit=12.1' 'gcc=12.4' 'gxx=12.4'
fi
export CONDA_PREFIX="$sam_root/env"
export CUDA_HOME="$sam_root/env"
export PATH="$sam_root/env/bin:$PATH"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
# Compile only for the requested architecture; RTX 8000 is Turing (7.5).
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
export MAX_JOBS="${MAX_JOBS:-4}"
export FORCE_CUDA=1
export CC="$sam_root/env/bin/x86_64-conda-linux-gnu-cc"
export CXX="$sam_root/env/bin/x86_64-conda-linux-gnu-c++"
# Current conda-forge CUDA packages place headers in targets/, not include/.
export CPATH="$sam_root/env/targets/x86_64-linux/include"
export LIBRARY_PATH="$sam_root/env/targets/x86_64-linux/lib"
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121
export PIP_CONSTRAINT="$script_dir/sam3d-constraints.txt"
python -m pip install -r "$script_dir/sam3d-runtime.txt"
python -m pip install --no-build-isolation \
    'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47' \
    'gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7' \
    'MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b'
python -m pip install kaolin==0.17.0 \
    --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html
# The UV baker uses utils3d's CUDA rasterizer even with rendering_engine=pytorch3d.
python -m pip install --no-build-isolation \
    'nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8'
python -m pip install --no-deps -e "$sam_root/repo"
# The official dependency list requests FlashAttention 2, which doesn't support Turing.
# This integration uses xformers attention and float16 instead. No FlashAttention install.
python "$sam_root/repo/patching/hydra"
python -m pip freeze > "$sam_root/logs/runtime-freeze.txt"
git -C "$sam_root/repo" rev-parse HEAD > "$sam_root/logs/repo-revision.txt"
printf '%s\n' 'Runtime packages installed. Run kuafu assets sam3d-doctor to validate imports and GPU kernels.'
