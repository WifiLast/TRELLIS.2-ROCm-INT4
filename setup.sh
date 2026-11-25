#!/bin/bash
# Build .venv for TRELLIS.2-ROCm-INT4 on AMD (ROCm) or NVIDIA (CUDA) GPUs.
#   ./setup.sh            platform detected (nvidia-smi -> cuda, rocminfo -> rocm)
#   ./setup.sh --cuda     force NVIDIA      ./setup.sh --rocm   force AMD
# env: TORCH_INDEX_URL (PyTorch wheel index; default: rocm7.2 with a ROCm 7 install, else rocm6.4 / cu128). CUDA builds
#      need a CUDA Toolkit whose major version matches the PyTorch wheel (set CUDA_HOME if several are installed).
set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PLATFORM="${TRELLIS_PLATFORM:-}"
for arg in "$@"; do
    case "$arg" in
        --cuda) PLATFORM=cuda ;;
        --rocm) PLATFORM=rocm ;;
        -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done
if [ -z "$PLATFORM" ]; then
    if command -v nvidia-smi &> /dev/null; then
        PLATFORM=cuda
    elif command -v rocminfo &> /dev/null; then
        PLATFORM=rocm
    else
        echo "Error: no GPU platform found (neither nvidia-smi nor rocminfo); use --cuda or --rocm." >&2
        exit 1
    fi
fi
echo "Platform: $PLATFORM"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first."
    exit 1
fi

echo "uv is installed, proceeding..."

# Check if .venv exists and remove it
if [ -d ".venv" ]; then
    echo "Removing existing .venv directory..."
    rm -rf .venv
fi

# Create new venv with Python 3.11
echo "Creating new virtual environment with Python 3.11..."
uv venv --python 3.11

echo "Virtual environment created successfully!"

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

source .venv/bin/activate

if [ "$PLATFORM" = "rocm" ]; then
    # PyTorch wheel for the installed ROCm major version: the extensions are compiled with the system hipcc and must
    # use the same HIP runtime as torch
    ROCM_VER="$(cat "${ROCM_PATH:-/opt/rocm}/.info/version" 2>/dev/null || true)"
    case "$ROCM_VER" in
        7.*) DEFAULT_INDEX=https://download.pytorch.org/whl/rocm7.2 ;;
        *)   DEFAULT_INDEX=https://download.pytorch.org/whl/rocm6.4 ;;
    esac
    echo "Installing PyTorch with ROCm support (system ROCm ${ROCM_VER:-unknown})..."
    uv pip install torch torchvision --index-url "${TORCH_INDEX_URL:-$DEFAULT_INDEX}"
    # flash-attn below: Triton backend (the default CK backend is a very long hipcc build)
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
else
    echo "Installing PyTorch with CUDA support..."
    uv pip install torch torchvision --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
fi

echo "Installing additional dependencies..."
uv pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers zstandard huggingface_hub OpenEXR
uv pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
uv pip install pillow-simd
uv pip install kornia timm
# flash-attn has no kernels before Ampere (compute capability 8.0): those GPUs use PyTorch SDPA (run.sh)
MIN_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -n | head -1 || true)"
if [ "$PLATFORM" = "cuda" ] && [ -n "$MIN_CC" ] && [ "${MIN_CC%%.*}" -lt 8 ]; then
    echo "GPU compute capability $MIN_CC < 8.0: skipping flash-attn (attention runs on PyTorch SDPA)"
else
    uv pip install flash-attn==2.8.3 --no-build-isolation
fi

echo "Dependencies installed successfully!"

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# AMD: the CUDA extensions are converted with hipify and patched for ROCm / RDNA3
build_rocm() {
# Torch headers after hipify: older hipify-perl writes ATen/cuda/HIPContext.h, hipify-perl 7.x leaves
# ATen/cuda/CUDAContext.h / c10/cuda/*.h untouched; the ROCm torch wheel has them only under ATen/hip, c10/hip.
fix_torch_includes() {
    find . -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" \) -not -path "*/eigen/*" \
        -exec sed -i -e 's|ATen/cuda/HIPContext\.h|ATen/hip/HIPContext.h|g' \
                     -e 's|ATen/cuda/CUDAContext\.h|ATen/hip/HIPContext.h|g' \
                     -e 's|c10/cuda/CUDAStream\.h|c10/hip/HIPStream.h|g' \
                     -e 's|c10/cuda/CUDAGuard\.h|c10/hip/HIPGuard.h|g' {} +
}

# PyTorch >= 2.10 headers need C++20; the extensions' setup.py files ask for C++17.
cxx20() {
    find "$1" -name setup.py -not -path "*/eigen/*" \
        -exec sed -i -e 's/c++17/c++20/g' -e 's/^cpp_standard = 17/cpp_standard = 20/' {} +
}

# nvdiffrast-hip
if [ -d "./nvdiffrast-hip" ]; then
    rm -rf ./nvdiffrast-hip
fi

git clone https://github.com/CalebisGross/TRELLIS-AMD ./trellis
cd ./trellis
git checkout 23a1b81906e39bb730659b71760c69440ea0c001
mv ./extensions/nvdiffrast-hip/ ../
cd ..
rm -rf ./trellis
cd ./nvdiffrast-hip
# Rasterizer fixes for RDNA3 (see patches/nvdiffrast-hip-rocm-fixes.patch):
# lane masks under divergence (dropped/duplicated triangles), depth peeling,
# __syncwarp memory fence, coarse warp-emit arrays back in LDS.
patch -p1 < "$SCRIPT_DIR/patches/nvdiffrast-hip-rocm-fixes.patch"
uv pip install . --no-build-isolation

cd ..

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Prepare nvdiffrec with hipify conversion
if [ -d "./nvdiffrec" ]; then
    rm -rf ./nvdiffrec
fi

echo "Cloning nvdiffrec repository..."
git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git
cd ./nvdiffrec

echo "Converting CUDA code to ROCm using hipify..."
# Find and convert all CUDA files to HIP
find . -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" \) -exec hipify-perl {} -inplace \;

echo "Applying ROCm-specific fixes..."
# Fix __shfl_*_sync mask - ROCm requires 64-bit mask instead of 32-bit
find . -type f \( -name "*.cu" -o -name "*.hip" -o -name "*.cpp" -o -name "*.h" \) -exec sed -i 's/__shfl\([^(]*\)_sync(0xFFFFFFFF,/__shfl\1_sync(0xFFFFFFFFFFFFFFFFULL,/g' {} \;

# Torch CUDA headers -> ATen/hip, c10/hip (see fix_torch_includes)
fix_torch_includes

# Remove CUDAUtils.h include as it's not needed and not available in ROCm PyTorch
find . -type f \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's|#include <ATen/cuda/CUDAUtils\.h>|// #include <ATen/cuda/CUDAUtils.h> // Removed for ROCm|g' {} \;

# Fix getCurrentCUDAStream to getCurrentHIPStream for ROCm
find . -type f \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's/at::cuda::getCurrentCUDAStream/at::hip::getCurrentHIPStream/g' {} \;

# Update setup.py to use CppExtension instead of CUDAExtension
if [ -f "setup.py" ]; then
    # Change imports
    sed -i 's/from torch.utils.cpp_extension import BuildExtension, CUDAExtension/from torch.utils.cpp_extension import BuildExtension, CppExtension/g' setup.py

    # Change CUDAExtension to CppExtension
    sed -i 's/CUDAExtension(/CppExtension(/g' setup.py

    # Add ROCm include path to c_flags (needed for .cpp files that include HIP headers)
    sed -i "s/c_flags = \['-DNVDR_TORCH'\]/c_flags = ['-DNVDR_TORCH', '-I\/opt\/rocm\/include']/g" setup.py

    # Change nvcc_flags to hipcc_flags (variable name and references)
    sed -i "s/nvcc_flags = /hipcc_flags = /g" setup.py
    sed -i "s/nvcc_flags/hipcc_flags/g" setup.py

    # Change 'nvcc': to 'hipcc': in extra_compile_args
    sed -i "s/'nvcc':/'hipcc':/g" setup.py

    # Fix linker flags for ROCm
    sed -i "s/'-lcuda', '-lnvrtc'/'-lamdhip64'/g" setup.py
    sed -i "s/'cuda.lib', 'advapi32.lib', 'nvrtc.lib'/'amdhip64.lib', 'advapi32.lib'/g" setup.py

    # Note: We keep .cu extensions in setup.py even after hipify conversion
    # The files remain as .cu but contain HIP code, which hipcc can compile
    # hipify modifies files in-place, so torch_bindings.cpp stays as torch_bindings.cpp
fi

cd ..
echo "Installing nvdiffrec..."
uv pip install ./nvdiffrec --no-build-isolation

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# CuMesh with hipify conversion
if [ -d "./CuMesh" ]; then
    rm -rf ./CuMesh
fi

echo "Cloning CuMesh repository..."
git clone https://github.com/JeffreyXiang/CuMesh.git --recursive
cd ./CuMesh

echo "Converting CUDA code to ROCm using hipify..."
# Hipify main sources
find src -name "*.cu" -o -name "*.cpp" | xargs -I {} hipify-perl {} -inplace
# Hipify cubvh sources
find third_party/cubvh/src -name "*.cu" -o -name "*.cpp" | xargs -I {} hipify-perl {} -inplace
# Hipify headers (exclude Eigen - it already supports HIP)
find . -name "*.cuh" -o -name "*.h" | grep -v ".prehip" | grep -v "third_party/cubvh/third_party/eigen" | xargs -I {} hipify-perl {} -inplace

echo "Applying ROCm-specific fixes..."
# Fix __shfl_*_sync mask - ROCm requires 64-bit mask
find . -type f \( -name "*.cu" -o -name "*.hip" -o -name "*.cpp" -o -name "*.h" -o -name "*.cuh" \) -exec sed -i 's/__shfl\([^(]*\)_sync(0xFFFFFFFF,/__shfl\1_sync(0xFFFFFFFFFFFFFFFFULL,/g' {} \;

# Torch CUDA headers -> ATen/hip, c10/hip (see fix_torch_includes)
fix_torch_includes

# Remove CUDAUtils includes
find . -type f \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's|#include <ATen/cuda/CUDAUtils\.h>|// #include <ATen/cuda/CUDAUtils.h> // Removed for ROCm|g' {} \;

# Fix stream API
find . -type f \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's/at::cuda::getCurrentCUDAStream/at::hip::getCurrentHIPStream/g' {} \;

# Fix cuda::std namespace to std (thrust doesn't have tuple in ROCm)
sed -i 's/::cuda::std::/::std::/g' src/clean_up.cu

# Apply comprehensive fix to clean_up.cu for ROCm compatibility
echo "Applying ROCm compatibility fixes to clean_up.cu..."

# Add thrust includes after hipcub include
sed -i '/#include <hipcub\/hipcub.hpp>/a #include <thrust/sort.h>\n#include <thrust/device_ptr.h>' src/clean_up.cu

# Replace int3_decomposer with int3_comparator
sed -i '/struct int3_decomposer/,/^};/c\
// Comparator for int3 to sort lexicographically (x, y, z)\
struct int3_comparator\
{\
    __host__ __device__ bool operator()(const int3\& a, const int3\& b) const\
    {\
        if (a.x != b.x) return a.x < b.x;\
        if (a.y != b.y) return a.y < b.y;\
        return a.z < b.z;\
    }\
};' src/clean_up.cu

# Use Python to replace DeviceRadixSort with thrust (handles multiline better)
python3 << 'PYTHON_SCRIPT'
import re

with open('src/clean_up.cu', 'r') as f:
    content = f.read()

# Pattern to match the two DeviceRadixSort calls plus the resize between them
# (CuMesh since 4843631 passes the CUDA stream as the last argument)
pattern = r'    CUDA_CHECK\(hipcub::DeviceRadixSort::SortPairs\(\s*nullptr,.*?\)\);\s*this->cub_temp_storage\.resize\(temp_storage_bytes\);\s*CUDA_CHECK\(hipcub::DeviceRadixSort::SortPairs\(.*?int3_decomposer\{\}\s*(,\s*stream\s*)?\)\);'

def replacement(m):
    # with a stream argument the preceding kernels ran on `stream`: wait for them before the null-stream copies
    sync = '    CUDA_CHECK(hipStreamSynchronize(stream));\n' if m.group(1) else ''
    return sync + '''    // Use thrust::sort_by_key instead of hipcub with decomposer (not supported in ROCm)
    CUDA_CHECK(hipMemcpy(cu_sorted_faces_output, cu_sorted_faces, F * sizeof(int3), hipMemcpyDeviceToDevice));
    CUDA_CHECK(hipMemcpy(cu_sorted_indices_output, cu_sorted_face_indices, F * sizeof(int), hipMemcpyDeviceToDevice));

    thrust::device_ptr<int3> faces_ptr(cu_sorted_faces_output);
    thrust::device_ptr<int> indices_ptr(cu_sorted_indices_output);
    thrust::sort_by_key(faces_ptr, faces_ptr + F, indices_ptr, int3_comparator());

    CUDA_CHECK(hipDeviceSynchronize());'''

content, n = re.subn(pattern, replacement, content, flags=re.DOTALL)
if n != 1:
    raise SystemExit(f"CuMesh clean_up.cu: DeviceRadixSort patch matched {n} times (expected 1); CuMesh changed upstream")

with open('src/clean_up.cu', 'w') as f:
    f.write(content)
PYTHON_SCRIPT

# Add __host__ to Vec3f constructors for host-side compatibility (both declarations and implementations)
# Fix declarations in class
sed -i 's/__device__ __forceinline__ Vec3f();/__host__ __device__ __forceinline__ Vec3f();/g' src/dtypes.cuh
sed -i 's/__device__ __forceinline__ Vec3f(float x, float y, float z);/__host__ __device__ __forceinline__ Vec3f(float x, float y, float z);/g' src/dtypes.cuh
sed -i 's/__device__ __forceinline__ Vec3f(float3 v);/__host__ __device__ __forceinline__ Vec3f(float3 v);/g' src/dtypes.cuh
# Fix implementations
sed -i 's/__device__ __forceinline__ Vec3f::Vec3f/__host__ __device__ __forceinline__ Vec3f::Vec3f/g' src/dtypes.cuh

# Update CuMesh setup.py
if [ -f "setup.py" ]; then
    # Change CUDAExtension to CppExtension
    sed -i 's/from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION/from torch.utils.cpp_extension import CppExtension, BuildExtension, IS_HIP_EXTENSION/g' setup.py
    sed -i 's/CUDAExtension(/CppExtension(/g' setup.py

    # Change nvcc to hipcc in extra_compile_args
    sed -i 's/"nvcc":/"hipcc":/g' setup.py

    # Add ROCm include path for C++ files
    sed -i 's/"cxx": \["-O3", "-std=c++17"\]/"cxx": ["-O3", "-std=c++17", "-I\/opt\/rocm\/include"]/g' setup.py
fi

# Update cubvh setup.py
if [ -f "third_party/cubvh/setup.py" ]; then
    sed -i 's/from torch.utils.cpp_extension import BuildExtension, CUDAExtension/from torch.utils.cpp_extension import BuildExtension, CppExtension/g' third_party/cubvh/setup.py
    sed -i 's/CUDAExtension(/CppExtension(/g' third_party/cubvh/setup.py
    sed -i "s/'nvcc':/'hipcc':/g" third_party/cubvh/setup.py
fi

# Fix Eigen for HIP in cubvh
echo "Fixing Eigen for ROCm/HIP compatibility..."
# Fix Eigen Core to detect HIP using __HIP_PLATFORM_AMD__
if [ -f "third_party/cubvh/third_party/eigen/Eigen/Core" ]; then
    sed -i 's|#if defined(EIGEN_CUDACC)|#if defined(__HIPCC__) \|\| defined(__HIP_PLATFORM_AMD__)\
#include <hip/hip_runtime.h>\
#elif defined(EIGEN_CUDACC)|g' third_party/cubvh/third_party/eigen/Eigen/Core
fi

# Fix Eigen Meta.h to use correct HIP math constants path
if [ -f "third_party/cubvh/third_party/eigen/Eigen/src/Core/util/Meta.h" ]; then
    sed -i 's|#include "Eigen/src/Core/arch/HIP/hcc/hip/hip_math_constants.h"|#include <hip/hip_math_constants.h>|g' third_party/cubvh/third_party/eigen/Eigen/src/Core/util/Meta.h
fi

# Fix cubvh thrust and stream APIs
echo "Fixing cubvh thrust and stream APIs..."
find third_party/cubvh -type f \( -name "*.cu" -o -name "*.cpp" -o -name "*.h" -o -name "*.cuh" \) -exec sed -i 's/thrust::cuda::/thrust::hip::/g' {} \;
find third_party/cubvh -type f \( -name "*.cu" -o -name "*.cpp" -o -name "*.h" -o -name "*.cuh" \) -exec sed -i 's/at::cuda::getCurrentCUDAStream/at::hip::getCurrentHIPStream/g' {} \;

cd ..
echo "Installing CuMesh..."
cxx20 ./CuMesh
uv pip install ./CuMesh --no-build-isolation

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# FlexGEMM with hipify conversion
if [ -d "./FlexGEMM" ]; then
    rm -rf ./FlexGEMM
fi

echo "Cloning FlexGEMM repository..."
git clone https://github.com/JeffreyXiang/FlexGEMM.git
cd ./FlexGEMM

echo "Converting CUDA code to ROCm using hipify..."
find flex_gemm/kernels/cuda -type f \( -name "*.cu" -o -name "*.cpp" -o -name "*.cuh" -o -name "*.h" \) | xargs -I {} hipify-perl {} -inplace

echo "Applying ROCm-specific fixes to FlexGEMM..."
# Disable TF32 for ROCm (TF32 is NVIDIA-specific, ROCm only supports IEEE)
sed -i 's/allow_tf32 = True/allow_tf32 = False/g' flex_gemm/kernels/triton/spconv/config.py

# Update setup.py to use CppExtension instead of CUDAExtension
if [ -f "setup.py" ]; then
    sed -i 's/from torch.utils.cpp_extension import CUDAExtension, BuildExtension/from torch.utils.cpp_extension import CppExtension, BuildExtension/g' setup.py
    sed -i 's/CUDAExtension(/CppExtension(/g' setup.py
    sed -i 's/"nvcc":/"hipcc":/g' setup.py
fi

cd ..
echo "Installing FlexGEMM..."
cxx20 ./FlexGEMM
uv pip install ./FlexGEMM --no-build-isolation

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# o-voxel with hipify conversion
if [ -d "./o-voxel-hip" ]; then
    rm -rf ./o-voxel-hip
fi

echo "Copying o-voxel to o-voxel-hip..."
cp -r ./o-voxel ./o-voxel-hip
cd ./o-voxel-hip

echo "Converting CUDA code to ROCm using hipify..."
# Hipify main sources
find src -type f \( -name "*.cu" -o -name "*.cpp" -o -name "*.cuh" -o -name "*.h" \) | xargs -I {} hipify-perl {} -inplace

echo "Applying ROCm-specific fixes to o-voxel..."
# Fix __shfl_*_sync mask - ROCm requires 64-bit mask
find . -type f \( -name "*.cu" -o -name "*.hip" -o -name "*.cpp" -o -name "*.h" -o -name "*.cuh" \) -exec sed -i 's/__shfl\([^(]*\)_sync(0xFFFFFFFF,/__shfl\1_sync(0xFFFFFFFFFFFFFFFFULL,/g' {} \;

# Torch CUDA headers -> ATen/hip, c10/hip (see fix_torch_includes)
fix_torch_includes

# Remove CUDAUtils includes
find . -type f \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's|#include <ATen/cuda/CUDAUtils\.h>|// #include <ATen/cuda/CUDAUtils.h> // Removed for ROCm|g' {} \;

# Fix stream API
find . -type f \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's/at::cuda::getCurrentCUDAStream/at::hip::getCurrentHIPStream/g' {} \;

# Remove local struct definitions that conflict with HIP types
# HIP already provides float3, int3, int4 but NOT bool3
sed -i '/^struct float3 {float x, y, z;/d' src/convert/flexible_dual_grid.cpp
sed -i '/^struct int3 {int x, y, z;/d' src/convert/flexible_dual_grid.cpp
sed -i '/^struct int4 {int x, y, z, w;/d' src/convert/flexible_dual_grid.cpp
# Keep bool3 as it's not provided by HIP

# Update setup.py to use CppExtension instead of CUDAExtension
if [ -f "setup.py" ]; then
    # Change CUDAExtension to CppExtension
    sed -i 's/from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION/from torch.utils.cpp_extension import CppExtension, BuildExtension, IS_HIP_EXTENSION/g' setup.py
    sed -i 's/CUDAExtension(/CppExtension(/g' setup.py

    # Change nvcc to hipcc in extra_compile_args
    sed -i 's/"nvcc":/"hipcc":/g' setup.py

    # Add ROCm include path and Eigen path for C++ files
    # Use Python to dynamically insert the path
    python3 << 'PYTHON_SCRIPT'
import re
with open('setup.py', 'r') as f:
    content = f.read()

# Replace cxx flags
content = re.sub(
    r'"cxx": \["-O3", "-std=c\+\+17"\]',
    '"cxx": ["-I/opt/rocm/include", f"-I{os.path.join(ROOT, \'third_party/eigen\')}", "-O3", "-std=c++17"]',
    content
)

with open('setup.py', 'w') as f:
    f.write(content)
PYTHON_SCRIPT
fi

# Download Eigen if not present
if [ ! -d "third_party/eigen" ] || [ -z "$(ls -A third_party/eigen 2>/dev/null)" ]; then
    echo "Downloading Eigen library..."
    mkdir -p third_party
    rm -rf third_party/eigen
    git clone --depth 1 https://gitlab.com/libeigen/eigen.git third_party/eigen
fi

# Update pyproject.toml to use local CuMesh and FlexGEMM
if [ -f "pyproject.toml" ]; then
    # Comment out git dependencies - we'll use already installed local versions
    sed -i 's|"cumesh @ git+https://github.com/JeffreyXiang/CuMesh.git",|# "cumesh @ git+https://github.com/JeffreyXiang/CuMesh.git",  # Use locally installed version|g' pyproject.toml
    sed -i 's|"flex_gemm @ git+https://github.com/JeffreyXiang/FlexGEMM.git",|# "flex_gemm @ git+https://github.com/JeffreyXiang/FlexGEMM.git",  # Use locally installed version|g' pyproject.toml
fi

cd ..
echo "Installing o-voxel-hip..."
cxx20 ./o-voxel-hip
uv pip install ./o-voxel-hip --no-build-isolation
}

# NVIDIA: the extensions build unchanged (as in microsoft/TRELLIS.2's setup.sh)
build_cuda() {
    rm -rf ./nvdiffrast ./nvdiffrec ./CuMesh ./FlexGEMM ./o-voxel-cuda

    echo "Installing nvdiffrast..."
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git ./nvdiffrast
    uv pip install ./nvdiffrast --no-build-isolation

    echo "Installing nvdiffrec..."
    git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git ./nvdiffrec
    uv pip install ./nvdiffrec --no-build-isolation

    echo "Installing CuMesh..."
    git clone --recursive https://github.com/JeffreyXiang/CuMesh.git ./CuMesh
    uv pip install ./CuMesh --no-build-isolation

    echo "Installing FlexGEMM..."
    git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git ./FlexGEMM
    uv pip install ./FlexGEMM --no-build-isolation

    echo "Installing o-voxel..."
    cp -r ./o-voxel ./o-voxel-cuda
    if [ -z "$(ls -A o-voxel-cuda/third_party/eigen 2>/dev/null)" ]; then
        rm -rf o-voxel-cuda/third_party/eigen
        git clone --depth 1 https://gitlab.com/libeigen/eigen.git o-voxel-cuda/third_party/eigen
    fi
    # CuMesh / FlexGEMM are installed above; keep pip from fetching them again from git
    sed -i 's|^\(\s*\)"cumesh @ git+|\1# "cumesh @ git+|; s|^\(\s*\)"flex_gemm @ git+|\1# "flex_gemm @ git+|' o-voxel-cuda/pyproject.toml
    uv pip install ./o-voxel-cuda --no-build-isolation
}

if [ "$PLATFORM" = "rocm" ]; then
    build_rocm
else
    build_cuda
fi

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

uv pip install transparent-background
echo "$PLATFORM" > .venv/trellis_platform
echo "Setup finished ($PLATFORM)."
