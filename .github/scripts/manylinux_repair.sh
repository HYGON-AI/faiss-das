#!/usr/bin/env bash

# Build a portable manylinux wheel while leaving DTK runtime libraries external.
# Usage: manylinux_repair.sh [input-dir] [output-dir] [faiss-library-dir]

if [[ -f /opt/dtk/env.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/dtk/env.sh
fi

set -euo pipefail

input_dir="${1:-build/faiss/python/dist}"
output_dir="${2:-dist-manylinux}"
faiss_library_dir="${3:-build/faiss}"

pip install --no-cache-dir auditwheel patchelf

# _swigfaiss.so links to libfaiss.so from the build tree. auditwheel must be
# able to locate it in order to copy it into the repaired wheel.
export LD_LIBRARY_PATH="${faiss_library_dir}:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/opt/intel/mkl/lib/intel64_lin:${LD_LIBRARY_PATH}"
export AUDITWHEEL_LD_LIBRARY_PATH="${faiss_library_dir}:${AUDITWHEEL_LD_LIBRARY_PATH:-}"

glibc_version="$(ldd --version | awk '{print $NF}' | head -n1 | tr '.' '_')"
platform="manylinux_${glibc_version}_x86_64"

# These libraries belong to the installed DTK/driver runtime and must not be
# bundled into the Faiss wheel.
excluded_libraries=(
  libgalaxyhip.so
  libgalaxyhip.so.5
  libamdhip64.so
  libamdhip64.so.6
  libamd_comgr.so
  libhsa-runtime64.so
  libhipblas.so
  libhipblas.so.2
  libhipfft.so
  libhiprand.so
  libhiprand.so.1
  libhipsolver.so
  libhipsparse.so
  librocblas.so
  librocblas.so.4
  librocfft.so
  librocrand.so
  librocsparse.so
  librocsparse.so.1
  librocsolver.so
  libhiprtc.so
  libomp.so
  libomp.so.5
)

exclude_args=()
for library in "${excluded_libraries[@]}"; do
  exclude_args+=(--exclude "${library}")
done

mkdir -p "${output_dir}"
auditwheel repair --plat "${platform}" --strip \
  "${exclude_args[@]}" \
  -w "${output_dir}" "${input_dir}"/*.whl
