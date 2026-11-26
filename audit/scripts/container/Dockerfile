# syntax=docker/dockerfile:1
# p104-llama: llama.cpp (+p104 patches) for the P104-100 (sm_61). Build context is prepared by build.sh:
#   src/    = git archive of the requested commit (no .git)       patch.diff = optional extra patch (may be empty)
# CUDA 12.9.1 (Jul 2025) still supports sm_61 (CUDA 13 drops it). Host uses CUDA 12.0 + gcc-12; we keep gcc-12.
ARG CUDA=12.9.1
FROM nvidia/cuda:${CUDA}-devel-ubuntu24.04 AS build
ARG GIT_COMMIT=unknown
ARG JOBS=2
# CPU ISA: host build used GGML_NATIVE=ON on a Xeon W-2123 (Skylake-X). We pin the equivalent explicitly so the image
# is reproducible (NATIVE=OFF) yet keeps the same CPU kernels. Override with --build-arg CPU_OPTS=... if needed.
ARG CPU_OPTS="-DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_AVX512=ON"
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build gcc-12 g++-12 git patch libssl-dev libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY src/ /src/
COPY patch.diff /tmp/patch.diff
RUN if [ -s /tmp/patch.diff ]; then patch -p1 --no-backup-if-mismatch < /tmp/patch.diff; fi
# -j2 and no GPU access: only nvcc/gcc run. Parallelism is capped (7.5 GiB host RAM; nvcc template TUs are large).
# CHANGES.md #1: devel image has only the libcuda stub (no libcuda.so.1): executables failed to link (undefined cu* symbols).
RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
 && cmake -S . -B build -G Ninja \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_CUDA_HOST_COMPILER=gcc-12 \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61-real -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=OFF \
      -DGGML_NATIVE=OFF ${CPU_OPTS} -DBUILD_SHARED_LIBS=ON -DGGML_CCACHE=OFF \
      -DLLAMA_BUILD_COMMIT=${GIT_COMMIT} \
    && cmake --build build -j${JOBS} --target llama-server llama-bench llama-perplexity test-backend-ops \
    && mkdir -p /out/bin /out/lib \
    && cp build/bin/llama-server build/bin/llama-bench build/bin/llama-perplexity build/bin/test-backend-ops /out/bin/ \
    && cp -a build/bin/*.so* /out/lib/
RUN ls /out/bin | grep -q llama-server && sha256sum /out/bin/llama-server /out/lib/libggml-cuda.so* > /out/HASHES.txt

FROM nvidia/cuda:${CUDA}-runtime-ubuntu24.04
ARG GIT_COMMIT=unknown
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libssl3t64 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && (id ubuntu >/dev/null 2>&1 && userdel -r ubuntu || true) \
    && useradd -u 1000 -m -s /usr/sbin/nologin llama
COPY --from=build /out/bin/ /app/
COPY --from=build /out/lib/ /app/
COPY --from=build /out/HASHES.txt /app/HASHES.txt
ENV LD_LIBRARY_PATH=/app PATH=/app:$PATH NVIDIA_DRIVER_CAPABILITIES=compute,utility CUDA_DEVICE_ORDER=PCI_BUS_ID
LABEL lab.git_commit=${GIT_COMMIT} lab.arch=sm_61
USER 1000:1000
WORKDIR /app
# No models inside: bind-mount ~/models read-only at /models. No ENTRYPOINT: callers give the binary (llama-bench ...).
CMD ["llama-server", "--help"]
