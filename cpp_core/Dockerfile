# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — build proxy_server + FetchContent deps (httplib, simdjson, nlohmann)
# ---------------------------------------------------------------------------
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV ONNXRUNTIME_VERSION=1.17.1
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime prebuilt Linux x64 CPU binaries
RUN curl -fsSL \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" \
    -o /tmp/onnxruntime.tgz \
    && mkdir -p "${ONNXRUNTIME_ROOT}" \
    && tar -xzf /tmp/onnxruntime.tgz -C "${ONNXRUNTIME_ROOT}" --strip-components=1 \
    && rm /tmp/onnxruntime.tgz

WORKDIR /build

COPY CMakeLists.txt .
COPY src/ src/

RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT}" \
    && cmake --build build --config Release --target proxy_server -j"$(nproc)"

# ---------------------------------------------------------------------------
# Stage 2 — minimal runtime image
# ---------------------------------------------------------------------------
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime shared library
COPY --from=builder /opt/onnxruntime/lib/libonnxruntime.so* /usr/lib/
RUN ldconfig

# Binary lives at /app/cpp_core so ../models resolves to the mounted /app/models volume
WORKDIR /app/cpp_core
COPY --from=builder /build/build/proxy_server ./proxy_server

EXPOSE 8000

# Models are bind-mounted at /app/models (see docker-compose.yml)
ENTRYPOINT ["./proxy_server", "../models/all-MiniLM-L6-v2-int8.onnx"]
