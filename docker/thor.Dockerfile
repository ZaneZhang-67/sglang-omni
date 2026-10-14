# NVIDIA PyTorch 26.03 for Jetson ARM64 (CUDA 13.2).
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch@sha256:9b1b9d6477afce05a5e941a3711093366b43de1b4eb7c1963251ade6a07c34fe
FROM ${PYTORCH_IMAGE}

ARG APT_PROXY=DIRECT
ARG NVIDIA_PYPI_INDEX=https://pypi.nvidia.com
ARG OPENFST_VERSION=1.8.3
ARG OPENFST_SHA256=077714159d5cf3e38a80b6c6656d3ccc2c8b8b6c50bb41bb65c5fec10796bf53

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get \
        -o Acquire::http::Proxy="${APT_PROXY}" \
        -o Acquire::https::Proxy="${APT_PROXY}" \
        update \
    && apt-get \
        -o Acquire::http::Proxy="${APT_PROXY}" \
        -o Acquire::https::Proxy="${APT_PROXY}" \
        install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        libsndfile1 \
        sox \
    && rm -rf /var/lib/apt/lists/*

# Pynini 2.1.6 requires OpenFst 1.8.3 with the GRM extensions.
RUN curl --fail --location \
        "https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-${OPENFST_VERSION}.tar.gz" \
        --output "/tmp/openfst-${OPENFST_VERSION}.tar.gz" \
    && echo "${OPENFST_SHA256}  /tmp/openfst-${OPENFST_VERSION}.tar.gz" \
        | sha256sum --check - \
    && tar -xzf "/tmp/openfst-${OPENFST_VERSION}.tar.gz" -C /tmp \
    && cd "/tmp/openfst-${OPENFST_VERSION}" \
    && ./configure --enable-grm \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && rm -rf \
        "/tmp/openfst-${OPENFST_VERSION}" \
        "/tmp/openfst-${OPENFST_VERSION}.tar.gz"

ENV VIRTUAL_ENV=/opt/sglang-omni
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
ENV UV_LINK_MODE=copy

RUN uv venv "${VIRTUAL_ENV}" --python python3

RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" --prerelease=allow \
        torch==2.13.0 \
        torchvision==0.28.0 \
        torchaudio==2.11.0 \
        torchcodec==0.15.0 \
    && uv pip install --python "${VIRTUAL_ENV}/bin/python" --prerelease=allow \
        --index "${NVIDIA_PYPI_INDEX}" \
        cuda-tile==1.6.0rc5 \
    && uv cache clean

COPY . /workspace/sglang-omni
WORKDIR /workspace/sglang-omni

ARG GRPCIO_VERSION=1.81.1

# gRPC 1.81.1 satisfies SMG while preserving CUTLASS DSL's Protobuf 6 stack.
RUN CPPFLAGS=-I/usr/local/include \
    LDFLAGS="-L/usr/local/lib -Wl,-rpath,/usr/local/lib" \
    uv pip install --python "${VIRTUAL_ENV}/bin/python" --prerelease=allow \
        -e . \
        "grpcio==${GRPCIO_VERSION}" \
        "grpcio-health-checking==${GRPCIO_VERSION}" \
        "grpcio-reflection==${GRPCIO_VERSION}" \
    && uv cache clean

CMD ["/bin/bash"]
