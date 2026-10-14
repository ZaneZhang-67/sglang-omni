# Installation — Jetson AGX Thor

This experimental path builds SGLang-Omni on the
[NVIDIA Jetson AGX Thor Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/).
It currently validates one model and workflow: Qwen3-ASR 1.7B transcription on
one Thor GPU. It does not claim support for every SGLang-Omni model or for
Jetson Orin.

## Validated environment

- Jetson AGX Thor Developer Kit (ARM64, SM110)
- Jetson Linux R39.2 and Ubuntu 24.04
- NVIDIA driver 595.78 with CUDA driver API 13.2
- Docker with NVIDIA Container Runtime
- NVIDIA PyTorch 26.03 base image

The image installs PyTorch 2.13 and the SGLang-Omni dependency set in an
isolated virtual environment. Pynini is built against OpenFst 1.8.3 because
Ubuntu 24.04 provides an older, incompatible OpenFst release.

## Build the image

Run the build from the repository root:

```bash
docker build --network host \
  -f docker/thor.Dockerfile \
  -t sglang-omni-thor:qwen3-asr-prototype \
  .
```

If the Thor host reaches external package repositories through a local SOCKS
proxy, pass the same proxy to the package managers without disabling TLS:

```bash
docker build --network host \
  --build-arg ALL_PROXY=socks5h://127.0.0.1:1080 \
  --build-arg APT_PROXY=socks5h://127.0.0.1:1080 \
  -f docker/thor.Dockerfile \
  -t sglang-omni-thor:qwen3-asr-prototype \
  .
```

## Download Qwen3-ASR

Keep model weights outside the image so the same image can be rebuilt without
downloading the checkpoint again:

```bash
docker volume create sglang-omni-thor-models

MODEL_REVISION=7278e1e70fe206f11671096ffdd38061171dd6e5
docker run --rm \
  --network host \
  -v sglang-omni-thor-models:/models \
  sglang-omni-thor:qwen3-asr-prototype \
  hf download Qwen/Qwen3-ASR-1.7B \
    --revision "${MODEL_REVISION}" \
    --local-dir /models/Qwen3-ASR-1.7B
```

`MODEL_REVISION` identifies the model snapshot validated by the Qwen3-ASR
cookbook. It is separate from the SGLang-Omni source revision.

## Start the server

The settings below limit the prototype to one running request and about 20% of
Thor's unified memory. On the validated 122.86 GiB device, startup used about
24 GiB after CUDA graph capture.

```bash
docker run -d \
  --name sglang-omni-thor-qwen3-asr \
  --runtime nvidia \
  --network host \
  --ipc host \
  -v sglang-omni-thor-models:/models:ro \
  sglang-omni-thor:qwen3-asr-prototype \
  sgl-omni serve \
    --model-path /models/Qwen3-ASR-1.7B \
    --model-name Qwen/Qwen3-ASR-1.7B \
    --asr.engine.max_running_requests 1 \
    --asr.engine.mem_fraction_static 0.2 \
    --host 0.0.0.0 \
    --port 8000
```

Initial startup captures prefill, decode, and audio-encoder CUDA graphs. Follow
the logs until Uvicorn reports that application startup is complete:

```bash
docker logs -f sglang-omni-thor-qwen3-asr
```

## Verify transcription

Check server health:

```bash
curl --fail http://127.0.0.1:8000/health
```

From the repository root, transcribe the checked-in WAV sample:

```bash
curl --fail --max-time 300 \
  http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@tests/data/query_to_cars.wav \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F response_format=verbose_json
```

The validated response contains:

```json
{
  "text": "How many cars are there in the picture?",
  "language": "English",
  "duration": 4.62
}
```

## Known limitations

- Only Qwen3-ASR 1.7B has been validated on Thor.
- Jetson Orin is not covered by this image or test result.
- Jetson NVML does not report process memory through the same interface as a
  discrete GPU. SGLang falls back to PyTorch memory reporting.
- `uv pip check` reports the NVIDIA cuSPARSELt wheel as a different platform
  because it is tagged `manylinux2014_sbsa`. Its shared library loads on the
  validated Thor system, and the Qwen3-ASR path runs successfully.
- The CUDA 13 NIXL wheel exports `nixl_cu13`, while the current optional relay
  probe imports `nixl`. Single-stage Qwen3-ASR does not use the NIXL relay.
