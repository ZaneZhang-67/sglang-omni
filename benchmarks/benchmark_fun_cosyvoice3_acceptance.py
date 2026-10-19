# SPDX-License-Identifier: Apache-2.0
"""Run Fun-CosyVoice3 FA backend performance and accuracy acceptance."""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import requests
import soundfile as sf
import torch
from jiwer import process_characters, process_words

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    run_tts_seedtts_benchmark,
)
from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity
from benchmarks.metrics.speaker_similarity_assets import (
    ensure_speaker_similarity_assets,
)
from benchmarks.tasks.asr import normalize_text
from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    PackedDiT,
    PackedRowAttention,
    pack_rows,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cache import (
    PrefixRowAttention,
    PrefixStepLayout,
)
from sglang_omni.models.fun_cosyvoice3.stages import (
    CosyVoice3Vocoder,
    FlowBatchInput,
    HiftStepRow,
    build_prefix_pool,
    create_vocoder_executor,
    load_cosyvoice3_flow_hift,
    pack_flow_inputs,
    patch_chunk_mask,
    prepare_flow_conditioning,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (
    PRE_LOOKAHEAD_LEN,
    TOKEN_HOP_LEN,
    TOKEN_MAX_HOP_LEN,
    next_stream_hop_len,
    pad_flow_prompt_to_hop,
)
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import causal_hop_frames

AttentionBackend = Literal["sglang_fa3", "flashinfer_fa2"]
BACKENDS: tuple[AttentionBackend, ...] = ("sglang_fa3", "flashinfer_fa2")
ROUND_ORDER: tuple[tuple[str, AttentionBackend], ...] = (
    ("a1", "sglang_fa3"),
    ("b1", "flashinfer_fa2"),
    ("b2", "flashinfer_fa2"),
    ("a2", "sglang_fa3"),
)
DEFAULT_SEEDTTS_DATASET = "zhaochenyang20/seed-tts-eval-arrow"
DEFAULT_PREFIX_CACHE_GB = 24.0
SAMPLE_RATE = 24_000
FLOW_NOISE_SEED = 20261019
PROFILE_TENSOR_SEED = 20261020
SIMILARITY_BATCH_SIZE = 8
WAVEFORM_PATHS = {
    "whole_history": "whole_history_waveform",
    "streaming_prefix": "streaming_prefix_waveform",
}

CAPTURE_HOOK_SOURCE = '''"""Capture completed Fun-CosyVoice3 AR outputs."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from sglang_omni.models.fun_cosyvoice3 import request_builders
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State

capture_dir = Path(os.environ["COSYVOICE_FROZEN_AR_CAPTURE_DIR"])
original_apply_result = request_builders.apply_sglang_cosyvoice3_result
capture_index = 0


def cpu_clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def capture_apply_result(payload, request_data):
    global capture_index
    completed_payload = original_apply_result(payload, request_data)
    state = FunCosyVoice3State.from_dict(completed_payload.data)
    if state.seed is None:
        raise RuntimeError("Frozen AR capture requires an explicit request seed")
    if state.audio_codes is None:
        raise RuntimeError("Frozen AR capture requires audio_codes")
    if not request_data.output_codes:
        raise RuntimeError("Frozen AR capture received no raw speech tokens")
    capture_index += 1
    raw_audio_codes = torch.stack(request_data.output_codes, dim=0).to(torch.long)
    artifact = {
        "request_id": completed_payload.request_id,
        "text": state.text,
        "seed": state.seed,
        "raw_audio_codes": cpu_clone(raw_audio_codes),
        "audio_codes": torch.as_tensor(state.audio_codes, dtype=torch.long).clone(),
        "flow_prompt_speech_token": cpu_clone(request_data.flow_prompt_speech_token),
        "flow_prompt_speech_feat": cpu_clone(request_data.flow_prompt_speech_feat),
        "flow_embedding": cpu_clone(request_data.flow_embedding),
    }
    capture_dir.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, capture_dir / f"case-{capture_index:02d}.pt")
    return completed_payload


request_builders.apply_sglang_cosyvoice3_result = capture_apply_result
'''


class ReplayArguments(argparse.Namespace):
    model_path: Path
    frozen_ar_dir: Path
    output_dir: Path
    backends: list[AttentionBackend]
    device: str
    prefix_cache_gb: float


class FreezeArguments(argparse.Namespace):
    model_path: Path
    meta: str
    split: str
    output_dir: Path
    frozen_ar_dir: Path
    max_samples: int
    seed: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    host: str
    port: int
    startup_timeout: int
    backend: AttentionBackend
    packed_flow_cuda_graphs: bool


class AbbaArguments(argparse.Namespace):
    model_path: Path
    meta: str
    split: str
    output_dir: Path
    concurrencies: tuple[int, ...]
    max_samples: int | None
    warmup: int
    seed: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    host: str
    port: int
    startup_timeout: int
    packed_flow_cuda_graphs: bool


class QualityArguments(argparse.Namespace):
    replay_dir: Path
    output_dir: Path
    speaker_cache_dir: Path
    asr_base_url: str
    asr_model: str
    lang: str
    device: str
    backends: list[AttentionBackend]
    paths: list[str]
    phase: str


class ProfileArguments(argparse.Namespace):
    model_path: Path
    frozen_ar: Path
    output_dir: Path
    warmup: int
    iterations: int
    attention_repeats: int
    prefix_cache_gb: float
    device: str
    gpu_id: int
    packed_flow_cuda_graphs: bool


class ProfileCall(Protocol):
    def __call__(self) -> None: ...


class AttentionFactory(Protocol):
    def __call__(self) -> PackedRowAttention | PrefixRowAttention: ...


@dataclass(frozen=True, kw_only=True)
class FrozenArCase:
    sample_id: str
    request_id: str
    text: str
    seed: int
    reference_audio: Path
    raw_audio_codes: torch.Tensor
    audio_codes: torch.Tensor
    prompt_speech_token: torch.Tensor
    prompt_speech_features: torch.Tensor
    speaker_embedding: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class StreamingReplay:
    mel: torch.Tensor
    waveform: torch.Tensor
    hop_mels: list[torch.Tensor]
    hop_token_ends: list[int]


@dataclass(frozen=True, kw_only=True)
class RoundRecord:
    concurrency: int
    round_name: str
    backend: AttentionBackend
    output_dir: Path
    speed_results: dict[str, object]


def require_tensor(payload: dict[str, object], key: str) -> torch.Tensor:
    value = payload[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Frozen AR field {key!r} must be a tensor")
    else:
        return value


def load_frozen_case(path: Path) -> FrozenArCase:
    payload: object = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise TypeError(f"Frozen AR file must contain a string-keyed dict: {path}")
    else:
        typed_payload: dict[str, object] = payload
    request_id = typed_payload["request_id"]
    sample_id = typed_payload["sample_id"]
    text = typed_payload["text"]
    seed = typed_payload["seed"]
    reference_audio = typed_payload["reference_audio"]
    if (
        not isinstance(request_id, str)
        or not isinstance(sample_id, str)
        or not isinstance(text, str)
        or not isinstance(reference_audio, str)
    ):
        raise TypeError(
            f"Frozen AR identifiers, text and reference_audio must be strings: {path}"
        )
    elif not isinstance(seed, int):
        raise TypeError(f"Frozen AR seed must be an int: {path}")
    else:
        pass
    reference_path = Path(reference_audio)
    if not reference_path.is_absolute():
        reference_path = path.parent / reference_path
    else:
        pass
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    else:
        pass
    return FrozenArCase(
        sample_id=sample_id,
        request_id=request_id,
        text=text,
        seed=seed,
        reference_audio=reference_path.resolve(),
        raw_audio_codes=require_tensor(typed_payload, "raw_audio_codes"),
        audio_codes=require_tensor(typed_payload, "audio_codes"),
        prompt_speech_token=require_tensor(typed_payload, "flow_prompt_speech_token"),
        prompt_speech_features=require_tensor(typed_payload, "flow_prompt_speech_feat"),
        speaker_embedding=require_tensor(typed_payload, "flow_embedding"),
    )


def set_case_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_flow_input(frozen: FrozenArCase, token_mel_ratio: int) -> FlowBatchInput:
    prompt_token, prompt_features = pad_flow_prompt_to_hop(
        frozen.prompt_speech_token.to(torch.int32),
        frozen.prompt_speech_features,
        token_mel_ratio=token_mel_ratio,
    )
    return FlowBatchInput(
        token=frozen.audio_codes.reshape(1, -1).to(torch.int32),
        prompt_token=prompt_token,
        prompt_feat=prompt_features,
        embedding=frozen.speaker_embedding,
    )


def replay_whole_history(
    vocoder: CosyVoice3Vocoder, flow_input: FlowBatchInput, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(vocoder.flow.parameters()).device
    set_case_seed(seed)
    with vocoder.stream_context:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            mel = vocoder.leftover_batch([flow_input])[0]
        waveform = vocoder.mel2wav_batch([mel])[0]
    torch.cuda.synchronize(device)
    return mel.detach().cpu(), waveform.detach().cpu()


def replay_streaming_prefix(
    vocoder: CosyVoice3Vocoder, flow_input: FlowBatchInput, seed: int
) -> StreamingReplay:
    device = next(vocoder.flow.parameters()).device
    token_count = int(flow_input.token.shape[1])
    token_offset = 0
    hop_tokens = TOKEN_HOP_LEN
    mel_history: torch.Tensor | None = None
    emitted_samples = 0
    waveform_parts: list[torch.Tensor] = []
    hop_mels: list[torch.Tensor] = []
    hop_token_ends: list[int] = []
    cache = None
    set_case_seed(seed)
    with vocoder.stream_context:
        while token_count >= token_offset + hop_tokens + PRE_LOOKAHEAD_LEN:
            token_end = token_offset + hop_tokens + PRE_LOOKAHEAD_LEN
            step_input = FlowBatchInput(
                token=flow_input.token[:, :token_end],
                prompt_token=flow_input.prompt_token,
                prompt_feat=flow_input.prompt_feat,
                embedding=flow_input.embedding,
            )
            total_frames = causal_hop_frames(
                step_input, token_mel_ratio=vocoder.flow.token_mel_ratio
            )
            if cache is None:
                cache = vocoder.prefix_cache_rows(total_frames)
                if cache is None:
                    raise RuntimeError("Prefix cache could not allocate the first hop")
                else:
                    pass
            elif not vocoder.grow_prefix_cache(cache, total_frames):
                raise RuntimeError("Prefix cache could not grow for a replay hop")
            else:
                pass
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                step_mel = vocoder.hop_batch_prefix([step_input], [cache])[0]
            new_mel = step_mel[
                :, :, token_offset * vocoder.flow.token_mel_ratio :
            ].detach()
            mel_history = (
                new_mel
                if mel_history is None
                else torch.cat([mel_history, new_mel], dim=2)
            )
            waveform_delta, emitted_samples = vocoder.hift_step(
                [
                    HiftStepRow(
                        history=mel_history,
                        emitted_samples=emitted_samples,
                        is_final=False,
                    )
                ]
            )[0]
            waveform_parts.append(waveform_delta)
            hop_mels.append(new_mel.cpu())
            hop_token_ends.append(token_end)
            token_offset += hop_tokens
            hop_tokens = next_stream_hop_len(hop_tokens, max_hop_len=TOKEN_MAX_HOP_LEN)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            final_mel = vocoder.leftover_batch([flow_input])[0]
        new_mel = final_mel[
            :, :, token_offset * vocoder.flow.token_mel_ratio :
        ].detach()
        mel_history = (
            new_mel if mel_history is None else torch.cat([mel_history, new_mel], dim=2)
        )
        waveform_delta, emitted_samples = vocoder.hift_step(
            [
                HiftStepRow(
                    history=mel_history,
                    emitted_samples=emitted_samples,
                    is_final=True,
                )
            ]
        )[0]
        waveform_parts.append(waveform_delta)
        hop_mels.append(new_mel.cpu())
        hop_token_ends.append(token_count)
    vocoder.release_prefix_cache(cache)
    torch.cuda.synchronize(device)
    waveform = torch.cat(waveform_parts, dim=1)
    return StreamingReplay(
        mel=mel_history.detach().cpu(),
        waveform=waveform.detach().cpu(),
        hop_mels=hop_mels,
        hop_token_ends=hop_token_ends,
    )


def run_backend(
    backend: AttentionBackend,
    frozen_cases: list[FrozenArCase],
    arguments: ReplayArguments,
) -> None:
    set_case_seed(FLOW_NOISE_SEED)
    flow, hift = load_cosyvoice3_flow_hift(
        str(arguments.model_path),
        device=arguments.device,
        autocast_dtype=torch.bfloat16,
        flow_attention_backend=backend,
    )
    device = torch.device(arguments.device)
    flow_noise = flow.decoder.rand_noise
    noise_generator = torch.Generator(device="cpu")
    noise_generator.manual_seed(FLOW_NOISE_SEED)
    fixed_noise = torch.randn(
        flow_noise.shape,
        dtype=flow_noise.dtype,
        device="cpu",
        generator=noise_generator,
    )
    flow_noise.copy_(fixed_noise.to(device=flow_noise.device))
    flow.prefix_pool = build_prefix_pool(
        flow, device, torch.bfloat16, arguments.prefix_cache_gb
    )
    vocoder = CosyVoice3Vocoder(
        flow,
        hift,
        autocast_dtype=torch.bfloat16,
        hift_dtype="float32",
    )
    first_flow_input = make_flow_input(frozen_cases[0], vocoder.flow.token_mel_ratio)
    assert flow.packed_estimator is not None
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        conditioning = prepare_flow_conditioning(
            flow,
            pack_flow_inputs(flow.flow, [first_flow_input]),
            finalize=False,
        )
        flow.packed_estimator.build_modulations(
            conditioning.time_span, conditioning.speaker_embedding.dtype
        )
    backend_dir = arguments.output_dir / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    manifest_cases: list[dict[str, object]] = []
    for index, frozen in enumerate(frozen_cases, start=1):
        flow_input = make_flow_input(frozen, vocoder.flow.token_mel_ratio)
        whole_mel, whole_waveform = replay_whole_history(
            vocoder, flow_input, frozen.seed
        )
        streaming = replay_streaming_prefix(vocoder, flow_input, frozen.seed)
        case_name = f"case-{index:02d}"
        torch.save(
            {
                "audio_codes": frozen.audio_codes,
                "raw_audio_codes": frozen.raw_audio_codes,
                "whole_history_mel": whole_mel,
                "whole_history_waveform": whole_waveform,
                "streaming_prefix_mel": streaming.mel,
                "streaming_prefix_waveform": streaming.waveform,
                "streaming_prefix_hop_mels": streaming.hop_mels,
                "streaming_prefix_hop_token_ends": streaming.hop_token_ends,
            },
            backend_dir / f"{case_name}.pt",
        )
        manifest_cases.append(
            {
                "case": case_name,
                "sample_id": frozen.sample_id,
                "request_id": frozen.request_id,
                "text": frozen.text,
                "seed": frozen.seed,
                "reference_audio": str(frozen.reference_audio),
                "raw_speech_tokens": int(frozen.raw_audio_codes.numel()),
                "speech_tokens": int(frozen.audio_codes.numel()),
                "whole_history_mel_frames": int(whole_mel.shape[2]),
                "whole_history_audio_samples": int(whole_waveform.shape[1]),
                "whole_history_audio_seconds": (
                    int(whole_waveform.shape[1]) / SAMPLE_RATE
                ),
                "streaming_prefix_mel_frames": int(streaming.mel.shape[2]),
                "streaming_prefix_audio_samples": int(streaming.waveform.shape[1]),
                "streaming_prefix_audio_seconds": (
                    int(streaming.waveform.shape[1]) / SAMPLE_RATE
                ),
                "streaming_prefix_hop_token_ends": streaming.hop_token_ends,
            }
        )
        print(
            f"backend={backend} case={case_name} tokens={frozen.audio_codes.numel()} "
            f"whole_seconds={whole_waveform.shape[1] / SAMPLE_RATE:.3f} "
            f"streaming_seconds={streaming.waveform.shape[1] / SAMPLE_RATE:.3f}"
        )
    with (backend_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "backend": backend,
                "device": arguments.device,
                "dtype": "bfloat16",
                "dit_compiled": False,
                "flow_noise_seed": FLOW_NOISE_SEED,
                "prefix_cache_gb": arguments.prefix_cache_gb,
                "cases": manifest_cases,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
    del vocoder, hift, flow
    gc.collect()
    torch.cuda.empty_cache()


def tensor_difference(base: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    shape_equal = tuple(base.shape) == tuple(candidate.shape)
    if not shape_equal:
        return {
            "shape_equal": False,
            "base_shape": list(base.shape),
            "candidate_shape": list(candidate.shape),
            "bit_exact": False,
        }
    difference = candidate.float() - base.float()
    base_norm = float(torch.linalg.vector_norm(base.float()))
    difference_norm = float(torch.linalg.vector_norm(difference))
    return {
        "shape_equal": True,
        "base_shape": list(base.shape),
        "candidate_shape": list(candidate.shape),
        "bit_exact": bool(torch.equal(base, candidate)),
        "mean_absolute_error": float(difference.abs().mean()),
        "max_absolute_error": float(difference.abs().max()),
        "rmse": float(torch.sqrt(torch.mean(difference.square()))),
        "l2": difference_norm,
        "relative_l2": difference_norm / base_norm if base_norm > 0 else math.inf,
    }


def load_replay_artifact(path: Path) -> dict[str, object]:
    payload: object = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise TypeError(f"Replay artifact must contain a string-keyed dict: {path}")
    else:
        return payload


def compare_backends(
    frozen_cases: list[FrozenArCase], arguments: ReplayArguments
) -> dict[str, object]:
    base_backend, candidate_backend = arguments.backends
    comparisons: list[dict[str, object]] = []
    for index, frozen in enumerate(frozen_cases, start=1):
        case_name = f"case-{index:02d}"
        base = load_replay_artifact(
            arguments.output_dir / base_backend / f"{case_name}.pt"
        )
        candidate = load_replay_artifact(
            arguments.output_dir / candidate_backend / f"{case_name}.pt"
        )
        base_hops = base["streaming_prefix_hop_mels"]
        candidate_hops = candidate["streaming_prefix_hop_mels"]
        if not isinstance(base_hops, list) or not isinstance(candidate_hops, list):
            raise TypeError("Replay hop mels must be lists")
        else:
            pass
        hop_differences: list[dict[str, object]] = []
        for hop_index, (base_hop, candidate_hop) in enumerate(
            zip(base_hops, candidate_hops, strict=True), start=1
        ):
            if not isinstance(base_hop, torch.Tensor) or not isinstance(
                candidate_hop, torch.Tensor
            ):
                raise TypeError("Replay hop mel entries must be tensors")
            else:
                pass
            hop_differences.append(
                {
                    "hop": hop_index,
                    **tensor_difference(base_hop, candidate_hop),
                }
            )
        tensor_keys = (
            "whole_history_mel",
            "whole_history_waveform",
            "streaming_prefix_mel",
            "streaming_prefix_waveform",
        )
        path_differences: dict[str, object] = {}
        for key in tensor_keys:
            base_tensor = base[key]
            candidate_tensor = candidate[key]
            if not isinstance(base_tensor, torch.Tensor) or not isinstance(
                candidate_tensor, torch.Tensor
            ):
                raise TypeError(f"Replay field {key!r} must be a tensor")
            else:
                path_differences[key] = tensor_difference(base_tensor, candidate_tensor)
        base_codes = base["audio_codes"]
        candidate_codes = candidate["audio_codes"]
        if not isinstance(base_codes, torch.Tensor) or not isinstance(
            candidate_codes, torch.Tensor
        ):
            raise TypeError("Replay audio_codes must be tensors")
        else:
            pass
        comparisons.append(
            {
                "case": case_name,
                "request_id": frozen.request_id,
                "seed": frozen.seed,
                "speech_tokens": int(frozen.audio_codes.numel()),
                "ar_tokens_bit_exact": bool(torch.equal(base_codes, candidate_codes)),
                "paths": path_differences,
                "streaming_prefix_hops": hop_differences,
            }
        )
    report: dict[str, object] = {
        "base": base_backend,
        "candidate": candidate_backend,
        "cases": comparisons,
    }
    with (arguments.output_dir / "comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def parse_concurrencies(value: str) -> tuple[int, ...]:
    concurrencies = tuple(int(part.strip()) for part in value.split(","))
    if not concurrencies or min(concurrencies) <= 0:
        raise argparse.ArgumentTypeError(
            "concurrencies must be positive comma-separated integers"
        )
    else:
        return concurrencies


def server_command(
    model_path: Path,
    host: str,
    port: int,
    backend: AttentionBackend,
    *,
    packed_flow_cuda_graphs: bool,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--vocoder.factory.flow_attention_backend",
        backend,
        "--vocoder.factory.enable_flow_cuda_graph",
        "false",
        "--vocoder.factory.enable_flow_prefix_cuda_graph",
        str(packed_flow_cuda_graphs).lower(),
        "--vocoder.factory.enable_flow_whole_history_cuda_graph",
        str(packed_flow_cuda_graphs).lower(),
    ]


def verify_backend_log(log_path: Path, backend: AttentionBackend) -> None:
    expected = f"Fun-CosyVoice3 Flow row attention on cuda:0: {backend}"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if expected not in log_text:
        raise RuntimeError(
            f"Server log does not confirm requested backend {backend!r}: {log_path}"
        )
    else:
        pass


def seedtts_config(
    *,
    model_path: Path,
    meta: str,
    output_dir: Path,
    host: str,
    port: int,
    max_samples: int | None,
    warmup: int,
    concurrency: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    lang: str,
) -> TtsSeedttsBenchmarkConfig:
    return TtsSeedttsBenchmarkConfig(
        model=str(model_path),
        meta=meta,
        base_url=f"http://{host}:{port}",
        output_dir=str(output_dir),
        max_samples=max_samples,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        seed=seed,
        warmup=warmup,
        concurrency=concurrency,
        stream=True,
        collect_stream_outcomes=True,
        disable_tqdm=True,
        lang=lang,
    )


def run_freeze(arguments: FreezeArguments) -> None:
    if arguments.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    elif list(arguments.frozen_ar_dir.glob("*.pt")):
        raise RuntimeError(
            f"Frozen AR directory already contains .pt files: {arguments.frozen_ar_dir}"
        )
    else:
        pass
    samples = load_seedtts_samples(
        arguments.meta,
        arguments.max_samples,
        split=arguments.split,
    )
    if len(samples) != arguments.max_samples:
        raise RuntimeError(
            f"Requested {arguments.max_samples} SeedTTS samples, got {len(samples)}"
        )
    else:
        pass
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    arguments.frozen_ar_dir.mkdir(parents=True, exist_ok=True)
    log_path = arguments.output_dir / "freeze-server.log"
    benchmark_dir = arguments.output_dir / "freeze-seedtts"
    config = seedtts_config(
        model_path=arguments.model_path,
        meta=arguments.meta,
        output_dir=benchmark_dir,
        host=arguments.host,
        port=arguments.port,
        max_samples=arguments.max_samples,
        warmup=0,
        concurrency=1,
        seed=arguments.seed,
        temperature=arguments.temperature,
        top_p=arguments.top_p,
        top_k=arguments.top_k,
        repetition_penalty=arguments.repetition_penalty,
        lang=arguments.split,
    )
    process: subprocess.Popen[bytes] | None = None
    with tempfile.TemporaryDirectory(prefix="cosyvoice-capture-hook-") as hook_dir:
        hook_path = Path(hook_dir)
        (hook_path / "sitecustomize.py").write_text(
            CAPTURE_HOOK_SOURCE,
            encoding="utf-8",
        )
        server_environment = os.environ.copy()
        existing_python_path = server_environment.get("PYTHONPATH")
        server_environment["PYTHONPATH"] = (
            f"{hook_path}{os.pathsep}{existing_python_path}"
            if existing_python_path
            else str(hook_path)
        )
        server_environment["COSYVOICE_FROZEN_AR_CAPTURE_DIR"] = str(
            arguments.frozen_ar_dir.resolve()
        )
        try:
            process = start_server_from_cmd(
                server_command(
                    arguments.model_path,
                    arguments.host,
                    arguments.port,
                    arguments.backend,
                    packed_flow_cuda_graphs=arguments.packed_flow_cuda_graphs,
                ),
                log_path,
                arguments.port,
                timeout=arguments.startup_timeout,
                env=server_environment,
            )
            speed_results = asyncio.run(
                run_tts_seedtts_benchmark(
                    config,
                    samples=samples,
                    save_audio=False,
                )
            )
        finally:
            if process is not None:
                stop_server(process)
            else:
                pass
    verify_backend_log(log_path, arguments.backend)
    frozen_paths = sorted(arguments.frozen_ar_dir.glob("case-*.pt"))
    if len(frozen_paths) != len(samples):
        raise RuntimeError(
            f"Captured {len(frozen_paths)} frozen AR cases for {len(samples)} samples"
        )
    else:
        pass
    references_dir = arguments.frozen_ar_dir / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    frozen_manifest: list[dict[str, object]] = []
    for index, (path, sample) in enumerate(zip(frozen_paths, samples, strict=True), 1):
        artifact = load_replay_artifact(path)
        if artifact["text"] != sample.target_text:
            raise RuntimeError(
                f"Captured case order differs from SeedTTS at case {index}: "
                f"{artifact['text']!r} != {sample.target_text!r}"
            )
        else:
            pass
        reference_suffix = Path(sample.ref_audio).suffix or ".wav"
        reference_name = f"case-{index:02d}{reference_suffix}"
        reference_path = references_dir / reference_name
        shutil.copy2(sample.ref_audio, reference_path)
        artifact["sample_id"] = sample.sample_id
        artifact["reference_audio"] = str(Path("references") / reference_name)
        torch.save(artifact, path)
        audio_codes = artifact["audio_codes"]
        if not isinstance(audio_codes, torch.Tensor):
            raise TypeError("Captured audio_codes must be a tensor")
        else:
            pass
        frozen_manifest.append(
            {
                "case": f"case-{index:02d}",
                "sample_id": sample.sample_id,
                "target_text": sample.target_text,
                "speech_tokens": int(audio_codes.numel()),
                "reference_audio": str(Path("references") / reference_name),
            }
        )
    summary = speed_results["summary"]
    if summary["completed_requests"] != len(samples) or summary["failed_requests"]:
        raise RuntimeError(f"SeedTTS freeze generation failed: {summary}")
    else:
        pass
    report = {
        "dataset": arguments.meta,
        "split": arguments.split,
        "backend": arguments.backend,
        "packed_flow_cuda_graphs_enabled": arguments.packed_flow_cuda_graphs,
        "seed": arguments.seed,
        "sampling": {
            "temperature": arguments.temperature,
            "top_p": arguments.top_p,
            "top_k": arguments.top_k,
            "repetition_penalty": arguments.repetition_penalty,
        },
        "speed_summary": summary,
        "cases": frozen_manifest,
    }
    report_path = arguments.frozen_ar_dir / "manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"frozen_ar_manifest={report_path}")


def validate_round(speed_results: dict[str, object], expected_requests: int) -> None:
    summary = speed_results["summary"]
    per_request = speed_results["per_request"]
    if not isinstance(summary, dict) or not isinstance(per_request, list):
        raise TypeError("Speed results must contain summary and per_request records")
    elif summary["completed_requests"] != expected_requests:
        raise RuntimeError(
            f"Completed {summary['completed_requests']} of {expected_requests} requests"
        )
    elif summary["failed_requests"] != 0:
        raise RuntimeError(f"Benchmark reported {summary['failed_requests']} failures")
    else:
        pass
    for record in per_request:
        if not isinstance(record, dict):
            raise TypeError("Per-request speed records must be dictionaries")
        elif record["completion_tokens"] is None:
            raise RuntimeError(f"Request {record['id']} has no completion token count")
        elif record["audio_duration_s"] <= 0:
            raise RuntimeError(f"Request {record['id']} has no audio duration")
        else:
            pass


def run_round(
    arguments: AbbaArguments,
    base_config: TtsSeedttsBenchmarkConfig,
    samples: list[SampleInput],
    concurrency: int,
    round_name: str,
    backend: AttentionBackend,
) -> RoundRecord:
    point_name = f"c{concurrency:02d}/{round_name}-{backend.replace('_', '-')}"
    point_output_dir = arguments.output_dir / point_name
    point_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = point_output_dir / "server.log"
    process: subprocess.Popen[bytes] | None = None
    try:
        process = start_server_from_cmd(
            server_command(
                arguments.model_path,
                arguments.host,
                arguments.port,
                backend,
                packed_flow_cuda_graphs=arguments.packed_flow_cuda_graphs,
            ),
            log_path,
            arguments.port,
            timeout=arguments.startup_timeout,
        )
        config = replace(
            base_config,
            concurrency=concurrency,
            output_dir=str(point_output_dir),
        )
        speed_results = asyncio.run(run_tts_seedtts_benchmark(config, samples=samples))
        validate_round(speed_results, len(samples))
    finally:
        if process is not None:
            stop_server(process)
        else:
            pass
    verify_backend_log(log_path, backend)
    return RoundRecord(
        concurrency=concurrency,
        round_name=round_name,
        backend=backend,
        output_dir=point_output_dir,
        speed_results=speed_results,
    )


def request_signature(record: RoundRecord) -> list[tuple[str, int, float]]:
    per_request = record.speed_results["per_request"]
    if not isinstance(per_request, list):
        raise TypeError("per_request must be a list")
    else:
        pass
    signature: list[tuple[str, int, float]] = []
    for request in per_request:
        if not isinstance(request, dict):
            raise TypeError("Per-request speed records must be dictionaries")
        else:
            signature.append(
                (
                    str(request["id"]),
                    int(request["completion_tokens"]),
                    float(request["audio_duration_s"]),
                )
            )
    return sorted(signature)


def request_values(records: list[RoundRecord], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        per_request = record.speed_results["per_request"]
        if not isinstance(per_request, list):
            raise TypeError("per_request must be a list")
        else:
            pass
        for request in per_request:
            if not isinstance(request, dict):
                raise TypeError("Per-request speed records must be dictionaries")
            else:
                value = request[key]
            if value is None:
                raise RuntimeError(f"Per-request metric {key!r} is missing")
            else:
                values.append(float(value))
    return values


def aggregate_backend(records: list[RoundRecord]) -> dict[str, object]:
    latency_seconds = request_values(records, "latency_s")
    first_audio_seconds = request_values(records, "audio_ttfp_s")
    rtfs = request_values(records, "rtf")
    audio_seconds = request_values(records, "audio_duration_s")
    completion_tokens = request_values(records, "completion_tokens")
    round_throughputs: list[float] = []
    for record in records:
        summary = record.speed_results["summary"]
        if not isinstance(summary, dict):
            raise TypeError("summary must be a dictionary")
        else:
            round_throughputs.append(float(summary["throughput_qps"]))
    return {
        "rounds": [record.round_name for record in records],
        "completed_requests": len(latency_seconds),
        "latency_median_s": float(np.median(latency_seconds)),
        "latency_p95_s": float(np.percentile(latency_seconds, 95)),
        "ttfa_median_s": float(np.median(first_audio_seconds)),
        "ttfa_p95_s": float(np.percentile(first_audio_seconds, 95)),
        "rtf_median": float(np.median(rtfs)),
        "rtf_p95": float(np.percentile(rtfs, 95)),
        "throughput_qps_mean": statistics.mean(round_throughputs),
        "throughput_qps_rounds": round_throughputs,
        "completion_tokens_total": int(sum(completion_tokens)),
        "completion_tokens_per_request": [int(value) for value in completion_tokens],
        "audio_seconds_total": sum(audio_seconds),
        "audio_seconds_per_request": audio_seconds,
    }


def aggregate_concurrency(records: list[RoundRecord]) -> dict[str, object]:
    base_records = [record for record in records if record.backend == BACKENDS[0]]
    candidate_records = [record for record in records if record.backend == BACKENDS[1]]
    signatures = [request_signature(record) for record in records]
    token_signatures = [
        [(request_id, tokens) for request_id, tokens, _ in signature]
        for signature in signatures
    ]
    return {
        "concurrency": records[0].concurrency,
        "round_order": [record.round_name for record in records],
        "backend_order": [record.backend for record in records],
        "completion_tokens_identical": all(
            signature == token_signatures[0] for signature in token_signatures[1:]
        ),
        "token_and_audio_outputs_identical": all(
            signature == signatures[0] for signature in signatures[1:]
        ),
        "per_round_outputs": {
            record.round_name: [
                {
                    "id": request_id,
                    "completion_tokens": tokens,
                    "audio_duration_s": audio_seconds,
                }
                for request_id, tokens, audio_seconds in request_signature(record)
            ]
            for record in records
        },
        "base": aggregate_backend(base_records),
        "candidate": aggregate_backend(candidate_records),
    }


def run_abba(arguments: AbbaArguments) -> None:
    if arguments.warmup < 10:
        raise ValueError("--warmup must be at least 10")
    else:
        pass
    samples = load_seedtts_samples(
        arguments.meta,
        arguments.max_samples,
        split=arguments.split,
    )
    if len(samples) < max(arguments.concurrencies):
        raise RuntimeError(
            f"Workload has {len(samples)} samples but concurrency reaches "
            f"{max(arguments.concurrencies)}"
        )
    else:
        pass
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    base_config = seedtts_config(
        model_path=arguments.model_path,
        meta=arguments.meta,
        output_dir=arguments.output_dir,
        host=arguments.host,
        port=arguments.port,
        max_samples=arguments.max_samples,
        warmup=arguments.warmup,
        concurrency=1,
        seed=arguments.seed,
        temperature=arguments.temperature,
        top_p=arguments.top_p,
        top_k=arguments.top_k,
        repetition_penalty=arguments.repetition_penalty,
        lang=arguments.split,
    )
    concurrency_reports: list[dict[str, object]] = []
    for concurrency in arguments.concurrencies:
        round_records: list[RoundRecord] = []
        for round_name, backend in ROUND_ORDER:
            print(
                f"concurrency={concurrency} round={round_name} backend={backend}",
                flush=True,
            )
            round_records.append(
                run_round(
                    arguments,
                    base_config,
                    samples,
                    concurrency,
                    round_name,
                    backend,
                )
            )
        concurrency_reports.append(aggregate_concurrency(round_records))
    device_index = torch.cuda.current_device()
    report = {
        "model_path": str(arguments.model_path),
        "dataset": arguments.meta,
        "split": arguments.split,
        "gpu": {
            "name": torch.cuda.get_device_name(device_index),
            "compute_capability": list(torch.cuda.get_device_capability(device_index)),
        },
        "warmup_requests": arguments.warmup,
        "seed": arguments.seed,
        "sampling": {
            "temperature": arguments.temperature,
            "top_p": arguments.top_p,
            "top_k": arguments.top_k,
            "repetition_penalty": arguments.repetition_penalty,
        },
        "flow_cuda_graphs_enabled": False,
        "packed_flow_cuda_graphs_enabled": arguments.packed_flow_cuda_graphs,
        "backend_note": (
            "sglang_fa3 is the requested SGLang backend name; its underlying "
            "kernel depends on GPU architecture and installed runtime."
        ),
        "concurrencies": concurrency_reports,
    }
    report_path = arguments.output_dir / "abba_summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"abba_report={report_path}")


def export_waveforms(
    replay_dir: Path,
    output_dir: Path,
    backends: tuple[str, ...],
    waveform_paths: dict[str, str],
) -> list[dict[str, object]]:
    base_manifest = json.loads(
        (replay_dir / backends[0] / "manifest.json").read_text(encoding="utf-8")
    )
    cases = base_manifest["cases"]
    entries: list[dict[str, object]] = []
    for backend in backends:
        backend_dir = replay_dir / backend
        manifest = json.loads(
            (backend_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest["cases"] != cases:
            raise RuntimeError(f"Frozen replay manifests differ for {backend}")
        else:
            pass
        for case in cases:
            case_name = case["case"]
            tensors = torch.load(
                backend_dir / f"{case_name}.pt",
                map_location="cpu",
                weights_only=True,
            )
            for path_name, tensor_name in waveform_paths.items():
                waveform = tensors[tensor_name].reshape(-1).to(torch.float32).numpy()
                wav_path = (
                    output_dir / "audio" / backend / path_name / f"{case_name}.wav"
                )
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(wav_path, waveform, SAMPLE_RATE, subtype="FLOAT")
                entries.append(
                    {
                        "case": case_name,
                        "sample_id": case["sample_id"],
                        "backend": backend,
                        "path": path_name,
                        "target_text": case["text"],
                        "speech_tokens": int(case["speech_tokens"]),
                        "audio_seconds": float(waveform.shape[0] / SAMPLE_RATE),
                        "wav_path": str(wav_path.resolve()),
                        "reference_audio": case["reference_audio"],
                    }
                )
    return entries


def transcribe_quality(
    entries: list[dict[str, object]],
    asr_base_url: str,
    asr_model: str,
    lang: str,
) -> None:
    endpoint = f"{asr_base_url.rstrip('/')}/v1/audio/transcriptions"
    for entry in entries:
        wav_path = Path(str(entry["wav_path"]))
        with wav_path.open("rb") as audio_file:
            response = requests.post(
                endpoint,
                data={
                    "model": asr_model,
                    "language": lang,
                    "response_format": "json",
                },
                files={"file": (wav_path.name, audio_file, "audio/wav")},
                timeout=300,
                proxies={"http": None, "https": None},
            )
        response.raise_for_status()
        hypothesis = str(response.json()["text"])
        reference_normalized = normalize_text(str(entry["target_text"]), lang)
        hypothesis_normalized = normalize_text(hypothesis, lang)
        word_measures = process_words(reference_normalized, hypothesis_normalized)
        character_measures = process_characters(
            "".join(reference_normalized.split()),
            "".join(hypothesis_normalized.split()),
        )
        entry["asr_text"] = hypothesis
        entry["reference_normalized"] = reference_normalized
        entry["hypothesis_normalized"] = hypothesis_normalized
        entry["wer"] = float(word_measures.wer)
        entry["word_substitutions"] = int(word_measures.substitutions)
        entry["word_deletions"] = int(word_measures.deletions)
        entry["word_insertions"] = int(word_measures.insertions)
        entry["word_hits"] = int(word_measures.hits)
        entry["cer"] = float(character_measures.cer)
        entry["character_substitutions"] = int(character_measures.substitutions)
        entry["character_deletions"] = int(character_measures.deletions)
        entry["character_insertions"] = int(character_measures.insertions)
        entry["character_hits"] = int(character_measures.hits)


def score_similarity(
    entries: list[dict[str, object]],
    speaker_cache_dir: Path,
    device: str,
) -> None:
    assets = ensure_speaker_similarity_assets(cache_dir=speaker_cache_dir)
    scorer = WavLMSpeakerSimilarity(
        finetune_checkpoint=assets.finetune_checkpoint,
        wavlm_base=assets.wavlm_base,
        device=device,
    )
    for start in range(0, len(entries), SIMILARITY_BATCH_SIZE):
        batch = entries[start : start + SIMILARITY_BATCH_SIZE]
        scores = scorer.score_batch(
            [str(entry["reference_audio"]) for entry in batch],
            [str(entry["wav_path"]) for entry in batch],
        )
        for entry, score in zip(batch, scores, strict=True):
            entry["speaker_similarity"] = float(score)


def quality_group(rows: list[dict[str, object]]) -> dict[str, object]:
    word_errors = sum(
        int(row["word_substitutions"])
        + int(row["word_deletions"])
        + int(row["word_insertions"])
        for row in rows
    )
    reference_words = sum(
        int(row["word_substitutions"])
        + int(row["word_deletions"])
        + int(row["word_hits"])
        for row in rows
    )
    character_errors = sum(
        int(row["character_substitutions"])
        + int(row["character_deletions"])
        + int(row["character_insertions"])
        for row in rows
    )
    reference_characters = sum(
        int(row["character_substitutions"])
        + int(row["character_deletions"])
        + int(row["character_hits"])
        for row in rows
    )
    return {
        "samples": len(rows),
        "wer_corpus": word_errors / reference_words,
        "wer_mean": statistics.mean(float(row["wer"]) for row in rows),
        "cer_corpus": character_errors / reference_characters,
        "cer_mean": statistics.mean(float(row["cer"]) for row in rows),
        "speaker_similarity_mean": statistics.mean(
            float(row["speaker_similarity"]) for row in rows
        ),
    }


def summarize_quality(
    entries: list[dict[str, object]],
    backends: tuple[str, ...],
    waveform_paths: dict[str, str],
) -> dict[str, object]:
    groups: dict[str, dict[str, object]] = {}
    for backend in backends:
        for path_name in waveform_paths:
            rows = [
                entry
                for entry in entries
                if entry["backend"] == backend and entry["path"] == path_name
            ]
            if not rows:
                raise RuntimeError(f"No quality rows for {backend}/{path_name}")
            else:
                groups[f"{backend}/{path_name}"] = quality_group(rows)

    comparisons: dict[str, dict[str, object]] = {}
    all_passed = True
    for candidate_backend in backends[1:]:
        candidate_comparisons: dict[str, object] = {}
        for path_name in waveform_paths:
            base = groups[f"{backends[0]}/{path_name}"]
            candidate = groups[f"{candidate_backend}/{path_name}"]
            path_entries = [entry for entry in entries if entry["path"] == path_name]
            transcripts = {
                (str(entry["case"]), str(entry["backend"])): str(entry["asr_text"])
                for entry in path_entries
            }
            wer_delta = float(candidate["wer_corpus"]) - float(base["wer_corpus"])
            cer_delta = float(candidate["cer_corpus"]) - float(base["cer_corpus"])
            similarity_delta = float(candidate["speaker_similarity_mean"]) - float(
                base["speaker_similarity_mean"]
            )
            passed = wer_delta <= 0 and cer_delta <= 0 and similarity_delta >= 0
            all_passed = all_passed and passed
            candidate_comparisons[path_name] = {
                "candidate_minus_base_wer_corpus": wer_delta,
                "candidate_minus_base_cer_corpus": cer_delta,
                "candidate_minus_base_speaker_similarity": similarity_delta,
                "asr_transcripts_exact": all(
                    transcripts[(case, backends[0])]
                    == transcripts[(case, candidate_backend)]
                    for case in {str(entry["case"]) for entry in path_entries}
                ),
                "passes_no_quality_regression": passed,
            }
        comparisons[candidate_backend] = candidate_comparisons
    return {
        "groups": groups,
        "comparisons": comparisons,
        "passes_no_quality_regression": all_passed,
    }


def run_quality(arguments: QualityArguments) -> None:
    backends = tuple(arguments.backends)
    waveform_paths = {path: WAVEFORM_PATHS[path] for path in arguments.paths}
    if len(backends) != 2:
        raise ValueError("--backends requires exactly one base and one candidate")
    else:
        pass
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    entries_path = arguments.output_dir / "quality_entries.json"
    if arguments.phase == "similarity":
        entries = json.loads(entries_path.read_text(encoding="utf-8"))
    else:
        entries = export_waveforms(
            arguments.replay_dir,
            arguments.output_dir,
            backends,
            waveform_paths,
        )
        transcribe_quality(
            entries,
            arguments.asr_base_url,
            arguments.asr_model,
            arguments.lang,
        )
        entries_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if arguments.phase == "asr":
        print(f"quality_entries={entries_path}")
        return
    else:
        pass
    score_similarity(entries, arguments.speaker_cache_dir, arguments.device)
    report = {
        "base": backends[0],
        "candidate": backends[1],
        "sample_rate": SAMPLE_RATE,
        "summary": summarize_quality(entries, backends, waveform_paths),
        "per_sample": entries,
    }
    report_path = arguments.output_dir / "quality_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"quality_report={report_path}")


def summarize_milliseconds(samples_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def measure_profile_call(
    vocoder: CosyVoice3Vocoder,
    call: ProfileCall,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, dict[str, object]]:
    device = next(vocoder.flow.parameters()).device
    for _ in range(warmup):
        with (
            torch.inference_mode(),
            vocoder.stream_context,
            torch.autocast(
                device_type=device.type,
                dtype=vocoder.autocast_dtype,
                enabled=vocoder.autocast_dtype is not None,
            ),
        ):
            for _ in range(repeats):
                call()
    torch.cuda.synchronize(device)

    cuda_samples_ms: list[float] = []
    wall_samples_ms: list[float] = []
    for _ in range(iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_started_seconds = time.perf_counter()
        with (
            torch.inference_mode(),
            vocoder.stream_context,
            torch.autocast(
                device_type=device.type,
                dtype=vocoder.autocast_dtype,
                enabled=vocoder.autocast_dtype is not None,
            ),
        ):
            start_event.record()
            for _ in range(repeats):
                call()
            end_event.record()
        end_event.synchronize()
        cuda_samples_ms.append(float(start_event.elapsed_time(end_event)) / repeats)
        wall_samples_ms.append(
            (time.perf_counter() - wall_started_seconds) * 1000 / repeats
        )
    return {
        "cuda": {
            "summary": summarize_milliseconds(cuda_samples_ms),
            "samples_ms": cuda_samples_ms,
        },
        "wall": {
            "summary": summarize_milliseconds(wall_samples_ms),
            "samples_ms": wall_samples_ms,
        },
    }


def measure_attention_planning(
    factory: AttentionFactory,
    device: torch.device,
    iterations: int,
) -> dict[str, object]:
    factory()
    torch.cuda.synchronize(device)
    samples_ms: list[float] = []
    for _ in range(iterations):
        started_seconds = time.perf_counter()
        factory()
        torch.cuda.synchronize(device)
        samples_ms.append((time.perf_counter() - started_seconds) * 1000)
    return {
        "summary": summarize_milliseconds(samples_ms),
        "samples_ms": samples_ms,
    }


def fixed_profile_tensor(
    shape: tuple[int, ...],
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator).to(device=device, dtype=dtype)


def run_profile_streaming_flow(
    vocoder: CosyVoice3Vocoder, flow_input: FlowBatchInput
) -> None:
    token_count = int(flow_input.token.shape[1])
    token_offset = 0
    hop_tokens = TOKEN_HOP_LEN
    cache = None
    try:
        while token_count >= token_offset + hop_tokens + PRE_LOOKAHEAD_LEN:
            token_end = token_offset + hop_tokens + PRE_LOOKAHEAD_LEN
            step_input = FlowBatchInput(
                token=flow_input.token[:, :token_end],
                prompt_token=flow_input.prompt_token,
                prompt_feat=flow_input.prompt_feat,
                embedding=flow_input.embedding,
            )
            total_frames = causal_hop_frames(
                step_input, token_mel_ratio=vocoder.flow.token_mel_ratio
            )
            if cache is None:
                cache = vocoder.prefix_cache_rows(total_frames)
                if cache is None:
                    raise RuntimeError("Prefix cache could not allocate the first hop")
                else:
                    pass
            elif not vocoder.grow_prefix_cache(cache, total_frames):
                raise RuntimeError("Prefix cache could not grow for a profile hop")
            else:
                pass
            vocoder.hop_batch_prefix([step_input], [cache])
            token_offset += hop_tokens
            hop_tokens = next_stream_hop_len(hop_tokens, max_hop_len=TOKEN_MAX_HOP_LEN)
        vocoder.leftover_batch([flow_input])
    finally:
        vocoder.release_prefix_cache(cache)


def profile_speed_comparison(
    base_median_ms: float, candidate_median_ms: float
) -> dict[str, float]:
    return {
        "base_median_ms": base_median_ms,
        "candidate_median_ms": candidate_median_ms,
        "latency_reduction_percent": (
            (base_median_ms - candidate_median_ms) / base_median_ms * 100
        ),
        "speedup_ratio": base_median_ms / candidate_median_ms,
        "speedup_percent": ((base_median_ms / candidate_median_ms - 1) * 100),
    }


def run_profile_round(
    round_name: str,
    backend: AttentionBackend,
    frozen: FrozenArCase,
    arguments: ProfileArguments,
) -> dict[str, object]:
    scheduler = create_vocoder_executor(
        str(arguments.model_path),
        device=arguments.device,
        gpu_id=arguments.gpu_id,
        dtype="bfloat16",
        max_batch_size=1,
        enable_dit_torch_compile=True,
        enable_flow_cuda_graph=False,
        enable_flow_prefix_cuda_graph=arguments.packed_flow_cuda_graphs,
        enable_flow_whole_history_cuda_graph=arguments.packed_flow_cuda_graphs,
        enable_hift_cuda_graph=False,
        flow_prefix_cache_gb=arguments.prefix_cache_gb,
        flow_attention_backend=backend,
    )
    vocoder = scheduler.vocoder
    estimator = vocoder.flow.packed_estimator
    if not isinstance(estimator, PackedDiT):
        raise RuntimeError("Profile requires the PackedDiT estimator")
    elif estimator.attention_backend != backend:
        raise RuntimeError(
            f"Requested backend {backend!r}, got {estimator.attention_backend!r}"
        )
    else:
        pass
    device = next(vocoder.flow.parameters()).device
    flow_input = make_flow_input(frozen, vocoder.flow.token_mel_ratio)

    def whole_flow_call() -> None:
        vocoder.leftover_batch([flow_input])

    def streaming_flow_call() -> None:
        run_profile_streaming_flow(vocoder, flow_input)

    scopes: dict[str, object] = {
        "flow_whole_history": measure_profile_call(
            vocoder,
            whole_flow_call,
            warmup=arguments.warmup,
            iterations=arguments.iterations,
            repeats=1,
        ),
        "flow_streaming_prefix": measure_profile_call(
            vocoder,
            streaming_flow_call,
            warmup=arguments.warmup,
            iterations=arguments.iterations,
            repeats=1,
        ),
    }

    with (
        torch.inference_mode(),
        vocoder.stream_context,
        torch.autocast(
            device_type=device.type,
            dtype=vocoder.autocast_dtype,
            enabled=vocoder.autocast_dtype is not None,
        ),
    ):
        packed = pack_flow_inputs(vocoder.flow.flow, [flow_input])
        conditioning = prepare_flow_conditioning(vocoder.flow, packed, finalize=True)
        rows = pack_rows(conditioning.mel_lengths, device)
        twin_rows = pack_rows(rows.lengths * 2, device)
    first_block_attention = estimator.dit.transformer_blocks[0].attn
    attention_heads = int(first_block_attention.heads)
    attention_inner_dim = int(first_block_attention.inner_dim)
    attention_head_dim = attention_inner_dim // attention_heads
    attention_dtype = conditioning.speaker_embedding.dtype
    whole_query = fixed_profile_tensor(
        (1, twin_rows.total, attention_inner_dim),
        seed=PROFILE_TENSOR_SEED,
        dtype=attention_dtype,
        device=device,
    )
    whole_key = fixed_profile_tensor(
        (1, twin_rows.total, attention_inner_dim),
        seed=PROFILE_TENSOR_SEED + 1,
        dtype=attention_dtype,
        device=device,
    )
    whole_value = fixed_profile_tensor(
        (1, twin_rows.total, attention_inner_dim),
        seed=PROFILE_TENSOR_SEED + 2,
        dtype=attention_dtype,
        device=device,
    )

    def whole_attention_factory() -> PackedRowAttention:
        return estimator.row_attention(
            twin_rows, streaming=False, dtype=attention_dtype
        )

    whole_plan = measure_attention_planning(
        whole_attention_factory, device, arguments.iterations
    )
    whole_attention = whole_attention_factory()

    def whole_attention_call() -> None:
        whole_attention(whole_query, whole_key, whole_value)

    scopes["attention_whole_history"] = {
        "planning_wall": whole_plan,
        "kernel": measure_profile_call(
            vocoder,
            whole_attention_call,
            warmup=arguments.warmup,
            iterations=arguments.iterations,
            repeats=arguments.attention_repeats,
        ),
        "repeats_per_sample": arguments.attention_repeats,
        "layout": {
            "row_lengths": list(twin_rows.lengths),
            "total_frames": twin_rows.total,
            "max_frames": twin_rows.width,
            "qkv_shape": list(whole_query.shape),
        },
    }

    second_hop_tokens = next_stream_hop_len(
        TOKEN_HOP_LEN, max_hop_len=TOKEN_MAX_HOP_LEN
    )
    second_token_end = TOKEN_HOP_LEN + second_hop_tokens + PRE_LOOKAHEAD_LEN
    if int(flow_input.token.shape[1]) < second_token_end:
        raise RuntimeError(
            f"Prefix attention profile needs at least {second_token_end} frozen "
            f"speech tokens, got {flow_input.token.shape[1]}"
        )
    else:
        pass
    first_step_input = FlowBatchInput(
        token=flow_input.token[:, : TOKEN_HOP_LEN + PRE_LOOKAHEAD_LEN],
        prompt_token=flow_input.prompt_token,
        prompt_feat=flow_input.prompt_feat,
        embedding=flow_input.embedding,
    )
    second_step_input = FlowBatchInput(
        token=flow_input.token[:, :second_token_end],
        prompt_token=flow_input.prompt_token,
        prompt_feat=flow_input.prompt_feat,
        embedding=flow_input.embedding,
    )
    first_total_frames = causal_hop_frames(
        first_step_input, token_mel_ratio=vocoder.flow.token_mel_ratio
    )
    second_total_frames = causal_hop_frames(
        second_step_input, token_mel_ratio=vocoder.flow.token_mel_ratio
    )
    prefix_cache = vocoder.prefix_cache_rows(first_total_frames)
    if prefix_cache is None or vocoder.flow.prefix_pool is None:
        raise RuntimeError("Prefix cache could not allocate the attention profile")
    elif not vocoder.grow_prefix_cache(prefix_cache, second_total_frames):
        vocoder.release_prefix_cache(prefix_cache)
        raise RuntimeError("Prefix cache could not grow the attention profile")
    else:
        pass
    committed_frames = first_total_frames // estimator.chunk_size * estimator.chunk_size
    for cache_row in prefix_cache:
        cache_row.committed_frames = committed_frames
    new_frames = second_total_frames - committed_frames
    twin_caches = [prefix_cache[0], prefix_cache[1]]
    twin_new_frames = [new_frames, new_frames]
    end_frames = [
        cache_row.committed_frames + new_frame_count
        for cache_row, new_frame_count in zip(twin_caches, twin_new_frames, strict=True)
    ]
    prefix_pool = vocoder.flow.prefix_pool

    def prefix_attention_factory() -> PrefixRowAttention:
        return PrefixRowAttention(
            rows=twin_caches,
            new_frames=twin_new_frames,
            layout=PrefixStepLayout(
                half_frames=new_frames,
                row_slots=1,
                segment_count=sum(
                    -(-end_frame // estimator.chunk_size)
                    - cache_row.committed_frames // estimator.chunk_size
                    for cache_row, end_frame in zip(
                        twin_caches, end_frames, strict=True
                    )
                ),
                page_table_width=max(end_frames),
            ),
            padding_block=prefix_pool.padding_block,
            chunk_size=estimator.chunk_size,
            device=device,
            flashinfer_runner=estimator.flashinfer_runner,
            heads=attention_heads,
            head_dim=attention_head_dim,
            dtype=attention_dtype,
        )

    try:
        prefix_plan = measure_attention_planning(
            prefix_attention_factory, device, arguments.iterations
        )
        prefix_attention = prefix_attention_factory()
        prefix_query_frames = sum(twin_new_frames)
        prefix_query = fixed_profile_tensor(
            (1, prefix_query_frames, attention_inner_dim),
            seed=PROFILE_TENSOR_SEED + 3,
            dtype=attention_dtype,
            device=device,
        )
        prefix_key = fixed_profile_tensor(
            (1, prefix_query_frames, attention_inner_dim),
            seed=PROFILE_TENSOR_SEED + 4,
            dtype=attention_dtype,
            device=device,
        )
        prefix_value = fixed_profile_tensor(
            (1, prefix_query_frames, attention_inner_dim),
            seed=PROFILE_TENSOR_SEED + 5,
            dtype=attention_dtype,
            device=device,
        )
        key_pool = prefix_pool.keys[0][0]
        value_pool = prefix_pool.values[0][0]
        page_indices = torch.unique(prefix_attention.page_table).to(torch.long)
        pool_page_shape = (
            len(page_indices),
            key_pool.shape[1],
            attention_heads,
            attention_head_dim,
        )
        key_pool.index_copy_(
            0,
            page_indices,
            fixed_profile_tensor(
                pool_page_shape,
                seed=PROFILE_TENSOR_SEED + 6,
                dtype=attention_dtype,
                device=device,
            ),
        )
        value_pool.index_copy_(
            0,
            page_indices,
            fixed_profile_tensor(
                pool_page_shape,
                seed=PROFILE_TENSOR_SEED + 7,
                dtype=attention_dtype,
                device=device,
            ),
        )

        def prefix_attention_call() -> None:
            prefix_attention(
                prefix_query,
                prefix_key,
                prefix_value,
                key_pool,
                value_pool,
                attention_heads,
                attention_head_dim,
            )

        scopes["attention_streaming_prefix"] = {
            "planning_wall": prefix_plan,
            "kernel": measure_profile_call(
                vocoder,
                prefix_attention_call,
                warmup=arguments.warmup,
                iterations=arguments.iterations,
                repeats=arguments.attention_repeats,
            ),
            "repeats_per_sample": arguments.attention_repeats,
            "layout": {
                "first_total_frames": first_total_frames,
                "committed_prefix_frames": committed_frames,
                "second_total_frames": second_total_frames,
                "new_frames_per_cfg_half": new_frames,
                "query_frames": prefix_query_frames,
                "qkv_shape": list(prefix_query.shape),
                "cache_sequence_lengths": prefix_attention.cache_seqlens.tolist(),
                "cumulative_query_lengths": (prefix_attention.cu_seqlens_q.tolist()),
                "page_table_shape": list(prefix_attention.page_table.shape),
                "page_table_sha256": hashlib.sha256(
                    prefix_attention.page_table.detach()
                    .cpu()
                    .contiguous()
                    .numpy()
                    .tobytes()
                ).hexdigest(),
            },
        }
    finally:
        vocoder.release_prefix_cache(prefix_cache)

    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    report = {
        "round": round_name,
        "requested_backend": backend,
        "resolved_packed_backend": estimator.attention_backend,
        "flashinfer_kernel_backend": ("fa2" if backend == "flashinfer_fa2" else None),
        "prefix_cache_capacity_frames": prefix_pool.free_frames,
        "gpu": {
            "name": torch.cuda.get_device_name(device_index),
            "compute_capability": list(torch.cuda.get_device_capability(device_index)),
        },
        "scopes": scopes,
    }
    print(
        f"round={round_name} backend={backend} "
        f"flow_whole_ms={scopes['flow_whole_history']['cuda']['summary']['median_ms']:.3f} "
        f"flow_streaming_ms={scopes['flow_streaming_prefix']['cuda']['summary']['median_ms']:.3f} "
        f"attention_whole_ms={scopes['attention_whole_history']['kernel']['cuda']['summary']['median_ms']:.4f} "
        f"attention_prefix_ms={scopes['attention_streaming_prefix']['kernel']['cuda']['summary']['median_ms']:.4f}",
        flush=True,
    )
    del scheduler, vocoder
    gc.collect()
    torch.cuda.empty_cache()
    return report


def combined_profile_scope(
    rounds: list[dict[str, object]],
    backend: AttentionBackend,
    scope_name: str,
    measurement_name: str | None,
) -> dict[str, dict[str, float]]:
    selected = [
        round_report
        for round_report in rounds
        if round_report["requested_backend"] == backend
    ]
    combined: dict[str, dict[str, float]] = {}
    for clock_name in ("cuda", "wall"):
        if measurement_name is None:
            samples_ms = [
                float(sample_ms)
                for round_report in selected
                for sample_ms in round_report["scopes"][scope_name][clock_name][
                    "samples_ms"
                ]
            ]
        else:
            samples_ms = [
                float(sample_ms)
                for round_report in selected
                for sample_ms in round_report["scopes"][scope_name][measurement_name][
                    clock_name
                ]["samples_ms"]
            ]
        combined[clock_name] = summarize_milliseconds(samples_ms)
    return combined


def combined_profile_plan(
    rounds: list[dict[str, object]],
    backend: AttentionBackend,
    scope_name: str,
) -> dict[str, float]:
    samples_ms = [
        float(sample_ms)
        for round_report in rounds
        if round_report["requested_backend"] == backend
        for sample_ms in round_report["scopes"][scope_name]["planning_wall"][
            "samples_ms"
        ]
    ]
    return summarize_milliseconds(samples_ms)


def run_profile(arguments: ProfileArguments) -> None:
    if arguments.warmup < 10:
        raise ValueError("--warmup must be at least 10")
    elif arguments.iterations <= 0:
        raise ValueError("--iterations must be positive")
    elif arguments.attention_repeats <= 0:
        raise ValueError("--attention-repeats must be positive")
    elif arguments.prefix_cache_gb <= 0:
        raise ValueError("--prefix-cache-gb must be greater than zero")
    else:
        pass
    frozen = load_frozen_case(arguments.frozen_ar)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    rounds = [
        run_profile_round(round_name, backend, frozen, arguments)
        for round_name, backend in ROUND_ORDER
    ]
    for scope_name in (
        "attention_whole_history",
        "attention_streaming_prefix",
    ):
        layouts = [
            round_report["scopes"][scope_name]["layout"] for round_report in rounds
        ]
        if any(layout != layouts[0] for layout in layouts[1:]):
            raise RuntimeError(
                f"Profile rounds did not use the same {scope_name} layout"
            )
        else:
            pass
    combined: dict[str, object] = {}
    comparisons: dict[str, object] = {}
    for scope_name in ("flow_whole_history", "flow_streaming_prefix"):
        base = combined_profile_scope(
            rounds, BACKENDS[0], scope_name, measurement_name=None
        )
        candidate = combined_profile_scope(
            rounds, BACKENDS[1], scope_name, measurement_name=None
        )
        combined[scope_name] = {"base": base, "candidate": candidate}
        comparisons[scope_name] = {
            "cuda": profile_speed_comparison(
                base["cuda"]["median_ms"], candidate["cuda"]["median_ms"]
            ),
            "wall": profile_speed_comparison(
                base["wall"]["median_ms"], candidate["wall"]["median_ms"]
            ),
        }
    for scope_name in (
        "attention_whole_history",
        "attention_streaming_prefix",
    ):
        base_kernel = combined_profile_scope(
            rounds, BACKENDS[0], scope_name, measurement_name="kernel"
        )
        candidate_kernel = combined_profile_scope(
            rounds, BACKENDS[1], scope_name, measurement_name="kernel"
        )
        base_plan = combined_profile_plan(rounds, BACKENDS[0], scope_name)
        candidate_plan = combined_profile_plan(rounds, BACKENDS[1], scope_name)
        combined[scope_name] = {
            "base": {"planning_wall": base_plan, "kernel": base_kernel},
            "candidate": {
                "planning_wall": candidate_plan,
                "kernel": candidate_kernel,
            },
        }
        comparisons[scope_name] = {
            "planning_wall": profile_speed_comparison(
                base_plan["median_ms"], candidate_plan["median_ms"]
            ),
            "kernel_cuda": profile_speed_comparison(
                base_kernel["cuda"]["median_ms"],
                candidate_kernel["cuda"]["median_ms"],
            ),
            "kernel_wall": profile_speed_comparison(
                base_kernel["wall"]["median_ms"],
                candidate_kernel["wall"]["median_ms"],
            ),
        }
    report = {
        "model_path": str(arguments.model_path),
        "frozen_ar": str(arguments.frozen_ar),
        "request": {
            "request_id": frozen.request_id,
            "seed": frozen.seed,
            "speech_tokens": int(frozen.audio_codes.numel()),
        },
        "gpu": rounds[0]["gpu"],
        "base": BACKENDS[0],
        "candidate": BACKENDS[1],
        "round_order": [round_name for round_name, _ in ROUND_ORDER],
        "backend_order": [backend for _, backend in ROUND_ORDER],
        "warmup": arguments.warmup,
        "iterations_per_round": arguments.iterations,
        "attention_repeats": arguments.attention_repeats,
        "dtype": "bfloat16",
        "dit_compiled": True,
        "flow_cuda_graphs_enabled": False,
        "packed_flow_cuda_graphs_enabled": arguments.packed_flow_cuda_graphs,
        "prefix_cache_gb_requested": arguments.prefix_cache_gb,
        "fixed_qkv_seed": PROFILE_TENSOR_SEED,
        "backend_note": (
            "sglang_fa3 is the requested SGLang backend name; on Ampere it is "
            "a compatibility route and must not be reported as native FA3."
        ),
        "scope_definitions": {
            "end_to_end": (
                "Use the abba command: AR + Flow/PackedDiT + HiFT + HTTP streaming."
            ),
            "flow_whole_history": (
                "Frozen AR input through buffered Flow/PackedDiT only; excludes AR, "
                "HiFT and HTTP."
            ),
            "flow_streaming_prefix": (
                "Frozen AR input through every PrefixKV Flow hop and final "
                "whole-history Flow call; excludes AR, HiFT and HTTP."
            ),
            "attention_whole_history": (
                "Attention kernel only with identical fixed Q/K/V and packed row "
                "layout; planning is reported separately."
            ),
            "attention_streaming_prefix": (
                "PrefixKV attention kernel only on a filled-prefix hop with identical "
                "fixed Q/K/V and semantic page layout; planning is reported separately."
            ),
        },
        "rounds": rounds,
        "combined": combined,
        "comparisons": comparisons,
    }
    report_path = arguments.output_dir / "profile_summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"profile_report={report_path}")


def run_replay(arguments: ReplayArguments) -> None:
    if arguments.prefix_cache_gb <= 0:
        raise ValueError("--prefix-cache-gb must be greater than zero")
    elif len(arguments.backends) not in (1, 2):
        raise ValueError(
            "--backends accepts one replay backend or two backends to compare"
        )
    else:
        pass
    frozen_paths = sorted(arguments.frozen_ar_dir.glob("*.pt"))
    if not frozen_paths:
        raise FileNotFoundError(f"No .pt files found in {arguments.frozen_ar_dir}")
    else:
        pass
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_cases = [load_frozen_case(path) for path in frozen_paths]
    patch_chunk_mask()
    for backend in arguments.backends:
        run_backend(backend, frozen_cases, arguments)
    if len(arguments.backends) == 2:
        compare_backends(frozen_cases, arguments)
        print(f"report={arguments.output_dir / 'comparison.json'}")
    else:
        print(
            f"manifest={arguments.output_dir / arguments.backends[0] / 'manifest.json'}"
        )


def add_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.21)


def add_server_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=int, default=600)


def add_packed_graph_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--packed-flow-cuda-graphs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Capture and replay PackedDiT PrefixKV and whole-history CUDA graphs. "
            "Use this flag in a separate ABBA/profile run when scanning graph off/on."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fun-CosyVoice3 acceptance: SeedTTS AR freeze, frozen replay accuracy, "
            "quality evaluation, end-to-end ABBA performance and nested timing."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser(
        "freeze",
        help="Capture deterministic AR outputs from the official SeedTTS workload.",
    )
    freeze.add_argument("--model-path", type=Path, required=True)
    freeze.add_argument("--meta", default=DEFAULT_SEEDTTS_DATASET)
    freeze.add_argument("--split", choices=("en", "zh"), default="zh")
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--frozen-ar-dir", type=Path, required=True)
    freeze.add_argument("--max-samples", type=int, default=5)
    freeze.add_argument("--backend", choices=BACKENDS, default=BACKENDS[0])
    add_sampling_arguments(freeze)
    add_server_arguments(freeze)
    add_packed_graph_argument(freeze)

    replay = commands.add_parser(
        "replay",
        help="Replay frozen AR cases through both PackedDiT backends and compare tensors.",
    )
    replay.add_argument("--model-path", type=Path, required=True)
    replay.add_argument("--frozen-ar-dir", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    replay.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS)
    )
    replay.add_argument("--device", default="cuda:0")
    replay.add_argument(
        "--prefix-cache-gb", type=float, default=DEFAULT_PREFIX_CACHE_GB
    )

    quality = commands.add_parser(
        "quality",
        help="Measure frozen replay WER, CER and WavLM speaker similarity.",
    )
    quality.add_argument("--replay-dir", type=Path, required=True)
    quality.add_argument("--output-dir", type=Path, required=True)
    quality.add_argument("--speaker-cache-dir", type=Path, required=True)
    quality.add_argument("--asr-base-url", default="http://127.0.0.1:8090")
    quality.add_argument("--asr-model", required=True)
    quality.add_argument("--lang", choices=("en", "zh"), default="zh")
    quality.add_argument("--device", default="cuda:0")
    quality.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS)
    )
    quality.add_argument(
        "--paths",
        nargs="+",
        choices=tuple(WAVEFORM_PATHS),
        default=list(WAVEFORM_PATHS),
    )
    quality.add_argument(
        "--phase",
        choices=("all", "asr", "similarity"),
        default="all",
    )

    abba = commands.add_parser(
        "abba",
        help="Run strict A-B-B-A end-to-end serving comparisons on SeedTTS.",
    )
    abba.add_argument("--model-path", type=Path, required=True)
    abba.add_argument("--meta", default=DEFAULT_SEEDTTS_DATASET)
    abba.add_argument("--split", choices=("en", "zh"), default="zh")
    abba.add_argument("--output-dir", type=Path, required=True)
    abba.add_argument(
        "--concurrencies",
        type=parse_concurrencies,
        default=(1, 4, 8, 16),
    )
    abba.add_argument("--max-samples", type=int)
    abba.add_argument("--warmup", type=int, default=10)
    add_sampling_arguments(abba)
    add_server_arguments(abba)
    add_packed_graph_argument(abba)

    profile = commands.add_parser(
        "profile",
        help=("Run A-B-B-A Flow/PackedDiT and attention-only timing on frozen AR."),
    )
    profile.add_argument("--model-path", type=Path, required=True)
    profile.add_argument("--frozen-ar", type=Path, required=True)
    profile.add_argument("--output-dir", type=Path, required=True)
    profile.add_argument("--warmup", type=int, default=10)
    profile.add_argument("--iterations", type=int, default=20)
    profile.add_argument("--attention-repeats", type=int, default=100)
    profile.add_argument(
        "--prefix-cache-gb", type=float, default=DEFAULT_PREFIX_CACHE_GB
    )
    profile.add_argument("--device", default="cuda")
    profile.add_argument("--gpu-id", type=int, default=0)
    add_packed_graph_argument(profile)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.command == "freeze":
        run_freeze(arguments)
    elif arguments.command == "replay":
        run_replay(arguments)
    elif arguments.command == "quality":
        run_quality(arguments)
    elif arguments.command == "abba":
        run_abba(arguments)
    elif arguments.command == "profile":
        run_profile(arguments)
    else:
        raise RuntimeError(f"Unsupported command {arguments.command!r}")


if __name__ == "__main__":
    main()
else:
    pass
