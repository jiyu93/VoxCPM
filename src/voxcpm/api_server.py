import argparse
import asyncio
import base64
import binascii
import gc
import hashlib
import hmac
import json
import os
import re
import time
import uuid
import wave
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Literal

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".webm"}
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
DEFAULT_HF_MODEL_ID = "openbmb/VoxCPM2"
VOICE_MODES = {"auto", "plain", "design", "clone", "continuation", "hybrid"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _new_id() -> str:
    return uuid.uuid4().hex


def _clean_instruction(instruction: str | None) -> str:
    value = (instruction or "").strip()
    return re.sub(r"[()（）]", "", value).strip()


def _compose_voxcpm_text(text: str, instruction: str | None) -> str:
    clean = _clean_instruction(instruction)
    text = text.strip()
    return f"({clean}){text}" if clean else text


@dataclass
class ServerSettings:
    model: str = os.getenv("VOXCPM_MODEL", DEFAULT_HF_MODEL_ID)
    output_dir: Path = Path(os.getenv("VOXCPM_OUTPUT_DIR", "outputs/api"))
    host: str = os.getenv("VOXCPM_HOST", "0.0.0.0")
    port: int = _env_int("VOXCPM_PORT", 7862)
    api_key: str | None = os.getenv("VOXCPM_API_KEY")
    cors_origins: list[str] | None = None
    device: str | None = os.getenv("VOXCPM_DEVICE", "auto")
    cache_dir: str | None = os.getenv("VOXCPM_CACHE_DIR")
    local_files_only: bool = _env_bool("VOXCPM_LOCAL_FILES_ONLY", False)
    lazy_load: bool = _env_bool("VOXCPM_LAZY_LOAD", True)
    optimize: bool = _env_bool("VOXCPM_OPTIMIZE", True)
    load_denoiser: bool = _env_bool("VOXCPM_LOAD_DENOISER", True)
    zipenhancer_model: str | None = os.getenv("VOXCPM_ZIPENHANCER_MODEL")
    allow_local_paths: bool = _env_bool("VOXCPM_ALLOW_LOCAL_PATHS", False)
    max_upload_mb: int = _env_int("VOXCPM_MAX_UPLOAD_MB", 50)
    max_queue_size: int = _env_int("VOXCPM_MAX_QUEUE_SIZE", 100)
    default_cfg_value: float = _env_float("VOXCPM_DEFAULT_CFG_VALUE", 2.0)
    default_inference_timesteps: int = _env_int("VOXCPM_DEFAULT_INFERENCE_TIMESTEPS", 10)
    lora_weights_path: str | None = os.getenv("VOXCPM_LORA_PATH")
    lora_enable_lm: bool = _env_bool("VOXCPM_LORA_ENABLE_LM", True)
    lora_enable_dit: bool = _env_bool("VOXCPM_LORA_ENABLE_DIT", True)
    lora_enable_proj: bool = _env_bool("VOXCPM_LORA_ENABLE_PROJ", False)
    lora_r: int = _env_int("VOXCPM_LORA_R", 32)
    lora_alpha: int = _env_int("VOXCPM_LORA_ALPHA", 16)
    lora_dropout: float = _env_float("VOXCPM_LORA_DROPOUT", 0.0)

    def __post_init__(self) -> None:
        self.output_dir = self.output_dir.expanduser().resolve()
        if self.cors_origins is None:
            raw = os.getenv("VOXCPM_CORS_ORIGINS", "")
            self.cors_origins = [origin.strip() for origin in raw.split(",") if origin.strip()]


class GenerationSettings(BaseModel):
    cfg_value: float = Field(2.0, ge=0.1, le=10.0)
    inference_timesteps: int = Field(10, ge=1, le=100)
    min_len: int = Field(2, ge=0)
    max_len: int = Field(4096, ge=1)
    normalize_text: bool = True
    denoise_reference_audio: bool = False
    retry_badcase: bool = True
    retry_badcase_max_times: int = Field(3, ge=0, le=10)
    retry_badcase_ratio_threshold: float = Field(6.0, gt=0.0)


class InlineAudio(BaseModel):
    audio_base64: str = Field(..., min_length=1)
    filename: str = "reference.wav"
    content_type: str | None = None


class SynthesisVoice(BaseModel):
    mode: Literal["auto", "plain", "design", "clone", "continuation", "hybrid"] = "auto"
    instruction: str | None = None
    reference_id: str | None = None
    reference_audio: InlineAudio | None = None
    reference_audio_path: str | None = None
    prompt_id: str | None = None
    prompt_audio: InlineAudio | None = None
    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    use_prompt_as_reference: bool = True


class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: SynthesisVoice = Field(default_factory=SynthesisVoice)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)


class StreamSynthesisRequest(SynthesisRequest):
    chunk_encoding: Literal["base64_f32le"] = "base64_f32le"


class ReferenceJsonRequest(BaseModel):
    audio_base64: str = Field(..., min_length=1)
    filename: str = "reference.wav"
    content_type: str | None = None


class ReferenceAsset(BaseModel):
    reference_id: str
    sha256: str
    original_filename: str
    path: str
    content_type: str | None = None
    size_bytes: int
    created_at: float
    last_used_at: float | None = None
    use_count: int = 0


class ResolvedSynthesis(BaseModel):
    text: str
    mode: Literal["plain", "design", "clone", "continuation", "hybrid"]
    instruction: str | None = None
    reference_audio_path: str | None = None
    reference_id: str | None = None
    prompt_audio_path: str | None = None
    prompt_id: str | None = None
    prompt_text: str | None = None
    generation: GenerationSettings


class SynthesisRecord(BaseModel):
    synthesis_id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"] = "queued"
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: str = "queued"
    created_at: float
    updated_at: float
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_seconds: float | None = None
    queue_position: int | None = None
    mode: Literal["plain", "design", "clone", "continuation", "hybrid"]
    text: str
    instruction: str | None = None
    reference_id: str | None = None
    prompt_id: str | None = None
    has_prompt_text: bool = False
    output_path: str | None = None
    audio_url: str | None = None
    duration_seconds: float | None = None
    sample_rate: int | None = None
    error: str | None = None


class SynthesisSubmitResponse(BaseModel):
    synthesis_id: str
    status: str
    progress: float
    message: str
    synthesis_url: str
    events_url: str
    audio_url: str
    queue_position: int | None = None


class LoRAStatus(BaseModel):
    configured: bool
    enabled: bool | None = None
    weights_path: str | None = None
    enable_lm: bool
    enable_dit: bool
    enable_proj: bool
    r: int
    alpha: int
    dropout: float


class VoxCPMService:
    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        self.model: VoxCPM | None = None
        self.load_started_at: float | None = None
        self.loaded_at: float | None = None
        self.load_error: str | None = None
        self.load_lock = asyncio.Lock()
        self.infer_lock = asyncio.Lock()
        self.synthesis_lock = RLock()
        self.reference_lock = RLock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.synthesis_records: dict[str, SynthesisRecord] = {}
        self.requests: dict[str, ResolvedSynthesis] = {}
        self.subscribers: dict[str, set[asyncio.Queue[None]]] = {}
        self.queued_ids: deque[str] = deque()
        self.queue: asyncio.Queue[str] | None = None
        self.worker_task: asyncio.Task[None] | None = None
        self.active_synthesis_id: str | None = None
        self.active_started_at: float | None = None
        self.active_streams = 0

        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        (self.settings.output_dir / "results").mkdir(parents=True, exist_ok=True)
        self.references_dir = self.settings.output_dir / "references"
        self.references_dir.mkdir(parents=True, exist_ok=True)
        self.reference_registry_path = self.references_dir / "references.json"
        self.references: dict[str, ReferenceAsset] = self._load_reference_registry()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.model is not None:
            return
        self.load_started_at = time.time()
        self.load_error = None
        try:
            lora_config = None
            if self.settings.lora_weights_path:
                lora_config = LoRAConfig(
                    enable_lm=self.settings.lora_enable_lm,
                    enable_dit=self.settings.lora_enable_dit,
                    enable_proj=self.settings.lora_enable_proj,
                    r=self.settings.lora_r,
                    alpha=self.settings.lora_alpha,
                    dropout=self.settings.lora_dropout,
                )
            self.model = VoxCPM.from_pretrained(
                hf_model_id=self.settings.model,
                load_denoiser=self.settings.load_denoiser,
                zipenhancer_model_id=self.settings.zipenhancer_model or "iic/speech_zipenhancer_ans_multiloss_16k_base",
                cache_dir=self.settings.cache_dir,
                local_files_only=self.settings.local_files_only,
                optimize=self.settings.optimize,
                device=self.settings.device,
                lora_config=lora_config,
                lora_weights_path=self.settings.lora_weights_path,
            )
            self.loaded_at = time.time()
        except Exception as exc:
            self.load_error = repr(exc)
            raise

    async def ensure_loaded(self) -> VoxCPM:
        if self.model is None:
            async with self.load_lock:
                if self.model is None:
                    try:
                        await asyncio.to_thread(self.load)
                    except Exception as exc:
                        raise HTTPException(status_code=503, detail=f"Failed to load VoxCPM model: {exc}") from exc
        if self.model is None:
            raise RuntimeError("VoxCPM model is not loaded")
        return self.model

    def unload(self) -> None:
        if self.active_synthesis_id is not None or self.active_streams > 0:
            raise HTTPException(status_code=409, detail="Cannot unload while synthesis is running")
        self.model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        if self.queue is None:
            self.queue = asyncio.Queue(maxsize=self.settings.max_queue_size)
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        if self.worker_task is None:
            return
        self.worker_task.cancel()
        try:
            await self.worker_task
        except asyncio.CancelledError:
            pass

    def _load_reference_registry(self) -> dict[str, ReferenceAsset]:
        if not self.reference_registry_path.exists():
            return {}
        try:
            raw = json.loads(self.reference_registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        references = {}
        for item in raw.get("references", []):
            asset = ReferenceAsset.model_validate(item)
            references[asset.reference_id] = asset
        return references

    def _save_reference_registry(self) -> None:
        payload = {
            "references": [
                asset.model_dump() for asset in sorted(self.references.values(), key=lambda item: item.created_at)
            ]
        }
        tmp_path = self.reference_registry_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.reference_registry_path)

    def _register_reference(
        self,
        *,
        tmp_path: Path,
        sha256: str,
        suffix: str,
        original_filename: str,
        content_type: str | None,
        size_bytes: int,
    ) -> ReferenceAsset:
        reference_id = sha256
        with self.reference_lock:
            existing = self.references.get(reference_id)
            if existing is not None and Path(existing.path).exists():
                tmp_path.unlink(missing_ok=True)
                return existing
            destination = self.references_dir / f"{sha256}{suffix}"
            if destination.exists():
                tmp_path.unlink(missing_ok=True)
            else:
                tmp_path.replace(destination)
            asset = ReferenceAsset(
                reference_id=reference_id,
                sha256=sha256,
                original_filename=original_filename,
                path=str(destination),
                content_type=content_type,
                size_bytes=size_bytes,
                created_at=time.time(),
            )
            self.references[reference_id] = asset
            self._save_reference_registry()
            return asset

    async def create_reference_from_upload(self, upload: UploadFile) -> ReferenceAsset:
        suffix = Path(upload.filename or "").suffix.lower() or ".wav"
        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported audio extension: {suffix}")
        tmp_path = self.references_dir / f".{_new_id()}.tmp"
        digest = hashlib.sha256()
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        written = 0
        with tmp_path.open("wb") as file:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    tmp_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413, detail=f"Reference audio exceeds {self.settings.max_upload_mb} MB"
                    )
                digest.update(chunk)
                file.write(chunk)
        if written == 0:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Uploaded audio is empty")
        return self._register_reference(
            tmp_path=tmp_path,
            sha256=digest.hexdigest(),
            suffix=suffix,
            original_filename=upload.filename or f"reference{suffix}",
            content_type=upload.content_type,
            size_bytes=written,
        )

    def create_reference_from_base64(
        self, audio_base64: str, filename: str, content_type: str | None = None
    ) -> ReferenceAsset:
        suffix = Path(filename).suffix.lower() or ".wav"
        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported audio extension: {suffix}")
        try:
            if "," in audio_base64:
                audio_base64 = audio_base64.split(",", 1)[1]
            data = base64.b64decode(audio_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid base64 audio") from exc
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Reference audio exceeds {self.settings.max_upload_mb} MB")
        if not data:
            raise HTTPException(status_code=400, detail="Audio payload is empty")
        tmp_path = self.references_dir / f".{_new_id()}.tmp"
        tmp_path.write_bytes(data)
        return self._register_reference(
            tmp_path=tmp_path,
            sha256=hashlib.sha256(data).hexdigest(),
            suffix=suffix,
            original_filename=filename,
            content_type=content_type,
            size_bytes=len(data),
        )

    def list_references(self) -> list[ReferenceAsset]:
        with self.reference_lock:
            return sorted(self.references.values(), key=lambda item: item.created_at, reverse=True)

    def get_reference(self, reference_id: str) -> ReferenceAsset:
        with self.reference_lock:
            asset = self.references.get(reference_id)
            if asset is None:
                raise HTTPException(status_code=404, detail="Reference not found")
            if not Path(asset.path).exists():
                raise HTTPException(status_code=404, detail="Reference audio file not found")
            return asset

    def resolve_reference_path(self, reference_id: str) -> Path:
        with self.reference_lock:
            asset = self.get_reference(reference_id)
            updated = asset.model_copy(update={"last_used_at": time.time(), "use_count": asset.use_count + 1})
            self.references[reference_id] = updated
            self._save_reference_registry()
            return Path(updated.path)

    def delete_reference(self, reference_id: str) -> ReferenceAsset:
        with self.reference_lock:
            asset = self.references.get(reference_id)
            if asset is None:
                raise HTTPException(status_code=404, detail="Reference not found")
            for synthesis_id, request in self.requests.items():
                record = self.synthesis_records.get(synthesis_id)
                if (
                    record
                    and record.status in {"queued", "running"}
                    and reference_id
                    in {
                        request.reference_id,
                        request.prompt_id,
                    }
                ):
                    raise HTTPException(status_code=409, detail="Reference is used by an active synthesis")
            self.references.pop(reference_id)
            self._save_reference_registry()
        Path(asset.path).unlink(missing_ok=True)
        return asset

    def resolve_local_audio_path(self, value: str | None, field_name: str) -> Path | None:
        if not value:
            return None
        if not self.settings.allow_local_paths:
            raise HTTPException(
                status_code=400, detail=f"{field_name} is disabled. Upload audio or use base64 instead."
            )
        path = Path(value).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=400, detail=f"{field_name} does not exist: {value}")
        if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported audio extension: {path.suffix}")
        return path

    def resolve_inline_or_registered_audio(
        self,
        *,
        reference_id: str | None,
        inline: InlineAudio | None,
        local_path: str | None,
        field_name: str,
    ) -> tuple[Path | None, str | None]:
        if reference_id:
            return self.resolve_reference_path(reference_id), reference_id
        if inline is not None:
            asset = self.create_reference_from_base64(inline.audio_base64, inline.filename, inline.content_type)
            return self.resolve_reference_path(asset.reference_id), asset.reference_id
        return self.resolve_local_audio_path(local_path, field_name), None

    def infer_mode(
        self,
        *,
        requested_mode: str,
        instruction: str | None,
        reference_path: Path | None,
        prompt_path: Path | None,
        prompt_text: str | None,
    ) -> Literal["plain", "design", "clone", "continuation", "hybrid"]:
        has_instruction = bool(_clean_instruction(instruction))
        has_reference = reference_path is not None
        has_prompt = prompt_path is not None and bool((prompt_text or "").strip())
        if requested_mode != "auto":
            if requested_mode == "design" and (has_reference or has_prompt):
                raise HTTPException(status_code=400, detail="mode=design does not accept reference or prompt audio")
            if requested_mode == "clone" and (not has_reference or has_prompt):
                raise HTTPException(status_code=400, detail="mode=clone requires reference audio and no prompt audio")
            if requested_mode == "continuation" and not has_prompt:
                raise HTTPException(status_code=400, detail="mode=continuation requires prompt audio and prompt_text")
            if requested_mode == "hybrid" and not (has_reference and has_prompt):
                raise HTTPException(
                    status_code=400, detail="mode=hybrid requires reference audio plus prompt audio/text"
                )
            return requested_mode  # type: ignore[return-value]
        if has_reference and has_prompt:
            return "hybrid"
        if has_prompt:
            return "continuation"
        if has_reference:
            return "clone"
        if has_instruction:
            return "design"
        return "plain"

    def resolve_synthesis(self, body: SynthesisRequest) -> ResolvedSynthesis:
        voice = body.voice
        generation = body.generation
        if generation.max_len < generation.min_len:
            raise HTTPException(status_code=400, detail="generation.max_len must be greater than generation.min_len")

        reference_path, resolved_reference_id = self.resolve_inline_or_registered_audio(
            reference_id=voice.reference_id,
            inline=voice.reference_audio,
            local_path=voice.reference_audio_path,
            field_name="voice.reference_audio_path",
        )
        prompt_path, resolved_prompt_id = self.resolve_inline_or_registered_audio(
            reference_id=voice.prompt_id,
            inline=voice.prompt_audio,
            local_path=voice.prompt_audio_path,
            field_name="voice.prompt_audio_path",
        )
        prompt_text = (voice.prompt_text or "").strip() or None
        if prompt_path is not None and prompt_text is None:
            raise HTTPException(status_code=400, detail="voice.prompt_text is required when prompt audio is provided")
        if prompt_text is not None and prompt_path is None:
            raise HTTPException(status_code=400, detail="prompt audio is required when voice.prompt_text is provided")
        if reference_path is None and prompt_path is not None and voice.use_prompt_as_reference:
            reference_path = prompt_path
            resolved_reference_id = resolved_prompt_id
        mode = self.infer_mode(
            requested_mode=voice.mode,
            instruction=voice.instruction,
            reference_path=reference_path,
            prompt_path=prompt_path,
            prompt_text=prompt_text,
        )
        return ResolvedSynthesis(
            text=body.text.strip(),
            mode=mode,
            instruction=_clean_instruction(voice.instruction) or None,
            reference_audio_path=str(reference_path) if reference_path else None,
            reference_id=resolved_reference_id or voice.reference_id,
            prompt_audio_path=str(prompt_path) if prompt_path else None,
            prompt_id=resolved_prompt_id or voice.prompt_id,
            prompt_text=prompt_text,
            generation=generation,
        )

    async def enqueue(self, request: ResolvedSynthesis) -> SynthesisRecord:
        if self.queue is None:
            await self.start()
        if self.queue is None:
            raise HTTPException(status_code=503, detail="Synthesis queue is not available")
        if self.queue.full():
            raise HTTPException(status_code=429, detail="Synthesis queue is full")
        synthesis_id = _new_id()
        now = time.time()
        record = SynthesisRecord(
            synthesis_id=synthesis_id,
            created_at=now,
            updated_at=now,
            mode=request.mode,
            text=request.text,
            instruction=request.instruction,
            reference_id=request.reference_id,
            prompt_id=request.prompt_id,
            has_prompt_text=bool(request.prompt_text),
            queue_position=self.queue.qsize() + 1,
        )
        with self.synthesis_lock:
            self.synthesis_records[synthesis_id] = record
            self.requests[synthesis_id] = request
            self.queued_ids.append(synthesis_id)
        await self.queue.put(synthesis_id)
        return self.get_synthesis(synthesis_id)

    def _queue_position(self, synthesis_id: str) -> int | None:
        try:
            return list(self.queued_ids).index(synthesis_id) + 1
        except ValueError:
            return None

    def get_synthesis(self, synthesis_id: str) -> SynthesisRecord:
        with self.synthesis_lock:
            record = self.synthesis_records.get(synthesis_id)
            if record is None:
                raise HTTPException(status_code=404, detail="Synthesis not found")
            return record.model_copy(update={"queue_position": self._queue_position(synthesis_id)})

    def list_synthesis_records(self, limit: int = 50) -> list[SynthesisRecord]:
        with self.synthesis_lock:
            records = sorted(self.synthesis_records.values(), key=lambda item: item.created_at, reverse=True)[:limit]
            return [
                record.model_copy(update={"queue_position": self._queue_position(record.synthesis_id)})
                for record in records
            ]

    def subscribe(self, synthesis_id: str) -> asyncio.Queue[None]:
        queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        self.subscribers.setdefault(synthesis_id, set()).add(queue)
        return queue

    def unsubscribe(self, synthesis_id: str, queue: asyncio.Queue[None]) -> None:
        subscribers = self.subscribers.get(synthesis_id)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            self.subscribers.pop(synthesis_id, None)

    def _notify(self, synthesis_id: str) -> None:
        loop = self.loop
        if loop is None or loop.is_closed():
            return

        def notify() -> None:
            for queue in list(self.subscribers.get(synthesis_id, ())):
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                queue.put_nowait(None)

        loop.call_soon_threadsafe(notify)

    def _set_status(
        self,
        synthesis_id: str,
        *,
        status_value: Literal["queued", "running", "succeeded", "failed", "cancelled"],
        progress: float | None = None,
        message: str | None = None,
        error: str | None = None,
        output_path: Path | None = None,
        duration_seconds: float | None = None,
        elapsed_seconds: float | None = None,
        sample_rate: int | None = None,
    ) -> None:
        changed = False
        with self.synthesis_lock:
            record = self.synthesis_records.get(synthesis_id)
            if record is None:
                return
            if record.status == "cancelled" and status_value != "cancelled":
                return
            now = time.time()
            update: dict[str, Any] = {"status": status_value, "updated_at": now}
            if status_value == "running" and record.started_at is None:
                update["started_at"] = now
            if status_value in TERMINAL_STATUSES:
                update["finished_at"] = now
            if progress is not None:
                update["progress"] = min(1.0, max(0.0, progress))
            if message is not None:
                update["message"] = message
            if error is not None:
                update["error"] = error
            if output_path is not None:
                update["output_path"] = str(output_path)
            if duration_seconds is not None:
                update["duration_seconds"] = duration_seconds
            if elapsed_seconds is not None:
                update["elapsed_seconds"] = elapsed_seconds
            if sample_rate is not None:
                update["sample_rate"] = sample_rate
            updated = record.model_copy(update=update)
            changed = updated != record
            self.synthesis_records[synthesis_id] = updated
        if changed:
            self._notify(synthesis_id)

    def cancel_synthesis(self, synthesis_id: str) -> SynthesisRecord:
        record = self.get_synthesis(synthesis_id)
        if record.status in TERMINAL_STATUSES:
            return record
        with self.synthesis_lock:
            try:
                self.queued_ids.remove(synthesis_id)
            except ValueError:
                pass
        self._set_status(synthesis_id, status_value="cancelled", progress=1.0, message="cancelled")
        return self.get_synthesis(synthesis_id)

    async def _worker_loop(self) -> None:
        if self.queue is None:
            return
        while True:
            synthesis_id = await self.queue.get()
            try:
                with self.synthesis_lock:
                    try:
                        self.queued_ids.remove(synthesis_id)
                    except ValueError:
                        pass
                    record = self.synthesis_records.get(synthesis_id)
                    request = self.requests.get(synthesis_id)
                if record is None or request is None or record.status == "cancelled":
                    continue
                await self._run_synthesis(synthesis_id, request)
            finally:
                self.queue.task_done()

    async def _run_synthesis(self, synthesis_id: str, request: ResolvedSynthesis) -> None:
        started = time.perf_counter()
        output_path = self.settings.output_dir / "results" / f"{synthesis_id}.wav"
        self.active_synthesis_id = synthesis_id
        self.active_started_at = time.time()
        self._set_status(synthesis_id, status_value="running", progress=0.01, message="loading model...")
        try:
            model = await self.ensure_loaded()
            sample_rate = int(model.tts_model.sample_rate)
            final_text = _compose_voxcpm_text(request.text, request.instruction)
            self._set_status(
                synthesis_id, status_value="running", progress=0.05, message="waiting for inference slot..."
            )
            async with self.infer_lock:
                self._set_status(synthesis_id, status_value="running", progress=0.1, message="generating audio...")
                wav = await asyncio.to_thread(
                    model.generate,
                    text=final_text,
                    prompt_wav_path=request.prompt_audio_path,
                    prompt_text=request.prompt_text,
                    reference_wav_path=request.reference_audio_path,
                    cfg_value=request.generation.cfg_value,
                    inference_timesteps=request.generation.inference_timesteps,
                    min_len=request.generation.min_len,
                    max_len=request.generation.max_len,
                    normalize=request.generation.normalize_text,
                    denoise=request.generation.denoise_reference_audio
                    and (request.prompt_audio_path is not None or request.reference_audio_path is not None),
                    retry_badcase=request.generation.retry_badcase,
                    retry_badcase_max_times=request.generation.retry_badcase_max_times,
                    retry_badcase_ratio_threshold=request.generation.retry_badcase_ratio_threshold,
                )
            if self.get_synthesis(synthesis_id).status == "cancelled":
                output_path.unlink(missing_ok=True)
                return
            self._set_status(synthesis_id, status_value="running", progress=0.95, message="saving audio...")
            sf.write(str(output_path), np.asarray(wav), sample_rate)
            elapsed = time.perf_counter() - started
            self._set_status(
                synthesis_id,
                status_value="succeeded",
                progress=1.0,
                message="completed",
                output_path=output_path,
                duration_seconds=get_wav_duration(output_path),
                elapsed_seconds=elapsed,
                sample_rate=sample_rate,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            if self.get_synthesis(synthesis_id).status == "cancelled":
                return
            self._set_status(
                synthesis_id,
                status_value="failed",
                progress=1.0,
                message="failed",
                error=str(exc),
                elapsed_seconds=elapsed,
            )
        finally:
            self.active_synthesis_id = None
            self.active_started_at = None

    def lora_status(self) -> LoRAStatus:
        enabled = None
        configured = bool(self.settings.lora_weights_path)
        if self.model is not None:
            configured = bool(self.model.lora_enabled)
            if configured and hasattr(self.model.tts_model, "lora_enabled"):
                enabled = bool(self.model.tts_model.lora_enabled)
        return LoRAStatus(
            configured=configured,
            enabled=enabled,
            weights_path=self.settings.lora_weights_path,
            enable_lm=self.settings.lora_enable_lm,
            enable_dit=self.settings.lora_enable_dit,
            enable_proj=self.settings.lora_enable_proj,
            r=self.settings.lora_r,
            alpha=self.settings.lora_alpha,
            dropout=self.settings.lora_dropout,
        )

    def set_lora_enabled(self, enabled: bool) -> LoRAStatus:
        if self.model is None:
            raise HTTPException(status_code=409, detail="Model is not loaded")
        if not self.model.lora_enabled:
            raise HTTPException(status_code=409, detail="LoRA is not configured for this model instance")
        self.model.set_lora_enabled(enabled)
        return self.lora_status()


def get_wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except wave.Error:
        return None


def create_auth_dependency(settings: ServerSettings):
    async def require_api_key(request: Request) -> None:
        if not settings.api_key:
            return
        authorization = request.headers.get("authorization", "")
        token = request.headers.get("x-api-key", "")
        if authorization.lower().startswith("bearer "):
            token = authorization.split(" ", 1)[1].strip()
        if not hmac.compare_digest(token, settings.api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    return require_api_key


def synthesis_response(request: Request, record: SynthesisRecord) -> SynthesisRecord:
    return record.model_copy(
        update={"audio_url": str(request.url_for("get_synthesis_audio", synthesis_id=record.synthesis_id))}
    )


def submit_response(request: Request, record: SynthesisRecord) -> SynthesisSubmitResponse:
    return SynthesisSubmitResponse(
        synthesis_id=record.synthesis_id,
        status=record.status,
        progress=record.progress,
        message=record.message,
        synthesis_url=str(request.url_for("get_synthesis", synthesis_id=record.synthesis_id)),
        events_url=str(request.url_for("get_synthesis_events", synthesis_id=record.synthesis_id)),
        audio_url=str(request.url_for("get_synthesis_audio", synthesis_id=record.synthesis_id)),
        queue_position=record.queue_position,
    )


def sse_message(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _next_chunk(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


async def resolve_upload_reference(
    service: VoxCPMService, upload: UploadFile | None, reference_id: str | None
) -> tuple[Path | None, str | None]:
    if reference_id:
        return service.resolve_reference_path(reference_id), reference_id
    if upload is not None:
        asset = await service.create_reference_from_upload(upload)
        return service.resolve_reference_path(asset.reference_id), asset.reference_id
    return None, None


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or ServerSettings()
    service = VoxCPMService(settings)
    require_api_key = create_auth_dependency(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service
        await service.start()
        if not settings.lazy_load:
            await service.ensure_loaded()
        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(
        title="VoxCPM2 API Server",
        version="0.2.0",
        description="VoxCPM-native FastAPI service for synthesis, voice design, cloning, and streaming.",
        lifespan=lifespan,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "name": "VoxCPM2 API Server",
            "docs_url": "/docs",
            "health_url": "/health",
            "runtime_url": "/v1/runtime",
            "model_url": "/v1/model",
        }

    @app.get("/health")
    async def health() -> JSONResponse:
        payload = {"ok": service.load_error is None, "model_loaded": service.loaded, "load_error": service.load_error}
        return JSONResponse(payload, status_code=200 if service.load_error is None else 503)

    @app.get("/v1/runtime", dependencies=[Depends(require_api_key)])
    async def runtime() -> dict[str, Any]:
        queue_size = service.queue.qsize() if service.queue is not None else 0
        running = sum(1 for item in service.synthesis_records.values() if item.status == "running")
        queued = sum(1 for item in service.synthesis_records.values() if item.status == "queued")
        payload: dict[str, Any] = {
            "model_loaded": service.loaded,
            "model": settings.model,
            "configured_device": settings.device,
            "load_started_at": service.load_started_at,
            "loaded_at": service.loaded_at,
            "load_error": service.load_error,
            "queue_size": queue_size,
            "queued_synthesis_jobs": queued,
            "running_synthesis_jobs": running,
            "active_synthesis_id": service.active_synthesis_id,
            "active_started_at": service.active_started_at,
            "active_streams": service.active_streams,
            "max_queue_size": settings.max_queue_size,
            "worker_concurrency": 1,
            "references": len(service.references),
            "synthesis_count": len(service.synthesis_records),
        }
        try:
            import torch

            payload["torch"] = {
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            }
            if torch.cuda.is_available():
                payload["cuda_devices"] = [
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "memory_allocated": torch.cuda.memory_allocated(index),
                        "memory_reserved": torch.cuda.memory_reserved(index),
                    }
                    for index in range(torch.cuda.device_count())
                ]
        except Exception as exc:
            payload["torch_error"] = str(exc)
        return payload

    @app.post("/v1/runtime/load", dependencies=[Depends(require_api_key)])
    async def load_runtime() -> dict[str, Any]:
        await service.ensure_loaded()
        return {"model_loaded": True, "loaded_at": service.loaded_at}

    @app.post("/v1/runtime/unload", dependencies=[Depends(require_api_key)])
    async def unload_runtime() -> dict[str, Any]:
        service.unload()
        return {"model_loaded": False}

    @app.get("/v1/model", dependencies=[Depends(require_api_key)])
    async def model() -> dict[str, Any]:
        sample_rate = int(service.model.tts_model.sample_rate) if service.model is not None else None
        return {
            "model": settings.model,
            "model_loaded": service.loaded,
            "device": settings.device,
            "sample_rate": sample_rate,
            "outputs_dir": str(settings.output_dir),
            "allow_local_paths": settings.allow_local_paths,
            "default_generation": {
                "cfg_value": settings.default_cfg_value,
                "inference_timesteps": settings.default_inference_timesteps,
            },
            "voice_modes": sorted(VOICE_MODES),
            "capabilities": {
                "plain_tts": True,
                "voice_design": True,
                "reference_clone": True,
                "prompt_continuation": True,
                "hybrid_clone": True,
                "streaming": True,
                "reference_registry": True,
                "lora": service.lora_status().configured,
            },
            "lora": service.lora_status().model_dump(),
        }

    @app.get("/v1/model/lora", response_model=LoRAStatus, dependencies=[Depends(require_api_key)])
    async def get_lora_status() -> LoRAStatus:
        return service.lora_status()

    @app.post("/v1/model/lora/enable", response_model=LoRAStatus, dependencies=[Depends(require_api_key)])
    async def enable_lora() -> LoRAStatus:
        await service.ensure_loaded()
        return service.set_lora_enabled(True)

    @app.post("/v1/model/lora/disable", response_model=LoRAStatus, dependencies=[Depends(require_api_key)])
    async def disable_lora() -> LoRAStatus:
        await service.ensure_loaded()
        return service.set_lora_enabled(False)

    @app.post("/v1/references", response_model=ReferenceAsset, dependencies=[Depends(require_api_key)])
    async def create_reference(audio: UploadFile = File(...)) -> ReferenceAsset:
        return await service.create_reference_from_upload(audio)

    @app.post("/v1/references/json", response_model=ReferenceAsset, dependencies=[Depends(require_api_key)])
    async def create_reference_json(body: ReferenceJsonRequest) -> ReferenceAsset:
        return service.create_reference_from_base64(body.audio_base64, body.filename, body.content_type)

    @app.get("/v1/references", response_model=list[ReferenceAsset], dependencies=[Depends(require_api_key)])
    async def list_references() -> list[ReferenceAsset]:
        return service.list_references()

    @app.get("/v1/references/{reference_id}", response_model=ReferenceAsset, dependencies=[Depends(require_api_key)])
    async def get_reference(reference_id: str) -> ReferenceAsset:
        return service.get_reference(reference_id)

    @app.delete("/v1/references/{reference_id}", response_model=ReferenceAsset, dependencies=[Depends(require_api_key)])
    async def delete_reference(reference_id: str) -> ReferenceAsset:
        return service.delete_reference(reference_id)

    @app.post(
        "/v1/synthesis/jobs",
        response_model=SynthesisSubmitResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_key)],
    )
    async def create_synthesis_job(request: Request, body: SynthesisRequest) -> SynthesisSubmitResponse:
        resolved = service.resolve_synthesis(body)
        record = await service.enqueue(resolved)
        return submit_response(request, record)

    @app.post(
        "/v1/synthesis/jobs/multipart",
        response_model=SynthesisSubmitResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_key)],
    )
    async def create_synthesis_job_multipart(
        request: Request,
        text: str = Form(...),
        voice_mode: Literal["auto", "plain", "design", "clone", "continuation", "hybrid"] = Form("auto"),
        instruction: str | None = Form(None),
        reference_id: str | None = Form(None),
        reference_audio: UploadFile | None = File(None),
        reference_audio_path: str | None = Form(None),
        prompt_id: str | None = Form(None),
        prompt_audio: UploadFile | None = File(None),
        prompt_audio_path: str | None = Form(None),
        prompt_text: str | None = Form(None),
        use_prompt_as_reference: bool = Form(True),
        cfg_value: float = Form(settings.default_cfg_value),
        inference_timesteps: int = Form(settings.default_inference_timesteps),
        min_len: int = Form(2),
        max_len: int = Form(4096),
        normalize_text: bool = Form(True),
        denoise_reference_audio: bool = Form(False),
        retry_badcase: bool = Form(True),
        retry_badcase_max_times: int = Form(3),
        retry_badcase_ratio_threshold: float = Form(6.0),
    ) -> SynthesisSubmitResponse:
        ref_path, ref_id = await resolve_upload_reference(service, reference_audio, reference_id)
        prompt_path, resolved_prompt_id = await resolve_upload_reference(service, prompt_audio, prompt_id)
        voice = SynthesisVoice(
            mode=voice_mode,
            instruction=instruction,
            reference_id=ref_id,
            reference_audio_path=str(ref_path) if ref_path else reference_audio_path,
            prompt_id=resolved_prompt_id,
            prompt_audio_path=str(prompt_path) if prompt_path else prompt_audio_path,
            prompt_text=prompt_text,
            use_prompt_as_reference=use_prompt_as_reference,
        )
        body = SynthesisRequest(
            text=text,
            voice=voice,
            generation=GenerationSettings(
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                min_len=min_len,
                max_len=max_len,
                normalize_text=normalize_text,
                denoise_reference_audio=denoise_reference_audio,
                retry_badcase=retry_badcase,
                retry_badcase_max_times=retry_badcase_max_times,
                retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
            ),
        )
        resolved = service.resolve_synthesis(body)
        record = await service.enqueue(resolved)
        return submit_response(request, record)

    @app.get("/v1/synthesis/jobs", response_model=list[SynthesisRecord], dependencies=[Depends(require_api_key)])
    async def list_synthesis_jobs(request: Request, limit: int = 50) -> list[SynthesisRecord]:
        return [synthesis_response(request, item) for item in service.list_synthesis_records(max(1, min(limit, 200)))]

    @app.get(
        "/v1/synthesis/jobs/{synthesis_id}",
        response_model=SynthesisRecord,
        name="get_synthesis",
        dependencies=[Depends(require_api_key)],
    )
    async def get_synthesis(request: Request, synthesis_id: str) -> SynthesisRecord:
        return synthesis_response(request, service.get_synthesis(synthesis_id))

    @app.post(
        "/v1/synthesis/jobs/{synthesis_id}/cancel",
        response_model=SynthesisRecord,
        dependencies=[Depends(require_api_key)],
    )
    async def cancel_synthesis(request: Request, synthesis_id: str) -> SynthesisRecord:
        return synthesis_response(request, service.cancel_synthesis(synthesis_id))

    @app.get(
        "/v1/synthesis/jobs/{synthesis_id}/events",
        name="get_synthesis_events",
        dependencies=[Depends(require_api_key)],
    )
    async def get_synthesis_events(request: Request, synthesis_id: str) -> StreamingResponse:
        service.get_synthesis(synthesis_id)

        async def event_stream():
            updates = service.subscribe(synthesis_id)
            last_payload = None
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    record = synthesis_response(request, service.get_synthesis(synthesis_id))
                    payload = record.model_dump()
                    if payload != last_payload:
                        yield sse_message("synthesis", payload)
                        last_payload = payload
                    if record.status in TERMINAL_STATUSES:
                        yield sse_message("done", payload)
                        break
                    try:
                        await asyncio.wait_for(updates.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                service.unsubscribe(synthesis_id, updates)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get(
        "/v1/synthesis/jobs/{synthesis_id}/audio",
        name="get_synthesis_audio",
        dependencies=[Depends(require_api_key)],
    )
    async def get_synthesis_audio(synthesis_id: str) -> FileResponse:
        record = service.get_synthesis(synthesis_id)
        if record.status != "succeeded" or not record.output_path:
            raise HTTPException(status_code=409, detail=f"Synthesis is not ready: {record.status}")
        output_path = Path(record.output_path)
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Synthesis audio file not found")
        return FileResponse(output_path, media_type="audio/wav", filename=f"{record.synthesis_id}.wav")

    @app.delete("/v1/synthesis/jobs/{synthesis_id}", dependencies=[Depends(require_api_key)])
    async def delete_synthesis(synthesis_id: str) -> dict[str, Any]:
        record = service.get_synthesis(synthesis_id)
        if record.status == "running":
            raise HTTPException(status_code=409, detail="Running synthesis cannot be deleted; cancel it first")
        if record.status == "queued":
            service.cancel_synthesis(synthesis_id)
        with service.synthesis_lock:
            service.synthesis_records.pop(synthesis_id, None)
            service.requests.pop(synthesis_id, None)
        if record.output_path:
            Path(record.output_path).unlink(missing_ok=True)
        return {"deleted": True, "synthesis_id": synthesis_id}

    @app.post("/v1/synthesis/stream/events", dependencies=[Depends(require_api_key)])
    async def stream_synthesis_events(body: StreamSynthesisRequest) -> StreamingResponse:
        resolved = service.resolve_synthesis(body)

        async def event_stream():
            iterator = None
            service.active_streams += 1
            try:
                model = await service.ensure_loaded()
                final_text = _compose_voxcpm_text(resolved.text, resolved.instruction)
                sample_rate = int(model.tts_model.sample_rate)
                yield sse_message(
                    "metadata",
                    {
                        "sample_rate": sample_rate,
                        "channels": 1,
                        "audio_format": "f32le",
                        "chunk_encoding": body.chunk_encoding,
                        "mode": resolved.mode,
                    },
                )
                async with service.infer_lock:
                    iterator = model.generate_streaming(
                        text=final_text,
                        prompt_wav_path=resolved.prompt_audio_path,
                        prompt_text=resolved.prompt_text,
                        reference_wav_path=resolved.reference_audio_path,
                        cfg_value=resolved.generation.cfg_value,
                        inference_timesteps=resolved.generation.inference_timesteps,
                        min_len=resolved.generation.min_len,
                        max_len=resolved.generation.max_len,
                        normalize=resolved.generation.normalize_text,
                        denoise=resolved.generation.denoise_reference_audio
                        and (resolved.prompt_audio_path is not None or resolved.reference_audio_path is not None),
                        retry_badcase=False,
                        retry_badcase_max_times=resolved.generation.retry_badcase_max_times,
                        retry_badcase_ratio_threshold=resolved.generation.retry_badcase_ratio_threshold,
                    )
                    index = 0
                    while True:
                        chunk = await asyncio.to_thread(_next_chunk, iterator)
                        if chunk is None:
                            break
                        audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
                        yield sse_message(
                            "chunk",
                            {
                                "index": index,
                                "samples": int(audio.shape[0]),
                                "audio_base64": base64.b64encode(audio.tobytes()).decode("ascii"),
                            },
                        )
                        index += 1
                yield sse_message("done", {"chunks": index})
            except Exception as exc:
                yield sse_message("error", {"error": str(exc)})
            finally:
                if iterator is not None and hasattr(iterator, "close"):
                    iterator.close()
                service.active_streams -= 1

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VoxCPM2 FastAPI server")
    parser.add_argument("--host", default=os.getenv("VOXCPM_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=_env_int("VOXCPM_PORT", 7862))
    parser.add_argument("--model", default=os.getenv("VOXCPM_MODEL", DEFAULT_HF_MODEL_ID))
    parser.add_argument("--output_dir", default=os.getenv("VOXCPM_OUTPUT_DIR", "outputs/api"))
    parser.add_argument("--api_key", default=os.getenv("VOXCPM_API_KEY"))
    parser.add_argument("--cors_origins", default=os.getenv("VOXCPM_CORS_ORIGINS", ""))
    parser.add_argument("--device", default=os.getenv("VOXCPM_DEVICE", "auto"))
    parser.add_argument("--cache_dir", default=os.getenv("VOXCPM_CACHE_DIR"))
    parser.add_argument("--local_files_only", action="store_true", default=_env_bool("VOXCPM_LOCAL_FILES_ONLY", False))
    parser.add_argument(
        "--lazy_load", dest="lazy_load", action="store_true", default=_env_bool("VOXCPM_LAZY_LOAD", True)
    )
    parser.add_argument("--eager_load", dest="lazy_load", action="store_false")
    parser.add_argument(
        "--no_optimize", dest="optimize", action="store_false", default=_env_bool("VOXCPM_OPTIMIZE", True)
    )
    parser.add_argument(
        "--no_denoiser", dest="load_denoiser", action="store_false", default=_env_bool("VOXCPM_LOAD_DENOISER", True)
    )
    parser.add_argument("--zipenhancer_model", default=os.getenv("VOXCPM_ZIPENHANCER_MODEL"))
    parser.add_argument(
        "--allow_local_paths", action="store_true", default=_env_bool("VOXCPM_ALLOW_LOCAL_PATHS", False)
    )
    parser.add_argument("--max_upload_mb", type=int, default=_env_int("VOXCPM_MAX_UPLOAD_MB", 50))
    parser.add_argument("--max_queue_size", type=int, default=_env_int("VOXCPM_MAX_QUEUE_SIZE", 100))
    parser.add_argument("--default_cfg_value", type=float, default=_env_float("VOXCPM_DEFAULT_CFG_VALUE", 2.0))
    parser.add_argument(
        "--default_inference_timesteps", type=int, default=_env_int("VOXCPM_DEFAULT_INFERENCE_TIMESTEPS", 10)
    )
    parser.add_argument("--lora_path", default=os.getenv("VOXCPM_LORA_PATH"))
    parser.add_argument("--lora_disable_lm", action="store_true")
    parser.add_argument("--lora_disable_dit", action="store_true")
    parser.add_argument("--lora_enable_proj", action="store_true", default=_env_bool("VOXCPM_LORA_ENABLE_PROJ", False))
    parser.add_argument("--lora_r", type=int, default=_env_int("VOXCPM_LORA_R", 32))
    parser.add_argument("--lora_alpha", type=int, default=_env_int("VOXCPM_LORA_ALPHA", 16))
    parser.add_argument("--lora_dropout", type=float, default=_env_float("VOXCPM_LORA_DROPOUT", 0.0))
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload for development")
    return parser


def main() -> None:
    import uvicorn

    args = build_arg_parser().parse_args()
    settings = ServerSettings(
        model=args.model,
        output_dir=Path(args.output_dir),
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        cors_origins=[origin.strip() for origin in args.cors_origins.split(",") if origin.strip()],
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        lazy_load=args.lazy_load,
        optimize=args.optimize,
        load_denoiser=args.load_denoiser,
        zipenhancer_model=args.zipenhancer_model,
        allow_local_paths=args.allow_local_paths,
        max_upload_mb=args.max_upload_mb,
        max_queue_size=args.max_queue_size,
        default_cfg_value=args.default_cfg_value,
        default_inference_timesteps=args.default_inference_timesteps,
        lora_weights_path=args.lora_path,
        lora_enable_lm=not args.lora_disable_lm,
        lora_enable_dit=not args.lora_disable_dit,
        lora_enable_proj=args.lora_enable_proj,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    if args.reload:
        os.environ["VOXCPM_MODEL"] = settings.model
        os.environ["VOXCPM_OUTPUT_DIR"] = str(settings.output_dir)
        if settings.api_key:
            os.environ["VOXCPM_API_KEY"] = settings.api_key
        os.environ["VOXCPM_CORS_ORIGINS"] = ",".join(settings.cors_origins or [])
        os.environ["VOXCPM_DEVICE"] = settings.device or "auto"
        if settings.cache_dir:
            os.environ["VOXCPM_CACHE_DIR"] = settings.cache_dir
        os.environ["VOXCPM_LOCAL_FILES_ONLY"] = str(settings.local_files_only).lower()
        os.environ["VOXCPM_LAZY_LOAD"] = str(settings.lazy_load).lower()
        os.environ["VOXCPM_OPTIMIZE"] = str(settings.optimize).lower()
        os.environ["VOXCPM_LOAD_DENOISER"] = str(settings.load_denoiser).lower()
        if settings.zipenhancer_model:
            os.environ["VOXCPM_ZIPENHANCER_MODEL"] = settings.zipenhancer_model
        os.environ["VOXCPM_ALLOW_LOCAL_PATHS"] = str(settings.allow_local_paths).lower()
        os.environ["VOXCPM_MAX_UPLOAD_MB"] = str(settings.max_upload_mb)
        os.environ["VOXCPM_MAX_QUEUE_SIZE"] = str(settings.max_queue_size)
        os.environ["VOXCPM_DEFAULT_CFG_VALUE"] = str(settings.default_cfg_value)
        os.environ["VOXCPM_DEFAULT_INFERENCE_TIMESTEPS"] = str(settings.default_inference_timesteps)
        uvicorn.run("voxcpm.api_server:create_app", factory=True, host=settings.host, port=settings.port, reload=True)
    else:
        uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
