from __future__ import annotations

import logging
from dataclasses import dataclass

from .asr_service import ASRService, QwenASRService
from .config import Settings
from .tts_service import QwenTTSService, TTSService

logger = logging.getLogger(__name__)


@dataclass
class RuntimeStatus:
    ready: bool = False
    loading: bool = True
    error: str | None = None
    gpu: str = "unknown"


class InferenceRuntime:
    def __init__(
        self,
        settings: Settings,
        asr_service: ASRService | None = None,
        tts_service: TTSService | None = None,
    ):
        self.settings = settings
        self.asr = asr_service or QwenASRService(settings)
        self.tts = tts_service or QwenTTSService(settings)
        self.status = RuntimeStatus()

    @staticmethod
    def _gpu_snapshot(label: str) -> str:
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("%s: CUDA is unavailable", label)
                return "CUDA unavailable"
            device = torch.cuda.current_device()
            name = torch.cuda.get_device_name(device)
            free, total = torch.cuda.mem_get_info(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            logger.info(
                "%s: gpu=%s free_mib=%.1f total_mib=%.1f allocated_mib=%.1f reserved_mib=%.1f",
                label,
                name,
                free / 2**20,
                total / 2**20,
                allocated / 2**20,
                reserved / 2**20,
            )
            return name
        except Exception as exc:
            logger.warning("Could not read GPU state at %s: %s", label, exc)
            return "unknown"

    def load(self) -> None:
        self.status.loading = True
        self.status.ready = False
        self.status.error = None
        try:
            self.status.gpu = self._gpu_snapshot("before speech model load")
            logger.info("Loading ASR model %s", self.settings.asr_model)
            self.asr.load()
            self._gpu_snapshot("after ASR load")
            if self.settings.tts_enabled:
                logger.info("Loading TTS model %s", self.settings.tts_model)
                self.tts.load()
                self.status.gpu = self._gpu_snapshot("after TTS load")
            else:
                logger.info("SPEECH_TTS_ENABLED=false; serving ASR only")
            self.status.ready = True
        except Exception as exc:
            logger.exception("Speech model loading failed")
            self.status.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.status.loading = False
