"""악기 분리 (Demucs).

Meta의 Demucs v4(htdemucs 계열)로 곡을 악기별 트랙(stem)으로 나눕니다.

* htdemucs_6s (기본): vocals, drums, bass, guitar, piano, other  ← 신스·현악·패드 등은 대부분 other
* htdemucs     : vocals, drums, bass, other
* htdemucs_ft  : htdemucs를 악기별로 미세조정한 버전 (더 정확, 약 4배 느림)

처음 실행할 때 모델 가중치(수십 MB)를 인터넷에서 내려받습니다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Stem:
    name: str
    path: Path
    share: float  # 원곡 대비 에너지 비율 (0~1)


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def separate(audio: Path, outdir: Path, model: str = "htdemucs_6s",
             device: str | None = None, log=print) -> list[Stem]:
    """audio를 악기별 wav로 나눠 outdir에 저장하고, 각 stem 정보를 돌려줍니다."""
    try:
        from demucs.api import Separator, save_audio
    except ImportError as e:  # pragma: no cover - 안내 메시지용
        raise RuntimeError(
            "악기 분리에는 demucs가 필요합니다:  pip install -r requirements-separate.txt"
        ) from e

    import random

    import torch

    # Demucs는 무작위 시간 이동(shift)을 쓰므로, 같은 곡이면 같은 결과가 나오도록 시드 고정
    random.seed(0)
    torch.manual_seed(0)
    device = device or _pick_device()
    log(f"      Demucs 모델 '{model}' 사용 ({device})")
    sep = Separator(model=model, device=device, progress=False)
    origin, stems = sep.separate_audio_file(Path(audio))

    outdir.mkdir(parents=True, exist_ok=True)
    total = float((origin ** 2).mean()) or 1.0
    result = []
    for name, wav in stems.items():
        path = outdir / f"{name}.wav"
        save_audio(wav, str(path), samplerate=sep.samplerate)
        result.append(Stem(name, path, float((wav ** 2).mean()) / total))
    return result
