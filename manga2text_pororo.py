from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import manga2text_pipeline as _pipeline


def _ensure_comic_translate_import_path() -> Path:
    candidates = []

    env_path = os.environ.get("COMIC_TRANSLATE_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            Path("/content/comic-translate"),
            Path.cwd() / "comic-translate",
        ]
    )

    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return path

    raise RuntimeError(
        "comic-translate 저장소를 찾지 못했습니다. "
        "Colab의 '저장소 가져오기' 셀을 먼저 실행하세요."
    )


def resolve_ocr_configuration(
    source_language: str,
    ocr_backend: str = "auto",
    reading_direction: str = "auto",
) -> dict[str, str]:
    if source_language not in _pipeline.PADDLE_LANG_BY_SOURCE:
        raise ValueError(f"지원하지 않는 언어: {source_language}")

    if ocr_backend == "auto":
        if source_language == "ko":
            resolved_backend = "pororo"
        elif source_language == "ja":
            resolved_backend = "manga"
        else:
            resolved_backend = "paddle"
    else:
        resolved_backend = ocr_backend

    if reading_direction == "auto":
        resolved_direction = "rtl" if source_language == "ja" else "ltr"
    else:
        resolved_direction = reading_direction

    return {
        "source_language": source_language,
        "ocr_backend": resolved_backend,
        "paddle_lang": _pipeline.PADDLE_LANG_BY_SOURCE[source_language],
        "reading_direction": resolved_direction,
    }


def load_ocr_backend(
    backend: str,
    paddle_lang: str = "korean",
    paddle_device: str = "cpu",
    pororo_device: str | None = None,
):
    if backend != "pororo":
        return _pipeline._base.load_ocr_backend(
            backend=backend,
            paddle_lang=paddle_lang,
            paddle_device=paddle_device,
        )

    _ensure_comic_translate_import_path()

    from modules.ocr.pororo.engine import PororoOCREngine

    requested_device = pororo_device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if requested_device == "cuda" and not torch.cuda.is_available():
        print(
            "[Pororo OCR] PyTorch CUDA를 사용할 수 없습니다. "
            "CPU로 fallback합니다."
        )
        requested_device = "cpu"

    print(
        "[Pororo OCR] backend=PyTorch/BrainOCR, "
        f"device={requested_device}"
    )

    model = PororoOCREngine()
    model.initialize(
        lang="ko",
        device=requested_device,
        use_text_lines=False,
    )
    return model


def run_pororo_ocr(model, crop: Image.Image) -> str:
    image_array = np.array(crop.convert("RGB"))

    # PororoOCREngine wraps PororoOcr as `model`.
    # Koharu already supplies the text crop, so we only need Pororo's
    # recognition result for this crop; Pororo may internally refine text lines.
    model.model.run_ocr(image_array)
    result = model.model.get_ocr_result()

    texts: list[str] = []
    for text in result.get("description", []):
        text = str(text).strip()
        if text:
            texts.append(text)

    return " ".join(texts).strip()


def run_ocr(
    backend: str,
    model,
    crop: Image.Image,
) -> str:
    if backend == "pororo":
        return run_pororo_ocr(model, crop)

    return _pipeline._base.run_ocr(
        backend=backend,
        model=model,
        crop=crop,
    )


def install_pororo_overrides() -> None:
    """Patch package globals used by process_pages at runtime."""
    _pipeline.load_ocr_backend = load_ocr_backend
    _pipeline.run_ocr = run_ocr
    _pipeline.resolve_ocr_configuration = resolve_ocr_configuration
