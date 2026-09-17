from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import manga2text_pipeline as _pipeline


SUPPORTED_SOURCE_LANGUAGES = {"ko", "ja"}
_OCR_CACHE: dict[tuple[str, str], Any] = {}


def _ensure_comic_translate_import_path() -> Path:
    """Expose only the pieces of comic-translate needed by Pororo OCR.

    comic-translate's ``modules.utils`` package executes ``textblock`` from its
    ``__init__.py``. That path eventually imports PySide6, which is a desktop UI
    dependency and is irrelevant in Colab.  We therefore register
    ``modules.utils`` as a lightweight namespace package that points at the same
    directory without executing its ``__init__.py``.
    """

    candidates: list[Path] = []

    env_path = os.environ.get("COMIC_TRANSLATE_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            Path("/content/comic-translate"),
            Path.cwd() / "comic-translate",
        ]
    )

    root: Path | None = None

    for path in candidates:
        if path.exists():
            root = path
            break

    if root is None:
        raise RuntimeError(
            "comic-translate 저장소를 찾지 못했습니다. "
            "Colab의 '저장소 가져오기' 셀을 먼저 실행하세요."
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    utils_path = root / "modules" / "utils"

    # Avoid importing modules/utils/__init__.py because it imports textblock,
    # which imports language_utils -> PySide6.  Pororo only needs individual
    # utility modules such as torch_autocast.
    if "modules.utils" not in sys.modules:
        utils_package = types.ModuleType("modules.utils")
        utils_package.__path__ = [str(utils_path)]
        utils_package.__package__ = "modules.utils"
        sys.modules["modules.utils"] = utils_package

    return root


def resolve_ocr_configuration(
    source_language: str,
    ocr_backend: str = "auto",
    reading_direction: str = "auto",
) -> dict[str, str]:
    if source_language not in SUPPORTED_SOURCE_LANGUAGES:
        raise ValueError(
            f"현재 노트북은 한국어(ko)와 일본어(ja)만 지원합니다: {source_language}"
        )

    if ocr_backend == "auto":
        resolved_backend = "pororo" if source_language == "ko" else "manga"
    else:
        resolved_backend = ocr_backend

    allowed_backends = {"pororo", "manga"}
    if resolved_backend not in allowed_backends:
        raise ValueError(
            f"현재 OCR backend는 PororoOCR/MangaOCR만 지원합니다: {resolved_backend}"
        )

    if reading_direction == "auto":
        resolved_direction = "rtl" if source_language == "ja" else "ltr"
    else:
        resolved_direction = reading_direction

    return {
        "source_language": source_language,
        "ocr_backend": resolved_backend,
        "reading_direction": resolved_direction,
    }


def load_ocr_backend(
    backend: str,
    pororo_device: str | None = None,
):
    if backend == "manga":
        cache_key = ("manga", "default")

        if cache_key not in _OCR_CACHE:
            from manga_ocr import MangaOcr

            print("[MangaOCR] loading Japanese OCR model")
            _OCR_CACHE[cache_key] = MangaOcr()

        return _OCR_CACHE[cache_key]

    if backend != "pororo":
        raise ValueError(f"지원하지 않는 OCR backend: {backend}")

    _ensure_comic_translate_import_path()

    # Import the Pororo core directly.  Do not import PororoOCREngine because
    # that wrapper pulls in PPOCR and desktop UI dependencies we do not use.
    from modules.ocr.pororo.main import PororoOcr

    requested_device = pororo_device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[Pororo OCR] CUDA를 사용할 수 없어 CPU로 fallback합니다.")
        requested_device = "cpu"

    cache_key = ("pororo", requested_device)

    if cache_key not in _OCR_CACHE:
        print(f"[Pororo OCR] PyTorch BrainOCR device={requested_device}")

        _OCR_CACHE[cache_key] = PororoOcr(
            model="brainocr",
            lang="ko",
            device=requested_device,
            use_text_lines=False,
        )

    return _OCR_CACHE[cache_key]


def run_pororo_ocr(model, crop: Image.Image) -> str:
    image_array = np.array(crop.convert("RGB"))

    model.run_ocr(image_array)
    result = model.get_ocr_result()

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

    if backend == "manga":
        return model(crop).strip()

    raise RuntimeError(f"알 수 없는 OCR backend: {backend}")


def auto_detect_source_language(
    page_paths: list[Path],
    detector,
    class_thresholds: dict[int, float],
    crop_padding: int = 8,
    max_crops: int = 3,
    pororo_device: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Choose Korean vs Japanese using the OCR backends we actually use."""

    print("[Auto language] 한국어/일본어 두 후보만 비교합니다.")

    crops = _pipeline._base._sample_text_regions(
        page_paths=page_paths,
        detector=detector,
        class_thresholds=class_thresholds,
        crop_padding=crop_padding,
        max_crops=max_crops,
    )

    if not crops:
        raise RuntimeError("자동 언어 판별용 텍스트 영역을 찾지 못했습니다.")

    candidates = {
        "ko": "pororo",
        "ja": "manga",
    }
    results: dict[str, Any] = {}

    for language, backend in candidates.items():
        print(f"[Auto language] testing {language} / {backend}")

        model = load_ocr_backend(
            backend=backend,
            pororo_device=pororo_device,
        )

        texts: list[str] = []

        for crop in crops:
            text = run_ocr(
                backend=backend,
                model=model,
                crop=crop,
            ).strip()

            if text:
                texts.append(text)

        joined = "\n".join(texts)
        score = _pipeline._base._script_score(joined, language)

        results[language] = {
            "backend": backend,
            "score": score,
            "sample": joined[:240],
        }

        print(f"  score={score:.3f} | sample={joined[:100]!r}")

    selected_language = max(
        results,
        key=lambda language: results[language]["score"],
    )

    print(f"[Auto language] selected: {selected_language}")

    return selected_language, results


def install_pororo_overrides() -> None:
    """Patch package globals used by process_pages at runtime."""

    _pipeline.load_ocr_backend = load_ocr_backend
    _pipeline.run_ocr = run_ocr
    _pipeline.resolve_ocr_configuration = resolve_ocr_configuration
