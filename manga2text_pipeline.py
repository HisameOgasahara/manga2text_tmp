from __future__ import annotations

import importlib.util
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


CLASS_NAMES = {
    0: "text",
    1: "onomatopoeia",
    2: "bubble",
    3: "panel",
}

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
KANA_RE = re.compile(r"[\u3040-\u30ff]")


# -----------------------------------------------------------------------------
# 1. Input inspection / preview
# -----------------------------------------------------------------------------

def classify_inputs(input_dir: Path) -> dict[str, list[Path]]:
    """Classify uploaded files into images, PDFs, and unsupported files."""
    groups = {
        "images": [],
        "pdfs": [],
        "unsupported": [],
    }

    for path in sorted(input_dir.iterdir()):
        suffix = path.suffix.lower()

        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            groups["images"].append(path)
        elif suffix == ".pdf":
            groups["pdfs"].append(path)
        else:
            groups["unsupported"].append(path)

    return groups


def describe_input_mode(groups: dict[str, list[Path]]) -> str:
    image_count = len(groups["images"])
    pdf_count = len(groups["pdfs"])

    if image_count == 1 and pdf_count == 0:
        return "단일 이미지"

    if image_count > 1 and pdf_count == 0:
        return "여러 이미지"

    if image_count == 0 and pdf_count == 1:
        return "PDF"

    if image_count == 0 and pdf_count > 1:
        return "여러 PDF"

    if image_count > 0 and pdf_count > 0:
        return "이미지 + PDF 혼합"

    return "지원되는 입력 없음"


def make_preview_images(
    input_dir: Path,
    max_items: int = 8,
    pdf_preview_pages: int = 3,
    pdf_dpi: int = 90,
) -> list[tuple[str, Image.Image]]:
    """Create lightweight thumbnails for uploaded images and PDFs."""
    import fitz

    groups = classify_inputs(input_dir)
    previews: list[tuple[str, Image.Image]] = []

    for image_path in groups["images"]:
        if len(previews) >= max_items:
            break

        with Image.open(image_path) as image:
            previews.append(
                (
                    image_path.name,
                    image.convert("RGB").copy(),
                )
            )

    for pdf_path in groups["pdfs"]:
        if len(previews) >= max_items:
            break

        document = fitz.open(pdf_path)
        preview_count = min(
            len(document),
            pdf_preview_pages,
            max_items - len(previews),
        )

        zoom = pdf_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_index in range(preview_count):
            page = document[page_index]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)

            image = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )

            label = f"{pdf_path.name} / p.{page_index + 1}"
            previews.append((label, image))

        document.close()

    return previews


# -----------------------------------------------------------------------------
# 2. PDF / image preparation
# -----------------------------------------------------------------------------

def _render_pdf_page(
    pdf_path: Path,
    output_dir: Path,
    page_index: int,
    dpi: int,
) -> Path:
    """Render one PDF page. Each worker opens its own document for thread safety."""
    import fitz

    document = fitz.open(pdf_path)
    page = document[page_index]

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    image = Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )

    output_path = output_dir / (
        f"{pdf_path.stem}_page_{page_index + 1:04d}.png"
    )
    image.save(output_path)

    document.close()
    return output_path


def pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 200,
    page_limit: int | None = None,
    workers: int = 4,
) -> list[Path]:
    """Render PDF pages in parallel using independent PyMuPDF documents."""
    import fitz

    output_dir.mkdir(parents=True, exist_ok=True)

    document = fitz.open(pdf_path)
    page_count = len(document)
    document.close()

    if page_limit is not None:
        page_count = min(page_count, page_limit)

    page_indices = list(range(page_count))

    if workers <= 1:
        return [
            _render_pdf_page(
                pdf_path=pdf_path,
                output_dir=output_dir,
                page_index=page_index,
                dpi=dpi,
            )
            for page_index in page_indices
        ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _render_pdf_page,
                pdf_path,
                output_dir,
                page_index,
                dpi,
            )
            for page_index in page_indices
        ]

        output_paths = [
            future.result()
            for future in futures
        ]

    return sorted(output_paths)


def _copy_image(input_path: Path, page_dir: Path) -> Path:
    destination = page_dir / input_path.name
    shutil.copy2(input_path, destination)
    return destination


def collect_page_images(
    input_dir: Path,
    page_dir: Path,
    pdf_dpi: int = 200,
    page_limit: int | None = None,
    workers: int = 4,
) -> list[Path]:
    """Prepare images from mixed image/PDF uploads."""
    page_dir.mkdir(parents=True, exist_ok=True)
    groups = classify_inputs(input_dir)

    page_paths: list[Path] = []

    image_paths = groups["images"]
    if image_paths:
        if workers <= 1:
            copied = [
                _copy_image(path, page_dir)
                for path in image_paths
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                copied = list(
                    executor.map(
                        lambda path: _copy_image(path, page_dir),
                        image_paths,
                    )
                )

        page_paths.extend(copied)

    for pdf_path in groups["pdfs"]:
        pdf_output_dir = page_dir / pdf_path.stem

        converted = pdf_to_images(
            pdf_path=pdf_path,
            output_dir=pdf_output_dir,
            dpi=pdf_dpi,
            page_limit=page_limit,
            workers=workers,
        )

        page_paths.extend(converted)

    return sorted(page_paths, key=lambda path: str(path).lower())


# -----------------------------------------------------------------------------
# 3. Koharu RF-DETR detector
# -----------------------------------------------------------------------------

def load_koharu_detector(
    repo_id: str = "mayocream/koharu-layout-rfdetr-seg-2xl-1152",
):
    from huggingface_hub import hf_hub_download

    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
    )

    loader_path = hf_hub_download(
        repo_id=repo_id,
        filename="load_model.py",
    )

    spec = importlib.util.spec_from_file_location(
        "koharu_layout_loader",
        loader_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Koharu RF-DETR loader를 불러오지 못했습니다.")

    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)

    detector = loader.load_model(weights_path)
    return detector


def detect_regions(
    detector,
    image: Image.Image,
    class_thresholds: dict[int, float],
    include_sfx: bool = False,
    minimum_threshold: float = 0.20,
) -> list[dict[str, Any]]:
    detections = detector.predict(
        image,
        threshold=minimum_threshold,
        shape=(1152, 1152),
        include_source_image=False,
    )

    regions: list[dict[str, Any]] = []

    for box, class_id, score in zip(
        detections.xyxy,
        detections.class_id,
        detections.confidence,
    ):
        class_id = int(class_id)
        score = float(score)

        required_score = class_thresholds[class_id]
        if score < required_score:
            continue

        if class_id == 0:
            pass
        elif class_id == 1 and include_sfx:
            pass
        else:
            continue

        x1, y1, x2, y2 = [
            int(round(value))
            for value in box
        ]

        regions.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "score": score,
                "bbox": [x1, y1, x2, y2],
            }
        )

    return regions


def crop_region(
    image: Image.Image,
    bbox: list[int],
    padding: int = 8,
) -> Image.Image:
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)

    return image.crop((x1, y1, x2, y2))


def sort_regions_reading_order(
    regions: list[dict[str, Any]],
    direction: str = "rtl",
    row_tolerance: int = 80,
) -> list[dict[str, Any]]:
    if not regions:
        return []

    def center(region: dict[str, Any]) -> tuple[float, float]:
        x1, y1, x2, y2 = region["bbox"]
        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
        )

    remaining = list(regions)
    ordered: list[dict[str, Any]] = []

    while remaining:
        remaining.sort(
            key=lambda region: center(region)[1]
        )

        anchor = remaining[0]
        _, anchor_y = center(anchor)

        same_row = [
            region
            for region in remaining
            if abs(center(region)[1] - anchor_y) <= row_tolerance
        ]

        same_row.sort(
            key=lambda region: center(region)[0],
            reverse=(direction == "rtl"),
        )

        ordered.extend(same_row)

        same_row_ids = {
            id(region)
            for region in same_row
        }

        remaining = [
            region
            for region in remaining
            if id(region) not in same_row_ids
        ]

    return ordered


# -----------------------------------------------------------------------------
# 4. OCR backends
# -----------------------------------------------------------------------------

def load_ocr_backend(
    backend: str,
    paddle_lang: str = "korean",
    paddle_device: str = "cpu",
):
    if backend == "manga":
        from manga_ocr import MangaOcr
        return MangaOcr()

    if backend == "paddle":
        from paddleocr import PaddleOCR

        return PaddleOCR(
            lang=paddle_lang,
            device=paddle_device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    raise ValueError(
        'OCR backend은 "manga" 또는 "paddle"이어야 합니다.'
    )


def run_manga_ocr(model, crop: Image.Image) -> str:
    return model(crop).strip()


def run_paddle_ocr(model, crop: Image.Image) -> str:
    image_array = np.array(crop.convert("RGB"))
    recognized_texts: list[str] = []

    for result in model.predict(image_array):
        data = result.json if hasattr(result, "json") else result

        if "res" in data:
            data = data["res"]

        texts = data.get("rec_texts", [])

        for text in texts:
            text = str(text).strip()
            if text:
                recognized_texts.append(text)

    return "\n".join(recognized_texts)


def run_ocr(
    backend: str,
    model,
    crop: Image.Image,
) -> str:
    if backend == "manga":
        return run_manga_ocr(model, crop)

    if backend == "paddle":
        return run_paddle_ocr(model, crop)

    raise RuntimeError("알 수 없는 OCR backend입니다.")


# -----------------------------------------------------------------------------
# 5. Language detection
# -----------------------------------------------------------------------------

def build_language_detector():
    from lingua import Language, LanguageDetectorBuilder

    detector = LanguageDetectorBuilder.from_languages(
        Language.KOREAN,
        Language.JAPANESE,
        Language.CHINESE,
        Language.ENGLISH,
    ).build()

    language_to_code = {
        Language.KOREAN: "ko",
        Language.JAPANESE: "ja",
        Language.CHINESE: "zh",
        Language.ENGLISH: "en",
    }

    return detector, language_to_code


def detect_language(
    text: str,
    detector,
    language_to_code: dict,
) -> str:
    text = text.strip()

    if not text:
        return "unknown"

    if HANGUL_RE.search(text):
        return "ko"

    if KANA_RE.search(text):
        return "ja"

    language = detector.detect_language_of(text)

    if language is None:
        return "unknown"

    return language_to_code.get(language, "unknown")


# -----------------------------------------------------------------------------
# 6. Small local translation LLM
# -----------------------------------------------------------------------------

def load_translation_model(
    model_name: str = "Qwen/Qwen3-1.7B",
):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )

    model.eval()
    return tokenizer, model


SOURCE_LANGUAGE_NAMES = {
    "ja": "일본어",
    "zh": "중국어",
    "en": "영어",
    "unknown": "외국어",
}


@torch.inference_mode()
def translate_to_korean(
    text: str,
    source_language: str,
    tokenizer,
    model,
    max_new_tokens: int = 256,
) -> str:
    if source_language == "ko":
        return text

    source_name = SOURCE_LANGUAGE_NAMES.get(
        source_language,
        "외국어",
    )

    messages = [
        {
            "role": "system",
            "content": (
                "너는 만화와 웹툰 대사 번역기다. "
                "원문의 의미, 인물 말투, 존댓말/반말, 감정 표현을 최대한 유지한다. "
                "설명이나 주석을 붙이지 말고 번역문만 출력한다."
            ),
        },
        {
            "role": "user",
            "content": (
                f"다음 {source_name} 만화 대사를 자연스러운 한국어로 번역해.\n\n"
                f"{text}"
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    new_tokens = generated[
        0,
        inputs["input_ids"].shape[1]:,
    ]

    translated = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    return translated.strip()


# -----------------------------------------------------------------------------
# 7. Full page pipeline
# -----------------------------------------------------------------------------

def process_pages(
    page_paths: list[Path],
    detector,
    ocr_backend: str,
    ocr_model,
    language_detector,
    language_to_code: dict,
    class_thresholds: dict[int, float],
    reading_direction: str,
    row_tolerance: int,
    crop_padding: int,
    include_sfx: bool,
    enable_translation: bool,
    translation_tokenizer=None,
    translation_model=None,
    max_new_tokens: int = 256,
) -> list[dict[str, Any]]:
    from tqdm.auto import tqdm

    records: list[dict[str, Any]] = []

    for page_number, page_path in enumerate(
        tqdm(page_paths, desc="페이지 처리"),
        start=1,
    ):
        image = Image.open(page_path).convert("RGB")

        regions = detect_regions(
            detector=detector,
            image=image,
            class_thresholds=class_thresholds,
            include_sfx=include_sfx,
        )

        regions = sort_regions_reading_order(
            regions=regions,
            direction=reading_direction,
            row_tolerance=row_tolerance,
        )

        for region_index, region in enumerate(
            regions,
            start=1,
        ):
            crop = crop_region(
                image=image,
                bbox=region["bbox"],
                padding=crop_padding,
            )

            original_text = run_ocr(
                backend=ocr_backend,
                model=ocr_model,
                crop=crop,
            )

            if not original_text:
                continue

            language = detect_language(
                text=original_text,
                detector=language_detector,
                language_to_code=language_to_code,
            )

            if enable_translation and language != "ko":
                translated_text = translate_to_korean(
                    text=original_text,
                    source_language=language,
                    tokenizer=translation_tokenizer,
                    model=translation_model,
                    max_new_tokens=max_new_tokens,
                )
            else:
                translated_text = original_text

            records.append(
                {
                    "page": page_number,
                    "page_file": page_path.name,
                    "order": region_index,
                    "bbox": region["bbox"],
                    "detector_class": region["class_name"],
                    "detector_score": round(region["score"], 4),
                    "ocr_backend": ocr_backend,
                    "language": language,
                    "original": original_text,
                    "korean": translated_text,
                }
            )

    return records


# -----------------------------------------------------------------------------
# 8. Save results
# -----------------------------------------------------------------------------

def save_results(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "dialogues.jsonl"
    txt_path = output_dir / "dialogues.txt"

    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with txt_path.open("w", encoding="utf-8") as file:
        current_page = None

        for record in records:
            if record["page"] != current_page:
                current_page = record["page"]
                file.write(
                    f"\n=== Page {current_page} ===\n"
                )

            file.write(record["korean"] + "\n")

    return jsonl_path, txt_path
