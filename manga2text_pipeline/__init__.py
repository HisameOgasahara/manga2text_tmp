from __future__ import annotations

import importlib.util
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image

# Load the original single-file implementation under a private module name.
# Because this package directory has the same import name as manga2text_pipeline.py,
# Python imports this package first. We then re-export the original API and override
# only process_pages with bubble-aware grouping.
_ORIGINAL_PATH = Path(__file__).resolve().parent.parent / "manga2text_pipeline.py"
_spec = importlib.util.spec_from_file_location("_manga2text_pipeline_base", _ORIGINAL_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("기존 manga2text_pipeline.py를 불러오지 못했습니다.")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

for _name in dir(_base):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_base, _name)


def _center(bbox: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _area(bbox: list[int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _center_inside(inner_bbox: list[int], outer_bbox: list[int]) -> bool:
    cx, cy = _center(inner_bbox)
    x1, y1, x2, y2 = outer_bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _normalize_dialogue(text: str) -> str:
    # PaddleOCR commonly returns one recognized line per newline. Inside one
    # speech bubble those line breaks are layout, not separate dialogue records.
    return re.sub(r"\s+", " ", text).strip()


def _raw_layout_regions(
    detector,
    image: Image.Image,
    class_thresholds: dict[int, float],
    include_sfx: bool,
    minimum_threshold: float = 0.20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detections = detector.predict(
        image,
        threshold=minimum_threshold,
        shape=(1152, 1152),
        include_source_image=False,
    )

    text_regions: list[dict[str, Any]] = []
    bubbles: list[dict[str, Any]] = []

    for box, class_id, score in zip(
        detections.xyxy,
        detections.class_id,
        detections.confidence,
    ):
        class_id = int(class_id)
        score = float(score)
        if class_id not in class_thresholds or score < class_thresholds[class_id]:
            continue

        x1, y1, x2, y2 = [int(round(value)) for value in box]
        region = {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "score": score,
            "bbox": [x1, y1, x2, y2],
        }

        if class_id == 0 or (class_id == 1 and include_sfx):
            text_regions.append(region)
        elif class_id == 2:
            bubbles.append(region)

    return text_regions, bubbles


def _group_regions_by_bubble(
    text_regions: list[dict[str, Any]],
    bubbles: list[dict[str, Any]],
    reading_direction: str,
    row_tolerance: int,
) -> list[dict[str, Any]]:
    # A text region is assigned to the smallest detected bubble that contains
    # its center. Choosing the smallest containing bubble avoids a large,
    # overlapping bubble stealing text from a tighter match.
    assignments: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(len(bubbles))
    }
    standalone: list[dict[str, Any]] = []

    for text_region in text_regions:
        candidates = [
            (index, bubble)
            for index, bubble in enumerate(bubbles)
            if _center_inside(text_region["bbox"], bubble["bbox"])
        ]

        if not candidates:
            standalone.append(text_region)
            continue

        bubble_index, _ = min(candidates, key=lambda item: _area(item[1]["bbox"]))
        assignments[bubble_index].append(text_region)

    grouped: list[dict[str, Any]] = []

    for bubble_index, bubble in enumerate(bubbles):
        members = assignments[bubble_index]
        if not members:
            continue

        members = sort_regions_reading_order(
            members,
            direction=reading_direction,
            row_tolerance=row_tolerance,
        )
        grouped.append(
            {
                "group_type": "bubble",
                "bbox": bubble["bbox"],
                "bubble_bbox": bubble["bbox"],
                "bubble_score": bubble["score"],
                "text_regions": members,
            }
        )

    for text_region in standalone:
        grouped.append(
            {
                "group_type": "text",
                "bbox": text_region["bbox"],
                "bubble_bbox": None,
                "bubble_score": None,
                "text_regions": [text_region],
            }
        )

    return sort_regions_reading_order(
        grouped,
        direction=reading_direction,
        row_tolerance=row_tolerance,
    )


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
    debug: bool = True,
    debug_samples_per_page: int = 3,
) -> list[dict[str, Any]]:
    from tqdm.auto import tqdm

    records: list[dict[str, Any]] = []
    language_counter = Counter()

    for page_number, page_path in enumerate(
        tqdm(page_paths, desc="페이지 처리"),
        start=1,
    ):
        image = Image.open(page_path).convert("RGB")

        text_regions, bubbles = _raw_layout_regions(
            detector=detector,
            image=image,
            class_thresholds=class_thresholds,
            include_sfx=include_sfx,
        )
        groups = _group_regions_by_bubble(
            text_regions=text_regions,
            bubbles=bubbles,
            reading_direction=reading_direction,
            row_tolerance=row_tolerance,
        )

        if debug:
            bubble_groups = sum(group["group_type"] == "bubble" for group in groups)
            standalone_groups = len(groups) - bubble_groups
            print(f"\n[Page {page_number:03d}] {page_path.name}")
            print(f"  text regions    : {len(text_regions)}")
            print(f"  bubbles detected: {len(bubbles)}")
            print(f"  bubble records  : {bubble_groups}")
            print(f"  standalone text : {standalone_groups}")

        page_ocr_count = 0
        page_translation_count = 0

        for group_index, group in enumerate(groups, start=1):
            pieces: list[str] = []
            text_bboxes: list[list[int]] = []
            text_scores: list[float] = []

            for text_region in group["text_regions"]:
                crop = crop_region(
                    image=image,
                    bbox=text_region["bbox"],
                    padding=crop_padding,
                )
                piece = run_ocr(
                    backend=ocr_backend,
                    model=ocr_model,
                    crop=crop,
                )
                piece = _normalize_dialogue(piece)
                if not piece:
                    continue

                pieces.append(piece)
                text_bboxes.append(text_region["bbox"])
                text_scores.append(round(text_region["score"], 4))

            if not pieces:
                continue

            # One detected bubble becomes exactly one dialogue record.
            original_text = _normalize_dialogue(" ".join(pieces))
            page_ocr_count += 1

            language = detect_language(
                text=original_text,
                detector=language_detector,
                language_to_code=language_to_code,
            )
            language_counter[language] += 1

            should_translate = enable_translation and language != "ko"
            if should_translate:
                page_translation_count += 1
                translated_text = translate_to_korean(
                    text=original_text,
                    source_language=language,
                    tokenizer=translation_tokenizer,
                    model=translation_model,
                    max_new_tokens=max_new_tokens,
                )
            else:
                translated_text = original_text

            if debug and page_ocr_count <= debug_samples_per_page:
                print(
                    f"  OCR[{group_index:02d}] type={group['group_type']} "
                    f"lines={len(pieces)} lang={language} translate={should_translate}"
                )
                print(f"    original: {original_text[:160]!r}")
                if should_translate:
                    print(f"    korean  : {translated_text[:160]!r}")

            is_bubble = group["group_type"] == "bubble"
            records.append(
                {
                    "page": page_number,
                    "page_file": page_path.name,
                    "order": group_index,
                    "bbox": group["bbox"],
                    "region_type": group["group_type"],
                    "bubble_bbox": group["bubble_bbox"],
                    "bubble_score": (
                        round(group["bubble_score"], 4)
                        if group["bubble_score"] is not None
                        else None
                    ),
                    "text_bboxes": text_bboxes,
                    "text_scores": text_scores,
                    "detector_class": "bubble" if is_bubble else group["text_regions"][0]["class_name"],
                    "detector_score": (
                        round(group["bubble_score"], 4)
                        if is_bubble
                        else round(group["text_regions"][0]["score"], 4)
                    ),
                    "ocr_backend": ocr_backend,
                    "language": language,
                    "original": original_text,
                    "korean": translated_text,
                }
            )

        if debug:
            print(f"  OCR records: {page_ocr_count}")
            print(f"  translated : {page_translation_count}")

    print("\n[Pipeline summary]")
    print(f"  pages          : {len(page_paths)}")
    print(f"  output records : {len(records)}")
    print(f"  languages      : {dict(language_counter)}")

    return records
