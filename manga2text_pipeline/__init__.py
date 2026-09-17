from __future__ import annotations

import importlib.util
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Load the original single-file implementation under a private module name.
_ORIGINAL_PATH = Path(__file__).resolve().parent.parent / "manga2text_pipeline.py"
_spec = importlib.util.spec_from_file_location("_manga2text_pipeline_base", _ORIGINAL_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError("기존 manga2text_pipeline.py를 불러오지 못했습니다.")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

for _name in dir(_base):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_base, _name)


# -----------------------------------------------------------------------------
# Reading-order configuration
# -----------------------------------------------------------------------------

READING_ORDER_PRESETS = {
    "한국/영문 페이지형": {
        "panel_direction": "ltr",
        "bubble_direction": "ltr",
        "vertical_priority": False,
    },
    "일본 원서": {
        "panel_direction": "rtl",
        "bubble_direction": "rtl",
        "vertical_priority": False,
    },
    "세로 웹툰": {
        "panel_direction": "ltr",
        "bubble_direction": "ltr",
        "vertical_priority": True,
    },
    "일본 만화 번역본(LTR 대사)": {
        "panel_direction": "rtl",
        "bubble_direction": "ltr",
        "vertical_priority": False,
    },
}

_READING_ORDER_CONFIG = {
    "mode": "legacy",
    "preset": None,
    "panel_direction": None,
    "bubble_direction": None,
    "vertical_priority": False,
}


def set_reading_order_preset(preset: str) -> dict[str, Any]:
    if preset not in READING_ORDER_PRESETS:
        raise ValueError(
            f"알 수 없는 읽기 순서 프리셋: {preset}. "
            f"가능한 값: {list(READING_ORDER_PRESETS)}"
        )
    values = READING_ORDER_PRESETS[preset]
    _READING_ORDER_CONFIG.update(
        {
            "mode": "preset",
            "preset": preset,
            **values,
        }
    )
    return get_reading_order_config()


def set_reading_order_manual(
    panel_direction: str,
    bubble_direction: str,
    vertical_priority: bool = False,
) -> dict[str, Any]:
    for name, value in {
        "panel_direction": panel_direction,
        "bubble_direction": bubble_direction,
    }.items():
        if value not in {"ltr", "rtl"}:
            raise ValueError(f"{name}은 'ltr' 또는 'rtl'이어야 합니다.")

    _READING_ORDER_CONFIG.update(
        {
            "mode": "manual",
            "preset": None,
            "panel_direction": panel_direction,
            "bubble_direction": bubble_direction,
            "vertical_priority": bool(vertical_priority),
        }
    )
    return get_reading_order_config()


def reset_reading_order_config() -> dict[str, Any]:
    _READING_ORDER_CONFIG.update(
        {
            "mode": "legacy",
            "preset": None,
            "panel_direction": None,
            "bubble_direction": None,
            "vertical_priority": False,
        }
    )
    return get_reading_order_config()


def get_reading_order_config() -> dict[str, Any]:
    return dict(_READING_ORDER_CONFIG)


def _resolved_order_config(legacy_direction: str) -> dict[str, Any]:
    config = get_reading_order_config()
    if config["mode"] == "legacy":
        return {
            "mode": "legacy",
            "preset": None,
            "panel_direction": legacy_direction,
            "bubble_direction": legacy_direction,
            "vertical_priority": False,
        }
    return config


# -----------------------------------------------------------------------------
# Layout helpers
# -----------------------------------------------------------------------------

def _center(bbox: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _area(bbox: list[int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection_area(bbox_a: list[int], bbox_b: list[int]) -> int:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _iou(bbox_a: list[int], bbox_b: list[int]) -> float:
    intersection = _intersection_area(bbox_a, bbox_b)
    if intersection <= 0:
        return 0.0
    union = _area(bbox_a) + _area(bbox_b) - intersection
    return intersection / union if union > 0 else 0.0


def _containment_ratio(inner_bbox: list[int], outer_bbox: list[int]) -> float:
    inner_area = _area(inner_bbox)
    if inner_area <= 0:
        return 0.0
    return _intersection_area(inner_bbox, outer_bbox) / inner_area


def _center_inside(inner_bbox: list[int], outer_bbox: list[int]) -> bool:
    cx, cy = _center(inner_bbox)
    x1, y1, x2, y2 = outer_bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _normalize_dialogue(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _sort_layout_regions(
    regions: list[dict[str, Any]],
    direction: str,
    row_tolerance: int,
    vertical_priority: bool = False,
) -> list[dict[str, Any]]:
    if not regions:
        return []

    if vertical_priority:
        # Webtoon-style: top-to-bottom dominates. Horizontal position is only
        # a tie-breaker when two items are nearly level.
        sign = 1 if direction == "ltr" else -1
        return sorted(
            regions,
            key=lambda region: (
                _center(region["bbox"])[1],
                sign * _center(region["bbox"])[0],
            ),
        )

    return sort_regions_reading_order(
        regions,
        direction=direction,
        row_tolerance=row_tolerance,
    )


def _coerce_mask(mask: Any) -> np.ndarray | None:
    if mask is None:
        return None
    try:
        array = np.asarray(mask)
    except Exception:
        return None
    array = np.squeeze(array)
    if array.ndim != 2 or array.size == 0:
        return None
    return array.astype(bool)


def _extract_detection_masks(detections) -> list[np.ndarray | None]:
    raw_masks = getattr(detections, "mask", None)
    if raw_masks is None:
        return []
    try:
        return [_coerce_mask(mask) for mask in raw_masks]
    except TypeError:
        return []


def _mask_containment_ratio(
    text_mask: np.ndarray | None,
    bubble_mask: np.ndarray | None,
) -> float:
    if text_mask is None or bubble_mask is None:
        return 0.0
    if text_mask.shape != bubble_mask.shape:
        return 0.0
    text_area = int(text_mask.sum())
    if text_area <= 0:
        return 0.0
    intersection = int(np.logical_and(text_mask, bubble_mask).sum())
    return intersection / text_area


def _deduplicate_text_regions(
    regions: list[dict[str, Any]],
    iou_threshold: float = 0.78,
    containment_threshold: float = 0.95,
) -> list[dict[str, Any]]:
    """Remove near-duplicate detector regions while preferring the fuller crop."""
    if not regions:
        return []

    # Prefer a larger region first so a full sentence/column is retained instead
    # of a smaller overlapping fragment that would cause clipped duplicate OCR.
    ordered = sorted(
        regions,
        key=lambda region: (
            -_area(region["bbox"]),
            -float(region["score"]),
        ),
    )

    kept: list[dict[str, Any]] = []
    for region in ordered:
        bbox = region["bbox"]
        duplicate = False
        for existing in kept:
            existing_bbox = existing["bbox"]
            if _iou(bbox, existing_bbox) >= iou_threshold:
                duplicate = True
                break
            # A small region almost completely contained by an already-kept
            # larger region is normally the same text detected twice.
            if _containment_ratio(bbox, existing_bbox) >= containment_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(region)

    return kept


def _raw_layout_regions(
    detector,
    image: Image.Image,
    class_thresholds: dict[int, float],
    include_sfx: bool,
    minimum_threshold: float = 0.20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    detections = detector.predict(
        image,
        threshold=minimum_threshold,
        shape=(1152, 1152),
        include_source_image=False,
    )

    detection_masks = _extract_detection_masks(detections)
    text_regions: list[dict[str, Any]] = []
    bubbles: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []

    for detection_index, (box, class_id, score) in enumerate(
        zip(
            detections.xyxy,
            detections.class_id,
            detections.confidence,
        )
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
            "mask": (
                detection_masks[detection_index]
                if detection_index < len(detection_masks)
                else None
            ),
        }

        if class_id == 0 or (class_id == 1 and include_sfx):
            text_regions.append(region)
        elif class_id == 2:
            bubbles.append(region)
        elif class_id == 3:
            panels.append(region)

    return text_regions, bubbles, panels


def _assign_panel_index(
    bbox: list[int],
    panels: list[dict[str, Any]],
) -> int | None:
    candidates = [
        (index, panel)
        for index, panel in enumerate(panels)
        if _center_inside(bbox, panel["bbox"])
    ]
    if not candidates:
        return None
    index, _ = min(candidates, key=lambda item: _area(item[1]["bbox"]))
    return index


def _choose_bubble_for_text(
    text_region: dict[str, Any],
    bubbles: list[dict[str, Any]],
) -> int | None:
    """Assign text to the bubble with the strongest segmentation/spatial overlap."""
    text_bbox = text_region["bbox"]
    text_mask = text_region.get("mask")
    candidates: list[tuple[float, float, float, int]] = []

    for bubble_index, bubble in enumerate(bubbles):
        bubble_bbox = bubble["bbox"]
        bbox_overlap = _containment_ratio(text_bbox, bubble_bbox)
        mask_overlap = _mask_containment_ratio(text_mask, bubble.get("mask"))
        center_inside = _center_inside(text_bbox, bubble_bbox)

        # Segmentation overlap has priority. If masks are unavailable or do not
        # share the same raster shape, use a conservative bbox fallback.
        if mask_overlap > 0:
            if mask_overlap < 0.20:
                continue
        elif bbox_overlap < 0.60 and not center_inside:
            continue

        candidates.append(
            (
                mask_overlap,
                bbox_overlap,
                -float(_area(bubble_bbox)),
                bubble_index,
            )
        )

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][3]


def _group_regions_by_bubble(
    text_regions: list[dict[str, Any]],
    bubbles: list[dict[str, Any]],
    panels: list[dict[str, Any]],
    panel_direction: str,
    bubble_direction: str,
    row_tolerance: int,
    vertical_priority: bool,
) -> list[dict[str, Any]]:
    assignments: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(len(bubbles))
    }
    standalone: list[dict[str, Any]] = []

    for text_region in text_regions:
        bubble_index = _choose_bubble_for_text(text_region, bubbles)
        if bubble_index is None:
            standalone.append(text_region)
            continue
        assignments[bubble_index].append(text_region)

    groups: list[dict[str, Any]] = []
    for bubble_index, bubble in enumerate(bubbles):
        members = assignments[bubble_index]
        if not members:
            continue

        # Detector may emit both a full-text region and smaller overlapping
        # fragments for the same bubble. Remove those before OCR.
        members = _deduplicate_text_regions(members)
        members = _sort_layout_regions(
            members,
            direction=bubble_direction,
            row_tolerance=row_tolerance,
            vertical_priority=vertical_priority,
        )
        groups.append(
            {
                "group_type": "bubble",
                "bbox": bubble["bbox"],
                "bubble_bbox": bubble["bbox"],
                "bubble_score": bubble["score"],
                "text_regions": members,
                "panel_index": _assign_panel_index(bubble["bbox"], panels),
            }
        )

    # Deduplicate only within spatially overlapping standalone text. This keeps
    # captions/narration outside speech bubbles while avoiding obvious repeats.
    standalone = _deduplicate_text_regions(standalone)
    for text_region in standalone:
        groups.append(
            {
                "group_type": "text",
                "bbox": text_region["bbox"],
                "bubble_bbox": None,
                "bubble_score": None,
                "text_regions": [text_region],
                "panel_index": _assign_panel_index(text_region["bbox"], panels),
            }
        )

    # Reading-order logic intentionally unchanged by the grouping patch.
    ordered_panels = _sort_layout_regions(
        panels,
        direction=panel_direction,
        row_tolerance=row_tolerance,
        vertical_priority=vertical_priority,
    )
    panel_rank = {id(panel): rank for rank, panel in enumerate(ordered_panels)}
    original_panel_to_rank: dict[int, int] = {}
    for original_index, panel in enumerate(panels):
        original_panel_to_rank[original_index] = panel_rank.get(id(panel), len(ordered_panels))

    buckets: dict[int | None, list[dict[str, Any]]] = {}
    for group in groups:
        buckets.setdefault(group["panel_index"], []).append(group)

    ordered_groups: list[dict[str, Any]] = []
    panel_indices = sorted(
        [idx for idx in buckets if idx is not None],
        key=lambda idx: original_panel_to_rank.get(idx, 10**9),
    )
    for panel_index in panel_indices:
        ordered_groups.extend(
            _sort_layout_regions(
                buckets[panel_index],
                direction=bubble_direction,
                row_tolerance=row_tolerance,
                vertical_priority=vertical_priority,
            )
        )

    if None in buckets:
        ordered_groups.extend(
            _sort_layout_regions(
                buckets[None],
                direction=bubble_direction,
                row_tolerance=row_tolerance,
                vertical_priority=vertical_priority,
            )
        )

    return ordered_groups


# -----------------------------------------------------------------------------
# Bubble-aware, panel-aware pipeline
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
    debug: bool = True,
    debug_samples_per_page: int = 3,
) -> list[dict[str, Any]]:
    from tqdm.auto import tqdm

    records: list[dict[str, Any]] = []
    language_counter = Counter()
    order_config = _resolved_order_config(reading_direction)

    if debug:
        print("[Reading order]")
        print(f"  mode              : {order_config['mode']}")
        print(f"  preset            : {order_config['preset']}")
        print(f"  panel direction   : {order_config['panel_direction']}")
        print(f"  bubble direction  : {order_config['bubble_direction']}")
        print(f"  vertical priority : {order_config['vertical_priority']}")

    for page_number, page_path in enumerate(
        tqdm(page_paths, desc="페이지 처리"),
        start=1,
    ):
        image = Image.open(page_path).convert("RGB")

        text_regions, bubbles, panels = _raw_layout_regions(
            detector=detector,
            image=image,
            class_thresholds=class_thresholds,
            include_sfx=include_sfx,
        )
        groups = _group_regions_by_bubble(
            text_regions=text_regions,
            bubbles=bubbles,
            panels=panels,
            panel_direction=order_config["panel_direction"],
            bubble_direction=order_config["bubble_direction"],
            row_tolerance=row_tolerance,
            vertical_priority=order_config["vertical_priority"],
        )

        if debug:
            bubble_groups = sum(group["group_type"] == "bubble" for group in groups)
            standalone_groups = len(groups) - bubble_groups
            print(f"\n[Page {page_number:03d}] {page_path.name}")
            print(f"  panels detected : {len(panels)}")
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
                    "panel_index": group["panel_index"],
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
                    "reading_order_mode": order_config["mode"],
                    "reading_order_preset": order_config["preset"],
                    "panel_direction": order_config["panel_direction"],
                    "bubble_direction": order_config["bubble_direction"],
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