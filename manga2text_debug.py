from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from manga2text_pipeline import (
    CLASS_NAMES,
    crop_region,
    detect_regions,
    run_ocr,
    sort_regions_reading_order,
)


def diagnose_page(
    page_path: str | Path,
    detector,
    ocr_model,
    ocr_backend: str,
    class_thresholds: dict[int, float],
    include_sfx: bool = False,
    crop_padding: int = 8,
    reading_direction: str = "ltr",
    row_tolerance: int = 80,
    show_below_threshold: bool = True,
    detector_floor: float = 0.05,
    max_crops: int = 20,
) -> list[dict[str, Any]]:
    """Visualize detector boxes and OCR results for one manga/webtoon page.

    The diagnostic intentionally runs the detector twice conceptually:
    - accepted regions: the exact regions the normal pipeline would OCR
    - low-threshold candidates: regions found by the detector but rejected by
      the configured class threshold

    This makes it easy to distinguish detector misses from OCR failures.
    """
    page_path = Path(page_path)
    image = Image.open(page_path).convert("RGB")

    accepted = detect_regions(
        detector=detector,
        image=image,
        class_thresholds=class_thresholds,
        include_sfx=include_sfx,
        minimum_threshold=min(detector_floor, min(class_thresholds.values())),
    )
    accepted = sort_regions_reading_order(
        accepted,
        direction=reading_direction,
        row_tolerance=row_tolerance,
    )

    # Get raw detector candidates at a low floor so rejected text/SFX boxes are visible.
    raw = detector.predict(
        image,
        threshold=detector_floor,
        shape=(1152, 1152),
        include_source_image=False,
    )

    candidates: list[dict[str, Any]] = []
    for box, class_id, score in zip(raw.xyxy, raw.class_id, raw.confidence):
        class_id = int(class_id)
        score = float(score)
        if class_id not in (0, 1):
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        candidates.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "score": score,
                "bbox": [x1, y1, x2, y2],
                "accepted": (
                    score >= class_thresholds[class_id]
                    and (class_id == 0 or include_sfx)
                ),
            }
        )

    diagnostics: list[dict[str, Any]] = []
    for index, region in enumerate(accepted[:max_crops], start=1):
        crop = crop_region(
            image=image,
            bbox=region["bbox"],
            padding=crop_padding,
        )
        text = run_ocr(
            backend=ocr_backend,
            model=ocr_model,
            crop=crop,
        ).strip()
        diagnostics.append(
            {
                "index": index,
                **region,
                "ocr_text": text,
                "crop": crop,
            }
        )

    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(image)
    ax.set_title(
        f"Detector diagnostic: {page_path.name}\n"
        "solid = accepted, dashed = detected but filtered"
    )
    ax.axis("off")

    if show_below_threshold:
        for region in candidates:
            if region["accepted"]:
                continue
            x1, y1, x2, y2 = region["bbox"]
            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=1.2,
                linestyle="--",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                max(0, y1 - 4),
                f"filtered {region['class_name']} {region['score']:.2f}",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 1},
            )

    for item in diagnostics:
        x1, y1, x2, y2 = item["bbox"]
        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2.0,
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(0, y1 - 4),
            f"#{item['index']} {item['class_name']} {item['score']:.2f}",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "pad": 1},
        )

    plt.show()

    if not diagnostics:
        print("[진단] 정상 임계값을 통과한 OCR 대상 영역이 없습니다.")
        print("       점선 bbox가 보이면 detector threshold/filter 문제일 가능성이 큽니다.")
        print("       점선 bbox도 없으면 detector 자체가 해당 대사를 못 찾은 것입니다.")
        return diagnostics

    print(f"[진단] OCR 대상 영역: {len(diagnostics)}개")

    columns = 3
    rows = (len(diagnostics) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4.5 * rows))

    if rows == 1 and columns == 1:
        axes = [axes]
    else:
        try:
            axes = axes.flatten()
        except AttributeError:
            axes = [axes]

    for ax, item in zip(axes, diagnostics):
        ax.imshow(item["crop"])
        text = item["ocr_text"] if item["ocr_text"] else "<OCR EMPTY>"
        ax.set_title(
            f"#{item['index']} score={item['score']:.2f}\n{text}",
            fontsize=10,
        )
        ax.axis("off")

    for ax in axes[len(diagnostics):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    empty_count = sum(not item["ocr_text"] for item in diagnostics)
    if empty_count:
        print(f"[진단] bbox는 잡혔지만 OCR이 빈 결과인 영역: {empty_count}개")
        print("       이 경우 PaddleOCR/전처리/crop 크기 쪽을 먼저 의심하면 됩니다.")
    else:
        print("[진단] 정상 bbox들은 모두 OCR 문자열을 반환했습니다.")
        print("       빠진 대사가 화면에서 bbox 자체가 없으면 detector 문제입니다.")

    return diagnostics


def diagnose_first_pages(
    page_paths: list[str | Path],
    detector,
    ocr_model,
    ocr_backend: str,
    class_thresholds: dict[int, float],
    page_count: int = 3,
    **kwargs,
) -> dict[str, list[dict[str, Any]]]:
    """Run diagnose_page on the first few prepared page images."""
    output: dict[str, list[dict[str, Any]]] = {}
    for page_path in page_paths[:page_count]:
        print("\n" + "=" * 80)
        print(f"[페이지 진단] {page_path}")
        output[str(page_path)] = diagnose_page(
            page_path=page_path,
            detector=detector,
            ocr_model=ocr_model,
            ocr_backend=ocr_backend,
            class_thresholds=class_thresholds,
            **kwargs,
        )
    return output
