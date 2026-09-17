from __future__ import annotations


def show_reading_order_controls():
    """Display Colab/Jupyter controls for manga reading order."""
    import ipywidgets as widgets
    from IPython.display import display

    from manga2text_pipeline import (
        get_reading_order_config,
        set_reading_order_manual,
        set_reading_order_preset,
    )

    preset = widgets.Dropdown(
        options=[
            "한국/영문 페이지형",
            "일본 원서",
            "세로 웹툰",
            "일본 만화 번역본(LTR 대사)",
            "수동 설정",
        ],
        value="한국/영문 페이지형",
        description="프리셋",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="520px"),
    )

    panel_direction = widgets.Dropdown(
        options=[
            ("왼쪽 → 오른쪽", "ltr"),
            ("오른쪽 → 왼쪽", "rtl"),
        ],
        value="ltr",
        description="패널 방향",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="520px"),
    )

    bubble_direction = widgets.Dropdown(
        options=[
            ("왼쪽 → 오른쪽", "ltr"),
            ("오른쪽 → 왼쪽", "rtl"),
        ],
        value="ltr",
        description="말풍선 방향",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="520px"),
    )

    vertical_priority = widgets.Checkbox(
        value=False,
        description="위 → 아래 우선 (세로 웹툰)",
        indent=False,
        layout=widgets.Layout(width="520px"),
    )

    apply_button = widgets.Button(
        description="읽기 순서 적용",
        button_style="primary",
    )
    status = widgets.Output()

    def set_manual_enabled(change=None):
        manual = preset.value == "수동 설정"
        panel_direction.disabled = not manual
        bubble_direction.disabled = not manual
        vertical_priority.disabled = not manual

    def apply(_):
        with status:
            status.clear_output()
            if preset.value == "수동 설정":
                config = set_reading_order_manual(
                    panel_direction=panel_direction.value,
                    bubble_direction=bubble_direction.value,
                    vertical_priority=vertical_priority.value,
                )
            else:
                config = set_reading_order_preset(preset.value)

            print("[읽기 순서 적용 완료]")
            print("모드            :", config["mode"])
            print("프리셋          :", config["preset"])
            print("패널 방향       :", config["panel_direction"])
            print("말풍선 방향     :", config["bubble_direction"])
            print("위→아래 우선    :", config["vertical_priority"])

    preset.observe(set_manual_enabled, names="value")
    apply_button.on_click(apply)
    set_manual_enabled()

    print("프리셋을 고르거나 '수동 설정'을 선택해 항목을 하나씩 지정하세요.")
    display(
        widgets.VBox(
            [
                preset,
                panel_direction,
                bubble_direction,
                vertical_priority,
                apply_button,
                status,
            ]
        )
    )

    return {
        "preset": preset,
        "panel_direction": panel_direction,
        "bubble_direction": bubble_direction,
        "vertical_priority": vertical_priority,
        "apply_button": apply_button,
        "status": status,
        "get_config": get_reading_order_config,
    }
