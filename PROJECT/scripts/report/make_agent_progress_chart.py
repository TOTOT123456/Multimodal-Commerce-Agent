#!/usr/bin/env python3
"""Render the single-page 100-tool Agent decision-gain overview."""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA = (
    PROJECT_ROOT
    / "data/tool_call_trace/_aux/project_progress_chart_20260804/chart_data.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "KNOWLEDGE/100工具电商导购Agent_全链路决策增益图.png"

WIDTH = 5120
HEIGHT = 3400
WEB_WIDTH = 2200
WEB_HEIGHT = 1320

BLACK = "#000000"
DARK = "#1A1A1A"
WHITE = "#FFFFFF"
GREEN = "#76B900"
GREEN_DARK = "#4D7800"
GREEN_SOFT = "#EAF3DF"
GREEN_PALE = "#F3F8EC"
SOFT = "#F7F7F7"
HAIR = "#CCCCCC"
GRID = "#E3E3E3"
MUTED = "#757575"
MID = "#5E5E5E"
ORANGE = "#F59D15"
RED = "#D93A32"
PURPLE = "#6D55A4"

CJK_FONT = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
LATIN_BOLD_FONT = Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf")
LATIN_REGULAR_FONT = Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf")


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False, latin: bool = False) -> ImageFont.FreeTypeFont:
    if latin:
        candidate = LATIN_BOLD_FONT if bold else LATIN_REGULAR_FONT
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.truetype(str(CJK_FONT), size=size)


def is_ascii_token(token: str) -> bool:
    return bool(token) and all(ord(char) < 128 for char in token)


def tokenize(text: str) -> list[str]:
    pattern = r"[A-Za-z0-9@._%+/:=\-]+|[\u3400-\u9fff]|[^\s]"
    return re.findall(pattern, text)


def text_runs(value: str) -> list[tuple[str, bool]]:
    """Split a line into Latin and CJK runs for deterministic font fallback."""
    runs: list[tuple[str, bool]] = []
    current = ""
    current_latin: bool | None = None
    for char in value:
        latin = ord(char) < 128
        if current and latin != current_latin:
            runs.append((current, bool(current_latin)))
            current = char
        else:
            current += char
        current_latin = latin
    if current:
        runs.append((current, bool(current_latin)))
    return runs


def mixed_text_length(
    draw: ImageDraw.ImageDraw,
    value: str,
    size: int,
    *,
    bold: bool = False,
) -> float:
    return sum(
        draw.textlength(run, font=font(size, bold=bold, latin=latin))
        for run, latin in text_runs(value)
    )


def wrapped_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    size: int,
    max_width: int,
    *,
    bold: bool = False,
) -> list[str]:
    result: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            result.append("")
            continue
        current = ""
        previous = ""
        for token in tokenize(paragraph):
            spacer = " " if current and is_ascii_token(previous) and is_ascii_token(token) else ""
            candidate = current + spacer + token
            if current and mixed_text_length(draw, candidate, size, bold=bold) > max_width:
                result.append(current)
                current = token
            else:
                current = candidate
            previous = token
        if current:
            result.append(current)
    return result


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    size: int,
    fill: str = DARK,
    *,
    bold: bool = False,
    anchor: str = "la",
    latin: bool = False,
    stroke: int | None = None,
) -> None:
    width = stroke if stroke is not None else (1 if bold and size >= 28 else 0)
    if "\n" in value:
        x, y = xy
        for index, line in enumerate(value.split("\n")):
            text(
                draw,
                (x, y + index * (size + 5)),
                line,
                size,
                fill,
                bold=bold,
                anchor=anchor,
                latin=latin,
                stroke=stroke,
            )
        return
    if latin:
        draw.text(
            xy,
            value,
            font=font(size, bold=bold, latin=True),
            fill=fill,
            anchor=anchor,
            stroke_width=width,
            stroke_fill=fill,
        )
        return

    total_width = mixed_text_length(draw, value, size, bold=bold)
    x, y = xy
    horizontal = anchor[0]
    if horizontal == "m":
        cursor = x - total_width / 2
    elif horizontal == "r":
        cursor = x - total_width
    else:
        cursor = float(x)
    run_anchor = "l" + anchor[1]
    for run, run_is_latin in text_runs(value):
        run_font = font(size, bold=bold, latin=run_is_latin)
        draw.text(
            (cursor, y),
            run,
            font=run_font,
            fill=fill,
            anchor=run_anchor,
            stroke_width=width,
            stroke_fill=fill,
        )
        cursor += draw.textlength(run, font=run_font)


def paragraph(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    size: int,
    max_width: int,
    fill: str = DARK,
    *,
    bold: bool = False,
    line_gap: int = 10,
    max_lines: int | None = None,
) -> int:
    lines = wrapped_lines(draw, value, size, max_width, bold=bold)
    if max_lines is not None:
        lines = lines[:max_lines]
    x, y = xy
    line_height = size + line_gap
    for index, line in enumerate(lines):
        text(draw, (x, y + index * line_height), line, size, fill, bold=bold)
    return len(lines) * line_height


def card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str = WHITE,
    outline: str = HAIR,
    width: int = 2,
    accent: str | None = None,
) -> None:
    draw.rectangle(box, fill=fill, outline=outline, width=width)
    if accent:
        x0, y0, x1, _ = box
        draw.rectangle((x0, y0, x1, y0 + 8), fill=accent)


def chip(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    label: str,
    *,
    fill: str,
    fg: str,
    size: int = 24,
    padding_x: int = 18,
    height: int = 44,
    outline: str | None = None,
) -> int:
    width = int(mixed_text_length(draw, label, size)) + padding_x * 2
    x, y = xy
    draw.rectangle((x, y, x + width, y + height), fill=fill, outline=outline, width=2)
    text(draw, (x + width // 2, y + height // 2), label, size, fg, anchor="mm")
    return width


def section_title(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    index: str,
    title_value: str,
    subtitle: str,
) -> None:
    x, y = xy
    draw.rectangle((x, y + 3, x + 56, y + 59), fill=GREEN)
    text(draw, (x + 28, y + 31), index, 25, BLACK, bold=True, anchor="mm", latin=True)
    text(draw, (x + 78, y), title_value, 40, BLACK, bold=True)
    text(draw, (x + 78, y + 54), subtitle, 25, MUTED)


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def draw_header(draw: ImageDraw.ImageDraw, data: dict[str, Any]) -> None:
    draw.rectangle((0, 0, WIDTH, 430), fill=BLACK)
    draw.rectangle((160, 74, 310, 89), fill=GREEN)
    text(draw, (160, 112), data["title"], 74, WHITE, bold=True)
    text(draw, (163, 214), data["subtitle"], 31, "#D2D2D2")

    x = 163
    for badge in data["header_badges"]:
        badge_width = chip(
            draw,
            (x, 292),
            badge,
            fill=DARK,
            fg=WHITE,
            outline="#434343",
            size=25,
            height=50,
        )
        x += badge_width + 14

    draw.line((3870, 62, 3870, 364), fill="#4A4A4A", width=2)
    text(draw, (3950, 70), "FINAL · 9K FULL TRACE", 25, GREEN, bold=True, latin=True)
    text(draw, (3942, 115), fmt_pct(data["final"]["full_trace"]), 116, WHITE, bold=True, latin=True)
    text(
        draw,
        (3948, 263),
        f"{data['final']['success_count']:,} / {data['final']['total_count']:,} 条完整成功",
        29,
        "#D6D6D6",
    )
    text(draw, (3948, 320), data["final"]["endpoint"], 24, "#AFAFAF")


def draw_main_curve(draw: ImageDraw.ImageDraw, data: dict[str, Any]) -> None:
    panel = (120, 470, 5000, 1770)
    card(draw, panel, fill=WHITE, outline=HAIR, width=2, accent=GREEN)
    points = data["system_curve"]
    section_title(
        draw,
        (170, 510),
        "01",
        "同一 9K 口径的累计 Full Trace 主线",
        "冻结 9K（3K 闲聊 + 6K 电商）；起点没有 Intent 与闲聊出口，3K 闲聊全部失败",
    )

    left, right = 400, 4790
    top, bottom = 650, 1260
    y_min, y_max = 15.0, 90.0

    def y_of(value: float) -> int:
        return int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top))

    for tick in range(20, 91, 10):
        y = y_of(float(tick))
        draw.line((left, y, right, y), fill=GRID, width=2)
        text(draw, (left - 26, y), f"{tick}%", 25, MUTED, anchor="rm", latin=True)
    draw.line((left, top, left, bottom), fill=MID, width=3)
    draw.line((left, bottom, right, bottom), fill=MID, width=3)
    text(draw, (170, 932), "Full Trace", 24, MUTED)
    text(draw, (170, 972), "（越高越好）", 22, MUTED)

    step = (right - left) / (len(points) - 1)
    coords = [(int(left + index * step), y_of(point["value"])) for index, point in enumerate(points)]

    area = [(coords[0][0], bottom), *coords, (coords[-1][0], bottom)]
    draw.polygon(area, fill=GREEN_PALE)
    draw.line(coords, fill=GREEN, width=12, joint="curve")

    for index, ((x, y), point) in enumerate(zip(coords, points)):
        if index:
            previous = points[index - 1]
            delta = point["value"] - previous["value"]
            delta_count = point["count"] - previous["count"]
            mid_x = int((coords[index - 1][0] + x) / 2)
            mid_y = int((coords[index - 1][1] + y) / 2) - 28
            label = f"{delta:+.2f}pp / {delta_count:+,} 条"
            label_width = int(mixed_text_length(draw, label, 18)) + 20
            draw.rectangle(
                (mid_x - label_width // 2, mid_y - 17, mid_x + label_width // 2, mid_y + 17),
                fill=BLACK,
            )
            text(draw, (mid_x, mid_y), label, 18, WHITE, anchor="mm", latin=True)

        retrospective = bool(point.get("retrospective"))
        radius = 17 if index != len(points) - 1 else 24
        if retrospective:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=WHITE, outline=ORANGE, width=7)
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=GREEN, outline=BLACK, width=4)
        if index == len(points) - 1:
            draw.ellipse((x - 38, y - 38, x + 38, y + 38), outline=GREEN, width=5)

        value_y = y - 46 - (index % 2) * 30
        value_fill = ORANGE if retrospective else BLACK
        text(draw, (x, value_y), fmt_pct(point["value"]), 28, value_fill, bold=True, anchor="ms", latin=True)

        card_left, card_right, card_gap = 150, 4970, 8
        card_width = (card_right - card_left - card_gap * (len(points) - 1)) / len(points)
        card_x0 = int(card_left + index * (card_width + card_gap))
        card_x1 = int(card_x0 + card_width)
        card_y0 = 1330
        card(draw, (card_x0, card_y0, card_x1, 1682), fill=SOFT, outline=HAIR, width=2, accent=ORANGE if retrospective else GREEN)
        draw.rectangle((card_x0 + 14, card_y0 + 22, card_x0 + 52, card_y0 + 60), fill=BLACK)
        text(draw, (card_x0 + 33, card_y0 + 41), str(index), 18, WHITE, bold=True, anchor="mm", latin=True)
        text(draw, (card_x0 + 62, card_y0 + 20), point["stage"], 21, BLACK, bold=True)
        text(draw, (card_x0 + 14, card_y0 + 82), point["line1"], 18, DARK)
        text(draw, (card_x0 + 14, card_y0 + 119), point["line2"], 17, MUTED)
        if point.get("line3"):
            text(draw, (card_x0 + 14, card_y0 + 154), point["line3"], 17, MUTED)
        draw.line((card_x0 + 14, card_y0 + 202, card_x1 - 14, card_y0 + 202), fill=HAIR, width=2)
        text(draw, (card_x0 + 14, card_y0 + 220), f"{point['count']:,} / 9,000", 22, BLACK, bold=True, latin=True)
        text(draw, (card_x0 + 14, card_y0 + 263), point["evidence"], 16, MUTED)
        if retrospective:
            chip(draw, (card_x0 + 14, card_y0 + 302), "RETROSPECTIVE", fill="#FFF2DD", fg="#A86300", size=15, height=30)


def draw_selection_card(draw: ImageDraw.ImageDraw, data: dict[str, Any]) -> None:
    x0, y0, x1, y1 = 120, 1810, 1640, 3160
    card(draw, (x0, y0, x1, y1), fill=WHITE, outline=HAIR, width=2, accent=BLACK)
    section_title(draw, (170, 1852), "02", "基座池（先比较，再锁定）", "局部指标与端到端结果同时看，避免把最大模型直接当作最优系统")

    row_y = 1985
    for item in data["base_pool"]:
        draw.rectangle((170, row_y, 1590, row_y + 205), fill=SOFT, outline=GRID, width=2)
        draw.rectangle((170, row_y, 182, row_y + 205), fill=GREEN if item["selected"] else HAIR)
        text(draw, (208, row_y + 22), item["module"], 28, BLACK, bold=True)
        text(draw, (208, row_y + 66), item["choice"], 23, GREEN_DARK, bold=True)
        paragraph(draw, (208, row_y + 105), item["comparison"], 21, 1325, MUTED, line_gap=7, max_lines=3)
        row_y += 225

    note_y = 2885
    draw.rectangle((170, note_y, 1590, 3110), fill=BLACK)
    draw.rectangle((170, note_y, 184, 3110), fill=GREEN)
    text(draw, (210, note_y + 24), "系统级反例", 25, GREEN, bold=True)
    paragraph(draw, (210, note_y + 70), data["base_pool_note"], 22, 1330, WHITE, line_gap=9, max_lines=5)


def draw_retrieval_card(draw: ImageDraw.ImageDraw, data: dict[str, Any]) -> None:
    x0, y0, x1, y1 = 1680, 1810, 3480, 3160
    card(draw, (x0, y0, x1, y1), fill=WHITE, outline=HAIR, width=2, accent=GREEN)
    section_title(draw, (1730, 1852), "03", "检索证据轨（Legacy 6K AllHit@10）", "覆盖率不是 Full Trace；它解释候选质量如何逐层抬高，并保留非单调决策")

    values = data["retrieval_curve"]
    left, right = 1780, 3385
    top, bottom = 2045, 2520
    y_min, y_max = 40.0, 100.0

    def y_of(value: float) -> int:
        return int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top))

    for tick in (40, 60, 80, 100):
        y = y_of(float(tick))
        draw.line((left, y, right, y), fill=GRID, width=2)
        text(draw, (left - 18, y), str(tick), 19, MUTED, anchor="rm", latin=True)
    coords = []
    step = (right - left) / (len(values) - 1)
    for index, point in enumerate(values):
        coords.append((int(left + index * step), y_of(point["value"])))
    draw.line(coords, fill=GREEN, width=9, joint="curve")

    for index, ((x, y), point) in enumerate(zip(coords, values)):
        negative = index and point["value"] < values[index - 1]["value"]
        point_color = ORANGE if negative else GREEN
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=point_color, outline=BLACK, width=3)
        label_y = y - 37 if index % 2 == 0 else y + 25
        text(draw, (x, label_y), fmt_pct(point["value"]), 21, ORANGE if negative else BLACK, bold=True, anchor="ms", latin=True)
        stage_y = 2552 + (index % 2) * 34
        text(draw, (x, stage_y), point["short"], 18, DARK, anchor="ma")

    draw.rectangle((1740, 2635, 3420, 2792), fill=GREEN_SOFT, outline="#B7D58E", width=2)
    paragraph(draw, (1770, 2663), data["retrieval_note"], 22, 1620, DARK, line_gap=8, max_lines=4)

    funnel_y = 2835
    funnel = data["retrieval_funnel"]
    cell_width = 400
    for index, item in enumerate(funnel):
        cell_x = 1740 + index * (cell_width + 18)
        draw.rectangle((cell_x, funnel_y, cell_x + cell_width, 3095), fill=SOFT, outline=HAIR, width=2)
        text(draw, (cell_x + 20, funnel_y + 20), item["value"], 31, BLACK, bold=True, latin=True)
        text(draw, (cell_x + 20, funnel_y + 69), item["label"], 21, GREEN_DARK, bold=True)
        paragraph(draw, (cell_x + 20, funnel_y + 112), item["detail"], 19, cell_width - 40, MUTED, line_gap=6, max_lines=4)


def draw_dpo_card(draw: ImageDraw.ImageDraw, data: dict[str, Any]) -> None:
    x0, y0, x1, y1 = 3520, 1810, 5000, 3160
    card(draw, (x0, y0, x1, y1), fill=WHITE, outline=HAIR, width=2, accent=GREEN)
    section_title(draw, (3570, 1852), "04", "三阶段偏好优化（共 9,200 对）", "chosen=标注工具链；rejected=对应父模型真实错误；冻结样本不进入训练或选参")

    cards = data["dpo_stages"]
    stage_y = 1990
    for index, item in enumerate(cards):
        box_y = stage_y + index * 242
        draw.rectangle((3570, box_y, 4950, box_y + 222), fill=SOFT, outline=HAIR, width=2)
        draw.rectangle((3570, box_y, 3700, box_y + 222), fill=BLACK)
        text(draw, (3635, box_y + 56), f"DPO-{index + 1}", 22, GREEN, bold=True, anchor="mm", latin=True)
        text(draw, (3635, box_y + 118), f"{item['pairs']:,}", 31, WHITE, bold=True, anchor="mm", latin=True)
        text(draw, (3635, box_y + 165), "pairs", 18, "#BFBFBF", anchor="mm", latin=True)
        text(draw, (3730, box_y + 22), item["title"], 25, BLACK, bold=True)
        text(draw, (3730, box_y + 64), item["source"], 20, DARK)
        text(draw, (3730, box_y + 102), item["split"], 19, MUTED)
        text(draw, (3730, box_y + 142), item["result"], 21, GREEN_DARK, bold=True)
        text(draw, (3730, box_y + 180), item["checkpoint"], 18, MUTED)

    chart_y = 2735
    text(draw, (3570, chart_y), "隔离确认 TraceEM（选参轨）", 23, BLACK, bold=True)
    confirmation = data["dpo_confirmation"]
    cx = [3650, 4250, 4850]
    cy = []
    for value in confirmation:
        cy.append(int(2940 - (value - 70.0) / 8.0 * 150))
    draw.line(list(zip(cx, cy)), fill=GREEN, width=8)
    for index, (x, y) in enumerate(zip(cx, cy)):
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=GREEN, outline=BLACK, width=3)
        text(draw, (x, y - 26), fmt_pct(confirmation[index]), 20, BLACK, bold=True, anchor="ms", latin=True)
        text(draw, (x, 2972), f"DPO-{index + 1}", 18, MUTED, anchor="ma", latin=True)

    draw.rectangle((3570, 3020, 4950, 3112), fill="#FFF2DD", outline="#E9B867", width=2)
    paragraph(draw, (3595, 3038), data["dpo_rejected_note"], 19, 1330, "#7A4A00", line_gap=6, max_lines=3)


def draw_footer(draw: ImageDraw.ImageDraw, data: dict[str, Any]) -> None:
    draw.rectangle((0, 3200, WIDTH, HEIGHT), fill=BLACK)
    draw.rectangle((150, 3240, 166, 3360), fill=GREEN)
    text(draw, (195, 3237), "口径与边界", 24, GREEN, bold=True)
    text(draw, (195, 3280), data["scope"], 21, WHITE)
    text(draw, (195, 3322), data["evidence_line"], 18, "#BDBDBD")

    errors = data["final_errors"]
    right_x = 3560
    text(draw, (right_x, 3237), "FINAL ERROR BUDGET（1,355）", 22, GREEN, bold=True, latin=True)
    bar_x0, bar_y0, bar_w, bar_h = right_x, 3284, 1240, 30
    total = sum(item["count"] for item in errors)
    cursor = bar_x0
    colors = [ORANGE, RED, "#8B8B8B", PURPLE]
    for item, color in zip(errors, colors):
        width = max(5, int(bar_w * item["count"] / total))
        draw.rectangle((cursor, bar_y0, cursor + width, bar_y0 + bar_h), fill=color)
        cursor += width
    label_x = right_x
    for item, color in zip(errors, colors):
        draw.rectangle((label_x, 3332, label_x + 12, 3344), fill=color)
        text(draw, (label_x + 20, 3322), f"{item['label']} {item['count']}", 18, "#D2D2D2")
        label_x += 295


def render_web_curve(data: dict[str, Any], output: Path) -> None:
    """Render a taller, narrower main curve for the project conclusion page."""
    image = Image.new("RGB", (WEB_WIDTH, WEB_HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)
    points = data["system_curve"]

    # Editorial header: keep the web figure self-contained when shared alone.
    draw.rectangle((0, 0, WEB_WIDTH, 205), fill=BLACK)
    draw.rectangle((76, 42, 200, 55), fill=GREEN)
    text(draw, (76, 72), "PROJECT CONCLUSION / FROZEN 9K", 24, GREEN, bold=True, latin=True)
    text(draw, (76, 111), f"{len(points)} 个决策点，如何把端到端成功率推到 84.94%", 45, WHITE, bold=True)
    text(draw, (78, 169), "同一 9K Full Trace 口径 | 3K 闲聊 + 6K 电商 | 逐条静态聚合", 22, "#CFCFCF")
    draw.line((1745, 36, 1745, 171), fill="#454545", width=2)
    text(draw, (1810, 49), "FINAL", 21, GREEN, bold=True, latin=True)
    text(draw, (2105, 72), fmt_pct(data["final"]["full_trace"]), 69, WHITE, bold=True, anchor="ra", latin=True)
    text(draw, (2105, 153), f"{data['final']['success_count']:,} / 9,000", 25, "#D0D0D0", anchor="ra", latin=True)

    left, right = 150, 2070
    top, bottom = 306, 814
    y_min, y_max = 15.0, 90.0

    def y_of(value: float) -> int:
        return int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top))

    step = (right - left) / (len(points) - 1)
    coords = [(int(left + index * step), y_of(point["value"])) for index, point in enumerate(points)]

    # Decision bands are derived from contiguous phases in chart_data.json.
    phase_styles = [
        (SOFT, DARK),
        ("#E9E9E9", DARK),
        (GREEN_SOFT, GREEN_DARK),
        ("#DDEBCB", GREEN_DARK),
    ]
    phase_ranges: list[tuple[int, int, str]] = []
    phase_start = 0
    for index in range(1, len(points) + 1):
        if index == len(points) or points[index].get("phase") != points[phase_start].get("phase"):
            phase_ranges.append((phase_start, index - 1, points[phase_start].get("phase_label", "")))
            phase_start = index
    boundaries = []
    for index, (start, end, label) in enumerate(phase_ranges):
        x0 = left if start == 0 else int((coords[start - 1][0] + coords[start][0]) / 2) + 2
        x1 = right if end == len(points) - 1 else int((coords[end][0] + coords[end + 1][0]) / 2)
        fill, fg = phase_styles[index % len(phase_styles)]
        boundaries.append((x0, x1, label, fill, fg))
    for x0, x1, label, fill, fg in boundaries:
        draw.rectangle((x0, 232, x1, 274), fill=fill, outline=HAIR, width=1)
        text(draw, ((x0 + x1) // 2, 253), label, 18, fg, bold=True, anchor="mm")

    for tick in range(20, 91, 10):
        y = y_of(float(tick))
        draw.line((left, y, right, y), fill=GRID, width=2)
        text(draw, (left - 22, y), f"{tick}%", 20, MUTED, anchor="rm", latin=True)
    for x, _ in coords:
        draw.line((x, top, x, bottom), fill="#F0F0F0", width=1)
    draw.line((left, top, left, bottom), fill=MID, width=2)
    draw.line((left, bottom, right, bottom), fill=MID, width=2)
    text(draw, (48, 519), "FULL TRACE", 18, MUTED, bold=True, latin=True)
    text(draw, (48, 547), "越高越好", 18, MUTED)

    area = [(coords[0][0], bottom), *coords, (coords[-1][0], bottom)]
    draw.polygon(area, fill=GREEN_PALE)
    draw.line(coords, fill=GREEN, width=10, joint="curve")

    # Segment gains sit on alternating sides of the curve to stay legible.
    for index in range(1, len(points)):
        previous = points[index - 1]
        point = points[index]
        x = int((coords[index - 1][0] + coords[index][0]) / 2)
        y = int((coords[index - 1][1] + coords[index][1]) / 2) + (38 if index % 2 else -46)
        if index == 1:
            y -= 76
        label = f"{point['value'] - previous['value']:+.2f}pp / {point['count'] - previous['count']:+,}"
        label_width = int(mixed_text_length(draw, label, 17, bold=True)) + 24
        draw.rectangle((x - label_width // 2, y - 17, x + label_width // 2, y + 17), fill=BLACK)
        text(draw, (x, y), label, 17, WHITE, bold=True, anchor="mm", latin=True, stroke=0)

    for index, ((x, y), point) in enumerate(zip(coords, points)):
        retrospective = bool(point.get("retrospective"))
        radius = 14 if index != len(points) - 1 else 19
        if retrospective:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=WHITE, outline=ORANGE, width=6)
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=GREEN, outline=BLACK, width=3)
        if index == len(points) - 1:
            draw.ellipse((x - 31, y - 31, x + 31, y + 31), outline=GREEN, width=4)
        text(
            draw,
            (x, y - 34),
            fmt_pct(point["value"]),
            27,
            ORANGE if retrospective else BLACK,
            bold=True,
            anchor="ms",
            latin=True,
        )

    # Compact point captions retain the decision and the exact accumulated count.
    cell_top, cell_bottom = 850, 1092
    cell_width = (right - left) / len(points)
    for index, point in enumerate(points):
        x0 = int(left + index * cell_width)
        x1 = int(left + (index + 1) * cell_width)
        accent = ORANGE if point.get("retrospective") else GREEN
        draw.rectangle((x0, cell_top, x1, cell_bottom), fill=SOFT, outline=HAIR, width=1)
        draw.rectangle((x0, cell_top, x1, cell_top + 6), fill=accent)
        draw.rectangle((x0 + 14, cell_top + 18, x0 + 47, cell_top + 51), fill=BLACK)
        text(draw, (x0 + 30, cell_top + 35), str(index), 16, WHITE, bold=True, anchor="mm", latin=True)
        text(draw, (x0 + 52, cell_top + 20), point["stage"], 16, BLACK, bold=True)
        paragraph(draw, (x0 + 10, cell_top + 69), point["web_detail"], 14, int(cell_width - 20), MUTED, line_gap=5, max_lines=4)
        draw.line((x0 + 14, cell_bottom - 47, x1 - 14, cell_bottom - 47), fill=HAIR, width=1)
        text(draw, (x0 + 10, cell_bottom - 35), f"{point['count']:,} / 9K", 15, BLACK, bold=True, latin=True)

    # Closing strip states the net gain, final route, and evidence semantics.
    draw.rectangle((0, 1130, WEB_WIDTH, WEB_HEIGHT), fill=BLACK)
    draw.rectangle((70, 1166, 82, 1284), fill=GREEN)
    text(draw, (112, 1157), "累计净增益", 19, GREEN, bold=True)
    total_delta = points[-1]["value"] - points[0]["value"]
    total_count_delta = points[-1]["count"] - points[0]["count"]
    text(draw, (112, 1192), f"{total_delta:+.2f}pp", 43, WHITE, bold=True, latin=True)
    text(draw, (112, 1251), f"{points[0]['count']:,} -> {points[-1]['count']:,}（{total_count_delta:+,} 条）", 20, "#C9C9C9")

    draw.line((660, 1162, 660, 1287), fill="#444444", width=2)
    text(draw, (700, 1157), "最终部署链路", 19, GREEN, bold=True)
    text(draw, (700, 1194), "0.6B 路由 + 精排 20 / 规划 10", 26, WHITE, bold=True)
    text(draw, (700, 1239), "单 Planner -> Intent -> RAG -> SFT -> DPO", 20, "#C9C9C9")

    draw.line((1535, 1162, 1535, 1287), fill="#444444", width=2)
    text(draw, (1575, 1157), "读图口径", 19, GREEN, bold=True)
    draw.ellipse((1575, 1204, 1595, 1224), fill=WHITE, outline=ORANGE, width=4)
    text(draw, (1610, 1198), "空心橙点：回溯补测，不参与选型", 17, "#D0D0D0")
    draw.ellipse((1575, 1248, 1595, 1268), fill=GREEN, outline=WHITE, width=2)
    text(draw, (1610, 1242), "实心绿点：正式系统端点", 17, "#D0D0D0")

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True, dpi=(180, 180))


def validate(data: dict[str, Any]) -> None:
    points = data.get("system_curve", [])
    if not 2 <= len(points) <= 16:
        raise ValueError("system_curve must contain between two and sixteen stages")
    if any("web_detail" not in point or "phase" not in point for point in points):
        raise ValueError("each system stage must define web_detail and phase")
    if points[-1]["count"] != data["final"]["success_count"]:
        raise ValueError("final curve count does not match header")
    if abs(points[-1]["value"] - data["final"]["full_trace"]) > 1e-9:
        raise ValueError("final curve percentage does not match header")
    if sum(item["pairs"] for item in data.get("dpo_stages", [])) != 9200:
        raise ValueError("DPO preference-pair total must be 9,200")
    if sum(item["count"] for item in data.get("final_errors", [])) != 1355:
        raise ValueError("final error decomposition must total 1,355")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--web-out", type=Path)
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    validate(data)

    image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)
    draw_header(draw, data)
    draw_main_curve(draw, data)
    draw_selection_card(draw, data)
    draw_retrieval_card(draw, data)
    draw_dpo_card(draw, data)
    draw_footer(draw, data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out, format="PNG", optimize=True, dpi=(180, 180))
    print(f"chart -> {args.out} ({WIDTH}x{HEIGHT})")
    if args.web_out:
        render_web_curve(data, args.web_out)
        print(f"web curve -> {args.web_out} ({WEB_WIDTH}x{WEB_HEIGHT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
