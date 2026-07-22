from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from rich.color import Color
from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import RenderResult
from textual.message import Message
from textual.widgets import Static

from aigen.runtime_profiles import resolve_project_path


class SAMPromptCanvas(Static):
    """Terminal image canvas for user-directed SAM box and point prompts."""

    class PromptChanged(Message):
        def __init__(
            self,
            *,
            box: str,
            positive_points: str,
            negative_points: str,
        ) -> None:
            super().__init__()
            self.box = box
            self.positive_points = positive_points
            self.negative_points = negative_points

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("markup", False)
        kwargs.setdefault("classes", "sam-prompt-canvas")
        super().__init__("", **kwargs)
        self._image: np.ndarray | None = None
        self._image_path: Path | None = None
        self._mode = "auto"
        self._box: tuple[int, int, int, int] | None = None
        self._positive_points: list[tuple[int, int]] = []
        self._negative_points: list[tuple[int, int]] = []
        self._pending_box_corner: tuple[int, int] | None = None

    def set_state(
        self,
        image_path: str,
        *,
        prompt_mode: str,
        box: str,
        positive_points: str,
        negative_points: str,
    ) -> None:
        path = resolve_project_path(image_path) if image_path.strip() else None
        if path != self._image_path:
            self._image_path = path
            self._image = self._load_image(path)
        self._mode = prompt_mode
        self._box = self._parse_box(box)
        self._positive_points = self._parse_points(positive_points)
        self._negative_points = self._parse_points(negative_points)
        self._pending_box_corner = None
        self.refresh(layout=False)

    def clear_prompts(self) -> None:
        self._box = None
        self._positive_points.clear()
        self._negative_points.clear()
        self._pending_box_corner = None
        self._emit_prompt_changed()
        self.refresh(layout=False)

    def on_click(self, event: events.Click) -> None:
        if event.button not in (1, 3):
            return
        point = self._source_point(event)
        if point is None:
            return
        event.stop()
        selecting_box = self._mode == "box" or (
            self._mode == "box+points"
            and (self._box is None or self._pending_box_corner is not None)
        )
        if selecting_box and event.button == 1:
            if self._pending_box_corner is None:
                self._pending_box_corner = point
                self._box = None
                self.refresh(layout=False)
                return
            self._box = self._normalized_box(self._pending_box_corner, point)
            self._pending_box_corner = None
            self._emit_prompt_changed()
            return
        if self._mode not in {"points", "box+points"}:
            return
        if self._mode == "box+points" and self._box is None:
            return
        target = self._negative_points if event.button == 3 or event.shift else self._positive_points
        target.append(point)
        self._emit_prompt_changed()

    def render(self) -> RenderResult:
        if self._image is None:
            return Text("No input image", style="dim", no_wrap=True)
        layout = self._layout()
        if layout is None:
            return Text("", no_wrap=True)
        left, top_lines, width, height = layout
        image = np.asarray(
            Image.fromarray(self._image, mode="RGB").resize(
                (width, height),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        ).copy()
        self._draw_prompts(image, width, height)
        lines = [Text(" " * self.content_size.width, end="") for _ in range(self.content_size.height)]
        styles: dict[tuple[int, int, int, int, int, int], Style] = {}
        for row in range(height // 2):
            line = Text(" " * left, end="")
            upper = image[row * 2]
            lower = image[row * 2 + 1]
            for top_pixel, bottom_pixel in zip(upper, lower, strict=True):
                top_rgb = tuple(int(value) for value in top_pixel)
                bottom_rgb = tuple(int(value) for value in bottom_pixel)
                key = (*top_rgb, *bottom_rgb)
                style = styles.get(key)
                if style is None:
                    style = Style(
                        color=Color.from_rgb(*top_rgb),
                        bgcolor=Color.from_rgb(*bottom_rgb),
                    )
                    styles[key] = style
                line.append("▀", style=style)
            line.append(" " * (self.content_size.width - left - width))
            lines[top_lines + row] = line
        return Text("\n").join(lines)

    def _layout(self) -> tuple[int, int, int, int] | None:
        if self._image is None or self.content_size.width < 1 or self.content_size.height < 1:
            return None
        source_height, source_width = self._image.shape[:2]
        max_width = self.content_size.width
        max_height = self.content_size.height * 2
        height = min(max_height, max(2, round(max_width * source_height / source_width)))
        height -= height % 2
        height = max(2, height)
        width = min(max_width, max(1, round(height * source_width / source_height)))
        if width > max_width:
            width = max_width
            height = max(2, round(width * source_height / source_width))
            height -= height % 2
        lines = height // 2
        return (
            (max_width - width) // 2,
            (self.content_size.height - lines) // 2,
            width,
            height,
        )

    def _source_point(self, event: events.MouseEvent) -> tuple[int, int] | None:
        layout = self._layout()
        if layout is None:
            return None
        left, top_lines, width, height = layout
        offset = event.get_content_offset(self)
        if offset is None:
            return None
        x = offset.x - left
        y = (offset.y - top_lines) * 2
        if not 0 <= x < width or not 0 <= y < height:
            return None
        source_height, source_width = self._image.shape[:2]  # type: ignore[union-attr]
        return (
            round(x * (source_width - 1) / max(1, width - 1)),
            round(y * (source_height - 1) / max(1, height - 1)),
        )

    def _draw_prompts(self, image: np.ndarray, width: int, height: int) -> None:
        if self._box is not None:
            left, top, right, bottom = self._box
            for x in range(left, right + 1):
                self._paint_source(image, x, top, (255, 210, 40), width, height)
                self._paint_source(image, x, bottom, (255, 210, 40), width, height)
            for y in range(top, bottom + 1):
                self._paint_source(image, left, y, (255, 210, 40), width, height)
                self._paint_source(image, right, y, (255, 210, 40), width, height)
        if self._pending_box_corner is not None:
            self._paint_marker(image, self._pending_box_corner, (110, 220, 255), width, height)
        for point in self._positive_points:
            self._paint_marker(image, point, (80, 255, 120), width, height)
        for point in self._negative_points:
            self._paint_marker(image, point, (255, 90, 110), width, height)

    def _paint_marker(
        self,
        image: np.ndarray,
        point: tuple[int, int],
        color: tuple[int, int, int],
        width: int,
        height: int,
    ) -> None:
        x, y = self._display_point(point, width, height)
        for delta_y in (-1, 0, 1):
            for delta_x in (-1, 0, 1):
                self._paint_pixel(image, x + delta_x, y + delta_y, color)

    def _paint_source(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        color: tuple[int, int, int],
        width: int,
        height: int,
    ) -> None:
        self._paint_pixel(image, *self._display_point((x, y), width, height), color)

    def _display_point(self, point: tuple[int, int], width: int, height: int) -> tuple[int, int]:
        source_height, source_width = self._image.shape[:2]  # type: ignore[union-attr]
        return (
            round(point[0] * (width - 1) / max(1, source_width - 1)),
            round(point[1] * (height - 1) / max(1, source_height - 1)),
        )

    @staticmethod
    def _paint_pixel(
        image: np.ndarray,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            image[y, x] = color

    def _emit_prompt_changed(self) -> None:
        self.post_message(
            self.PromptChanged(
                box=self._format_box(self._box),
                positive_points=self._format_points(self._positive_points),
                negative_points=self._format_points(self._negative_points),
            )
        )
        self.refresh(layout=False)

    @staticmethod
    def _load_image(path: Path | None) -> np.ndarray | None:
        if path is None or not path.is_file():
            return None
        try:
            with Image.open(path) as image:
                return np.asarray(
                    ImageOps.exif_transpose(image).convert("RGB"),
                    dtype=np.uint8,
                ).copy()
        except OSError:
            return None

    @staticmethod
    def _parse_box(value: str) -> tuple[int, int, int, int] | None:
        try:
            coordinates = tuple(int(part.strip()) for part in value.split(","))
        except ValueError:
            return None
        if len(coordinates) != 4:
            return None
        left, top, right, bottom = coordinates
        if right <= left or bottom <= top:
            return None
        return coordinates

    @staticmethod
    def _parse_points(value: str) -> list[tuple[int, int]]:
        points: list[tuple[int, int]] = []
        for item in value.split(";"):
            if not item.strip():
                continue
            try:
                coordinates = tuple(int(part.strip()) for part in item.split(","))
            except ValueError:
                continue
            if len(coordinates) == 2:
                points.append(coordinates)
        return points

    @staticmethod
    def _normalized_box(
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        return (
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[0], second[0]),
            max(first[1], second[1]),
        )

    @staticmethod
    def _format_box(box: tuple[int, int, int, int] | None) -> str:
        return "" if box is None else ",".join(str(value) for value in box)

    @staticmethod
    def _format_points(points: list[tuple[int, int]]) -> str:
        return ";".join(f"{x},{y}" for x, y in points)
