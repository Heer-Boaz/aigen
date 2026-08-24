#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


CANVAS_SIZE = 512
MAX_SPRITE_EXTENT = 416
CONTACT_CELL_SIZE = 160
CONTACT_LABEL_HEIGHT = 24
CONTACT_COLUMNS = 8
BACKGROUNDS = (
    (255, 255, 255),
    (244, 244, 240),
    (232, 236, 240),
    (246, 238, 232),
    (232, 240, 236),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--canvas", type=int, default=CANVAS_SIZE)
    parser.add_argument(
        "--global-scale",
        type=int,
        default=None,
        help="Use one integer scale for every sprite instead of filling the canvas",
    )
    args = parser.parse_args()

    recipe_path = args.recipe.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True)
    train_dir = output_dir / "train"
    train_dir.mkdir()

    records = []
    metadata = []
    contact_images = []
    for index, item in enumerate(recipe["items"], start=1):
        sprite = _load_sprite(item, recipe_path.parent)
        canvas, scale = _compose_canvas(
            sprite,
            BACKGROUNDS[(index - 1) % len(BACKGROUNDS)],
            args.canvas,
            args.global_scale,
        )
        stem = f"{index:03d}_{item['name']}"
        image_path = train_dir / f"{stem}.png"
        caption = f"{recipe['trigger_token']}. {item['caption']}"
        canvas.save(image_path)
        metadata.append({"file_name": image_path.name, "prompt": caption})
        record = {
            "name": item["name"],
            "source": item["source"],
            "source_kind": item["kind"],
            "component": item.get("component"),
            "caption": caption,
            "image": image_path.name,
            "sha256": _sha256(image_path),
            "native_size": list(sprite.size),
            "integer_scale": scale,
        }
        records.append(record)
        contact_images.append((item["name"], canvas))

    _save_contact_sheet(contact_images, output_dir / "contact_sheet.png")
    with (train_dir / "metadata.jsonl").open("w", encoding="utf-8") as stream:
        for item in metadata:
            stream.write(json.dumps(item) + "\n")
    manifest = {
        "kind": "flux2-sprite-style-lora-dataset",
        "source_recipe": recipe_path.as_posix(),
        "trigger_token": recipe["trigger_token"],
        "image_count": len(records),
        "canvas_size": [args.canvas, args.canvas],
        "sprite_extent": MAX_SPRITE_EXTENT if args.global_scale is None else None,
        "global_scale": args.global_scale,
        "resize_filter": "nearest",
        "backgrounds": [list(color) for color in BACKGROUNDS],
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _load_sprite(item: dict[str, object], recipe_dir: Path) -> Image.Image:
    source = (recipe_dir / str(item["source"])).resolve()
    with Image.open(source) as image:
        if item["kind"] == "frame":
            rgba = image.convert("RGBA")
            box = rgba.getchannel("A").getbbox()
            if box is None:
                raise ValueError(f"Empty sprite frame: {source}")
            return rgba.crop(box)
        if item["kind"] == "sheet_component":
            return _sheet_component(image.convert("RGB"), int(item["component"]), source)
    raise ValueError(f"Unknown sprite source kind: {item['kind']}")


def _sheet_component(sheet: Image.Image, component: int, source: Path) -> Image.Image:
    pixels = np.asarray(sheet)
    mask = np.any(pixels != pixels[0, 0], axis=2)
    labels, component_count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if component < 1 or component > component_count:
        raise ValueError(f"Component {component} does not exist in {source}")
    component_slice = ndimage.find_objects(labels)[component - 1]
    if component_slice is None:
        raise ValueError(f"Component {component} is empty in {source}")
    y_slice, x_slice = component_slice
    component_mask = labels[component_slice] == component
    crop = pixels[component_slice]
    rgba = np.zeros((*component_mask.shape, 4), dtype=np.uint8)
    rgba[..., :3] = crop
    rgba[..., 3] = component_mask.astype(np.uint8) * 255
    return Image.fromarray(rgba, "RGBA")


def _compose_canvas(
    sprite: Image.Image,
    background: tuple[int, int, int],
    canvas_size: int,
    global_scale: int | None,
) -> tuple[Image.Image, int]:
    # A per-sprite scale makes every figure fill the canvas, which erases the two
    # things a sprite-style LoRA has to learn: relative body size and pixel grain.
    # A global scale keeps both constant across the dataset.
    if global_scale is not None:
        scale = global_scale
    else:
        scale = max(1, MAX_SPRITE_EXTENT // max(sprite.size))
    scaled_size = (sprite.width * scale, sprite.height * scale)
    if max(scaled_size) > canvas_size:
        raise ValueError(
            f"Sprite {sprite.size} at scale {scale} does not fit canvas {canvas_size}"
        )
    scaled = sprite.resize(scaled_size, Image.Resampling.NEAREST)
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (*background, 255))
    position = ((canvas_size - scaled.width) // 2, (canvas_size - scaled.height) // 2)
    canvas.alpha_composite(scaled, position)
    return canvas.convert("RGB"), scale


def _save_contact_sheet(items: list[tuple[str, Image.Image]], output: Path) -> None:
    rows = (len(items) + CONTACT_COLUMNS - 1) // CONTACT_COLUMNS
    cell_height = CONTACT_CELL_SIZE + CONTACT_LABEL_HEIGHT
    sheet = Image.new("RGB", (CONTACT_COLUMNS * CONTACT_CELL_SIZE, rows * cell_height), (40, 40, 40))
    draw = ImageDraw.Draw(sheet)
    for index, (name, image) in enumerate(items):
        x = index % CONTACT_COLUMNS * CONTACT_CELL_SIZE
        y = index // CONTACT_COLUMNS * cell_height
        thumb = image.resize((CONTACT_CELL_SIZE, CONTACT_CELL_SIZE), Image.Resampling.NEAREST)
        sheet.paste(thumb, (x, y + CONTACT_LABEL_HEIGHT))
        draw.text((x + 4, y + 5), name, fill="white")
    sheet.save(output)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
