"""
clean_backgrounds.py

Maak de achtergrond van afbeeldingen wit en verwijder kleine AI-artefacten in de achtergrond.

Gebruik:
  python tools/clean_backgrounds.py /home/boaz/aigen/assets/lora/JSEED/train

Opties:
  --inplace    : Vervang originele bestanden (gevaarlijk)
  --white-threshold : afstandsdrempel (0-441) tot wit (standaard 40)
  --max-blob   : maximale grootte (in pixels) van donkere blobs in de achtergrond die worden verwijderd (standaard 800)

Dit script gebruikt OpenCV en Pillow. Installeer vereisten indien nodig:
  pip install opencv-python pillow numpy tqdm

"""
import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def process_image(in_path: Path, out_path: Path, white_thresh: float = 40.0, max_blob: int = 800) -> bool:
    # Lees de afbeelding met behoud van alfa als aanwezig
    img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Kan bestand niet lezen: {in_path}")
        return False

    # Zorg dat we RGB (BGR in OpenCV) zonder alfa door compositing op wit
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        # alpha compositing op wit
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3].astype(np.float32) / 255.0
        white = np.ones_like(bgr) * 255.0
        img_rgb = (bgr * alpha[:, :, None] + white * (1.0 - alpha[:, :, None])).astype(np.uint8)
    else:
        img_rgb = img[:, :, :3]

    h, w = img_rgb.shape[:2]

    # Bereken kleurafstand tot wit en bepaal "near-white" pixels
    diff = 255.0 - img_rgb.astype(np.float32)
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    bg_like = (dist < white_thresh)

    # Morfologische kernels
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # Pixels die NIET op wit lijken (mogelijke onderwerp pixels)
    mask_non_bg = (~bg_like).astype(np.uint8) * 255

    # Sluit kleine gaten en verwijder ruis zodat we een stabiele "subject seed" krijgen
    mask_clean = cv2.morphologyEx(mask_non_bg, cv2.MORPH_CLOSE, kernel_med, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Preserve thin dark strokes (line-art): detect dark pixels and edges and
    # re-union with the cleaned mask so fine lines (mouth, eyes) are not
    # accidentally removed by the closing/opening steps.
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # detect both dark pixels and Canny edges; combine and dilate to strengthen
    stroke_thresh = 220
    dark_mask = (gray < stroke_thresh).astype(np.uint8) * 255
    edges = cv2.Canny(gray, 50, 150)
    edge_mask = (edges > 0).astype(np.uint8) * 255
    stroke_mask = cv2.bitwise_or(dark_mask, edge_mask)
    # Dilate rather than open: we want to thicken and preserve thin strokes
    stroke_mask = cv2.dilate(stroke_mask, kernel_small, iterations=1)
    mask_clean = cv2.bitwise_or(mask_clean, stroke_mask)

    # Vind grootste verbonden component — dit is meestal het onderwerp (figure)
    num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
    fg_seed = np.zeros((h, w), dtype=np.uint8)
    if num_cc > 1:
        # kies component met grootste area (excl achtergrond label 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_i = 1 + int(np.argmax(areas))

        # Start met de grootste component als onderwerp-seed
        fg_seed[labels == max_i] = 1

        # Breid selectie uit: includeer andere componenten die dichtbij of binnen
        # de vergrote bounding box van het hoofdonderwerp liggen, en iteratief
        # componenten die via een kleine dilatie aansluiten.
        left = int(stats[max_i, cv2.CC_STAT_LEFT])
        top = int(stats[max_i, cv2.CC_STAT_TOP])
        width = int(stats[max_i, cv2.CC_STAT_WIDTH])
        height = int(stats[max_i, cv2.CC_STAT_HEIGHT])
        margin = max(10, int(0.02 * max(h, w)))
        ex_left = max(0, left - margin)
        ex_top = max(0, top - margin)
        ex_right = min(w, left + width + margin)
        ex_bottom = min(h, top + height + margin)

        included = {max_i}
        changed = True
        while changed:
            changed = False
            # gecombineerde mask van momenteel inbegrepen componenten
            combined_mask = np.isin(labels, list(included)).astype(np.uint8)
            dil = cv2.dilate(combined_mask, kernel_small, iterations=5)
            for i in range(1, num_cc):
                if i in included:
                    continue
                li = int(stats[i, cv2.CC_STAT_LEFT])
                ti = int(stats[i, cv2.CC_STAT_TOP])
                wi = int(stats[i, cv2.CC_STAT_WIDTH])
                hi = int(stats[i, cv2.CC_STAT_HEIGHT])
                # bbox intersects expanded main bbox?
                intersects = not (li + wi < ex_left or li > ex_right or ti + hi < ex_top or ti > ex_bottom)
                # or component touches dilated combined mask
                touches = np.any(dil[labels == i] > 0)
                if intersects or touches:
                    included.add(i)
                    changed = True

        # Also include any components that contain stroke pixels inside the
        # expanded main bbox. This catches thin, isolated strokes (mouth,
        # nose lines) that may not be connected to the main component.
        try:
            stroke_labels = np.unique(labels[(stroke_mask > 0) & (np.arange(h)[:, None] >= ex_top) & (np.arange(h)[:, None] < ex_bottom)])
        except Exception:
            # fallback: scan bbox directly
            stroke_area = stroke_mask[ex_top:ex_bottom, ex_left:ex_right]
            if stroke_area.size > 0:
                labs = np.unique(labels[ex_top:ex_bottom, ex_left:ex_right][stroke_area > 0])
                for lab in labs:
                    if lab != 0:
                        included.add(int(lab))

        # Stel fg_seed op basis van alle inbegrepen componenten
        fg_seed[:] = 0
        for i in included:
            fg_seed[labels == i] = 1

    else:
        # fallback: gebruik mask_clean direct
        fg_seed = (mask_clean > 0).astype(np.uint8)

    # pad/expand subject seed licht om dunne kenmerken niet af te kappen
    fg_seed = cv2.dilate(fg_seed, kernel_small, iterations=2)

    # Achtergrond = alles dat niet tot het onderwerp behoort
    bg_mask = (fg_seed == 0).astype(np.uint8) * 255

    # Verwijder kleine donkere blobs in achtergrond (artefact cleanup).
    # Alleen verwijder componenten die niet duidelijk binnen/tegen het onderwerp
    # aanliggen - dit voorkomt dat kleine lijntjes zoals een mond per ongeluk
    # weggepoetst worden.
    fg_mask = 255 - bg_mask
    num_cc2, labels2, stats2, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)

    # bereken onderwerp-bbox uit fg_seed (indien aanwezig) om nabijheidstest uit te voeren
    ys, xs = np.where(fg_seed > 0)
    if ys.size > 0:
        subj_top = int(ys.min())
        subj_bottom = int(ys.max())
        subj_left = int(xs.min())
        subj_right = int(xs.max())
    else:
        subj_top = subj_left = subj_right = subj_bottom = None

    for i in range(1, num_cc2):
        area = int(stats2[i, cv2.CC_STAT_AREA])
        if area < max_blob:
            comp_mask = (labels2 == i)
            # als component enige overlap heeft met het onderwerp, sla over
            if np.any(fg_seed[comp_mask > 0] > 0):
                continue
            # als component bbox in of direct naast het onderwerp valt, sla over
            if subj_top is not None:
                li = int(stats2[i, cv2.CC_STAT_LEFT])
                ti = int(stats2[i, cv2.CC_STAT_TOP])
                wi = int(stats2[i, cv2.CC_STAT_WIDTH])
                hi = int(stats2[i, cv2.CC_STAT_HEIGHT])
                # intersectie test met kleine marge
                margin = max(6, int(0.01 * max(h, w)))
                if not (li + wi < subj_left - margin or li > subj_right + margin or ti + hi < subj_top - margin or ti > subj_bottom + margin):
                    continue
            # anders is het waarschijnlijk een kleine artefact in de achtergrond -> verwijder
            bg_mask[comp_mask] = 255

    # final smoothing
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    # Zet achtergrondpixels strikt wit, maar bescherm het onderwerp
    out = img_rgb.copy()
    out[bg_mask == 255] = 255

    # Randverzachting: blur op de grensgebieden om harde randen te verzachten
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    dil = cv2.dilate(edges, kernel_small, iterations=1)
    border_area = (dil > 0) & (bg_mask == 255)
    if np.any(border_area):
        blurred = cv2.GaussianBlur(out, (5, 5), 0)
        out[border_area] = blurred[border_area]

    # Schrijf naar uitpad (PNG om kwaliteit en transparantie te behouden)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(out_path), out)
    if not success:
        print(f"Kon niet schrijven: {out_path}")
    return success


def find_images(folder: Path):
    exts = {'.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff'}
    files = [p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()]
    files.sort()
    return files


def main():
    p = argparse.ArgumentParser(description='Maak witte, schone achtergronden voor trainingsafbeeldingen')
    p.add_argument('folder', help='Bronmap met afbeeldingen')
    p.add_argument('--out', help='Doelmap (standaard: <folder>_clean)', default=None)
    p.add_argument('--inplace', help='Overschrijf originele bestanden', action='store_true')
    p.add_argument('--white-threshold', type=float, default=40.0, help='Drempelwaarde (afstand tot wit)')
    p.add_argument('--max-blob', type=int, default=800, help='Max grootte (px) van donkere blobs verwijderd uit achtergrond')
    args = p.parse_args()

    src = Path(args.folder)
    if not src.exists() or not src.is_dir():
        print('Bronmap bestaat niet of is geen map:', src)
        sys.exit(1)

    if args.inplace:
        out_root = src
    else:
        out_root = Path(str(src) + '_clean') if args.out is None else Path(args.out)

    images = find_images(src)
    if not images:
        print('Geen afbeeldingsbestanden gevonden in', src)
        sys.exit(0)

    print(f'Bezig met verwerken van {len(images)} afbeeldingen -> {out_root} (white_thresh={args.white_threshold}, max_blob={args.max_blob})')
    processed = 0
    failed = []
    for img in tqdm(images):
        rel = img.name
        out_path = out_root / rel
        if args.inplace:
            # schrijf naar tijdelijke en vervang later
            tmp = out_root / (rel + '.tmp.png')
            ok = process_image(img, tmp, white_thresh=args.white_threshold, max_blob=args.max_blob)
            if ok:
                tmp.replace(img)
                processed += 1
            else:
                failed.append(img)
        else:
            ok = process_image(img, out_path, white_thresh=args.white_threshold, max_blob=args.max_blob)
            if ok:
                processed += 1
            else:
                failed.append(img)

    print(f'Klaar. Verwerkt: {processed}, mislukt: {len(failed)}')
    if failed:
        for f in failed:
            print('Mislukt:', f)


if __name__ == '__main__':
    main()
