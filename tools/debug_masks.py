"""Sla debug masks op voor één afbeelding om te inspecteren waarom details verdwijnen."""
import sys
from pathlib import Path
import cv2
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python tools/debug_masks.py <image>")
    sys.exit(1)

p = Path(sys.argv[1])
img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
if img is None:
    print('Kon niet lezen', p)
    sys.exit(1)

# Composite to RGB if needed
if img.ndim == 2:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
elif img.shape[2] == 4:
    bgr = img[:, :, :3].astype(np.float32)
    alpha = img[:, :, 3].astype(np.float32) / 255.0
    white = np.ones_like(bgr) * 255.0
    img_rgb = (bgr * alpha[:, :, None] + white * (1.0 - alpha[:, :, None])).astype(np.uint8)
else:
    img_rgb = img[:, :, :3]

h, w = img_rgb.shape[:2]

# parameters (match script)
white_thresh = 30.0
max_blob = 800
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

# compute masks
diff = 255.0 - img_rgb.astype(float)
dist = np.sqrt(np.sum(diff * diff, axis=2))
bg_like = (dist < white_thresh)
mask_non_bg = (~bg_like).astype(np.uint8) * 255
mask_clean = cv2.morphologyEx(mask_non_bg, cv2.MORPH_CLOSE, kernel_med, iterations=2)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel_small, iterations=1)

gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
stroke_mask = (gray < 200).astype(np.uint8) * 255
stroke_mask = cv2.morphologyEx(stroke_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
mask_union = cv2.bitwise_or(mask_clean, stroke_mask)

num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask_union, connectivity=8)
# choose largest
fg_seed = np.zeros((h, w), dtype=np.uint8)
if num_cc > 1:
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_i = 1 + int(np.argmax(areas))
    fg_seed[labels == max_i] = 1

# save debug images
out_dir = p.parent / 'debug_masks'
out_dir.mkdir(exist_ok=True)
cv2.imwrite(str(out_dir / (p.stem + '_mask_non_bg.png')), mask_non_bg)
cv2.imwrite(str(out_dir / (p.stem + '_mask_clean.png')), mask_clean)
cv2.imwrite(str(out_dir / (p.stem + '_stroke_mask.png')), stroke_mask)
cv2.imwrite(str(out_dir / (p.stem + '_mask_union.png')), mask_union)
cv2.imwrite(str(out_dir / (p.stem + '_fg_seed.png')), (fg_seed*255).astype('uint8'))

print('Wrote debug masks to', out_dir)
