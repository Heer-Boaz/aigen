"""Isolate the main figure in an image and stack figures with head-count rules."""
from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage


def figures(path, min_frac=0.15):
    im = Image.open(path).convert('RGB')
    a = np.asarray(im).astype(int)
    bg = a[2, 2]
    mask = np.abs(a - bg).sum(2) > 60
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    out = []
    for idx, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        ys, xs = sl
        h = ys.stop - ys.start
        if h < min_frac * a.shape[0]:
            continue
        out.append((im.crop((xs.start, ys.start, xs.stop, ys.stop)), h,
                    int((lab[sl] == idx).sum())))
    out.sort(key=lambda t: -t[2])          # biggest blob first = the figure
    return [(im, h) for im, h, _ in out]


def strip(items, out, H=620):
    ims = []
    for label, im in items:
        w = max(1, int(im.width * H / im.height))
        ims.append((label, im.resize((w, H), Image.NEAREST)))
    W = sum(i.width for _, i in ims) + 30 * len(ims)
    s = Image.new('RGB', (W, H + 24), 'white')
    d = ImageDraw.Draw(s)
    x = 0
    for label, i in ims:
        s.paste(i, (x, 24))
        for k, col in ((4, (255, 0, 0)), (5, (0, 140, 255)), (6, (0, 180, 0))):
            y = 24 + H // k
            d.line([(x, y), (x + i.width, y)], fill=col, width=2)
        d.text((x + 3, 6), label, fill='black')
        x += i.width + 30
    s.save(out)
