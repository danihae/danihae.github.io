from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageFilter

SRC = "20210601_10kPa_day28_SCU.lif - Series005.tif"
INTERVAL_S = 0.06855769455432892
OUT = Path("assets/bg.webp")
SIZE = 760
LOOP = 32
DURATION = round(INTERVAL_S * 1000)


def pick_window():
    tif = tifffile.TiffFile(SRC)
    n = len(tif.pages)
    sig = np.empty(n)
    prev = tif.pages[0].asarray().astype(np.float32)
    for i in range(1, n):
        cur = tif.pages[i].asarray().astype(np.float32)
        sig[i] = np.abs(cur - prev).mean()
        prev = cur
    sig[0] = sig[1]
    smooth = np.convolve(sig, np.ones(9) / 9, mode="same")
    best, best_score = 0, np.inf
    for s in range(0, n - LOOP, 4):
        a = tif.pages[s].asarray().astype(np.float32)
        b = tif.pages[s + LOOP].asarray().astype(np.float32)
        score = np.abs(a - b).mean() - smooth[s : s + LOOP].mean()
        if score < best_score:
            best_score, best = score, s
    window = np.stack([tif.pages[i].asarray() for i in range(best, best + LOOP + 1)])
    tif.close()
    print(f"window: frames {best}-{best + LOOP} (score {best_score:.2f})")
    return window


def main():
    window = pick_window().astype(np.float32)
    vmin, vmax = np.percentile(window, [1, 99.7])

    frames = []
    for f in range(len(window)):
        base = np.clip((window[f] - vmin) / (vmax - vmin), 0, 1)
        img = Image.fromarray((base * 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(1.0)).convert("RGB")
        frames.append(img.resize((SIZE, SIZE), Image.LANCZOS))

    Path("assets").mkdir(exist_ok=True)

    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION,
        loop=0,
        quality=48,
        method=6,
    )
    print(f"saved {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {len(frames)} frames @ {DURATION} ms)")


if __name__ == "__main__":
    main()
