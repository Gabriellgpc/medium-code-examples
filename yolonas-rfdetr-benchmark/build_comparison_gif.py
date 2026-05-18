"""Assemble output/comparison/compare_*.jpg into a Medium-ready GIF.

Usage:
    uv run python build_comparison_gif.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent
SRC_DIR = ROOT / "output" / "comparison"
OUT_PATH = ROOT / "output" / "comparison.gif"

TARGET_WIDTH = 1080      # Medium displays at <= ~1080 px wide
FRAME_DURATION_MS = 1500  # 1.5 s per frame — slow enough to read FPS overlay
LOOP = 0                  # 0 = infinite


def main() -> None:
    paths = sorted(SRC_DIR.glob("compare_*.jpg"))
    if not paths:
        raise SystemExit(f"No compare_*.jpg files in {SRC_DIR}")

    frames: list[Image.Image] = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if w != TARGET_WIDTH:
            new_h = round(h * TARGET_WIDTH / w)
            im = im.resize((TARGET_WIDTH, new_h), Image.LANCZOS)
        # Convert to a palette image for compact GIF.
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=256))

    frames[0].save(
        OUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=LOOP,
        optimize=True,
        disposal=2,
    )

    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")
    print(f"  frames: {len(frames)}")
    print(f"  size:   {size_mb:.2f} MB")
    print(f"  dims:   {frames[0].size[0]}x{frames[0].size[1]}")
    print(f"  speed:  {FRAME_DURATION_MS} ms / frame")


if __name__ == "__main__":
    main()
