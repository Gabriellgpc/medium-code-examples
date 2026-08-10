"""Kaggle session step 1: fetch SN-GSR-2025 and build the training annotations.

Paste as notebook cells, or run as a script in a Kaggle notebook.

Design notes, both forced by Kaggle's disk layout:
  * /kaggle/working is 20 GB and is *persisted* as the notebook output;
  * anything else (we use /kaggle/tmp) gets ~60 GB of scratch that is discarded.
So the 32 GB of zips and frames live in scratch, and only the annotations and
checkpoints — a few MB — are written to /kaggle/working. Nothing is uploaded
from a laptop: SN-GSR-2025 is public on the Hub, and Kaggle pulls it at
datacenter speed.

Session limit is 9 h, so treat the download as repeatable, not precious.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

SPLITS = ["train", "valid"]        # add "test" when you need the eval set too
SCRATCH = Path("/kaggle/tmp/gsr")  # ~60 GB, not persisted
OUT = Path("/kaggle/working/gsr")  # 20 GB, persisted as the notebook output
REPO_ID = "SoccerNet/SN-GSR-2025"


def sh(*cmd: str) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def fetch() -> Path:
    """Download the split zips from the Hub and expand them into scratch."""
    from huggingface_hub import hf_hub_download

    SCRATCH.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        target = SCRATCH / split
        if target.exists() and any(target.iterdir()):
            print(f"{split}: already expanded, skipping")
            continue
        print(f"{split}: downloading…", flush=True)
        zip_path = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset",
            filename=f"{split}.zip", local_dir=str(SCRATCH / "_zips"),
        )
        print(f"{split}: expanding…", flush=True)
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)
        # The zip is 9-10 GB; drop it once expanded so scratch does not fill up.
        os.remove(zip_path)
    return SCRATCH


def install_package() -> None:
    """Put soccernet_tracking_edge on the path.

    Source arrives as the private Kaggle Dataset ``soccernet-tracking-edge-src``.
    Kaggle expands an uploaded archive into a folder named after it, so the tree
    lands at ``…/soccernet_tracking_edge_src/src/soccernet_tracking_edge`` rather
    than at ``…/src``. Rather than hard-code either shape, find the package.
    """
    for candidate in Path("/kaggle/input").glob("**/soccernet_tracking_edge/__init__.py"):
        src = candidate.parent.parent          # …/src
        sys.path.insert(0, str(src))
        print(f"using source from {src}")
        return
    # Fallback: the public repo, once these modules are pushed there.
    sh(sys.executable, "-m", "pip", "install", "-q",
       "git+https://github.com/CondadosAI/soccernet-tracking-edge")


def prepare(split_root: Path, split: str) -> dict:
    """Build COCO + pitch lines + ball track for one split."""
    from soccernet_tracking_edge.core import gamestate

    seqs = sorted(p for p in split_root.iterdir() if p.is_dir() and gamestate.is_gamestate(p))
    print(f"{split}: {len(seqs)} sequences")
    out = OUT / split
    stats = gamestate.build_coco(seqs, out / "detection.json", split_root)
    lines = gamestate.build_pitch_lines(seqs, out / "pitch_lines.json", split_root)
    ball = gamestate.build_ball_track(seqs, out / "ball_track.json", split_root)

    sizes = sorted(stats.pop("ball_sizes_px"))
    summary = {
        "split": split, "sequences": len(seqs), **stats,
        "pitch_line_frames": lines["frames"],
        "pitch_line_types": lines["distinct_lines"],
        "ball_visible_pct": ball["ball_visible_pct"],
        "ball_px": {
            "n": len(sizes),
            "median": sizes[len(sizes) // 2] if sizes else None,
            "p10": sizes[int(0.10 * len(sizes))] if sizes else None,
            "p90": sizes[int(0.90 * len(sizes))] if sizes else None,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    install_package()
    root = fetch()
    OUT.mkdir(parents=True, exist_ok=True)
    all_stats = {split: prepare(root / split, split) for split in SPLITS}
    (OUT / "summary.json").write_text(json.dumps(all_stats, indent=2))
    print("\nannotations written to", OUT)
    print("frames stay in scratch:", SCRATCH, "(re-download next session)")
