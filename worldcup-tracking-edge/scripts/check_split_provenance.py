"""Do SN-GSR and SoccerNet-Tracking share footage?

This matters because the RF-DETR baseline in Step 3 was fine-tuned on
SoccerNet-Tracking and scored on SN-GSR valid. If the two datasets draw from the
same matches and clips, that baseline may have trained on frames it was then
tested on, and its mAP is inflated.

Sequence names cannot answer it — SNMOT-xxx against SNGS-xxx tells you nothing
about the underlying footage. The clip metadata can: SoccerNet-Tracking's
``gameinfo.ini`` carries ``gameID``, ``actionPosition`` and ``clipStart``, which
together identify a specific 30-second window of a specific match.

The SN-GSR annotations live inside multi-gigabyte per-split zips on the Hub, and
downloading them to read a few kilobytes of JSON would be absurd. A zip's central
directory sits at the end of the file and each member can be fetched by byte
range, so this reads the index and pulls only the label files.

    python scripts/check_split_provenance.py --split valid --sample 20
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

import requests

HF = "https://huggingface.co/datasets/SoccerNet/SN-GSR-2025/resolve/main"
EOCD_SIG = b"PK\x05\x06"
EOCD64_LOCATOR_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"


def get_range(url: str, start: int, end: int) -> bytes:
    r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
    r.raise_for_status()
    return r.content


def remote_size(url: str) -> int:
    r = requests.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def read_central_directory(url: str, size: int) -> list[dict]:
    """Parse the zip index, handling zip64 (these archives are well over 4 GB)."""
    tail_len = min(size, 128 * 1024)
    tail = get_range(url, size - tail_len, size - 1)
    idx = tail.rfind(EOCD_SIG)
    if idx < 0:
        raise SystemExit("no end-of-central-directory found")

    cd_size, cd_offset = struct.unpack("<II", tail[idx + 12 : idx + 20])
    loc = tail.rfind(EOCD64_LOCATOR_SIG, 0, idx)
    if loc >= 0 and (cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF):
        eocd64_off = struct.unpack("<Q", tail[loc + 8 : loc + 16])[0]
        head = get_range(url, eocd64_off, eocd64_off + 55)
        if head[:4] == EOCD64_SIG:
            cd_size, cd_offset = struct.unpack("<QQ", head[40:56])

    cd = get_range(url, cd_offset, cd_offset + cd_size - 1)
    entries, pos = [], 0
    while pos + 46 <= len(cd):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            break
        # Bytes 10..46 of a central-directory header: method, time, date, crc,
        # compressed size, uncompressed size, name/extra/comment lengths, disk,
        # internal and external attributes, then the local-header offset.
        (method, _, _, _crc, csize, usize, nlen, elen, clen, _, _, _, lho) = struct.unpack(
            "<HHHIIIHHHHHII", cd[pos + 10 : pos + 46]
        )
        name = cd[pos + 46 : pos + 46 + nlen].decode("utf-8", "replace")
        extra = cd[pos + 46 + nlen : pos + 46 + nlen + elen]

        # zip64 extra field: sizes and the local-header offset move here once any
        # of them exceeds 32 bits, which they do in an 11 GB archive.
        if 0xFFFFFFFF in (csize, usize, lho):
            ep = 0
            while ep + 4 <= len(extra):
                tag, tsize = struct.unpack("<HH", extra[ep : ep + 4])
                if tag == 0x0001:
                    vals = extra[ep + 4 : ep + 4 + tsize]
                    vp = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", vals[vp : vp + 8])[0]
                        vp += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", vals[vp : vp + 8])[0]
                        vp += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", vals[vp : vp + 8])[0]
                    break
                ep += 4 + tsize
        entries.append({"name": name, "method": method, "csize": csize,
                        "usize": usize, "offset": lho})
        pos += 46 + nlen + elen + clen
    return entries


def fetch_member(url: str, entry: dict) -> bytes:
    """Range-read one member and inflate it."""
    header = get_range(url, entry["offset"], entry["offset"] + 29)
    nlen, elen = struct.unpack("<HH", header[26:30])
    data_start = entry["offset"] + 30 + nlen + elen
    raw = get_range(url, data_start, data_start + entry["csize"] - 1)
    if entry["method"] == 0:
        return raw
    return zlib.decompress(raw, -zlib.MAX_WBITS)


def local_tracking_fingerprints(root: Path) -> dict[tuple, str]:
    """(gameID, clipStart) -> sequence name, from SoccerNet-Tracking gameinfo.ini."""
    import configparser

    out = {}
    for ini in sorted(root.rglob("gameinfo.ini")):
        cp = configparser.ConfigParser()
        cp.read(ini)
        sec = cp["Sequence"]
        out[(sec.get("gameID"), sec.get("clipStart"))] = sec.get("name", ini.parent.name)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid")
    ap.add_argument("--sample", type=int, default=20, help="sequences to inspect")
    ap.add_argument("--tracking-root", default="data/soccernet/tracking-2023")
    ap.add_argument("--out", default="output/split_provenance.json")
    args = ap.parse_args()

    url = f"{HF}/{args.split}.zip"
    size = remote_size(url)
    print(f"{args.split}.zip: {size / 1e9:.2f} GB", flush=True)
    entries = read_central_directory(url, size)
    labels = [e for e in entries if e["name"].endswith("Labels-GameState.json")]
    print(f"members: {len(entries)}, label files: {len(labels)}", flush=True)

    picked = labels[: args.sample]
    infos = []
    for e in picked:
        try:
            info = json.loads(fetch_member(url, e).decode("utf-8"))["info"]
        except Exception as exc:  # noqa: BLE001
            print(f"  {e['name']}: {type(exc).__name__}: {exc}", flush=True)
            continue
        infos.append(info)
        print(f"  {info.get('name', e['name'])}: "
              f"game_id={info.get('game_id')} clip_start={info.get('clip_start')} "
              f"action={info.get('action_class')} seq={info.get('seq_length')}", flush=True)

    tracking = local_tracking_fingerprints(Path(args.tracking_root))
    print(f"\nlocal SoccerNet-Tracking sequences on disk: {len(tracking)}")
    for k, v in tracking.items():
        print(f"  gameID={k[0]} clipStart={k[1]} -> {v}")

    gsr_keys = {(str(i.get("game_id")), str(i.get("clip_start"))) for i in infos}
    shared = [k for k in tracking if (str(k[0]), str(k[1])) in gsr_keys]
    print(f"\nexact (gameID, clipStart) matches: {len(shared)}")
    for k in shared:
        print(f"  {tracking[k]} == a {args.split} sequence")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "split": args.split,
        "gsr_sequences_inspected": len(infos),
        "gsr_info_keys": sorted({k for i in infos for k in i}),
        "gsr_infos": infos,
        "local_tracking_fingerprints": {f"{k[0]}|{k[1]}": v for k, v in tracking.items()},
        "exact_matches": [tracking[k] for k in shared],
    }, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
