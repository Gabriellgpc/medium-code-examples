"""Render markdown tables in article.md to PNGs and emit a Medium-ready article.

Usage:
    uv run python render_tables.py

Inputs:
    article.md
Outputs:
    output/article_images/tables/table_NN.png
    article_medium.md
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
ARTICLE = ROOT / "article.md"
OUT_MD = ROOT / "article_medium.md"
TABLES_DIR = ROOT / "output" / "article_images" / "tables"
TABLES_REL = "output/article_images/tables"


def parse_row(line: str) -> list[str]:
    raw = line.strip()
    if raw.startswith("|"):
        raw = raw[1:]
    if raw.endswith("|"):
        raw = raw[:-1]
    return [c.strip() for c in raw.split("|")]


def parse_alignments(sep_cells: list[str]) -> list[str]:
    aligns = []
    for c in sep_cells:
        c = c.strip()
        left = c.startswith(":")
        right = c.endswith(":")
        if left and right:
            aligns.append("center")
        elif right:
            aligns.append("right")
        else:
            aligns.append("left")
    return aligns


def find_tables(lines: list[str]):
    """Return list of (start_idx, end_idx_inclusive, header, aligns, rows)."""
    tables = []
    sep_re = re.compile(r"^\|[\s\-:|]+\|$")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if line.strip().startswith("|") and i + 1 < len(lines) and sep_re.match(lines[i + 1].rstrip("\n").strip()):
            start = i
            header = parse_row(line)
            aligns = parse_alignments(parse_row(lines[i + 1]))
            j = i + 2
            rows = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(parse_row(lines[j]))
                j += 1
            tables.append((start, j - 1, header, aligns, rows))
            i = j
        else:
            i += 1
    return tables


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_CODE_RE = re.compile(r"`([^`]+)`")


def clean_cell(text: str) -> tuple[str, bool]:
    """Strip markdown formatting from a cell. Returns (plain_text, is_bold)."""
    plain = _LINK_RE.sub(r"\1", text)
    plain = _CODE_RE.sub(r"\1", plain)
    is_bold = bool(_BOLD_RE.search(plain))
    plain = _BOLD_RE.sub(r"\1", plain)
    return plain.strip(), is_bold


def wrap_cell(text: str, max_chars: int) -> list[str]:
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]
    lines = textwrap.wrap(
        text,
        width=max_chars,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return lines or [text]


# Layout constants — measured against DejaVu Sans 11pt at dpi=220.
CHAR_W = 0.095          # inch per character (generous for proportional font)
LINE_H = 0.26           # inch per line of text
H_PAD = 0.18            # inch padding on each side of a cell
V_PAD = 0.14            # inch padding on top/bottom of a cell
MAX_COL_CHARS = 32      # widest column before wrapping kicks in
MIN_COL_CHARS = 5       # minimum column width

HEADER_FACE = "#1f2933"
HEADER_TEXT = "#ffffff"
BORDER = "#dfe3e8"
ALT_FACE = "#f7f8fa"
ROW_FACE = "#ffffff"
BOLD_COLOR = "#0b5fff"
BODY_COLOR = "#222222"
HEADER_FONT = 11
BODY_FONT = 10.5


def render_table(
    header: list[str],
    aligns: list[str],
    rows: list[list[str]],
    output_path: Path,
) -> None:
    n_cols = len(header)

    # Clean every cell and remember bold flags.
    header_clean = [clean_cell(h)[0] for h in header]
    body_clean: list[list[str]] = []
    body_bold: list[list[bool]] = []
    for row in rows:
        padded = list(row) + [""] * (n_cols - len(row))
        cleaned = [clean_cell(c) for c in padded]
        body_clean.append([t for t, _ in cleaned])
        body_bold.append([b for _, b in cleaned])

    # Choose a width per column: longest cell content, capped at MAX_COL_CHARS.
    col_chars: list[int] = []
    for c in range(n_cols):
        longest = len(header_clean[c])
        for row in body_clean:
            longest = max(longest, len(row[c]))
        col_chars.append(max(MIN_COL_CHARS, min(longest, MAX_COL_CHARS)))

    # Wrap any cell that exceeds its column width.
    header_wrapped = [wrap_cell(header_clean[c], col_chars[c]) for c in range(n_cols)]
    body_wrapped: list[list[list[str]]] = [
        [wrap_cell(row[c], col_chars[c]) for c in range(n_cols)]
        for row in body_clean
    ]

    # Derive heights from wrapped line counts.
    header_lines = max(len(c) for c in header_wrapped)
    row_line_counts = [max(len(c) for c in row) for row in body_wrapped]

    col_widths_in = [c * CHAR_W + 2 * H_PAD for c in col_chars]
    header_h_in = header_lines * LINE_H + 2 * V_PAD
    row_heights_in = [n * LINE_H + 2 * V_PAD for n in row_line_counts]

    fig_w = sum(col_widths_in)
    fig_h = header_h_in + sum(row_heights_in)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.invert_yaxis()
    ax.axis("off")

    x_starts = [0.0]
    for w in col_widths_in[:-1]:
        x_starts.append(x_starts[-1] + w)

    def draw_cell(
        x: float,
        y: float,
        w: float,
        h: float,
        text_lines: list[str],
        *,
        face: str,
        text_color: str,
        weight: str,
        fontsize: float,
        align: str,
    ) -> None:
        ax.add_patch(
            patches.Rectangle(
                (x, y), w, h,
                facecolor=face,
                edgecolor=BORDER,
                linewidth=0.6,
            )
        )
        text = "\n".join(text_lines)
        if align == "left":
            tx, ha = x + H_PAD, "left"
        elif align == "right":
            tx, ha = x + w - H_PAD, "right"
        else:
            tx, ha = x + w / 2, "center"
        ax.text(
            tx,
            y + h / 2,
            text,
            ha=ha,
            va="center",
            color=text_color,
            fontsize=fontsize,
            fontweight=weight,
            linespacing=1.15,
        )

    # Header row.
    y = 0.0
    for c in range(n_cols):
        draw_cell(
            x_starts[c],
            y,
            col_widths_in[c],
            header_h_in,
            header_wrapped[c],
            face=HEADER_FACE,
            text_color=HEADER_TEXT,
            weight="bold",
            fontsize=HEADER_FONT,
            align="center",
        )
    y += header_h_in

    # Body rows.
    for r_idx, row_lines in enumerate(body_wrapped):
        h = row_heights_in[r_idx]
        face = ALT_FACE if r_idx % 2 == 0 else ROW_FACE
        for c in range(n_cols):
            is_bold = body_bold[r_idx][c]
            draw_cell(
                x_starts[c],
                y,
                col_widths_in[c],
                h,
                row_lines[c],
                face=face,
                text_color=BOLD_COLOR if is_bold else BODY_COLOR,
                weight="bold" if is_bold else "normal",
                fontsize=BODY_FONT,
                align=aligns[c] if c < len(aligns) else "left",
            )
        y += h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, facecolor="white", pad_inches=0.05, bbox_inches="tight")
    plt.close(fig)


def strip_bold(text: str) -> str:
    return clean_cell(text)[0]


def main() -> None:
    lines = ARTICLE.read_text().splitlines(keepends=True)
    tables = find_tables(lines)
    print(f"Found {len(tables)} tables")

    # Render tables (1-indexed)
    table_paths = []
    for idx, (_s, _e, header, aligns, rows) in enumerate(tables, start=1):
        out = TABLES_DIR / f"table_{idx:02d}.png"
        render_table(header, aligns, rows, out)
        rel = f"{TABLES_REL}/table_{idx:02d}.png"
        table_paths.append(rel)
        print(f"  table_{idx:02d}.png  ({len(rows)} rows x {len(header)} cols)")

    # Rebuild article with tables replaced by image references
    out_lines: list[str] = []
    i = 0
    t_idx = 0
    table_starts = {t[0]: t for t in tables}
    while i < len(lines):
        if i in table_starts:
            _s, e, header, _a, _r = table_starts[i]
            rel = table_paths[t_idx]
            caption = " · ".join(strip_bold(h) for h in header)
            out_lines.append(f"![{caption}]({rel})\n")
            out_lines.append("\n")
            t_idx += 1
            i = e + 1
        else:
            out_lines.append(lines[i])
            i += 1

    # Prepend a short header note listing images to upload
    inline_imgs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", "".join(out_lines))
    header_note = [
        "<!--\n",
        "MEDIUM PASTE NOTES\n",
        "------------------\n",
        "1. Medium's editor does not render markdown tables; every results table in\n",
        "   this file is replaced by a PNG under output/article_images/tables/.\n",
        "2. Drag-and-drop each image into the Medium editor at its location.\n",
        "3. Images to upload (in order of appearance):\n",
    ]
    for p in inline_imgs:
        header_note.append(f"     - {p}\n")
    header_note.append("-->\n\n")

    OUT_MD.write_text("".join(header_note + out_lines))
    print(f"\nWrote {OUT_MD.relative_to(ROOT)}  ({len(inline_imgs)} total images, {len(tables)} are table PNGs)")


if __name__ == "__main__":
    main()
