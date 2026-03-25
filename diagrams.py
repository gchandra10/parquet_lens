# diagrams.py — all matplotlib diagram functions
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def bytes_human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# DIAGRAM HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
# Strategy: white/near-white box fills with DARK text = always readable.
# Border color identifies the category. No dark fills with colored text.

BG   = "#111318"   # figure background
C_BG = "#FFFFFF"   # box fill  (white)
C_TX = "#111318"   # box text  (near-black)
C_MU = "#555E70"   # muted / sublabel text
C_TT = "#F0F2F8"   # title / axis label text

# 8 distinct, saturated border/accent colors — readable on both white and dark bg
ACCENTS = ["#1565C0","#2E7D32","#E65100","#B71C1C",
           "#6A1B9A","#00838F","#F57F17","#880E4F"]

# Specific semantic colors
A_MAGIC  = "#B71C1C"   # red  – magic bytes / header
A_FOOTER = "#6A1B9A"   # purple – footer
A_DICT   = "#F57F17"   # amber – dictionary page
A_PAGE   = "#00838F"   # teal  – data page
A_RG     = ["#1565C0","#2E7D32","#E65100","#B71C1C","#6A1B9A"]

def _accent(i): return ACCENTS[i % len(ACCENTS)]
def _rg_color(i): return A_RG[i % len(A_RG)]

def _setup(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

def _box(ax, x, y, w, h, label, sublabel=None,
         border="#1565C0", lw=2.0, fontsize=10,
         fill=C_BG, text_color=C_TX, sub_color=C_MU,
         bold=True, zorder=3, radius=0.04):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad={radius}",
                          linewidth=lw, edgecolor=border,
                          facecolor=fill, zorder=zorder)
    ax.add_patch(rect)
    ty = y + h/2 + (0.06 if sublabel else 0)
    ax.text(x + w/2, ty, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color=text_color, fontfamily="monospace", zorder=zorder+1)
    if sublabel:
        ax.text(x + w/2, y + 0.13, sublabel,
                ha="center", va="center", fontsize=7,
                color=sub_color, fontfamily="monospace", zorder=zorder+1)


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 1 — Full File Layout
# ═══════════════════════════════════════════════════════════════════════════════
def draw_file_layout(info):
    n_rg   = info["num_row_groups"]
    show_rg = min(n_rg, 3)
    fig_h  = 3.0 + show_rg * 1.4

    fig, ax = plt.subplots(figsize=(14, fig_h))
    _setup(fig, ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, fig_h)

    ax.text(5, fig_h - 0.25, "Parquet File Layout  (top → bottom = start → end of file)",
            ha="center", va="top", fontsize=13, fontweight="bold",
            color=C_TT, fontfamily="monospace")

    BH   = 0.70   # block height
    GAP  = 0.22
    y    = fig_h - 0.85
    X0, W = 0.6, 8.8

    # ── MAGIC BYTES (start)
    _box(ax, X0, y, W, BH,
         "MAGIC BYTES  ▸  PAR1",
         "4 bytes  •  file signature  •  offset 0",
         border=A_MAGIC, lw=3, fontsize=11)
    ax.text(X0 + W + 0.15, y + BH/2, "START",
            va="center", fontsize=8, fontweight="bold",
            color="white", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc=A_MAGIC, ec="white", lw=1))
    y -= BH + GAP

    # ── Row groups
    for i in range(show_rg):
        rg     = info["row_groups"][i]
        rc     = _rg_color(i)
        _box(ax, X0, y, W, BH,
             f"ROW GROUP {i}",
             f"{rg['num_rows']:,} rows  •  {bytes_human(rg['total_bytes'])}",
             border=rc, lw=2.5, fontsize=11)

        # mini column strips inside the row-group bar
        n_cols  = min(len(rg["columns"]), 8)
        strip_w = (W - 0.3) / n_cols
        for ci in range(n_cols):
            col   = rg["columns"][ci]
            cc    = _accent(ci)
            sx    = X0 + 0.15 + ci * strip_w
            srect = FancyBboxPatch((sx, y + 0.06), strip_w - 0.04, BH * 0.35,
                                   boxstyle="round,pad=0.02", linewidth=1.5,
                                   edgecolor=cc, facecolor=cc, zorder=4)
            ax.add_patch(srect)
            lbl = col["path"][:7] + "…" if len(col["path"]) > 7 else col["path"]
            ax.text(sx + (strip_w-0.04)/2, y + 0.06 + BH*0.175,
                    lbl, ha="center", va="center",
                    fontsize=5.5, color="white",
                    fontfamily="monospace", zorder=5)
        y -= BH + GAP

    if n_rg > show_rg:
        ax.text(5, y + 0.15,
                f"  ⋮  {n_rg - show_rg} more row group(s)  ⋮  ",
                ha="center", va="center", fontsize=9,
                color=C_MU, style="italic", fontfamily="monospace")
        y -= 0.4

    # ── Footer metadata
    _box(ax, X0, y, W, BH,
         "FOOTER METADATA  (FileMetaData — Thrift)",
         f"schema + row-group offsets + stats  •  {bytes_human(info['serialized_size'])}",
         border=A_FOOTER, lw=2.5, fontsize=11)
    y -= BH + GAP

    # ── Footer length + trailing magic
    _box(ax, X0, y, W, BH,
         "FOOTER LENGTH  (4 bytes)   +   MAGIC PAR1  (4 bytes)",
         "last 8 bytes of file — reader entry point",
         border=A_FOOTER, lw=2.5, fontsize=10)
    ax.text(X0 + W + 0.15, y + BH/2, "END",
            va="center", fontsize=8, fontweight="bold",
            color="white", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc=A_FOOTER, ec="white", lw=1))

    plt.tight_layout(pad=0.4)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 2 — Row Group Deep Dive
# ═══════════════════════════════════════════════════════════════════════════════
def draw_row_group_detail(info, rg_indices=(0, 1)):
    n_cols     = min(info["num_columns"], 6)
    n_rgs      = min(len(rg_indices), info["num_row_groups"])
    rg_indices = list(rg_indices[:n_rgs])

    COL_W   = 1.75
    COL_GAP = 0.14
    RG_LBL  = 1.8
    LM      = 0.3
    RG_H    = 4.0
    RG_GAP  = 0.8

    fig_w = LM + RG_LBL + n_cols * (COL_W + COL_GAP) + 0.4
    fig_h = 1.0 + n_rgs * (RG_H + RG_GAP)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _setup(fig, ax)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)

    ax.text(fig_w/2, fig_h - 0.3,
            "Row Group Structure — Column Chunks",
            ha="center", va="top", fontsize=13, fontweight="bold",
            color=C_TT, fontfamily="monospace")

    for ri, rg_idx in enumerate(rg_indices):
        rg      = info["row_groups"][rg_idx]
        rc      = _rg_color(rg_idx)
        y_top   = fig_h - 0.7 - ri * (RG_H + RG_GAP)
        y_bot   = y_top - RG_H

        # Row-group label panel (left)
        _box(ax, LM, y_bot, RG_LBL, RG_H,
             f"ROW GROUP {rg_idx}",
             f"{rg['num_rows']:,} rows",
             border=rc, lw=3, fontsize=10,
             fill="#EEF4FF" if ri==0 else "#F0FFF4")

        # Column chunks
        for ci in range(n_cols):
            cd  = rg["columns"][ci]
            cc  = _accent(ci)
            cx  = LM + RG_LBL + COL_GAP + ci * (COL_W + COL_GAP)

            # Column outer box
            _box(ax, cx, y_bot, COL_W, RG_H, "",
                 border=cc, lw=2.5, fill="#F8F9FA", zorder=2)

            # Column name header stripe
            name = cd["path"][:12] + "…" if len(cd["path"]) > 12 else cd["path"]
            stripe = FancyBboxPatch((cx, y_top - 0.42), COL_W, 0.38,
                                    boxstyle="round,pad=0.02", linewidth=0,
                                    edgecolor=cc, facecolor=cc, zorder=4)
            ax.add_patch(stripe)
            ax.text(cx + COL_W/2, y_top - 0.23, name,
                    ha="center", va="center", fontsize=7.5, fontweight="bold",
                    color="white", fontfamily="monospace", zorder=5)

            # Dict page (amber)
            has_dict = cd.get("has_dict_page", False)
            inner_y  = y_bot + 0.12
            if has_dict:
                _box(ax, cx + 0.1, inner_y, COL_W - 0.2, 0.42,
                     "Dict Page",
                     border=A_DICT, lw=1.8, fontsize=7,
                     fill="#FFF8E1", text_color="#5D4037", zorder=3)
                inner_y += 0.50

            # 3 data pages
            avail_h = (y_top - 0.46) - inner_y - 0.06
            ph      = avail_h / 3.1
            for pi in range(3):
                py = inner_y + pi * (ph + 0.06)
                _box(ax, cx + 0.1, py, COL_W - 0.2, ph,
                     f"Page {pi}",
                     border=cc, lw=1.3, fontsize=7,
                     fill="#EDF3FF" if ci%2==0 else "#F0FFF4",
                     text_color=C_TX, zorder=3)

            # Encoding badge below
            enc       = cd["encoding"]
            enc_short = (enc.replace("DELTA_BINARY_PACKED","DBP")
                            .replace("DELTA_LENGTH_BYTE_ARRAY","DLBA")
                            .replace("RLE_DICTIONARY","RLE+DICT")
                            .replace("PLAIN_DICTIONARY","PLAIN+D"))
            ax.text(cx + COL_W/2, y_bot + 0.06,
                    enc_short, ha="center", va="bottom",
                    fontsize=6.5, color=C_TX, fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.22",
                              fc=cc, ec=cc, lw=0))
            # white text on colored badge
            ax.texts[-1].set_color("white")

    # Legend
    ax.plot([LM], [0.25], "s", color=A_DICT, ms=8)
    ax.text(LM + 0.18, 0.25, "Dict Page", va="center",
            fontsize=8, color=C_TT, fontfamily="monospace")
    ax.plot([LM + 1.5], [0.25], "s", color=A_PAGE, ms=8)
    ax.text(LM + 1.68, 0.25, "Data Pages (encoded + compressed)",
            va="center", fontsize=8, color=C_TT, fontfamily="monospace")

    plt.tight_layout(pad=0.4)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 3 — Schema & Data Types
# ═══════════════════════════════════════════════════════════════════════════════
def draw_schema_tree(info):
    cols  = info["columns"][:16]
    n     = len(cols)
    fig_h = max(5, 1.6 + n * 0.72)

    fig, ax = plt.subplots(figsize=(14, fig_h))
    _setup(fig, ax)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, fig_h)

    ax.text(7, fig_h - 0.3,
            "Schema — Columns, Types & Storage",
            ha="center", va="top", fontsize=13, fontweight="bold",
            color=C_TT, fontfamily="monospace")

    # Root node
    ROOT_X, ROOT_Y = 2.0, fig_h - 0.85
    _box(ax, 0.3, ROOT_Y - 0.28, 3.4, 0.54,
         "schema  (message)",
         border=A_FOOTER, lw=2.5, fontsize=9,
         fill="#EDE7F6", text_color="#4A148C")

    # Column headers
    HY = ROOT_Y - 0.78
    for lbl, xc in [("COLUMN NAME",4.2),("DATA TYPE",7.1),
                     ("ENCODING",9.4),("COMPRESSION",11.3),("SIZE ▶",13.1)]:
        ax.text(xc, HY, lbl, ha="center", va="center", fontsize=7,
                fontweight="bold", color=C_MU, fontfamily="monospace")

    ROW_Y0   = HY - 0.22
    ROW_STEP = 0.68

    for i, col in enumerate(cols):
        cy = ROW_Y0 - i * ROW_STEP
        cc = _accent(i)

        # Connector line
        ax.plot([ROOT_X, ROOT_X, 3.6],
                [ROOT_Y - 0.28, cy, cy],
                color=cc, lw=1.3, alpha=0.6, zorder=1)

        # Column name box — WHITE fill, colored border
        _box(ax, 3.65, cy - 0.20, 2.75, 0.40,
             col["name"][:18] + ("…" if len(col["name"])>18 else ""),
             border=cc, lw=2, fontsize=8.5,
             fill="#FFFFFF", text_color=C_TX)

        # Data type badge — solid color, white text
        ax.text(7.1, cy, col["dtype"],
                ha="center", va="center", fontsize=7.5,
                color="white", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.28",
                          fc="#1565C0", ec="#1565C0", lw=0))

        # Encoding badge
        if col["encodings"]:
            enc   = col["encodings"][0]
            enc_s = (enc.replace("DELTA_BINARY_PACKED","DBP")
                        .replace("DELTA_LENGTH_BYTE_ARRAY","DLBA")
                        .replace("RLE_DICTIONARY","RLE+D")
                        .replace("PLAIN_DICTIONARY","PLN+D"))
            ax.text(9.4, cy, enc_s,
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.28",
                              fc="#2E7D32", ec="#2E7D32", lw=0))

        # Compression badge
        if col["compressions"]:
            ax.text(11.3, cy, col["compressions"][0],
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.28",
                              fc="#E65100", ec="#E65100", lw=0))

        # Size bar
        if col["total_bytes"] > 0:
            max_b = max(c["total_bytes"] for c in cols if c["total_bytes"] > 0)
            blen  = 0.9 * col["total_bytes"] / max_b
            ax.barh(cy, blen, height=0.28, left=12.5,
                    color=cc, alpha=1.0, zorder=2)

    # Alternating row shading
    for i in range(n):
        cy = ROW_Y0 - i * ROW_STEP
        if i % 2 == 1:
            ax.axhspan(cy - ROW_STEP/2, cy + ROW_STEP/2,
                       xmin=3.65/14, xmax=13.5/14,
                       color="#F5F5F5", alpha=0.08, zorder=0)

    plt.tight_layout(pad=0.4)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 4 — Encoding Breakdown
# ═══════════════════════════════════════════════════════════════════════════════
def draw_encoding_diagram(info):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)

    # ── Left: donut chart ─────────────────────────────────────────────────────
    ax1.set_facecolor(BG)
    enc_counts = {}
    for col in info["columns"]:
        for e in col["encodings"]:
            enc_counts[e] = enc_counts.get(e, 0) + 1

    labels = list(enc_counts.keys())
    sizes  = list(enc_counts.values())
    colors = [_accent(i) for i in range(len(labels))]

    wedges, _, autotexts = ax1.pie(
        sizes, labels=None, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.72,
        wedgeprops=dict(linewidth=3, edgecolor=BG, width=0.50)
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontfamily("monospace")

    legend_handles = [mpatches.Patch(color=c, label=l)
                      for c, l in zip(colors, labels)]
    ax1.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9,
               facecolor="#1A1D24", edgecolor="#2E3240",
               labelcolor=C_TT)
    ax1.set_title("Encoding Distribution", color=C_TT, fontsize=12,
                  fontfamily="monospace", fontweight="bold", pad=14)

    # ── Right: horizontal bar chart ───────────────────────────────────────────
    ax2.set_facecolor("#1A1D24")
    cols_s  = sorted(info["columns"],
                     key=lambda c: c["total_bytes"], reverse=True)[:14]
    names   = [c["name"][:14] for c in cols_s]
    sizes_k = [c["total_bytes"] / 1024 for c in cols_s]
    bcolors = [_accent(i) for i in range(len(names))]

    bars = ax2.barh(range(len(names)), sizes_k,
                    color=bcolors, alpha=1.0,
                    edgecolor=BG, linewidth=0.5, height=0.7)

    for bar, val in zip(bars, sizes_k):
        ax2.text(bar.get_width() + max(sizes_k) * 0.01,
                 bar.get_y() + bar.get_height()/2,
                 f"{val:.0f} KB",
                 va="center", fontsize=8,
                 color=C_TT, fontfamily="monospace",
                 fontweight="bold")

    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9,
                        color=C_TT, fontfamily="monospace")
    ax2.set_xlabel("Size (KB)", color=C_MU, fontsize=9, fontfamily="monospace")
    ax2.set_title("Column Storage Size", color=C_TT, fontsize=12,
                  fontfamily="monospace", fontweight="bold", pad=14)
    ax2.tick_params(colors=C_MU)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#2E3240")
    ax2.grid(axis="x", color="#2E3240", linewidth=0.7)
    ax2.set_axisbelow(True)

    plt.tight_layout(pad=1.5)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 5 — Footer Metadata
# ═══════════════════════════════════════════════════════════════════════════════
def draw_footer_diagram(info):
    fig, ax = plt.subplots(figsize=(14, 8))
    _setup(fig, ax)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)

    ax.text(7, 7.75, "Footer — FileMetaData  (Thrift Binary)",
            ha="center", va="top", fontsize=13, fontweight="bold",
            color=C_TT, fontfamily="monospace")

    # Outer border
    outer = FancyBboxPatch((0.3, 0.3), 13.4, 7.1,
                           boxstyle="round,pad=0.12", linewidth=3,
                           edgecolor=A_FOOTER, facecolor="#F3E8FF")
    ax.add_patch(outer)

    # Left panel: fields
    fields = [
        ("version",       str(info["format_version"]),   "#6A1B9A"),
        ("num_rows",      f"{info['num_rows']:,}",        "#1565C0"),
        ("created_by",    info["created_by"][:28],        "#2E7D32"),
        ("schema[]",      f"{info['num_columns']} fields","#00838F"),
        ("row_groups[]",  f"{info['num_row_groups']} groups","#E65100"),
        ("key_value_meta","optional KV pairs",            "#880E4F"),
    ]
    for fi, (fname, fval, fc) in enumerate(fields):
        fy = 6.8 - fi * 0.88
        rect = FancyBboxPatch((0.55, fy - 0.30), 3.5, 0.58,
                              boxstyle="round,pad=0.04", linewidth=2,
                              edgecolor=fc, facecolor="#FFFFFF")
        ax.add_patch(rect)
        ax.text(0.75, fy, fname, va="center", fontsize=9,
                fontweight="bold", color=fc, fontfamily="monospace")
        ax.text(3.95, fy, fval, va="center", ha="right",
                fontsize=8.5, color=C_TX, fontfamily="monospace")

    # Arrow: row_groups → right panel
    ax.annotate("",
                xy=(4.6, 3.6), xytext=(4.1, 3.6),
                arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.15",
                                color="#E65100", lw=2.0))

    # Right panel: row group detail
    ax.text(9.0, 6.9, "Row Group Metadata",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color=C_TX, fontfamily="monospace")

    rg_fields = [
        ("num_rows",       "row count in this group"),
        ("total_byte_size","uncompressed bytes"),
        ("columns[ ]",     "list of ColumnChunkMetaData"),
        ("  file_offset",  "byte offset in file"),
        ("  encodings",    "list of encodings used"),
        ("  codec",        "compression codec"),
        ("  statistics",   "min / max / null_count"),
    ]
    for fi, (fn, fd) in enumerate(rg_fields):
        fy   = 6.4 - fi * 0.72
        fc   = _accent(fi + 2)
        rect = FancyBboxPatch((4.7, fy - 0.25), 8.7, 0.48,
                              boxstyle="round,pad=0.03", linewidth=1.5,
                              edgecolor=fc, facecolor="#FFFFFF")
        ax.add_patch(rect)
        ax.text(4.9, fy, fn, va="center", fontsize=8.5,
                fontweight="bold", color=fc, fontfamily="monospace")
        ax.text(13.3, fy, "→ " + fd, va="center", ha="right",
                fontsize=7.5, color=C_TX, fontfamily="monospace")

    # Footer size callout
    _box(ax, 5.0, 0.5, 3.8, 0.9,
         f"Footer size:  {bytes_human(info['serialized_size'])}",
         "Thrift compact binary  •  end of file",
         border=A_FOOTER, lw=2, fontsize=10,
         fill="#EDE7F6", text_color="#4A148C")

    # Thrift note
    _box(ax, 9.2, 0.5, 4.7, 0.9,
         "Last 8 bytes = footer_len (4B) + PAR1 (4B)",
         "Reader seeks to end, reads length, jumps to footer",
         border="#880E4F", lw=2, fontsize=9,
         fill="#FCE4EC", text_color="#880E4F")

    plt.tight_layout(pad=0.4)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 6 — Page Anatomy
# ═══════════════════════════════════════════════════════════════════════════════
def draw_page_anatomy(info):
    fig, ax = plt.subplots(figsize=(14, 7.5))
    _setup(fig, ax)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)

    ax.text(7, 7.3, "Column Chunk  ▸  Page Anatomy",
            ha="center", va="top", fontsize=13, fontweight="bold",
            color=C_TT, fontfamily="monospace")

    # Left side: structural boxes
    boxes = [
        # (label, sublabel, x, y, w, h, border, fill, text_color)
        ("Dictionary Page",
         "Optional. Stores unique values.\nData pages reference via integer index.",
         0.4, 4.8, 2.8, 1.3,
         A_DICT, "#FFF8E1", "#5D4037"),
        ("Data Page V1 / V2",
         "Contains encoded values,\nrepetition & definition levels.",
         3.6, 4.8, 3.2, 1.3,
         A_PAGE, "#E0F7FA", "#006064"),
        ("Page Header  (Thrift)",
         "type, uncompressed_page_size,\ncompressed_page_size, crc",
         3.6, 3.1, 3.2, 1.3,
         "#1565C0", "#E3F2FD", "#0D47A1"),
        ("Rep Levels",
         "Nested types\n(LIST, MAP...)",
         3.6, 1.3, 0.95, 1.3,
         "#2E7D32", "#E8F5E9", "#1B5E20"),
        ("Def Levels",
         "NULL encoding\nper value",
         4.65, 1.3, 0.95, 1.3,
         "#E65100", "#FBE9E7", "#BF360C"),
        ("Encoded Values",
         "Actual data bytes\nafter encoding + codec",
         5.7, 1.3, 1.1, 1.3,
         "#B71C1C", "#FCE4EC", "#880E4F"),
    ]

    for label, sub, bx, by, bw, bh, bc, bf, btc in boxes:
        _box(ax, bx, by, bw, bh, label, sub,
             border=bc, lw=2.5, fontsize=9.5,
             fill=bf, text_color=btc, sub_color=btc, radius=0.06)

    # Compression wrapper dashed border
    comp = FancyBboxPatch((3.4, 1.1), 3.6, 5.3,
                          boxstyle="round,pad=0.08",
                          linewidth=1.8, edgecolor="#78909C",
                          facecolor="none", linestyle="--", zorder=1)
    ax.add_patch(comp)
    ax.text(5.2, 6.55,
            "compressed with SNAPPY / GZIP / ZSTD / LZ4 / BROTLI",
            ha="center", va="center", fontsize=8, color="#78909C",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3",
                      fc=BG, ec="#78909C", lw=1.2))

    # Arrows
    for (x1,y1),(x2,y2) in [((3.2,5.45),(3.5,5.45)),
                              ((5.2,4.8),(5.2,4.4)),
                              ((5.2,3.1),(5.2,2.6))]:
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.22,head_length=0.14",
                                   color=C_MU, lw=1.5))

    # Right side: encoding reference table
    ax.text(8.3, 6.9, "Key Encodings",
            ha="left", va="top", fontsize=11, fontweight="bold",
            color=C_TT, fontfamily="monospace")

    enc_rows = [
        ("PLAIN",              "#1565C0", "Raw bytes, type-dependent width"),
        ("RLE",                "#2E7D32", "Run-length encoding"),
        ("RLE_DICTIONARY",     "#E65100", "Dict index stored with RLE"),
        ("DELTA_BINARY_PACKED","#B71C1C", "Delta + bit-packing for ints"),
        ("BYTE_STREAM_SPLIT",  "#6A1B9A", "Byte interleave for floats"),
    ]
    for ei, (ename, ec, edesc) in enumerate(enc_rows):
        ey = 6.3 - ei * 0.88
        _box(ax, 8.2, ey - 0.24, 2.6, 0.44,
             ename,
             border=ec, lw=2, fontsize=7.5,
             fill="#FFFFFF", text_color=ec)
        ax.text(11.0, ey, edesc,
                va="center", fontsize=8.5, color=C_TT,
                fontfamily="monospace")

    plt.tight_layout(pad=0.4)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 7 — Compression Ratio
# ═══════════════════════════════════════════════════════════════════════════════
def draw_compression_chart(info):
    col_data = [c for rg in info["row_groups"]
                for c in rg["columns"]
                if c["uncompressed"] > 0 and c["compressed"] > 0]
    agg = {}
    for c in col_data:
        n = c["path"]
        if n not in agg:
            agg[n] = {"comp": 0, "uncomp": 0}
        agg[n]["comp"]   += c["compressed"]
        agg[n]["uncomp"] += c["uncompressed"]
    agg = {k: v for k, v in agg.items() if v["uncomp"] > 0}

    if not agg:
        fig, ax = plt.subplots(figsize=(8, 3))
        _setup(fig, ax)
        ax.text(0.5, 0.5, "No compression statistics available.",
                ha="center", va="center",
                color=C_TT, fontsize=12, transform=ax.transAxes)
        return fig

    names   = list(agg.keys())[:14]
    ratios  = [agg[n]["uncomp"] / agg[n]["comp"] for n in names]
    savings = [(1 - agg[n]["comp"] / agg[n]["uncomp"]) * 100 for n in names]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5),
                                   gridspec_kw={"height_ratios": [3, 1.2]})
    fig.patch.set_facecolor(BG)

    x = np.arange(len(names))

    WHITE = "#FFFFFF"
    LABEL_BG = "#1A1D24"   # dark bg for both axes

    # ── Ratio bars
    ax1.set_facecolor(LABEL_BG)
    bcolors = [_accent(i) for i in range(len(names))]
    bars = ax1.bar(x, ratios, color=bcolors,
                   edgecolor=LABEL_BG, linewidth=0.8, width=0.65, zorder=3)
    ax1.axhline(1.0, color="#EF5350", linewidth=1.5,
                linestyle="--", zorder=4, label="1× (no gain)")
    for bar, r in zip(bars, ratios):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05,
                 f"{r:.1f}×",
                 ha="center", va="bottom", fontsize=8.5,
                 color=WHITE, fontweight="bold", fontfamily="monospace",
                 zorder=5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([n[:13] for n in names], rotation=30, ha="right",
                        fontsize=8.5, color=WHITE, fontfamily="monospace")
    ax1.set_ylabel("Compression Ratio (×)", color=WHITE,
                   fontsize=9, fontfamily="monospace")
    ax1.set_title("Per-Column Compression Ratio & Space Savings",
                  color=WHITE, fontsize=12, fontfamily="monospace",
                  fontweight="bold", pad=10)
    ax1.tick_params(colors=WHITE, labelsize=8, labelcolor=WHITE)
    ax1.legend(fontsize=8, facecolor=LABEL_BG,
               edgecolor="#444", labelcolor=WHITE)
    for sp in ax1.spines.values(): sp.set_edgecolor("#444")
    ax1.grid(axis="y", color="#333", linewidth=0.7, zorder=0)
    ax1.set_axisbelow(True)

    # ── Savings %
    ax2.set_facecolor(LABEL_BG)
    cmap = LinearSegmentedColormap.from_list(
        "sv", ["#EF5350", "#FFA726", "#66BB6A"])
    scolors = [cmap(max(0, min(s, 100)) / 100) for s in savings]
    ax2.bar(x, savings, color=scolors,
            edgecolor=LABEL_BG, linewidth=0.5, zorder=3)
    for xi, sv in zip(x, savings):
        ax2.text(xi, max(sv, 0) + 0.5, f"{sv:.0f}%",
                 ha="center", va="bottom", fontsize=7.5,
                 color=WHITE, fontweight="bold", fontfamily="monospace",
                 zorder=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([])
    ax2.set_ylabel("Savings %", color=WHITE, fontsize=8, fontfamily="monospace")
    ax2.tick_params(colors=WHITE, labelcolor=WHITE)
    for sp in ax2.spines.values(): sp.set_edgecolor("#444")
    ax2.grid(axis="y", color="#333", linewidth=0.7, zorder=0)
    ax2.set_axisbelow(True)

    plt.tight_layout(pad=0.6)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════