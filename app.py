# app.py — Parquet File Architecture Explorer
# Run with:  streamlit run app.py

import streamlit as st
import pyarrow.parquet as pq
import pyarrow as pa
import duckdb
import pandas as pd
import numpy as np
import io, tempfile, os, warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings("ignore")

from diagrams import (
    draw_file_layout, draw_row_group_detail, draw_schema_tree,
    draw_encoding_diagram, draw_footer_diagram, draw_page_anatomy,
    draw_compression_chart,
)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_BYTES = 1 * 100 * 1024 * 1024
st.set_page_config(
    page_title="Parquet File Explorer",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
.nav-group-label {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #9AA0A6;
    padding: 14px 0 4px 0;
    font-family: Inter, sans-serif;
    display: block;
}
.nav-item {
    display: block;
    width: 100%;
    text-align: left;
    padding: 7px 10px;
    border-radius: 6px;
    font-size: 0.84rem;
    font-family: Inter, sans-serif;
    font-weight: 500;
    color: #3C4043;
    background: transparent;
    cursor: pointer;
    text-decoration: none;
    border: none;
    margin-bottom: 1px;
    box-sizing: border-box;
}
.nav-item:hover {
    background: #F1F3F4;
    color: #1A73E8;
    text-decoration: none;
}
.nav-item-active {
    background: #E8F0FE;
    color: #1A73E8;
    font-weight: 600;
}
/* Nav buttons — left aligned, all levels */
[data-testid="stVerticalBlock"] [data-testid="stButton"] > button {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 7px 12px !important;
    border-radius: 6px !important;
    font-size: 0.84rem !important;
    font-family: Inter, sans-serif !important;
    font-weight: 500 !important;
    color: #3C4043 !important;
    width: 100% !important;
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    cursor: pointer !important;
    margin: 2px 0 !important;
    min-height: 0 !important;
    line-height: 1.4 !important;
}
[data-testid="stVerticalBlock"] [data-testid="stButton"] > button:hover {
    background: #F1F3F4 !important;
    color: #1A73E8 !important;
}
/* Streamlit button internals: button > div > p */
[data-testid="stVerticalBlock"] [data-testid="stButton"] > button > div {
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stVerticalBlock"] [data-testid="stButton"] > button > div > p,
[data-testid="stVerticalBlock"] [data-testid="stButton"] > button p {
    text-align: left !important;
    margin: 0 auto 0 0 !important;
    padding: 0 !important;
    font-size: 0.84rem !important;
    font-family: Inter, sans-serif !important;
    font-weight: 500 !important;
    color: inherit !important;
    width: auto !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def bytes_human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024: return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"

ACCENTS = ["#1565C0","#2E7D32","#E65100","#B71C1C",
           "#6A1B9A","#00838F","#F57F17","#880E4F"]
def accent(i): return ACCENTS[i % len(ACCENTS)]

def info_box(html):
    st.markdown(f"<div class='info-box'>{html}</div>", unsafe_allow_html=True)

def section(txt):
    st.markdown(f"<div class='section-header'>{txt}</div>", unsafe_allow_html=True)

def show_fig(fig):
    st.pyplot(fig, width="stretch", clear_figure=True)


# ── Metadata extraction ───────────────────────────────────────────────────────
def extract_metadata(pf):
    meta, schema = pf.metadata, pf.schema_arrow
    info = {
        "num_rows": meta.num_rows, "num_columns": len(schema),
        "num_row_groups": meta.num_row_groups,
        "created_by": meta.created_by or "unknown",
        "format_version": str(meta.format_version),
        "serialized_size": meta.serialized_size,
        "row_groups": [], "columns": [], "schema": schema,
    }
    col_stats = {}
    for rg_idx in range(meta.num_row_groups):
        rg = meta.row_group(rg_idx)
        rg_info = {"index": rg_idx, "num_rows": rg.num_rows,
                   "total_bytes": rg.total_byte_size, "columns": []}
        for col_idx in range(rg.num_columns):
            col   = rg.column(col_idx)
            stats = col.statistics

            def _safe(obj, *attrs, default=None):
                for a in attrs:
                    try:
                        v = getattr(obj, a, None)
                        if v is not None: return v
                    except: pass
                return default

            def _fmt(v):
                if v is None: return None
                if isinstance(v, (bytes, bytearray)):
                    try: return v.decode("utf-8")
                    except: return v.hex()
                if isinstance(v, float): return f"{v:,.6g}"
                if isinstance(v, int):   return f"{v:,}"
                return str(v)

            all_encs = [str(e) for e in col.encodings] if col.encodings else ["UNKNOWN"]
            has_mm   = _safe(stats, "has_min_max", default=False)
            ci = {
                "path": col.path_in_schema, "file_offset": col.file_offset,
                "compressed": col.total_compressed_size,
                "uncompressed": col.total_uncompressed_size,
                "encoding": all_encs[0], "all_encodings": ", ".join(all_encs),
                "compression": str(col.compression),
                "has_dict_page": col.has_dictionary_page,
                "has_stats": stats is not None, "has_min_max": bool(has_mm),
                "min_value": _fmt(_safe(stats, "min", "min_value")) if has_mm else None,
                "max_value": _fmt(_safe(stats, "max", "max_value")) if has_mm else None,
                "null_count":  _safe(stats, "null_count"),
                "num_values":  _safe(stats, "num_values"),
                "distinct_count": _safe(stats, "distinct_count"),
                "physical_type": str(_safe(stats, "physical_type") or ""),
                "logical_type":  str(_safe(stats, "logical_type")  or ""),
            }
            rg_info["columns"].append(ci)
            nm = col.path_in_schema
            if nm not in col_stats:
                col_stats[nm] = {"total_bytes": 0, "encodings": set(), "compressions": set()}
            col_stats[nm]["total_bytes"]    += col.total_compressed_size
            col_stats[nm]["encodings"].add(all_encs[0])
            col_stats[nm]["compressions"].add(str(col.compression))
        info["row_groups"].append(rg_info)
    for name, field in zip([f.name for f in schema], schema):
        info["columns"].append({
            "name": name, "dtype": str(field.type),
            "total_bytes": col_stats.get(name, {}).get("total_bytes", 0),
            "encodings":   list(col_stats.get(name, {}).get("encodings", set())),
            "compressions":list(col_stats.get(name, {}).get("compressions", set())),
        })
    return info


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE RENDERERS
# ═══════════════════════════════════════════════════════════════════════════════

def page_introduction():
    st.markdown("""
<h2 style="font-family:Inter,sans-serif;font-weight:700;color:#1A1A1A;margin-bottom:4px;">
  What is Apache Parquet?</h2>
<p style="color:#5F6368;font-size:0.95rem;margin-bottom:20px;">
  A free, open-source columnar storage format optimised for analytics workloads.</p>
""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, icon, title, color, body in [
        (c1,"📋","Columnar Storage","#1565C0",
         "Data is stored column-by-column. To read 3 of 100 columns, Parquet reads ~3% of the file. Row formats read 100%."),
        (c2,"🗃️","Row Groups","#2E7D32",
         "Files split into horizontal row groups (~128 MB). Each is independently readable with its own statistics."),
        (c3,"⚡","Encode then Compress","#E65100",
         "Values are first encoded (RLE, dictionary, delta), then compressed (Snappy, ZSTD, GZIP). Two-stage reduction."),
    ]:
        col.markdown(f"""
<div class="metric-card" style="border-top:3px solid {color};">
  <div style="font-size:1.4rem;margin-bottom:8px;">{icon}</div>
  <div style="font-weight:600;font-size:0.95rem;color:#1A1A1A;margin-bottom:6px;">{title}</div>
  <div style="font-size:0.875rem;color:#5F6368;line-height:1.6;">{body}</div>
</div>""", unsafe_allow_html=True)

    section("How a Parquet file is read (step by step)")
    info_box("""<ol style="margin:0;padding-left:20px;line-height:2.2;">
      <li>Open file → jump to <b>last 8 bytes</b> → read footer length + PAR1 magic</li>
      <li>Seek back → read <b>footer (FileMetaData)</b>: schema, row group offsets, column stats</li>
      <li><b>Predicate pushdown</b>: skip row groups where min/max stats exclude matching rows</li>
      <li><b>Projection pushdown</b>: seek directly to only the needed column chunks</li>
      <li>Decompress + decode only the selected pages within each column chunk</li>
    </ol>""")

    section("Key terminology")
    t1, t2 = st.columns(2)
    for col, terms in [(t1, [
        ("Row Group",        "Horizontal partition of rows (~128 MB). Unit of parallelism."),
        ("Column Chunk",     "All data for one column within one row group."),
        ("Page",             "Smallest unit (~1 MB) inside a column chunk. Unit of compression."),
        ("Dictionary Page",  "Stores unique values once; data pages store integer indexes."),
        ("Footer",           "Thrift-encoded index at end of file. Contains all metadata."),
    ]), (t2, [
        ("PLAIN",              "Raw bytes. No transformation. Baseline encoding."),
        ("RLE_DICTIONARY",     "Dict indexes + run-length. Best for low-cardinality columns."),
        ("DELTA_BINARY_PACKED","Differences between values + bit-pack. Best for monotone integers."),
        ("Predicate Pushdown", "Skip entire row groups using footer min/max stats."),
        ("Projection Pushdown","Read only needed columns. Columnar layout makes this free."),
    ])]:
        for term, defn in terms:
            col.markdown(f"""
<div style="border-left:3px solid #1A73E8;padding:8px 12px;margin-bottom:8px;
            background:#F8F9FA;border-radius:0 6px 6px 0;">
  <div style="font-weight:600;font-size:0.875rem;color:#1A1A1A;">{term}</div>
  <div style="font-size:0.825rem;color:#5F6368;margin-top:2px;">{defn}</div>
</div>""", unsafe_allow_html=True)


def page_file_layout(info):
    section("Full Parquet File Byte Layout")
    info_box("Every Parquet file starts and ends with the 4-byte magic string <code>PAR1</code>. "
             "Between them: row groups containing column chunks, then Thrift-encoded footer metadata. "
             "The reader always starts from the <b>end of the file</b> — reads the last 8 bytes "
             "to find the footer length, then seeks back to read schema and row group offsets.")
    show_fig(draw_file_layout(info))


def page_row_groups(info):
    section("Row Groups — Horizontal Partitions")
    info_box("The file is split into <b>row groups</b> (~128 MB each). "
             "Each row group holds one <b>column chunk</b> per column. Column chunks are stored "
             "contiguously per column enabling efficient columnar reads. An optional "
             "<b>dictionary page</b> at the front of each chunk stores unique values.")

    n_rgs = info["num_row_groups"]
    max_show = 2
    show_n = min(n_rgs, max_show)

    if n_rgs > max_show:
        st.caption(
            f"⚠️ Diagram shows first {show_n} of {n_rgs} row groups to keep it readable. "
            f"All {n_rgs} row groups are fully listed in the statistics table below "
            f"and in **Footer Deep Dive → Column Statistics per Row Group**."
        )
    else:
        st.caption(f"Showing all {n_rgs} row group(s).")

    show_fig(draw_row_group_detail(
        info, rg_indices=list(range(show_n))))

    section(f"Row Group Statistics — all {n_rgs} row groups")
    st.dataframe(pd.DataFrame([{
        "Row Group":  rg["index"],
        "Rows":       f"{rg['num_rows']:,}",
        "Total Size": bytes_human(rg["total_bytes"]),
        "Columns":    len(rg["columns"]),
        "Avg Col Size": bytes_human(rg["total_bytes"] // len(rg["columns"]) if rg["columns"] else 0),
    } for rg in info["row_groups"]]), width="stretch", hide_index=True)


def page_page_anatomy(info):
    section("Page Anatomy — Smallest Unit Inside a Column Chunk")
    info_box("Each column chunk is split into <b>pages</b> (~1 MB) — the smallest unit of "
             "encoding and compression. A page has: a <b>Thrift-serialized header</b> "
             "(sizes, CRC), optional <b>repetition &amp; definition levels</b> (nested types), "
             "then encoded + compressed values.")
    show_fig(draw_page_anatomy(info))


def page_schema(info):
    section("Schema — Column Names, Types & Storage")
    info_box("The schema is stored in the footer and describes every column: name, logical type "
             "(Arrow), physical type (Parquet), encoding, and compression. All column metadata "
             "is known <b>before reading any data</b>.")
    show_fig(draw_schema_tree(info))
    st.dataframe(pd.DataFrame([{
        "Column": c["name"], "Arrow Type": c["dtype"],
        "Encodings": ", ".join(c["encodings"]),
        "Compression": ", ".join(c["compressions"]),
        "Total Size": bytes_human(c["total_bytes"])
    } for c in info["columns"]]), width="stretch", hide_index=True)


def page_columnar_layout(info):
    section("Columnar Physical Layout — How Columns Are Stored on Disk")
    info_box("Within each row group, <b>all pages of column 0</b> are written first, "
             "then all pages of column 1, and so on. To read one column, the reader seeks "
             "to that column's offset and reads only its bytes — skipping all others entirely.")

    n_cols = min(info["num_columns"], 12)
    n_rgs  = min(info["num_row_groups"], 3)
    fig, ax = plt.subplots(figsize=(14, 2.4 + n_rgs * 1.6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2.4 + n_rgs * 1.6)
    ax.text(7, 2.1 + n_rgs * 1.6,
            "Physical byte layout — left → right = file start → end of each row group",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="#1A1A1A", fontfamily="monospace")
    for ri in range(n_rgs):
        rg    = info["row_groups"][ri]
        y     = 1.6 + (n_rgs - 1 - ri) * 1.6
        cols  = rg["columns"][:n_cols]
        total = sum(c["compressed"] for c in cols) or 1
        x     = 0.3
        ax.text(0.3, y + 1.05, f"ROW GROUP {ri}  ({rg['num_rows']:,} rows  ·  {bytes_human(rg['total_bytes'])})",
                fontsize=9, fontweight="bold", color="#1A1A1A", fontfamily="monospace")
        for ci, col in enumerate(cols):
            bw = max(col["compressed"] / total * 13.4, 0.12)
            cc = accent(ci)
            ax.add_patch(FancyBboxPatch((x, y), bw, 0.72,
                         boxstyle="round,pad=0.02", linewidth=1.5,
                         edgecolor="white", facecolor=cc))
            if bw > 0.5:
                lbl = col["path"][:6] + "…" if len(col["path"]) > 6 else col["path"]
                ax.text(x + bw/2, y + 0.36, lbl, ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold", fontfamily="monospace")
            x += bw
    handles = [mpatches.Patch(color=accent(i), label=info["columns"][i]["name"][:14])
               for i in range(min(n_cols, len(info["columns"])))]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
              ncol=min(n_cols, 6), fontsize=7.5, facecolor="white",
              edgecolor="#E0E0E0", framealpha=1)
    plt.tight_layout(pad=0.4)
    show_fig(fig)

    info_box("<b>Why this matters:</b><br>"
             "• <b>Row format</b> (CSV, JSON): reads every column byte even if you need just 2 columns<br>"
             "• <b>Parquet</b>: seeks directly to the 2 column chunks → reads only their bytes<br>"
             "Each colour block above is one column chunk. They are <b>contiguous per column</b> — "
             "that's what makes the seek possible.")

    section("Column sizes within Row Group 0")
    if info["row_groups"]:
        rg0   = info["row_groups"][0]
        total = sum(c["compressed"] for c in rg0["columns"]) or 1
        st.dataframe(pd.DataFrame([{
            "Column": c["path"],
            "Compressed": bytes_human(c["compressed"]),
            "% of Row Group": f"{c['compressed']/total*100:.1f}%",
            "Uncompressed": bytes_human(c["uncompressed"]),
            "Ratio": f"{c['uncompressed']/c['compressed']:.1f}×" if c["compressed"] else "—",
        } for c in rg0["columns"]]), width="stretch", hide_index=True)


def page_encoding_explainer(info):
    section("Encoding Explainer — How Raw Values Are Transformed Before Compression")
    info_box("Parquet applies an <b>encoding</b> before compression that exploits patterns "
             "in the data. Encoding reduces size structurally; compression reduces it statistically. "
             "Together they achieve much better results than either alone.")

    t1, t2, t3, t4 = st.tabs(["PLAIN","RLE / RLE_DICTIONARY","DELTA_BINARY_PACKED","BYTE_STREAM_SPLIT"])
    with t1:
        st.markdown("#### PLAIN — Raw bytes, no transformation")
        info_box("The baseline. Each value stored in raw binary. "
                 "INT32 = 4 bytes, DOUBLE = 8 bytes, STRING = 4-byte length prefix + bytes.")
        c1, c2 = st.columns(2)
        c1.markdown("**Input**"); c1.code("VendorID: [1, 2, 1, 2, 1, 2, 1, 2]")
        c2.markdown("**PLAIN (INT32 = 4 bytes each)**")
        c2.code("01 00 00 00  02 00 00 00  01 00 00 00 ...\n= 32 bytes for 8 values")
        plain = [c["name"] for c in info["columns"] if "PLAIN" in " ".join(c["encodings"])]
        if plain: st.markdown(f"**Columns using PLAIN in your file:** `{'`, `'.join(plain[:8])}`")

    with t2:
        st.markdown("#### RLE — Run-Length Encoding  |  RLE_DICTIONARY")
        info_box("RLE stores repeated runs as (value, count) pairs. "
                 "RLE_DICTIONARY builds a dictionary of unique values, stores integer indexes, "
                 "then RLE-encodes the indexes. Extremely effective for low-cardinality columns.")
        c1, c2 = st.columns(2)
        c1.markdown("**Input (repeated values)**"); c1.code("[1,1,1,1,2,2,2,1,1,1,1,1]")
        c1.markdown("**RLE encoded**"); c1.code("(1,4),(2,3),(1,5)\n3 pairs vs 12 values → 75% smaller")
        c2.markdown("**Dictionary encoding**")
        c2.code('Dict: {0:"N", 1:"Y"}\nData: [0,0,1,0,0,0,1,1,0,0]\nIndex = 1 bit per value\n→ 87% reduction vs raw string')
        rle = [c["name"] for c in info["columns"] if any("RLE" in e or "DICT" in e for e in c["encodings"])]
        if rle: st.markdown(f"**Columns using RLE/Dictionary:** `{'`, `'.join(rle[:8])}`")
        else:   st.info("No RLE/Dictionary columns detected. File likely uses PLAIN + codec compression.")

    with t3:
        st.markdown("#### DELTA_BINARY_PACKED — Delta + bit-packing for integers")
        info_box("Stores the <b>difference</b> between consecutive values, then bit-packs "
                 "the deltas using the minimum bits needed. Extremely effective for "
                 "monotonically increasing values like timestamps or sequential IDs.")
        c1, c2 = st.columns(2)
        c1.markdown("**Input (unix timestamps ms)**")
        c1.code("[1609459200000,\n 1609459260000,\n 1609459320000,\n 1609459380000]")
        c2.markdown("**Delta encoded**")
        c2.code("First: 1609459200000\nDeltas: [60000, 60000, 60000]\nAll fit in 17 bits\n→ ~85% smaller than INT64")
        int_cols = [c["name"] for c in info["columns"] if "int" in c["dtype"] or "timestamp" in c["dtype"]]
        if int_cols: st.markdown(f"**Integer/timestamp columns:** `{'`, `'.join(int_cols[:6])}`")

    with t4:
        st.markdown("#### BYTE_STREAM_SPLIT — Byte interleaving for floats")
        info_box("Splits float bytes by position across all values, grouping similar bytes together. "
                 "Makes the data more compressible. Best for high-cardinality float/double columns.")
        c1, c2 = st.columns(2)
        c1.markdown("**3 floats, 4 bytes each (hex)**")
        c1.code("[3F 80 00 00]\n[3F 00 00 00]\n[40 00 00 00]")
        c2.markdown("**BYTE_STREAM_SPLIT output**")
        c2.code("Byte-0 stream: 3F 3F 40\nByte-1 stream: 80 00 00\nByte-2 stream: 00 00 00\nByte-3 stream: 00 00 00\n→ better ZSTD compression")
        float_cols = [c["name"] for c in info["columns"] if "float" in c["dtype"] or "double" in c["dtype"]]
        if float_cols: st.markdown(f"**Float/double columns:** `{'`, `'.join(float_cols[:6])}`")

    section("Encoding distribution in your file")
    show_fig(draw_encoding_diagram(info))


def page_compression(info):
    section("Compression Ratios per Column")
    info_box("After encoding, Parquet applies a <b>block compression codec</b> "
             "(Snappy, ZSTD, GZIP, LZ4, Brotli). "
             "Ratio = uncompressed ÷ compressed. A 4× ratio = column uses ¼ of its raw size. "
             "Low-cardinality integers compress best. High-cardinality floats compress worst.")
    show_fig(draw_compression_chart(info))


def page_predicate_pushdown(info):
    section("Predicate Pushdown Simulator")
    info_box("Query engines read <b>footer statistics</b> (min/max per column per row group) "
             "and skip entire row groups without touching their data pages. "
             "This is the single biggest performance feature of Parquet — zero I/O for skipped groups.")

    # Unique column names that have min/max stats in at least one row group
    cols_with_stats = list(dict.fromkeys(
        col["name"] for col in info["columns"]
        if any(
            c["path"] == col["name"] and c.get("has_min_max")
            for rg in info["row_groups"]
            for c in rg["columns"]
        )
    ))

    if not cols_with_stats:
        st.warning("No min/max statistics found in this file's footer. Writer may have disabled statistics.")
        return

    c1, c2, c3 = st.columns([2, 1, 1])
    sel_col  = c1.selectbox("Column for WHERE clause", cols_with_stats)
    operator = c2.selectbox("Operator", [">", ">=", "<", "<=", "=", "!="])
    fval_str = c3.text_input("Value", placeholder="e.g. 50.0")
    if not fval_str:
        info_box("Enter a filter value above to see which row groups would be skipped vs scanned.")
        return

    try:    fval = float(fval_str.replace(",", ""))
    except: fval = fval_str

    results = []
    total_rows = skipped_rows = total_bytes = skipped_bytes = 0
    for rg in info["row_groups"]:
        for c in rg["columns"]:
            if c["path"] != sel_col: continue
            mn, mx    = c.get("min_value"), c.get("max_value")
            total_rows  += rg["num_rows"];  total_bytes  += rg["total_bytes"]
            skipped = False; reason = "Scan — no stats"
            if mn is not None and mx is not None:
                try:
                    mnf, mxf = float(str(mn).replace(",","")), float(str(mx).replace(",",""))
                    if   operator == ">"  and mxf <= fval: skipped, reason = True, f"max={mx} ≤ {fval}"
                    elif operator == ">=" and mxf <  fval: skipped, reason = True, f"max={mx} < {fval}"
                    elif operator == "<"  and mnf >= fval: skipped, reason = True, f"min={mn} ≥ {fval}"
                    elif operator == "<=" and mnf >  fval: skipped, reason = True, f"min={mn} > {fval}"
                    elif operator == "="  and (mxf < fval or mnf > fval):
                        skipped, reason = True, f"[{mn},{mx}] excludes {fval}"
                    elif operator == "!=" and mnf == mxf == fval:
                        skipped, reason = True, f"all values = {fval}"
                    else: reason = f"[{mn}, {mx}] may contain {fval} → SCAN"
                except: reason = f"[{mn}, {mx}] — string compare not simulated"
            if skipped: skipped_rows += rg["num_rows"]; skipped_bytes += rg["total_bytes"]
            results.append({"Row Group": rg["index"], "Rows": f"{rg['num_rows']:,}",
                            "Min": mn or "—", "Max": mx or "—",
                            "Size": bytes_human(rg["total_bytes"]),
                            "Decision": "⏭️ SKIP" if skipped else "🔍 SCAN",
                            "Reason": reason})

    pct_r = skipped_rows/total_rows*100 if total_rows else 0
    pct_b = skipped_bytes/total_bytes*100 if total_bytes else 0
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Skipped", f"{sum(1 for r in results if '⏭️' in r['Decision'])} / {len(results)} groups")
    m2.metric("Rows avoided", f"{skipped_rows:,} ({pct_r:.0f}%)")
    m3.metric("Bytes saved",  bytes_human(skipped_bytes))
    m4.metric("I/O reduction", f"{pct_b:.0f}%")

    df = pd.DataFrame(results)
    st.dataframe(df.style.apply(
        lambda row: ["background-color:#E6F4EA" if "SKIP" in str(row["Decision"])
                     else "background-color:#FCE8E6"] * len(row), axis=1),
        width="stretch", hide_index=True)

    if pct_b > 0:
        st.success(f"WHERE {sel_col} {operator} {fval_str} → skips {pct_b:.0f}% of file bytes "
                   f"({bytes_human(skipped_bytes)} never read from disk).")
    else:
        st.info(f"All row groups must be scanned for WHERE {sel_col} {operator} {fval_str}.")


def page_projection_pushdown(info):
    section("Projection Pushdown — Reading Only the Columns You Need")
    info_box("Because each column is stored contiguously, a query selecting 3 of 20 columns "
             "reads only those 3 column chunks. The reader seeks past all other columns entirely.")

    col_names = [c["name"] for c in info["columns"]]
    selected  = st.multiselect("Select columns your query needs", col_names, default=col_names[:3])
    if not selected: st.warning("Select at least one column."); return

    total_b = sum(c["total_bytes"] for c in info["columns"]) or 1
    sel_b   = sum(c["total_bytes"] for c in info["columns"] if c["name"] in selected)
    skip_b  = total_b - sel_b
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Columns selected", f"{len(selected)} of {len(col_names)}")
    m2.metric("Bytes read",   bytes_human(sel_b))
    m3.metric("Bytes skipped", bytes_human(skip_b))
    m4.metric("I/O used", f"{sel_b/total_b*100:.1f}%")

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FFFFFF")
    names = [c["name"][:16] for c in info["columns"]]
    sizes = [c["total_bytes"]/1024 for c in info["columns"]]
    cols  = ["#1A73E8" if c["name"] in selected else "#E0E0E0" for c in info["columns"]]
    ax.bar(range(len(names)), sizes, color=cols, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8, fontfamily="monospace")
    ax.set_ylabel("Size (KB)", color="#5F6368", fontsize=9)
    ax.set_title("Column sizes — blue = read, grey = skipped by projection pushdown",
                 fontsize=11, fontweight="bold", color="#1A1A1A")
    for sp in ax.spines.values(): sp.set_edgecolor("#E0E0E0")
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.7); ax.set_axisbelow(True)
    plt.tight_layout(); show_fig(fig)


def page_read_cost(info, file_size):
    section("Read Cost Estimator — Parquet vs Row-Based Format")
    info_box("Estimate the I/O cost of different query shapes. "
             "Adjust columns selected and row groups scanned to see how Parquet compares "
             "to a hypothetical row-based format (CSV, JSON).")
    c1, c2 = st.columns(2)
    n_cols = c1.slider("Columns selected", 1, info["num_columns"], min(3, info["num_columns"]))
    n_rgs  = c2.slider("Row groups scanned (after predicate pushdown)",
                       1, info["num_row_groups"], info["num_row_groups"])

    col_frac = n_cols / info["num_columns"]
    rg_frac  = n_rgs  / info["num_row_groups"]
    pq_bytes  = file_size * col_frac * rg_frac
    row_bytes = file_size * rg_frac
    saving    = (row_bytes - pq_bytes) / row_bytes * 100 if row_bytes else 0

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Parquet reads",   bytes_human(pq_bytes))
    m2.metric("Row-based reads", bytes_human(row_bytes))
    m3.metric("Parquet saves",   bytes_human(row_bytes - pq_bytes))
    m4.metric("I/O reduction",   f"{saving:.0f}%")

    fig, ax = plt.subplots(figsize=(9, 3))
    fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FFFFFF")
    bar_vals = [row_bytes/1024**2, pq_bytes/1024**2]
    bar_cols = ["#E0E0E0", "#1A73E8"]
    bar_labs = [f"Row-based\n{bytes_human(row_bytes)}", f"Parquet\n{bytes_human(pq_bytes)}"]
    ax.barh([1,0], bar_vals, color=bar_cols, edgecolor="white", height=0.5)
    ax.set_yticks([0,1]); ax.set_yticklabels(bar_labs, fontsize=11, fontfamily="monospace")
    ax.set_xlabel("MB read from disk", color="#5F6368", fontsize=9)
    for b, v in zip(ax.patches, bar_vals):
        ax.text(b.get_width()+0.3, b.get_y()+b.get_height()/2,
                f"{v:.1f} MB", va="center", fontsize=10, fontweight="bold", fontfamily="monospace")
    for sp in ax.spines.values(): sp.set_edgecolor("#E0E0E0")
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.7); ax.set_axisbelow(True)
    plt.tight_layout(); show_fig(fig)


def page_data_preview(info, tmp_path, duck_df):
    section("Data Preview — First 100 Rows")
    info_box("Actual data read by <b>DuckDB</b> using native Parquet support.")
    con = duckdb.connect()
    df  = con.execute(f"SELECT * FROM read_parquet('{tmp_path}') LIMIT 100").df()
    con.close()
    st.dataframe(df, width="stretch")
    section("Column Profile")
    st.dataframe(pd.DataFrame([{
        "Column": col, "Dtype": str(duck_df[col].dtype),
        "Null % (sample)": round(duck_df[col].isnull().mean()*100,1),
        "Unique (sample)": duck_df[col].nunique()
    } for col in duck_df.columns[:30]]), width="stretch", hide_index=True)


def page_value_distribution(info, tmp_path):
    section("Value Distribution — Column Histograms")
    info_box("Understanding value distribution explains compression ratios. "
             "Low cardinality → dictionary encoding wins. High-cardinality floats → harder to compress.")
    sel = st.selectbox("Select column", [c["name"] for c in info["columns"]])
    con = duckdb.connect()
    try:
        df   = con.execute(f'SELECT "{sel}" FROM read_parquet(\'{tmp_path}\') '
                           f'WHERE "{sel}" IS NOT NULL LIMIT 50000').df()
        vals = df.iloc[:, 0]
        is_num = pd.api.types.is_numeric_dtype(vals)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#F8F9FA")
        if is_num:
            ax1.set_facecolor("#FFFFFF")
            ax1.hist(vals.dropna(), bins=50, color="#1A73E8", edgecolor="white", linewidth=0.5)
            ax1.set_title(f"Distribution — {sel}", fontsize=11, fontweight="bold", color="#1A1A1A")
            ax1.set_xlabel(sel, color="#5F6368", fontsize=9)
            ax1.set_ylabel("Frequency", color="#5F6368", fontsize=9)
            for sp in ax1.spines.values(): sp.set_edgecolor("#E0E0E0")
            ax1.grid(axis="y", color="#E0E0E0", linewidth=0.7); ax1.tick_params(colors="#5F6368")
            ax2.set_facecolor("#FFFFFF")
            ax2.boxplot(vals.dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#E8F0FE", color="#1A73E8"),
                        medianprops=dict(color="#B71C1C", linewidth=2),
                        whiskerprops=dict(color="#5F6368"), capprops=dict(color="#5F6368"),
                        flierprops=dict(marker=".", color="#E65100", alpha=0.4, ms=3))
            ax2.set_title(f"Box plot — {sel}", fontsize=11, fontweight="bold", color="#1A1A1A")
            ax2.set_ylabel("Value", color="#5F6368", fontsize=9)
            for sp in ax2.spines.values(): sp.set_edgecolor("#E0E0E0")
            ax2.tick_params(colors="#5F6368")
        else:
            freq = vals.value_counts().head(20)
            ax1.set_facecolor("#FFFFFF")
            ax1.barh(range(len(freq)), freq.values,
                     color=[accent(i) for i in range(len(freq))], edgecolor="white")
            ax1.set_yticks(range(len(freq)))
            ax1.set_yticklabels([str(v)[:20] for v in freq.index], fontsize=8, fontfamily="monospace")
            ax1.set_xlabel("Count", color="#5F6368", fontsize=9)
            ax1.set_title(f"Top {len(freq)} values — {sel}", fontsize=11, fontweight="bold", color="#1A1A1A")
            for sp in ax1.spines.values(): sp.set_edgecolor("#E0E0E0")
            ax1.grid(axis="x", color="#E0E0E0", linewidth=0.7)
            ax2.set_facecolor("#F8F9FA")
            top5  = vals.value_counts().head(5)
            other = len(vals) - top5.sum()
            pv    = list(top5.values) + ([other] if other > 0 else [])
            pl    = [str(v)[:12] for v in top5.index] + (["Other"] if other > 0 else [])
            ax2.pie(pv, labels=pl, colors=[accent(i) for i in range(len(pl))],
                    autopct="%1.0f%%", startangle=90,
                    wedgeprops=dict(linewidth=2, edgecolor="white"),
                    textprops=dict(fontsize=8))
            ax2.set_title(f"Top 5 share — {sel}", fontsize=11, fontweight="bold", color="#1A1A1A")
        plt.tight_layout(); show_fig(fig)

        stats_d = {"Count": f"{len(vals):,}", "Unique": f"{vals.nunique():,}",
                   "Cardinality": "Low<100" if vals.nunique()<100 else "Med<10k" if vals.nunique()<10000 else "High"}
        if is_num:
            stats_d.update({"Min": f"{vals.min():,.4g}", "Max": f"{vals.max():,.4g}",
                            "Mean": f"{vals.mean():,.4g}", "Nulls": f"{vals.isnull().sum():,}"})
        cols = st.columns(min(len(stats_d), 4))
        for i, (k, v) in enumerate(stats_d.items()):
            cols[i % 4].metric(k, v)

        n_u = vals.nunique()
        if n_u < 100:
            st.success(f"✅ Low cardinality ({n_u} unique) → RLE_DICTIONARY is optimal.")
        elif n_u < 10000:
            st.info(f"ℹ️ Medium cardinality ({n_u} unique) → dictionary may help partially.")
        else:
            st.warning(f"⚠️ High cardinality ({n_u} unique) → PLAIN + codec or DELTA is better than dictionary.")
    except Exception as e:
        st.error(f"Could not load column: {e}")
    finally:
        con.close()


def page_null_heatmap(info, tmp_path):
    section("Null Heatmap — Data Quality Across Row Groups")
    info_box("Shows <b>null percentage per column per row group</b> from footer statistics. "
             "Uneven distribution can indicate partitioning effects or data quality issues. "
             "Null-heavy columns compress better.")
    col_names = [c["name"] for c in info["columns"]]
    rg_labels = [f"RG {rg['index']} ({rg['num_rows']//1000}k)" for rg in info["row_groups"]]
    matrix    = np.zeros((len(col_names), len(info["row_groups"])))
    for ri, rg in enumerate(info["row_groups"]):
        col_map = {c["path"]: c for c in rg["columns"]}
        for ci, cname in enumerate(col_names):
            c = col_map.get(cname)
            if c and c.get("null_count") is not None and c.get("num_values"):
                total = (c["null_count"] or 0) + (c["num_values"] or 0)
                matrix[ci, ri] = (c["null_count"] or 0) / total * 100 if total else 0
    fig, ax = plt.subplots(figsize=(max(8, len(info["row_groups"])*1.5+4),
                                    max(6, len(col_names)*0.45+2)))
    fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#F8F9FA")
    cmap = LinearSegmentedColormap.from_list("nulls", ["#E8F5E9","#FFF9C4","#FFCCBC","#B71C1C"])
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(rg_labels))); ax.set_xticklabels(rg_labels, fontsize=8, fontfamily="monospace")
    ax.set_yticks(range(len(col_names))); ax.set_yticklabels(col_names, fontsize=8, fontfamily="monospace")
    for ci in range(len(col_names)):
        for ri in range(len(info["row_groups"])):
            v = matrix[ci, ri]
            ax.text(ri, ci, f"{v:.0f}%", ha="center", va="center",
                    fontsize=7, color="#1A1A1A" if v < 60 else "white", fontfamily="monospace")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02).set_label("Null %", color="#5F6368", fontsize=9)
    ax.set_title("Null % per column per row group  (green=0%, red=100%)",
                 color="#1A1A1A", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout(); show_fig(fig)


def page_cardinality(info, tmp_path):
    section("Cardinality Analysis — Distinct Values per Column")
    info_box("<b>Cardinality</b> = number of distinct values. "
             "Low cardinality → dictionary encoding is highly efficient. "
             "High cardinality → dictionary grows too large; PLAIN or delta encoding is better.")
    con = duckdb.connect()
    try:
        col_names = [c["name"] for c in info["columns"]]
        exprs     = ", ".join([f'approx_count_distinct("{c}") as "{c}"' for c in col_names])
        result    = con.execute(f"SELECT {exprs} FROM read_parquet('{tmp_path}')").df()
        rows = []
        for c in info["columns"]:
            nd = int(result[c["name"]].iloc[0]) if c["name"] in result.columns else None
            card = ("Very Low <100" if nd and nd<100 else "Low <1k" if nd and nd<1000
                    else "Medium <100k" if nd and nd<100000 else "High")
            best = ("RLE_DICTIONARY" if nd and nd<1000
                    else "DELTA_BINARY_PACKED" if "int" in c["dtype"] or "timestamp" in c["dtype"]
                    else "BYTE_STREAM_SPLIT" if "float" in c["dtype"] or "double" in c["dtype"]
                    else "PLAIN")
            rows.append({
                "Column": c["name"], "Type": c["dtype"],
                "Approx Distinct": f"{nd:,}" if nd else "—",
                "Cardinality": card,
                "Actual Encoding": c["encodings"][0] if c["encodings"] else "—",
                "Recommended": best,
                "Match": "✅" if c["encodings"] and c["encodings"][0]==best else "⚠️",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        dists  = [int(r["Approx Distinct"].replace(",","")) if r["Approx Distinct"] != "—" else 0 for r in rows]
        colors = ["#1A73E8" if d<1000 else "#E65100" if d<100000 else "#B71C1C" for d in dists]
        fig, ax = plt.subplots(figsize=(13, 5))
        fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#FFFFFF")
        ax.bar(range(len(rows)), dists, color=colors, edgecolor="white")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r["Column"][:14] for r in rows], rotation=35, ha="right",
                           fontsize=8, fontfamily="monospace")
        ax.set_ylabel("Approx distinct values", color="#5F6368", fontsize=9)
        ax.set_title("Column Cardinality  (blue=low ✅, orange=medium, red=high)",
                     fontsize=11, fontweight="bold", color="#1A1A1A")
        for sp in ax.spines.values(): sp.set_edgecolor("#E0E0E0")
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.7); ax.set_axisbelow(True)
        plt.tight_layout(); show_fig(fig)
    except Exception as e:
        st.error(f"Could not compute cardinality: {e}")
    finally:
        con.close()


def page_footer(info, file_bytes):
    section("File-Level Metadata")
    _buf     = io.BytesIO(file_bytes)
    meta_obj = pq.ParquetFile(_buf).metadata
    kv_meta  = meta_obj.metadata or {}
    if kv_meta:
        kv_meta = {(k.decode() if isinstance(k,bytes) else k):
                   (v.decode() if isinstance(v,bytes) else v) for k,v in kv_meta.items()}
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Format Version", info["format_version"])
    m2.metric("Total Rows",     f"{info['num_rows']:,}")
    m3.metric("Row Groups",     info["num_row_groups"])
    m4.metric("Footer Size",    bytes_human(info["serialized_size"]))
    st.markdown(f"""<div class="info-box" style="margin-top:8px;">
      <b>Created by:</b> <code>{info['created_by']}</code><br>
      <b>Schema fields:</b> {info['num_columns']} columns<br>
      <b>Key-value metadata entries:</b> {len(kv_meta)}
      {"— <code>"+"</code>, <code>".join(list(kv_meta.keys())[:8])+"</code>" if kv_meta else ""}
    </div>""", unsafe_allow_html=True)
    show_fig(draw_footer_diagram(info))
    with st.expander("Raw Arrow Schema"):
        st.code(str(info["schema"]), language="text")
    if kv_meta:
        with st.expander("Key-Value Metadata"):
            for k,v in kv_meta.items(): st.text(f"{k}: {v[:200]}")

    section("Column Statistics per Row Group")
    info_box("Every column chunk stores statistics in the footer. "
             "Fields: min, max, null count, num values, distinct count, physical type, "
             "logical type, file offset, compression ratio, space saved.")
    dtype_map = {c["name"]: c["dtype"] for c in info["columns"]}
    for rg in info["row_groups"]:
        label = (f"Row Group {rg['index']}  —  {rg['num_rows']:,} rows  ·  "
                 f"{bytes_human(rg['total_bytes'])}  ·  {len(rg['columns'])} columns")
        with st.expander(label, expanded=(rg["index"]==0)):
            stat_rows = []
            for c in rg["columns"]:
                ratio   = f"{c['uncompressed']/c['compressed']:.2f}×" if c["compressed"]>0 else "—"
                savings = f"{(1-c['compressed']/c['uncompressed'])*100:.1f}%" if c["uncompressed"]>0 else "—"
                stat_rows.append({
                    "Column":         c["path"],
                    "Arrow Type":     dtype_map.get(c["path"],"—"),
                    "Physical Type":  c.get("physical_type") or "—",
                    "Logical Type":   c.get("logical_type")  or "—",
                    "Min":            c.get("min_value") or "—",
                    "Max":            c.get("max_value") or "—",
                    "Num Values":     f"{c['num_values']:,}"     if c.get("num_values")     is not None else "—",
                    "Null Count":     f"{c['null_count']:,}"     if c.get("null_count")     is not None else "—",
                    "Distinct Count": f"{c['distinct_count']:,}" if c.get("distinct_count") is not None else "—",
                    "Compressed":     bytes_human(c["compressed"]),
                    "Uncompressed":   bytes_human(c["uncompressed"]),
                    "Ratio":          ratio,
                    "Space Saved":    savings,
                    "All Encodings":  c.get("all_encodings") or "—",
                    "Codec":          c["compression"],
                    "Dict Page":      "✅" if c["has_dict_page"] else "❌",
                    "Has Min/Max":    "✅" if c.get("has_min_max") else "❌",
                    "File Offset":    f"{c['file_offset']:,}" if c.get("file_offset") else "—",
                })
            st.dataframe(pd.DataFrame(stat_rows), width="stretch", hide_index=True,
                         column_config={"Min": st.column_config.TextColumn("Min", width="medium"),
                                        "Max": st.column_config.TextColumn("Max", width="medium")})


# ═══════════════════════════════════════════════════════════════════════════════
# NAV STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
NAV = [
    ("Getting Started", [
        ("intro",        "📖  Introduction"),
    ]),
    ("File Structure", [
        ("file_layout",  "📦  File Layout"),
        ("row_groups",   "🗃️  Row Groups"),
        ("page_anatomy", "📄  Page Anatomy"),
    ]),
    ("Schema & Types", [
        ("schema",       "🌲  Schema Tree"),
    ]),
    ("Storage & Encoding", [
        ("col_layout",   "🔲  Columnar Layout"),
        ("encoding",     "🔣  Encoding Explainer"),
        ("compression",  "🗜️  Compression Ratios"),
    ]),
    ("Data Profiling", [
        ("data",         "🔍  Data Preview"),
        ("distribution", "📊  Value Distribution"),
        ("nulls",        "🔳  Null Heatmap"),
        ("cardinality",  "🔢  Cardinality Analysis"),
    ]),
    ("Query Performance", [
        ("predicate",    "⚡  Predicate Pushdown"),
        ("projection",   "🎯  Projection Pushdown"),
        ("read_cost",    "📐  Read Cost Estimator"),
    ]),
    ("Metadata", [
        ("footer",       "📋  Footer Deep Dive"),
    ]),
]

FILE_SECTIONS = {"file_layout","row_groups","page_anatomy","schema","col_layout",
                 "encoding","compression","predicate","projection","read_cost",
                 "data","distribution","nulls","cardinality","footer"}


def render_header():
    st.markdown("""
<div style="padding:2px 0 12px 0;border-bottom:1px solid #E0E0E0;margin-bottom:16px;">
  <div style="display:flex;align-items:center;justify-content:space-between;">
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="font-size:1.6rem;">🗂️</span>
      <span style="font-family:Inter,sans-serif;font-weight:700;font-size:1.6rem;
                   color:#1A1A1A;letter-spacing:-0.02em;">
        Parquet File Architecture Explorer
      </span>
    </div>
    <div style="font-family:Inter,sans-serif;font-size:0.75rem;color:#9AA0A6;
                text-align:right;white-space:nowrap;">
      Conceived by <strong style="color:#5F6368;"><a href="https://ganeshchandrasekaran.com">Ganesh Chandrasekaran</a></strong>
      &nbsp;·&nbsp; Built by <strong style="color:#5F6368;">Claude (Anthropic)</strong>
    </div>
  </div>
  <p style="font-family:Inter,sans-serif;color:#5F6368;font-size:0.88rem;
            margin:4px 0 0 46px;line-height:1.4;">
    Drop a <code style="background:#F1F3F4;padding:1px 5px;border-radius:4px;
    font-size:0.82rem;">.parquet</code> file to explore its full internal structure.
  </p>
</div>
""", unsafe_allow_html=True)


def render_footer_bar():
    st.markdown("""
<div style="margin-top:48px;padding:16px 0 8px 0;border-top:1px solid #E0E0E0;text-align:center;">
  <span style="font-family:Inter,sans-serif;font-size:0.78rem;color:#9AA0A6;">
    Conceived &amp; designed by <strong style="color:#5F6368;"><a href="https://ganeshchandrasekaran.com">Ganesh Chandrasekaran</a></strong>
    &nbsp;·&nbsp; Built by <strong style="color:#5F6368;">Claude (Anthropic)</strong>
    &nbsp;·&nbsp;
    <code style="font-size:0.74rem;background:#F1F3F4;padding:1px 5px;border-radius:3px;">pyarrow</code>
    <code style="font-size:0.74rem;background:#F1F3F4;padding:1px 5px;border-radius:3px;">duckdb</code>
    <code style="font-size:0.74rem;background:#F1F3F4;padding:1px 5px;border-radius:3px;">matplotlib</code>
  </span>
</div>
""", unsafe_allow_html=True)


def render_metrics(info, file_size):
    for col, label, val, color in zip(st.columns(5), [
        "File Size","Total Rows","Columns","Row Groups","Footer Size"],[
        bytes_human(file_size), f"{info['num_rows']:,}", str(info["num_columns"]),
        str(info["num_row_groups"]), bytes_human(info["serialized_size"])],[
        "#1A73E8","#188038","#E37400","#C5221F","#6A1B9A"]):
        col.markdown(f"""
<div class="metric-card" style="border-top:3px solid {color};">
  <div style="font-size:0.7rem;font-weight:600;color:#5F6368;text-transform:uppercase;
              letter-spacing:0.06em;margin-bottom:8px;font-family:Inter,sans-serif;">{label}</div>
  <div style="font-size:1.8rem;font-weight:700;color:#1A1A1A;font-family:Inter,sans-serif;
              letter-spacing:-0.02em;line-height:1;">{val}</div>
</div>""", unsafe_allow_html=True)


def render_nav(has_file):
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "intro"

    for group, pages in NAV:
        if not has_file and group != "Getting Started":
            continue
        st.markdown(f"<div class='nav-group-label'>{group}</div>",
                    unsafe_allow_html=True)
        for key, label in pages:
            is_active = st.session_state.nav_page == key
            clean = label.strip()
            if is_active:
                # Active: plain styled div, no button needed
                st.markdown(f"""
<div style="text-align:left;background:#E8F0FE;border-radius:6px;
padding:7px 12px;margin:2px 0;font-size:0.84rem;font-family:Inter,sans-serif;
font-weight:600;color:#1A73E8;line-height:1.4;">{clean}</div>
""", unsafe_allow_html=True)
            else:
                # Render the button FIRST, full width, then float label on top
                clicked = st.button(clean, key=f"nav_{key}",
                                    use_container_width=True, help=None)
                if clicked:
                    st.session_state.nav_page = key
                    st.rerun()

    return st.session_state.nav_page


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    render_header()
    st.markdown("""
<div style="font-size:0.8rem;color:#5F6368;font-family:Inter,sans-serif;margin-bottom:6px;">
  📂 Drop your <code style="background:#F1F3F4;padding:1px 5px;border-radius:3px;
  font-size:0.78rem;">.parquet</code> file &nbsp;·&nbsp;
  <span style="color:#E37400;">⚠ Best under 75 MB for speed</span>
  &nbsp;·&nbsp; Max 100 MB accepted
</div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader(" ", type=["parquet"],
                                accept_multiple_files=False,
                                label_visibility="collapsed")
    has_file = uploaded is not None

    nav_col, content_col = st.columns([1, 5])

    with nav_col:
        active = render_nav(has_file)

    with content_col:
        if not has_file:
            page_introduction() if active == "intro" else st.info("Upload a Parquet file to explore this section.")
            render_footer_bar()
            return

        file_bytes = uploaded.read()
        file_size  = len(file_bytes)
        if file_size > MAX_BYTES:
            st.error(f"File too large: {bytes_human(file_size)}. Maximum is 100 MB.")
            return

        with st.spinner("Reading Parquet metadata…"):
            try:
                pf   = pq.ParquetFile(io.BytesIO(file_bytes))
                info = extract_metadata(pf)
            except Exception as e:
                st.error(f"Failed to parse Parquet file: {e}"); return

        tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        tmp.write(file_bytes); tmp.flush(); tmp.close()
        tmp_path = tmp.name

        con     = duckdb.connect()
        duck_df = con.execute(f"SELECT * FROM read_parquet('{tmp_path}') LIMIT 5").df()
        con.close()

        st.markdown(f"<div style='font-size:0.85rem;color:#5F6368;margin-bottom:8px;"
                    f"font-family:Inter,sans-serif;'>📄 <b>{uploaded.name}</b></div>",
                    unsafe_allow_html=True)
        render_metrics(info, file_size)
        st.markdown("<hr style='border:none;border-top:1px solid #E0E0E0;margin:16px 0'>",
                    unsafe_allow_html=True)

        p = active
        if   p == "intro":        page_introduction()
        elif p == "file_layout":  page_file_layout(info)
        elif p == "row_groups":   page_row_groups(info)
        elif p == "page_anatomy": page_page_anatomy(info)
        elif p == "schema":       page_schema(info)
        elif p == "col_layout":   page_columnar_layout(info)
        elif p == "encoding":     page_encoding_explainer(info)
        elif p == "compression":  page_compression(info)
        elif p == "predicate":    page_predicate_pushdown(info)
        elif p == "projection":   page_projection_pushdown(info)
        elif p == "read_cost":    page_read_cost(info, file_size)
        elif p == "data":         page_data_preview(info, tmp_path, duck_df)
        elif p == "distribution": page_value_distribution(info, tmp_path)
        elif p == "nulls":        page_null_heatmap(info, tmp_path)
        elif p == "cardinality":  page_cardinality(info, tmp_path)
        elif p == "footer":       page_footer(info, file_bytes)

        render_footer_bar()
        try: os.unlink(tmp_path)
        except: pass


if __name__ == "__main__":
    main()