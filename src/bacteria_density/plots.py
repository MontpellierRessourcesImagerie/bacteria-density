import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_plot(x, y, vertical_lines, colname, out_dir, dpi=150, figsize=(8,4)):
    """
    Plot y vs x, draw vertical lines and save a PNG, then close the figure.
    x, y : 1D numpy arrays (same length). Only finite pairs will be plotted.
    vertical_lines : iterable of x positions where a vertical dashed line is drawn.
    colname : used for ylabel and filename.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return False

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x[valid], y[valid], '-', linewidth=1)
    ax.set_xlabel("Cumulative distance")
    ax.set_ylabel(colname)
    ax.set_title(colname)
    ax.grid(True, linestyle=':', linewidth=0.5)

    ymin, ymax = ax.get_ylim()
    for vx in vertical_lines:
        ax.axvline(x=vx, color='k', linestyle='--', linewidth=1, alpha=0.7)

    # safe filename
    filename = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in colname).strip("_")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{filename}.png")
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def main(csv_path, out_dir="plots", x_column="Cumulative distance"):
    """
    Load CSV preserving blank rows, construct a continuous (summed) X axis from x_column
    across blocks separated by blank rows, draw vertical lines at blank rows, and
    call save_plot for each numeric column (except x_column).
    """
    df = pd.read_csv(csv_path, skip_blank_lines=False)

    if x_column not in df.columns:
        raise ValueError(f"'{x_column}' not found in CSV columns: {list(df.columns)}")

    # detect fully-empty rows (all NaN) -> separators
    blank_mask = df.isna().all(axis=1)

    # build blocks (start inclusive, end exclusive)
    blocks = []
    start = 0
    for i, is_blank in enumerate(blank_mask):
        if is_blank:
            if i > start:
                blocks.append((start, i))
            start = i + 1
    if start < len(df):
        blocks.append((start, len(df)))

    # build continuous x axis
    x_adj = np.full(len(df), np.nan, dtype=float)
    offset = 0.0
    for a, b in blocks:
        block = df.iloc[a:b]
        xvals = pd.to_numeric(block[x_column], errors="coerce").values
        if np.any(np.isfinite(xvals)):
            x_adj[a:b] = xvals + offset
            # last finite value in this block
            for v in xvals[::-1]:
                if np.isfinite(v):
                    offset += float(v)
                    break
        else:
            # nothing to add; offset unchanged; x_adj remains NaN in this block
            pass

    # compute vertical line x positions for each blank row:
    vertical_lines = []
    for i, is_blank in enumerate(blank_mask):
        if not is_blank:
            continue
        # find last finite x_adj before this blank row
        j = i - 1
        pos = None
        while j >= 0:
            if np.isfinite(x_adj[j]):
                pos = float(x_adj[j])
                break
            j -= 1
        if pos is None:
            pos = 0.0
        vertical_lines.append(pos)

    # determine numeric columns to plot (exclude x_column)
    numeric_cols = []
    for col in df.columns:
        if col == x_column:
            continue
        series_num = pd.to_numeric(df[col], errors="coerce")
        if series_num.notna().any():
            numeric_cols.append(col)

    # loop and plot
    for col in numeric_cols:
        y = pd.to_numeric(df[col], errors="coerce").values.astype(float)
        saved = save_plot(x_adj, y, vertical_lines, col, out_dir)
        if saved:
            print("Saved:", col)
        else:
            print("Skipped (no valid data):", col)


# Example usage:
if __name__ == "__main__":
    main(
        "/home/clement/Documents/projects/2119-bacteria-density/small-data/output/id_002/id_002_merged_results.csv", 
        out_dir="/home/clement/Documents/projects/2119-bacteria-density/small-data/output/id_002/plots", 
        x_column="Cumulative distance"
    )
