import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans

# -----------------------------
# Paper-style plotting helpers
# -----------------------------

def set_paper_style():
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.25,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

# Colorblind-friendly palettes (Tableau/ColorBrewer-inspired)
PALETTE_QUADS = {
    'Q1': '#4E79A7',  # blue
    'Q2': '#F28E2B',  # orange
    'Q3': '#59A14F',  # green
    'Q4': '#E15759',  # red
}
PALETTE_IO = {
    'inner': '#76B7B2',
    'outer': '#EDC948',
}
PALETTE_SEQ = ['#4E79A7', '#59A14F', '#F28E2B', '#E15759', '#B07AA1', '#9C755F', '#FF9DA7']


def save_both(fig, out_path: Path):
    out_path = Path(out_path)
    fig.savefig(out_path.with_suffix('.png'), bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.svg'), bbox_inches='tight')

# Output dir for CSVs and plots (shared with location_plot)
OUT_DIR = Path("ucnet_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

 # -----------------------------
# Set camera intrinsic parameters (example values)
fx = 383.41082763671875
fy = 383.41082763671875
ppx = 320.9901123046875
ppy = 242.16427612304688

# -----------------------------
# Location utilities (from location_plot)
# -----------------------------

def to_gray01(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        g = img
    g = g.astype(np.float32)
    if g.max() <= 1.0:
        return np.clip(g, 0, 1)
    return np.clip(g / 255.0, 0, 1)


def binarize(mask_like: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    g = to_gray01(mask_like)
    return (g > thresh).astype(np.uint8)


def compute_placenta_mask_from_img(img: np.ndarray) -> np.ndarray:
    g01 = to_gray01(img)
    return (g01 > 0).astype(np.uint8)


def placenta_centroid(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([ys.mean(), xs.mean()], dtype=np.float32)


def max_radius(mask: np.ndarray, center_rc: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    pts = np.stack([ys, xs], axis=1).astype(np.float32)
    d = np.linalg.norm(pts - center_rc[None, :], axis=1)
    return float(d.max())


def quadrant_of(centroid_rc: np.ndarray, placenta_c_rc: np.ndarray) -> str:
    if centroid_rc[0] < placenta_c_rc[0] and centroid_rc[1] < placenta_c_rc[1]:
        return "Q1"
    if centroid_rc[0] < placenta_c_rc[0] and centroid_rc[1] >= placenta_c_rc[1]:
        return "Q2"
    if centroid_rc[0] >= placenta_c_rc[0] and centroid_rc[1] < placenta_c_rc[1]:
        return "Q3"
    return "Q4"


# -----------------------------
# File matching functions
# -----------------------------
def extract_datetime(filename):
    """
    Extract datetime pattern from filename.
    Expected format: YYYY-MM-DD_HH-MM-SS (e.g. '2023-01-01_12-30-15')
    """
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
    return match.group(1) if match else None


def get_matched_files(gt_dir, depth_dir, mask_dir):
    """
    For the given directories, find files whose names contain a datetime stamp.
    Returns a dictionary keyed by the datetime string with values being a dictionary
    containing the full paths for the GT image, depth CSV, and mask image.
    """
    gt_files = {extract_datetime(f): f for f in os.listdir(gt_dir) if f.endswith('.jpg')}
    mask_files = {extract_datetime(f): f for f in os.listdir(mask_dir) if f.endswith('.jpg')}
    depth_files = {extract_datetime(f): f for f in os.listdir(depth_dir) if f.endswith('.csv')}
    common_keys = set(gt_files.keys()).intersection(depth_files.keys()).intersection(mask_files.keys())
    matched = {
        key: {
            'gt': os.path.join(gt_dir, gt_files[key]),
            'depth': os.path.join(depth_dir, depth_files[key]),
            'mask': os.path.join(mask_dir, mask_files[key])
        }
        for key in common_keys
    }
    return matched


# -----------------------------
# Loading functions
# -----------------------------
def load_gt_image(filepath):
    """
    Loads the GT image in grayscale and binarizes it.
    Returns a binary mask (values 0 or 1).
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # Adjust the threshold as needed; here we assume 127 is a good cutoff.
    _, bin_img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return bin_img


def load_depth_csv(filepath):
    """
    Loads a depth map stored in a CSV file.
    Assumes the CSV contains a 2D array of depth values.
    """
    depth = np.loadtxt(filepath, delimiter=',')
    return depth


# -----------------------------
# 3D conversion helper
# -----------------------------
def pixel_to_3d(u, v, Z, fx, fy, ppx, ppy):
    """
    Back-project a pixel (u,v) with depth Z to 3D using the pinhole model.
    Returns a numpy array [X, Y, Z].
    """
    X = (u - ppx) * Z / fx
    Y = (v - ppy) * Z / fy
    return np.array([X, Y, Z])


# -----------------------------
# Volume estimation function
# -----------------------------
def compute_cc_volumes(gt_bin, depth_map, fx, fy, ppx, ppy):
    """
    For each connected component (CC) in the binary GT mask (gt_bin),
    compute an approximate volume by:
      1. Extracting the pixel indices belonging to the CC.
      2. For each pixel, using its depth from depth_map to back-project to 3D.
      3. Computing the convex hull of the 3D points.
      4. Using the hull volume as the estimated volume.

    Returns:
        volumes: a list of volume values (one per CC, in the same 3D units as depth_map).
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gt_bin, connectivity=8)
    volumes = []
    for label in range(1, num_labels):  # skip background label 0
        # Find pixel indices for this component.
        indices = np.where(labels == label)
        if len(indices[0]) == 0:
            continue
        pts_3d = []
        for i in range(len(indices[0])):
            v = indices[0][i]  # row index
            u = indices[1][i]  # column index
            Z = depth_map[v, u]
            # Skip invalid depth values (assume <= 0 is invalid)
            if Z <= 0:
                continue
            pt = pixel_to_3d(u, v, Z, fx, fy, ppx, ppy)
            pts_3d.append(pt)
        pts_3d = np.array(pts_3d)
        if pts_3d.shape[0] < 4:
            # Not enough points for a 3D hull; volume set to 0.
            volumes.append(0)
        else:
            try:
                hull = ConvexHull(pts_3d)
                volumes.append(hull.volume)
            except Exception as e:
                # If hull computation fails, assign volume 0.
                volumes.append(0)
    return volumes


def compute_cc_volume_half_ellipsoid(gt_bin, depth_map, fx, fy, ppx, ppy):
    """
    For each connected component (CC) ...
    NOTE: The returned list order corresponds to connectedComponents labels 1..num_labels-1.
    using a half 3D ellipsoid model. Outliers are removed in x, y, and z dimensions using an IQR filter.

    Args:
        gt_bin (np.ndarray): Binary GT mask (values 0 or 1).
        depth_map (np.ndarray): 2D array of depth values.
        fx, fy, ppx, ppy: Camera intrinsic parameters.

    Returns:
        volumes (list): A list of volume estimates (one per CC) in the same cubic units as the depth_map.
    """
    # Run connected component analysis on the GT mask.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gt_bin.astype(np.uint8), connectivity=8)
    volumes = []

    for label in range(1, num_labels):  # skip background
        # Get pixel indices for this component.
        indices = np.where(labels == label)
        if len(indices[0]) == 0:
            continue

        # Create an array of points: columns are [x, y, z] where x = column index, y = row index.
        xs = indices[1].astype(np.float32)
        ys = indices[0].astype(np.float32)
        zs = depth_map[indices].astype(np.float32)
        pts = np.stack((xs, ys, zs), axis=1)  # shape: [N, 3]

        # Compute median and IQR for each coordinate.
        med = np.median(pts, axis=0)
        q1 = np.percentile(pts, 25, axis=0)
        q3 = np.percentile(pts, 75, axis=0)
        iqr = q3 - q1
        # Define bounds.
        lower_bound = med - 1.5 * iqr
        upper_bound = med + 1.5 * iqr

        # Filter points that are within bounds in all three dimensions.
        mask = ((pts[:, 0] >= lower_bound[0]) & (pts[:, 0] <= upper_bound[0]) &
                (pts[:, 1] >= lower_bound[1]) & (pts[:, 1] <= upper_bound[1]) &
                (pts[:, 2] >= lower_bound[2]) & (pts[:, 2] <= upper_bound[2]))
        filtered_pts = pts[mask]

        if filtered_pts.shape[0] < 4:
            # Not enough points for a 3D ellipsoid approximation; volume set to 0.
            volumes.append(0)
            continue

        # Compute the new bounding box (in pixel coordinates) from the filtered points.
        x_min, x_max = filtered_pts[:, 0].min(), filtered_pts[:, 0].max()
        y_min, y_max = filtered_pts[:, 1].min(), filtered_pts[:, 1].max()
        # Instead of using the median depth, compute the robust depth range.
        z_min = filtered_pts[:, 2].min()
        z_max = filtered_pts[:, 2].max()
        # Use the thickness of the missing region (in depth) as the full extent in the z-dimension.
        c = (z_max - z_min) / 2.0

        # Convert the bounding box width and height (in pixels) to real-world measurements.
        # For width: (pixel width * Z) / fx, and for height: (pixel height * Z) / fy.
        # Here we use the median depth for width and height conversion.
        Z_med = np.median(filtered_pts[:, 2])
        width_real = ((x_max - x_min) * Z_med) / fx
        height_real = ((y_max - y_min) * Z_med) / fy

        # Define semi-axes for the ellipsoid.
        a = width_real / 2.0  # semi-axis along width
        b = height_real / 2.0  # semi-axis along height
        # c is computed from the robust depth range above.

        # Compute the volume of a half ellipsoid: V = (2/3) * pi * a * b * c.
        volume = (2.0 / 3.0) * np.pi * a * b * c
        volumes.append(volume)

    return volumes
# -----------------------------
# Main function to calculate volumes and create output DataFrame and histogram.
# -----------------------------
def load_mask_image(mask_path):
    """
    Loads the placenta mask image.
    """
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Mask image not found: {mask_path}")
    return img


def check_gt_touching_edge(gt_bin, mask_img, threshold=10):
    """
    Checks if the GT connected component touches or is close to the edge of the placenta mask.

    Instead of only checking if any nonzero pixel in the GT binary image lies exactly on the image boundary,
    this function checks if any nonzero pixel is within 'threshold' pixels from any border.

    Args:
        gt_bin (np.ndarray): Binary GT image.
        mask_img (np.ndarray): Placenta mask image (not used in this implementation but available for extension).
        threshold (int): Number of pixels from the edge to consider as "touching".
                         For threshold=0, it behaves as before (only exact boundary pixels).

    Returns:
        bool: True if any nonzero GT pixel is within 'threshold' pixels of any border, False otherwise.
    """
    if gt_bin is None or mask_img is None:
        return False

    h, w = gt_bin.shape
    # Find coordinates of all nonzero pixels in gt_bin.
    coords = np.column_stack(np.where(gt_bin > 0))
    if coords.size == 0:
        return False

    # Check if any coordinate is within 'threshold' pixels from any edge.
    for r, c in coords:
        if r < threshold or r >= h - threshold or c < threshold or c >= w - threshold:
            return True
    return False

def calculate_volumes(gt_dir, depth_dir, mask_dir, fx, fy, ppx, ppy):
    """
    For each matched file (GT, depth, and mask), compute volumes for each connected component.
    If the GT connected component touches the edge of the placenta (from the mask), then
    adjust the volume (e.g. divide by 2).
    """
    matched_files = get_matched_files(gt_dir, depth_dir, mask_dir)
    records = []
    all_volumes = []
    for dt, files in matched_files.items():
        gt_path = files['gt']
        depth_path = files['depth']
        mask_path = files['mask']

        gt_bin = load_gt_image(gt_path).astype(np.uint8)
        depth_map = load_depth_csv(depth_path)
        mask_img = load_mask_image(mask_path)

        # Placenta region & geometry
        placenta_mask = compute_placenta_mask_from_img(mask_img)
        pc = placenta_centroid(placenta_mask)
        if pc is None:
            continue
        placenta_area = int((placenta_mask > 0).sum())
        r_max = max_radius(placenta_mask, pc)

        # CC labeling (order must match our later volume list)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gt_bin, connectivity=8)

        # Volumes for all CCs (label order 1..num_labels-1)
        volumes_all = compute_cc_volume_half_ellipsoid(gt_bin, depth_map, fx, fy, ppx, ppy)

        # Edge-touch adjustment (global, same as previous behavior)
        touches_edge = check_gt_touching_edge(gt_bin, mask_img, threshold=20)
        if touches_edge:
            volumes_all = [v / 2 for v in volumes_all]

        # Per-CC records
        for label in range(1, num_labels):
            area_px = int(stats[label, cv2.CC_STAT_AREA])
            if area_px == 0:
                continue
            # centroid (OpenCV gives x,y)
            cx, cy = centroids[label]
            cc_rc = np.array([cy, cx], dtype=np.float32)
            rel_size_pct = (area_px / max(placenta_area, 1)) * 100.0
            q = quadrant_of(cc_rc, pc)
            inner_outer = "inner" if np.linalg.norm(cc_rc - pc) < 0.5 * r_max else "outer"
            vol_mm3 = float(volumes_all[label-1]) if (label-1) < len(volumes_all) else 0.0
            vol_cc = vol_mm3 / 1000.0

            records.append({
                'datetime': dt,
                'label': int(label),
                'volume': vol_mm3,
                'volume_cc': vol_cc,
                'quadrant': q,
                'inner_outer': inner_outer,
                'relative_size_pct': float(rel_size_pct),
                'touches_edge': bool(touches_edge)
            })
            all_volumes.append(vol_mm3)
    df = pd.DataFrame(records)
    return df, all_volumes


# -----------------------------
# Example usage
# -----------------------------
if __name__ == '__main__':
    # Set the directories for your GT images and depth CSV files.
    root_dir = Path.cwd()
    depth_dir = root_dir / "Images" / "csv_files"
    gt_dir = root_dir / "Images" / "gt"
    mask_dir = root_dir / "Images" /  "detectron_masked_images"

    df_volumes, volumes_list = calculate_volumes(gt_dir, depth_dir, mask_dir, fx, fy, ppx, ppy)
    set_paper_style()

    # Save enriched per-CC CSV
    csv_out = OUT_DIR / "location_volume_components.csv"
    df_volumes.to_csv(csv_out, index=False)
    print(f"Wrote per-CC CSV → {csv_out}")

    # Standardized volume bins (in cc)
    df_volumes["volume_cc"] = df_volumes["volume"] / 1000.0

    # 1) Quadrant distribution (paper style)
    quad_counts = df_volumes["quadrant"].value_counts().reindex(["Q1","Q2","Q3","Q4"], fill_value=0)
    fig = plt.figure(figsize=(6,4))
    colors = [PALETTE_QUADS[q] for q in quad_counts.index]
    bars = plt.bar(quad_counts.index, quad_counts.values, color=colors)
    plt.xlabel("Quadrant")
    plt.ylabel("Missing Regions (count)")
    plt.title("Distribution by Quadrant")
    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2, y, f"{int(y)}", ha='center', va='bottom')
    plt.tight_layout()
    save_both(fig, OUT_DIR / "quadrant_distribution")
    plt.show()

    plt.close(fig)

    # 2) Inner/Outer distribution (paper style)
    io_counts = df_volumes["inner_outer"].value_counts().reindex(["inner","outer"], fill_value=0)
    fig = plt.figure(figsize=(6,4))
    colors = [PALETTE_IO[k] for k in io_counts.index]
    bars = plt.bar(io_counts.index, io_counts.values, color=colors)
    plt.xlabel("Location")
    plt.ylabel("Missing Regions (count)")
    plt.title("Inner vs Outer")
    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2, y, f"{int(y)}", ha='center', va='bottom')
    plt.tight_layout()
    save_both(fig, OUT_DIR / "inner_outer_distribution")
    plt.show()

    plt.close(fig)

    # 3) Relative size bins (paper style)
    size_bins = [0,1,2,5,10,1000]
    size_labels = ["<1%","1-2%","2-5%","5-10%",">=10%"]
    size_cats = pd.cut(df_volumes["relative_size_pct"], bins=size_bins, labels=size_labels, right=False, include_lowest=True)
    size_counts = size_cats.value_counts().reindex(size_labels, fill_value=0)
    fig = plt.figure(figsize=(8,5))
    colors = PALETTE_SEQ[:len(size_labels)]
    bars = plt.bar(size_counts.index.astype(str), size_counts.values, color=colors, edgecolor='black', linewidth=0.6)
    plt.xlabel("Missing Region Size (% of Placenta Area)")
    plt.ylabel("Missing Regions (count)")
    plt.title("Distribution of Missing Region Sizes")
    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2, y, f"{int(y)}", ha='center', va='bottom')
    plt.tight_layout()
    save_both(fig, OUT_DIR / "relative_size_bins")
    plt.show()

    plt.close(fig)

    # 4) Volume bins (paper style, cc)
    vol_edges = [0, 0.1, 0.5, 1, 2, 5, 10, np.inf]
    vol_labels = ["<0.1","0.1-0.5","0.5-1","1-2","2-5","5-10",">10"]
    vol_cats = pd.cut(df_volumes["volume_cc"], bins=vol_edges, labels=vol_labels, right=False, include_lowest=True)
    vol_counts = vol_cats.value_counts().reindex(vol_labels, fill_value=0)
    fig = plt.figure(figsize=(9,5))
    colors = PALETTE_SEQ[:len(vol_labels)]
    bars = plt.bar(vol_counts.index.astype(str), vol_counts.values, color=colors, edgecolor='black', linewidth=0.6)
    plt.xlabel("Missing Region Volume (cc)")
    plt.ylabel("Missing Regions (count)")
    plt.title("Distribution of Missing Region Volumes")
    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2, y, f"{int(y)}", ha='center', va='bottom')
    plt.tight_layout()
    save_both(fig, OUT_DIR / "volume_bins")
    plt.show()

    plt.close(fig)

    # 5) 2D scatter: Relative size vs Volume (paper style)
    fig = plt.figure(figsize=(6.2,5.2))
    plt.scatter(df_volumes["relative_size_pct"], df_volumes["volume_cc"], s=18, alpha=0.8, edgecolors='none')
    plt.xlabel("Relative Size (% of Placenta Area)")
    plt.ylabel("Volume (cc)")
    plt.title("Relative Size vs Volume per Missing Region")
    plt.tight_layout()
    save_both(fig, OUT_DIR / "relative_size_vs_volume_2d")
    plt.show()

    plt.close(fig)

    volumes_array = np.array(volumes_list).reshape(-1, 1)
    threshold_mm3 = 20 * 1000

    # Separate volumes into small (<= threshold) and big (> threshold)
    small_volumes = [vol for vol in volumes_list if vol <= threshold_mm3]
    big_volumes = [vol for vol in volumes_list if vol > threshold_mm3]

    # Volumes in cc and nice, fixed bins to keep labels compact
    volumes_cc = [vol / 1000.0 for vol in volumes_list if vol > 0]
    bins = [(0, 5), (5, 10), (10, 15), (15, 20)]
    bin_names = [f"{int(b[0])}–{int(b[1])}" for b in bins]  # en dash, no unit per tick
    big_bin_name = "20+"

    # Count the volumes in each bin.
    bin_counts = {name: 0 for name in bin_names}
    big_count = 0

    for vol in volumes_cc:
        if vol < 0:
            continue  # ignore negative (if any)
        if vol < 5:
            bin_counts[bin_names[0]] += 1
        elif vol < 10:
            bin_counts[bin_names[1]] += 1
        elif vol < 15:
            bin_counts[bin_names[2]] += 1
        elif vol < 20:
            bin_counts[bin_names[3]] += 1
        else:
            big_count += 1

    # Combine counts into a list and create final labels.
    final_labels = bin_names + [big_bin_name]
    final_counts = [bin_counts[name] for name in bin_names] + [big_count]

    # Optionally, print out the ranges (fixed for bins, and for big we show min and max of volumes >20 cc)
    big_vols = [vol for vol in volumes_cc if vol >= 20]
    if big_vols:
        big_range = (min(big_vols), max(big_vols))
    else:
        big_range = (0, 0)

    print("Volume bins (in cc):")
    for name in bin_names:
        print(f"{name} cc")
    print(f"{big_bin_name}: range = {big_range[0]:.2f} to {big_range[1]:.2f}")

    # Create a bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(final_labels, final_counts, color='skyblue', edgecolor='black')
    plt.xlabel("Volume Bins (cc)")
    plt.ylabel("Count of Missing Regions")
    plt.title("Counts of Missing Region Volumes (in cc)")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{int(height)}",
                 ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    plt.savefig("volume_bins_bar.png", dpi=150)
    plt.show()
    # =============================
    # Composite 2x2 panel for paper: (a) Quadrants, (b) Inner/Outer, (c) Relative Size, (d) Volume Bins
    # =============================
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Quadrant distribution
    quad_counts = df_volumes["quadrant"].value_counts().reindex(["Q1", "Q2", "Q3", "Q4"], fill_value=0)
    colors_a = [PALETTE_QUADS[q] for q in quad_counts.index]
    bars_a = axes[0, 0].bar(quad_counts.index, quad_counts.values, color=colors_a)
    axes[0, 0].set_xlabel("Quadrant")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("(a) Distribution by Quadrant")
    for b in bars_a:
        y = b.get_height()
        axes[0, 0].text(b.get_x() + b.get_width() / 2, y, f"{int(y)}", ha='center', va='bottom', fontsize=10)

    # (b) Inner/Outer
    io_counts = df_volumes["inner_outer"].value_counts().reindex(["inner", "outer"], fill_value=0)
    colors_b = [PALETTE_IO[k] for k in io_counts.index]
    bars_b = axes[0, 1].bar(io_counts.index, io_counts.values, color=colors_b)
    axes[0, 1].set_xlabel("Location")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("(b) Inner vs Outer")
    for b in bars_b:
        y = b.get_height()
        axes[0, 1].text(b.get_x() + b.get_width() / 2, y, f"{int(y)}", ha='center', va='bottom', fontsize=10)

    # (c) Relative size bins
    size_bins = [0, 1, 2, 5, 10, 1000]
    size_labels = ["<1%", "1-2%", "2-5%", "5-10%", ">=10%"]
    size_cats = pd.cut(df_volumes["relative_size_pct"], bins=size_bins, labels=size_labels, right=False,
                       include_lowest=True)
    size_counts = size_cats.value_counts().reindex(size_labels, fill_value=0)
    colors_c = PALETTE_SEQ[:len(size_labels)]
    bars_c = axes[1, 0].bar(size_counts.index.astype(str), size_counts.values, color=colors_c, edgecolor='black',
                            linewidth=0.6)
    axes[1, 0].set_xlabel("Missing Region Size (% of Placenta Area)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("(c) Missing Region Sizes")
    for b in bars_c:
        y = b.get_height()
        axes[1, 0].text(b.get_x() + b.get_width() / 2, y, f"{int(y)}", ha='center', va='bottom', fontsize=10)

    # (d) Volume bins (user-preferred style)
    bars_d = axes[1, 1].bar(final_labels, final_counts, color='skyblue', edgecolor='black')
    axes[1, 1].set_xlabel("Volume Bins (cc)")
    axes[1, 1].margins(x=0.02)
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("(d) Missing Region Volumes")
    for b in bars_d:
        y = b.get_height()
        axes[1, 1].text(b.get_x() + b.get_width() / 2, y, f"{int(y)}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    panel_path = OUT_DIR / "panel_summary"
    fig.savefig(panel_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(panel_path.with_suffix('.svg'), dpi=300, bbox_inches='tight')
    plt.show()
# if __name__ == '__main__':
#     # ----------------------------
#     # 1. Calculate Volumes
#     # ----------------------------
#     root_dir = Path.cwd()
#     depth_directory = root_dir / "Images" / "csv_files"
#     gt_directory = root_dir / "Images" / "gt"
#     mask_dir = root_dir / "Images" /  "detectron_masked_images"
#
#
#     # Calculate volumes (assume calculate_volumes is defined)
#     df_volumes, volumes_list = calculate_volumes(gt_directory, depth_directory,mask_dir, fx, fy, ppx, ppy)
#     # Convert volumes from mm^3 to cc (1 cc = 1000 mm^3)
#     df_volumes["volume_cc"] = df_volumes["volume"] / 1000.0
#
#     # ----------------------------
#     # 2. Read the previously updated CSV
#     # ----------------------------
#     updated_csv_path = "ucnet/placenta_results/tp_fn_analysis_updated.csv"
#     df_updated = pd.read_csv(updated_csv_path)
#
#     # Extract datetime from the Filename
#     df_updated["datetime"] = df_updated["Filename"].apply(extract_datetime)
#     # Create a ranking for multiple CCs per datetime
#     df_updated["cc_rank"] = df_updated.groupby("datetime").cumcount()
#
#     # ----------------------------
#     # 3. Prepare the volume DataFrame for merging
#     # ----------------------------
#     # Ensure the 'datetime' column in df_volumes is a string
#     df_volumes["datetime"] = df_volumes["datetime"].astype(str)
#     # Create a ranking for volume entries per datetime
#     df_volumes["cc_rank"] = df_volumes.groupby("datetime").cumcount()
#
#     # ----------------------------
#     # 4. Merge Volume Information on datetime and cc_rank
#     # ----------------------------
#     df_merged = pd.merge(
#         df_updated,
#         df_volumes[["datetime", "cc_rank", "volume_cc"]],
#         on=["datetime", "cc_rank"],
#         how="left"
#     )
#
#     # ----------------------------
#     # 5. Save the Final CSV
#     # ----------------------------
#     final_csv_path = updated_csv_path.replace("_updated.csv", "_final.csv")
#     df_merged.to_csv(final_csv_path, index=False, sep="\t")
#     print(f"Final CSV with Volume_cc column saved to: {final_csv_path}")