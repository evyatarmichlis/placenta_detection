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
# Set camera intrinsic parameters (example values)
fx = 383.41082763671875
fy = 383.41082763671875
ppx = 320.9901123046875
ppy = 242.16427612304688


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


def get_matched_files(gt_dir, depth_dir):
    """
    For the given directories, find files whose names contain a datetime stamp.
    Returns a dictionary keyed by the datetime string with values being a dictionary
    containing the full paths for the GT image and the corresponding depth CSV.
    """
    gt_files = {extract_datetime(f): f for f in os.listdir(gt_dir) if f.endswith('.jpg')}
    depth_files = {extract_datetime(f): f for f in os.listdir(depth_dir) if f.endswith('.csv')}
    common_keys = set(gt_files.keys()).intersection(depth_files.keys())
    matched = {key: {'gt': os.path.join(gt_dir, gt_files[key]),
                     'depth': os.path.join(depth_dir, depth_files[key])}
               for key in common_keys}
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


# -----------------------------
# Main function to calculate volumes and create output DataFrame and histogram.
# -----------------------------
def calculate_volumes(gt_dir, depth_dir, fx, fy, ppx, ppy):
    matched_files = get_matched_files(gt_dir, depth_dir)
    records = []
    all_volumes = []
    for dt, files in matched_files.items():
        gt_path = files['gt']
        depth_path = files['depth']
        gt_bin = load_gt_image(gt_path)
        depth_map = load_depth_csv(depth_path)
        volumes = compute_cc_volumes(gt_bin, depth_map, fx, fy, ppx, ppy)
        for vol in volumes:
            records.append({'datetime': dt, 'volume': vol})
            all_volumes.append(vol)
    df = pd.DataFrame(records)
    return df, all_volumes


# -----------------------------
# Example usage
# -----------------------------
if __name__ == '__main__':
    # Set the directories for your GT images and depth CSV files.
    root_dir = Path.cwd()
    depth_directory = root_dir / "Images" / "csv_files"
    gt_directory = root_dir / "Images" / "gt"

    df_volumes, volumes_list = calculate_volumes(gt_directory, depth_directory, fx, fy, ppx, ppy)
    volumes_array = np.array(volumes_list).reshape(-1, 1)
    threshold_mm3 = 20 * 1000

    # Separate volumes into small (<= threshold) and big (> threshold)
    small_volumes = [vol for vol in volumes_list if vol <= threshold_mm3]
    big_volumes = [vol for vol in volumes_list if vol > threshold_mm3]

    # --- Cluster the small volumes only ---
    n_clusters_small = 4  # number of clusters for the small volumes
    if len(small_volumes) > 0:
        small_volumes_array = np.array(small_volumes).reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters_small, random_state=42)
        kmeans.fit(small_volumes_array)
        labels_small = kmeans.labels_  # Cluster labels for each small volume

        # Get the cluster centers as a flat array and sort them (smallest first)
        cluster_centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(cluster_centers)
        # Create a mapping from original cluster label to a rank-based cluster name.
        # e.g. the cluster with the smallest center becomes "Cluster 1", etc.
        label_to_cluster = {}
        for rank, orig_label in enumerate(sorted_indices):
            label_to_cluster[orig_label] = f'Cluster {rank + 1}'
    else:
        labels_small = np.array([])
        label_to_cluster = {}

    # --- Count and record ranges for the small clusters ---
    small_cluster_counts = {f'Cluster {i + 1}': 0 for i in range(n_clusters_small)}
    small_cluster_ranges = {f'Cluster {i + 1}': [float('inf'), float('-inf')] for i in range(n_clusters_small)}

    for vol, label in zip(small_volumes, labels_small):
        cluster_name = label_to_cluster[label]
        small_cluster_counts[cluster_name] += 1
        small_cluster_ranges[cluster_name][0] = min(small_cluster_ranges[cluster_name][0], vol)
        small_cluster_ranges[cluster_name][1] = max(small_cluster_ranges[cluster_name][1], vol)

    # --- For big volumes, assign them all to a single group "big" ---
    big_cluster_count = len(big_volumes)
    if big_volumes:
        big_range = [min(big_volumes), max(big_volumes)]
    else:
        big_range = [0, 0]

    # --- Combine the results ---
    final_cluster_counts = {}
    final_cluster_ranges = {}

    # Add small clusters (sorted as "Cluster 1", "Cluster 2", …)
    for i in range(n_clusters_small):
        cluster_name = f'Cluster {i + 1}'
        final_cluster_counts[cluster_name] = small_cluster_counts[cluster_name]
        final_cluster_ranges[cluster_name] = small_cluster_ranges[cluster_name]
    # Add the big group.
    final_cluster_counts['big'] = big_cluster_count
    final_cluster_ranges['big'] = big_range

    # --- Create custom labels for the plot ---
    # For small clusters, we convert the mm³ values to cc (divide by 1000) and create a range label.
    custom_labels = []
    for i in range(n_clusters_small):
        cluster_name = f'Cluster {i + 1}'
        low, high = final_cluster_ranges[cluster_name]
        # Convert to cc.
        label_str = f"{low / 1000:.2f}-{high / 1000:.2f}"
        custom_labels.append(label_str)
    # For the big group:
    low_big, high_big = final_cluster_ranges['big']
    big_label = f"{low_big / 1000:.2f}-{high_big / 1000:.2f}"
    custom_labels.append(big_label)

    # --- Print the volume ranges for each group (in cc) ---
    print("Volume cluster ranges (in cc):")
    for i in range(n_clusters_small):
        cluster_name = f'Cluster {i + 1}'
        low, high = final_cluster_ranges[cluster_name]
        print(f"{cluster_name}: range = {low / 1000:.2f} to {high / 1000:.2f}")
    print(f"big: range = {low_big / 1000:.2f} to {high_big / 1000:.2f}")

    # --- Create a bar chart showing the count of missing regions per group ---
    group_names = [f'Cluster {i + 1}' for i in range(n_clusters_small)] + ['big']
    counts = [final_cluster_counts[name] for name in group_names]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(custom_labels, counts, color='skyblue', edgecolor='black')
    plt.xlabel("Volume Range (cc)")
    plt.ylabel("Count of Missing Regions")
    plt.title("Counts of Missing Region Volumes by Cluster Group (in cc)")
    # Annotate each bar with its count.
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{int(height)}",
                 ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig("volume_cluster_bar_30_big_cc_custom_labels.png", dpi=150)
    plt.show()