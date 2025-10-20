import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
# -----------------------------
# USER CONFIGURATION
# -----------------------------
# Set camera intrinsic parameters
fx = 383.41082763671875
fy = 383.41082763671875
ppx = 320.9901123046875
ppy = 242.16427612304688

ROOT_DIR = Path.cwd()
RGB_DIR = ROOT_DIR / "Images" / "detectron_masked_images"
DEPTH_DIR = ROOT_DIR / "Images" / "csv_files"
GT_DIR = ROOT_DIR / "Images" / "gt"

# ==============================================================================
# >> IMPORTANT <<
# SPECIFY THE FILENAMES YOU WANT TO HIGHLIGHT HERE
# Just provide the base filename, not the full path.
# The script will find any Connected Component from these files.
# ==============================================================================

HIGHLIGHT_FILENAMES = [
    "Real defect maternal_color-image_2025-06-12_07-44-52_masked.jpg",
    "Real defect maternal_color-image_2025-06-12_08-15-21_masked.jpg",
    "Real defect maternal_color-image_2025-06-13_08-17-21_masked.jpg"
]

# ==============================================================================


# -----------------------------
# 1. DATA LOADING & MATCHING
# (No changes in this section)
# -----------------------------
def extract_datetime(filename):
    """Extracts datetime pattern YYYY-MM-DD_HH-MM-SS from a filename."""
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
    return match.group(1) if match else None


def get_matched_files(rgb_dir, gt_dir, depth_dir):
    """Matches RGB, GT, and Depth files based on a common datetime stamp."""
    rgb_files = {extract_datetime(f): f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png'))}
    gt_files = {extract_datetime(f): f for f in os.listdir(gt_dir) if f.endswith('.jpg')}
    depth_files = {extract_datetime(f): f for f in os.listdir(depth_dir) if f.endswith('.csv')}

    common_keys = set(rgb_files.keys()) & set(gt_files.keys()) & set(depth_files.keys())

    matched = {
        key: {
            'rgb': os.path.join(rgb_dir, rgb_files[key]),
            'gt': os.path.join(gt_dir, gt_files[key]),
            'depth': os.path.join(depth_dir, depth_files[key])
        }
        for key in common_keys if key is not None
    }
    print(f"Found {len(matched)} matched sets of RGB, GT, and Depth files.")
    return matched


# -----------------------------
# 2. FEATURE EXTRACTION
# (No changes in this section)
# -----------------------------
def load_and_prep_images(file_paths):
    rgb_img = cv2.imread(file_paths['rgb'])
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    gt_img_gray = cv2.imread(file_paths['gt'], cv2.IMREAD_GRAYSCALE)
    _, gt_binary = cv2.threshold(gt_img_gray, 127, 255, cv2.THRESH_BINARY)
    depth_map = np.loadtxt(file_paths['depth'], delimiter=',')
    gray_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    _, placenta_mask = cv2.threshold(gray_rgb, 5, 255, cv2.THRESH_BINARY)
    return rgb_img, gt_binary, depth_map, placenta_mask


def compute_cc_volume_half_ellipsoid(cc_mask, depth_map, fx, fy):
    indices = np.where(cc_mask > 0)
    if len(indices[0]) < 4: return 0.0
    ys, xs = indices
    zs = depth_map[indices]
    valid_mask = zs > 0
    if np.sum(valid_mask) < 4: return 0.0
    xs, ys, zs = xs[valid_mask], ys[valid_mask], zs[valid_mask]
    x_min, x_max = np.percentile(xs, [5, 95])
    y_min, y_max = np.percentile(ys, [5, 95])
    z_min, z_max = np.percentile(zs, [5, 95])
    median_z = np.median(zs)
    if median_z <= 0: return 0.0
    width_real = ((x_max - x_min) * median_z) / fx
    height_real = ((y_max - y_min) * median_z) / fy
    depth_extent = z_max - z_min
    a, b, c = width_real / 2.0, height_real / 2.0, depth_extent / 2.0
    volume = (2.0 / 3.0) * np.pi * a * b * c
    return volume,width_real,height_real,depth_extent


def extract_features_for_image(file_paths):
    try:
        rgb_img, gt_binary, depth_map, placenta_mask = load_and_prep_images(file_paths)
    except Exception as e:
        print(f"Error loading files for {file_paths['rgb']}: {e}")
        return []

    placenta_area = np.sum(placenta_mask > 0)
    if placenta_area == 0: return []
    M = cv2.moments(placenta_mask)
    if M["m00"] == 0: return []
    placenta_cx, placenta_cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    placenta_centroid = np.array([placenta_cx, placenta_cy])
    placenta_coords = np.argwhere(placenta_mask > 0)[:, ::-1]
    distances = np.linalg.norm(placenta_coords - placenta_centroid, axis=1)
    max_dist_from_center = np.max(distances) if len(distances) > 0 else 1.0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gt_binary, connectivity=8)

    features_list = []
    for i in range(1, num_labels):
        cc_mask = (labels == i).astype(np.uint8) * 255
        cc_coords = np.where(cc_mask > 0)
        volume,width,height,depth = compute_cc_volume_half_ellipsoid(cc_mask, depth_map, fx, fy)
        cc_area_2d = stats[i, cv2.CC_STAT_AREA]
        relative_area_2d = (cc_area_2d / placenta_area) * 100.0
        cc_centroid = centroids[i]
        dist = np.linalg.norm(cc_centroid - placenta_centroid)
        normalized_distance = dist / max_dist_from_center if max_dist_from_center > 0 else 0
        cc_pixels = rgb_img[cc_coords]
        mean_color = np.mean(cc_pixels, axis=0) if len(cc_pixels) > 0 else [0, 0, 0]
        std_color = np.std(cc_pixels, axis=0) if len(cc_pixels) > 0 else [0, 0, 0]

        features_list.append({
            'filepath': file_paths['rgb'], 'cc_label': i, 'cc': volume/1000,
            'relative_area_pct': relative_area_2d, 'norm_dist_from_center': normalized_distance,
            'mean_r': mean_color[0], 'mean_g': mean_color[1], 'mean_b': mean_color[2],
            'std_r': std_color[0], 'std_g': std_color[1], 'std_b': std_color[2],
            'width':width,"height":height,"depth":depth
        })
    return features_list


# -------------------------------------
# 3. VISUALIZATION (MODIFIED)
# -------------------------------------

def visualize_highlighted_samples(tsne_results, df):
    """
    Creates a scatter plot of t-SNE results, highlighting specific samples.
    Args:
        tsne_results (np.ndarray): The 2D coordinates from t-SNE.
        df (pd.DataFrame): The dataframe containing the 'is_highlighted' column.
    """
    if 'is_highlighted' not in df.columns:
        print("Error: 'is_highlighted' column not found in DataFrame.")
        return

    # Separate the data into normal and highlighted groups
    normal_indices = df[~df['is_highlighted']].index
    highlight_indices = df[df['is_highlighted']].index

    plt.figure(figsize=(14, 12))

    # Plot normal samples first (smaller, semi-transparent)
    plt.scatter(
        tsne_results[normal_indices, 0],
        tsne_results[normal_indices, 1],
        c='cornflowerblue',
        s=30,
        alpha=0.5,
        label=f'Normal Samples (n={len(normal_indices)})'
    )

    # Plot highlighted samples on top (larger, opaque, with an edge)
    if len(highlight_indices) > 0:
        plt.scatter(
            tsne_results[highlight_indices, 0],
            tsne_results[highlight_indices, 1],
            c='red',
            s=150,
            alpha=1.0,
            label=f'Highlighted Samples (n={len(highlight_indices)})',
            edgecolors='black'
        )

    plt.title('t-SNE Visualization of Dataset with Highlighted Samples')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def visualize_pca_biplot(pca_results, pca_model, df, feature_cols):
    """
    Creates a PCA biplot showing data points and feature vectors.
    This is key to understanding what the principal components represent.
    """
    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot the data points (scores)
    normal_indices = df[~df['is_highlighted']].index
    highlight_indices = df[df['is_highlighted']].index

    ax.scatter(pca_results[normal_indices, 0], pca_results[normal_indices, 1], c='cornflowerblue', s=30, alpha=0.5,
               label=f'Normal Samples (n={len(normal_indices)})')
    if len(highlight_indices) > 0:
        ax.scatter(pca_results[highlight_indices, 0], pca_results[highlight_indices, 1], c='red', s=150, alpha=1.0,
                   label=f'Highlighted Samples (n={len(highlight_indices)})', edgecolors='black')


    explained_var = pca_model.explained_variance_ratio_
    ax.set_xlabel(f'Principal Component 1 ({explained_var[0]:.1%} variance)', fontsize=14)
    ax.set_ylabel(f'Principal Component 2 ({explained_var[1]:.1%} variance)', fontsize=14)
    ax.set_title('PCA Biplot of Missing Part Features', fontsize=16)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_feature_distributions(df, feature_cols):
    """
    For each feature, plots the distribution for all samples and overlays
    the distribution for highlighted samples.
    """
    print("\n--- Generating Feature Distribution Plots ---")

    highlighted_df = df[df['is_highlighted']]
    if highlighted_df.empty:
        print("No highlighted samples to plot. Skipping distribution plots.")
        return

    # Determine grid size for subplots
    num_features = len(feature_cols)
    cols = 3
    rows = (num_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = axes.flatten()  # Flatten to 1D array for easy iteration

    for i, col in enumerate(feature_cols):
        ax = axes[i]

        # Plot the distribution for ALL samples
        sns.histplot(df[col], ax=ax, color="cornflowerblue", label="All Samples", stat="density", kde=True)

        # Overlay the distribution for HIGHLIGHTED samples
        # Using a Kernel Density Estimate (KDE) plot for the highlight group is often clearer than a histogram
        # if there are few samples. We also add a "rug" plot to show exact values.
        sns.kdeplot(highlighted_df[col], ax=ax, color="red", label="Highlighted", lw=3)
        sns.rugplot(highlighted_df[col], ax=ax, color="red", height=0.05)

        ax.set_title(f'Distribution of "{col}"')
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(pad=3.0)
    plt.suptitle("Feature Distributions: All Samples vs. Highlighted Samples", fontsize=16, y=1.02)
    plt.show()
# =============================
# MAIN EXECUTION WORKFLOW (MODIFIED)
# =============================
if __name__ == '__main__':
    # Step 1: Find all matched files
    matched_files = get_matched_files(RGB_DIR, GT_DIR, DEPTH_DIR)

    # Step 2: Extract features for every CC in every image
    all_features = []
    for i, (dt, file_paths) in enumerate(matched_files.items()):
        print(f"Processing image {i + 1}/{len(matched_files)}: {Path(file_paths['rgb']).name}")
        features = extract_features_for_image(file_paths)
        all_features.extend(features)

    if not all_features:
        print("No features were extracted. Please check your directories and file contents.")
    else:
        # Create a DataFrame from the features
        features_df = pd.DataFrame(all_features)
        print(f"\nExtracted {len(features_df)} features from {features_df['filepath'].nunique()} images.")

        # Step 3: Identify the samples to be highlighted
        # Get the base filename from the full path
        features_df['filename'] = features_df['filepath'].apply(lambda p: Path(p).name)
        # Create a new boolean column: True if the filename is in our highlight list
        features_df['is_highlighted'] = features_df['filename'].isin(HIGHLIGHT_FILENAMES)

        num_highlighted = features_df['is_highlighted'].sum()
        print(
            f"\nFound and marked {num_highlighted} data points from {len(HIGHLIGHT_FILENAMES)} specified files for highlighting.")

        # Step 4: Prepare data for t-SNE
        feature_cols = [
            'cc', 'relative_area_pct', 'norm_dist_from_center',
            'mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b',
            'width',"height","depth"
        ]

        features_df.dropna(subset=feature_cols, inplace=True)
        data_to_visualize = features_df[feature_cols].values

        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data_to_visualize)

        print("\nRunning t-SNE for visualization... (this may take a moment)")
        perplexity_val = min(30, len(scaled_features) - 1)
        if perplexity_val <= 0:
            print("Not enough data points to run t-SNE. Need at least 2.")
        else:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
            tsne_results = tsne.fit_transform(scaled_features)

            visualize_highlighted_samples(tsne_results, features_df)
            plot_feature_distributions(features_df, feature_cols)
            pca = PCA(n_components=2)
            pca_results = pca.fit_transform(scaled_features)
            visualize_pca_biplot(pca_results, pca, features_df, feature_cols)
        # Save the final dataframe with all features and highlight status
        features_df.to_csv("cluster_res/placenta_features_with_highlights.csv", index=False)
        print("\nSaved detailed feature data to 'placenta_features_with_highlights.csv'")