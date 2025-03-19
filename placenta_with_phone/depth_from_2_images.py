import os
import sys
import subprocess
import glob


def run_command(cmd, cwd=None):
    print("Running command: " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def get_first_image_path(images_folder):
    # Look for common image extensions
    extensions = ["*.jpg", "*.png", "*.jpeg", "*.JPG"]
    for ext in extensions:
        files = glob.glob(os.path.join(images_folder, ext))
        if files:
            return os.path.basename(files[0])
    return None


def main(dataset_path):
    # Define paths
    images_path = os.path.join(dataset_path, "images")
    database_path = os.path.join(dataset_path, "database.db")
    sparse_path = os.path.join(dataset_path, "sparse")
    dense_path = os.path.join(dataset_path, "dense")

    # Create necessary directories
    os.makedirs(sparse_path, exist_ok=True)
    os.makedirs(dense_path, exist_ok=True)

    # 1. Feature Extraction
    cmd_feature_extractor = [
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", images_path
    ]
    run_command(cmd_feature_extractor)

    # 2. Feature Matching
    cmd_exhaustive_matcher = [
        "colmap", "exhaustive_matcher",
        "--database_path", database_path
    ]
    run_command(cmd_exhaustive_matcher)

    # 3. Sparse Reconstruction (Mapping)
    os.makedirs(os.path.join(sparse_path, "0"), exist_ok=True)
    cmd_mapper = [
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", images_path,
        "--output_path", sparse_path
    ]
    run_command(cmd_mapper)

    # 4. Image Undistortion for Dense Reconstruction
    sparse_model_path = os.path.join(sparse_path, "0")
    cmd_undistorter = [
        "colmap", "image_undistorter",
        "--image_path", images_path,
        "--input_path", sparse_model_path,
        "--output_path", dense_path,
        "--output_type", "COLMAP",
        "--max_image_size", "2000"
    ]
    run_command(cmd_undistorter)

    # 5. Dense Stereo Matching (PatchMatch Stereo)
    cmd_patch_match = [
        "colmap", "patch_match_stereo",
        "--workspace_path", dense_path,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true"
    ]
    try:
        run_command(cmd_patch_match)
    except subprocess.CalledProcessError as e:
        print("Error: COLMAP patch-match stereo requires a CUDA-enabled GPU.")
        print("Dense stereo reconstruction cannot be performed on this system.")
        sys.exit(1)

    # 6. Optionally, locate the depth map for one image.
    first_image = get_first_image_path(images_path)
    if not first_image:
        print("No image files found in", images_path)
        return

    base_name = os.path.splitext(first_image)[0]
    # Check common locations for depth maps (typically stored as .exr files)
    candidate_dirs = [os.path.join(dense_path, "stereo", "depths"),
                      os.path.join(dense_path, "stereo", "depth_maps")]
    depth_map_file = None
    for d in candidate_dirs:
        potential = os.path.join(d, base_name + ".exr")
        if os.path.exists(potential):
            depth_map_file = potential
            break

    if depth_map_file:
        print(f"Depth map for image '{first_image}' found at: {depth_map_file}")
    else:
        print(f"Could not find a depth map for image '{first_image}' in expected locations.")
if __name__ == "__main__":

    dataset_path = 'dataset'
    main(dataset_path)