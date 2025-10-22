import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from transformers import pipeline
import torch
from accelerate.test_utils.testing import get_backend
from PIL import Image

# Get the device from accelerate.
device, _, _ = get_backend()

# Set the checkpoint for the depth-estimation model.
checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
# Create the depth-estimation pipeline. (This pipeline predicts depth maps from color images.)
pipe = pipeline("depth-estimation", model=checkpoint, device=device)

# Define your directories.
root_dir = Path.cwd().parent
input_dir = root_dir / "Images" / "images"  # Folder containing your input images.
output_dir = root_dir / "Images" / "new_depth_files"  # Folder where the new depth CSVs will be saved.
output_dir.mkdir(parents=True, exist_ok=True)

# Process every JPEG image in the input directory.
for img_file in input_dir.glob("*.jpg"):
    print(f"Processing {img_file.name} ...")
    # Open the image and ensure it's in RGB mode.
    image = Image.open(img_file).convert("RGB")

    # Run the depth-estimation pipeline on the image.
    predictions = pipe(image)

    # The pipeline returns a dict with a "depth" key.
    depth_img = predictions["depth"]

    # (Optional) You can display the depth image:
    # plt.imshow(depth_img, cmap="gray")
    # plt.title(img_file.name)
    # plt.show()

    # Convert the predicted depth image to a NumPy array.
    depth_array = np.array(depth_img)

    # Save the depth map as a CSV file. The filename is based on the input image name.
    csv_filename = output_dir / (img_file.stem + ".csv")
    np.savetxt(csv_filename, depth_array, delimiter=",")
    print(f"Saved predicted depth for {img_file.name} to {csv_filename}")