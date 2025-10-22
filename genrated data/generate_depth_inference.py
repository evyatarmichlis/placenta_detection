import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoConfig
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define an inference dataset that only loads color images
class InferenceDataset(Dataset):
    def __init__(self, color_images_dir, image_processor):
        """
        Args:
            color_images_dir (str or Path): Directory containing the images.
            image_processor: Hugging Face image processor.
        """
        self.color_images_dir = Path(color_images_dir)
        self.image_processor = image_processor
        # Look for common image formats
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(list(self.color_images_dir.glob(ext)))
        self.image_files = sorted(self.image_files)
        print(f"Found {len(self.image_files)} images in {color_images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        processed = self.image_processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)
        # Convert image_path to string to avoid collate issues
        return {"pixel_values": pixel_values, "image_path": str(image_path)}


# Set up paths
root_dir = Path.cwd().parent  # Modify this if needed
images_dir = root_dir / "Images" / "images"  # Update this to your images directory
output_dir = root_dir / "detectron_generated_depths"  # Where to save depth maps
output_dir.mkdir(parents=True, exist_ok=True)

# Load model and processor
print("Loading model and processor...")
checkpoint = "Intel/zoedepth-nyu-kitti"  # Base modelÎÎÍ¸ƒÏÏÏ
config = AutoConfig.from_pretrained(checkpoint)

# If you want to use the fine-tuned model, uncomment this:
# fine_tuned_checkpoint = "results/checkpoint-2315"
# model = AutoModelForDepthEstimation.from_pretrained(fine_tuned_checkpoint, config=config)
# print("Using fine-tuned model from:", fine_tuned_checkpoint)

# If you want to use the base model instead, comment the above and uncomment this:
model = AutoModelForDepthEstimation.from_pretrained(checkpoint)
print("Using base model:", checkpoint)

model = model.to(device)
model.eval()
image_processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=False)

# Create the inference dataset and dataloader
inference_dataset = InferenceDataset(images_dir, image_processor)
inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False)  # Batch size 1 for visualization


# Function to save depth maps as both CSV and visualizations
def save_depth_map(depth_tensor, original_path, output_dir, visualize=True):
    depth_np = depth_tensor.cpu().numpy()

    # Create filenames
    original_path = Path(original_path)
    base_name = original_path.stem

    # Save as CSV
    csv_path = output_dir / f"{base_name}.csv"
    np.savetxt(csv_path, depth_np, delimiter=",")

    # Save visualization
    if visualize:
        plt.figure(figsize=(10, 7.5))
        plt.imshow(depth_np, cmap='plasma')  # plasma or viridis are good colormaps for depth
        plt.colorbar(label='Depth')
        plt.title(f"Depth Map: {base_name}")
        plt.axis('off')

        # Save visualization
        vis_path = output_dir / f"{base_name}_depth_vis.png"
        plt.savefig(vis_path, bbox_inches='tight', dpi=150)
        plt.close()

    return csv_path


# Run inference and save predicted depths
print(f"Processing {len(inference_dataset)} images...")
with torch.no_grad():
    for i, batch in enumerate(inference_dataloader):
        # Get current image path for display
        img_path = batch["image_path"][0]
        print(f"Processing image {i + 1}/{len(inference_dataset)}: {Path(img_path).name}")

        # Process through model
        pixel_values = batch["pixel_values"].to(device)
        outputs = model(pixel_values)

        # Get predicted depths
        predicted_depth = outputs["predicted_depth"]  # (B, H, W)

        # 1) add channel dimension  →  (B, 1, H, W)
        if predicted_depth.dim() == 3:  # ZoeDepth case
            predicted_depth = predicted_depth.unsqueeze(1)

        # 2) resize *exactly* to 480 × 640
        predicted_depth = F.interpolate(
            predicted_depth,
            size=(480, 640),  # (height, width)
            mode="bicubic",  # or "bilinear" / "nearest"
            align_corners=False
        )

        # 3) drop the channel dimension for later convenience  →  (B, H, W)
        predicted_depth = predicted_depth.squeeze(1)
        # Save depth map with visualization
        saved_path = save_depth_map(
            predicted_depth[0],  # First (and only) item in batch
            batch["image_path"][0],
            output_dir,
            visualize=True
        )

        print(f"  Saved to: {saved_path}")

print(f"Done! All depth maps saved to {output_dir}")

# If you want to visualize one example at the end
if len(inference_dataset) > 0:
    example_idx = 0  # Change this to view a different example
    example_path = inference_dataset.image_files[example_idx]
    example_depth_path = output_dir / f"{example_path.stem}_depth_vis.png"

    if example_depth_path.exists():
        print(f"\nDisplaying example depth map: {example_path.name}")
        plt.figure(figsize=(15, 7))

        # Original image
        plt.subplot(1, 2, 1)
        img = Image.open(example_path)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')

        # Depth map
        plt.subplot(1, 2, 2)
        depth_vis = Image.open(example_depth_path)
        plt.imshow(depth_vis)
        plt.title("Depth Map")
        plt.axis('off')

        plt.tight_layout()
        plt.show()