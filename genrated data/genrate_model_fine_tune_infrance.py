import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoConfig

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class EnhancedDepthProcessor:
    """
    Processor for enhancing depth maps, particularly for medical images with backgrounds.
    """

    def __init__(self, background_threshold=0.01):
        self.background_threshold = background_threshold

    def process_depth(self, depth_map):
        """
        Apply enhancement pipeline to depth map
        """
        # Create a copy to avoid modifying the original
        processed = depth_map.copy()

        # Create mask for non-background pixels
        mask = processed > self.background_threshold

        # Skip processing if no foreground pixels
        if not np.any(mask):
            return processed

        # Get only non-zero pixels for processing
        non_zero_pixels = processed[mask]

        # Normalize to [0,255] range for OpenCV operations
        stretched = non_zero_pixels * 255

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(stretched.astype(np.uint8))

        # Apply gamma correction
        gamma = 1.2  # Tune this value for your dataset
        gamma_corrected = np.power(contrast_enhanced / 255.0, gamma) * 255
        gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

        # Edge enhancement using Laplacian
        laplacian = cv2.Laplacian(gamma_corrected, cv2.CV_64F, ksize=3)
        enhanced_edges = gamma_corrected - 0.5 * laplacian  # Boosts local differences

        # Sharpening
        enhanced_edges_uint8 = np.clip(enhanced_edges, 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(enhanced_edges_uint8, (3, 3), 0)
        sharpened = cv2.addWeighted(enhanced_edges_uint8, 1.5, blurred, -0.5, 0)

        # Apply the processed values back to the original array, keeping background as zero
        result = np.zeros_like(processed)
        result[mask] = sharpened.reshape(-1) / 255.0  # Normalize back to [0,1] range

        return result


class InferenceDataset(Dataset):
    def __init__(self, color_images_dir, image_processor):
        """
        Args:
            color_images_dir (str or Path): Directory containing the masked color images.
            image_processor: Hugging Face image processor.
        """
        self.color_images_dir = Path(color_images_dir)
        self.image_processor = image_processor
        self.image_files = sorted(list(self.color_images_dir.glob("*.jpg")))
        print(f"Found {len(self.image_files)} images for inference")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")

        # Process the image
        processed = self.image_processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)

        # Get image as numpy array for masking
        image_np = np.array(image)

        # Create a mask from the image (assumes background is black)
        mask = (image_np.sum(axis=2) > 0).astype(np.float32)

        return {
            "pixel_values": pixel_values,
            "image_path": str(image_path),
            "mask": mask
        }


def save_visualization(original_depth, processed_depth, save_path, image_name):
    """Save a visualization of the original and processed depth maps"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    im1 = ax1.imshow(original_depth, cmap='viridis')
    ax1.set_title("Raw Depth")
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(processed_depth, cmap='viridis')
    ax2.set_title("Enhanced Depth")
    fig.colorbar(im2, ax=ax2)

    plt.suptitle(f"Depth Map: {image_name}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # Directories setup
    root_dir = Path.cwd().parent
    processed_dir = root_dir / "processed"
    root_dir = Path.cwd().parent
    data_dir = "detectron_seed_"
    seed = 1
    color_images_dir =     Path(f'./placenta_data/{data_dir}{seed}/val/img/')
    # Output directories
    generated_depths_dir = Path(f'./placenta_data/{data_dir}{seed}/val/gen_depth/')
    generated_depths_dir.mkdir(parents=True, exist_ok=True)
    # For visualizations
    visualization_dir = processed_dir / "depth_visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    # Optional: Separate directory for enhanced versions
    enhanced_depths_dir = processed_dir / "enhanced_depths"
    enhanced_depths_dir.mkdir(parents=True, exist_ok=True)

    # Model path - using your fine-tuned model
    model_path = "ucnet/final_depth_model"  # Path to your fine-tuned model
    base_model = "Intel/zoedepth-nyu-kitti"

    print(f"Loading model from {model_path}")

    # Load the model with the appropriate configuration
    config = AutoConfig.from_pretrained(base_model)
    image_processor = AutoImageProcessor.from_pretrained(base_model)

    # Attempt to load your fine-tuned model
    try:
        model = AutoModelForDepthEstimation.from_pretrained(model_path, config=config)
        print("Successfully loaded your fine-tuned model")
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        print("Falling back to checkpoint model as a temporary measure")
        model = AutoModelForDepthEstimation.from_pretrained("results/checkpoint-6180", config=config)

    model = model.to(device)
    model.eval()

    # Create depth processor for post-processing
    depth_processor = EnhancedDepthProcessor(background_threshold=0.01)

    # Create dataset and dataloader
    inference_dataset = InferenceDataset(color_images_dir, image_processor)
    inference_dataloader = DataLoader(inference_dataset, batch_size=4, shuffle=False)

    # Process all images
    print("Starting inference...")
    with torch.no_grad():
        for batch in tqdm(inference_dataloader, desc="Processing images"):
            pixel_values = batch["pixel_values"].to(device)

            # Generate depth predictions
            outputs = model(pixel_values)
            predicted_depth = outputs["predicted_depth"]

            # Optional: Resize if needed
            # predicted_depth = F.interpolate(predicted_depth, size=(480, 640), mode='bilinear', align_corners=False)

            # Process each image in the batch
            for i in range(len(pixel_values)):
                # Get the depth map
                depth_map = predicted_depth[i].squeeze().cpu().numpy()

                # Get the image path and mask
                image_path = Path(batch["image_path"][i])
                mask = batch["mask"][i].cpu().numpy()  # Convert tensor to numpy

                # Check for shape mismatch and resize mask if needed
                if mask.shape != depth_map.shape:
                    print(f"Resizing mask from {mask.shape} to {depth_map.shape}")
                    mask = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

                # Apply mask to ensure background is zero
                masked_depth = depth_map * mask

                # Apply post-processing
                enhanced_depth = depth_processor.process_depth(masked_depth)

                raw_save_path = generated_depths_dir / f"{image_path.stem}.csv"
                np.savetxt(raw_save_path, masked_depth, delimiter=",")

                # Save the enhanced depth map
                enhanced_save_path = enhanced_depths_dir / f"{image_path.stem}.csv"
                np.savetxt(enhanced_save_path, enhanced_depth, delimiter=",")

                # Create and save visualization
                vis_save_path = visualization_dir / f"{image_path.stem}.png"
                save_visualization(masked_depth, enhanced_depth, vis_save_path, image_path.name)

    print("Inference complete!")
    print(f"Raw depth maps saved to: {generated_depths_dir}")
    print(f"Enhanced depth maps saved to: {enhanced_depths_dir}")
    print(f"Visualizations saved to: {visualization_dir}")


if __name__ == "__main__":
    main()