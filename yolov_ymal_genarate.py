import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import yaml


class PlacentaYOLOConverter:
    def __init__(self, organized_data_dir, yolo_output_dir, min_contour_area=100):
        """
        Convert organized placenta dataset to YOLO format.

        Args:
            organized_data_dir (str): Directory containing organized dataset
            yolo_output_dir (str): Directory where YOLO format dataset will be saved
            min_contour_area (int): Minimum contour area to consider (to filter noise)
        """
        self.organized_data_dir = organized_data_dir
        self.yolo_output_dir = yolo_output_dir
        self.min_contour_area = min_contour_area

    def mask_to_bboxes(self, mask):
        """
        Convert a binary mask to multiple YOLO format bounding boxes.

        Args:
            mask (numpy.ndarray): Binary mask image

        Returns:
            list: List of [x_center, y_center, width, height] for each contour
        """
        # Ensure binary mask
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Threshold if not already binary
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        image_h, image_w = mask.shape

        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < self.min_contour_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Convert to YOLO format (normalized coordinates)
            x_center = (x + w / 2) / image_w
            y_center = (y + h / 2) / image_h
            width = w / image_w
            height = h / image_h

            bboxes.append([x_center, y_center, width, height])

        return bboxes

    def setup_yolo_directories(self):
        """Create YOLO dataset directory structure."""
        for split in ['train', 'val', 'test']:
            (self.yolo_output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.yolo_output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    def create_yaml(self):
        """Create dataset.yaml file for YOLO training."""
        yaml_content = {
            'path': str(self.yolo_output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,  # number of classes
            'names': ['defect']  # class names
        }

        yaml_path = self.yolo_output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)

    def convert_dataset(self):
        """Convert the organized dataset to YOLO format."""
        self.setup_yolo_directories()

        # Process each split
        all_stats = {}
        for split in ['train', 'val', 'test']:
            print(f"\nProcessing {split} split...")
            split_stats = {'total_images': 0, 'total_defects': 0, 'images_with_defects': 0}

            # Process positive samples (with defects)
            pos_dir = self.organized_data_dir / split / 'positive'
            for img_file in tqdm(list(pos_dir.glob('*color-image*.jpg')), desc=f"Processing {split} positive"):
                # Get corresponding mask file
                mask_file = pos_dir / f"gt_mask-image_{img_file.name.split('color-image_')[1]}"
                if not mask_file.exists():
                    print(f"Warning: No mask found for {img_file}")
                    continue

                # Convert mask to bboxes
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                bboxes = self.mask_to_bboxes(mask)

                # Copy image to YOLO directory
                shutil.copy2(img_file, self.yolo_output_dir / split / 'images' / img_file.name)
                split_stats['total_images'] += 1

                # Create YOLO format label file
                if bboxes:
                    label_file = self.yolo_output_dir / split / 'labels' / f"{img_file.stem}.txt"
                    with open(label_file, 'w') as f:
                        for bbox in bboxes:
                            f.write(f"0 {' '.join(map(str, bbox))}\n")
                    split_stats['total_defects'] += len(bboxes)
                    split_stats['images_with_defects'] += 1
                else:
                    print(f"Warning: No valid contours found in {mask_file}")

            # Process negative samples (no defects)
            neg_dir = self.organized_data_dir / split / 'negative'
            for img_file in tqdm(list(neg_dir.glob('*color-image*.jpg')), desc=f"Processing {split} negative"):
                if img_file.name.startswith('gt_'):
                    continue

                # Copy image to YOLO directory
                shutil.copy2(img_file, self.yolo_output_dir / split / 'images' / img_file.name)
                split_stats['total_images'] += 1

                # Create empty label file (no defects)
                label_file = self.yolo_output_dir / split / 'labels' / f"{img_file.stem}.txt"
                with open(label_file, 'w') as f:
                    pass  # Empty file for negative samples

            all_stats[split] = split_stats

        # Create dataset.yaml
        self.create_yaml()

        # Print detailed statistics
        self.print_statistics(all_stats)

    def print_statistics(self, stats):
        """Print detailed dataset statistics."""
        print("\nDataset Statistics:")
        for split, split_stats in stats.items():
            print(f"\n{split}:")
            print(f"  Total images: {split_stats['total_images']}")
            print(f"  Images with defects: {split_stats['images_with_defects']}")
            print(f"  Total defect instances: {split_stats['total_defects']}")
            avg_defects = (split_stats['total_defects'] / split_stats['images_with_defects']
                           if split_stats['images_with_defects'] > 0 else 0)
            print(f"  Average defects per positive image: {avg_defects:.2f}")


def visualize_annotations(image_path, label_path, output_path=None):
    """
    Visualize YOLO format annotations on an image.

    Args:
        image_path (str): Path to the image
        label_path (str): Path to the YOLO format label file
        output_path (str, optional): Path to save the visualization
    """
    # Read image
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]

    # Read labels
    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        with open(label_path, 'r') as f:
            labels = f.readlines()

        # Draw each bbox
        for label in labels:
            class_id, x_center, y_center, width, height = map(float, label.strip().split())

            # Convert normalized coordinates to pixel coordinates
            x = int((x_center - width / 2) * w)
            y = int((y_center - height / 2) * h)
            box_w = int(width * w)
            box_h = int(height * h)

            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(str(output_path), image)
    else:
        cv2.imshow('Annotations', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # Example usage

    current_file = Path(__file__)
    root_dir = current_file.parent
    output_base_dir = root_dir / f"dataset_split_seed_0"

    organized_data_dir = output_base_dir
    yolo_output_dir = root_dir/'yolov_dir'

    # Convert dataset
    converter = PlacentaYOLOConverter(
        organized_data_dir=organized_data_dir,
        yolo_output_dir=yolo_output_dir,
        min_contour_area=100  # Adjust this threshold based on your needs
    )
    converter.convert_dataset()

    # Visualize some examples (optional)
    train_images_dir = Path(yolo_output_dir) / 'train' / 'images'
    train_labels_dir = Path(yolo_output_dir) / 'train' / 'labels'

    # Visualize first 5 images with annotations
    for img_path in list(train_images_dir.glob('*.jpg'))[:5]:
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        vis_path = Path(yolo_output_dir) / 'visualizations' / f"{img_path.stem}_annotated.jpg"
        vis_path.parent.mkdir(exist_ok=True)
        visualize_annotations(img_path, label_path, vis_path)


if __name__ == "__main__":
    main()