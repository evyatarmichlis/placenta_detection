import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import random
import yaml
from PIL import Image
import re
from ultralytics import YOLO
import matplotlib.pyplot as plt


class PlacentaYOLOPipeline:
    def __init__(self, color_images_dir, gt_dir, output_dir, min_contour_area=100):
        """
        Initialize the pipeline for preparing and training YOLO model.

        Args:
            color_images_dir (str): Directory with color images
            gt_dir (str): Directory with ground truth masks
            output_dir (str): Directory for YOLO dataset and results
            min_contour_area (int): Minimum contour area to consider
        """
        self.color_images_dir = color_images_dir
        self.gt_dir = gt_dir
        self.output_dir = output_dir
        self.min_contour_area = min_contour_area

        # Create YOLO directory structure
        self.yolo_dir = self.output_dir / 'yolo_dataset'
        for split in ['train', 'val', 'test']:
            (self.yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    def extract_datetime(self, filename):
        """Extract datetime pattern from filename."""
        match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
        return match.group(1) if match else None

    def mask_to_bboxes(self, mask, scale_factor=1.2):
        """
        Convert binary mask to YOLO format bounding boxes,
        and enlarge them by `scale_factor` around the center.
        """
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        image_h, image_w = mask.shape

        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Convert to normalized YOLO coords
            x_center = (x + w / 2) / image_w
            y_center = (y + h / 2) / image_h
            width = w / image_w
            height = h / image_h

            # -----------------------
            # Enlarge bounding box
            # -----------------------
            # 1) Multiply width & height by the scale factor
            new_width = width * scale_factor
            new_height = height * scale_factor

            # 2) Convert to "left, right, top, bottom" in [0..1]
            left = x_center - new_width / 2
            right = x_center + new_width / 2
            top = y_center - new_height / 2
            bottom = y_center + new_height / 2

            # 3) Clamp to image boundaries [0,1]
            left = max(left, 0.0)
            right = min(right, 1.0)
            top = max(top, 0.0)
            bottom = min(bottom, 1.0)

            # 4) Recompute center and size after clamping
            x_center = (left + right) / 2
            y_center = (top + bottom) / 2
            width = right - left
            height = bottom - top

            # If the box became too small or invalid, skip
            if width <= 0 or height <= 0:
                continue

            bboxes.append([x_center, y_center, width, height])

        return bboxes

    def prepare_dataset(self, split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}, seed=42):
        """Prepare YOLO format dataset."""
        random.seed(seed)
        for split in ['train', 'val', 'test']:
            images_dir = self.yolo_dir / split / 'images'
            labels_dir = self.yolo_dir / split / 'labels'
            # Remove old directories if they exist
            shutil.rmtree(images_dir, ignore_errors=True)
            shutil.rmtree(labels_dir, ignore_errors=True)
            # Recreate clean directories
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

        random.seed(seed)

        # Match images with GT masks
        print("Matching images with ground truth masks...")
        gt_files = {self.extract_datetime(f.name): f for f in self.gt_dir.glob('*.jpg')}

        matched_files = []
        for img_file in self.color_images_dir.glob('*.jpg'):
            img_datetime = self.extract_datetime(img_file.name)
            if img_datetime and img_datetime in gt_files:
                matched_files.append((img_file, gt_files[img_datetime]))

        # Split dataset
        random.shuffle(matched_files)
        total = len(matched_files)
        train_end = int(total * split_ratios['train'])
        val_end = train_end + int(total * split_ratios['val'])

        splits = {
            'train': matched_files[:train_end],
            'val': matched_files[train_end:val_end],
            'test': matched_files[val_end:]
        }

        # Process each split
        stats = {split: {'total': 0, 'with_defects': 0, 'total_defects': 0}
                 for split in splits}

        for split, files in splits.items():
            print(f"\nProcessing {split} split...")
            for img_file, mask_file in tqdm(files):
                # Copy image
                color_img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)


                lab_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab_img)

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_channel = clahe.apply(l_channel)
                lab_img = cv2.merge((l_channel, a_channel, b_channel))
                color_img_clahe = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
                out_img_path = self.yolo_dir / split / 'images' / img_file.name
                cv2.imwrite(str(out_img_path), color_img_clahe)  # Save processed image
                stats[split]['total'] += 1

                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                bboxes = self.mask_to_bboxes(mask)

                # Create label file
                label_file = self.yolo_dir / split / 'labels' / f"{img_file.stem}.txt"
                if bboxes:
                    with open(label_file, 'w') as f:
                        for bbox in bboxes:
                            f.write(f"0 {' '.join(map(str, bbox))}\n")
                    stats[split]['with_defects'] += 1
                    stats[split]['total_defects'] += len(bboxes)
                else:
                    # Create empty file for negative samples
                    label_file.touch()

        # Create dataset.yaml
        yaml_content = {
            'path': str(self.yolo_dir.absolute()),
            'train': str(Path('train') / 'images'),
            'val': str(Path('val') / 'images'),
            'test': str(Path('test') / 'images'),
            'nc': 1,
            'names': ['defect']
        }

        with open(self.yolo_dir / 'dataset.yaml', 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        # Print statistics
        print("\nDataset Statistics:")
        for split, stat in stats.items():
            print(f"\n{split}:")
            print(f"  Total images: {stat['total']}")
            print(f"  Images with defects: {stat['with_defects']}")
            print(f"  Total defect instances: {stat['total_defects']}")
            if stat['with_defects'] > 0:
                print(f"  Average defects per positive image: "
                      f"{stat['total_defects'] / stat['with_defects']:.2f}")

    def train_model(self, epochs=100, imgsz=640, batch=16, model_size='l'):
        """Train YOLOv8 model."""
        print("\nStarting model training...")

        # Initialize model
        model = YOLO(f'yolov8{model_size}.pt')

        # Train
        results = model.train(
            data=str(self.yolo_dir / 'dataset.yaml'),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=str(self.output_dir / 'runs'),
            name='train',
            plots=True
        )

        return model, results

    def evaluate_model(self, model):
        """Evaluate the trained model."""
        print("\nEvaluating model...")

        # Run validation
        metrics = model.val(data=str(self.yolo_dir / 'dataset.yaml'),
                            split='test')

        results = model.predict(
            source=self.yolo_dir/'test'/'images' ,
            save=True,  # Save the results to a folder
            imgsz=640,  # Image size
            conf=0.25,  # Confidence threshold for predictions
            project=self.yolo_dir/"predictions"  # or use save_dir

        )

def main():
    # Set paths
    current_file = Path(__file__)
    root_dir = current_file.parent
    color_images_dir = root_dir / "Images" / "masked_images"
    gt_dir = root_dir / "Images" / "gt"
    output_dir = root_dir / "yolo_detection"

    # Initialize pipeline
    pipeline = PlacentaYOLOPipeline(
        color_images_dir=color_images_dir,
        gt_dir=gt_dir,
        output_dir=output_dir,
        min_contour_area=100  # Adjust based on your defect sizes
    )

    # Prepare dataset
    pipeline.prepare_dataset()

    # Train model
    model, results = pipeline.train_model(
        epochs=300,
        imgsz=640,
        batch=32,
        model_size='m'  # 'n' for nano, 's' for small, 'm' for medium
    )
    # model = YOLO('yolo_detection/runs/train13/weights/best.pt')
    # Evaluate model
    metrics = pipeline.evaluate_model(model)


if __name__ == "__main__":
    main()