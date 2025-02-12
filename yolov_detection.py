import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import random
import yaml
import re
from ultralytics import YOLO
import matplotlib.pyplot as plt


class PlacentaYOLOPipeline:
    def __init__(self, color_images_dir, gt_dir, output_dir, min_contour_area=100):
        """
        Initialize the pipeline for preparing and training YOLO model.

        Args:
            color_images_dir (Path): Directory with color images.
            gt_dir (Path): Directory with ground truth masks.
            output_dir (Path): Directory for YOLO dataset and results.
            min_contour_area (int): Minimum contour area to consider.
        """
        self.color_images_dir = Path(color_images_dir)
        self.gt_dir = Path(gt_dir)
        self.output_dir = Path(output_dir)
        self.min_contour_area = min_contour_area

        # Create YOLO directory structure
        self.yolo_dir = self.output_dir / 'yolo_dataset'
        for split in ['train', 'val', 'test']:
            (self.yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    def extract_datetime(self, filename):
        """
        Extract datetime pattern from filename.
        Expected pattern: YYYY-MM-DD_HH-MM-SS
        """
        match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
        return match.group(1) if match else None

    def extract_hour_key(self, filename):
        """
        Extract a key representing the date and hour from the filename.
        For example, for '2023-01-01_12-30-15.jpg', return '2023-01-01_12'
        """
        dt_str = self.extract_datetime(filename)
        return dt_str[:13] if dt_str else None

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
            x_center = (x + w / 2) / image_w
            y_center = (y + h / 2) / image_h
            width = w / image_w
            height = h / image_h

            # Enlarge the bounding box by the scale factor
            new_width = width * scale_factor
            new_height = height * scale_factor

            # Compute new left, right, top, and bottom (clamped to [0,1])
            left = max(x_center - new_width / 2, 0.0)
            right = min(x_center + new_width / 2, 1.0)
            top = max(y_center - new_height / 2, 0.0)
            bottom = min(y_center + new_height / 2, 1.0)

            # Recompute center and size after clamping
            x_center = (left + right) / 2
            y_center = (top + bottom) / 2
            width = right - left
            height = bottom - top

            if width <= 0 or height <= 0:
                continue

            bboxes.append([x_center, y_center, width, height])

        return bboxes

    def prepare_dataset(self, split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}, seed=42):
        """
        Prepare a YOLO-format dataset by grouping images by hour (to avoid temporal leakage)
        and then splitting the hour groups into train, val, and test sets.
        """
        # Clean existing directories
        for split in ['train', 'val', 'test']:
            images_dir = self.yolo_dir / split / 'images'
            labels_dir = self.yolo_dir / split / 'labels'
            shutil.rmtree(images_dir, ignore_errors=True)
            shutil.rmtree(labels_dir, ignore_errors=True)
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

        # Match images with ground truth masks
        print("Matching images with ground truth masks...")
        gt_files = {self.extract_datetime(f.name): f for f in self.gt_dir.glob('*.jpg')}
        matched_files = []
        for img_file in self.color_images_dir.glob('*.jpg'):
            img_datetime = self.extract_datetime(img_file.name)
            if img_datetime and img_datetime in gt_files:
                matched_files.append((img_file, gt_files[img_datetime]))

        # Group matched files by the hour key (e.g. "2023-01-01_12")
        hour_groups = {}
        for img_file, mask_file in matched_files:
            key = self.extract_hour_key(img_file.name)
            if key is None:
                continue
            hour_groups.setdefault(key, []).append((img_file, mask_file))

        # Sort and shuffle the hour keys
        group_keys = list(hour_groups.keys())
        group_keys.sort()
        random.seed(seed)
        random.shuffle(group_keys)

        n_total = len(group_keys)
        n_train = int(split_ratios['train'] * n_total)
        n_val = int(split_ratios['val'] * n_total)
        # Remaining groups go to test.
        train_keys = group_keys[:n_train]
        val_keys = group_keys[n_train:n_train+n_val]
        test_keys = group_keys[n_train+n_val:]

        # Merge files from each hour group for each split
        splits = {
            'train': [item for key in train_keys for item in hour_groups[key]],
            'val': [item for key in val_keys for item in hour_groups[key]],
            'test': [item for key in test_keys for item in hour_groups[key]]
        }

        stats = {split: {'total': 0, 'with_defects': 0, 'total_defects': 0}
                 for split in splits}

        # Process each split
        for split, files in splits.items():
            print(f"\nProcessing {split} split with {len(files)} samples...")
            for img_file, mask_file in tqdm(files):
                # Read and process the color image (apply CLAHE)
                color_img = cv2.imread(str(img_file))
                lab_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab_img)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_channel = clahe.apply(l_channel)
                lab_img = cv2.merge((l_channel, a_channel, b_channel))
                color_img_clahe = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

                # Save processed image
                out_img_path = self.yolo_dir / split / 'images' / img_file.name
                cv2.imwrite(str(out_img_path), color_img_clahe)
                stats[split]['total'] += 1

                # Process the ground truth mask
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                bboxes = self.mask_to_bboxes(mask)
                label_file = self.yolo_dir / split / 'labels' / f"{img_file.stem}.txt"
                if bboxes:
                    with open(label_file, 'w') as f:
                        for bbox in bboxes:
                            f.write("0 " + " ".join(map(str, bbox)) + "\n")
                    stats[split]['with_defects'] += 1
                    stats[split]['total_defects'] += len(bboxes)
                else:
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
                print(f"  Avg defects per positive image: {stat['total_defects'] / stat['with_defects']:.2f}")

    def train_model(self, epochs=100, imgsz=640, batch=16, model_size='l'):
        """Train YOLOv8 model."""
        print("\nStarting model training...")
        model = YOLO(f'yolov8{model_size}.pt')
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
        metrics = model.val(data=str(self.yolo_dir / 'dataset.yaml'), split='test')
        results = model.predict(
            source=self.yolo_dir / 'test' / 'images',
            save=True,
            imgsz=640,
            conf=0.20,
            project=self.yolo_dir / "predictions"
        )
        return metrics, results


def main():
    # Set paths (update these paths as needed)
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
        min_contour_area=100
    )

    # Prepare dataset with hour-based splitting
    pipeline.prepare_dataset(seed=1)

    # Train model
    model, results = pipeline.train_model(
        epochs=300,
        imgsz=640,
        batch=32,
        model_size='m'  # options: 'n', 's', 'm'
    )

    # Evaluate model
    pipeline.evaluate_model(model)


if __name__ == "__main__":
    main()