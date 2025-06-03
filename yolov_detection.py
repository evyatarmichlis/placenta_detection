import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import yaml
import re
from ultralytics import YOLO
import matplotlib.pyplot as plt


class PlacentaYOLOPipeline:
    def __init__(self, data_dir, seed, output_dir, min_contour_area=100):
        """
        Initialize the pipeline for preparing and training YOLO model using existing splits.

        Args:
            data_dir (str): Base data directory name prefix.
            seed (int): Seed number used in the directory structure.
            output_dir (Path): Directory for YOLO dataset and results.
            min_contour_area (int): Minimum contour area to consider.
        """
        self.data_dir = data_dir
        self.seed = seed
        self.base_path = Path(f'ucnet/placenta_data/{data_dir}{seed}')
        self.output_dir = Path(output_dir)
        self.min_contour_area = min_contour_area

        # Helper method for extracting datetime from filenames
        def extract_datetime(filename):
            """Extract datetime pattern from filename (YYYY-MM-DD_HH-MM-SS)"""
            match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', str(filename))
            return match.group(1) if match else None

        self.extract_datetime = extract_datetime

        # Define paths for existing splits
        self.split_paths = {
            'train': {
                'img': self.base_path / 'train' / 'img',
                'gt': self.base_path / 'train' / 'gt'
            },
            'val': {
                'img': self.base_path / 'val' / 'img',
                'gt': self.base_path / 'val' / 'gt'
            },
            'test': {
                'img': self.base_path / 'test' / 'img',
                'gt': self.base_path / 'test' / 'gt'
            }
        }

        # Create YOLO directory structure
        self.yolo_dir = self.output_dir / f'yolo_dataset_{seed}'
        for split in ['train', 'val', 'test']:
            (self.yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

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

    def prepare_dataset(self):
        """
        Prepare a YOLO-format dataset using the existing splits.
        """
        # Clean existing directories
        for split in ['train', 'val', 'test']:
            images_dir = self.yolo_dir / split / 'images'
            labels_dir = self.yolo_dir / split / 'labels'
            shutil.rmtree(images_dir, ignore_errors=True)
            shutil.rmtree(labels_dir, ignore_errors=True)
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

        # Process each split
        stats = {split: {'total': 0, 'with_defects': 0, 'total_defects': 0}
                 for split in ['train', 'val', 'test']}

        for split in ['train', 'val', 'test']:
            print(f"\nProcessing {split} split...")
            img_dir = self.split_paths[split]['img']
            gt_dir = self.split_paths[split]['gt']

            # Get all image and GT files
            img_files = list(img_dir.glob('*.jpg'))
            gt_files = list(gt_dir.glob('*.jpg'))

            # Create dictionaries with datetime as keys
            img_dict = {}
            for f in img_files:
                dt = self.extract_datetime(f.name)
                if dt:
                    img_dict[dt] = f

            gt_dict = {}
            for f in gt_files:
                dt = self.extract_datetime(f.name)
                if dt:
                    gt_dict[dt] = f

            # Find common datetime keys
            common_datetimes = set(img_dict.keys()) & set(gt_dict.keys())

            print(f"Found {len(common_datetimes)} matching image-mask pairs in {split} split")

            for dt in tqdm(common_datetimes):
                img_file = img_dict[dt]
                mask_file = gt_dict[dt]

                # Read and process the color image (apply CLAHE)
                color_img = cv2.imread(str(img_file))
                lab_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab_img)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_channel = clahe.apply(l_channel)
                lab_img = cv2.merge((l_channel, a_channel, b_channel))
                color_img_clahe = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

                # Save processed image
                filename = f"{dt}.jpg"
                out_img_path = self.yolo_dir / split / 'images' / filename
                cv2.imwrite(str(out_img_path), color_img_clahe)
                stats[split]['total'] += 1

                # Process the ground truth mask
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                bboxes = self.mask_to_bboxes(mask)

                # Make sure the label filename matches the image filename (without extension)
                # This is critical for YOLO to find the corresponding labels
                label_filename = f"{dt}.txt"
                label_file = self.yolo_dir / split / 'labels' / label_filename

                if bboxes:
                    with open(label_file, 'w') as f:
                        for bbox in bboxes:
                            # Format must be: class_id center_x center_y width height
                            # with values normalized between 0 and 1
                            f.write("0 " + " ".join([f"{coord:.6f}" for coord in bbox]) + "\n")
                    stats[split]['with_defects'] += 1
                    stats[split]['total_defects'] += len(bboxes)
                else:
                    # Create empty file for no defects (required by YOLO)
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
            name=f'train_seed{self.seed}',
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
            project=str(self.output_dir / 'predictions'),
            name=f'test_seed{self.seed}'
        )
        return metrics, results


def main():
    # Set parameters
    data_dir = "detectron_seed_"
    seed = 1  # Change this according to your seed value
    output_dir = Path("./yolo_detection")  # Update as needed

    # Initialize pipeline with existing splits
    pipeline = PlacentaYOLOPipeline(
        data_dir=data_dir,
        seed=seed,
        output_dir=output_dir,
        min_contour_area=100
    )

    # Prepare dataset using existing splits
    pipeline.prepare_dataset()

    # Train model
    model, results = pipeline.train_model(
        epochs=300,
        imgsz=640,
        batch=8,
        model_size='l'  # options: 'n', 's', 'm'
    )

    # Evaluate model
    pipeline.evaluate_model(model)


if __name__ == "__main__":
    main()