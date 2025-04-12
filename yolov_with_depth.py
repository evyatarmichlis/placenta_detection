import os
import re

import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import random
import yaml
import torch
import torch.nn as nn
from ultralytics import YOLO
import pandas as pd
from find_local_max import FindLocalMax
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
class MultiStreamYOLOTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)

    def train(self, train_loader, val_loader, epochs=100, lr=0.001):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(epochs):
            for m in self.model.modules():
                if isinstance(m, nn.Module):
                    m.train()

            total_loss = 0
            for batch_idx, (rgb, depth, targets) in enumerate(train_loader):
                rgb = rgb.to(self.device)
                depth = depth.to(self.device)
                batch_targets = []
                for target in targets:
                    batch_targets.append({
                        'boxes': target['boxes'].to(self.device),
                        'labels': target['labels'].to(self.device)
                    })

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = self.model(rgb, depth, batch_targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                if batch_idx % 10 == 0:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

            if epoch % 5 == 0:
                self.validate(val_loader)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0

        for rgb, depth, targets in val_loader:
            rgb = rgb.to(self.device)
            depth = depth.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(rgb, depth, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')


def train_model(self, epochs=100, batch_size=16, model_size='m'):
    model = MultiStreamYOLOResNet(f'yolov8{model_size}.pt')
    trainer = MultiStreamYOLOTrainer(model)

    train_dataset = MultiStreamDatasetResNet(
        self.dataset_dir / 'train',
        img_size=640
    )

    val_dataset = MultiStreamDatasetResNet(
        self.dataset_dir / 'val',
        img_size=640
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    trainer.train(train_loader, val_loader, epochs=epochs)
    return model

def collate_fn(batch):
    rgb, depth, targets = zip(*batch)
    rgb = torch.stack(rgb)
    depth = torch.stack(depth)
    return rgb, depth, targets

class DepthFeatureExtractor:
    def __init__(self, neighborhood_size=10, threshold=1000):
        self.neighborhood_size = neighborhood_size
        self.threshold = threshold

    def process_depth_map(self, csv_path, segment_path=None):
        """Process depth data to extract meaningful features using FindLocalMax."""
        # Initialize FindLocalMax with the depth data
        local_max = FindLocalMax(
            csv_path=csv_path,
            segment_path=segment_path,
            threshold=self.threshold,
            ground_truth=None
        )

        # Get features from FindLocalMax
        maxima_coords, magnitude, orientation = local_max.detect_local_maxima(
            neighborhood_size=self.neighborhood_size,
            plot=False
        )

        # Create feature channels
        features = {
            'maxima': maxima_coords.to_numpy(),
            'magnitude': magnitude.to_numpy(),
            'orientation': orientation.to_numpy()
        }

        return features


class DepthStreamResNet(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super().__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.adaptation = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.p5_conv = nn.Conv2d(1024, 1024, kernel_size=1)
        self.p4_conv = nn.Conv2d(1024, 512, kernel_size=1)
        self.p3_conv = nn.Conv2d(1024, 256, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        features = self.adaptation(x)

        p5 = self.p5_conv(features)
        p4 = self.p4_conv(features)
        p4 = F.interpolate(p4, size=(features.shape[-2] * 2, features.shape[-1] * 2), mode='bilinear',
                           align_corners=False)
        p3 = self.p3_conv(features)
        p3 = F.interpolate(p3, size=(features.shape[-2] * 4, features.shape[-1] * 4), mode='bilinear',
                           align_corners=False)

        return p5, p4, p3


class MultiStreamYOLOResNet(nn.Module):
    def __init__(self, base_model='yolov8m.pt'):
        super().__init__()
        base = YOLO(base_model)
        self.backbone = base.model.model[0:9]
        self.head = base.model.model[9:]
        self.depth_stream = DepthStreamResNet(pretrained=True)

        # Adjust fusion modules to match YOLO dimensions
        self.fusion_modules = nn.ModuleDict({
            'p5_fusion': nn.Sequential(
                nn.Conv2d(576 + 1024, 576, kernel_size=1),
                nn.BatchNorm2d(576),
                nn.ReLU(inplace=True)
            ),
            'p4_fusion': nn.Sequential(
                nn.Conv2d(384 + 512, 384, kernel_size=1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True)
            ),
            'p3_fusion': nn.Sequential(
                nn.Conv2d(192 + 256, 192, kernel_size=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True)
            )
        })

        self.cross_attention = CrossScaleAttention()

    def forward(self, rgb, depth, targets=None):
        rgb_features = []
        x = rgb
        for m in self.backbone:
            x = m(x)
            rgb_features.append(x)

        # Get features at correct scales
        p5_rgb = rgb_features[-1]  # 576 channels
        p4_rgb = rgb_features[5]  # 384 channels
        p3_rgb = rgb_features[3]  # 192 channels

        # Get depth features
        p5_depth, p4_depth, p3_depth = self.depth_stream(depth)

        # Fuse features
        p5 = self.fusion_modules['p5_fusion'](torch.cat([p5_rgb, p5_depth], dim=1))
        p4 = self.fusion_modules['p4_fusion'](torch.cat([p4_rgb, p4_depth], dim=1))
        p3 = self.fusion_modules['p3_fusion'](torch.cat([p3_rgb, p3_depth], dim=1))

        # Process through YOLO head
        x = p5
        for m in self.head:
            if isinstance(m, nn.ModuleList):
                curr_x = []
                for i, subm in enumerate(m):
                    if i == 0:
                        curr_x.append(subm(x))
                    elif i == 1:
                        curr_x.append(subm(p4))
                    else:
                        curr_x.append(subm(p3))
                x = curr_x
            else:
                x = m(x)

        return x if targets is None else self.compute_loss(x, targets)

    def compute_loss(self, outputs, targets):
        """Compute YOLOv8 loss combining box, class, and object losses."""
        # Initialize loss components
        box_loss = torch.zeros(1, device=targets[0]['boxes'].device)
        obj_loss = torch.zeros(1, device=targets[0]['boxes'].device)
        cls_loss = torch.zeros(1, device=targets[0]['boxes'].device)

        # Parameters
        box_weight = 0.05
        obj_weight = 1.0
        cls_weight = 0.5

        # Process each output level (P3, P4, P5)
        for output in outputs:
            # Decode predictions
            pred_boxes = output[..., :4]  # Box predictions
            pred_obj = output[..., 4:5]  # Objectness predictions
            pred_cls = output[..., 5:]  # Class predictions

            # Get target assignments
            target_boxes = targets[0]['boxes']
            target_cls = targets[0]['labels']

            # Compute IoU between predictions and targets
            ious = box_iou(pred_boxes, target_boxes)

            # Box loss (GIoU Loss)
            box_loss += (1.0 - giou(pred_boxes, target_boxes)).mean()

            # Objectness loss (BCE)
            obj_targets = (ious > 0.5).float()
            obj_loss += F.binary_cross_entropy_with_logits(pred_obj, obj_targets)

            # Classification loss (BCE)
            cls_loss += F.binary_cross_entropy_with_logits(pred_cls, F.one_hot(target_cls, pred_cls.shape[-1]))

        # Combine losses with weights
        total_loss = box_weight * box_loss + obj_weight * obj_loss + cls_weight * cls_loss

        return total_loss

def box_iou(box1, box2):
    """Compute IoU between boxes."""
    # Convert to x1, y1, x2, y2 format
    b1_x1, b1_y1 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2
    b1_x2, b1_y2 = box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2
    b2_x1, b2_y1 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2
    b2_x2, b2_y2 = box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2

    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-6)

def giou(box1, box2):
    """Compute GIoU between boxes."""
    iou = box_iou(box1, box2)

    # Find enclosing box
    c_x1 = torch.min(box1[..., 0] - box1[..., 2] / 2, box2[..., 0] - box2[..., 2] / 2)
    c_y1 = torch.min(box1[..., 1] - box1[..., 3] / 2, box2[..., 1] - box2[..., 3] / 2)
    c_x2 = torch.max(box1[..., 0] + box1[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2)
    c_y2 = torch.max(box1[..., 1] + box1[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2)

    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)

    return iou - (c_area - (box1[..., 2] * box1[..., 3] + box2[..., 2] * box2[..., 3] - c_area)) / c_area

class CrossScaleAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.ones(3))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, features):
        weights = self.softmax(self.attention_weights)
        return [f * w for f, w in zip(features, weights)]


class DepthFeatureProcessor:
    def __init__(self, img_size=640):
        self.img_size = img_size

    def process_features(self, depth_features):
        """Convert depth features to ResNet-compatible input."""
        # Ensure all features have the same shape
        maxima = depth_features['maxima'].astype(np.float32)
        magnitude = depth_features['magnitude'].astype(np.float32)
        orientation = depth_features['orientation'].astype(np.float32)

        # Resize all features to match target size
        maxima = cv2.resize(maxima, (self.img_size, self.img_size))
        magnitude = cv2.resize(magnitude, (self.img_size, self.img_size))
        orientation = cv2.resize(orientation, (self.img_size, self.img_size))

        # Normalize features
        maxima = self.normalize_feature(maxima)
        magnitude = self.normalize_feature(magnitude)
        orientation = self.normalize_feature(orientation)

        # Stack channels
        depth_input = np.stack([maxima, magnitude, orientation], axis=0)
        return torch.from_numpy(depth_input).float()

    @staticmethod
    def normalize_feature(feature):
        min_val = np.min(feature)
        max_val = np.max(feature)
        if max_val > min_val:
            return (feature - min_val) / (max_val - min_val)
        return feature


class MultiStreamDatasetResNet(torch.utils.data.Dataset):
    def __init__(self, data_dir, img_size=640):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.samples = list(self.data_dir.glob('images/*.jpg'))
        self.depth_processor = DepthFeatureProcessor(img_size)

    def __getitem__(self, idx):
        img_path = self.samples[idx]

        # Load and preprocess RGB image
        rgb_img = cv2.imread(str(img_path))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (self.img_size, self.img_size))
        rgb_img = rgb_img.transpose(2, 0, 1)  # HWC to CHW format
        rgb_img = torch.from_numpy(rgb_img).float() / 255.0

        # Load depth features
        depth_feature_path = self.data_dir / 'depth_features' / f"{img_path.stem}.npz"
        depth_features = np.load(depth_feature_path)
        depth_tensor = self.depth_processor.process_features(depth_features)

        # Load labels
        label_path = self.data_dir / 'labels' / f"{img_path.stem}.txt"
        if label_path.exists() and label_path.stat().st_size > 0:
            labels = np.loadtxt(str(label_path))
            if len(labels.shape) == 1:
                labels = labels.reshape(1, -1)
            target = {
                'boxes': torch.tensor(labels[:, 1:5]),
                'labels': torch.tensor(labels[:, 0], dtype=torch.long)
            }
        else:
            target = {
                'boxes': torch.zeros((0, 4)),
                'labels': torch.zeros(0, dtype=torch.long)
            }

        return rgb_img, depth_tensor, target

    def __len__(self):
        return len(self.samples)


class PlacentaYOLODepthPipeline:
    def __init__(self, color_images_dir, depth_dir, gt_dir, output_dir, min_contour_area=100):
        """
        Initialize the pipeline for multi-stream YOLO training.

        Args:
            color_images_dir (str/Path): Directory with RGB images
            depth_dir (str/Path): Directory with depth CSV files
            gt_dir (str/Path): Directory with ground truth masks
            output_dir (str/Path): Directory for output
            min_contour_area (int): Minimum contour area to consider
        """
        self.color_images_dir = Path(color_images_dir)
        self.depth_dir = Path(depth_dir)
        self.gt_dir = Path(gt_dir)
        self.output_dir = Path(output_dir)
        self.min_contour_area = min_contour_area

        # Initialize depth feature extractor
        self.depth_extractor = DepthFeatureExtractor()

        # Create directory structure
        self.dataset_dir = self.output_dir / 'multistream_dataset'
        for split in ['train', 'val', 'test']:
            (self.dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / split / 'depth_features').mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    def extract_datetime(self, filename):
        """Extract datetime from filename."""
        match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
        return match.group(1) if match else None

    def prepare_dataset(self, split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}, seed=42):
        """Prepare dataset with both RGB and depth information."""
        random.seed(seed)

        # Match RGB images with depth maps and GT masks
        print("Matching RGB, depth, and ground truth files...")
        gt_files = {self.extract_datetime(f.name): f for f in self.gt_dir.glob('*.jpg')}
        depth_files = {self.extract_datetime(f.name): f for f in self.depth_dir.glob('*.csv')}

        matched_files = []
        for img_file in self.color_images_dir.glob('*.jpg'):
            img_datetime = self.extract_datetime(img_file.name)
            if img_datetime and img_datetime in gt_files and img_datetime in depth_files:
                matched_files.append((
                    img_file,
                    depth_files[img_datetime],
                    gt_files[img_datetime]
                ))


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
        for split, files in splits.items():
            print(f"\nProcessing {split} split...")
            for img_file, depth_file, gt_file in tqdm(files):
                shutil.copy2(img_file, self.dataset_dir / split / 'images' / img_file.name)

                # Extract and save depth features
                depth_features = self.depth_extractor.process_depth_map(
                    csv_path=str(depth_file),
                    segment_path=str(img_file)  # Use RGB image for segmentation
                )
                # Save depth features
                depth_feature_file = self.dataset_dir / split / 'depth_features' / f"{img_file.stem}.npz"
                np.savez(depth_feature_file, **depth_features)

                # Process ground truth mask to YOLO format
                mask = cv2.imread(str(gt_file))
                bboxes = self.mask_to_bboxes(mask)

                # Save YOLO format labels
                label_file = self.dataset_dir / split / 'labels' / f"{img_file.stem}.txt"
                if bboxes:
                    with open(label_file, 'w') as f:
                        for bbox in bboxes:
                            f.write(f"0 {' '.join(map(str, bbox))}\n")
                else:
                    label_file.touch()  # Create empty file for negative samples

    def mask_to_bboxes(self, mask):
        """Convert binary mask to YOLO format bounding boxes."""
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
            # Convert to YOLO format (normalized center coordinates and dimensions)
            x_center = (x + w / 2) / image_w
            y_center = (y + h / 2) / image_h
            width = w / image_w
            height = h / image_h

            bboxes.append([x_center, y_center, width, height])

        return bboxes

    def train_model(self, epochs=100, batch_size=16, model_size='m'):
        """Train the multi-stream model."""
        print("\nInitializing multi-stream training...")

        # Initialize model and trainer
        model = MultiStreamYOLOResNet(f'yolov8{model_size}.pt')
        trainer = MultiStreamYOLOTrainer(model, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Create datasets
        train_dataset = MultiStreamDatasetResNet(
            self.dataset_dir / 'train',
            img_size=640
        )

        val_dataset = MultiStreamDatasetResNet(
            self.dataset_dir / 'val',
            img_size=640
        )

        # Create data loaders with collate_fn
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )

        trainer.train(train_loader, val_loader, epochs=epochs, lr=0.001)

        return model

    def evaluate_model(self, model):
        """Evaluate the trained model."""
        print("\nEvaluating model...")

        test_dataset = MultiStreamDataset(
            self.dataset_dir / 'test',
            img_size=640
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )

        model.eval()
        results = []

        with torch.no_grad():
            for batch in test_loader:
                rgb_imgs, depth_features, labels = batch
                predictions = model(rgb_imgs, depth_features)
                results.append(predictions)

        return results


class MultiStreamDataset(Dataset):
    def __init__(self, data_dir, img_size=640):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.samples = list(self.data_dir.glob('images/*.jpg'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]

        # Load RGB image
        rgb_img = cv2.imread(str(img_path))
        rgb_img = cv2.resize(rgb_img, (self.img_size, self.img_size))
        rgb_img = torch.from_numpy(rgb_img).float() / 255.0

        # Load depth features
        depth_feature_path = self.data_dir / 'depth_features' / f"{img_path.stem}.npz"
        depth_features = np.load(depth_feature_path)
        depth_tensor = torch.from_numpy(np.stack([
            depth_features['maxima'],
            depth_features['magnitude'],
            depth_features['orientation']
        ])).float()

        label_path = self.data_dir / 'labels' / f"{img_path.stem}.txt"
        if label_path.exists() and label_path.stat().st_size > 0:
            labels = torch.from_numpy(np.loadtxt(str(label_path))).float()
        else:
            labels = torch.zeros((0, 5))  # Empty label tensor

        return rgb_img, depth_tensor, labels


def main():
    # Set paths
    root_dir = Path.cwd()
    color_images_dir = root_dir / "Images" / "color_images"
    depth_dir = root_dir / "Images" / "csv_files"
    gt_dir = root_dir / "Images" / "gt"
    output_dir = root_dir / "multistream_detection"

    # Initialize pipeline
    pipeline = PlacentaYOLODepthPipeline(
        color_images_dir=color_images_dir,
        depth_dir=depth_dir,
        gt_dir=gt_dir,
        output_dir=output_dir,
        min_contour_area=100
    )

    if not (pipeline.dataset_dir / 'train').exists():
        pipeline.prepare_dataset()

    # Train model
    model = pipeline.train_model(
        epochs=100,
        batch_size=16,
        model_size='m'
    )

    # Evaluate model
    results = pipeline.evaluate_model(model)


if __name__ == "__main__":
    main()