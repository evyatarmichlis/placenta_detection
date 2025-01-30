import os
import re
import shutil
from collections import Counter
from pathlib import Path
import random

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights
from find_local_max import FindLocalMax

matplotlib.use('Agg')
import matplotlib.pyplot as plt
class PlacentaHHADataset(Dataset):
    def __init__(self, data_dir, depth_features_dir, transform=None, is_train=False):
        self.data_dir = Path(data_dir)
        self.depth_features_dir = Path(depth_features_dir)
        self.is_train = is_train

        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.transform = transform if transform else (
            self.train_transform if is_train else self.base_transform
        )

        # Map date_string -> depth_file.npz
        self.depth_files = {}
        for depth_path in self.depth_features_dir.glob('*.npz'):
            date_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', depth_path.stem)
            if date_match:
                self.depth_files[date_match.group(1)] = depth_path

        self.samples = []
        # Add all samples (image path, depth path, label)
        for label_dir, label_value in [('positive', 1), ('negative', 0)]:
            dir_path = self.data_dir / label_dir
            if dir_path.exists():
                for img_path in dir_path.glob('*.jpg'):
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', img_path.stem)
                    if date_match and date_match.group(1) in self.depth_files:
                        depth_path = self.depth_files[date_match.group(1)]
                        self.samples.append((str(img_path), str(depth_path), label_value))

    def __getitem__(self, idx):
        img_path, depth_path, label = self.samples[idx]

        # ---- Load RGB image ----
        rgb_img = Image.open(img_path).convert('RGB')
        rgb_img = self.transform(rgb_img)

        # ---- Load HHA ----
        depth_data = np.load(depth_path)
        hha = depth_data['hha']  # (3, H, W)
        hha = torch.from_numpy(hha).float()  # Convert to PyTorch tensor
        hha = F.interpolate(hha.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

        return rgb_img, hha, label

    def __len__(self):
        return len(self.samples)

class PlacentaDataset(Dataset):
    def __init__(self, data_dir, depth_features_dir, transform=None, is_train=False):
        self.data_dir = Path(data_dir)
        self.depth_features_dir = Path(depth_features_dir)
        self.is_train = is_train

        # Same transforms as before for RGB
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform = transform if transform else (self.train_transform if is_train else self.base_transform)

        self.depth_files = {}
        for depth_path in self.depth_features_dir.glob('*.npz'):
            date_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', depth_path.stem)
            if date_match:
                self.depth_files[date_match.group(1)] = depth_path

        self.samples = []
        # Look in positive directory
        for label_dir, label_value in [('positive', 1), ('negative', 0)]:
            dir_path = self.data_dir / label_dir
            if dir_path.exists():
                for img_path in dir_path.glob('*.jpg'):
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', img_path.stem)
                    if date_match and date_match.group(1) in self.depth_files:
                        depth_path = self.depth_files[date_match.group(1)]
                        self.samples.append((str(img_path), str(depth_path), label_value))

        if len(self.samples) == 0:
            print(f"Warning: No samples found in {self.data_dir}")
            print(f"Available depth dates: {list(self.depth_files.keys())}")
            print(f"Available image files: {list(self.data_dir.glob('**/*.jpg'))}")

    def __getitem__(self, idx):
        img_path, depth_path, label = self.samples[idx]

        # Load RGB image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)  # Convert PIL image to numpy array
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # Convert to LAB color space
        l_channel, a_channel, b_channel = cv2.split(lab_image)  # Split channels
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab_image = cv2.merge((l_channel, a_channel, b_channel))
        image_clahe = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)  # Convert back to RGB
        image = Image.fromarray(image_clahe)
        image = self.transform(image)

        # Load depth features
        depth_data = np.load(depth_path)
        depth_feature = depth_data['orientation']
        depth_feature = torch.from_numpy(depth_feature).float().unsqueeze(0)

        depth_feature = F.interpolate(
            depth_feature.unsqueeze(0),  # Add batch dimension
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        return image, depth_feature, label

    def __len__(self):
        return len(self.samples)


class DualStreamPlacentaClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()

        # Start with pretrained ResNet50
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Modify first conv layer to accept 4 channels
        old_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
            stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias
        )

        # Initialize weights for the new conv layer
        with torch.no_grad():
            # Keep RGB weights from pretrained model
            self.backbone.features[0][0].weight[:, :3, :, :] = old_conv.weight
            # Initialize depth channel weights using mean of RGB weights
            self.backbone.features[0][0].weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)

        # Optionally freeze the backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze the modified first conv layer
            for param in self.backbone.features[0][0].parameters():
                param.requires_grad = True

        # Replace classifier head
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb, depth):
        x = torch.cat([rgb, depth], dim=1)
        return self.backbone(x)


class LateFusionTwoStream(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # RGB stream
        self.rgb_net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features_rgb = self.rgb_net.classifier[1].in_features
        self.rgb_net.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features_rgb, num_classes)
        )

        # HHA stream
        self.hha_net = efficientnet_b0(weights=None)
        in_features_hha = self.hha_net.classifier[1].in_features
        self.hha_net.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features_hha, num_classes)
        )

    def forward(self, rgb, hha):
        logits_rgb = self.rgb_net(rgb)  # Output shape: (B, num_classes)
        logits_hha = self.hha_net(hha)  # Output shape: (B, num_classes)

        # Late fusion: Add logits
        fused_logits = logits_rgb + logits_hha
        return fused_logits


class PlacentaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for rgb, hha, labels in train_loader:
            rgb, hha, labels = rgb.to(device), hha.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(rgb, hha)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} Train Acc: {train_acc:.2f}% Loss: {running_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for rgb, hha, labels in val_loader:
                rgb, hha, labels = rgb.to(device), hha.to(device), labels.to(device)
                outputs = model(rgb, hha)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = 100. * val_correct / val_total
        print(f"Validation Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_hha_model.pth")

    print("Best Validation Acc:", best_val_acc)


def evaluate_model(model, test_loader, device,seed):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for rgb, hha, labels in test_loader:
            rgb, hha, labels = rgb.to(device), hha.to(device), labels.to(device)
            outputs = model(rgb, hha)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Defect'],
                yticklabels=['Normal', 'Defect'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"results/cm_seed_{seed}.png")

    print(f"\nTest Set Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


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
        original_depth = np.genfromtxt(csv_path, delimiter=',')[1:, :]

        normalized_depth = local_max.read_csv_and_norm(save=False)
        maxima_coords, magnitude, orientation = local_max.detect_local_maxima(
            neighborhood_size=self.neighborhood_size,
            plot=False
        )

        # Create HHA-like feature map
        hha = np.stack([normalized_depth, orientation, magnitude], axis=0)

        # Save all relevant features
        features = {
            'hha': hha,  # HHA representation
            'maxima': maxima_coords.to_numpy(),
            'magnitude': magnitude.to_numpy(),
            'orientation': orientation.to_numpy(),
            'original_depth': original_depth,
            'normalized_depth': normalized_depth
        }

        return features

class BalancedPlacentaDataOrganizer:
    def __init__(self, color_images_dir, depth_dir, gt_dir, output_base_dir, num_ensembles=1):
        self.color_images_dir = Path(color_images_dir)
        self.depth_dir = Path(depth_dir)
        self.gt_dir = Path(gt_dir)
        self.output_base_dir = Path(output_base_dir)
        self.num_ensembles = num_ensembles
        self.depth_extractor = DepthFeatureExtractor()

    def organize_datasets(self, split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}, seed=42):
        self._remove_output_directory()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        image_files = self._get_matched_files()
        random.shuffle(image_files)

        # Calculate split sizes
        total_samples = len(image_files)
        train_size = int(total_samples * split_ratios['train'])
        val_size = int(total_samples * split_ratios['val'])

        # Split into train/val/test
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size + val_size]
        test_files = image_files[train_size + val_size:]

        # Separate training positives and negatives
        train_positives = [f for f in train_files if f[3] == 'positive']
        train_negatives = [f for f in train_files if f[3] == 'negative']

        print(f"\nTraining set composition:")
        print(f"Positive samples: {len(train_positives)}")
        print(f"Negative samples: {len(train_negatives)}")

        # Create balanced splits for each ensemble member
        for ensemble_idx in range(self.num_ensembles):
            ensemble_dir = self.output_base_dir / f'ensemble_{ensemble_idx}'

            # Create directory structure
            for split in ['train', 'val', 'test']:
                (ensemble_dir / split / 'images').mkdir(parents=True, exist_ok=True)
                (ensemble_dir / split / 'depth_features').mkdir(parents=True, exist_ok=True)
                for label in ['positive', 'negative']:
                    (ensemble_dir / split / label).mkdir(parents=True, exist_ok=True)

            # Balance training set
            sample_size = min(len(train_positives), len(train_negatives))
            model_positives = random.sample(train_positives, sample_size)
            model_negatives = random.sample(train_negatives, sample_size)
            balanced_train = model_positives + model_negatives
            random.shuffle(balanced_train)

            splits = {
                'train': balanced_train,
                'val': val_files,
                'test': test_files
            }

            # Process each split
            for split, files in splits.items():
                print(f"\nProcessing {split} split for ensemble {ensemble_idx}...")
                for img_file, depth_file, gt_file, label in tqdm(files):
                    # Copy RGB image
                    dest_img_path = ensemble_dir / split / label / img_file.name
                    shutil.copy2(img_file, dest_img_path)

                    # Process and save depth features
                    depth_features = self.depth_extractor.process_depth_map(
                        csv_path=str(depth_file),
                        segment_path=str(img_file)
                    )
                    depth_feature_file = ensemble_dir / split / 'depth_features' / f"{img_file.stem}.npz"
                    np.savez(depth_feature_file, **depth_features)

    def _get_matched_files(self):
        """Get all matched RGB, depth, and GT files with labels."""
        gt_files = {self.extract_datetime(f.name): f for f in self.gt_dir.glob('*.jpg')}
        depth_files = {self.extract_datetime(f.name): f for f in self.depth_dir.glob('*.csv')}

        matched_files = []
        for img_file in self.color_images_dir.glob('*.jpg'):
            img_datetime = self.extract_datetime(img_file.name)
            if img_datetime and img_datetime in gt_files and img_datetime in depth_files:
                # Check if it's a positive or negative sample
                mask = Image.open(gt_files[img_datetime]).convert('L')
                mask_array = np.array(mask)
                label = 'positive' if np.any(mask_array > 0) else 'negative'

                matched_files.append((
                    img_file,
                    depth_files[img_datetime],
                    gt_files[img_datetime],
                    label
                ))

        return matched_files

    def extract_datetime(self, filename):
        """Extract datetime from filename."""
        match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
        return match.group(1) if match else None

    def _remove_output_directory(self):
        if os.path.exists(self.output_base_dir):
            print(f"\nRemoving existing output directory: {self.output_base_dir}")
            shutil.rmtree(self.output_base_dir)
        os.makedirs(self.output_base_dir)

    def _print_dataset_statistics(self):
        # Track all used samples
        used_positives = set()
        used_negatives = set()

        for ensemble_idx in range(self.num_ensembles):
            ensemble_dir = self.output_base_dir / f'ensemble_{ensemble_idx}'
            train_pos_path = ensemble_dir / 'train' / 'positive'
            train_neg_path = ensemble_dir / 'train' / 'negative'

            pos_files = {f.name for f in train_pos_path.glob('*.jpg')}
            neg_files = {f.name for f in train_neg_path.glob('*.jpg')}

            used_positives.update(pos_files)
            used_negatives.update(neg_files)

            print(f"\nEnsemble {ensemble_idx}:")
            for split in ['train', 'val', 'test']:
                pos_count = len(list((ensemble_dir / split / 'positive').glob('*.jpg')))
                neg_count = len(list((ensemble_dir / split / 'negative').glob('*.jpg')))
                split_total = pos_count + neg_count
                print(f'{split}: Total={split_total}, '
                      f'Positive={pos_count} ({pos_count / split_total * 100:.1f}%), '
                      f'Negative={neg_count} ({neg_count / split_total * 100:.1f}%)')

        print("\nOverall sample coverage:")
        print(f"Unique positive samples used: {len(used_positives)}")
        print(f"Unique negative samples used: {len(used_negatives)}")

class EnsemblePlacentaClassifier:
    def __init__(self, num_ensembles=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_ensembles = num_ensembles
        # Initialize with DualStreamPlacentaClassifier instead of PlacentaClassifier
        self.models = [DualStreamPlacentaClassifier().to(self.device) for _ in range(num_ensembles)]

    def train_ensemble(self, ensemble_datasets, num_epochs=30):
        for idx, model in enumerate(self.models):
            best_model = self.models[idx]
            print(f"\nTraining model {idx + 1}/{len(self.models)}")
            train_loader = DataLoader(ensemble_datasets[idx]['train'], batch_size=8, shuffle=True)
            val_loader = DataLoader(ensemble_datasets[idx]['val'], batch_size=8)

            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

            best_val_acc = 0
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for rgb, depth, labels in train_loader:
                    rgb, depth, labels = rgb.to(self.device), depth.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()

                    outputs = model(rgb, depth)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                train_acc = 100. * correct / total


                # Validation
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for rgb, depth, labels in val_loader:
                        rgb, depth, labels = rgb.to(self.device), depth.to(self.device), labels.to(self.device)
                        outputs = model(rgb, depth)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()

                val_acc = 100. * correct / total
                train_loss = running_loss / len(train_loader)
                print(f'Epoch {epoch + 1}/{num_epochs}:')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'Val Acc: {val_acc:.2f}%')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), f'best_dual_stream_model_{idx}.pth')
                    best_model = self.models[idx]
            self.models[idx] = best_model

    def predict(self, rgb, depth):
        predictions = []
        probabilities = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                outputs = model(rgb, depth)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                predictions.append(predicted)
                probabilities.append(probs)

        predictions = torch.stack(predictions)
        probabilities = torch.stack(probabilities)
        mean_probs = torch.mean(probabilities, dim=0)
        majority_vote = torch.mode(predictions, dim=0).values
        return majority_vote, mean_probs

    def evaluate_ensemble(self, test_loader):
        all_labels = []
        ensemble_preds = []
        individual_preds = [[] for _ in self.models]

        # Get predictions
        for rgb, depth, labels in test_loader:
            rgb, depth = rgb.to(self.device), depth.to(self.device)
            predictions = self.predict(rgb, depth)
            majority_vote = predictions[0]  # Get just the majority vote
            ensemble_preds.extend(majority_vote.cpu().numpy())
            all_labels.extend(labels.numpy())

            # Get individual model predictions
            for i, model in enumerate(self.models):
                model.eval()
                with torch.no_grad():
                    outputs = model(rgb, depth)
                    _, predicted = outputs.max(1)
                    individual_preds[i].extend(predicted.cpu().numpy())

        # Calculate ensemble metrics
        ensemble_metrics = {
            'accuracy': accuracy_score(all_labels, ensemble_preds),
            'precision': precision_score(all_labels, ensemble_preds, average='weighted'),
            'recall': recall_score(all_labels, ensemble_preds, average='weighted'),
            'f1': f1_score(all_labels, ensemble_preds, average='weighted')
        }

        # Calculate confusion matrix for ensemble
        ensemble_cm = confusion_matrix(all_labels, ensemble_preds)

        # Calculate metrics for individual models
        individual_metrics = []
        for i, preds in enumerate(individual_preds):
            metrics = {
                'model_id': i,
                'accuracy': accuracy_score(all_labels, preds),
                'precision': precision_score(all_labels, preds, average='weighted'),
                'recall': recall_score(all_labels, preds, average='weighted'),
                'f1': f1_score(all_labels, preds, average='weighted')
            }
            individual_metrics.append(metrics)

        return ensemble_metrics, ensemble_cm, individual_metrics


def main():
    # Set paths
    root_dir = Path.cwd()
    color_images_dir = root_dir / "Images" / "masked_images"
    depth_dir = root_dir / "Images" / "csv_files"
    gt_dir = root_dir / "Images" / "gt"
    seed = 2  # Fixed seed for reproducibility
    output_base_dir = root_dir / f"dual_stream_dataset_split_seed_{seed}"
    # Organize the dataset if it doesn't exist
    if not output_base_dir.exists():
        print(f"\nOrganizing dataset for seed {seed}...")
        organizer = BalancedPlacentaDataOrganizer(
            color_images_dir=color_images_dir,
            depth_dir=depth_dir,
            gt_dir=gt_dir,
            output_base_dir=output_base_dir
        )
        organizer.organize_datasets(seed=seed)
    else:
        print(f"\nUsing existing dataset for seed {seed}")

    # Initialize the dataset
    train_dataset = PlacentaHHADataset(
        data_dir=output_base_dir / 'ensemble_0' / 'train',
        depth_features_dir=output_base_dir / 'ensemble_0' / 'train' / 'depth_features',
        is_train=True
    )
    val_dataset = PlacentaHHADataset(
        data_dir=output_base_dir / 'ensemble_0' / 'val',
        depth_features_dir=output_base_dir / 'ensemble_0' / 'val' / 'depth_features'
    )
    test_dataset = PlacentaHHADataset(
        data_dir=output_base_dir / 'ensemble_0' / 'test',
        depth_features_dir=output_base_dir / 'ensemble_0' / 'test' / 'depth_features'
    )

    # Create dataloaders
    labels = [sample[2] for sample in train_dataset.samples]  # Extract labels from dataset
    label_counts = Counter(labels)  # Count occurrences of each label
    print(f"Class distribution in training set: {label_counts}")


    # Create a WeightedRandomSampler to balance the classes
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize the dual-stream model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LateFusionTwoStream(num_classes=2).to(device)

    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    print("\nStarting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, device=device)

    # Evaluate the model
    print("\nEvaluating on the test set...")
    evaluate_model(model, test_loader, device,seed=seed)


if __name__ == "__main__":
    main()

