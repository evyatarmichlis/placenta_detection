import os
import shutil
from pathlib import Path
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class PlacentaDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=False):
        self.data_dir = Path(data_dir)
        self.is_train = is_train

        # Base transforms for all sets
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Additional augmentations for training
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)
        ])

        # Use provided transform if given, otherwise use default transforms
        self.transform = transform if transform else (self.train_transform if is_train else self.base_transform)

        self.samples = []
        for label in ['positive', 'negative']:
            label_dir = self.data_dir / label
            for img_path in label_dir.glob('*.jpg'):
                if not img_path.name.startswith('gt_'):  # Skip GT files
                    self.samples.append((str(img_path), 1 if label == 'positive' else 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label



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


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device,seed):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_placenta_model_seed_{seed}.pth')


def evaluate_model(model, test_loader, device,seed):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
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


class BalancedPlacentaDataOrganizer:
    def __init__(self, color_images_dir, gt_dir, output_base_dir, num_ensembles=5):
        self.color_images_dir = Path(color_images_dir)
        self.gt_dir = Path(gt_dir)
        self.output_base_dir = Path(output_base_dir)
        self.num_ensembles = num_ensembles

    def organize_datasets(self, split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}, seed=42):
        # Clean up output directory first
        self._remove_output_directory()

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Get all images and their labels
        image_files = self._get_labeled_images()
        random.shuffle(image_files)

        # Calculate split sizes
        total_samples = len(image_files)
        train_size = int(total_samples * split_ratios['train'])
        val_size = int(total_samples * split_ratios['val'])
        test_size = total_samples - train_size - val_size

        # Split into train/val/test
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size + val_size]
        test_files = image_files[train_size + val_size:]

        # Separate training positives and negatives
        train_positives = [f for f in train_files if f[1] == 'positive']
        train_negatives = [f for f in train_files if f[1] == 'negative']

        print(f"\nTraining set composition:")
        print(f"Positive samples: {len(train_positives)}")
        print(f"Negative samples: {len(train_negatives)}")

        # Create different balanced splits for each ensemble member
        for ensemble_idx in range(self.num_ensembles):
            ensemble_dir = self.output_base_dir / f'ensemble_{ensemble_idx}'

            # Randomly sample positives
            sample_size = min(len(train_positives), len(train_negatives))
            model_positives = random.sample(train_positives, sample_size)
            # Randomly sample negatives to match positive size
            model_negatives = random.sample(train_negatives, sample_size)

            # Create balanced training set for this model
            balanced_train = model_positives + model_negatives
            random.shuffle(balanced_train)

            splits = {
                'train': balanced_train,
                'val': val_files,
                'test': test_files
            }

            # Create directories and copy files
            self._create_split_directories(ensemble_dir, splits)

            print(f"\nEnsemble {ensemble_idx} training set:")
            print(f"Positives: {len(model_positives)}")
            print(f"Negatives: {len(model_negatives)}")

        print(f"\nCreated {self.num_ensembles} ensemble datasets with random sampling")
        self._print_dataset_statistics()

    def _get_labeled_images(self):
        def extract_datetime(filename):
            import re
            match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
            return match.group(1) if match else None

        gt_files = {extract_datetime(f.name): f for f in self.gt_dir.glob('*.jpg')}
        image_files = []

        for img_file in self.color_images_dir.glob('*.jpg'):
            img_datetime = extract_datetime(img_file.name)
            if img_datetime and img_datetime in gt_files:
                mask = Image.open(gt_files[img_datetime]).convert('L')
                mask_array = np.array(mask)
                is_defect = np.any(mask_array > 0)
                image_files.append((img_file, 'positive' if is_defect else 'negative'))

        return image_files

    def _create_split_directories(self, ensemble_dir, splits):
        if os.path.exists(ensemble_dir):
            shutil.rmtree(ensemble_dir)

        for split in splits:
            for label in ['positive', 'negative']:
                os.makedirs(ensemble_dir / split / label, exist_ok=True)

        for split, files in splits.items():
            for img_file, label in files:
                dest_path = ensemble_dir / split / label / img_file.name
                shutil.copy2(img_file, dest_path)

    def _remove_output_directory(self):
        """Remove the entire output directory and its contents if it exists."""
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
    def __init__(self, num_models= 5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_models = num_models
        self.models = [PlacentaClassifier().to(self.device) for _ in range(num_models)]

    def train_ensemble(self, ensemble_datasets, num_epochs=30):
        for idx, model in enumerate(self.models):
            print(f"\nTraining model {idx + 1}/{len(self.models)}")
            train_loader = DataLoader(ensemble_datasets[idx]['train'], batch_size=32, shuffle=True)
            val_loader = DataLoader(ensemble_datasets[idx]['val'], batch_size=32)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

            best_val_acc = 0
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                train_acc = 100. * correct / total
                train_loss = running_loss / len(train_loader)
                # Validation
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()

                val_acc = 100. * correct / total
                # print(f'Epoch {epoch + 1}/{num_epochs}:')
                # print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                # print(f'Val Acc: {val_acc:.2f}%')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), f'best_model_{idx}.pth')

    def predict(self, x):
        predictions = []
        probabilities = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                predictions.append(predicted)
                probabilities.append(probs)

        # Return both majority vote and mean probabilities
        predictions = torch.stack(predictions)
        probabilities = torch.stack(probabilities)
        mean_probs = torch.mean(probabilities, dim=0)
        majority_vote = torch.mode(predictions, dim=0).values
        return majority_vote, mean_probs

    def evaluate_ensemble(self, test_loader):
        """
        Evaluate the ensemble model and individual models on test data
        """
        all_labels = []
        ensemble_preds = []
        individual_preds = [[] for _ in self.models]

        # Get predictions
        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            predictions = self.predict(inputs)
            if isinstance(predictions, tuple):
                majority_vote = predictions[0]  # Get just the majority vote
            else:
                majority_vote = predictions
            ensemble_preds.extend(majority_vote.cpu().numpy())
            all_labels.extend(labels.numpy())

            # Get individual model predictions
            for i, model in enumerate(self.models):
                model.eval()
                with torch.no_grad():
                    outputs = model(inputs)
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
    # Set up paths
    for seed in [2,88,107,3345]:
        current_file = Path(__file__)
        root_dir = current_file.parent
        color_images_dir = root_dir / "Images" / "color_images"
        gt_dir = root_dir / "Images" / "gt"
        output_base_dir = root_dir / f"dataset_split_seed_{seed}"
        organizer = BalancedPlacentaDataOrganizer(color_images_dir, gt_dir, output_base_dir)
        organizer.organize_datasets(seed=seed)

        # Train ensemble
        ensemble = EnsemblePlacentaClassifier()

        # Load datasets for each ensemble model
        ensemble_datasets = []
        for i in range(ensemble.num_models):
            datasets = {
                'train': PlacentaDataset(output_base_dir / f'ensemble_{i}' / 'train', is_train=True),
                'val': PlacentaDataset(output_base_dir / f'ensemble_{i}' / 'val'),
                'test': PlacentaDataset(output_base_dir / f'ensemble_{i}' / 'test')
            }
            ensemble_datasets.append(datasets)

        # Train the ensemble
        ensemble.train_ensemble(ensemble_datasets)
        print("\nEvaluating ensemble performance...")
        test_loader = DataLoader(ensemble_datasets[0]['test'], batch_size=32, shuffle=False)
        ensemble_metrics, ensemble_cm, individual_metrics = ensemble.evaluate_ensemble(test_loader)

        # Print results
        print(f"\nEnsemble Model Performance: seed {seed}")
        for metric, value in ensemble_metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

        print("\nIndividual Model Performance:")
        for i, metrics in enumerate(individual_metrics):
            print(f"\nModel {i}:")
            for metric, value in metrics.items():
                if metric != 'model_id':
                    print(f"{metric.capitalize()}: {value:.4f}")
        print("###################ensemble_cm###################")
        print(ensemble_cm)

if __name__ == "__main__":
    main()




# baseline

# Mean and standard deviation of metrics:
# accuracy:
# Mean: 0.8078
# Std:  0.0048
# precision:
# Mean: 0.8253
# Std:  0.0109
# recall:
# Mean: 0.9387
# Std:  0.0234
# f1:
# Mean: 0.8780
# Std:  0.0045

# # with augmentation:
# accuracy:
# Mean: 0.8039
# Std:  0.0164
# precision:
# Mean: 0.8350
# Std:  0.0042
# recall:
# Mean: 0.9152
# Std:  0.0251
# f1:
# Mean: 0.8731
# Std:  0.0110
#
# Process finished with exit code 0



# Evaluating ensemble performance...
# [[25  0]
#  [ 6 23]]
#
# Ensemble Model Performance:
# Accuracy: 0.8889
# Precision: 0.9104
# Recall: 0.8889
# F1: 0.8884
#
# Individual Model Performance:
#
# Model 0:
# Accuracy: 0.7222
# Precision: 0.7593
# Recall: 0.7222
# F1: 0.7052
#
# Model 1:
# Accuracy: 0.7963
# Precision: 0.8370
# Recall: 0.7963
# F1: 0.7931
#
# Model 2:
# Accuracy: 0.8148
# Precision: 0.8176
# Recall: 0.8148
# F1: 0.8151
#
# Model 3:
# Accuracy: 0.7778
# Precision: 0.7778
# Recall: 0.7778
# F1: 0.7778
#
# Model 4:
# Accuracy: 0.8148
# Precision: 0.8482
# Recall: 0.8148
# F1: 0.8128
#
# Process finished with exit code 0