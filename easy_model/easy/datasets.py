import pickle
from collections import Counter

from torchvision import transforms, datasets
from args import args
import numpy as np
import torch
import json
import os

class CPUDataset():
    def __init__(self, data, targets, transforms = [], batch_size = args.batch_size, use_hd = False):
        self.data = data
        if torch.is_tensor(data):
            self.length = data.shape[0]
        else:
            self.length = len(self.data)
        self.targets = targets
        assert(self.length == targets.shape[0])
        self.batch_size = batch_size
        self.transforms = transforms
        self.use_hd = use_hd
    def __getitem__(self, idx):
        if self.use_hd:
            elt = transforms.ToTensor()(np.array(Image.open(self.data[idx]).convert('RGB')))
        else:
            elt = self.data[idx]
        return self.transforms(elt), self.targets[idx]
    def __len__(self):
        return self.length
        
class EpisodicCPUDataset():
    def __init__(self, data, num_classes, transforms = [], episode_size = args.batch_size, use_hd = False):
        self.data = data
        if torch.is_tensor(data):
            self.length = data.shape[0]
        else:
            self.length = len(self.data)
        self.episode_size = (episode_size // args.n_ways) * args.n_ways
        self.transforms = transforms
        self.use_hd = use_hd
        self.num_classes = num_classes
        self.targets = []
        self.indices = []
        self.corrected_length = args.episodes_per_epoch * self.episode_size
        episodes = args.episodes_per_epoch
        for i in range(episodes):
            classes = np.random.permutation(np.arange(self.num_classes))[:args.n_ways]
            for c in range(args.n_ways):
                class_indices = np.random.permutation(np.arange(self.length // self.num_classes))[:self.episode_size // args.n_ways]
                self.indices += list(class_indices + classes[c] * (self.length // self.num_classes))
                self.targets += [c] * (self.episode_size // args.n_ways)
        self.indices = np.array(self.indices)
        self.targets = np.array(self.targets)

    def generate_next_episode(self, idx):
        if idx >= args.episodes_per_epoch:
            idx = 0
        classes = np.random.permutation(np.arange(self.num_classes))[:args.n_ways]
        n_samples = (self.episode_size // args.n_ways)
        for c in range(args.n_ways):
            class_indices = np.random.permutation(np.arange(self.length // self.num_classes))[:self.episode_size // args.n_ways]
            self.indices[idx * self.episode_size + c * n_samples: idx * self.episode_size + (c+1) * n_samples] = (class_indices + classes[c] * (self.length // self.num_classes))

    def __getitem__(self, idx):
        if idx % self.episode_size == 0:
            self.generate_next_episode((idx // self.episode_size) + 1)
        if self.use_hd:
            elt = transforms.ToTensor()(np.array(Image.open(self.data[self.indices[idx]]).convert('RGB')))
        else:
            elt = self.data[self.indices[idx]]
        return self.transforms(elt), self.targets[idx]

    def __len__(self):
        return self.corrected_length

class Dataset():
    def __init__(self, data, targets, transforms = [], batch_size = args.batch_size, shuffle = True, device = args.dataset_device):
        if torch.is_tensor(data):
            self.length = data.shape[0]
            self.data = data.to(device)
        else:
            self.length = len(self.data)
        self.targets = targets.to(device)
        assert(self.length == targets.shape[0])
        self.batch_size = batch_size
        self.transforms = transforms
        self.permutation = torch.arange(self.length)
        self.n_batches = self.length // self.batch_size + (0 if self.length % self.batch_size == 0 else 1)
        self.shuffle = shuffle
    def __iter__(self):
        if self.shuffle:
            self.permutation = torch.randperm(self.length)
        for i in range(self.n_batches):
            if torch.is_tensor(self.data):
                yield self.transforms(self.data[self.permutation[i * self.batch_size : (i+1) * self.batch_size]]), self.targets[self.permutation[i * self.batch_size : (i+1) * self.batch_size]]
            else:
                yield torch.stack([self.transforms(self.data[x]) for x in self.permutation[i * self.batch_size : (i+1) * self.batch_size]]), self.targets[self.permutation[i * self.batch_size : (i+1) * self.batch_size]]
    def __len__(self):
        return self.n_batches

class EpisodicDataset():
    def __init__(self, data, num_classes, transforms = [], episode_size = args.batch_size, device = args.dataset_device, use_hd = False):
        if torch.is_tensor(data):
            self.length = data.shape[0]
            self.data = data.to(device)
        else:
            self.data = data
            self.length = len(self.data)
        self.episode_size = episode_size
        self.transforms = transforms
        self.num_classes = num_classes
        self.n_batches = args.episodes_per_epoch
        self.use_hd = use_hd
        self.device = device
    def __iter__(self):
        for i in range(self.n_batches):
            classes = np.random.permutation(np.arange(self.num_classes))[:args.n_ways]
            indices = []
            for c in range(args.n_ways):
                class_indices = np.random.permutation(np.arange(self.length // self.num_classes))[:self.episode_size // args.n_ways]
                indices += list(class_indices + classes[c] * (self.length // self.num_classes))
            targets = torch.repeat_interleave(torch.arange(args.n_ways), self.episode_size // args.n_ways).to(self.device)
            if torch.is_tensor(self.data):
                yield self.transforms(self.data[indices]), targets
            else:
                if self.use_hd:
                    yield torch.stack([self.transforms(transforms.ToTensor()(np.array(Image.open(self.data[x]).convert('RGB'))).to(self.device)) for x in indices]), targets
                else:
                    yield torch.stack([self.transforms(self.data[x].to(self.device)) for x in indices]), targets
    def __len__(self):
        return self.n_batches

def iterator(data, target, transforms, forcecpu = False, shuffle = True, use_hd = False):
    if args.dataset_device == "cpu" or forcecpu:
        dataset = CPUDataset(data, target, transforms, use_hd = use_hd)
        return torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = shuffle, num_workers = min(8, os.cpu_count()))
    else:
        return Dataset(data, target, transforms, shuffle = shuffle)

def episodic_iterator(data, num_classes, transforms, forcecpu = False, use_hd = False):
    if args.dataset_device == "cpu" or forcecpu:
        dataset = EpisodicCPUDataset(data, num_classes, transforms, use_hd = use_hd)
        return torch.utils.data.DataLoader(dataset, batch_size = (args.batch_size // args.n_ways) * args.n_ways, shuffle = False, num_workers = min(8, os.cpu_count()))
    else:
        return EpisodicDataset(data, num_classes, transforms, use_hd = use_hd)


import random





from PIL import Image


def miniImageNet(use_hd=True):
    datasets = {}
    for subset in ["train", "val", "test"]:
        data = []
        target = []
        json_path = os.path.join(args.dataset_path, subset + '.json')

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        image_names = json_data['image_names']
        image_labels = json_data['image_labels']
        label_names = json_data['label_names']

        classes = list(set(image_labels))  # Extract unique classes
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for img_name, img_label in zip(image_names, image_labels):
            target.append(class_to_idx[img_label])
            path = os.path.join(args.dataset_path, 'images', img_name)

            if not use_hd:
                image = transforms.ToTensor()(np.array(Image.open(path).convert('RGB')))
                data.append(image)
            else:
                data.append(path)

        datasets[subset] = [data, torch.LongTensor(target)]
    print("load imageNet")
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    train_transforms = torch.nn.Sequential(transforms.RandomResizedCrop(84), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), transforms.RandomHorizontalFlip(), norm)
    all_transforms = torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84), norm) if args.sample_aug == 1 else torch.nn.Sequential(transforms.RandomResizedCrop(84), norm)
    if args.episodic:
        train_loader = episodic_iterator(datasets["train"][0], 64, transforms = train_transforms, forcecpu = True, use_hd = True)
    else:
        train_loader = iterator(datasets["train"][0], datasets["train"][1], transforms = train_transforms, forcecpu = True, use_hd = use_hd)
    train_clean = iterator(datasets["train"][0], datasets["train"][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = use_hd)
    val_loader = iterator(datasets["val"][0], datasets["val"][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = use_hd)
    test_loader = iterator(datasets["test"][0], datasets["test"][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = use_hd)
    return (train_loader, train_clean, val_loader, test_loader), [3, 84, 84], (64, 16, 20, 600), True, False


def evyatar_dataset():
    datasets = {}
    for subset in ["train", "val", "test"]:
        data = []
        target = []
        data_path = args.dataset_path+'json_files'
        json_path = os.path.join(data_path, subset + '.json')

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        image_names = json_data['image_names']
        image_labels = json_data['image_labels']
        classes = list(set(image_labels))  # Extract unique classes
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for img_name, img_label in zip(image_names, image_labels):
            target.append(class_to_idx[img_label])
            path = os.path.join(args.dataset_path, 'images', img_name)
            image = transforms.ToTensor()(np.array(Image.open(path).convert('RGB')))
            data.append(image)
        datasets[subset] = [data, torch.LongTensor(target)]
    print("load evyatar")
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    train_transforms = torch.nn.Sequential(transforms.RandomResizedCrop(84), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), transforms.RandomHorizontalFlip(), norm)
    all_transforms = torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84), norm) if args.sample_aug == 1 else torch.nn.Sequential(transforms.RandomResizedCrop(84), norm)
    if args.episodic:
        train_loader = episodic_iterator(datasets["train"][0], 64, transforms = train_transforms, forcecpu = True, use_hd = True)
    else:
        train_loader = iterator(datasets["train"][0], datasets["train"][1], transforms = train_transforms, forcecpu = True, use_hd = False)
    train_clean = iterator(datasets["train"][0], datasets["train"][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = False)
    val_loader = iterator(datasets["val"][0], datasets["val"][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = False)
    test_loader = iterator(datasets["test"][0], datasets["test"][1], transforms = all_transforms, forcecpu = True, shuffle = False, use_hd = False)
    train_label_number = len(np.unique(datasets["train"][1].numpy()))
    test_val_number =  len(np.unique(datasets["val"][1].numpy()))
    test_label_number = len(np.unique(datasets["test"][1].numpy()))

    return ((train_loader, train_clean, val_loader, test_loader), [3, 84, 84],
            (train_label_number+test_val_number+test_label_number, test_val_number, test_label_number, len(datasets["train"][1])//train_label_number), True, False)



def find_depth_path(rgb_path):
    """Attempts to find a corresponding depth image for a given RGB image path.

    Args:
        rgb_path (str): Path to the RGB image.

    Returns:
        str: Path to the depth image if found, otherwise None.
    """

    print(rgb_path)
    base_dir, filename = os.path.split(rgb_path)
    depth_dir = base_dir.replace('RGB_50', 'depth_50')
    for curr_ext in ['.png', '.jpg']:
        for ext in ['.png', '.jpg', '.jpeg','.bmp']:

            potential_depth_path = os.path.join(depth_dir, filename.replace(curr_ext, ext))
            print(f"potential_depth_path_{potential_depth_path}")

            if os.path.exists(potential_depth_path):
                return potential_depth_path
    return None













class DepthDataset:
    def __init__(self, args, ):
        self.args = args
        self.subsets = ["train", "val", "test"]

        # Calculate subset sizes
        self.subset_sizes = [
            len(json.load(open(os.path.join(args.dataset_path + "json_files", subset + ".json"))))
            for subset in self.subsets
        ]

        # Preload data
        self.data = {}
        self.targets = {}
        for subset in self.subsets:
            self.data[subset], self.targets[subset] = self.load_subset(subset)

    def load_subset(self, subset):
        data_path = self.args.dataset_path + 'json_files'
        json_path = os.path.join(data_path, subset + '.json')
        images = []
        targets = []

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        image_names = json_data['image_names']
        image_labels = json_data['image_labels']

        classes = list(set(image_labels))
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for img_name, img_label in zip(image_names, image_labels):
            target = class_to_idx[img_label]
            rgb_path = os.path.join(args.dataset_path, 'images', img_name)
            rgb_image = transforms.ToTensor()(np.array(Image.open(rgb_path).convert('RGB')))

            depth_path = find_depth_path(rgb_path)  # You'll need to provide this function
            if depth_path:
                depth_image = transforms.ToTensor()(np.array(Image.open(depth_path)))

            else:
                depth_image = rgb_image.mean(dim=0, keepdims=True)

            # image = torch.cat([rgb_image, depth_image], dim=0)
            image = (rgb_image, depth_image)
            images.append(image)
            targets.append(target)

        return images, torch.LongTensor(targets)

    def determine_max_size(self,subset):
        max_height = 0
        max_width = 0
        data_path = args.dataset_path + 'json_files'
        json_path = os.path.join(data_path, subset + '.json')
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        image_names = json_data['image_names']
        for img_name in image_names:
            rgb_path = os.path.join(args.dataset_path, 'images', img_name)
            rgb_image = Image.open(rgb_path).convert('RGB')
            h, w = rgb_image.size
            max_height = max(max_height, h)
            max_width = max(max_width, w)

        return max_height, max_width

    def pad_image(self, image, max_height, max_width):
        current_height, current_width = image.shape[1:]
        pad_bottom = max_height - current_height
        pad_right = max_width - current_width
        padding = ((0, 0), (0, pad_bottom), (0, pad_right))
        padded_image = np.pad(image, padding, mode='constant', constant_values=0)

        return padded_image

    def __getitem__(self, index):
        subset = self.subsets[index // self.subset_sizes[0]]

        rgb_image, depth_image = self.data[subset][index]  # Get original images
        target = self.targets[subset][index]

        max_height, max_width = self.determine_max_size(subset)

        depth_image = depth_image.resize((rgb_image.size()[2], rgb_image.size()[1]))

        image = torch.cat([rgb_image, depth_image], dim=0)
        image = self.pad_image(image, max_height, max_width)

        return image, target

    def calculate_mean_std(self, dataset_loader):
        """Calculates mean and standard deviation per channel for RGB and depth images separately"""

        rgb_channels_sum, rgb_channels_squared_sum, num_rgb_batches = 0, 0, 0
        depth_channels_sum, depth_channels_squared_sum, num_depth_batches = 0, 0, 0

        for data in dataset_loader:
            rgb_images, depth_images = data  # Unpack the tuples

            # Calculate statistics for RGB images
            rgb_channels_sum += torch.mean(rgb_images, dim=[0, 1, 2])
            rgb_channels_squared_sum += torch.mean(rgb_images ** 2, dim=[0, 1, 2])
            num_rgb_batches += 1

            # Calculate statistics for depth images

            depth_images = depth_images.to(torch.float32)
            depth_channels_sum += torch.mean(depth_images, dim=[0, 1, 2])
            depth_channels_squared_sum += torch.mean(depth_images ** 2, dim=[0, 1, 2])
            num_depth_batches += 1

        rgb_mean = rgb_channels_sum / num_rgb_batches
        rgb_std = (rgb_channels_squared_sum / num_rgb_batches - rgb_mean ** 2) ** 0.5

        depth_mean = depth_channels_sum / num_depth_batches
        depth_std = (depth_channels_squared_sum / num_depth_batches - depth_mean ** 2) ** 0.5

        return rgb_mean, rgb_std, depth_mean, depth_std



def calculate_additional_info(dataset,element_in_each_class=50):
    total_data=sum([len(dataset.data[k]) for k,v in dataset.data.items()])
    number_of_classes = total_data//element_in_each_class
    number_of_val_classes = len(dataset.data["val"])//element_in_each_class
    number_of_test_classes = len(dataset.data["test"])//element_in_each_class
    return (number_of_classes,number_of_val_classes,number_of_test_classes,element_in_each_class), True, False


class ApplySeparateTransforms(torch.nn.Module):
    def __init__(self, rgb_transform, depth_transform):
        super().__init__()
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

    def forward(self, image):
        rgb_image = image[0]
        depth_image = image[1]

        rgb_image = self.rgb_transform(rgb_image)
        depth_image = self.depth_transform(depth_image)

        return torch.cat([rgb_image, depth_image], dim=0)

def depth_main():
    dataset = DepthDataset(args)
    input_shape = [4, 84, 84]
    num_classes, few_shot, top_5 = calculate_additional_info(dataset)
    rgb_means, rgb_stds, depth_means, depth_stds = dataset.calculate_mean_std(dataset.data['train'])
    rgb_norm = transforms.Normalize(rgb_means, rgb_stds)
    depth_norm = transforms.Normalize(depth_means, depth_stds)
    norm = ApplySeparateTransforms(rgb_norm,depth_norm)
    train_transforms = torch.nn.Sequential(
        # transforms.RandomResizedCrop(84),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        # transforms.RandomHorizontalFlip(),
        norm
    )

    all_transforms = torch.nn.Sequential(
        # transforms.Resize(92),
        # transforms.CenterCrop(84),
        norm

    ) if args.sample_aug == 1 else torch.nn.Sequential(
        transforms.RandomResizedCrop(84),
        norm
    )
    if args.episodic:
        train_loader = episodic_iterator(dataset.data['train'], 64, transforms=train_transforms, forcecpu=True,
                                         use_hd=True)
    else:
        train_loader = iterator(dataset.data['train'], dataset.targets['train'], transforms=train_transforms, forcecpu=True,
                                use_hd=False)
    train_clean = iterator(dataset.data['train'], dataset.targets['train'], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=False)
    val_loader = iterator(dataset.data['val'], dataset.targets['val'], transforms=all_transforms, forcecpu=True,
                          shuffle=False, use_hd=False)
    test_loader = iterator(dataset.data['test'], dataset.targets['test'], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=False)
    return (train_loader, train_clean, val_loader, test_loader), input_shape,num_classes, few_shot, top_5

# def calculate_mean_std(dataset_loader):
#     """Calculates mean and standard deviation per channel across a dataset"""
#
#     channels_sum, channels_squared_sum, num_batches = 0, 0, 0
#
#     for data in dataset_loader:
#         channels_sum += torch.mean(data, dim=[0, 1, 2])  # Adjust the dimensions # Sum across batch, height, width dims
#         channels_squared_sum += torch.mean(data ** 2, dim=[0, 1, 2])
#         num_batches += 1
#
#     mean = channels_sum / num_batches
#     std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
#
#     return mean, std
# def depth_dataset():
#     """Creates a dataset loader with RGB images and corresponding depth images (if available)."""
#
#     datasets = {}
#     for subset in ["train", "val", "test"]:
#         data = []
#         target = []
#         data_path = args.dataset_path+'json_files'
#         json_path = os.path.join(data_path, subset + '.json')
#
#         with open(json_path, 'r') as f:
#             json_data = json.load(f)
#
#         image_names = json_data['image_names']
#         image_labels = json_data['image_labels']
#
#         classes = list(set(image_labels))  # Extract unique classes
#         class_to_idx = {cls: i for i, cls in enumerate(classes)}
#
#         for img_name, img_label in zip(image_names, image_labels):
#             target.append(class_to_idx[img_label])
#             rgb_path = os.path.join(args.dataset_path, 'images', img_name)
#             rgb_image = transforms.ToTensor()(np.array(Image.open(rgb_path).convert('RGB')))
#
#             depth_path = find_depth_path(rgb_path)
#             if depth_path:
#                 depth_image = transforms.ToTensor()(
#                     np.array(Image.open(depth_path).resize((rgb_image.size()[2], rgb_image.size()[1]))))
#
#             else:
#                 depth_image = rgb_image.mean(dim=0, keepdims=True)  # Grayscale conversion
#             image = torch.cat([rgb_image, depth_image], dim=0)
#
#             data.append(image)
#
#         datasets[subset] = [data, torch.LongTensor(target)]
#
#     print("load depth")
#     means, stds = calculate_mean_std(datasets["train"][0])
#     norm = transforms.Normalize(means, stds)
#     train_transforms = torch.nn.Sequential(
#         # transforms.RandomResizedCrop(84),
#         # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
#         # transforms.RandomHorizontalFlip(),
#         norm
#     )
#
#     all_transforms = torch.nn.Sequential(
#         # transforms.Resize(92),
#         # transforms.CenterCrop(84),
#         norm
#     ) if args.sample_aug == 1 else torch.nn.Sequential(
#         transforms.RandomResizedCrop(84),
#         norm
#     )
#     if args.episodic:
#         train_loader = episodic_iterator(datasets["train"][0], 64, transforms=train_transforms, forcecpu=True,
#                                          use_hd=True)
#     else:
#         train_loader = iterator(datasets["train"][0], datasets["train"][1], transforms=train_transforms, forcecpu=True,
#                                 use_hd=False)
#     train_clean = iterator(datasets["train"][0], datasets["train"][1], transforms=all_transforms, forcecpu=True,
#                            shuffle=False, use_hd=False)
#     val_loader = iterator(datasets["val"][0], datasets["val"][1], transforms=all_transforms, forcecpu=True,
#                           shuffle=False, use_hd=False)
#     test_loader = iterator(datasets["test"][0], datasets["test"][1], transforms=all_transforms, forcecpu=True,
#                            shuffle=False, use_hd=False)
#     train_label_number = len(np.unique(datasets["train"][1].numpy()))
#     val_label_number = len(np.unique(datasets["val"][1].numpy()))
#     test_label_number = len(np.unique(datasets["test"][1].numpy()))
#
#     return ((train_loader, train_clean, val_loader, test_loader), [4, 84, 84],
#             (train_label_number + val_label_number + test_label_number, val_label_number, test_label_number,
#              len(datasets["train"][1]) // train_label_number), True, False)
#


def get_dataset(dataset_name):
    if dataset_name.lower()=="evyatar":
        return evyatar_dataset()
    elif dataset_name.lower()=="depth":
        return depth_main()
    elif dataset_name.lower() == "miniimagenet":
        return miniImageNet()
    else:
        print("Unknown dataset!")
print("datasets, ", end='')
