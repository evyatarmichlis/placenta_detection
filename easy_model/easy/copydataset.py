import pickle
from collections import Counter

from torchvision import transforms, datasets
from args import args
import numpy as np
import torch
import json
import os


class CPUDataset():
    def __init__(self, data, targets, transforms=[], batch_size=args.batch_size, use_hd=False):
        self.data = data
        if torch.is_tensor(data):
            self.length = data.shape[0]
        else:
            self.length = len(self.data)
        self.targets = targets
        assert (self.length == targets.shape[0])
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
    def __init__(self, data, num_classes, transforms=[], episode_size=args.batch_size, use_hd=False):
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
                class_indices = np.random.permutation(np.arange(self.length // self.num_classes))[
                                :self.episode_size // args.n_ways]
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
            class_indices = np.random.permutation(np.arange(self.length // self.num_classes))[
                            :self.episode_size // args.n_ways]
            self.indices[idx * self.episode_size + c * n_samples: idx * self.episode_size + (c + 1) * n_samples] = (
                        class_indices + classes[c] * (self.length // self.num_classes))

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
    def __init__(self, data, targets, transforms=[], batch_size=args.batch_size, shuffle=True,
                 device=args.dataset_device):
        if torch.is_tensor(data):
            self.length = data.shape[0]
            self.data = data.to(device)
        else:
            self.length = len(self.data)
        self.targets = targets.to(device)
        assert (self.length == targets.shape[0])
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
                yield self.transforms(self.data[self.permutation[i * self.batch_size: (i + 1) * self.batch_size]]), \
                self.targets[self.permutation[i * self.batch_size: (i + 1) * self.batch_size]]
            else:
                yield torch.stack([self.transforms(self.data[x]) for x in
                                   self.permutation[i * self.batch_size: (i + 1) * self.batch_size]]), self.targets[
                    self.permutation[i * self.batch_size: (i + 1) * self.batch_size]]

    def __len__(self):
        return self.n_batches


class EpisodicDataset():
    def __init__(self, data, num_classes, transforms=[], episode_size=args.batch_size, device=args.dataset_device,
                 use_hd=False):
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
                class_indices = np.random.permutation(np.arange(self.length // self.num_classes))[
                                :self.episode_size // args.n_ways]
                indices += list(class_indices + classes[c] * (self.length // self.num_classes))
            targets = torch.repeat_interleave(torch.arange(args.n_ways), self.episode_size // args.n_ways).to(
                self.device)
            if torch.is_tensor(self.data):
                yield self.transforms(self.data[indices]), targets
            else:
                if self.use_hd:
                    yield torch.stack([self.transforms(
                        transforms.ToTensor()(np.array(Image.open(self.data[x]).convert('RGB'))).to(self.device)) for x
                                       in indices]), targets
                else:
                    yield torch.stack([self.transforms(self.data[x].to(self.device)) for x in indices]), targets

    def __len__(self):
        return self.n_batches


def iterator(data, target, transforms, forcecpu=False, shuffle=True, use_hd=False):
    if args.dataset_device == "cpu" or forcecpu:
        dataset = CPUDataset(data, target, transforms, use_hd=use_hd)
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                                           num_workers=min(8, os.cpu_count()))
    else:
        return Dataset(data, target, transforms, shuffle=shuffle)


def episodic_iterator(data, num_classes, transforms, forcecpu=False, use_hd=False):
    if args.dataset_device == "cpu" or forcecpu:
        dataset = EpisodicCPUDataset(data, num_classes, transforms, use_hd=use_hd)
        return torch.utils.data.DataLoader(dataset, batch_size=(args.batch_size // args.n_ways) * args.n_ways,
                                           shuffle=False, num_workers=min(8, os.cpu_count()))
    else:
        return EpisodicDataset(data, num_classes, transforms, use_hd=use_hd)


def create_dataset(train_data, test_data, train_targets, test_targets, train_transforms, test_transforms):
    train_loader = iterator(train_data[:args.dataset_size], train_targets[:args.dataset_size],
                            transforms=train_transforms)
    val_loader = iterator(train_data, train_targets, transforms=test_transforms)
    test_loader = iterator(test_data, test_targets, transforms=test_transforms)
    return train_loader, val_loader, test_loader


import random


def mnist():
    train_loader = datasets.MNIST(args.dataset_path, train=True, download=True)
    train_data = (train_loader.data.float() / 256).unsqueeze(1)
    train_targets = torch.LongTensor(train_loader.targets.clone())
    if args.dataset_size >= 0:
        data_per_class = []
        test = []
        for i in range(10):
            data_per_class.append(train_data[torch.where(train_targets == i)[0]][:args.dataset_size // 10])
            test.append(torch.zeros(args.dataset_size // 10) + i)
        train_data = torch.stack(data_per_class, dim=1).view(args.dataset_size, 1, 28, 28)
        train_targets = torch.arange(10).repeat(args.dataset_size // 10)
    test_loader = datasets.MNIST(args.dataset_path, train=False, download=True)
    test_data = (test_loader.data.float() / 256).unsqueeze(1)
    test_targets = torch.LongTensor(test_loader.targets.clone())
    all_transforms = transforms.Normalize((0.1302,), (0.3069,))
    loaders = create_dataset(train_data, test_data, train_targets, test_targets, all_transforms, all_transforms)
    return loaders, train_data.shape[1:], torch.max(train_targets).item() + 1, False, False


def fashion_mnist(data_augmentation=True):
    train_loader = datasets.FashionMNIST(args.dataset_path, train=True, download=True)
    train_data = (train_loader.data.float() / 256).unsqueeze(1)
    train_targets = torch.LongTensor(train_loader.targets)
    if args.dataset_size >= 0:
        data_per_class = []
        test = []
        for i in range(10):
            data_per_class.append(train_data[torch.where(train_targets == i)[0]][:args.dataset_size // 10])
            test.append(torch.zeros(args.dataset_size // 10) + i)
        train_data = torch.stack(data_per_class, dim=1).view(args.dataset_size, 1, 28, 28)
        train_targets = torch.arange(10).repeat(args.dataset_size // 10)
    test_loader = datasets.FashionMNIST(args.dataset_path, train=False, download=True)
    test_data = (test_loader.data.float() / 256).unsqueeze(1)
    test_targets = torch.LongTensor(test_loader.targets)
    norm = transforms.Normalize((0.2849,), (0.3516,))
    if data_augmentation:
        list_trans_train = torch.nn.Sequential(transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(),
                                               norm)
    all_transforms = norm
    loaders = create_dataset(train_data, test_data, train_targets, test_targets, list_trans_train, all_transforms)
    return loaders, train_data.shape[1:], torch.max(train_targets).item() + 1, False, False


def cifar10(data_augmentation=True):
    train_loader = datasets.CIFAR10(args.dataset_path, train=True, download=True)
    train_data = torch.stack(list(map(transforms.ToTensor(), train_loader.data)))
    train_targets = torch.LongTensor(train_loader.targets)
    if args.dataset_size >= 0:
        data_per_class = []
        test = []
        for i in range(10):
            data_per_class.append(train_data[torch.where(train_targets == i)[0]][:args.dataset_size // 10])
            test.append(torch.zeros(args.dataset_size // 10) + i)
        train_data = torch.stack(data_per_class, dim=1).view(args.dataset_size, 3, 32, 32)
        train_targets = torch.arange(10).repeat(args.dataset_size // 10)
    test_loader = datasets.CIFAR10(args.dataset_path, train=False, download=True)
    test_data = torch.stack(list(map(transforms.ToTensor(), test_loader.data)))
    test_targets = torch.LongTensor(test_loader.targets)
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    if data_augmentation:
        list_trans_train = torch.nn.Sequential(transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                               norm)
    else:
        list_trans_train = norm
    loaders = create_dataset(train_data, test_data, train_targets, test_targets, list_trans_train, norm)
    return loaders, train_data.shape[1:], torch.max(train_targets).item() + 1, False, False


def cifar100(data_augmentation=True):
    train_loader = datasets.CIFAR100(args.dataset_path, train=True, download=True)
    train_data = torch.stack(list(map(transforms.ToTensor(), train_loader.data)))
    train_targets = torch.LongTensor(train_loader.targets)
    if args.dataset_size >= 0:
        data_per_class = []
        test = []
        for i in range(10):
            data_per_class.append(train_data[torch.where(train_targets == i)[0]][:args.dataset_size // 10])
            test.append(torch.zeros(args.dataset_size // 10) + i)
        train_data = torch.stack(data_per_class, dim=1).view(args.dataset_size, 3, 32, 32)
        train_targets = torch.arange(10).repeat(args.dataset_size // 10)
    test_loader = datasets.CIFAR100(args.dataset_path, train=False, download=True)
    test_data = torch.stack(list(map(transforms.ToTensor(), test_loader.data)))
    test_targets = torch.LongTensor(test_loader.targets)
    norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    if data_augmentation:
        list_trans_train = torch.nn.Sequential(transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                               norm)
    else:
        list_trans_train = norm
    loaders = create_dataset(train_data, test_data, train_targets, test_targets, list_trans_train, norm)
    return loaders, train_data.shape[1:], torch.max(train_targets).item() + 1, False, True


from PIL import Image


def cifarfs(use_hd=True, data_augmentation=True):
    """
    CIFAR FS dataset
    Number of classes :
    - train: 64
    - val  : 16
    - novel: 20
    Number of samples per class: exactly 600
    Total number of images: 60000
    Images size : 32x32
    """
    datasets = {}
    classes = []
    total = 60000
    buffer = {'train': 0, 'val': 64, 'test': 64 + 16}
    for metaSub in ["meta-train", "meta-val", "meta-test"]:
        subset = metaSub.split('-')[-1]
        data = []
        target = []
        subset_path = os.path.join(args.dataset_path, 'cifar_fs', metaSub)
        classe_files = os.listdir(subset_path)

        for c, classe in enumerate(classe_files):
            files = os.listdir(os.path.join(subset_path, classe))
            count = 0
            for file in files:
                count += 1
                target.append(c + buffer[subset])

                path = os.path.join(subset_path, classe, file)
                if not use_hd:
                    image = transforms.ToTensor()(np.array(Image.open(path).convert('RGB')))
                    data.append(image)
                else:
                    data.append(path)

        datasets[subset] = [data, torch.LongTensor(target)]

    assert (len(datasets['train'][0]) + len(datasets['val'][0]) + len(
        datasets['test'][0]) == total), 'Total number of sample per class is not 600'

    image_size = 32
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transforms = torch.nn.Sequential(transforms.RandomResizedCrop(image_size),
                                           transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                           transforms.RandomHorizontalFlip(),
                                           norm)

    all_transforms = torch.nn.Sequential(transforms.Resize([int(1.15 * image_size), int(1.15 * image_size)]),
                                         transforms.CenterCrop(image_size),
                                         norm) if args.sample_aug == 1 else torch.nn.Sequential(
        transforms.RandomResizedCrop(image_size, scale=(0.14, 1)), norm)

    if args.episodic:
        train_loader = episodic_iterator(datasets['train'][0], 64, transforms=train_transforms, forcecpu=True,
                                         use_hd=True)
    else:
        train_loader = iterator(datasets['train'][0], datasets['train'][1], transforms=train_transforms, forcecpu=True,
                                use_hd=use_hd)
    train_clean = iterator(datasets["train"][0], datasets["train"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=use_hd)
    val_loader = iterator(datasets["val"][0], datasets["val"][1], transforms=all_transforms, forcecpu=True,
                          shuffle=False, use_hd=use_hd)
    test_loader = iterator(datasets["test"][0], datasets["test"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=use_hd)

    return (train_loader, train_clean, val_loader, test_loader), [3, image_size, image_size], (
    64, 16, 20, 600), True, False


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
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    train_transforms = torch.nn.Sequential(transforms.RandomResizedCrop(84),
                                           transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                           transforms.RandomHorizontalFlip(), norm)
    all_transforms = torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84),
                                         norm) if args.sample_aug == 1 else torch.nn.Sequential(
        transforms.RandomResizedCrop(84), norm)
    if args.episodic:
        train_loader = episodic_iterator(datasets["train"][0], 64, transforms=train_transforms, forcecpu=True,
                                         use_hd=True)
    else:
        train_loader = iterator(datasets["train"][0], datasets["train"][1], transforms=train_transforms, forcecpu=True,
                                use_hd=use_hd)
    train_clean = iterator(datasets["train"][0], datasets["train"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=use_hd)
    val_loader = iterator(datasets["val"][0], datasets["val"][1], transforms=all_transforms, forcecpu=True,
                          shuffle=False, use_hd=use_hd)
    test_loader = iterator(datasets["test"][0], datasets["test"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=use_hd)
    return (train_loader, train_clean, val_loader, test_loader), [3, 84, 84], (64, 16, 20, 600), True, False


def evyatar_dataset():
    datasets = {}
    for subset in ["train", "val", "test"]:
        data = []
        target = []
        data_path = args.dataset_path + 'json_files'
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
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    train_transforms = torch.nn.Sequential(transforms.RandomResizedCrop(84),
                                           transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                           transforms.RandomHorizontalFlip(), norm)
    all_transforms = torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84),
                                         norm) if args.sample_aug == 1 else torch.nn.Sequential(
        transforms.RandomResizedCrop(84), norm)
    if args.episodic:
        train_loader = episodic_iterator(datasets["train"][0], 64, transforms=train_transforms, forcecpu=True,
                                         use_hd=True)
    else:
        train_loader = iterator(datasets["train"][0], datasets["train"][1], transforms=train_transforms, forcecpu=True,
                                use_hd=False)
    train_clean = iterator(datasets["train"][0], datasets["train"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=False)
    val_loader = iterator(datasets["val"][0], datasets["val"][1], transforms=all_transforms, forcecpu=True,
                          shuffle=False, use_hd=False)
    test_loader = iterator(datasets["test"][0], datasets["test"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=False)
    train_label_number = len(np.unique(datasets["train"][1].numpy()))
    test_val_number = len(np.unique(datasets["val"][1].numpy()))
    test_label_number = len(np.unique(datasets["test"][1].numpy()))

    return ((train_loader, train_clean, val_loader, test_loader), [3, 84, 84],
            (train_label_number + test_val_number + test_label_number, test_val_number, test_label_number,
             len(datasets["train"][1]) // train_label_number), True, False)


def find_depth_path(rgb_path):
    """Attempts to find a corresponding depth image for a given RGB image path.

    Args:
        rgb_path (str): Path to the RGB image.

    Returns:
        str: Path to the depth image if found, otherwise None.
    """

    base_dir, filename = os.path.split(rgb_path)
    depth_dir = base_dir.replace('RGB_50', 'depth_50')
    for curr_ext in ['.png', '.jpg']:
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            potential_depth_path = os.path.join(depth_dir, filename.replace(curr_ext, ext))  # Assuming .jpg for RGB
            if os.path.exists(potential_depth_path):
                return potential_depth_path
    return None


class DepthDataset:
    def __init__(self, args):
        self.args = args
        self.subsets = ["train", "val", "test"]
        self.subset_sizes = [
            len(json.load(open(os.path.join(args.dataset_path + "json_files", subset + ".json"))))
            for subset in self.subsets
        ]

    class DepthDataset:
        def __init__(self, args):
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
                    depth_image = transforms.ToTensor()(
                        np.array(Image.open(depth_path).resize((rgb_image.size()[2], rgb_image.size()[1]))))
                else:
                    depth_image = rgb_image.mean(dim=0, keepdims=True)

                image = torch.cat([rgb_image, depth_image], dim=0)
                images.append(image)
                targets.append(target)

            return images, torch.LongTensor(targets)

        def determine_max_size(self, subset):
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

            image = self.data[subset][index]
            target = self.targets[subset][index]

            max_height, max_width = self.determine_max_size(subset)
            image = self.pad_image(image, max_height, max_width)

            return image, target

    def calculate_mean_std(self, dataset_loader):
        """Calculates mean and standard deviation per channel across a dataset"""

        channels_sum, channels_squared_sum, num_batches = 0, 0, 0

        for data in dataset_loader:
            channels_sum += torch.mean(data,
                                       dim=[0, 1, 2])  # Adjust the dimensions # Sum across batch, height, width dims
            channels_squared_sum += torch.mean(data ** 2, dim=[0, 1, 2])
            num_batches += 1

        mean = channels_sum / num_batches
        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

        return mean, std


def pad_image(image, max_height, max_width):
    current_height, current_width = image.shape[1:]  # Assuming image is a NumPy array

    pad_bottom = max_height - current_height
    pad_right = max_width - current_width

    padding = ((0, 0), (0, pad_bottom), (0, pad_right))  # Format for np.pad
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)  # Pad with 0

    return padded_image


def calculate_additional_info(dataset, element_in_each_class=50):
    return (sum(dataset.subset_sizes) // element_in_each_class, dataset.subset_sizes[1] // element_in_each_class,
            dataset.subset_sizes[2] // element_in_each_class, element_in_each_class), True, False


def depth_main():
    dataset = DepthDataset(args)
    input_shape = [4, 84, 84]
    num_classes, few_shot, top_5 = calculate_additional_info(dataset)
    means, stds = dataset.calculate_mean_std(dataset.subsets[0])
    norm = transforms.Normalize(means, stds)
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
        train_loader = episodic_iterator(datasets["train"][0], 64, transforms=train_transforms, forcecpu=True,
                                         use_hd=True)
    else:
        train_loader = iterator(datasets["train"][0], datasets["train"][1], transforms=train_transforms, forcecpu=True,
                                use_hd=False)
    train_clean = iterator(datasets["train"][0], datasets["train"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=False)
    val_loader = iterator(datasets["val"][0], datasets["val"][1], transforms=all_transforms, forcecpu=True,
                          shuffle=False, use_hd=False)
    test_loader = iterator(datasets["test"][0], datasets["test"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=False)
    return (train_loader, train_clean, val_loader, test_loader), input_shape, num_classes, few_shot, top_5, True, False


def calculate_mean_std(dataset_loader):
    """Calculates mean and standard deviation per channel across a dataset"""

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data in dataset_loader:
        channels_sum += torch.mean(data, dim=[0, 1, 2])  # Adjust the dimensions # Sum across batch, height, width dims
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 1, 2])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def depth_dataset():
    """Creates a dataset loader with RGB images and corresponding depth images (if available)."""

    datasets = {}
    for subset in ["train", "val", "test"]:
        data = []
        target = []
        data_path = args.dataset_path + 'json_files'
        json_path = os.path.join(data_path, subset + '.json')

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        image_names = json_data['image_names']
        image_labels = json_data['image_labels']

        classes = list(set(image_labels))  # Extract unique classes
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for img_name, img_label in zip(image_names, image_labels):
            target.append(class_to_idx[img_label])
            rgb_path = os.path.join(args.dataset_path, 'images', img_name)
            rgb_image = transforms.ToTensor()(np.array(Image.open(rgb_path).convert('RGB')))

            depth_path = find_depth_path(rgb_path)
            if depth_path:
                depth_image = transforms.ToTensor()(
                    np.array(Image.open(depth_path).resize((rgb_image.size()[2], rgb_image.size()[1]))))

            else:
                depth_image = rgb_image.mean(dim=0, keepdims=True)  # Grayscale conversion
            image = torch.cat([rgb_image, depth_image], dim=0)

            data.append(image)

        datasets[subset] = [data, torch.LongTensor(target)]

    print("load depth")
    means, stds = calculate_mean_std(datasets["train"][0])
    norm = transforms.Normalize(means, stds)
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
        train_loader = episodic_iterator(datasets["train"][0], 64, transforms=train_transforms, forcecpu=True,
                                         use_hd=True)
    else:
        train_loader = iterator(datasets["train"][0], datasets["train"][1], transforms=train_transforms, forcecpu=True,
                                use_hd=False)
    train_clean = iterator(datasets["train"][0], datasets["train"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=False)
    val_loader = iterator(datasets["val"][0], datasets["val"][1], transforms=all_transforms, forcecpu=True,
                          shuffle=False, use_hd=False)
    test_loader = iterator(datasets["test"][0], datasets["test"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=False)
    train_label_number = len(np.unique(datasets["train"][1].numpy()))
    test_val_number = len(np.unique(datasets["val"][1].numpy()))
    test_label_number = len(np.unique(datasets["test"][1].numpy()))

    return ((train_loader, train_clean, val_loader, test_loader), [4, 84, 84],
            (train_label_number + test_val_number + test_label_number, test_val_number, test_label_number,
             len(datasets["train"][1]) // train_label_number), True, False)


def tieredImageNet(use_hd=True):
    """
    tiredImagenet dataset
    Number of classes :
    - train: 351
    - val  : 97
    - novel: 160
    Number of samples per class: at most 1300
    Total number of images: 790400
    Images size : 84x84
    """
    datasets = {}
    total = 790400
    num_elements = {}
    buffer = {'train': 0, 'val': 351, 'test': 351 + 97}
    for subset in ['train', 'val', 'test']:
        data = []
        target = []
        num_elements[subset] = []
        if subset == 'train':
            data_train = []
            target_train = []
        subset_path = os.path.join(args.dataset_path, 'tieredimagenet', subset)
        classe_files = os.listdir(subset_path)

        for c, classe in enumerate(classe_files):
            files = os.listdir(os.path.join(subset_path, classe))
            count = 0
            for file in files:
                count += 1
                target.append(c + buffer[subset])
                if subset == 'train':
                    target_train.append(c)
                path = os.path.join(subset_path, classe, file)
                if not use_hd:
                    image = transforms.ToTensor()(np.array(Image.open(path).convert('RGB')))
                    data.append(image)
                    if subset == 'train':
                        data_train.append(image)
                else:
                    data.append(path)
                    if subset == 'train':
                        data_train.append(path)
            num_elements[subset].append(count)
            if count < 1300:
                for i in range(1300 - count):
                    target.append(c + buffer[subset])
                    if not use_hd:  # add the same element
                        image = transforms.ToTensor()(np.array(Image.open(path).convert('RGB')))
                        data.append(image)
                    else:
                        data.append(path)

        datasets[subset] = [data, torch.LongTensor(target)]

    datasets['train_base'] = [data_train, torch.LongTensor(target_train)]  # clean train without duplicates

    assert (len(datasets['train'][0]) + len(datasets['val'][0]) + len(
        datasets['test'][0]) == total), 'Total number of sample per class is not 1300'
    print()
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    train_transforms = torch.nn.Sequential(transforms.RandomResizedCrop(84),
                                           transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                           transforms.RandomHorizontalFlip(), norm)
    all_transforms = torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84),
                                         norm) if args.sample_aug == 1 else torch.nn.Sequential(
        transforms.RandomResizedCrop(84), norm)
    if args.episodic:
        train_loader = episodic_iterator(datasets["train_base"][0], 351, transforms=train_transforms, forcecpu=True,
                                         use_hd=True)
    else:
        train_loader = iterator(datasets["train_base"][0], datasets["train_base"][1], transforms=train_transforms,
                                forcecpu=True, use_hd=use_hd)
    train_clean = iterator(datasets["train"][0], datasets["train"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=use_hd)
    val_loader = iterator(datasets["val"][0], datasets["val"][1], transforms=all_transforms, forcecpu=True,
                          shuffle=False, use_hd=use_hd)
    test_loader = iterator(datasets["test"][0], datasets["test"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=use_hd)
    return (train_loader, train_clean, val_loader, test_loader), [3, 84, 84], (
    351, 97, 160, (num_elements['train'], num_elements['val'], num_elements['test'])), True, False


def fc100(use_hd=True):
    """
    fc100 dataset
    Number of classes :
    - train: 60
    - val  : 20
    - novel: 20
    Number of samples per class: exactly 600
    Total number of images: 60000
    Images size : 84x84
    """
    datasets = {}
    total = 60000
    buffer = {'train': 0, 'val': 60, 'test': 60 + 20}
    for subset in ['train', 'val', 'test']:
        data = []
        target = []
        subset_path = os.path.join(args.dataset_path, 'FC100', subset)
        classe_files = os.listdir(subset_path)

        for c, classe in enumerate(classe_files):
            files = os.listdir(os.path.join(subset_path, classe))
            for file in files:
                target.append(c + buffer[subset])
                path = os.path.join(subset_path, classe, file)
                if not use_hd:
                    image = transforms.ToTensor()(np.array(Image.open(path).convert('RGB')))
                    data.append(image)
                else:
                    data.append(path)
        datasets[subset] = [data, torch.LongTensor(target)]

    assert (len(datasets['train'][0]) + len(datasets['val'][0]) + len(
        datasets['test'][0]) == total), 'Total number of sample per class is not 1300'
    print()

    image_size = 84
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transforms = torch.nn.Sequential(transforms.RandomResizedCrop(image_size),
                                           transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                           transforms.RandomHorizontalFlip(),
                                           norm)

    all_transforms = torch.nn.Sequential(transforms.Resize(92),
                                         transforms.CenterCrop(image_size),
                                         norm) if args.sample_aug == 1 else torch.nn.Sequential(
        transforms.RandomResizedCrop(image_size, scale=(0.14, 1)), norm)
    if args.episodic:
        train_loader = episodic_iterator(datasets["train"][0], 60, transforms=train_transforms, forcecpu=True,
                                         use_hd=True)
    else:
        train_loader = iterator(datasets["train"][0], datasets["train"][1], transforms=train_transforms, forcecpu=True,
                                use_hd=use_hd)
    train_clean = iterator(datasets["train"][0], datasets["train"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=use_hd)
    val_loader = iterator(datasets["val"][0], datasets["val"][1], transforms=all_transforms, forcecpu=True,
                          shuffle=False, use_hd=use_hd)
    test_loader = iterator(datasets["test"][0], datasets["test"][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=use_hd)
    return (train_loader, train_clean, val_loader, test_loader), [3, 84, 84], (60, 20, 20, 600), True, False


def CUBfs(use_hd=True):
    datasets = {}
    num_elements = {}
    folders_path = os.path.join(args.dataset_path, 'CUB_200_2011')
    images_path = os.path.join(folders_path, 'CUB_200_2011', 'images')
    list_files = os.listdir(images_path)
    list_files.sort()
    num_elements = {}
    buffer = {'train': 0, 'val': 100, 'test': 150}
    class_names = {}
    for subset in ['train', 'val', 'test']:
        data = []
        target = []
        num_elements[subset] = []

        if subset == 'train':
            data_train = []
            target_train = []

        csv_path = os.path.join(folders_path, 'split', f'{subset}.csv')
        class_names[subset] = []
        with open(csv_path, "r") as f:
            start = 0
            for line in f:
                if start == 0:
                    start += 1
                else:
                    splits = line.split(",")
                    fn, c = splits[0], splits[1]
                    fn2 = ''.join([i for i in fn if not i.isdigit()])
                    fn2 = fn2.replace('.', '').replace('_', '').replace('jpg', '').lower()
                    if fn2 not in class_names[subset]:
                        class_names[subset].append(fn2)
        files = [fn for fn in list_files if
                 (''.join([i for i in fn if not i.isdigit()])).replace('.', '').replace('_', '').replace('jpg',
                                                                                                         '').lower() in
                 class_names[subset]]
        for c, folder in enumerate(files):
            count = 0
            images = os.listdir(os.path.join(images_path, folder))
            for file in images:
                count += 1
                target.append(c + buffer[subset])
                if subset == 'train':
                    target_train.append(c + buffer[subset])
                path = os.path.join(images_path, folder, file)
                if not use_hd:
                    image = transforms.ToTensor()(np.array(Image.open(path).convert('RGB')))
                    data.append(image)
                    if subset == 'train':
                        data_train.append(image)
                else:
                    data.append(path)
                    if subset == 'train':
                        data_train.append(path)
            num_elements[subset].append(count)
            if count < 60:
                for i in range(60 - count):
                    target.append(c + buffer[subset])
                    if not use_hd:  # add the same element
                        data.append(image)
                    else:
                        data.append(path)

        datasets[subset] = [data, torch.LongTensor(target)]
        if subset == 'train':
            datasets['train_base'] = [data_train, torch.LongTensor(target_train)]  # clean train without duplicates

    image_size = 84
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transforms = torch.nn.Sequential(transforms.RandomResizedCrop(image_size),
                                           transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                           transforms.RandomHorizontalFlip(),
                                           norm)

    all_transforms = torch.nn.Sequential(transforms.Resize([int(1.15 * image_size), int(1.15 * image_size)]),
                                         transforms.CenterCrop(image_size),
                                         norm) if args.sample_aug == 1 else torch.nn.Sequential(
        transforms.RandomResizedCrop(image_size, scale=(0.14, 1)), norm)
    if args.episodic:
        train_loader = episodic_iterator(datasets['train_base'][0], 100, transforms=train_transforms, forcecpu=True,
                                         use_hd=use_hd)
    else:
        train_loader = iterator(datasets['train_base'][0], datasets['train_base'][1], transforms=train_transforms,
                                forcecpu=True, use_hd=use_hd)
    train_clean = iterator(datasets['train'][0], datasets['train'][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=use_hd)
    val_loader = iterator(datasets['val'][0], datasets['val'][1], transforms=all_transforms, forcecpu=True,
                          shuffle=False, use_hd=use_hd)
    test_loader = iterator(datasets['test'][0], datasets['test'][1], transforms=all_transforms, forcecpu=True,
                           shuffle=False, use_hd=use_hd)

    return (train_loader, train_clean, val_loader, test_loader), [3, image_size, image_size], (
    100, 50, 50, (num_elements['train'], num_elements['val'], num_elements['test'])), True, False


def omniglotfs():
    base = torch.load(args.dataset_path + "omniglot/base.pt")
    base_data = base.reshape(-1, base.shape[2], base.shape[3], base.shape[4]).float()
    base_targets = torch.arange(base.shape[0]).unsqueeze(1).repeat(1, base.shape[1]).reshape(-1)
    val = torch.load(args.dataset_path + "omniglot/val.pt")
    val_data = val.reshape(-1, val.shape[2], val.shape[3], val.shape[4]).float()
    val_targets = torch.arange(val.shape[0]).unsqueeze(1).repeat(1, val.shape[1]).reshape(-1)
    novel = torch.load(args.dataset_path + "omniglot/novel.pt")
    novel_data = novel.reshape(-1, novel.shape[2], novel.shape[3], novel.shape[4]).float()
    novel_targets = torch.arange(novel.shape[0]).unsqueeze(1).repeat(1, novel.shape[1]).reshape(-1)
    train_transforms = torch.nn.Sequential(transforms.RandomCrop(100, padding=4),
                                           transforms.Normalize((0.0782), (0.2685)))
    all_transforms = torch.nn.Sequential(transforms.CenterCrop(100), transforms.Normalize((0.0782), (
        0.2685))) if args.sample_aug == 1 else torch.nn.Sequential(transforms.RandomCrop(100, padding=4),
                                                                   transforms.Normalize((0.0782), (0.2685)))
    if args.episodic:
        train_loader = episodic_iterator(base_data, base.shape[0], transforms=train_transforms)
    else:
        train_loader = iterator(base_data, base_targets, transforms=train_transforms)
    train_clean = iterator(base_data, base_targets, transforms=all_transforms, shuffle=False)
    val_loader = iterator(val_data, val_targets, transforms=all_transforms, shuffle=False)
    test_loader = iterator(novel_data, novel_targets, transforms=all_transforms, shuffle=False)
    return (train_loader, train_clean, val_loader, test_loader), [1, 100, 100], (
    base.shape[0], val.shape[0], novel.shape[0], novel.shape[1]), True, False


def miniImageNet84():
    with open(args.dataset_path + "miniimagenet/train.pkl", 'rb') as f:
        train_file = pickle.load(f)
    train, train_targets = [transforms.ToTensor()(x) for x in train_file["data"]], train_file["labels"]
    with open(args.dataset_path + "miniimagenet/test.pkl", 'rb') as f:
        test_file = pickle.load(f)
    test, test_targets = [transforms.ToTensor()(x) for x in test_file["data"]], test_file["labels"]
    with open(args.dataset_path + "miniimagenet/validation.pkl", 'rb') as f:
        validation_file = pickle.load(f)
    validation, validation_targets = [transforms.ToTensor()(x) for x in validation_file["data"]], validation_file[
        "labels"]
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    train_transforms = torch.nn.Sequential(transforms.RandomResizedCrop(84),
                                           transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                           transforms.RandomHorizontalFlip(), norm)
    all_transforms = torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84),
                                         norm) if args.sample_aug == 1 else torch.nn.Sequential(
        transforms.RandomResizedCrop(84), norm)
    if args.episodic:
        train_loader = episodic_iterator(train, 64, transforms=train_transforms, forcecpu=True)
    else:
        train_loader = iterator(train, train_targets, transforms=train_transforms, forcecpu=True)
    train_clean = iterator(train, train_targets, transforms=all_transforms, forcecpu=True, shuffle=False)
    val_loader = iterator(validation, validation_targets, transforms=all_transforms, forcecpu=True, shuffle=False)
    test_loader = iterator(test, test_targets, transforms=all_transforms, forcecpu=True, shuffle=False)
    return (train_loader, train_clean, val_loader, test_loader), [3, 84, 84], (64, 16, 20, 600), True, False


def get_dataset(dataset_name):


    if dataset_name.lower() == "cifar10":
        return cifar10(data_augmentation = True)
    elif dataset_name.lower()=="evyatar":
        return evyatar_dataset()
    elif dataset_name.lower()=="depth":
        return depth_dataset()
    elif dataset_name.lower() == "cifar100":
        return cifar100(data_augmentation = True)
    elif dataset_name.lower() == "cifarfs":
        return cifarfs(data_augmentation = True)
    elif dataset_name.lower() == "mnist":
        return mnist()
    elif dataset_name.lower() == "fashion":
        return fashion_mnist()
    elif dataset_name.lower() == "miniimagenet":
        return miniImageNet()
    elif dataset_name.lower() == "miniimagenet84":
        return miniImageNet84()
    elif dataset_name.lower() == "cubfs":
        return CUBfs()
    elif dataset_name.lower() == "omniglotfs":
        return omniglotfs()
    elif dataset_name.lower() == "tieredimagenet":
        return tieredImageNet()
    elif dataset_name.lower() == "fc100":
        return fc100()
    else:
        print("Unknown dataset!")

print("datasets, ", end='')
