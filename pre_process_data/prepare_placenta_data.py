import matplotlib
import os
import random
from pathlib import Path
from tqdm import tqdm
import shutil
import re
from PIL import Image
import numpy as np
import cv2
from pre_process_data import find_local_max

matplotlib.use('Agg')


def dilate_mask(mask, kernel_size=15):
    """
    Dilates a binary segmentation mask to make objects appear larger.

    Args:
        mask (numpy.ndarray): Binary segmentation mask.
        kernel_size (int): Size of the dilation kernel.

    Returns:
        numpy.ndarray: Dilated mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    return dilated_mask

class DatasetOrganizer:
    def __init__(self,
                 color_images_dir,
                 generated_depth_dir=None,
                 real_depth_dir=None,
                 gt_dir=None,
                 output_dir=None,
                 seed=42,
                 test_gt_addition=0):
        self.color_images_dir = Path(color_images_dir)
        self.generated_depth_dir = Path(generated_depth_dir) if generated_depth_dir else None
        self.real_depth_dir = Path(real_depth_dir) if real_depth_dir else None
        self.gt_dir = Path(gt_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.test_gt_addition = test_gt_addition

        # Create output directory structure
        for split in ['train', 'val', 'test']:
            split_path = self.output_dir / split
            if split_path.exists():
                shutil.rmtree(split_path)
            subdirs = ['img', 'gt', 'gray']
            if self.generated_depth_dir:
                subdirs.append('depth_generated')
            if self.real_depth_dir:
                subdirs.extend(['depth_real', 'original_depth'])
            for subdir in subdirs:
                (split_path / subdir).mkdir(parents=True, exist_ok=True)

    def extract_datetime(self, filename):
        match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
        return match.group(1) if match else None

    def extract_hour_key(self, filename):
        dt = self.extract_datetime(filename)
        return dt[:13] if dt else None

    def _get_matched_files(self):
        # Map files by datetime key
        gt_files = {self.extract_datetime(f.name): f for f in self.gt_dir.glob('*.jpg')}
        gen_depth = {} if not self.generated_depth_dir else {
            self.extract_datetime(f.name): f for f in self.generated_depth_dir.glob('*.csv')
        }
        real_depth = {} if not self.real_depth_dir else {
            self.extract_datetime(f.name): f for f in self.real_depth_dir.glob('*.csv')
        }

        print(f"\n🔍 Dataset Analysis:")
        print(f"  GT masks: {len(gt_files)}")
        if self.generated_depth_dir:
            print(f"  Generated depth: {len(gen_depth)}")
        if self.real_depth_dir:
            print(f"  Real depth: {len(real_depth)}")

        matched = []
        skipped = []
        seen = set()
        duplicates = []

        for img_file in self.color_images_dir.glob('*.jpg'):
            img = cv2.imread(str(img_file))
            if img is None or not img.any():
                print(f"🛑 Skipping unreadable: {img_file.name}")
                continue
            dt = self.extract_datetime(img_file.name)
            has_gt = dt in gt_files
            has_real = self.real_depth_dir and dt in real_depth
            has_gen = self.generated_depth_dir and dt in gen_depth
            # require GT + real depth
            if not (dt and has_gt and has_real):
                skipped.append(img_file.name)
                continue
            key = (dt, gen_depth.get(dt), real_depth.get(dt), gt_files[dt])
            if key in seen:
                duplicates.append(img_file.name)
                continue
            seen.add(key)

            matched.append({
                'img_file': img_file,
                'gt_file': gt_files[dt],
                'real_depth_file': real_depth[dt],
                'generated_depth_file': gen_depth.get(dt),
                'has_real_depth': True,
                'has_generated_depth': bool(has_gen)
            })

        if duplicates:
            print(f"⚠️ Skipped {len(duplicates)} duplicates")
        if skipped:
            print(f"ℹ️ Skipped {len(skipped)} missing GT or real depth")

        print(f"\n📊 Final samples: {len(matched)}")
        return matched

    def _report_missing_files(self, gt_files, generated_depth_files, real_depth_files):
        """Report missing GT, generated depth, or real depth files."""
        missing_gt = [img for img in self.color_images_dir.glob('*.jpg') if
                      self.extract_datetime(img.name) not in gt_files]
        missing_generated_depth = [img for img in self.color_images_dir.glob('*.jpg') if
                                   self.extract_datetime(img.name) not in generated_depth_files]

        if missing_gt:
            print("\n⚠️ Warning: Some RGB images do not have corresponding GT files:")
            for img in missing_gt[:10]:  # Show first 10
                print(f" - {img.name}")
            if len(missing_gt) > 10:
                print(f" ... and {len(missing_gt) - 10} more")

        if missing_generated_depth:
            print("\n⚠️ Warning: Some RGB images do not have corresponding Generated Depth files:")
            for img in missing_generated_depth[:10]:  # Show first 10
                print(f" - {img.name}")
            if len(missing_generated_depth) > 10:
                print(f" ... and {len(missing_generated_depth) - 10} more")

        if real_depth_files:
            missing_real_depth = [img for img in self.color_images_dir.glob('*.jpg') if
                                  self.extract_datetime(img.name) not in real_depth_files]
            if missing_real_depth:
                print(
                    f"\n📝 Info: {len(missing_real_depth)} RGB images do not have corresponding Real Depth files (this is expected)")

    def add_value_to_gt_locations(self, depth_data, gt_mask, addition_value):
        modified = depth_data.copy()
        modified[gt_mask > 0] += addition_value
        return modified




    def prepare_splits(self, split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}):
        matched = self._get_matched_files()
        # Force "real" files into test
        forced_test = [m for m in matched if 'real' in m['img_file'].name.lower()]
        print(f"✨ Forcing {len(forced_test)} 'real' samples into test set")
        remaining = [m for m in matched if m not in forced_test]

        # Group remaining by hour
        groups = {}
        for m in remaining:
            hour = self.extract_hour_key(m['img_file'].name)
            if hour:
                groups.setdefault(hour, []).append(m)

        keys = sorted(groups)
        random.seed(self.seed)
        random.shuffle(keys)
        n = len(keys)
        n_train = int(split_ratios['train'] * n)
        n_val = int(split_ratios['val'] * n)

        train_keys = keys[:n_train]
        val_keys = keys[n_train:n_train + n_val]
        test_keys = keys[n_train + n_val:]

        splits = {
            'train': [i for k in train_keys for i in groups[k]],
            'val': [i for k in val_keys for i in groups[k]],
            'test': [i for k in test_keys for i in groups[k]] + forced_test
        }

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for split, items in splits.items():
            print(f"\nProcessing {split} ({len(items)} samples)")
            if items:
                if items[0]['has_generated_depth']:
                    print("  - Data: RGB + GT + Real Depth + Generated Depth")
                else:
                    print("  - Data: RGB + GT + Real Depth")

            for info in tqdm(items):
                # Load GT mask and dilate
                gt_mask = np.array(Image.open(info['gt_file']).convert('L'))
                dilated = dilate_mask(gt_mask)
                dilated_path = self.output_dir / split / 'gt' / Path(info['gt_file']).name
                Image.fromarray(dilated).save(dilated_path)

                # CLAHE on color
                color = cv2.imread(str(info['img_file']))
                lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l_eq = clahe.apply(l)
                lab_eq = cv2.merge((l_eq, a, b))
                color_clahe = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
                out_img = self.output_dir / split / 'img' / info['img_file'].name
                cv2.imwrite(str(out_img), color_clahe)

                # Process real depth
                real_data = np.genfromtxt(info['real_depth_file'], delimiter=',')
                if real_data.shape == (481, 640):
                    real_data = real_data[1:, :]
                if split == 'test' and self.test_gt_addition:
                    real_data = self.add_value_to_gt_locations(real_data, gt_mask, self.test_gt_addition)
                temp_real = str(info['real_depth_file']).replace('.csv', '_temp.csv')
                np.savetxt(temp_real, real_data, delimiter=',')
                lm_real = find_local_max.FindLocalMax(csv_path=temp_real,
                                                      rgb_image_path=info['img_file'],
                                                      ground_truth=dilated_path,
                                                      real_depth=True)
                real_norm = lm_real.process(save=False)
                real_out = self.output_dir / split / 'depth_real' / f"{info['img_file'].stem}.png"
                Image.fromarray(real_norm.astype(np.uint8)).save(real_out)
                os.remove(temp_real)

                # Process generated depth if available
                if info['has_generated_depth']:
                    gen_data = np.genfromtxt(info['generated_depth_file'], delimiter=',')
                    if gen_data.shape == (481, 640):
                        gen_data = gen_data[1:, :]
                    if split == 'test' and self.test_gt_addition:
                        gen_data = self.add_value_to_gt_locations(gen_data, gt_mask, self.test_gt_addition)
                    temp_gen = str(info['generated_depth_file']).replace('.csv', '_temp.csv')
                    np.savetxt(temp_gen, gen_data, delimiter=',')
                    lm_gen = find_local_max.FindLocalMax(
                        csv_path=temp_gen,
                        rgb_image_path=info['img_file'],
                        ground_truth=dilated_path,
                        real_depth=False
                    )
                    gen_norm = lm_gen.process(save=False)
                    gen_out = self.output_dir / split / 'depth_generated' / f"{info['img_file'].stem}.png"
                    Image.fromarray(gen_norm.astype(np.uint8)).save(gen_out)
                    os.remove(temp_gen)

                # Grayscale CLAHE
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                gray_eq = clahe.apply(gray)
                gray_out = self.output_dir / split / 'gray' / f"{info['img_file'].stem}.png"
                cv2.imwrite(str(gray_out), gray_eq)

        # Summary
        print("\n📊 Splits prepared:")
        for s, items in splits.items():
            print(f"  {s}: {len(items)} samples")
        print("✅ Completed!")


if __name__ == '__main__':
    for seed in [42, 555, 613, 2025]:
        for test_gt_addition in [0]:
            print(f"\n🎲 Processing with seed {seed} and GT addition {test_gt_addition}")
            root_dir = Path.cwd().parent
            color_images_dir = root_dir / "Images" / "detectron_masked_images"
            generated_depth_dir = root_dir / "detectron_generated_depths"
            real_depth_dir = root_dir / "Images" / "csv_files"
            gt_dir = root_dir / "Images" / "gt"
            output_dir = root_dir / 'ucnet' / "placenta_data" / f"seed_{seed}_gt_add_{test_gt_addition}"

            os.makedirs(output_dir, exist_ok=True)

            # Initialize organizer with both real and generated depth directories
            organizer = DatasetOrganizer(
                color_images_dir=color_images_dir,
                generated_depth_dir=generated_depth_dir,
                gt_dir=gt_dir,
                output_dir=output_dir,
                real_depth_dir=real_depth_dir,
                seed=seed,
                test_gt_addition=test_gt_addition
            )

            organizer.prepare_splits()
            print(f"✅ Completed processing for seed {seed} with GT addition {test_gt_addition}")