import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import yaml
import re
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt


class PlacentaYOLOPipeline:
    def __init__(self, data_dir, seed, output_dir, min_contour_area=100, split_ratios=(0.7, 0.15, 0.15)):
        """
        Initialize the pipeline for preparing and training YOLO model using either:
        1) existing splits at {data_dir}/train|val|test/(img|gt), OR
        2) a single dataset folder {data_dir}/(img|gt) to be randomly split (deterministically by seed).

        Args:
            data_dir (str): Path to dataset root. Can be absolute or relative (e.g., 'placenta_data/detectron_seed_1' or 'ucnet/placenta_data/detectron_seed_1').
            seed (int): Random seed for deterministic splitting.
            output_dir (Path): Directory for YOLO dataset and results.
            min_contour_area (int): Minimum contour area to consider.
            split_ratios (tuple): (train, val, test) ratios, must sum to 1.0 for unsplit mode.
        """
        self.data_dir = data_dir
        self.seed = seed
        # Allow both 'placenta_data/...' and 'ucnet/placenta_data/...'
        root = Path(data_dir)
        if not root.exists():
            root = Path("../ucnet") / data_dir
        self.base_path = root
        self.output_dir = Path(output_dir)
        self.min_contour_area = min_contour_area
        self.split_ratios = split_ratios

        # Helper method for extracting datetime from filenames
        def extract_datetime(filename):
            """Extract datetime pattern from filename (YYYY-MM-DD_HH-MM-SS). Fallback to stem (basename without ext)."""
            s = str(filename)
            match = re.search(r'(\d{4}-\d{2}-\d{2}[_-]\d{2}[-_]\d{2}[-_]\d{2})', s)
            if match:
                return match.group(1)
            # fallback: try just the stem without extension
            try:
                return Path(s).stem
            except Exception:
                return None

        self.extract_datetime = extract_datetime

        # Detect whether predefined splits exist
        self.has_predefined_splits = (self.base_path / 'train').exists() and (self.base_path / 'val').exists() and (self.base_path / 'test').exists()

        # Define paths for existing splits (if present)
        self.split_paths = {
            'train': {
                'img': self.base_path / 'train' / 'img',
                'gt':  self.base_path / 'train' / 'gt'
            },
            'val': {
                'img': self.base_path / 'val' / 'img',
                'gt':  self.base_path / 'val' / 'gt'
            },
            'test': {
                'img': self.base_path / 'test' / 'img',
                'gt':  self.base_path / 'test' / 'gt'
            }
        }

        # Also record unsplit paths (if no predefined splits)
        self.unsplit_img_dir = self.base_path / 'img'
        self.unsplit_gt_dir  = self.base_path / 'gt'

        # Create YOLO directory structure
        self.yolo_dir = self.output_dir / f'yolo_dataset_{seed}'
        for split in ['train', 'val', 'test']:
            (self.yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    def extract_hour_key(self, filename: str):
        """
        Convert the datetime token to an hour bucket key: 'YYYY-MM-DD_HH'.
        Falls back to filename stem when missing.
        """
        dt = self.extract_datetime(filename)  # e.g., '2025-01-06_18-00-15'
        if not dt:
            return None
        # Normalize separators and cut to hour
        s = dt.replace('-', ':').replace('_', ':')
        # s looks like '2025:01:06:18:00:15'
        parts = s.split(':')
        if len(parts) >= 4:
            yyyy, mm, dd, hh = parts[:4]
            return f"{yyyy}-{mm}-{dd}_{hh}"
        # Fallback: try first 13 chars if already of form 'YYYY-MM-DD_HH...'
        return dt[:13] if len(dt) >= 13 else dt

    def _get_matched_files(self):
        """
        Produce a list of dicts [{'dt', 'img_file', 'gt_file'}, ...]
        by aggregating ALL pairs (predefined or unsplit).
        """
        all_pairs = {}
        if self.has_predefined_splits:
            for split in ['train', 'val', 'test']:
                pairs = self._gather_pairs(self.split_paths[split]['img'], self.split_paths[split]['gt'])
                all_pairs.update(pairs)
        else:
            pairs = self._gather_pairs(self.unsplit_img_dir, self.unsplit_gt_dir)
            all_pairs.update(pairs)

        out = []
        for dt, (img_p, gt_p) in all_pairs.items():
            out.append({"dt": dt, "img_file": img_p, "gt_file": gt_p})
        return out

    def prepare_splits(self, split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}):
        """
        Time-aware splitting:
          - Force any filename containing 'real' (case-insensitive) into test.
          - Group remaining samples by hour key (YYYY-MM-DD_HH) so temporally close frames
            stay in the same split.
          - Seed-shuffle group keys (deterministic by self.seed), then slice by ratios.
        Returns: dict split_name -> list of (dt, img_path, gt_path)
        """
        matched = self._get_matched_files()

        # Force “real” to test
        forced_test = [m for m in matched if 'real' in m['img_file'].name.lower()]
        print(f"✨ Forcing {len(forced_test)} 'real' samples into test set")
        remaining = [m for m in matched if m not in forced_test]

        # Group by hour key
        groups = {}
        unknown = []
        for m in remaining:
            hk = self.extract_hour_key(m['img_file'].name)
            if hk:
                groups.setdefault(hk, []).append(m)
            else:
                unknown.append(m)

        # Seed-shuffle group keys
        keys = sorted(groups)
        import random as _random
        rng = _random.Random(self.seed)
        rng.shuffle(keys)

        n = len(keys)
        n_train = int(round(split_ratios['train'] * n))
        n_val = int(round(split_ratios['val'] * n))
        n_test = n - n_train - n_val

        train_keys = keys[:n_train]
        val_keys = keys[n_train:n_train + n_val]
        test_keys = keys[n_train + n_val:]

        def expand(keys_):
            return [i for k in keys_ for i in groups[k]]

        train_list = expand(train_keys)
        val_list = expand(val_keys)
        test_list = expand(test_keys) + forced_test + unknown  # put unknown safely into test

        # Convert to the tuples your pipeline already uses
        splits = {
            'train': [(m['dt'], m['img_file'], m['gt_file']) for m in train_list],
            'val': [(m['dt'], m['img_file'], m['gt_file']) for m in val_list],
            'test': [(m['dt'], m['img_file'], m['gt_file']) for m in test_list],
        }

        # Stats
        for s in ['train', 'val', 'test']:
            print(f"{s}: {len(splits[s])} pairs (groups: "
                  f"{len(set(self.extract_hour_key(Path(t[1]).name) for t in splits[s] if self.extract_hour_key(Path(t[1]).name)))} )")

        # Audit files
        audit_dir = self.output_dir / "metrics" / f"split_seed{self.seed}"
        audit_dir.mkdir(parents=True, exist_ok=True)
        for s in ['train', 'val', 'test']:
            with open(audit_dir / f"{s}_items.txt", "w") as f:
                for dt, _, _ in splits[s]:
                    f.write(dt + "\n")

        return splits
    def _gather_pairs(self, img_dir: Path, gt_dir: Path):
        """
        Gather (image_path, mask_path) pairs matched by datetime token or filename stem.
        Supports common image extensions.
        """
        if not img_dir.exists() or not gt_dir.exists():
            return {}

        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
        img_files = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
        gt_files  = [p for p in gt_dir.rglob("*") if p.suffix.lower() in exts]

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

        common_datetimes = list(set(img_dict.keys()) & set(gt_dict.keys()))
        common_datetimes.sort()
        pairs = {dt: (img_dict[dt], gt_dict[dt]) for dt in common_datetimes}
        return pairs

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
        Prepare a YOLO-format dataset.
        If predefined splits exist under base_path/train|val|test, mirror them.
        Otherwise, split a single dataset at base_path/(img|gt) into train/val/test deterministically by self.seed.
        """
        # Clean YOLO output directories
        for split in ['train', 'val', 'test']:
            images_dir = self.yolo_dir / split / 'images'
            labels_dir = self.yolo_dir / split / 'labels'
            shutil.rmtree(images_dir, ignore_errors=True)
            shutil.rmtree(labels_dir, ignore_errors=True)
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

        stats = {split: {'total': 0, 'with_defects': 0, 'total_defects': 0} for split in ['train', 'val', 'test']}

        # Build time-aware, seed-deterministic splits from the entire dataset
        # (groups by hour key and forces files containing 'real' into test)
        splits_to_use = self.prepare_splits(
            split_ratios={
                'train': self.split_ratios[0],
                'val': self.split_ratios[1],
                'test': self.split_ratios[2]
            }
        )

        # Process each split
        for split in ['train', 'val', 'test']:
            for dt, img_file, mask_file in tqdm(splits_to_use[split]):
                # Read and process the color image (apply CLAHE)
                color_img = cv2.imread(str(img_file))
                if color_img is None:
                    # skip unreadable images
                    continue
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
                if mask is None:
                    # skip if mask unreadable
                    (self.yolo_dir / split / 'labels' / f"{dt}.txt").touch()
                    continue
                bboxes = self.mask_to_bboxes(mask)

                label_filename = f"{dt}.txt"
                label_file = self.yolo_dir / split / 'labels' / label_filename

                if bboxes:
                    with open(label_file, 'w') as f:
                        for bbox in bboxes:
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
        """Train YOLOv8 model and archive per-epoch CSV."""
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
        # Archive the training results.csv (per-epoch metrics) into ./metrics
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Most reliable source of the run directory after training:
            run_dir = Path(getattr(model.trainer, "save_dir", getattr(results, "save_dir", self.output_dir / "runs")))
            src_csv = run_dir / "results.csv"
            if src_csv.exists():
                dst_csv = metrics_dir / f"train_seed{self.seed}_results.csv"
                shutil.copy2(src_csv, dst_csv)
                print(f"Archived per-epoch CSV → {dst_csv}")
        except Exception as e:
            print(f"WARNING: could not archive training CSV: {e}")
        return model, results

    def _metrics_to_dict(self, metrics_obj):
        """
        Normalize Ultralytics val() metrics to a simple dict with keys:
        precision, recall, map50, map, map75 (if available), fitness (if available).
        Works across several Ultralytics versions by trying common fields.
        """
        out = {}

        # Prefer results_dict if available
        rd = getattr(metrics_obj, "results_dict", None)
        if isinstance(rd, dict):
            def pick(klist):
                for k in klist:
                    if k in rd:
                        return rd[k]
                return None
            out["precision"] = pick(["metrics/precision(B)", "precision", "box/precision"])
            out["recall"]    = pick(["metrics/recall(B)", "recall", "box/recall"])
            out["map50"]     = pick(["metrics/mAP50(B)", "mAP50", "box/mAP50"])
            out["map"]       = pick(["metrics/mAP50-95(B)", "mAP50-95", "box/mAP"])
            out["fitness"]   = pick(["fitness"])
            maps = pick(["maps", "box/maps"])
            if maps is not None and isinstance(maps, (list, tuple)) and len(maps) >= 3:
                out["map75"] = maps[2]
            return out

        # Fallback to attribute-style object (older API)
        box = getattr(metrics_obj, "box", None)
        if box is not None:
            out["precision"] = getattr(box, "p", None)
            out["recall"]    = getattr(box, "r", None)
            out["map50"]     = getattr(box, "map50", None)
            out["map"]       = getattr(box, "map", None)
            maps = getattr(box, "maps", None)
            if maps is not None and isinstance(maps, (list, tuple)) and len(maps) >= 3:
                out["map75"] = maps[2]
        return out

    @staticmethod
    def _iou_xyxy(a, b):
        """
        Compute IoU between two boxes in [x1,y1,x2,y2] format.
        """
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter + 1e-9
        return inter / union

    def _load_gt_xyxy(self, stem: str):
        """
        Load ground-truth boxes for an image stem from YOLO label .txt and convert to xyxy.
        Returns a list of [x1,y1,x2,y2]. If no file or empty → [].
        """
        img_path = self.yolo_dir / "test" / "images" / f"{stem}.jpg"
        if not img_path.exists():
            # try common alternatives
            for ext in [".png", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]:
                alt = img_path.with_suffix(ext)
                if alt.exists():
                    img_path = alt
                    break
        img = cv2.imread(str(img_path))
        if img is None:
            return []
        H, W = img.shape[:2]

        label_path = self.yolo_dir / "test" / "labels" / f"{stem}.txt"
        if not label_path.exists():
            return []

        gt_xyxy = []
        try:
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    # parts: class cx cy w h (normalized)
                    cx, cy, w, h = map(float, parts[1:6])
                    x1 = (cx - w / 2.0) * W
                    y1 = (cy - h / 2.0) * H
                    x2 = (cx + w / 2.0) * W
                    y2 = (cy + h / 2.0) * H
                    gt_xyxy.append([x1, y1, x2, y2])
        except Exception:
            pass
        return gt_xyxy

    def _compute_cc_and_image_metrics(self, results_list, iou_thresh=0.3, conf_thresh=0.20):
        """
        Compute:
          - CC-level metrics: TP/FP/FN from matching predicted boxes to GT boxes (IoU>=iou_thresh),
                              then Precision/Recall/F1.
          - Image-level metrics: presence/absence prediction vs. GT presence, yielding
                                 TP_img/FP_img/TN_img/FN_img and Accuracy/Precision/Recall/F1.

        Parameters
        ----------
        results_list : list of Ultralytics 'Results' (as returned by model.predict)
        iou_thresh : float, IoU threshold to count a detection as a TP at component level
        conf_thresh: float, confidence threshold to keep predictions

        Returns
        -------
        metrics_cc : dict with TP, FP, FN, precision, recall, f1
        metrics_img: dict with TP_img, FP_img, TN_img, FN_img, accuracy, precision, recall, f1
        per_image_rows : list of dict rows for optional CSV (per-image breakdown)
        """
        TP = FP = FN = 0
        TP_img = FP_img = TN_img = FN_img = 0
        per_image_rows = []

        for res in results_list:
            # res.path is full path to image file
            stem = Path(res.path).stem
            gt_boxes = self._load_gt_xyxy(stem)  # list of [x1,y1,x2,y2]

            # predicted boxes (xyxy) filtered by conf
            pred_xyxy = []
            if hasattr(res, "boxes") and res.boxes is not None:
                # res.boxes.xyxy is Nx4, res.boxes.conf Nx1
                xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") else res.boxes.xyxy
                confs = res.boxes.conf.cpu().numpy().reshape(-1) if hasattr(res.boxes.conf, "cpu") else np.array(res.boxes.conf).reshape(-1)
                for i in range(xyxy.shape[0]):
                    if confs[i] >= conf_thresh:
                        pred_xyxy.append(xyxy[i].tolist())

            # Greedy match: sort preds by confidence desc (if available), match best IoU gt not already matched
            # For simplicity, they are already in model order; we'll sort by area desc as fallback
            # (Ultralytics results usually are already sorted by conf)
            matched_gt = set()
            tp_local = 0
            fp_local = 0

            for p in pred_xyxy:
                # find best GT
                best_iou = 0.0
                best_j = -1
                for j, g in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    iou = self._iou_xyxy(p, g)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou >= iou_thresh and best_j >= 0:
                    matched_gt.add(best_j)
                    TP += 1
                    tp_local += 1
                else:
                    FP += 1
                    fp_local += 1

            # Any GT not matched → FN
            fn_local = len(gt_boxes) - len(matched_gt)
            FN += fn_local

            # Image-level presence
            gt_has = len(gt_boxes) > 0
            pred_has = len(pred_xyxy) > 0
            if pred_has and gt_has:
                TP_img += 1
            elif pred_has and not gt_has:
                FP_img += 1
            elif (not pred_has) and (not gt_has):
                TN_img += 1
            else:
                FN_img += 1

            per_image_rows.append({
                "image": stem,
                "n_gt": len(gt_boxes),
                "n_pred": len(pred_xyxy),
                "tp": tp_local,
                "fp": fp_local,
                "fn": fn_local,
                "gt_has": int(gt_has),
                "pred_has": int(pred_has),
                "tp_img": int(pred_has and gt_has),
                "fp_img": int(pred_has and not gt_has),
                "tn_img": int((not pred_has) and (not gt_has)),
                "fn_img": int((not pred_has) and gt_has),
            })

        # CC-level metrics
        prec_cc = TP / (TP + FP + 1e-9)
        rec_cc  = TP / (TP + FN + 1e-9)
        f1_cc   = 2 * prec_cc * rec_cc / (prec_cc + rec_cc + 1e-9)
        metrics_cc = {"TP": TP, "FP": FP, "FN": FN, "precision": float(prec_cc), "recall": float(rec_cc), "f1": float(f1_cc)}

        # Image-level metrics
        total_imgs = TP_img + FP_img + TN_img + FN_img
        acc_img = (TP_img + TN_img) / (total_imgs + 1e-9)
        prec_img = TP_img / (TP_img + FP_img + 1e-9)
        rec_img = TP_img / (TP_img + FN_img + 1e-9)
        f1_img = 2 * prec_img * rec_img / (prec_img + rec_img + 1e-9)
        metrics_img = {
            "TP_img": TP_img, "FP_img": FP_img, "TN_img": TN_img, "FN_img": FN_img,
            "accuracy": float(acc_img), "precision": float(prec_img), "recall": float(rec_img), "f1": float(f1_img),
            "total_images": int(total_imgs)
        }

        return metrics_cc, metrics_img, per_image_rows

    def evaluate_model(self, model):
        """Evaluate the trained model and write summary JSON/CSV, including CC-level and image-level metrics."""
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
        # Serialize standard Ultralytics summary
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        md = self._metrics_to_dict(metrics)
        md = {k: (None if v is None else float(v)) for k, v in md.items()}
        md["seed"] = int(self.seed)
        # JSON
        json_path = metrics_dir / f"test_seed{self.seed}_summary.json"
        with open(json_path, "w") as f:
            json.dump(md, f, indent=2)
        # CSV (single row)
        csv_path = metrics_dir / f"test_seed{self.seed}_summary.csv"
        cols = ["seed", "precision", "recall", "map50", "map", "map75", "fitness"]
        with open(csv_path, "w") as f:
            f.write(",".join(cols) + "\n")
            row = [md.get(c, None) for c in cols]
            def fmt(x):
                return "" if x is None else (str(int(x)) if isinstance(x, bool) else f"{x}")
            f.write(",".join(fmt(x) for x in row) + "\n")
        print(f"Saved test summary → {json_path} and {csv_path}")

        # ---- Custom CC-level and Image-level metrics ----
        cc_metrics, img_metrics, per_image_rows = self._compute_cc_and_image_metrics(results, iou_thresh=0.3, conf_thresh=0.20)

        # Save CC metrics
        cc_json = metrics_dir / f"test_seed{self.seed}_cc_metrics.json"
        with open(cc_json, "w") as f:
            json.dump(cc_metrics, f, indent=2)
        cc_csv = metrics_dir / f"test_seed{self.seed}_cc_metrics.csv"
        with open(cc_csv, "w") as f:
            f.write("TP,FP,FN,precision,recall,f1\n")
            f.write(",".join([str(cc_metrics[k]) for k in ["TP","FP","FN","precision","recall","f1"]]) + "\n")

        # Save image-level metrics
        img_json = metrics_dir / f"test_seed{self.seed}_image_metrics.json"
        with open(img_json, "w") as f:
            json.dump(img_metrics, f, indent=2)
        img_csv = metrics_dir / f"test_seed{self.seed}_image_metrics.csv"
        with open(img_csv, "w") as f:
            f.write("TP_img,FP_img,TN_img,FN_img,accuracy,precision,recall,f1,total_images\n")
            f.write(",".join([str(img_metrics[k]) for k in ["TP_img","FP_img","TN_img","FN_img","accuracy","precision","recall","f1","total_images"]]) + "\n")

        # Save per-image breakdown
        per_img_csv = metrics_dir / f"test_seed{self.seed}_per_image_breakdown.csv"
        if per_image_rows:
            headers = ["image","n_gt","n_pred","tp","fp","fn","gt_has","pred_has","tp_img","fp_img","tn_img","fn_img"]
            with open(per_img_csv, "w") as f:
                f.write(",".join(headers) + "\n")
                for r in per_image_rows:
                    f.write(",".join(str(r[h]) for h in headers) + "\n")
        print(f"Saved CC & Image metrics → {cc_json}, {img_json} and per-image breakdown → {per_img_csv}")

        return metrics, results


def aggregate_cross_seed_metrics(output_dir: Path, seeds_run=None):
    """
    Scan ./metrics for per-seed test summaries, then compute and save mean ± std.
    Writes:
      - cross_seed_summary.json
      - cross_seed_summary.csv
    """
    metrics_dir = output_dir / "metrics"
    rows = []
    if not metrics_dir.exists():
        print("No metrics directory found; skipping aggregation.")
        return

    for p in metrics_dir.glob("test_seed*_summary.json"):
        try:
            with open(p) as f:
                rows.append(json.load(f))
        except Exception as e:
            print(f"WARNING: could not load {p}: {e}")

    if not rows:
        print("No test summaries found; skipping aggregation.")
        return

    def agg(key):
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals:
            return None, None
        a = np.array(vals, float)
        mean = float(a.mean())
        std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        return mean, std

    summary = {}
    for k in ["precision", "recall", "map50", "map", "map75", "fitness"]:
        m, s = agg(k)
        if m is not None:
            summary[k] = {"mean": m, "std": s}

    # Print to console
    print("\n=== Cross-Seed Test Metrics (mean ± std) ===")
    for k, d in summary.items():
        print(f"{k:>9}: {d['mean']:.4f} ± {d['std']:.4f}")

    # Persist JSON + CSV
    metrics_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(metrics_dir / "cross_seed_summary.json", "w") as f:
            json.dump({
                "seeds": sorted({int(r["seed"]) for r in rows if "seed" in r}),
                "summary": summary
            }, f, indent=2)
    except Exception as e:
        print(f"WARNING: could not write cross_seed_summary.json: {e}")

    try:
        csv_path = metrics_dir / "cross_seed_summary.csv"
        with open(csv_path, "w") as f:
            f.write("metric,mean,std\n")
            for k, d in summary.items():
                f.write(f"{k},{d['mean']},{d['std']}\n")
        print(f"Saved aggregated summary → {metrics_dir / 'cross_seed_summary.json'} and {csv_path}")
    except Exception as e:
        print(f"WARNING: could not write cross_seed_summary.csv: {e}")


def main():
    data_dir = "../ucnet/placenta_data/detectron_seed_1"  # or: placenta_data/detectron_seed_1
    seeds =    [ 42, 555, 613, 2025]

    output_dir = Path("")

    for seed in tqdm(seeds):
        try:
            # Initialize pipeline (auto-detect predefined splits or split unsplit dataset deterministically by seed)
            pipeline = PlacentaYOLOPipeline(
                data_dir=data_dir,
                seed=seed,
                output_dir=output_dir,
                min_contour_area=100,
                split_ratios=(0.7, 0.15, 0.15),
            )

            # Prepare dataset using existing splits or split unsplit dataset
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
        except KeyboardInterrupt:
            print(f"\nInterrupted during seed {seed}. Proceeding to aggregation of available results...")
            break

    # Aggregate whatever test summaries exist
    aggregate_cross_seed_metrics(output_dir, seeds_run=seeds)


if __name__ == "__main__":
    main()
# """
#  seed 1 Class     Images  Instances      Box(P          R      mAP50  m