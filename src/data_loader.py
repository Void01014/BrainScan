import logging
import os
import random

import cv2
import numpy as np

logging.basicConfig(
    filename="loader.log",
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)

VALID_EXT = (".jpeg", ".jpg", ".bmp", ".png")
IMG_SIZE = (224, 224)


def get_classes(data_dir):
    """Class names = subfolder names."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")
    return sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )


def load_dataset(data_dir, img_size=IMG_SIZE, seed=42):
    """Return images (N,H,W,3) uint8, labels (N,) str, classes list."""
    classes = get_classes(data_dir)
    if not classes:
        raise ValueError(f"No class folders in {data_dir}")

    samples = []
    skipped = 0

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            path = os.path.join(cls_dir, fname)

            if not fname.lower().endswith(VALID_EXT):
                logging.warning("Invalid extension, skipped: %s", path)
                skipped += 1
                continue

            try:
                img = cv2.imread(path)
                if img is None:
                    raise ValueError("cv2.imread returned None")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                samples.append((img, cls))
            except Exception as e:
                logging.warning("Failed to load %s: %s", path, e)
                skipped += 1

    random.Random(seed).shuffle(samples)  # image and label shuffled together

    images = np.array([s[0] for s in samples], dtype=np.uint8)
    labels = np.array([s[1] for s in samples])

    print(f"Loaded: {len(images)} | Skipped: {skipped} | Classes: {classes}")
    return images, labels, classes
