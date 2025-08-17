#!/usr/bin/env python3
# Batch inference for melanoma vs benign using a trained VGG model.

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg19

PREPROCESS = {
    "vgg16": preprocess_vgg16,
    "vgg19": preprocess_vgg19,
}

def parse_args():
    p = argparse.ArgumentParser(description="Predict melanoma probability for images in a folder.")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--meta_path", type=str, required=True, help="Path to meta.json saved during training")
    p.add_argument("--class_names", type=str, required=True, help="Path to class_names.json saved during training")
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--out", type=str, default="predictions.csv")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()

def list_image_files(root):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    return [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True) if p.lower().endswith(exts)]

def load_image(path, img_size):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size), method="bilinear")
    img = tf.cast(img, tf.float32)  # keep 0..255 range for VGG preprocess
    return img

def main():
    args = parse_args()

    with open(args.meta_path, "r") as f:
        meta = json.load(f)
    with open(args.class_names, "r") as f:
        class_names = json.load(f)

    backbone = meta.get("backbone", "vgg16")
    img_size = int(meta.get("img_size", 224))
    preprocess = PREPROCESS[backbone]

    files = list_image_files(args.images_dir)
    if not files:
        raise SystemExit(f"No images found in {args.images_dir}")

    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.map(lambda p: (load_image(p, img_size), p), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = keras.models.load_model(args.model_path, compile=False)

    probs = []
    paths = []
    for batch_imgs, batch_paths in ds:
        # Apply the same preprocessing the model used internally during training:
        batch_imgs = preprocess(batch_imgs)
        p = model.predict(batch_imgs, verbose=0).reshape(-1)
        probs.extend(p.tolist())
        paths.extend([x.numpy().decode("utf-8") for x in batch_paths])

    # By convention, sigmoid is P(class 1). In Keras folder loading, class_names[1] is label 1.
    positive_name = class_names[1]
    negative_name = class_names[0]
    preds = (np.array(probs) >= 0.5).astype(int)
    pred_labels = [positive_name if v == 1 else negative_name for v in preds]

    df = pd.DataFrame({
        "filepath": paths,
        f"prob_{positive_name}": probs,
        "pred_label": pred_labels
    })
    df.to_csv(args.out, index=False)
    print(f"[INFO] Saved predictions to {args.out} ({len(df)} rows)")

if __name__ == "__main__":
    main()
