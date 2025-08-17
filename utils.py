#!/usr/bin/env python3
# Utilities for binary dataset loading, class weights, and plotting

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _make_ds_from_dir(dir_path, img_size, batch_size, shuffle=True):
    return keras.preprocessing.image_dataset_from_directory(
        dir_path,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="binary",  # 0/1 labels; class_names[1] maps to label 1
        shuffle=shuffle,
    )

def get_datasets_binary(data_dir: str, img_size: int, batch_size: int):
    """
    Returns: (train_ds, val_ds, class_names)
    Expects:
      data/
        train/
          benign/
          melanoma/
        val/   (optional; if missing, uses validation_split on train/)
          benign/
          melanoma/
    """
    train_root = os.path.join(data_dir, "train")
    val_root = os.path.join(data_dir, "val")

    if os.path.isdir(val_root):
        train_ds = _make_ds_from_dir(train_root, img_size, batch_size, shuffle=True)
        val_ds   = _make_ds_from_dir(val_root, img_size, batch_size, shuffle=False)
        class_names = train_ds.class_names
    else:
        train_ds = keras.preprocessing.image_dataset_from_directory(
            train_root,
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode="binary",
            shuffle=True,
        )
        val_ds = keras.preprocessing.image_dataset_from_directory(
            train_root,
            validation_split=0.2,
            subset="validation",
            seed=1337,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode="binary",
            shuffle=False,
        )
        class_names = train_ds.class_names

    # Cache & prefetch
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names

def compute_class_weights_binary(train_ds):
    """
    Compute class weights for binary labels from a tf.data Dataset.
    Returns dict {0: w0, 1: w1}
    """
    counts = np.zeros(2, dtype=np.int64)
    for _, y in train_ds.unbatch():
        # y is scalar float32 (0. or 1.)
        counts[int(y.numpy())] += 1
    total = counts.sum()
    weights = total / (2 * np.maximum(counts, 1))
    return {0: float(weights[0]), 1: float(weights[1])}

def plot_history_curves_binary(history, out_dir: str, prefix: str = ""):
    os.makedirs(out_dir, exist_ok=True)

    # Loss
    fig = plt.figure()
    plt.plot(history.get("loss", []), label="train")
    plt.plot(history.get("val_loss", []), label="val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("BinaryCrossentropy"); plt.legend()
    fig.savefig(os.path.join(out_dir, f"{prefix}_loss.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    # AUC
    if "auc" in history:
        fig = plt.figure()
        plt.plot(history.get("auc", []), label="train")
        plt.plot(history.get("val_auc", []), label="val")
        plt.title("AUC"); plt.xlabel("Epoch"); plt.ylabel("AUC"); plt.legend()
        fig.savefig(os.path.join(out_dir, f"{prefix}_auc.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)

    # Accuracy
    if "accuracy" in history:
        fig = plt.figure()
        plt.plot(history.get("accuracy", []), label="train")
        plt.plot(history.get("val_accuracy", []), label="val")
        plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
        fig.savefig(os.path.join(out_dir, f"{prefix}_acc.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)

    # Precision
    if "precision" in history:
        fig = plt.figure()
        plt.plot(history.get("precision", []), label="train")
        plt.plot(history.get("val_precision", []), label="val")
        plt.title("Precision"); plt.xlabel("Epoch"); plt.ylabel("Precision"); plt.legend()
        fig.savefig(os.path.join(out_dir, f"{prefix}_precision.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)

    # Recall
    if "recall" in history:
        fig = plt.figure()
        plt.plot(history.get("recall", []), label="train")
        plt.plot(history.get("val_recall", []), label="val")
        plt.title("Recall"); plt.xlabel("Epoch"); plt.ylabel("Recall"); plt.legend()
        fig.savefig(os.path.join(out_dir, f"{prefix}_recall.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
