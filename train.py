#!/usr/bin/env python3
# Train a melanoma vs benign classifier with VGG16/VGG19 transfer learning.

import os
import json
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as preprocess_vgg19

from utils import (
    set_seed,
    ensure_dir,
    get_datasets_binary,
    compute_class_weights_binary,
    plot_history_curves_binary,
)

BACKBONES = {
    "vgg16": (VGG16, preprocess_vgg16),
    "vgg19": (VGG19, preprocess_vgg19),
}

def build_model(img_size: int, backbone_name: str) -> tuple[keras.Model, keras.Model, str]:
    """Create VGG backbone (frozen) + binary head. Returns (model, base, preprocess_name)."""
    if backbone_name not in BACKBONES:
        raise ValueError(f"backbone must be one of {list(BACKBONES.keys())}")
    Backbone, preprocess = BACKBONES[backbone_name]

    inputs = keras.Input(shape=(img_size, img_size, 3))

    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")
    x = aug(inputs)

    # VGG preprocessing (expects 0..255 RGB tensors)
    x = layers.Lambda(preprocess, name="preprocess")(x)

    base = Backbone(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False  # Stage 1: freeze

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"Melanoma_{backbone_name.upper()}")
    return model, base, backbone_name

def parse_args():
    p = argparse.ArgumentParser(description="Melanoma vs Benign (VGG16/VGG19 transfer learning)")
    p.add_argument("--data_dir", type=str, default="data", help="Root with train/ val/ (optional test/)")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--backbone", type=str, default="vgg16", choices=list(BACKBONES.keys()))
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_finetune", type=float, default=1e-5)
    p.add_argument("--fine_tune_at", type=int, default=15, help="Unfreeze from this base layer index")
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir); ensure_dir(os.path.join(args.output_dir, "plots"))

    # Datasets (binary labels)
    train_ds, val_ds, class_names = get_datasets_binary(
        data_dir=args.data_dir, img_size=args.img_size, batch_size=args.batch_size
    )
    if len(class_names) != 2:
        raise SystemExit(f"Expected 2 classes, found: {class_names}")
    print(f"[INFO] Classes: {class_names}  (label 1 is '{class_names[1]}')")

    # Save class names & metadata
    with open(os.path.join(args.output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    # Build model
    model, base, backbone_name = build_model(args.img_size, args.backbone)

    # Loss/metrics
    metrics = [
        keras.metrics.AUC(name="auc"),
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]

    # Class weights for imbalance
    class_weights = compute_class_weights_binary(train_ds)
    print(f"[INFO] Class weights: {class_weights}")

    # Stage 1: train head only
    model.compile(optimizer=keras.optimizers.Adam(args.lr_head),
                  loss="binary_crossentropy",
                  metrics=metrics)

    ckpt_best = os.path.join(args.output_dir, "best_auc.h5")
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_best, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]

    print("[INFO] Stage 1: training head (VGG frozen)...")
    hist1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                      class_weight=class_weights, callbacks=callbacks, verbose=1)
    plot_history_curves_binary(hist1.history, os.path.join(args.output_dir, "plots"), prefix="stage1")
    with open(os.path.join(args.output_dir, "history_stage1.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in hist1.history.items()}, f)

    # Stage 2: fine-tune top layers
    print(f"[INFO] Stage 2: fine-tuning from layer index {args.fine_tune_at} ...")
    base.trainable = True
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= args.fine_tune_at)

    model.compile(optimizer=keras.optimizers.Adam(args.lr_finetune),
                  loss="binary_crossentropy",
                  metrics=metrics)

    callbacks_ft = [
        keras.callbacks.ModelCheckpoint(ckpt_best, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ]

    hist2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                      class_weight=class_weights, callbacks=callbacks_ft, verbose=1)
    plot_history_curves_binary(hist2.history, os.path.join(args.output_dir, "plots"), prefix="stage2")
    with open(os.path.join(args.output_dir, "history_stage2.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in hist2.history.items()}, f)

    # Save final + metadata
    final_path = os.path.join(args.output_dir, "last.h5")
    model.save(final_path)
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump({"backbone": backbone_name, "img_size": args.img_size}, f)
    print(f"[INFO] Saved final model to {final_path} and best-by-AUC to {ckpt_best}")

if __name__ == "__main__":
    main()
