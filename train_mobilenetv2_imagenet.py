"""
Reference training script: MobileNetV2 on ImageNet (ILSVRC-2012).

This is the recipe behind the pretrained MobileNetV2 backbone that
train_mask_detector.py fine-tunes for mask / no-mask classification
(Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks").
It is provided for documentation and is NOT meant to be run as part of this
project: full ImageNet training needs ~1.28M images (~150 GB) and multiple
GPUs for days. In practice we load the published weights via
`MobileNetV2(weights="imagenet")`.

Expected data layout (standard ImageNet folder structure):
    imagenet/train/<wnid>/*.JPEG    (1000 classes)
    imagenet/val/<wnid>/*.JPEG

Usage (if you really want to train):
    python train_mobilenetv2_imagenet.py --data imagenet --epochs 300 --batch-size 256
"""
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay

IMG_SIZE = 224
NUM_CLASSES = 1000
TRAIN_IMAGES = 1_281_167


def build_dataset(path, batch_size, training):
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        label_mode="categorical",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=training,
    )
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomZoom(0.1),
    ])

    def prep(x, y):
        if training:
            x = augment(x, training=True)
        return preprocess_input(x), y

    return ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="imagenet")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.045)
    ap.add_argument("--out", default="mobilenetv2_imagenet.h5")
    args = ap.parse_args()

    strategy = tf.distribute.MirroredStrategy()
    print(f"[INFO] training on {strategy.num_replicas_in_sync} device(s)")

    train_ds = build_dataset(os.path.join(args.data, "train"), args.batch_size, True)
    val_ds = build_dataset(os.path.join(args.data, "val"), args.batch_size, False)

    # Paper recipe: RMSprop (decay/momentum 0.9), lr 0.045 decayed by 0.98 per 2.5 epochs.
    steps_per_epoch = TRAIN_IMAGES // args.batch_size
    lr = ExponentialDecay(args.lr, decay_steps=int(2.5 * steps_per_epoch),
                          decay_rate=0.98, staircase=True)

    with strategy.scope():
        model = MobileNetV2(weights=None, classes=NUM_CLASSES,
                            input_shape=(IMG_SIZE, IMG_SIZE, 3), alpha=1.0)
        model.compile(
            optimizer=RMSprop(lr, rho=0.9, momentum=0.9, epsilon=1.0),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(5, name="top5")],
        )

    callbacks = [
        ModelCheckpoint(args.out, monitor="val_accuracy", save_best_only=True, verbose=1),
        CSVLogger("mobilenetv2_imagenet_log.csv"),
        TensorBoard(log_dir="logs/mobilenetv2_imagenet"),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    print(f"[INFO] saved best ImageNet model to {args.out}")


if __name__ == "__main__":
    main()
