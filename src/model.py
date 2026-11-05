"""VGG16 model builders for baseline and separately labeled extensions."""

from __future__ import annotations

import tensorflow as tf


def build_vgg16_model(
    learning_rate: float = 1e-4,
    fine_tune_block5: bool = False,
    augmentation: bool = False,
    input_shape: tuple[int, int, int] = (224, 224, 3),
) -> tf.keras.Model:
    """Build the paper head on ImageNet VGG16.

    The default has a fully frozen convolutional base and no augmentation.
    Optional augmentation and block-5 fine-tuning are explicit extensions.
    """

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    inputs = tf.keras.Input(shape=input_shape, name="ultrasound_image")
    x = inputs
    if augmentation:
        augmentation_layer = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal", seed=42),
                tf.keras.layers.RandomRotation(0.05, fill_mode="reflect", seed=43),
                tf.keras.layers.RandomZoom(0.10, fill_mode="reflect", seed=44),
            ],
            name="training_only_augmentation",
        )
        x = augmentation_layer(x)

    backbone = tf.keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    backbone.trainable = bool(fine_tune_block5)
    if fine_tune_block5:
        for layer in backbone.layers:
            layer.trainable = layer.name.startswith("block5_")
    else:
        for layer in backbone.layers:
            layer.trainable = False

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout_0_5")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="busi_vgg16")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

